/**
 * Codex CLI subprocess transport.
 *
 * Rust-side counterpart: src-tauri/src/commands/codex_cli.rs. The Rust
 * commands spawn `codex exec --json ... -`, write a reconstructed chat
 * transcript to stdin, and emit stdout back as `codex-cli:{streamId}`
 * events (one JSONL line per event). Codex is treated as a text-only
 * LLM backend here, not as a file-editing agent.
 */

import { invoke } from "@tauri-apps/api/core"
import { listen, type UnlistenFn } from "@tauri-apps/api/event"
import type { LlmConfig } from "@/stores/wiki-store"
import type { ChatMessage, RequestOverrides } from "./llm-providers"
import type { StreamCallbacks } from "./llm-client"

export const CODEX_CLI_TIMEOUT_MS = 30 * 60 * 1000

/**
 * Parse one JSONL event emitted by `codex exec --json`.
 *
 * Current Codex CLI emits final assistant text as:
 *   {"type":"item.completed","item":{"type":"agent_message","text":"..."}}
 *
 * If a future CLI starts sending cumulative agent-message updates, the
 * same prefix-diff logic keeps the UI from duplicating text.
 */
export function createCodexCliStreamParser() {
  let emittedFromAgentMessage = ""

  return function parseLine(rawLine: string): string | null {
    const line = rawLine.trim()
    if (!line) return null

    let evt: unknown
    try {
      evt = JSON.parse(line)
    } catch {
      return null
    }

    if (!evt || typeof evt !== "object") return null
    const obj = evt as Record<string, unknown>
    const type = obj.type

    if (type !== "item.completed" && type !== "item.updated") return null

    const item = obj.item as Record<string, unknown> | undefined
    if (item?.type !== "agent_message" || typeof item.text !== "string") {
      return null
    }

    const text = item.text
    if (!text) return null
    if (text.startsWith(emittedFromAgentMessage)) {
      const novel = text.slice(emittedFromAgentMessage.length)
      emittedFromAgentMessage = text
      return novel || null
    }
    emittedFromAgentMessage = text
    return text
  }
}

type CodexCliMessage = {
  role: ChatMessage["role"]
  content: string
}

type SpawnPayload = Record<string, unknown> & {
  streamId: string
  model: string
  messages: CodexCliMessage[]
}

function contentToText(content: ChatMessage["content"]): string {
  if (typeof content === "string") return content
  const image = content.find((block) => block.type === "image")
  if (image) {
    throw new Error(
      "Codex CLI provider is text-only in LLM Wiki. Use OpenAI, Anthropic, Gemini, Ollama, or a custom vision endpoint for image captioning.",
    )
  }
  return content.map((block) => (block.type === "text" ? block.text : "")).join("")
}

function toCodexCliMessages(messages: ChatMessage[]): CodexCliMessage[] {
  return messages.map((message) => ({
    role: message.role,
    content: contentToText(message.content),
  }))
}

export async function streamCodexCli(
  config: LlmConfig,
  messages: ChatMessage[],
  callbacks: StreamCallbacks,
  signal?: AbortSignal,
  overrides?: RequestOverrides,
): Promise<void> {
  const { onToken, onDone, onError } = callbacks

  if (import.meta.env?.DEV && overrides) {
    for (const key of ["temperature", "top_p", "top_k", "max_tokens", "stop", "reasoning"] as const) {
      if (overrides[key] !== undefined) {
        // eslint-disable-next-line no-console
        console.warn(`[codex-cli] ignoring unsupported override "${key}": CLI transport has no stable equivalent flag`)
      }
    }
  }

  const streamId = crypto.randomUUID()
  const parse = createCodexCliStreamParser()

  let unlistenData: UnlistenFn | undefined
  let unlistenDone: UnlistenFn | undefined
  let timeoutId: ReturnType<typeof setTimeout> | undefined
  let finished = false
  let resolveStream: () => void = () => {}
  const streamComplete = new Promise<void>((resolve) => {
    resolveStream = resolve
  })

  const UNPARSED_BUFFER_CAP = 4096
  const unparsedLines: string[] = []
  let unparsedSize = 0
  let emittedTokenChars = 0
  function captureUnparsed(line: string) {
    if (unparsedSize >= UNPARSED_BUFFER_CAP) return
    const trimmed = line.trim()
    if (trimmed.length === 0) return
    unparsedLines.push(line)
    unparsedSize += line.length + 1
  }

  const cleanup = () => {
    if (timeoutId !== undefined) {
      clearTimeout(timeoutId)
      timeoutId = undefined
    }
    unlistenData?.()
    unlistenDone?.()
  }

  const finishWith = (cb: () => void) => {
    if (finished) return
    finished = true
    cleanup()
    try {
      cb()
    } finally {
      resolveStream()
    }
  }

  const abortListener = () => {
    void invoke("codex_cli_kill", { streamId }).catch(() => {})
    finishWith(onDone)
  }
  signal?.addEventListener("abort", abortListener)

  try {
    const codexMessages = toCodexCliMessages(messages)

    unlistenData = await listen<string>(`codex-cli:${streamId}`, (event) => {
      const token = parse(event.payload)
      if (token !== null) {
        emittedTokenChars += token.length
        onToken(token)
      } else {
        captureUnparsed(event.payload)
      }
    })

    unlistenDone = await listen<{ code: number | null; stderr: string }>(
      `codex-cli:${streamId}:done`,
      (event) => {
        const code = event.payload?.code
        const stderr = event.payload?.stderr?.trim() ?? ""
        if (code !== null && code !== undefined && code !== 0) {
          finishWith(() =>
            onError(
              new Error(buildExitError(code, stderr, unparsedLines.join("\n"))),
            ),
          )
        } else if (emittedTokenChars === 0) {
          finishWith(() =>
            onError(
              new Error(buildEmptySuccessError(stderr, unparsedLines.join("\n"))),
            ),
          )
        } else {
          finishWith(onDone)
        }
      },
    )

    const payload: SpawnPayload = {
      streamId,
      model: config.model,
      messages: codexMessages,
    }
    timeoutId = setTimeout(() => {
      void invoke("codex_cli_kill", { streamId }).catch(() => {})
      finishWith(() => onError(new Error(buildTimeoutError())))
    }, CODEX_CLI_TIMEOUT_MS)

    const spawnPromise = invoke("codex_cli_spawn", payload).catch((err) => {
      finishWith(() => {
        const message = err instanceof Error ? err.message : String(err)
        if (/not found|No such file|executable file not found/i.test(message)) {
          onError(new Error(
            "Codex CLI not found. Install `codex`, run `codex login`, or pick a different provider.",
          ))
        } else {
          onError(err instanceof Error ? err : new Error(message))
        }
      })
    })

    // Keep the streamChat contract: callers that `await streamChat(...)`
    // must not continue until the subprocess has finished or failed.
    await Promise.race([spawnPromise, streamComplete])
    if (!finished) await streamComplete
  } catch (err) {
    finishWith(() => {
      const message = err instanceof Error ? err.message : String(err)
      if (/not found|No such file|executable file not found/i.test(message)) {
        onError(new Error(
          "Codex CLI not found. Install `codex`, run `codex login`, or pick a different provider.",
        ))
      } else {
        onError(err instanceof Error ? err : new Error(message))
      }
    })
  } finally {
    signal?.removeEventListener("abort", abortListener)
  }
}

export function buildTimeoutError(timeoutMs: number = CODEX_CLI_TIMEOUT_MS): string {
  return [
    `codex CLI timed out after ${Math.round(timeoutMs / 60000)} min.`,
    "The subprocess was stopped so the ingest queue can retry or fail cleanly.",
    "Try a smaller source, a lower context window, or run the same prompt in a terminal to inspect the CLI behavior.",
  ].join(" ")
}

export function buildEmptySuccessError(
  stderr: string,
  unparsedStdout: string = "",
): string {
  const trimmedStdout = unparsedStdout.trim()
  const trimmedStderr = stderr.trim()
  if (trimmedStdout) {
    return [
      "codex CLI exited successfully, but LLM Wiki could not parse any assistant response from stdout.",
      "This usually means the CLI JSONL schema changed or Codex emitted non-message events only.",
      "Captured stdout:\n",
      trimmedStdout,
      trimmedStderr ? `\n\n-- stderr --\n${trimmedStderr}` : "",
    ].join(" ").trim()
  }
  if (trimmedStderr) {
    return [
      "codex CLI exited successfully, but produced no assistant response.",
      `stderr: ${trimmedStderr}`,
    ].join(" ")
  }
  return [
    "codex CLI exited successfully, but produced no assistant response.",
    "Try running `codex exec --json -` in a terminal with the same prompt",
    "to see whether the CLI emitted an unexpected JSONL shape.",
  ].join(" ")
}

export function buildExitError(
  code: number,
  stderr: string,
  unparsedStdout: string = "",
): string {
  if (/not.*logged\s*in|please.*log\s*in|authentication|unauthorized|401/i.test(stderr)) {
    return [
      "Codex CLI is not authenticated.",
      "Please open a terminal and run `codex login`, then retry.",
      "(LLM Wiki only spawns the binary; it cannot complete the login flow on your behalf.)",
      stderr ? `\n\n-- stderr --\n${stderr}` : "",
    ].join(" ").trim()
  }
  if (stderr) {
    return `codex CLI exited with code ${code}: ${stderr}`
  }
  if (unparsedStdout.trim()) {
    return [
      `codex CLI exited with code ${code} (no stderr).`,
      "Captured stdout output that LLM Wiki could not parse:\n",
      unparsedStdout.trim(),
    ].join(" ")
  }
  return [
    `codex CLI exited silently with code ${code}.`,
    "Try running `codex exec --json -` in a terminal with the same prompt",
    "to see the underlying CLI error, or switch providers in Settings.",
  ].join(" ")
}
