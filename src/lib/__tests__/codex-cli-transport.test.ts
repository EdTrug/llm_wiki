import { describe, it, expect } from "vitest"
import {
  createCodexCliStreamParser,
  buildExitError,
  buildEmptySuccessError,
  buildTimeoutError,
} from "../codex-cli-transport"

describe("createCodexCliStreamParser", () => {
  it("emits text from a completed agent_message item", () => {
    const parse = createCodexCliStreamParser()
    const line = JSON.stringify({
      type: "item.completed",
      item: { id: "item_0", type: "agent_message", text: "pong" },
    })
    expect(parse(line)).toBe("pong")
  })

  it("emits only the novel tail when agent_message events are cumulative", () => {
    const parse = createCodexCliStreamParser()
    const mk = (text: string) =>
      JSON.stringify({
        type: "item.updated",
        item: { id: "item_0", type: "agent_message", text },
      })

    expect(parse(mk("Hello"))).toBe("Hello")
    expect(parse(mk("Hello world"))).toBe(" world")
    expect(parse(mk("Hello world!"))).toBe("!")
  })

  it("returns null for lifecycle events and non-message items", () => {
    const parse = createCodexCliStreamParser()
    expect(parse(JSON.stringify({ type: "thread.started", thread_id: "x" }))).toBeNull()
    expect(parse(JSON.stringify({ type: "turn.started" }))).toBeNull()
    expect(
      parse(JSON.stringify({ type: "item.completed", item: { type: "command_execution" } })),
    ).toBeNull()
    expect(parse(JSON.stringify({ type: "turn.completed", usage: {} }))).toBeNull()
  })

  it("returns null for malformed JSON or blank lines", () => {
    const parse = createCodexCliStreamParser()
    expect(parse("")).toBeNull()
    expect(parse("   ")).toBeNull()
    expect(parse("not json")).toBeNull()
    expect(parse("{bad json")).toBeNull()
  })
})

describe("buildExitError", () => {
  it("translates auth stderr into an actionable login hint", () => {
    const msg = buildExitError(1, "Authentication failed (401)")
    expect(msg).toMatch(/not authenticated/i)
    expect(msg).toMatch(/codex login/)
  })

  it("falls through to exit-code form for unrecognized stderr", () => {
    expect(buildExitError(2, "Unknown flag: --foo")).toBe(
      "codex CLI exited with code 2: Unknown flag: --foo",
    )
  })

  it("falls back to unparsed stdout when stderr is empty", () => {
    const stdout = '{"type":"error","message":"model not found"}'
    const msg = buildExitError(1, "", stdout)
    expect(msg).toContain("code 1")
    expect(msg).toContain("no stderr")
    expect(msg).toContain("model not found")
  })

  it("recommends terminal reproduction for silent exits", () => {
    const msg = buildExitError(1, "", "")
    expect(msg).toMatch(/silently/)
    expect(msg).toMatch(/codex exec/)
  })
})

describe("Codex CLI transport diagnostics", () => {
  it("treats exit 0 with only unparsed stdout as a schema/parser error", () => {
    const msg = buildEmptySuccessError("", '{"type":"turn.completed","usage":{}}')
    expect(msg).toMatch(/could not parse any assistant response/i)
    expect(msg).toContain("turn.completed")
  })

  it("treats exit 0 with no stdout as an empty assistant response", () => {
    const msg = buildEmptySuccessError("", "")
    expect(msg).toMatch(/produced no assistant response/i)
    expect(msg).toMatch(/codex exec/)
  })

  it("reports timeout in minutes", () => {
    const msg = buildTimeoutError(5 * 60 * 1000)
    expect(msg).toContain("5 min")
    expect(msg).toMatch(/stopped/)
  })
})
