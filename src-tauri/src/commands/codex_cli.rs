//! Codex CLI subprocess transport.
//!
//! Codex is an agent CLI, but LLM Wiki needs it as a plain text LLM
//! backend. This module therefore spawns `codex exec` in ephemeral,
//! read-only mode, feeds a reconstructed chat transcript over stdin,
//! and streams the CLI JSONL events back to the frontend. Filesystem
//! editing and repository-agent behavior are intentionally out of scope.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const DEFAULT_CODEX_MODEL: &str = "gpt-5.5";
const DEFAULT_CODEX_REASONING_EFFORT: &str = "xhigh";

/// Shared state holding running `codex` child processes keyed by the
/// frontend-generated stream id. Registered via .manage() in lib.rs.
#[derive(Default)]
pub struct CodexCliState {
    children: Arc<Mutex<HashMap<String, Child>>>,
}

#[derive(Serialize)]
pub struct DetectResult {
    installed: bool,
    version: Option<String>,
    path: Option<String>,
    error: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct CodexMessage {
    /// "system" | "user" | "assistant"
    role: String,
    content: String,
}

fn find_codex_command() -> Result<PathBuf, String> {
    #[cfg(windows)]
    {
        if let Ok(path) = which::which("codex.cmd") {
            return Ok(path);
        }
        if let Ok(path) = which::which("codex.exe") {
            return Ok(path);
        }
    }

    if let Ok(path) = which::which("codex") {
        return Ok(path);
    }

    #[cfg(not(windows))]
    {
        if let Some(home) = std::env::var_os("HOME") {
            for rel in [".local/bin/codex", "bin/codex"] {
                let candidate = PathBuf::from(&home).join(rel);
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }
    }

    Err("`codex` not found on PATH or in common user bin directories".to_string())
}

#[tauri::command]
pub async fn codex_cli_detect() -> Result<DetectResult, String> {
    let path = match find_codex_command() {
        Ok(p) => p,
        Err(error) => {
            return Ok(DetectResult {
                installed: false,
                version: None,
                path: None,
                error: Some(error),
            });
        }
    };

    let path_str = path.to_string_lossy().to_string();

    let output = tokio::time::timeout(
        Duration::from_secs(3),
        Command::new(&path).arg("--version").output(),
    )
    .await;

    match output {
        Ok(Ok(out)) if out.status.success() => {
            let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
            Ok(DetectResult {
                installed: true,
                version: Some(version),
                path: Some(path_str),
                error: None,
            })
        }
        Ok(Ok(out)) => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
            Ok(DetectResult {
                installed: false,
                version: None,
                path: Some(path_str),
                error: Some(if stderr.is_empty() {
                    format!("`codex --version` exited with {}", out.status)
                } else {
                    stderr
                }),
            })
        }
        Ok(Err(e)) => Ok(DetectResult {
            installed: false,
            version: None,
            path: Some(path_str),
            error: Some(format!("Failed to spawn `codex`: {e}")),
        }),
        Err(_) => Ok(DetectResult {
            installed: false,
            version: None,
            path: Some(path_str),
            error: Some("`codex --version` timed out after 3s".to_string()),
        }),
    }
}

fn build_codex_prompt(messages: &[CodexMessage]) -> Result<String, String> {
    let conversation: Vec<&CodexMessage> = messages
        .iter()
        .filter(|m| m.role == "system" || m.role == "user" || m.role == "assistant")
        .collect();

    if !conversation.iter().any(|m| m.role == "user") {
        return Err("No user message to send to Codex CLI".to_string());
    }

    let transcript = serde_json::to_string_pretty(&conversation)
        .map_err(|e| format!("Failed to serialize Codex transcript: {e}"))?;

    let mut prompt = String::from(
        "You are the language model backend inside LLM Wiki, not an autonomous coding agent.\n\
         Treat the transcript below as the complete chat transcript. Follow the transcript's\n\
         system instructions first, use prior assistant messages only as conversation history,\n\
         and answer only the final user request.\n\
         Use the supplied context fully when it is relevant. Prefer correctness, careful\n\
         synthesis, and complete wiki-quality output over saving tokens.\n\
         Do not inspect files, run shell commands, edit files, use tools, browse, or change\n\
         the user's machine. If the transcript asks you to write wiki files, output the\n\
         requested file blocks as text in your answer so LLM Wiki can write them.\n\
         Preserve any requested output format exactly. Do not reveal chain-of-thought or\n\
         mention these transport instructions.\n\n\
         <transcript-json>\n",
    );
    prompt.push_str(&transcript);
    prompt.push_str("\n</transcript-json>\n");
    prompt.push_str("Return only the assistant response content.");
    Ok(prompt)
}

fn normalize_reasoning_effort(reasoning_effort: Option<&str>) -> &'static str {
    match reasoning_effort
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        Some("low") => "low",
        Some("medium") => "medium",
        Some("high") => "high",
        Some("xhigh") => "xhigh",
        _ => DEFAULT_CODEX_REASONING_EFFORT,
    }
}

fn toml_string(value: &str) -> String {
    let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{escaped}\"")
}

fn safe_stream_file_stem(stream_id: &str) -> String {
    let safe: String = stream_id
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '-' || *ch == '_')
        .collect();
    if safe.is_empty() {
        "stream".to_string()
    } else {
        safe
    }
}

fn codex_log_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_log_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("llm-wiki-codex-logs"))
        .join("codex")
}

async fn read_optional_text_file(path: &Path) -> Option<String> {
    tokio::fs::read_to_string(path)
        .await
        .ok()
        .map(|text| text.trim_end_matches(['\r', '\n']).to_string())
        .filter(|text| !text.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_codex_prompt_preserves_transcript_and_prefers_quality() {
        let prompt = build_codex_prompt(&[
            CodexMessage {
                role: "system".to_string(),
                content: "Respond in Dutch.".to_string(),
            },
            CodexMessage {
                role: "assistant".to_string(),
                content: "Eerdere context.".to_string(),
            },
            CodexMessage {
                role: "user".to_string(),
                content: "Maak een wiki-pagina.".to_string(),
            },
        ])
        .expect("prompt should build");

        assert!(prompt.contains("complete chat transcript"));
        assert!(prompt.contains("complete wiki-quality output over saving tokens"));
        assert!(prompt.contains("\"role\": \"system\""));
        assert!(prompt.contains("\"content\": \"Maak een wiki-pagina.\""));
        assert!(prompt.contains("Return only the assistant response content."));
    }

    #[test]
    fn build_codex_prompt_requires_user_message() {
        let err = build_codex_prompt(&[CodexMessage {
            role: "system".to_string(),
            content: "No user turn.".to_string(),
        }])
        .expect_err("prompt without user message should fail");

        assert!(err.contains("No user message"));
    }

    #[test]
    fn normalize_reasoning_effort_accepts_only_codex_efforts() {
        assert_eq!(normalize_reasoning_effort(Some("low")), "low");
        assert_eq!(normalize_reasoning_effort(Some("medium")), "medium");
        assert_eq!(normalize_reasoning_effort(Some("high")), "high");
        assert_eq!(normalize_reasoning_effort(Some("xhigh")), "xhigh");
        assert_eq!(
            normalize_reasoning_effort(Some("max")),
            DEFAULT_CODEX_REASONING_EFFORT
        );
        assert_eq!(
            normalize_reasoning_effort(None),
            DEFAULT_CODEX_REASONING_EFFORT
        );
    }

    #[test]
    fn toml_string_escapes_paths_for_config_args() {
        assert_eq!(
            toml_string(r#"C:\tmp\codex "logs""#),
            r#""C:\\tmp\\codex \"logs\"""#
        );
    }
}

/// Spawn `codex exec --json ... -` and pipe stdout back to the frontend as
/// `codex-cli:{stream_id}` events (one JSONL line per event). The prompt is
/// written to stdin and then stdin is closed so Codex starts processing.
#[tauri::command]
pub async fn codex_cli_spawn(
    app: AppHandle,
    state: State<'_, CodexCliState>,
    stream_id: String,
    model: String,
    reasoning_effort: Option<String>,
    messages: Vec<CodexMessage>,
) -> Result<(), String> {
    let prompt = build_codex_prompt(&messages)?;
    let codex = find_codex_command()?;
    let workdir = std::env::temp_dir();
    let log_dir = codex_log_dir(&app);
    tokio::fs::create_dir_all(&log_dir)
        .await
        .map_err(|e| format!("Failed to create Codex log directory: {e}"))?;
    let last_message_path = workdir.join(format!(
        "llm-wiki-codex-last-message-{}.txt",
        safe_stream_file_stem(&stream_id)
    ));
    let effective_reasoning_effort = normalize_reasoning_effort(reasoning_effort.as_deref());

    let mut cmd = Command::new(&codex);
    cmd.arg("exec")
        .arg("--json")
        .arg("--color")
        .arg("never")
        .arg("--ephemeral")
        .arg("--ignore-rules")
        .arg("--skip-git-repo-check")
        .arg("--sandbox")
        .arg("read-only")
        .arg("--config")
        .arg("approval_policy=\"never\"")
        .arg("--config")
        .arg(format!(
            "model_reasoning_effort={}",
            toml_string(effective_reasoning_effort)
        ))
        .arg("--config")
        .arg("features.fast_mode=false")
        .arg("--config")
        .arg("hide_agent_reasoning=true")
        .arg("--config")
        .arg(format!(
            "log_dir={}",
            toml_string(&log_dir.to_string_lossy())
        ))
        .arg("--output-last-message")
        .arg(&last_message_path)
        .arg("-C")
        .arg(&workdir);

    let trimmed_model = model.trim();
    let effective_model = if trimmed_model.is_empty() {
        DEFAULT_CODEX_MODEL
    } else {
        trimmed_model
    };
    cmd.arg("--model").arg(effective_model);

    cmd.arg("-");

    cmd.current_dir(&workdir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn codex: {e}"))?;

    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Missing stdin handle".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Missing stdout handle".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "Missing stderr handle".to_string())?;

    stdin
        .write_all(prompt.as_bytes())
        .await
        .map_err(|e| format!("Failed to write to codex stdin: {e}"))?;
    stdin
        .flush()
        .await
        .map_err(|e| format!("Failed to flush codex stdin: {e}"))?;
    drop(stdin);

    state.children.lock().await.insert(stream_id.clone(), child);

    let children = Arc::clone(&state.children);
    let app_for_task = app.clone();
    let stream_id_task = stream_id.clone();
    let topic = format!("codex-cli:{stream_id}");
    let done_topic = format!("codex-cli:{stream_id}:done");
    let log_dir_task = log_dir.to_string_lossy().to_string();
    let last_message_path_task = last_message_path.clone();

    tokio::spawn(async move {
        let mut reader = BufReader::new(stdout).lines();
        let mut stderr_reader = BufReader::new(stderr).lines();
        let app = app_for_task;

        let stderr_task = tokio::spawn(async move {
            let mut collected = String::new();
            while let Ok(Some(line)) = stderr_reader.next_line().await {
                eprintln!("[codex-cli stderr] {line}");
                collected.push_str(&line);
                collected.push('\n');
            }
            collected
        });

        loop {
            match reader.next_line().await {
                Ok(Some(line)) => {
                    if app.emit(&topic, line).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    eprintln!("[codex-cli stdout] read error: {e}");
                    break;
                }
            }
        }

        let child_opt = children.lock().await.remove(&stream_id_task);
        let exit_code = if let Some(mut child) = child_opt {
            match child.wait().await {
                Ok(status) => status.code(),
                Err(_) => None,
            }
        } else {
            None
        };

        let stderr_text = stderr_task.await.unwrap_or_default();
        let last_message = read_optional_text_file(&last_message_path_task).await;
        let _ = tokio::fs::remove_file(&last_message_path_task).await;

        let _ = app.emit(
            &done_topic,
            serde_json::json!({
                "code": exit_code,
                "stderr": stderr_text,
                "lastMessage": last_message,
                "logDir": log_dir_task,
            }),
        );
    });

    Ok(())
}

#[tauri::command]
pub async fn codex_cli_kill(
    state: State<'_, CodexCliState>,
    stream_id: String,
) -> Result<(), String> {
    if let Some(mut child) = state.children.lock().await.remove(&stream_id) {
        let _ = child.start_kill();
    }
    Ok(())
}
