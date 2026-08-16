//! `ryu-research mcp` — the crate's own MCP stdio server (JSON-RPC 2.0).
//!
//! The 8 `research__*` tool schemas and their HTTP dispatch have always lived in
//! this crate ([`ryu_research::tool_specs`] / [`ryu_research::dispatch`]), but the
//! only thing that ever *served* them was a hardcoded provider inside Core
//! (`apps/core/src/sidecar/mcp/research.rs`) — an app-specific MCP module of
//! exactly the kind AGENTS.md forbids. This module is the severance: the same two
//! functions, served over stdin/stdout, so Core reaches them the generic way every
//! other app does — a `mcp_servers` entry in `manifest.json` spawning
//! `ryu-research mcp`, registered by `register_manifest_mcp_servers`.
//!
//! ## The contract Core's client actually reads
//!
//! `apps/core/src/sidecar/mcp/client.rs` is a minimal client, so the wire shape has
//! to be exact:
//!
//! - **Newline-delimited JSON**, one frame per line, on stdout. Nothing else may go
//!   to stdout — `main()` therefore points tracing at **stderr** in this mode (the
//!   client pumps child stderr into `tracing::debug!(target: "mcp", …)`).
//! - **`inputSchema`**, camelCase. The client reads `t.get("inputSchema")`; the
//!   snake_case field name a naive serialization would emit arrives as `None` and
//!   the model gets a schema-less tool.
//! - **Bare tool names** (`run`, not `research__run`). The registry applies the
//!   `<server-key>__<tool>` scheme itself, so the manifest key MUST be
//!   [`SERVER_NAME`] (`research`) for every existing `research__*` id — the
//!   workflow template `autoresearch.json` names six of them — to survive the
//!   severance byte-identically.
//! - `initialize` must return a non-error result; the client then fires
//!   `notifications/initialized` and expects **no** response to it.
//!
//! ## Error channel: JSON-RPC error, not `isError`
//!
//! [`ryu_research::dispatch`] already distinguishes the two failure kinds, and this
//! module preserves that distinction across the transport. A merely-unreachable
//! Python engine comes back as an `Ok` value (`{available: false, …}`) so the
//! agent's turn continues — that is a *result*, and it is returned as one. `Err` is
//! reserved for a malformed call (unknown tool, missing argument), and becomes a
//! JSON-RPC `error` frame, which Core's client turns back into `Err(anyhow!("MCP
//! error: …"))`. That is the same `Err` the in-process built-in raised, so callers
//! upstream of the registry cannot tell the transport changed.

use serde_json::{json, Value};

use crate::{dispatch, tool_specs};

/// The MCP protocol version this server speaks. Matches the `MCP_PROTOCOL_VERSION`
/// Core's client sends in `initialize`; a client asking for a different one still
/// gets this (the spec's negotiation is "server states what it speaks").
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// JSON-RPC 2.0 reserved error codes, the three this server can raise.
const ERR_PARSE: i64 = -32700;
const ERR_METHOD_NOT_FOUND: i64 = -32601;
const ERR_INVALID_PARAMS: i64 = -32602;

/// The `tools/list` payload: every spec from [`tool_specs`], with **bare** names and
/// a camelCase `inputSchema`.
pub fn tools_list_result() -> Value {
    let tools: Vec<Value> = tool_specs()
        .into_iter()
        .map(|spec| {
            json!({
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
            })
        })
        .collect();
    json!({ "tools": tools })
}

/// The `initialize` payload. `tools: {}` is the capability declaration that tells a
/// client `tools/list` is worth calling.
pub fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": { "tools": {} },
        "serverInfo": {
            "name": crate::SERVER_NAME,
            "version": env!("CARGO_PKG_VERSION"),
        },
    })
}

/// Wrap a dispatch result value as an MCP `tools/call` result.
///
/// `content` carries the compact JSON as text (what a model without structured-output
/// support reads); `structuredContent` repeats it verbatim, but **only when the value
/// is an object** — the spec types that field as an object, and `ledger` in read mode
/// can return an array.
fn tool_call_result(value: Value) -> Value {
    let text = serde_json::to_string(&value).unwrap_or_else(|_| "null".to_owned());
    let mut result = json!({
        "content": [{ "type": "text", "text": text }],
        "isError": false,
    });
    if value.is_object() {
        result["structuredContent"] = value;
    }
    result
}

fn ok_frame(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn err_frame(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() },
    })
}

/// Handle one decoded JSON-RPC frame.
///
/// Returns the response frame, or `None` for a notification (no `id`) — the spec
/// forbids responding to those, and Core's client would skip the frame anyway while
/// blocking on the id it asked for.
pub async fn handle_frame(client: &reqwest::Client, frame: &Value) -> Option<Value> {
    let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
    // A notification has no `id`. `null` counts as absent (JSON-RPC forbids a null
    // request id), which also keeps us from ever emitting `"id": null` responses.
    let id = match frame.get("id") {
        Some(v) if !v.is_null() => v.clone(),
        _ => return None,
    };

    match method {
        "initialize" => Some(ok_frame(id, initialize_result())),
        "tools/list" => Some(ok_frame(id, tools_list_result())),
        "tools/call" => {
            let params = frame.get("params").cloned().unwrap_or_else(|| json!({}));
            let Some(name) = params.get("name").and_then(Value::as_str) else {
                return Some(err_frame(
                    id,
                    ERR_INVALID_PARAMS,
                    "tools/call requires a string 'name'",
                ));
            };
            let arguments = params
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            match dispatch(client, name, arguments).await {
                Ok(value) => Some(ok_frame(id, tool_call_result(value))),
                // A malformed call (unknown tool / missing argument) — see the
                // module docs on why this is a protocol error, not `isError`.
                Err(e) => Some(err_frame(id, ERR_INVALID_PARAMS, format!("{e:#}"))),
            }
        }
        other => Some(err_frame(
            id,
            ERR_METHOD_NOT_FOUND,
            format!("unknown method '{other}'"),
        )),
    }
}

/// Serve MCP over stdin/stdout until EOF.
///
/// Exits `Ok(())` on EOF: Core's client shuts a server down by dropping its stdin,
/// so clean-exit-on-EOF is the normal termination, not a failure.
pub async fn run() -> anyhow::Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let client = reqwest::Client::new();
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(line) {
            Ok(frame) => handle_frame(&client, &frame).await,
            // Unparseable input has no id to answer under; JSON-RPC says respond
            // with a null id rather than stay silent.
            Err(e) => Some(json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": { "code": ERR_PARSE, "message": format!("invalid JSON: {e}") },
            })),
        };
        if let Some(response) = response {
            let mut out = serde_json::to_string(&response)?;
            out.push('\n');
            stdout.write_all(out.as_bytes()).await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: i64, method: &str, params: Value) -> Value {
        json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
    }

    /// The headline contract: `tools/list` advertises EXACTLY the 8 tools the
    /// in-process built-in did, under their bare names.
    #[tokio::test]
    async fn tools_list_advertises_the_eight_bare_tool_names() {
        let client = reqwest::Client::new();
        let resp = handle_frame(&client, &req(1, "tools/list", json!({})))
            .await
            .expect("tools/list is a request, not a notification");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 8, "got {tools:?}");

        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            vec![
                "list_experiments",
                "init_workspace",
                "read_file",
                "write_file",
                "run",
                "keep",
                "reset",
                "ledger",
            ]
        );
        // Bare, NOT `research__*`: the registry applies the `<server>__<tool>`
        // scheme itself. Pre-qualifying here would yield `research__research__run`.
        assert!(
            names.iter().all(|n| !n.contains("__")),
            "tool names must be bare, got {names:?}"
        );
    }

    /// The manifest key must be `SERVER_NAME`, because that is what makes the
    /// registry's ids come out as the `research__*` the workflow templates already
    /// name. Asserted here so a rename of either side fails loudly.
    #[test]
    fn qualified_ids_match_the_pre_severance_scheme() {
        for spec in tool_specs() {
            let id = format!("{}__{}", crate::SERVER_NAME, spec.name);
            assert!(id.starts_with("research__"), "got {id}");
        }
    }

    /// `inputSchema` — camelCase. Core's client reads that exact key; a snake_case
    /// `input_schema` would silently strip every schema.
    #[tokio::test]
    async fn every_tool_carries_a_camel_case_input_schema() {
        let client = reqwest::Client::new();
        let resp = handle_frame(&client, &req(1, "tools/list", json!({})))
            .await
            .unwrap();
        for tool in resp["result"]["tools"].as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            assert!(
                tool.get("input_schema").is_none(),
                "{name} must not emit snake_case input_schema"
            );
            let schema = tool
                .get("inputSchema")
                .unwrap_or_else(|| panic!("{name} has no inputSchema"));
            assert_eq!(schema["type"], "object", "{name} schema: {schema}");
            assert!(
                tool["description"].as_str().is_some_and(|d| !d.is_empty()),
                "{name} needs a description"
            );
        }
        // Spot-check one schema survived intact rather than being replaced by a stub.
        let init = resp["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "init_workspace")
            .unwrap();
        assert_eq!(init["inputSchema"]["required"][0], "experiment");
    }

    #[tokio::test]
    async fn initialize_states_the_protocol_and_tools_capability() {
        let client = reqwest::Client::new();
        let resp = handle_frame(&client, &req(7, "initialize", json!({})))
            .await
            .unwrap();
        assert_eq!(resp["id"], 7);
        assert_eq!(resp["jsonrpc"], "2.0");
        assert!(resp.get("error").is_none(), "initialize must not error");
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
        assert_eq!(resp["result"]["serverInfo"]["name"], crate::SERVER_NAME);
    }

    /// `notifications/initialized` carries no id — answering it would push an
    /// unmatched frame at a client that is blocking on a different id.
    #[tokio::test]
    async fn notifications_get_no_response() {
        let client = reqwest::Client::new();
        let frame =
            json!({ "jsonrpc": "2.0", "method": "notifications/initialized", "params": {} });
        assert!(handle_frame(&client, &frame).await.is_none());
        // An explicit `"id": null` is not a valid request id either.
        let nulled = json!({ "jsonrpc": "2.0", "id": null, "method": "tools/list" });
        assert!(handle_frame(&client, &nulled).await.is_none());
    }

    #[tokio::test]
    async fn unknown_method_is_a_method_not_found_error() {
        let client = reqwest::Client::new();
        let resp = handle_frame(&client, &req(3, "resources/list", json!({})))
            .await
            .unwrap();
        assert_eq!(resp["error"]["code"], ERR_METHOD_NOT_FOUND);
        assert!(resp.get("result").is_none());
    }

    /// A malformed call keeps the `Err` semantics the in-process built-in had: a
    /// JSON-RPC error frame, which Core's client re-raises as `Err`.
    #[tokio::test]
    async fn unknown_tool_and_missing_argument_are_protocol_errors() {
        let client = reqwest::Client::new();

        let unknown = handle_frame(
            &client,
            &req(4, "tools/call", json!({ "name": "no_such_tool" })),
        )
        .await
        .unwrap();
        assert_eq!(unknown["error"]["code"], ERR_INVALID_PARAMS);
        assert!(unknown["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no_such_tool"));

        // `read_file` requires workspace_id + path; omitting them must fail BEFORE
        // any HTTP is attempted (so this test needs no engine running).
        let missing = handle_frame(
            &client,
            &req(
                5,
                "tools/call",
                json!({ "name": "read_file", "arguments": {} }),
            ),
        )
        .await
        .unwrap();
        assert_eq!(missing["error"]["code"], ERR_INVALID_PARAMS);
        assert!(missing["error"]["message"]
            .as_str()
            .unwrap()
            .contains("workspace_id"));

        let no_name = handle_frame(&client, &req(6, "tools/call", json!({})))
            .await
            .unwrap();
        assert_eq!(no_name["error"]["code"], ERR_INVALID_PARAMS);
    }

    /// A DOWN engine is a result, not an error — the graceful degradation the
    /// in-process provider had must survive the transport.
    #[tokio::test]
    async fn an_unreachable_engine_returns_a_result_not_an_error() {
        let _g = crate::UpstreamGuard::set(Some("127.0.0.1:1"));
        let client = reqwest::Client::new();
        let resp = handle_frame(
            &client,
            &req(8, "tools/call", json!({ "name": "list_experiments" })),
        )
        .await
        .unwrap();
        assert!(
            resp.get("error").is_none(),
            "a down engine must not abort the turn: {resp}"
        );
        assert_eq!(resp["result"]["isError"], false);
        assert_eq!(resp["result"]["structuredContent"]["available"], false);
        // The text channel carries the same payload for a model with no structured
        // output support.
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"available\":false"), "got {text}");
    }

    #[test]
    fn a_non_object_result_omits_structured_content() {
        // `structuredContent` is typed as an object; `ledger` (read mode) can answer
        // with an array, which must ride the text channel alone.
        let arr = tool_call_result(json!([{ "commit": "abc" }]));
        assert!(arr.get("structuredContent").is_none());
        assert_eq!(arr["content"][0]["type"], "text");
        let obj = tool_call_result(json!({ "ok": true }));
        assert_eq!(obj["structuredContent"]["ok"], true);
    }
}
