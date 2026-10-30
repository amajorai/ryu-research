//! `ryu-research mcp` — the crate's own MCP stdio server (JSON-RPC 2.0).
//!
//! The 8 `research.*` tool schemas and their HTTP dispatch have always lived in
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
//! - **Bare tool names** (`run`, not `research.run`). The registry applies the
//!   `<server-key>.<tool>` scheme itself, so the manifest key MUST be
//!   [`SERVER_NAME`] (`research`) for every existing `research.*` id — the
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
const MODERN_PROTOCOL_VERSION: &str = "2026-07-28";
const META_PROTOCOL_VERSION: &str = "io.modelcontextprotocol/protocolVersion";
const META_CLIENT_INFO: &str = "io.modelcontextprotocol/clientInfo";
const META_CLIENT_CAPABILITIES: &str = "io.modelcontextprotocol/clientCapabilities";
const SERVER_INFO: &str = "io.modelcontextprotocol/serverInfo";

#[derive(Debug)]
struct ProtocolError {
    code: i64,
    message: &'static str,
    data: Option<Value>,
}

fn invalid_modern_params() -> ProtocolError {
    ProtocolError {
        code: ERR_INVALID_PARAMS,
        message:
            "modern MCP requests require protocolVersion and clientCapabilities in params._meta",
        data: None,
    }
}

fn request_is_modern(
    frame: &Value,
    method: &str,
    legacy_initialized: bool,
) -> Result<bool, ProtocolError> {
    let metadata = frame.get("params").and_then(|params| params.get("_meta"));
    if metadata.is_none() {
        if method == "server/discover" {
            return Err(invalid_modern_params());
        }
        return if method == "initialize" || legacy_initialized {
            Ok(false)
        } else {
            Err(invalid_modern_params())
        };
    }
    let Some(metadata) = metadata.and_then(Value::as_object) else {
        return Err(invalid_modern_params());
    };
    let is_modern = method == "server/discover"
        || metadata.contains_key(META_PROTOCOL_VERSION)
        || metadata.contains_key(META_CLIENT_CAPABILITIES);
    if !is_modern {
        return if method == "initialize" || legacy_initialized {
            Ok(false)
        } else {
            Err(invalid_modern_params())
        };
    }
    let version = metadata
        .get(META_PROTOCOL_VERSION)
        .and_then(Value::as_str)
        .ok_or_else(invalid_modern_params)?;
    if metadata
        .get(META_CLIENT_CAPABILITIES)
        .and_then(Value::as_object)
        .is_none()
    {
        return Err(invalid_modern_params());
    }
    if let Some(client_info) = metadata.get(META_CLIENT_INFO) {
        let valid_client_info = client_info.as_object().is_some_and(|info| {
            info.get("name").and_then(Value::as_str).is_some()
                && info.get("version").and_then(Value::as_str).is_some()
        });
        if !valid_client_info {
            return Err(invalid_modern_params());
        }
    }
    if version != MODERN_PROTOCOL_VERSION {
        return Err(ProtocolError {
            code: -32022,
            message: "Unsupported protocol version",
            data: Some(json!({
                "supported": [MODERN_PROTOCOL_VERSION],
                "requested": version,
            })),
        });
    }
    Ok(true)
}

fn protocol_error(id: Value, error: ProtocolError) -> Value {
    let mut details = json!({ "code": error.code, "message": error.message });
    if let Some(data) = error.data {
        details["data"] = data;
    }
    json!({ "jsonrpc": "2.0", "id": id, "error": details })
}

fn finalize_response(mut response: Value, modern: bool, method: &str, server_name: &str) -> Value {
    if !modern {
        return response;
    }
    let Some(result) = response.get_mut("result").and_then(Value::as_object_mut) else {
        return response;
    };
    result
        .entry("resultType".to_owned())
        .or_insert_with(|| json!("complete"));
    if method == "tools/list" {
        result
            .entry("ttlMs".to_owned())
            .or_insert_with(|| json!(300_000));
        result
            .entry("cacheScope".to_owned())
            .or_insert_with(|| json!("public"));
    }
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        SERVER_INFO.to_owned(),
        json!({ "name": server_name, "version": env!("CARGO_PKG_VERSION") }),
    );
    result.insert("_meta".to_owned(), Value::Object(metadata));
    response
}

fn discover_result() -> Value {
    json!({
        "supportedVersions": [MODERN_PROTOCOL_VERSION],
        "capabilities": { "tools": {} },
    })
}

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
                "annotations": spec.annotations,
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
    let mut legacy_initialized = true;
    handle_frame_with_legacy_state(client, frame, &mut legacy_initialized).await
}

async fn handle_frame_with_legacy_state(
    client: &reqwest::Client,
    frame: &Value,
    legacy_initialized: &mut bool,
) -> Option<Value> {
    let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
    // A notification has no `id`. `null` counts as absent (JSON-RPC forbids a null
    // request id), which also keeps us from ever emitting `"id": null` responses.
    let id = match frame.get("id") {
        Some(v) if !v.is_null() => v.clone(),
        _ => return None,
    };
    let modern = match request_is_modern(frame, method, *legacy_initialized) {
        Ok(modern) => modern,
        Err(error) => return Some(protocol_error(id, error)),
    };
    *legacy_initialized |= method == "initialize" && !modern;

    let response = match method {
        "initialize" if modern => {
            return Some(err_frame(
                id,
                ERR_METHOD_NOT_FOUND,
                "unknown method 'initialize'",
            ));
        }
        "server/discover" => ok_frame(id.clone(), discover_result()),
        "initialize" => ok_frame(id.clone(), initialize_result()),
        "tools/list" => ok_frame(id.clone(), tools_list_result()),
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
            // Campaign data is owned: every call except the public experiment
            // catalog runs under the verified caller scope Core stamped in
            // `_meta`, never one derived from the model-controlled arguments.
            let scope = match crate::caller::from_mcp_meta(params.get("_meta")) {
                Ok(scope) => Some(scope),
                Err(_) if name == "list_experiments" => None,
                Err(reason) => return Some(err_frame(id, ERR_INVALID_PARAMS, reason)),
            };
            let call = dispatch(client, name, arguments);
            let outcome = match scope {
                Some(scope) => crate::caller::with_caller(scope, call).await,
                None => call.await,
            };
            match outcome {
                Ok(value) => ok_frame(id.clone(), tool_call_result(value)),
                // A malformed call (unknown tool / missing argument) — see the
                // module docs on why this is a protocol error, not `isError`.
                Err(e) => return Some(err_frame(id, ERR_INVALID_PARAMS, format!("{e:#}"))),
            }
        }
        "ping" if !modern => ok_frame(id.clone(), json!({})),
        "ping" => return Some(err_frame(id, ERR_METHOD_NOT_FOUND, "unknown method 'ping'")),
        other => {
            return Some(err_frame(
                id,
                ERR_METHOD_NOT_FOUND,
                format!("unknown method '{other}'"),
            ))
        }
    };
    Some(finalize_response(
        response,
        modern,
        method,
        crate::SERVER_NAME,
    ))
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
    let mut legacy_initialized = false;

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(line) {
            Ok(frame) => {
                handle_frame_with_legacy_state(&client, &frame, &mut legacy_initialized).await
            }
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

    /// A `tools/call` carrying the node-operator scope Core stamps in `_meta`.
    fn node_call(id: i64, mut params: Value) -> Value {
        params["_meta"] = json!({ "ryu/caller": { "kind": "node" } });
        req(id, "tools/call", params)
    }

    /// A recording engine: answers every route and remembers each request's
    /// path plus the caller-scope headers it carried.
    async fn spawn_scope_recorder() -> (
        String,
        std::sync::Arc<tokio::sync::Mutex<Vec<(String, Option<String>, Option<String>)>>>,
    ) {
        use axum::extract::Request;
        use axum::Router;

        let log = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let recorded = log.clone();
        let app = Router::new().fallback(move |request: Request| {
            let recorded = recorded.clone();
            async move {
                let (path, headers) = {
                    let (parts, _body) = request.into_parts();
                    (parts.uri.path().to_owned(), parts.headers)
                };
                let header = |name: &str| {
                    headers
                        .get(name)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned)
                };
                let entry = (
                    path,
                    header("x-ryu-research-caller"),
                    header("x-ryu-research-caller-user"),
                );
                recorded.lock().await.push(entry);
                axum::Json(json!({ "campaigns": [] }))
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (addr.to_string(), log)
    }

    /// Campaign data is owned: a call without Core's verified caller scope, or
    /// with an unresolved one, never reaches the engine; a verified user's scope
    /// rides every engine hop and cannot be replaced through tool arguments.
    #[tokio::test]
    async fn tool_calls_carry_only_the_host_verified_caller_scope() {
        let (addr, log) = spawn_scope_recorder().await;
        let _g = crate::UpstreamGuard::set(Some(&addr));
        let client = reqwest::Client::new();

        for meta in [
            None,
            Some(json!({ "ryu/caller": { "kind": "unresolved" } })),
            Some(json!({ "ryu/caller": { "kind": "user" } })),
        ] {
            let mut params =
                json!({ "name": "get_campaign", "arguments": { "campaign_id": "victim" } });
            if let Some(meta) = meta.clone() {
                params["_meta"] = meta;
            }
            let resp = handle_frame(&client, &req(20, "tools/call", params))
                .await
                .unwrap();
            assert_eq!(resp["error"]["code"], ERR_INVALID_PARAMS, "{meta:?}");
        }
        assert!(
            log.lock().await.is_empty(),
            "unscoped calls reached the engine"
        );

        let resp = handle_frame(
            &client,
            &req(
                21,
                "tools/call",
                json!({
                    "name": "list_campaigns",
                    // Arguments cannot smuggle a different principal.
                    "arguments": { "_meta": { "ryu/caller": { "kind": "node" } } },
                    "_meta": { "ryu/caller": { "kind": "user", "userId": "bob", "orgId": "org-1" } }
                }),
            ),
        )
        .await
        .unwrap();
        assert!(resp.get("error").is_none(), "{resp}");
        let calls = log.lock().await.clone();
        assert_eq!(
            calls,
            vec![(
                "/campaigns".to_owned(),
                Some("user".to_owned()),
                Some("bob".to_owned())
            )]
        );
    }

    /// `tools/list` keeps the original eight tools and adds the four durable
    /// campaign/reasoning operations, all under bare names.
    #[tokio::test]
    async fn tools_list_advertises_the_twelve_bare_tool_names() {
        let client = reqwest::Client::new();
        let resp = handle_frame(&client, &req(1, "tools/list", json!({})))
            .await
            .expect("tools/list is a request, not a notification");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 12, "got {tools:?}");

        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(
            names,
            vec![
                "list_experiments",
                "init_workspace",
                "list_campaigns",
                "get_campaign",
                "read_file",
                "write_file",
                "run",
                "keep",
                "reset",
                "ledger",
                "next_variant",
                "finish_campaign",
            ]
        );
        // Bare names: the registry applies the `research.<tool>` namespace
        // itself. Pre-qualifying here would yield `research.research.run`.
        assert!(
            names.iter().all(|n| !n.contains('.')),
            "tool names must be bare, got {names:?}"
        );
    }

    /// The manifest key must be `SERVER_NAME`, because that is what makes the
    /// registry's ids come out as the `research.*` the workflow templates already
    /// name. Asserted here so a rename of either side fails loudly.
    #[test]
    fn qualified_ids_match_the_pre_severance_scheme() {
        for spec in tool_specs() {
            let id = format!("{}.{}", crate::SERVER_NAME, spec.name);
            assert!(id.starts_with("research."), "got {id}");
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

        let unknown = handle_frame(&client, &node_call(4, json!({ "name": "no_such_tool" })))
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
            &node_call(5, json!({ "name": "read_file", "arguments": {} })),
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
#[cfg(test)]
mod protocol_tests {
    use super::*;

    fn metadata(version: &str) -> Value {
        json!({
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientCapabilities": {},
        })
    }

    fn request(method: &str, meta: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": { "_meta": meta },
        })
    }

    #[test]
    fn modern_metadata_is_checked_without_downgrading_malformed_meta() {
        let modern = request("tools/list", metadata(MODERN_PROTOCOL_VERSION));
        assert!(request_is_modern(&modern, "tools/list", false).unwrap());

        let malformed = json!({
            "method": "tools/list",
            "params": { "_meta": "malformed" },
        });
        let error = request_is_modern(&malformed, "tools/list", true).unwrap_err();
        assert_eq!(error.code, -32602);

        let unsupported = request("tools/list", metadata("2025-11-25"));
        let error = request_is_modern(&unsupported, "tools/list", false).unwrap_err();
        assert_eq!(error.code, -32022);
        assert_eq!(
            error.data.as_ref().unwrap()["supported"][0],
            MODERN_PROTOCOL_VERSION
        );
        assert_eq!(error.data.as_ref().unwrap()["requested"], "2025-11-25");
    }

    #[test]
    fn discovery_has_result_metadata_but_only_list_results_have_cache_hints() {
        let discover = finalize_response(
            json!({ "jsonrpc": "2.0", "id": 1, "result": discover_result() }),
            true,
            "server/discover",
            "test-server",
        );
        assert_eq!(discover["result"]["resultType"], "complete");
        assert!(discover["result"].get("ttlMs").is_none());
        assert!(discover["result"].get("cacheScope").is_none());
        assert_eq!(
            discover["result"]["_meta"][SERVER_INFO]["name"],
            "test-server"
        );

        let list = finalize_response(
            json!({ "jsonrpc": "2.0", "id": 1, "result": { "tools": [] } }),
            true,
            "tools/list",
            "test-server",
        );
        assert_eq!(list["result"]["resultType"], "complete");
        assert_eq!(list["result"]["ttlMs"], 300_000);
        assert_eq!(list["result"]["cacheScope"], "public");
    }

    #[test]
    fn initialize_clients_keep_the_legacy_wire_shape() {
        let initialize = json!({
            "method": "initialize",
            "params": { "protocolVersion": "2024-11-05" },
        });
        assert!(!request_is_modern(&initialize, "initialize", false).unwrap());

        let legacy = finalize_response(
            json!({ "jsonrpc": "2.0", "id": 2, "result": { "tools": [] } }),
            false,
            "tools/list",
            "test-server",
        );
        assert_eq!(legacy["result"]["tools"], json!([]));
        assert!(legacy["result"].get("resultType").is_none());
        assert!(legacy["result"].get("ttlMs").is_none());
        assert!(legacy["result"].get("_meta").is_none());
    }
}
