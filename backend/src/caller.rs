//! The verified caller scope every engine request carries.
//!
//! Research campaigns are owned. On a shared (organization-bound) node one user's
//! agent must not list, read, or drive another user's campaigns, so every request
//! this crate sends to the Python engine names the principal it acts for, and the
//! engine authorizes the stored owner of every addressed campaign/workspace.
//!
//! The scope is never taken from tool arguments. It comes from host-stamped,
//! out-of-band metadata only:
//!
//! - **MCP (stdio)**: Core resolves the tool call's server-derived principal and
//!   sends it in `params._meta["ryu/caller"]` on the pipe it owns end to end
//!   ([`from_mcp_meta`]). A missing or unresolved scope fails closed.
//! - **HTTP sidecar**: Core's ext-proxy strips caller-supplied identity headers and
//!   stamps `x-ryu-caller-user-id` / `x-ryu-caller-org-id` for a verified user on a
//!   hop authenticated by `RYU_EXT_TOKEN` ([`from_core_headers`]). A hop with no
//!   verified user is Core's own node-token operator.
//!
//! The scope travels to the engine in `X-Ryu-Research-Caller*` headers on the
//! bearer-authenticated loopback hop ([`stamp`]).

use std::future::Future;

use axum::http::HeaderMap;
use serde_json::Value;

/// `_meta` key Core uses for the verified caller scope of a plugin MCP call.
pub const MCP_META_KEY: &str = "ryu/caller";

const ENGINE_CALLER_HEADER: &str = "X-Ryu-Research-Caller";
const ENGINE_CALLER_USER_HEADER: &str = "X-Ryu-Research-Caller-User";
const ENGINE_CALLER_ORG_HEADER: &str = "X-Ryu-Research-Caller-Org";

const CORE_CALLER_USER_HEADER: &str = "x-ryu-caller-user-id";
const CORE_CALLER_ORG_HEADER: &str = "x-ryu-caller-org-id";
const CORE_CALLER_AGENT_HEADER: &str = "x-ryu-caller-agent-id";

const MAX_ID_CHARS: usize = 256;

/// The principal one request acts for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallerScope {
    /// The node operator: a personal (unbound) node's single principal, or Core's
    /// own node-token caller. Sees every campaign.
    Node,
    /// A verified user (and organization, empty when none) on a shared node. Sees
    /// only the campaigns it created.
    User { user_id: String, org_id: String },
}

tokio::task_local! { static CURRENT: CallerScope; }

/// Run `future` with `scope` as the caller every engine request inside it carries.
pub async fn with_caller<F: Future>(scope: CallerScope, future: F) -> F::Output {
    CURRENT.scope(scope, future).await
}

/// The scope of the current task, if one was established.
#[must_use]
pub fn current() -> Option<CallerScope> {
    CURRENT.try_with(Clone::clone).ok()
}

/// Stamp the current caller scope on an engine request. Without a scope the
/// request carries none and the engine rejects every owned route (fail closed).
pub fn stamp(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match current() {
        Some(CallerScope::Node) => request.header(ENGINE_CALLER_HEADER, "node"),
        Some(CallerScope::User { user_id, org_id }) => request
            .header(ENGINE_CALLER_HEADER, "user")
            .header(ENGINE_CALLER_USER_HEADER, user_id)
            .header(ENGINE_CALLER_ORG_HEADER, org_id),
        None => request,
    }
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.chars().count() <= MAX_ID_CHARS
        && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn user_scope(user_id: &str, org_id: Option<&str>) -> Result<CallerScope, &'static str> {
    let user_id = user_id.trim();
    let org_id = org_id.map(str::trim).unwrap_or_default();
    if !valid_id(user_id) {
        return Err("the verified caller user id is malformed");
    }
    if !org_id.is_empty() && !valid_id(org_id) {
        return Err("the verified caller organization id is malformed");
    }
    Ok(CallerScope::User {
        user_id: user_id.to_owned(),
        org_id: org_id.to_owned(),
    })
}

/// Parse the scope Core stamped in an MCP request's `_meta`.
///
/// # Errors
///
/// Fails closed when the metadata is missing, malformed, or explicitly
/// `unresolved` (a shared node could not resolve the call's principal).
pub fn from_mcp_meta(meta: Option<&Value>) -> Result<CallerScope, &'static str> {
    let Some(scope) = meta.and_then(|meta| meta.get(MCP_META_KEY)) else {
        return Err("Research requires the host's verified caller scope");
    };
    match scope.get("kind").and_then(Value::as_str) {
        Some("node") => Ok(CallerScope::Node),
        Some("user") => user_scope(
            scope
                .get("userId")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            scope.get("orgId").and_then(Value::as_str),
        ),
        Some("unresolved") => {
            Err("Research is unavailable here: the caller's identity could not be resolved")
        }
        _ => Err("Research requires the host's verified caller scope"),
    }
}

/// Derive the scope from the identity headers Core's ext-proxy stamps.
///
/// Returns `None` (no scope, so the engine fails closed) for an agent-lane hop
/// that names an agent but no verified user, and for malformed identity values.
#[must_use]
pub fn from_core_headers(headers: &HeaderMap) -> Option<CallerScope> {
    let text = |name: &str| headers.get(name).and_then(|value| value.to_str().ok());
    match text(CORE_CALLER_USER_HEADER) {
        Some(user_id) => user_scope(user_id, text(CORE_CALLER_ORG_HEADER)).ok(),
        None if headers.contains_key(CORE_CALLER_USER_HEADER)
            || headers.contains_key(CORE_CALLER_AGENT_HEADER) =>
        {
            None
        }
        None => Some(CallerScope::Node),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn mcp_meta_parses_verified_scopes_and_fails_closed_otherwise() {
        assert_eq!(
            from_mcp_meta(Some(&json!({ "ryu/caller": { "kind": "node" } }))),
            Ok(CallerScope::Node)
        );
        assert_eq!(
            from_mcp_meta(Some(
                &json!({ "ryu/caller": { "kind": "user", "userId": "u1", "orgId": "o1" } })
            )),
            Ok(CallerScope::User {
                user_id: "u1".into(),
                org_id: "o1".into()
            })
        );
        assert_eq!(
            from_mcp_meta(Some(
                &json!({ "ryu/caller": { "kind": "user", "userId": "u1", "orgId": null } })
            )),
            Ok(CallerScope::User {
                user_id: "u1".into(),
                org_id: String::new()
            })
        );
        for rejected in [
            None,
            Some(json!({})),
            Some(json!({ "ryu/caller": { "kind": "unresolved" } })),
            Some(json!({ "ryu/caller": { "kind": "user" } })),
            Some(json!({ "ryu/caller": { "kind": "user", "userId": "bad\nid" } })),
            Some(json!({ "ryu/caller": { "kind": "admin" } })),
        ] {
            assert!(from_mcp_meta(rejected.as_ref()).is_err(), "{rejected:?}");
        }
    }

    #[test]
    fn core_headers_map_to_user_node_or_no_scope() {
        let mut headers = HeaderMap::new();
        assert_eq!(from_core_headers(&headers), Some(CallerScope::Node));
        headers.insert(CORE_CALLER_AGENT_HEADER, "agent-1".parse().unwrap());
        assert_eq!(from_core_headers(&headers), None);
        headers.insert(CORE_CALLER_USER_HEADER, "u1".parse().unwrap());
        headers.insert(CORE_CALLER_ORG_HEADER, "o1".parse().unwrap());
        assert_eq!(
            from_core_headers(&headers),
            Some(CallerScope::User {
                user_id: "u1".into(),
                org_id: "o1".into()
            })
        );
    }

    #[tokio::test]
    async fn stamp_carries_the_task_scope_only_inside_it() {
        let client = reqwest::Client::new();
        let outside = stamp(client.get("http://127.0.0.1:1/campaigns"))
            .build()
            .unwrap();
        assert!(outside.headers().get(ENGINE_CALLER_HEADER).is_none());
        let inside = with_caller(
            CallerScope::User {
                user_id: "u1".into(),
                org_id: String::new(),
            },
            async {
                stamp(client.get("http://127.0.0.1:1/campaigns"))
                    .build()
                    .unwrap()
            },
        )
        .await;
        assert_eq!(inside.headers()[ENGINE_CALLER_HEADER], "user");
        assert_eq!(inside.headers()[ENGINE_CALLER_USER_HEADER], "u1");
        assert_eq!(inside.headers()[ENGINE_CALLER_ORG_HEADER], "");
    }
}
