//! `ryu-research` — the standalone, out-of-process research sidecar.
//!
//! The out-of-process half of the `ryu_research` crate's in-process/out-of-process
//! duality (the same shape as `ryu-mail`): Core spawns THIS binary, health-checks
//! it, and reverse-proxies `/api/research/*` onto it through the generic ext-proxy
//! loader (`apps/core/src/sidecar/ext_proxy.rs`). Core does NOT contain this code,
//! so the research surface scales and fails independently of the rest of the node.
//!
//! It reuses the very same [`ryu_research::routes`] + [`ryu_research::ResearchCtx`]
//! the in-process merge uses — only nested under `/api/research` (Core forwards the
//! full mount path, `{mount}{sub_path}`) and guarded by the injected bearer.
//!
//! ## Two modes: HTTP sidecar (default) and `mcp` (stdio)
//!
//! Run bare, it is the HTTP sidecar described above. Run as **`ryu-research mcp`**
//! it instead serves the 12 `research.*` tools as a JSON-RPC 2.0 MCP server over
//! stdin/stdout ([`ryu_research::mcp_stdio`]) — the same [`ryu_research::tool_specs`]
//! and [`ryu_research::dispatch`] the crate always owned. Core spawns that through
//! the generic `mcp_servers` entry in `manifest.json`, which replaced the
//! app-specific `sidecar/mcp/research.rs` provider it used to hardcode. The two
//! modes are independent: the MCP mode binds no port and talks to the Python engine
//! directly, so it works whether or not the HTTP sidecar is running.
//!
//! ## Two hops, by design
//!
//! The autoresearch *engine* (git-versioned workspaces, metric ledger) is the
//! dependency-free Python service on :8087. This Rust sidecar is a thin JSON proxy
//! in front of it, so a request is `Core → ryu-research (Rust) → autoresearch
//! (Python :8087)` — TWO loopback hops. Acceptable: both are same-host loopback and
//! the proxied metadata calls are quick; the long work happens inside
//! the Python engine, unaffected by the extra hop.
//!
//! ## Engine lifecycle
//!
//! [`ryu_research::EngineSupervisor`] materializes the embedded Python payload and
//! lazily starts or adopts its authenticated loopback engine. HTTP and MCP modes
//! share the same supervisor contract; Core only starts this generic app sidecar.
//!
//! ## Security
//!
//! Binds LOOPBACK ONLY (127.0.0.1) and guards every route with the shared-secret
//! bearer `RYU_EXT_TOKEN` that Core mints per-plugin and stamps on both the proxied
//! hop and the health probe. Core is the auth front (`require_auth`), then
//! re-stamps `Authorization: Bearer <RYU_EXT_TOKEN>` on the loopback hop, so a
//! request that did NOT come through Core (any other local process on a shared
//! host) is rejected `401`. FAIL-CLOSED: with no token configured every route
//! rejects. Research has no external/public caller (unlike mail's inbound webhook),
//! so ALL routes are protected — there is no public sub-router.
//!
//! Port: `RYU_RESEARCH_PORT` env (Core injects the profile-shifted bind port via
//! the manifest's `port_env`), default `7995`.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::Request,
    http::{header::AUTHORIZATION, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Router,
};
use ryu_research::{routes, EngineSupervisor, ResearchCtx};

/// Default loopback port for the research sidecar (overridable via
/// `RYU_RESEARCH_PORT`). Distinct from browser (7993), mail (7996), the Python
/// autoresearch engine (8087), and every other declared sidecar port.
const DEFAULT_PORT: u16 = 7995;

// ── On-disk install check (faithful, dependency-free copy of Core's resolution) ──
//
// ── Bearer gate (fail-closed; no public routes) ──────────────────────────────

/// Guard every route with the injected shared-secret bearer. FAIL-CLOSED: with no
/// token configured (`expected` is `None`), reject all. Mirrors `ryu-mail`'s
/// `require_mail_token`, minus the public inbound carve-out research doesn't have.
async fn require_ext_token(req: Request, next: Next, expected: Option<&str>) -> Response {
    let Some(expected) = expected else {
        return StatusCode::UNAUTHORIZED.into_response();
    };
    let presented = req
        .headers()
        .get(AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    let ok = presented.is_some_and(|got| {
        ryu_sidecar_runtime::constant_time_eq(got.as_bytes(), expected.as_bytes())
    });
    if ok {
        next.run(req).await
    } else {
        StatusCode::UNAUTHORIZED.into_response()
    }
}

/// The argv sub-command that serves MCP over stdio instead of binding the HTTP
/// listener. Declared in `manifest.json`'s `mcp_servers.research.args`.
const MCP_SUBCOMMAND: &str = "mcp";

#[tokio::main]
async fn main() -> Result<()> {
    // `ryu-research mcp` — serve the `research.*` tools over stdin/stdout instead
    // of binding a port (see `ryu_research::mcp_stdio`). Logs MUST go to stderr
    // here: stdout is the JSON-RPC frame stream, and a single `info!` line on it
    // corrupts the transport. Core's MCP client already forwards a child's stderr
    // into `tracing::debug!(target: "mcp", …)`, so nothing is lost.
    //
    // No listener and no `RYU_EXT_TOKEN` gate in this mode: the "connection" IS the
    // pipe pair Core owns end-to-end, so there is no unauthenticated local caller
    // to fence off — the bearer exists only because the HTTP mode binds a port any
    // process on the host could reach.
    if std::env::args().nth(1).as_deref() == Some(MCP_SUBCOMMAND) {
        tracing_subscriber::fmt()
            .with_writer(std::io::stderr)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "info".into()),
            )
            .init();
        let engine = EngineSupervisor::new();
        engine.ensure_ready().await?;
        return ryu_research::mcp_stdio::run().await;
    }

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let port: u16 = std::env::var("RYU_RESEARCH_PORT")
        .ok()
        .and_then(|p| p.trim().parse().ok())
        .unwrap_or(DEFAULT_PORT);

    // Shared-secret bearer Core injects at spawn (the same `RYU_EXT_TOKEN` it stamps
    // on every proxied hop + the health probe). Every route requires it; FAIL-CLOSED
    // when absent.
    let token = std::env::var("RYU_EXT_TOKEN")
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());
    if token.is_some() {
        tracing::info!("ryu-research: all routes require the injected shared-secret bearer");
    } else {
        tracing::warn!(
            "ryu-research: no RYU_EXT_TOKEN set; all /api/research/* routes are FAIL-CLOSED (reject all). Core injects this token when it spawns the sidecar."
        );
    }

    // Reuse the crate's in-process router; nest it under `/api/research` because Core
    // forwards the FULL mount path (`{mount}{sub_path}`, e.g. `/api/research/status`).
    //
    // `/openapi.json` rides INSIDE the same bearer gate, at the SERVER ROOT. Core
    // fetches `http://127.0.0.1:<port>/openapi.json` on this sidecar's first Healthy
    // edge and lowers every operation it finds into searchable LLM tools, so routing
    // this one endpoint is what makes the whole `/api/research` surface callable by an
    // agent (`ryu_research::api::openapi()` was dead code until now — only tests read
    // it).
    //
    // Root, not under `/api/research`: Core tries the root FIRST and only falls back
    // to the mount-prefixed form, and keeping the document off the mount keeps it out
    // of the manifest's declared `http.routes[]` — anything declared there is
    // reachable through the generic ext-proxy, and the schema is Core's to read, not
    // an app surface. Inside the gate: Core stamps the injected `RYU_EXT_TOKEN` on the
    // fetch (the Python sidecars already require the bearer for everything but
    // `/health`), so the gate costs the fetcher nothing — while un-gated it would
    // disclose this app's entire internal API surface to any other process on
    // loopback.
    let engine = Arc::new(EngineSupervisor::new());
    let ctx = ResearchCtx::new(engine);
    let app = Router::new()
        .nest("/api/research", routes(ctx))
        .route("/health", axum::routing::get(|| async { "ok" }))
        .route(
            "/openapi.json",
            axum::routing::get(|| async { axum::Json(ryu_research::api::openapi()) }),
        )
        .layer(axum::middleware::from_fn(
            move |req: Request, next: Next| {
                let expected = token.clone();
                async move { require_ext_token(req, next, expected.as_deref()).await }
            },
        ));

    // LOOPBACK ONLY (belt) + shared-secret bearer (suspenders): Core is the auth
    // front and re-stamps the bearer on the proxied hop.
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("ryu-research sidecar listening on http://{addr}");

    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest is what actually spawns this binary in MCP mode, so its
    /// declaration and [`MCP_SUBCOMMAND`] must not drift. The server KEY is the
    /// load-bearing part: Core's registry ids are `<key>.<tool>`, so anything but
    /// `SERVER_NAME` silently renames all Research tools out from under the workflow
    /// templates that call `research.run` / `research.init_workspace`.
    #[test]
    fn manifest_mcp_declaration_matches_this_binary() {
        let manifest: serde_json::Value =
            serde_json::from_str(include_str!("../../manifest.json")).expect("manifest parses");
        let servers = manifest["mcp_servers"]
            .as_object()
            .expect("manifest declares mcp_servers");
        assert_eq!(servers.len(), 1, "one server: {servers:?}");
        let decl = &servers[ryu_research::SERVER_NAME];
        assert_eq!(decl["command"], env!("CARGO_BIN_NAME"));
        assert_eq!(decl["args"], serde_json::json!([MCP_SUBCOMMAND]));
        assert_eq!(
            decl["command_env"], "RYU_RESEARCH_BIN",
            "the dev-override env must match the sidecar entry's command_env"
        );
        // The sidecar entry and the MCP entry name the SAME binary — that is what
        // lets the managed-bin resolver find it, and what the ready notifier in
        // Core keys on when the download lands.
        assert_eq!(
            manifest["sidecars"][0]["process"]["command"],
            decl["command"]
        );
    }

    #[test]
    fn ct_eq_matches_only_identical_bytes() {
        assert!(ryu_sidecar_runtime::constant_time_eq(
            b"secret-token",
            b"secret-token"
        ));
        assert!(!ryu_sidecar_runtime::constant_time_eq(
            b"secret-token",
            b"secret-toke"
        ));
        assert!(!ryu_sidecar_runtime::constant_time_eq(
            b"secret-token",
            b"wrong-token!"
        ));
        assert!(!ryu_sidecar_runtime::constant_time_eq(b"", b"x"));
        assert!(ryu_sidecar_runtime::constant_time_eq(b"", b""));
    }

    // ── Fail-closed bearer gate (security-critical) ─────────────────────────────
    //
    // `require_ext_token`/`Next` have no public constructors, so we exercise the
    // REAL middleware by serving the same guarded router shape `main()` builds on an
    // ephemeral loopback port and driving it with real requests.

    async fn spawn_guarded(token: Option<String>) -> String {
        use axum::routing::get;
        let app = Router::new()
            .route("/guarded", get(|| async { "ok" }))
            .layer(axum::middleware::from_fn(
                move |req: Request, next: Next| {
                    let expected = token.clone();
                    async move { require_ext_token(req, next, expected.as_deref()).await }
                },
            ));
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn no_token_configured_rejects_all_fail_closed() {
        let base = spawn_guarded(None).await;
        let client = reqwest::Client::new();
        // Even a well-formed bearer is rejected when the server has no expected token.
        let resp = client
            .get(format!("{base}/guarded"))
            .header("Authorization", "Bearer anything")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 401);
    }

    #[tokio::test]
    async fn correct_bearer_passes_wrong_and_missing_rejected() {
        let base = spawn_guarded(Some("s3cr3t".to_owned())).await;
        let client = reqwest::Client::new();

        // Correct token → 200.
        let ok = client
            .get(format!("{base}/guarded"))
            .header("Authorization", "Bearer s3cr3t")
            .send()
            .await
            .unwrap();
        assert_eq!(ok.status().as_u16(), 200);

        // Wrong token → 401.
        let wrong = client
            .get(format!("{base}/guarded"))
            .header("Authorization", "Bearer nope")
            .send()
            .await
            .unwrap();
        assert_eq!(wrong.status().as_u16(), 401);

        // No Authorization header → 401.
        let missing = client.get(format!("{base}/guarded")).send().await.unwrap();
        assert_eq!(missing.status().as_u16(), 401);

        // Non-Bearer scheme → 401.
        let basic = client
            .get(format!("{base}/guarded"))
            .header("Authorization", "Basic s3cr3t")
            .send()
            .await
            .unwrap();
        assert_eq!(basic.status().as_u16(), 401);
    }

    #[tokio::test]
    async fn bearer_value_is_trimmed_before_compare() {
        // The gate trims the presented token, so surrounding spaces still match.
        let base = spawn_guarded(Some("tok".to_owned())).await;
        let client = reqwest::Client::new();
        let resp = client
            .get(format!("{base}/guarded"))
            .header("Authorization", "Bearer   tok  ")
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200);
    }
}
