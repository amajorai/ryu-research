//! Autoresearch data path — proxies `/api/research/*` to the research sidecar.
//!
//! The sidecar (`apps-store/research/sidecar`, Python stdlib HTTP on :8087) owns
//! the git-versioned experiment workspaces + run/ledger machinery; this module is
//! a thin proxy that forwards JSON to it, plus a `status` endpoint that reports
//! install/run state and mirrors the sidecar's experiment catalog.
//!
//! Per the Core-vs-Gateway rule this is **Core** — it decides *what runs* (which
//! experiment, in which workspace). The researcher agent's own model calls stay
//! Gateway-governed. The same sidecar calls are also exposed as `research.*`
//! MCP tools ([`crate::dispatch`]) so workflow `agent`/`tool` nodes drive the loop.
//!
//! The router is built with its own state ([`ResearchCtx`]) inside this crate so it
//! returns a state-less, mergeable `Router<()>`. Routes are declared relative to
//! `/api/research` (Core nests this service at that prefix behind the Research-App
//! gate), while the OpenAPI annotations keep the full external paths.

use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};

use crate::reasoning::{CapabilityBroker, CoreCapabilityClient};
use crate::ResearchHost;

/// This app's plugin id — the namespace half of every event id it may emit. Must
/// match `apps-store/research/manifest.json`'s `id` exactly or Core rejects the emit.
const PLUGIN_ID: &str = "@ryu/research";

/// Event: a fresh experiment workspace was initialised and git-committed.
const EV_WORKSPACE_CREATED: &str = "@ryu/research#workspace.created";

/// Router state for the research HTTP surface: the [`ResearchHost`] that lazily
/// starts the app-owned engine and reports its install state. Cloneable so the
/// router bakes a concrete state and returns `Router<()>`.
#[derive(Clone)]
pub struct ResearchCtx {
    host: Arc<dyn ResearchHost>,
    /// Raises `@ryu/research/*` app events back into Core. Built here rather than
    /// passed in so the constructor's signature (and every caller) is unchanged; it
    /// is cheap to clone and degrades entirely when the process is not Core-hosted —
    /// which is exactly the in-process/library and unit-test state.
    events: ryu_app_events::EventEmitter,
    capabilities: Option<Arc<dyn CapabilityBroker>>,
}

impl ResearchCtx {
    pub fn new(host: Arc<dyn ResearchHost>) -> Self {
        Self {
            host,
            events: ryu_app_events::EventEmitter::from_env(PLUGIN_ID),
            capabilities: CoreCapabilityClient::from_env(),
        }
    }
}

/// Build the `/api/research/*` router with its own state baked in, returning a
/// state-less `Router<()>` the host nests at `/api/research` behind the App gate.
pub fn routes(ctx: ResearchCtx) -> Router<()> {
    Router::new()
        .route("/status", get(research_status))
        .route("/campaigns", get(research_campaigns))
        .route("/campaigns/:id", get(research_campaign))
        .route("/campaigns/:id/analyze", post(research_analyze_campaign))
        .route("/workspace", post(research_init_workspace))
        .route("/workspace/:id/ledger", get(research_ledger))
        .with_state(ctx)
}

/// The OpenAPI sub-document for the research surface, merged into Core's spec when
/// the `research` feature is enabled.
pub fn openapi() -> utoipa::openapi::OpenApi {
    <ResearchApiDoc as utoipa::OpenApi>::openapi()
}

/// The document Core imports. `components(schemas(...))` is what turns
/// `request_body = InitWorkspaceBody` into a resolvable
/// `#/components/schemas/InitWorkspaceBody` entry: without it the operation still
/// carries a `$ref`, but the target is missing and Core's `resolve_ref` yields
/// nothing — a derived write tool with zero visible arguments. utoipa 5 also
/// auto-collects schemas reachable from the annotated paths, so this row is
/// belt-and-braces; it is listed explicitly anyway so the registration is
/// greppable and cannot be silently lost to an attribute edit.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        research_init_workspace,
        research_ledger,
        research_status,
        research_campaigns,
        research_campaign,
        research_analyze_campaign
    ),
    components(schemas(InitWorkspaceBody))
)]
struct ResearchApiDoc;

/// Request body for `POST /api/research/workspace`.
// Everything below is `//`, not `///`, ON PURPOSE: utoipa lifts a struct's doc
// comment into the schema's own `description`, so internal rationale written as
// `///` ships to the model alongside the arguments.
//
// The FIELD doc below is the opposite — utoipa lifts it into the property's
// `description`, and it is the only prose the model reads when choosing the
// argument. Before this type existed the annotation said
// `request_body = serde_json::Value`, so the only write tool this app exposes
// reached the model with no arguments at all: it could see "init a research
// workspace" and had no way to say which experiment.
//
// This type describes the wire shape; it is deliberately NOT used as the axum
// extractor, and it derives no `Deserialize`. `research_init_workspace` is a proxy:
// it forwards the body verbatim to the Python sidecar
// (`apps-store/research/sidecar`), which owns the contract of record. A Rust struct
// in the extract path would make this crate a gatekeeper for a schema it does not
// own, and would also reject a malformed body BEFORE the lazy `start_sidecar()`
// that this handler runs first. The annotation is the half Core reads, so typing it
// buys the argument without changing a single response.
#[derive(Debug, utoipa::ToSchema)]
pub struct InitWorkspaceBody {
    /// Name of the experiment to open a workspace for — one of the ids listed by
    /// `GET /api/research/status` under `experiments`.
    pub experiment: String,
    /// Human-readable campaign objective and metric direction.
    pub goal: Option<String>,
    /// Optional campaign label shown in Experiments.
    pub name: Option<String>,
    /// Character count at which a current RLM analysis becomes mandatory before
    /// another attempt can start. Defaults to the engine's safe threshold.
    pub reasoning_threshold_chars: Option<u64>,
}

/// Runs can be long, but these proxied calls (status/init/ledger) are quick.
/// A generous-but-bounded client keeps a hung sidecar from wedging the request.
fn research_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("ryu-core/0.1")
        .timeout(Duration::from_secs(30))
        .build()
        .expect("reqwest client")
}

/// `GET /api/research/status` — report install/run state and the sidecar's
/// experiment catalog. `running` is `false` (and `experiments` empty) when the
/// sidecar is not answering; never force-starts it.
#[utoipa::path(
    get,
    path = "/api/research/status",
    tag = "Research",
    summary = "report install/run state and the sidecar's",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn research_status(State(ctx): State<ResearchCtx>) -> impl IntoResponse {
    let client = research_client();
    let installed = ctx.host.is_installed();
    let running = crate::is_running_now(&client).await;

    let experiments = if running {
        match authorize_engine(client.get(format!("{}/experiments", crate::research_base_url())))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => resp
                .json::<Value>()
                .await
                .ok()
                .and_then(|v| v.get("experiments").cloned())
                .unwrap_or_else(|| json!([])),
            _ => json!([]),
        }
    } else {
        json!([])
    };

    Json(json!({
        "installed": installed,
        "running": running,
        "experiments": experiments,
    }))
}

/// `POST /api/research/workspace` — init a new experiment workspace. Lazily
/// starts the (off-by-default) sidecar so the flow works once installed, then
/// proxies to the sidecar's `POST /workspace/init`. On success it also raises
/// [`EV_WORKSPACE_CREATED`], so hooks and workflows can pick the loop up from here.
#[utoipa::path(
    post,
    path = "/api/research/workspace",
    tag = "Research",
    summary = "init a new experiment workspace. Lazily",
    request_body = InitWorkspaceBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn research_init_workspace(
    State(ctx): State<ResearchCtx>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    if let Err(e) = ctx.host.start_sidecar().await {
        tracing::debug!("research lazy start skipped: {e:#}");
    }
    let (status, Json(value)) = proxy_post("/workspace/init", body).await;

    // Raise the app event only when a workspace genuinely came into existence.
    // `pass_through` forwards the sidecar's error bodies with their status, so
    // gating on `workspace_id` (the init envelope's proof of creation) rather than
    // on "the call returned" is what keeps a failed init from firing a lie.
    if status.is_success() {
        if let Some(workspace_id) = value.get("workspace_id").and_then(Value::as_str) {
            ctx.events
                .emit(
                    EV_WORKSPACE_CREATED,
                    json!({
                        "workspace_id": workspace_id,
                        "experiment": value.get("experiment").cloned().unwrap_or(Value::Null),
                        // The files a researcher may edit — enough for a consumer to
                        // act on the new workspace. `program_md` is deliberately left
                        // out: it is a multi-KB instruction document, not a signal.
                        "mutable_files": value
                            .get("mutable_files")
                            .cloned()
                            .unwrap_or_else(|| json!([])),
                    }),
                )
                .await;
        }
    }

    (status, Json(value))
}

/// `GET /api/research/workspace/:id/ledger` — proxy the sidecar's ledger read.
#[utoipa::path(
    get,
    path = "/api/research/workspace/{id}/ledger",
    tag = "Research",
    summary = "proxy the sidecar's ledger read.",
    params(("id" = String, Path)),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
pub async fn research_ledger(
    State(ctx): State<ResearchCtx>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    // A ledger read is a real data request (unlike the passive `/status` poll), so
    // it wakes the idle-stopped sidecar on demand — the scale-from-zero half of the
    // Rivet-style idle-stop. Lazy-start (via the host) also refreshes the sidecar's
    // idle clock in Core, so an actively-read workspace is never reaped.
    if let Err(e) = ctx.host.start_sidecar().await {
        tracing::debug!("research lazy start skipped: {e:#}");
    }
    proxy_get(&format!("/workspace/{id}/ledger")).await
}

/// Forward a JSON body to a sidecar endpoint and pass the response through.
async fn proxy_post(endpoint: &str, body: Value) -> (StatusCode, Json<Value>) {
    let url = format!("{}{endpoint}", crate::research_base_url());
    let resp = match authorize_engine(research_client().post(&url))
        .json(&body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return unreachable_err(&url, e),
    };
    pass_through(resp).await
}

/// Forward a GET to a sidecar endpoint and pass the response through.
async fn proxy_get(endpoint: &str) -> (StatusCode, Json<Value>) {
    let url = format!("{}{endpoint}", crate::research_base_url());
    let resp = match authorize_engine(research_client().get(&url)).send().await {
        Ok(r) => r,
        Err(e) => return unreachable_err(&url, e),
    };
    pass_through(resp).await
}

/// `GET /api/research/campaigns` — list durable experiment campaigns. This is
/// the read model used by the Experiments companion; elapsed time is derived from
/// stored timestamps rather than persisted by a polling loop.
#[utoipa::path(
    get,
    path = "/api/research/campaigns",
    tag = "Research",
    summary = "List durable autoresearch campaigns",
    responses((status = 200, description = "Campaign summaries", body = serde_json::Value))
)]
pub async fn research_campaigns(State(ctx): State<ResearchCtx>) -> impl IntoResponse {
    wake_engine(&ctx).await;
    proxy_get("/campaigns").await
}

/// `GET /api/research/campaigns/:id` — load one campaign, its attempts, and
/// bounded RLM reasoning references.
#[utoipa::path(
    get,
    path = "/api/research/campaigns/{id}",
    tag = "Research",
    summary = "Read one autoresearch campaign with attempts and reasoning",
    params(("id" = String, Path)),
    responses((status = 200, description = "Campaign detail", body = serde_json::Value))
)]
pub async fn research_campaign(
    State(ctx): State<ResearchCtx>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    wake_engine(&ctx).await;
    if !valid_campaign_id(&id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "invalid campaign id" })),
        );
    }
    // The companion is the human-facing history browser, so request the complete
    // database projection. The MCP `get_campaign` tool intentionally omits this
    // flag and receives only the bounded recent window that is safe for a prompt.
    proxy_get(&format!("/campaigns/{id}?full=1")).await
}

/// `POST /api/research/campaigns/:id/analyze` — send the durable campaign
/// snapshot to the bound RLM provider without placing the history in the caller's
/// prompt. RLM keeps the recursive trace; Research stores only its ids and bounded
/// next-variant recommendation.
#[utoipa::path(
    post,
    path = "/api/research/campaigns/{id}/analyze",
    tag = "Research",
    summary = "Recursively analyze oversized campaign history before the next variant",
    params(("id" = String, Path)),
    responses((status = 200, description = "Recorded RLM analysis", body = serde_json::Value))
)]
#[allow(unreachable_code, unused_variables)]
pub async fn research_analyze_campaign(
    State(ctx): State<ResearchCtx>,
    Path(id): Path<String>,
    _request: axum::extract::Request,
) -> impl IntoResponse {
    if !valid_campaign_id(&id) {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "invalid campaign id" })),
        );
    }
    wake_engine(&ctx).await;

    // The request body is intentionally ignored. Research owns the query and the
    // proof policy, so callers can trigger preparation but cannot author evidence.
    let body = Value::Null;
    return match crate::reasoning::next_variant(
        &research_client(),
        ctx.capabilities.as_deref(),
        &id,
    )
    .await
    {
        Ok(value) => (StatusCode::OK, Json(value)),
        Err(error) => (
            StatusCode::FAILED_DEPENDENCY,
            Json(json!({ "error": error.to_string() })),
        ),
    };

    let (campaign_status, Json(campaign_envelope)) = proxy_get(&format!("/campaigns/{id}")).await;
    if !campaign_status.is_success() {
        return (campaign_status, Json(campaign_envelope));
    }
    let campaign = campaign_envelope
        .get("campaign")
        .unwrap_or(&campaign_envelope);
    if !campaign
        .get("reasoning_required")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return (
            StatusCode::CONFLICT,
            Json(json!({
                "error": "campaign history is still within the direct-reasoning threshold",
                "reasoning_required": false
            })),
        );
    }

    let Some(capabilities) = &ctx.capabilities else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": "RLM analysis is unavailable outside a Core-hosted Research sidecar"
            })),
        );
    };
    // The detail response is deliberately bounded for agent callers. Fetch the
    // full append-readable JSONL projection only inside this server-to-server hop;
    // it is handed to RLM as a context variable and never enters the caller prompt.
    let (history_status, Json(history_envelope)) =
        proxy_get(&format!("/campaigns/{id}/history")).await;
    if !history_status.is_success() {
        return (history_status, Json(history_envelope));
    }
    let Some(history) = history_envelope.get("document").and_then(Value::as_str) else {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": "research engine returned no campaign history document" })),
        );
    };
    let query = body
        .get("query")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|query| !query.is_empty())
        .unwrap_or(
            "Analyze this campaign's full experiment history. Identify exhausted ideas, failures, interactions, and the single best next hypothesis. Return one concrete, testable variant.",
        );
    let rlm = match capabilities
        .call(
            "rlm.query",
            &json!({
                "documents": [{
                    "source": format!("campaign:{id}/attempts.json"),
                    "text": history
                }],
                "context_name": format!("Experiment campaign {id}"),
                "query": query,
                "max_steps": 12,
                "max_model_calls": 64,
                "max_depth": 2,
                "wall_secs": 180
            }),
        )
        .await
    {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::FAILED_DEPENDENCY,
                Json(json!({ "error": error.to_string() })),
            )
        }
    };

    let recommendation = rlm
        .get("answer")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let context_id = rlm
        .get("context_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let run_id = rlm.get("id").and_then(Value::as_str).unwrap_or_default();
    if recommendation.is_empty() || context_id.is_empty() || run_id.is_empty() {
        return (
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": "RLM returned an incomplete analysis record" })),
        );
    }
    let after_attempt_ordinal = campaign
        .get("attempt_count")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            campaign
                .get("attempts")
                .and_then(Value::as_array)
                .map_or(0, |attempts| attempts.len() as u64)
        });
    let (record_status, Json(record)) = proxy_post(
        &format!("/campaigns/{id}/reasoning"),
        json!({
            "after_attempt_ordinal": after_attempt_ordinal,
            "rlm_context_id": context_id,
            "rlm_run_id": run_id,
            "recommendation": recommendation
        }),
    )
    .await;
    if !record_status.is_success() {
        return (record_status, Json(record));
    }
    (
        StatusCode::OK,
        Json(json!({ "analysis": record, "rlm": rlm })),
    )
}

async fn wake_engine(ctx: &ResearchCtx) {
    if let Err(error) = ctx.host.start_sidecar().await {
        tracing::debug!("research lazy start skipped: {error:#}");
    }
}

fn valid_campaign_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}

fn authorize_engine(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    if let Some(token) = crate::research_engine_token() {
        request.bearer_auth(token)
    } else {
        request
    }
}

fn unreachable_err(url: &str, e: reqwest::Error) -> (StatusCode, Json<Value>) {
    (
        StatusCode::BAD_GATEWAY,
        Json(json!({
            "error": format!(
                "research sidecar not reachable at {url}: {e}. Install it from the Store \
                 (or run `python -m ryu_research`) first."
            )
        })),
    )
}

async fn pass_through(resp: reqwest::Response) -> (StatusCode, Json<Value>) {
    let status = resp.status();
    let bytes = resp.bytes().await.unwrap_or_default();
    let value: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&bytes) }));
    if !status.is_success() {
        let code = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        return (code, Json(value));
    }
    (StatusCode::OK, Json(value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use axum::body::to_bytes;
    use axum::routing::{get, post};

    use crate::UpstreamGuard;

    /// A test double for the kernel coupling: records `start_sidecar` calls and can
    /// be told to fail that call (to pin that a start error does NOT abort the proxy).
    struct FakeHost {
        installed: bool,
        start_err: bool,
        start_calls: AtomicUsize,
    }

    impl FakeHost {
        fn new(installed: bool, start_err: bool) -> Arc<Self> {
            Arc::new(Self {
                installed,
                start_err,
                start_calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ResearchHost for FakeHost {
        async fn start_sidecar(&self) -> anyhow::Result<()> {
            self.start_calls.fetch_add(1, Ordering::SeqCst);
            if self.start_err {
                anyhow::bail!("simulated start failure");
            }
            Ok(())
        }
        fn is_installed(&self) -> bool {
            self.installed
        }
    }

    /// Spawn `app` on an ephemeral loopback port; return its `host:port` string.
    async fn spawn(app: Router<()>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr.to_string()
    }

    /// Drain an `IntoResponse` into (status, parsed-json).
    async fn read(resp: axum::response::Response) -> (StatusCode, Value) {
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let value = serde_json::from_slice(&bytes).unwrap_or(json!(null));
        (status, value)
    }

    fn ctx(host: Arc<FakeHost>) -> ResearchCtx {
        ResearchCtx::new(host)
    }

    #[tokio::test]
    async fn status_reports_not_running_when_upstream_dead() {
        // Point at a port with no listener → is_running_now() is false, so
        // experiments stays [] and running is false; installed mirrors the host.
        let _g = UpstreamGuard::set(Some("127.0.0.1:1"));
        let host = FakeHost::new(true, false);
        let resp = research_status(State(ctx(host))).await.into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["installed"], json!(true));
        assert_eq!(body["running"], json!(false));
        assert_eq!(body["experiments"], json!([]));
    }

    #[tokio::test]
    async fn status_installed_false_is_reported_through() {
        let _g = UpstreamGuard::set(Some("127.0.0.1:1"));
        let host = FakeHost::new(false, false);
        let resp = research_status(State(ctx(host))).await.into_response();
        let (_code, body) = read(resp).await;
        assert_eq!(body["installed"], json!(false));
    }

    #[tokio::test]
    async fn status_running_passes_through_experiment_catalog() {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/experiments",
                get(|| async {
                    Json(json!({ "experiments": [{ "id": "toy" }, { "id": "nanochat" }] }))
                }),
            );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_status(State(ctx(host))).await.into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["running"], json!(true));
        assert_eq!(
            body["experiments"],
            json!([{ "id": "toy" }, { "id": "nanochat" }])
        );
    }

    #[tokio::test]
    async fn status_running_but_experiments_errors_yields_empty_list() {
        let app = Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/experiments",
                get(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
            );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_status(State(ctx(host))).await.into_response();
        let (_code, body) = read(resp).await;
        assert_eq!(body["running"], json!(true));
        assert_eq!(body["experiments"], json!([]));
    }

    #[tokio::test]
    async fn init_workspace_starts_sidecar_and_proxies_body() {
        let app = Router::new().route(
            "/workspace/init",
            post(|Json(b): Json<Value>| async move { Json(json!({ "echo": b })) }),
        );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_init_workspace(
            State(ctx(host.clone())),
            Json(json!({ "experiment": "toy" })),
        )
        .await
        .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["echo"]["experiment"], json!("toy"));
        // Lazy-start was attempted exactly once.
        assert_eq!(host.start_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn init_workspace_proceeds_even_if_lazy_start_errors() {
        let app = Router::new().route(
            "/workspace/init",
            post(|| async { Json(json!({ "ok": true })) }),
        );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        // Host's start_sidecar fails — the request must still proxy through.
        let host = FakeHost::new(true, true);
        let resp = research_init_workspace(State(ctx(host.clone())), Json(json!({})))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["ok"], json!(true));
        assert_eq!(host.start_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn init_workspace_passes_the_init_envelope_through_untouched() {
        // The success path also raises `workspace.created`. That emit is best-effort
        // and unhosted here (no RYU_BIND/RYU_EXT_TOKEN), so what this pins is that
        // reading the body to build the payload does not consume or reshape it — the
        // caller still gets the sidecar's full init envelope.
        let app = Router::new().route(
            "/workspace/init",
            post(|| async {
                Json(json!({
                    "workspace_id": "9f3c1b2a",
                    "experiment": "toy",
                    "mutable_files": ["train.py"],
                    "program_md": "# instructions"
                }))
            }),
        );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_init_workspace(State(ctx(host)), Json(json!({ "experiment": "toy" })))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["workspace_id"], json!("9f3c1b2a"));
        assert_eq!(body["mutable_files"], json!(["train.py"]));
        assert_eq!(body["program_md"], json!("# instructions"));
    }

    #[tokio::test]
    async fn init_workspace_unreachable_upstream_is_502_with_hint() {
        let _g = UpstreamGuard::set(Some("127.0.0.1:1"));
        let host = FakeHost::new(true, false);
        let resp = research_init_workspace(State(ctx(host)), Json(json!({})))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::BAD_GATEWAY);
        let err = body["error"].as_str().unwrap();
        assert!(err.contains("not reachable"));
        assert!(err.contains("Store"));
    }

    #[tokio::test]
    async fn ledger_read_proxies_and_wakes_sidecar() {
        let app = Router::new().route(
            "/workspace/:id/ledger",
            get(|Path(id): Path<String>| async move { Json(json!({ "ws": id, "rows": [] })) }),
        );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_ledger(State(ctx(host.clone())), Path("ws-42".to_owned()))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["ws"], json!("ws-42"));
        assert_eq!(host.start_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn campaign_detail_requests_the_complete_companion_projection() {
        let app = Router::new().route(
            "/campaigns/:id",
            get(|uri: axum::http::Uri| async move {
                Json(json!({ "query": uri.query(), "campaign": { "id": "campaign-42" } }))
            }),
        );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_campaign(State(ctx(host.clone())), Path("campaign-42".to_owned()))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["query"], json!("full=1"));
        assert_eq!(host.start_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn campaign_detail_rejects_unsafe_ids_before_proxying() {
        let host = FakeHost::new(true, false);
        let resp = research_campaign(State(ctx(host.clone())), Path("../secret".to_owned()))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"], json!("invalid campaign id"));
        assert_eq!(host.start_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn pass_through_preserves_non_success_status_and_body() {
        // A 404 from the sidecar must be surfaced with its status, not masked as 200.
        let app = Router::new().route(
            "/workspace/:id/ledger",
            get(|| async {
                (
                    StatusCode::NOT_FOUND,
                    Json(json!({ "error": "no such workspace" })),
                )
            }),
        );
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_ledger(State(ctx(host)), Path("missing".to_owned()))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::NOT_FOUND);
        assert_eq!(body["error"], json!("no such workspace"));
    }

    #[tokio::test]
    async fn pass_through_wraps_non_json_body_as_raw() {
        // Sidecar returns 200 with a plain-text body → wrapped as { "raw": ... }.
        let app = Router::new().route("/workspace/init", post(|| async { "not json at all" }));
        let addr = spawn(app).await;
        let _g = UpstreamGuard::set(Some(&addr));

        let host = FakeHost::new(true, false);
        let resp = research_init_workspace(State(ctx(host)), Json(json!({})))
            .await
            .into_response();
        let (code, body) = read(resp).await;
        assert_eq!(code, StatusCode::OK);
        assert_eq!(body["raw"], json!("not json at all"));
    }

    #[test]
    fn routes_builds_a_stateless_router() {
        // The public router constructor bakes its state and returns Router<()>.
        let host = FakeHost::new(false, false);
        let _r: Router<()> = routes(ResearchCtx::new(host));
    }

    #[test]
    fn openapi_documents_all_research_paths() {
        let spec = openapi();
        let paths = &spec.paths.paths;
        assert!(paths.contains_key("/api/research/status"));
        assert!(paths.contains_key("/api/research/workspace"));
        assert!(paths.contains_key("/api/research/workspace/{id}/ledger"));
        assert!(paths.contains_key("/api/research/campaigns"));
        assert!(paths.contains_key("/api/research/campaigns/{id}"));
        assert!(paths.contains_key("/api/research/campaigns/{id}/analyze"));
    }

    // ── OpenAPI document ───────────────────────────────────────────────────────

    /// This app's own manifest, read at compile time. The route contract lives there,
    /// so the invariants below compare the document against the real declaration
    /// rather than against a second list that could drift from it.
    fn openapi_manifest() -> serde_json::Value {
        serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON")
    }

    /// The manifest sidecar whose HTTP surface this router serves: the one that
    /// declares an `http.mount`. Selected BY mount rather than by index because an app
    /// may declare a second, mountless sidecar (finetune already does), and
    /// `sidecars[0]` would then quietly start asserting against the wrong process.
    fn mounted_sidecar() -> serde_json::Value {
        openapi_manifest()["sidecars"]
            .as_array()
            .expect("sidecars must be an array")
            .iter()
            .find(|s| s["http"]["mount"].is_string())
            .expect("one sidecar must declare an http.mount")
            .clone()
    }

    /// A manifest route (relative to the mount, in axum's `:param` form) rewritten
    /// into the form the OpenAPI document uses (absolute, in `{param}` form).
    ///
    /// The two forms differ ON PURPOSE — the router registers paths relative to the
    /// mount because Core nests it there, while the `#[utoipa::path]` annotations carry
    /// the absolute EXTERNAL path a caller actually hits. Normalise here; do not
    /// "align" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn openapi_doc_is_served_and_non_empty() {
        // The doc is no longer dead code: Core fetches it to derive tools.
        assert!(!super::openapi().paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The direction that decides tool yield. Core's `ext_api::lower` keeps only the
        // document operations the manifest ALSO declares, so a declared route with no
        // `#[utoipa::path]` annotation is a tool that silently never exists — nothing
        // errors, the agent simply cannot call it. (The other direction is harmless: an
        // annotated path the manifest does not declare is dropped by the same filter.)
        let sidecar = mounted_sidecar();
        let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
        let doc = super::openapi();
        for route in sidecar["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
        {
            let path = route["path"].as_str().expect("a route path");
            let expected = doc_path_for(mount, path);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{path}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    // ── Request-body schema ────────────────────────────────────────────────────

    /// The one pointer Core reads to give a derived write tool its arguments.
    fn body_schema(wire: &Value, path: &str, method: &str) -> Value {
        wire.pointer(&format!(
            "/paths/{}/{method}/requestBody/content/application~1json/schema",
            path.replace('/', "~1")
        ))
        .unwrap_or_else(|| panic!("{method} {path} must declare a JSON request body"))
        .clone()
    }

    #[test]
    fn post_routes_document_their_request_body() {
        // The regression this locks down: the annotation used to say
        // `request_body = serde_json::Value`, which serialises to an untyped schema.
        // Core derives a tool per operation and fills `input_schema` from THIS node,
        // so an untyped body produced a tool the model could discover, could call,
        // and could never pass a single argument to — and `experiment` is the ONLY
        // argument this app's only write route takes, so the tool was inert.
        //
        // A `$ref` is the CORRECT and expected shape, not a near-miss: Core's
        // `openapi_import::resolve_ref` resolves it against `components.schemas`
        // before reading `properties`. So accept either a ref or inlined properties;
        // asserting "inlined" would fail on a healthy document.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let schema = body_schema(&wire, "/api/research/workspace", "post");
        assert!(
            schema.get("$ref").is_some() || schema.get("properties").is_some(),
            "a derived write tool for POST /api/research/workspace would have no arguments: \
             {schema}"
        );
    }

    #[test]
    fn every_request_body_ref_resolves_against_components() {
        // The half of the retrofit that a `$ref`-shaped assertion alone cannot see:
        // a `$ref` pointing at a schema that was never registered in
        // `components(schemas(...))` looks identical in the operation and still
        // yields zero arguments once Core tries to resolve it. Walk every request
        // body in the document and check the target actually exists and carries
        // properties.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let paths = wire["paths"].as_object().expect("paths must be an object");
        let mut checked = 0usize;
        for (path, item) in paths {
            for (method, op) in item.as_object().expect("a path item is an object") {
                let Some(schema) = op.pointer("/requestBody/content/application~1json/schema")
                else {
                    continue;
                };
                let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
                    // Inlined schemas are fine as long as they describe something.
                    // The failure this catches in practice is `request_body =
                    // Option<T>`, which utoipa renders as a nullable `oneOf` wrapper:
                    // Core resolves only a TOP-LEVEL `$ref`, so the wrapper reaches the
                    // importer unresolved and contributes no properties at all.
                    assert!(
                        schema.get("properties").is_some(),
                        "{method} {path} has a request-body schema Core cannot read \
                         (a `oneOf` here means `request_body = Option<T>` — use the \
                         plain type): {schema}"
                    );
                    checked += 1;
                    continue;
                };
                let name = reference
                    .strip_prefix("#/components/schemas/")
                    .unwrap_or_else(|| {
                        panic!("unexpected ref form '{reference}' at {method} {path}")
                    });
                let target = wire
                    .pointer(&format!("/components/schemas/{name}"))
                    .unwrap_or_else(|| {
                        panic!(
                            "{method} {path} refs '{name}' but it is missing from \
                             components.schemas — add it to components(schemas(..))"
                        )
                    });
                assert!(
                    target.get("properties").is_some(),
                    "{method} {path} refs '{name}', which has no properties: {target}"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 1,
            "only POST /workspace carries a body schema; analyze is bodyless, saw {checked}"
        );
    }

    #[test]
    fn the_experiment_argument_is_required_and_described() {
        // Doc comments on the body-struct fields are the whole payoff of the
        // retrofit: they are the only prose the model reads when choosing arguments.
        // `experiment` is also the one field the sidecar rejects the call without, so
        // it must reach Core's `required` list — a model that treats it as optional
        // sends `{}` and gets a 400 it cannot diagnose.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let body = &wire["components"]["schemas"]["InitWorkspaceBody"];
        let description = body["properties"]["experiment"]["description"]
            .as_str()
            .unwrap_or_default();
        assert!(
            description.contains("Name of the experiment"),
            "InitWorkspaceBody::experiment lost its doc comment, got {description:?}"
        );
        let required: Vec<&str> = body["required"]
            .as_array()
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        assert_eq!(required, ["experiment"]);
    }

    #[test]
    fn schema_descriptions_carry_no_internal_rationale() {
        // utoipa lifts a STRUCT's doc comment into the schema's own `description`,
        // exactly as it lifts field docs into property descriptions — so the `///`
        // paragraphs explaining why this type is not the axum extractor would ship to
        // the model as part of the tool. The convention that prevents it: one `///`
        // line naming the body, and every rationale paragraph below it demoted to
        // `//`. Wrapped prose is fine — the tell is VOCABULARY, so this greps for the
        // Rust implementation words that only ever appear in rationale, never in
        // something written for a caller.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        let schemas = wire["components"]["schemas"]
            .as_object()
            .expect("components.schemas must be an object");
        for (name, schema) in schemas {
            let mut descriptions = vec![schema.get("description")];
            if let Some(props) = schema.get("properties").and_then(Value::as_object) {
                descriptions.extend(props.values().map(|p| p.get("description")));
            }
            for description in descriptions.into_iter().flatten().filter_map(Value::as_str) {
                for leak in ["axum", "utoipa", "extractor", "Deserialize", "serde_json"] {
                    assert!(
                        !description.contains(leak),
                        "{name} ships the word '{leak}' to the model in a schema \
                         description — demote that rationale from `///` to `//`: \
                         {description:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn body_less_routes_declare_no_request_body() {
        // The other direction of the same bug. `/status` and the ledger read take no
        // JSON body — their handlers have no `Json` extractor. Declaring one would
        // document something the endpoint never reads, and (before the retrofit) an
        // untyped one at that.
        let wire = serde_json::to_value(super::openapi()).expect("the doc must serialize");
        for path in [
            "/api/research/status",
            "/api/research/workspace/{id}/ledger",
        ] {
            let op = wire
                .pointer(&format!("/paths/{}/get", path.replace('/', "~1")))
                .unwrap_or_else(|| panic!("{path} must have a GET operation"));
            assert!(
                op.get("requestBody").is_none(),
                "{path} takes no body but the document declares one"
            );
        }
        // …and the id the ledger read DOES take must still be an argument.
        let ledger = wire
            .pointer("/paths/~1api~1research~1workspace~1{id}~1ledger/get")
            .expect("the ledger operation");
        assert!(
            ledger.get("parameters").is_some(),
            "the ledger read must still document its workspace id"
        );
    }
}
