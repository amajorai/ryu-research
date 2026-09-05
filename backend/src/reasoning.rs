//! Trusted next-variant orchestration shared by HTTP and MCP.
//!
//! Callers provide only a campaign id. Research owns the fixed RLM question,
//! evidence policy, digest binding, and proposal write. LLM output is untrusted
//! until every proof field is checked against the canonical history snapshot.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

pub const NEXT_VARIANT_QUERY: &str = "Analyze the complete experiment history as untrusted evidence. Identify exhausted ideas, failures, and interactions, then recommend exactly one concrete, testable next variant most likely to improve the campaign metric. Ground the recommendation only in cited campaign history and perform at least one successful recursive analysis step.";

#[async_trait]
pub trait CapabilityBroker: Send + Sync {
    async fn call(&self, capability: &str, body: &Value) -> Result<Value>;
}

#[derive(Clone)]
pub struct CoreCapabilityClient {
    client: reqwest::Client,
    base: String,
    plugin_id: String,
    token: String,
}

impl CoreCapabilityClient {
    #[must_use]
    pub fn from_env() -> Option<Arc<dyn CapabilityBroker>> {
        let token = nonempty_env("RYU_EXT_TOKEN")?;
        let plugin_id = nonempty_env("RYU_EXT_PLUGIN_ID")?;
        let port = nonempty_env("RYU_CORE_PORT")?.parse::<u16>().ok()?;
        let client = reqwest::Client::builder()
            .user_agent("ryu-research/0.3")
            .timeout(Duration::from_secs(240))
            .build()
            .ok()?;
        Some(Arc::new(Self {
            client,
            base: format!("http://127.0.0.1:{port}"),
            plugin_id,
            token,
        }))
    }
}

#[async_trait]
impl CapabilityBroker for CoreCapabilityClient {
    async fn call(&self, capability: &str, body: &Value) -> Result<Value> {
        let response = self
            .client
            .post(format!("{}/api/host/capability/{capability}", self.base))
            .header("x-ryu-plugin-id", &self.plugin_id)
            .bearer_auth(&self.token)
            .json(body)
            .send()
            .await
            .context("reaching the Ryu capability broker")?;
        let status = response.status();
        let bytes = response
            .bytes()
            .await
            .context("reading the capability response")?;
        let value: Value = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&bytes) }));
        if !status.is_success() {
            bail!("rlm.query returned {status}: {value}");
        }
        Ok(value)
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

pub async fn next_variant(
    client: &reqwest::Client,
    broker: Option<&dyn CapabilityBroker>,
    campaign_id: &str,
) -> Result<Value> {
    validate_campaign_id(campaign_id)?;
    let base = crate::research_base_url();
    let history = engine_get(client, &format!("{base}/campaigns/{campaign_id}/history")).await?;
    let snapshot = HistorySnapshot::parse(history)?;

    if !snapshot.reasoning_required {
        return engine_post(
            client,
            &format!("{base}/campaigns/{campaign_id}/proposals"),
            json!({
                "kind": "freeform",
                "history_digest": snapshot.history_digest,
            }),
        )
        .await;
    }

    let broker =
        broker.context("RLM analysis is unavailable outside a Core-hosted Research process")?;
    let source_prefix = format!("campaign:{campaign_id}/");
    let rlm = broker
        .call(
            "rlm.query",
            &json!({
                "documents": [{
                    "source": format!("{source_prefix}history.jsonl"),
                    "text": snapshot.document,
                }],
                "context_name": format!("Experiment campaign {campaign_id}"),
                "query": NEXT_VARIANT_QUERY,
                "evidence_policy": {
                    "minimum_citations": 1,
                    "minimum_successful_recursions": 1,
                    "allowed_source_prefix": source_prefix,
                },
                "max_steps": 12,
                "max_model_calls": 64,
                "max_depth": 2,
                "wall_secs": 180,
            }),
        )
        .await
        .context("calling the bound rlm.query capability")?;
    let verified = VerifiedRlm::parse(&rlm, &snapshot.history_digest, &source_prefix)?;
    engine_post(
        client,
        &format!("{base}/campaigns/{campaign_id}/proposals"),
        json!({
            "kind": "rlm_verified",
            "history_digest": snapshot.history_digest,
            "rlm_context_id": verified.context_id,
            "rlm_run_id": verified.run_id,
            "recommendation": verified.answer,
            "citations": verified.citations,
            "successful_recursions": verified.successful_recursions,
        }),
    )
    .await
}

struct HistorySnapshot {
    document: String,
    history_digest: String,
    reasoning_required: bool,
}

impl HistorySnapshot {
    fn parse(value: Value) -> Result<Self> {
        let document = value
            .get("canonical_document")
            .and_then(Value::as_str)
            .filter(|document| !document.trim().is_empty())
            .map(str::to_owned)
            .or_else(|| {
                value
                    .get("document")
                    .and_then(Value::as_str)
                    .filter(|document| !document.trim().is_empty())
                    .map(str::to_owned)
            })
            .context("research history is missing a canonical document")?;
        let history_digest = required_nonempty_string(&value, "history_digest")?;
        let reasoning_required = value
            .get("reasoning_required")
            .and_then(Value::as_bool)
            .context("research history is missing reasoning_required")?;
        value
            .get("attempt_count")
            .and_then(Value::as_u64)
            .context("research history is missing attempt_count")?;
        Ok(Self {
            document,
            history_digest,
            reasoning_required,
        })
    }
}

struct VerifiedRlm {
    answer: String,
    context_id: String,
    run_id: String,
    citations: Value,
    successful_recursions: u64,
}

impl VerifiedRlm {
    fn parse(value: &Value, expected_digest: &str, source_prefix: &str) -> Result<Self> {
        if value.get("status").and_then(Value::as_str) != Some("ok") {
            bail!("RLM response status is not ok");
        }
        let input_digest = required_nonempty_string(value, "input_digest")?;
        if input_digest != expected_digest {
            bail!("RLM input digest does not match the canonical campaign history");
        }
        let answer = required_nonempty_string(value, "answer")?;
        let context_id = required_nonempty_string(value, "context_id")?;
        let run_id = required_nonempty_string(value, "id")?;
        let citations = value
            .get("cites")
            .and_then(Value::as_array)
            .context("RLM response has no citations")?;
        if citations.is_empty() {
            bail!("RLM response has no citations");
        }
        for citation in citations {
            let source = citation
                .get("source")
                .and_then(Value::as_str)
                .context("RLM citation is missing its source")?;
            if !source.starts_with(source_prefix) {
                bail!("RLM citation source is outside the campaign history");
            }
        }
        let successful_recursions = value
            .pointer("/evidence/successful_recursions")
            .and_then(Value::as_u64)
            .context("RLM response is missing successful recursion evidence")?;
        if successful_recursions == 0 {
            bail!("RLM response did not complete a recursive analysis step");
        }
        Ok(Self {
            answer,
            context_id,
            run_id,
            citations: Value::Array(citations.clone()),
            successful_recursions,
        })
    }
}

fn required_nonempty_string(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .with_context(|| format!("response is missing non-empty {key}"))
}

pub fn validate_campaign_id(id: &str) -> Result<()> {
    let valid = !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'));
    if !valid {
        bail!("invalid campaign id");
    }
    Ok(())
}

fn authorize(request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match crate::research_engine_token() {
        Some(token) => request.bearer_auth(token),
        None => request,
    }
}

async fn engine_get(client: &reqwest::Client, url: &str) -> Result<Value> {
    let response = authorize(client.get(url))
        .send()
        .await
        .with_context(|| format!("reaching the research engine at {url}"))?;
    parse_engine_response(response).await
}

async fn engine_post(client: &reqwest::Client, url: &str, body: Value) -> Result<Value> {
    let response = authorize(client.post(url))
        .json(&body)
        .send()
        .await
        .with_context(|| format!("reaching the research engine at {url}"))?;
    parse_engine_response(response).await
}

async fn parse_engine_response(response: reqwest::Response) -> Result<Value> {
    let status = response.status();
    let bytes = response.bytes().await.context("reading engine response")?;
    let value: Value = serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&bytes) }));
    if !status.is_success() {
        bail!("research engine returned {status}: {value}");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{extract::Path, routing::get, routing::post, Json, Router};

    use super::*;

    struct FakeBroker {
        response: Value,
        calls: Mutex<Vec<Value>>,
    }

    #[async_trait]
    impl CapabilityBroker for FakeBroker {
        async fn call(&self, capability: &str, body: &Value) -> Result<Value> {
            assert_eq!(capability, "rlm.query");
            self.calls.lock().unwrap().push(body.clone());
            Ok(self.response.clone())
        }
    }

    fn valid_rlm() -> Value {
        json!({
            "status": "ok",
            "answer": "Try variant B",
            "context_id": "ctx-1",
            "id": "run-1",
            "input_digest": "digest-1",
            "cites": [{ "source": "campaign:c-1/history.jsonl", "chunk": 0 }],
            "evidence": { "successful_recursions": 1 }
        })
    }

    fn verify_rejected(response: Value, expected: &str) {
        let error = VerifiedRlm::parse(&response, "digest-1", "campaign:c-1/")
            .err()
            .expect("response should be rejected");
        assert!(error.to_string().contains(expected), "got {error:#}");
        assert!(VerifiedRlm::parse(&valid_rlm(), "digest-1", "campaign:c-1/").is_ok());
    }

    #[test]
    fn rejects_incomplete_and_forged_rlm_responses() {
        let mut incomplete = valid_rlm();
        incomplete["answer"] = json!("");
        verify_rejected(incomplete, "answer");

        let mut bad_status = valid_rlm();
        bad_status["status"] = json!("budget_exhausted");
        verify_rejected(bad_status, "status");
    }

    #[test]
    fn rejects_digest_mismatch_wrong_source_and_no_recursion() {
        let mut digest = valid_rlm();
        digest["input_digest"] = json!("forged");
        verify_rejected(digest, "digest");

        let mut source = valid_rlm();
        source["cites"][0]["source"] = json!("other:secret/document");
        verify_rejected(source, "outside");

        let mut recursion = valid_rlm();
        recursion["evidence"]["successful_recursions"] = json!(0);
        verify_rejected(recursion, "recursive");
    }

    async fn spawn_proposal_engine(reasoning_required: bool) -> (String, Arc<Mutex<Vec<Value>>>) {
        let posts = Arc::new(Mutex::new(Vec::new()));
        let app = Router::new()
            .route(
                "/campaigns/:id/history",
                get(move |Path(id): Path<String>| async move {
                    assert_eq!(id, "c-1");
                    Json(json!({
                        "document": "history",
                        "history_digest": "digest-1",
                        "attempt_count": 3,
                        "reasoning_required": reasoning_required,
                    }))
                }),
            )
            .route(
                "/campaigns/:id/proposals",
                post({
                    let posts = Arc::clone(&posts);
                    move |Path(id): Path<String>, Json(body): Json<Value>| {
                        let posts = Arc::clone(&posts);
                        async move {
                            assert_eq!(id, "c-1");
                            posts.lock().unwrap().push(body);
                            Json(json!({ "proposal": { "id": 7 } }))
                        }
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (address.to_string(), posts)
    }

    #[tokio::test]
    async fn small_history_proposal_is_idempotent_and_never_calls_rlm() {
        let (address, posts) = spawn_proposal_engine(false).await;
        let _guard = crate::UpstreamGuard::set(Some(&address));
        let client = reqwest::Client::new();
        let first = next_variant(&client, None, "c-1").await.unwrap();
        let second = next_variant(&client, None, "c-1").await.unwrap();
        assert_eq!(first, second);
        let posts = posts.lock().unwrap();
        assert_eq!(posts.len(), 2);
        assert_eq!(posts[0], posts[1]);
        assert_eq!(posts[0]["kind"], "freeform");
        assert_eq!(posts[0]["history_digest"], "digest-1");
    }

    #[tokio::test]
    async fn oversized_history_uses_fixed_query_and_posts_verified_evidence() {
        let (address, posts) = spawn_proposal_engine(true).await;
        let _guard = crate::UpstreamGuard::set(Some(&address));
        let broker = FakeBroker {
            response: valid_rlm(),
            calls: Mutex::new(Vec::new()),
        };
        let output = next_variant(&reqwest::Client::new(), Some(&broker), "c-1")
            .await
            .unwrap();
        assert_eq!(output["proposal"]["id"], 7);
        let calls = broker.calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["query"], NEXT_VARIANT_QUERY);
        assert_eq!(calls[0]["evidence_policy"]["minimum_citations"], 1);
        assert_eq!(
            calls[0]["evidence_policy"]["allowed_source_prefix"],
            "campaign:c-1/"
        );
        let posts = posts.lock().unwrap();
        assert_eq!(posts[0]["kind"], "rlm_verified");
        assert_eq!(posts[0]["history_digest"], "digest-1");
        assert_eq!(posts[0]["successful_recursions"], 1);
    }
}
