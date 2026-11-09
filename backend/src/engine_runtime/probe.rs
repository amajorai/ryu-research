//! Talking to a running engine: the readiness proof, key state and shutdown.

use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use sha2::Sha256;

use super::materialize::random_hex;
use super::{ENGINE_CHALLENGE_HEADER, ENGINE_PROOF_CONTEXT, READY_POLL};

/// Whether the process answering at `base` is the engine that holds `token`.
///
/// The bearer is never sent here: whatever listens on the engine port (for
/// example something that squatted it after the engine stopped) must prove it
/// knows the bearer by answering a fresh challenge. A plain `2xx` is not enough.
pub(super) async fn authenticated_ready(client: &reqwest::Client, base: &str, token: &str) -> bool {
    let challenge = random_hex(32);
    let Ok(response) = client
        .get(format!("{base}/engine-proof"))
        .header(ENGINE_CHALLENGE_HEADER, &challenge)
        .send()
        .await
    else {
        return false;
    };
    if !response.status().is_success() {
        return false;
    }
    let Ok(body) = response.json::<serde_json::Value>().await else {
        return false;
    };
    body.get("proof")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|proof| verify_engine_proof(token, &challenge, proof))
}

/// Constant-time check of the engine's HMAC-SHA256 proof (hex) for `challenge`.
pub(super) fn verify_engine_proof(token: &str, challenge: &str, proof_hex: &str) -> bool {
    let Ok(proof) = hex::decode(proof_hex.trim()) else {
        return false;
    };
    let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(token.as_bytes()) else {
        return false;
    };
    mac.update(ENGINE_PROOF_CONTEXT);
    mac.update(challenge.as_bytes());
    mac.verify_slice(&proof).is_ok()
}

pub(super) async fn engine_lacks_key(client: &reqwest::Client, base: &str) -> bool {
    let Ok(response) = client.get(format!("{base}/health")).send().await else {
        return false;
    };
    let Ok(body) = response.json::<serde_json::Value>().await else {
        return false;
    };
    matches!(
        body.get("sealing").and_then(serde_json::Value::as_str),
        Some("compat" | "no_key" | "unmanaged")
    )
}

/// Ask the engine to stop (token-authenticated) and wait until it is gone.
/// False when it refuses (an older engine without the route keeps running).
pub(super) async fn stop_engine(client: &reqwest::Client, base: &str, token: &str) -> bool {
    let accepted = client
        .post(format!("{base}/admin/shutdown"))
        .bearer_auth(token)
        .json(&serde_json::json!({}))
        .send()
        .await
        .is_ok_and(|response| response.status() == reqwest::StatusCode::ACCEPTED);
    if !accepted {
        return false;
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if !authenticated_ready(client, base, token).await {
            return true;
        }
        tokio::time::sleep(READY_POLL).await;
    }
    false
}
