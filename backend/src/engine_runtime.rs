//! Self-contained lifecycle for the embedded Python autoresearch engine.
//!
//! Production builds carry an allowlisted engine payload in the Rust binary. On
//! first use it is published atomically into an immutable, digest-addressed
//! directory below the app's plugin data dir (`engine/<version>-<digest>`);
//! concurrent processes either publish the same payload
//! or adopt the winner. RESEARCH_DIR remains an explicit development override.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use tokio::process::Child;
use tokio::sync::Mutex;

use crate::ResearchHost;

mod materialize;
mod probe;
mod spawn;
#[cfg(test)]
mod tests;

pub use materialize::materialize_embedded;
use materialize::{create_private_dir, ensure_engine_token};
use probe::{authenticated_ready, engine_lacks_key, stop_engine};
use spawn::spawn_python;

// Cold-starting the embedded Python engine is materially slower on macOS CI
// than on a warm developer machine. Keep the wait bounded, but do not turn a
// valid release build into a platform-only timeout failure.
const READY_TIMEOUT: Duration = Duration::from_secs(60);
const READY_POLL: Duration = Duration::from_millis(125);
const DIGEST_MARKER: &str = ".embedded-digest";
/// Readiness handshake with the Python engine (`ryu_research/server.py`): the
/// supervisor sends a fresh nonce and the engine answers with an HMAC keyed by
/// the engine bearer, so the bearer itself is never sent to an unverified port.
const ENGINE_CHALLENGE_HEADER: &str = "x-ryu-engine-challenge";
const ENGINE_PROOF_CONTEXT: &[u8] = b"ryu-research-engine-proof-v1\n";

/// Key-carrying variables (Core delivers the first two with `crypto:seal`).
const SEAL_ENV_VARS: [&str; 4] = [
    "RYU_SEAL_KEY",
    "RYU_SEAL_SCOPE",
    "RYU_SEAL_RETIRED_KEYS",
    "RYU_MASTER_KEY",
];
static SEAL_ENV: OnceLock<Vec<(&'static str, String)>> = OnceLock::new();

/// Capture the sealing key this process was started with and remove it from
/// the process environment. Call once at the top of `main`, before any thread
/// or child exists; the Python engine receives the captured values explicitly
/// (its spawn clears the environment) and nothing else inherits them.
pub fn capture_seal_env() {
    let _ = captured_seal_env();
}

fn captured_seal_env() -> &'static [(&'static str, String)] {
    SEAL_ENV.get_or_init(|| {
        let mut captured = Vec::new();
        for name in SEAL_ENV_VARS {
            if let Some(value) = nonempty_env(name) {
                captured.push((name, value));
            }
            std::env::remove_var(name);
        }
        captured
    })
}

fn holds_delivered_key() -> bool {
    captured_seal_env()
        .iter()
        .any(|(name, _)| *name == "RYU_SEAL_KEY")
}

/// Whether Core started this process (sidecar or MCP server). The engine then
/// treats a missing key as "waiting for a Ryu update" (compat), never as a
/// standalone run that would generate its own key.
fn core_hosted() -> bool {
    ["RYU_PLUGIN_DATA_DIR", "RYU_EXT_PLUGIN_ID", "RYU_EXT_TOKEN"]
        .into_iter()
        .any(|name| nonempty_env(name).is_some())
}

/// The environment entries the engine needs for its sealing key state.
fn engine_seal_env() -> Vec<(&'static str, OsString)> {
    seal_env_for_engine(
        captured_seal_env(),
        core_hosted(),
        &crate::research_data_dir(),
    )
}

fn seal_env_for_engine(
    captured: &[(&'static str, String)],
    core_hosted: bool,
    data_dir: &Path,
) -> Vec<(&'static str, OsString)> {
    let mut env: Vec<(&'static str, OsString)> = captured
        .iter()
        .map(|(name, value)| (*name, OsString::from(value)))
        .collect();
    env.push(("RYU_RESEARCH_DATA_DIR", data_dir.as_os_str().to_owned()));
    if core_hosted {
        env.push(("RYU_PLUGIN_DATA_DIR", data_dir.as_os_str().to_owned()));
    }
    env
}

/// One compile-time-owned file in the Python engine distribution.
#[derive(Clone, Copy)]
pub struct EmbeddedFile {
    pub path: &'static str,
    pub bytes: &'static [u8],
}

macro_rules! embedded {
    ($path:literal) => {
        EmbeddedFile {
            path: $path,
            bytes: include_bytes!(concat!("../../sidecar/", $path)),
        }
    };
}

/// Explicit allowlist: adding Python code does not ship it until reviewed here.
pub const EMBEDDED_FILES: &[EmbeddedFile] = &[
    embedded!("pyproject.toml"),
    embedded!("ryu_research/__init__.py"),
    embedded!("ryu_research/__main__.py"),
    embedded!("ryu_research/server.py"),
    embedded!("ryu_research/experiments.py"),
    embedded!("ryu_research/campaigns.py"),
    embedded!("ryu_research/workspaces.py"),
    embedded!("ryu_research/ryu_content_seal.py"),
    embedded!("ryu_research/sealed_errors.py"),
    embedded!("ryu_research/sandbox.py"),
    embedded!("experiments/toy/experiment.toml"),
    embedded!("experiments/toy/program.md"),
    embedded!("experiments/toy/train.py"),
    embedded!("experiments/nanochat/experiment.toml"),
    embedded!("experiments/nanochat/program.md"),
    embedded!("experiments/nanochat/prepare.py"),
    embedded!("experiments/nanochat/train.py"),
];

fn start_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[derive(Clone, Debug)]
struct EngineTarget {
    base_url: String,
    host: String,
    port: u16,
    adopt_only: bool,
}

impl EngineTarget {
    fn from_env() -> Result<Self> {
        if let Some(raw) = nonempty_env("RYU_RESEARCH_UPSTREAM") {
            let normalized = if raw.starts_with("http://") || raw.starts_with("https://") {
                raw
            } else {
                format!("http://{raw}")
            };
            let url = reqwest::Url::parse(normalized.trim_end_matches('/'))
                .context("RYU_RESEARCH_UPSTREAM must be an http(s) URL or host:port")?;
            if !matches!(url.scheme(), "http" | "https")
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
            {
                bail!("RYU_RESEARCH_UPSTREAM must be an http(s) endpoint without credentials");
            }
            let host = url.host_str().expect("validated host").to_owned();
            let port = url
                .port_or_known_default()
                .context("RYU_RESEARCH_UPSTREAM has no port")?;
            return Ok(Self {
                base_url: url.as_str().trim_end_matches('/').to_owned(),
                host,
                port,
                adopt_only: true,
            });
        }

        let port = match nonempty_env("RESEARCH_PORT") {
            Some(value) => value
                .parse::<u16>()
                .context("RESEARCH_PORT must be a valid TCP port")?,
            None => crate::engine_port(),
        };
        Ok(Self {
            base_url: format!("http://127.0.0.1:{port}"),
            host: "127.0.0.1".to_owned(),
            port,
            adopt_only: false,
        })
    }
}

/// Owns or adopts the private Python engine used by HTTP and MCP modes.
pub struct EngineSupervisor {
    client: reqwest::Client,
    child: Mutex<Option<Child>>,
}

impl EngineSupervisor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("ryu-research-engine-supervisor/0.3")
                .timeout(Duration::from_secs(3))
                .build()
                .expect("valid engine HTTP client"),
            child: Mutex::new(None),
        }
    }

    /// Ensure an authenticated engine is ready.
    pub async fn ensure_ready(&self) -> Result<()> {
        let _start_guard = start_lock().lock().await;
        let target = EngineTarget::from_env()?;
        let token = ensure_engine_token()?;
        if authenticated_ready(&self.client, &target.base_url, &token).await {
            // An engine started without the key (for example by the MCP
            // server, which Core gives no key) is replaced once a process
            // holding the delivered key needs it, so its content gets sealed.
            if target.adopt_only
                || !holds_delivered_key()
                || !engine_lacks_key(&self.client, &target.base_url).await
                || !stop_engine(&self.client, &target.base_url, &token).await
            {
                return Ok(());
            }
            tracing::info!("restarting the research engine with its sealing key");
        }
        if target.adopt_only {
            bail!(
                "the explicit research engine at {} is not authenticated and ready",
                target.base_url
            );
        }
        if target.host != "127.0.0.1" && target.host != "localhost" && target.host != "::1" {
            bail!("Research only spawns its engine on loopback");
        }

        let code_dir = engine_code_dir()?;
        let workspaces = workspaces_dir();
        create_private_dir(&workspaces)?;
        let child = spawn_python(&code_dir, &workspaces, &target, &token).await?;
        *self.child.lock().await = Some(child);

        let deadline = Instant::now() + READY_TIMEOUT;
        loop {
            if authenticated_ready(&self.client, &target.base_url, &token).await {
                return Ok(());
            }
            if Instant::now() >= deadline {
                bail!(
                    "embedded research engine did not become authenticated and ready at {} within {}s",
                    target.base_url,
                    READY_TIMEOUT.as_secs()
                );
            }

            // A peer may have won the bind race. Keep probing if our child exits.
            let mut child_guard = self.child.lock().await;
            if let Some(child) = child_guard.as_mut() {
                if child
                    .try_wait()
                    .context("checking the research engine process")?
                    .is_some()
                {
                    *child_guard = None;
                }
            }
            drop(child_guard);
            tokio::time::sleep(READY_POLL).await;
        }
    }

    #[cfg(test)]
    async fn owns_process(&self) -> bool {
        self.child.lock().await.is_some()
    }
}

impl Default for EngineSupervisor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ResearchHost for EngineSupervisor {
    async fn start_sidecar(&self) -> Result<()> {
        self.ensure_ready().await
    }

    fn is_installed(&self) -> bool {
        // Materialization is a cache; the payload is installed in this executable.
        !EMBEDDED_FILES.is_empty()
    }
}

fn nonempty_env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn workspaces_dir() -> PathBuf {
    std::env::var_os("RESEARCH_WORKSPACES")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| crate::research_data_dir().join(crate::WORKSPACES_DIR_NAME))
}

fn engine_code_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("RESEARCH_DIR")
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        if !path.join("ryu_research").is_dir() || !path.join("experiments").is_dir() {
            bail!(
                "RESEARCH_DIR must contain ryu_research/ and experiments/: {}",
                path.display()
            );
        }
        return Ok(path);
    }
    materialize_embedded(&crate::research_data_dir())
}
