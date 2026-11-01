//! Self-contained lifecycle for the embedded Python autoresearch engine.
//!
//! Production builds carry an allowlisted engine payload in the Rust binary. On
//! first use it is published atomically into an immutable, digest-addressed
//! directory below the app's plugin data dir (`engine/<version>-<digest>`);
//! concurrent processes either publish the same payload
//! or adopt the winner. RESEARCH_DIR remains an explicit development override.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Stdio;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use hmac::{Hmac, Mac};
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::ResearchHost;

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

fn payload_digest() -> String {
    let mut hasher = Sha256::new();
    for file in EMBEDDED_FILES {
        hasher.update((file.path.len() as u64).to_le_bytes());
        hasher.update(file.path.as_bytes());
        hasher.update((file.bytes.len() as u64).to_le_bytes());
        hasher.update(file.bytes);
    }
    hex::encode(hasher.finalize())
}

fn safe_embedded_path(path: &str) -> Result<PathBuf> {
    let candidate = Path::new(path);
    if candidate.is_absolute()
        || candidate
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("unsafe embedded research path: {path}");
    }
    Ok(candidate.to_path_buf())
}

/// Publish the embedded payload as one immutable directory.
pub fn materialize_embedded(data_dir: &Path) -> Result<PathBuf> {
    let digest = payload_digest();
    let store = data_dir.join(crate::ENGINE_DIR_NAME);
    create_private_dir(&store)?;
    let destination = store.join(format!("{}-{}", env!("CARGO_PKG_VERSION"), digest));
    if valid_materialization(&destination, &digest) {
        return Ok(destination);
    }
    if destination.exists() {
        bail!(
            "embedded research directory exists without its valid digest marker: {}",
            destination.display()
        );
    }

    let staging = store.join(format!(".install-{}-{}", std::process::id(), random_hex(8)));
    create_private_dir(&staging)?;
    let write_result = (|| -> Result<()> {
        let mut seen = HashSet::new();
        for file in EMBEDDED_FILES {
            let relative = safe_embedded_path(file.path)?;
            if !seen.insert(relative.clone()) {
                bail!("duplicate embedded research path: {}", file.path);
            }
            atomic_write(&staging.join(relative), file.bytes)?;
        }
        atomic_write(&staging.join(DIGEST_MARKER), digest.as_bytes())?;
        Ok(())
    })();
    if let Err(error) = write_result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }

    match fs::rename(&staging, &destination) {
        Ok(()) => Ok(destination),
        Err(_) if destination.exists() && valid_materialization(&destination, &digest) => {
            let _ = fs::remove_dir_all(&staging);
            Ok(destination)
        }
        Err(error) => {
            let _ = fs::remove_dir_all(&staging);
            Err(error)
                .with_context(|| format!("publishing embedded engine to {}", destination.display()))
        }
    }
}

fn valid_materialization(path: &Path, digest: &str) -> bool {
    fs::read_to_string(path.join(DIGEST_MARKER))
        .ok()
        .is_some_and(|value| value == digest)
        && EMBEDDED_FILES
            .iter()
            .all(|file| path.join(file.path).is_file())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("embedded file has no parent")?;
    create_private_dir(parent)?;
    let temp = parent.join(format!(".write-{}-{}", std::process::id(), random_hex(6)));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(&temp)
        .with_context(|| format!("creating temporary embedded file {}", temp.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temp, path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(())
}

fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

fn ensure_engine_token() -> Result<String> {
    if let Some(token) = nonempty_env("RYU_RESEARCH_ENGINE_TOKEN") {
        return Ok(token);
    }
    let path = workspaces_dir().join(crate::ENGINE_TOKEN_FILE);
    if let Ok(token) = fs::read_to_string(&path) {
        let token = token.trim();
        if !token.is_empty() {
            return Ok(token.to_owned());
        }
    }
    let parent = path.parent().context("engine token has no parent")?;
    create_private_dir(parent)?;
    let token = random_hex(32);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            file.write_all(token.as_bytes())?;
            file.sync_all()?;
            Ok(token)
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let winner = fs::read_to_string(&path).with_context(|| {
                format!("reading concurrently-created token {}", path.display())
            })?;
            let winner = winner.trim();
            if winner.is_empty() {
                bail!("concurrently-created research engine token is empty");
            }
            Ok(winner.to_owned())
        }
        Err(error) => Err(error).with_context(|| format!("creating {}", path.display())),
    }
}

fn random_hex(byte_count: usize) -> String {
    let mut bytes = vec![0_u8; byte_count];
    OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}

/// Whether the process answering at `base` is the engine that holds `token`.
///
/// The bearer is never sent here: whatever listens on the engine port (for
/// example something that squatted it after the engine stopped) must prove it
/// knows the bearer by answering a fresh challenge. A plain `2xx` is not enough.
async fn authenticated_ready(client: &reqwest::Client, base: &str, token: &str) -> bool {
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
fn verify_engine_proof(token: &str, challenge: &str, proof_hex: &str) -> bool {
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

async fn engine_lacks_key(client: &reqwest::Client, base: &str) -> bool {
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
async fn stop_engine(client: &reqwest::Client, base: &str, token: &str) -> bool {
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

async fn spawn_python(
    code_dir: &Path,
    workspaces: &Path,
    target: &EngineTarget,
    token: &str,
) -> Result<Child> {
    let candidates = python_candidates(code_dir);
    let mut last_error = None;
    for program in candidates {
        let mut command = Command::new(&program);
        command
            .arg("-m")
            .arg("ryu_research")
            .current_dir(code_dir)
            .env_clear()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        copy_benign_environment(&mut command);
        command
            .env("PYTHONPATH", code_dir)
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("RESEARCH_HOST", &target.host)
            .env("RESEARCH_PORT", target.port.to_string())
            .env("RESEARCH_EXPERIMENTS", code_dir.join("experiments"))
            .env("RESEARCH_WORKSPACES", workspaces)
            .env("RYU_RESEARCH_ENGINE_TOKEN", token);
        // The sealing key and key-state hints go to the engine only; it
        // removes them from its own environment and never passes them on.
        for (name, value) in engine_seal_env() {
            command.env(name, value);
        }
        match command.spawn() {
            Ok(child) => return Ok(child),
            Err(error) if error.kind() == ErrorKind::NotFound => last_error = Some(error),
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("spawning {} -m ryu_research", program.to_string_lossy())
                })
            }
        }
    }
    Err(last_error.unwrap_or_else(|| std::io::Error::new(ErrorKind::NotFound, "Python not found")))
        .context("spawning the embedded research engine; set RESEARCH_PYTHON to Python 3")
}

fn python_candidates(code_dir: &Path) -> Vec<OsString> {
    if let Some(program) = std::env::var_os("RESEARCH_PYTHON").filter(|program| !program.is_empty())
    {
        return vec![program];
    }
    let venv = if cfg!(windows) {
        code_dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        code_dir.join(".venv").join("bin").join("python")
    };
    let mut candidates = Vec::new();
    if venv.is_file() {
        candidates.push(venv.into_os_string());
    }
    if cfg!(windows) {
        candidates.push(OsString::from("python"));
    } else {
        candidates.push(OsString::from("python3"));
        candidates.push(OsString::from("python"));
    }
    candidates
}

fn copy_benign_environment(command: &mut Command) {
    const ALLOWED: &[&str] = &[
        "APPDATA",
        "COMSPEC",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "DYLD_LIBRARY_PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LOCALAPPDATA",
        "NUMBER_OF_PROCESSORS",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROGRAMDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "SYSTEMROOT",
        "TEMP",
        "TERM",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    ];
    for (name, value) in std::env::vars_os() {
        let normalized = name.to_string_lossy().to_ascii_uppercase();
        if ALLOWED.contains(&normalized.as_str()) {
            command.env(name, value);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    struct EnvGuard {
        names: Vec<(&'static str, Option<OsString>)>,
    }

    impl EnvGuard {
        fn isolate(data_dir: &Path, port: u16) -> Self {
            let names = [
                "RYU_PLUGIN_DATA_DIR",
                "RESEARCH_DIR",
                "RESEARCH_WORKSPACES",
                "RYU_RESEARCH_UPSTREAM",
                "RYU_RESEARCH_ENGINE_TOKEN",
                "RESEARCH_PORT",
            ]
            .into_iter()
            .map(|name| (name, std::env::var_os(name)))
            .collect();
            std::env::set_var("RYU_PLUGIN_DATA_DIR", data_dir);
            std::env::remove_var("RESEARCH_DIR");
            std::env::set_var(
                "RESEARCH_WORKSPACES",
                data_dir.join(crate::WORKSPACES_DIR_NAME),
            );
            std::env::remove_var("RYU_RESEARCH_UPSTREAM");
            std::env::remove_var("RYU_RESEARCH_ENGINE_TOKEN");
            std::env::set_var("RESEARCH_PORT", port.to_string());
            Self { names }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (name, value) in &self.names {
                match value {
                    Some(value) => std::env::set_var(name, value),
                    None => std::env::remove_var(name),
                }
            }
        }
    }

    fn unused_port() -> u16 {
        std::net::TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral listener")
            .local_addr()
            .expect("listener address")
            .port()
    }

    fn python_available() -> bool {
        python_candidates(Path::new("missing-venv"))
            .into_iter()
            .any(|program| {
                std::process::Command::new(program)
                    .arg("--version")
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
            })
    }

    #[test]
    fn engine_receives_the_captured_key_and_its_key_state_hints() {
        let captured = [
            ("RYU_SEAL_KEY", "a2V5".to_owned()),
            ("RYU_SEAL_SCOPE", "plugin:@ryu/research".to_owned()),
        ];
        let dir = Path::new("/data/plugin-data/@ryu+research");
        let env = seal_env_for_engine(&captured, true, dir);
        let names: Vec<&str> = env.iter().map(|(name, _)| *name).collect();
        assert_eq!(
            names,
            [
                "RYU_SEAL_KEY",
                "RYU_SEAL_SCOPE",
                "RYU_RESEARCH_DATA_DIR",
                "RYU_PLUGIN_DATA_DIR"
            ]
        );
        // A standalone run never claims to be Core-hosted (so the engine may
        // use its own standalone key instead of compat mode).
        let standalone = seal_env_for_engine(&[], false, dir);
        assert_eq!(standalone.len(), 1);
        assert_eq!(standalone[0].0, "RYU_RESEARCH_DATA_DIR");
        assert!(EMBEDDED_FILES
            .iter()
            .any(|file| file.path == "ryu_research/ryu_content_seal.py"));
    }

    #[test]
    fn embedded_inventory_is_unique_and_traversal_free() {
        let mut paths = HashSet::new();
        assert!(!EMBEDDED_FILES.is_empty());
        for file in EMBEDDED_FILES {
            assert!(!file.bytes.is_empty(), "{} is empty", file.path);
            assert!(safe_embedded_path(file.path).is_ok(), "{}", file.path);
            assert!(paths.insert(file.path), "duplicate {}", file.path);
        }
    }

    #[test]
    fn empty_data_dir_materializes_the_complete_payload_idempotently() {
        let temp = tempfile::tempdir().unwrap();
        let first = materialize_embedded(temp.path()).unwrap();
        let second = materialize_embedded(temp.path()).unwrap();
        assert_eq!(first, second);
        assert!(valid_materialization(&first, &payload_digest()));
        for file in EMBEDDED_FILES {
            assert_eq!(fs::read(first.join(file.path)).unwrap(), file.bytes);
        }
    }

    #[tokio::test]
    async fn empty_data_dir_starts_and_second_supervisor_adopts_when_python_exists() {
        if !python_available() {
            return;
        }
        let _lock = crate::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let temp = tempfile::tempdir().unwrap();
        let _env = EnvGuard::isolate(temp.path(), unused_port());
        let engine_dir = materialize_embedded(temp.path()).unwrap();
        std::env::set_var("RESEARCH_DIR", &engine_dir);
        let first = Arc::new(EngineSupervisor::new());
        first.ensure_ready().await.unwrap();
        assert!(first.owns_process().await);
        assert!(materialize_embedded(temp.path()).unwrap().is_dir());

        let second = EngineSupervisor::new();
        second.ensure_ready().await.unwrap();
        assert!(!second.owns_process().await);
        assert!(crate::research_engine_token().is_some());
    }

    #[test]
    fn engine_proof_matches_the_python_engine() {
        // Known-answer vector shared with the Python engine's
        // `test_engine_proof_answers_a_challenge_without_receiving_the_bearer`.
        let token = "engine-proof-test-token";
        let challenge = "0123456789abcdef0123456789abcdef";
        let proof = "35b921cca206a80f65bc2b3c65f75a4136692b293e9f7d2846362e8a05ebeb00";
        assert!(verify_engine_proof(token, challenge, proof));
        assert!(!verify_engine_proof("another-token", challenge, proof));
        assert!(!verify_engine_proof(
            token,
            "fedcba9876543210fedcba9876543210",
            proof
        ));
        assert!(!verify_engine_proof(token, challenge, &proof[..62]));
        assert!(!verify_engine_proof(token, challenge, "not hex"));
        assert!(!verify_engine_proof(token, challenge, ""));
    }

    /// Something squatting the engine port that answers every request with
    /// `200` (optionally with a forged proof) and records what it receives.
    async fn spawn_port_squatter(body: &'static str) -> (String, Arc<std::sync::Mutex<Vec<u8>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        tokio::spawn(async move {
            while let Ok((mut stream, _)) = listener.accept().await {
                let mut buffer = vec![0_u8; 8192];
                let read = stream.read(&mut buffer).await.unwrap_or(0);
                sink.lock().unwrap().extend_from_slice(&buffer[..read]);
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(response.as_bytes()).await;
                let _ = stream.shutdown().await;
            }
        });
        (base, captured)
    }

    #[tokio::test]
    async fn a_port_squatter_is_never_ready_and_never_receives_the_bearer() {
        let token = "engine-bearer-sentinel-5e1f";
        let client = EngineSupervisor::new().client;
        for body in [
            r#"{"experiments":[]}"#,
            r#"{"proof":"35b921cca206a80f65bc2b3c65f75a4136692b293e9f7d2846362e8a05ebeb00"}"#,
        ] {
            let (base, captured) = spawn_port_squatter(body).await;
            assert!(
                !authenticated_ready(&client, &base, token).await,
                "a squatter answering {body} was treated as the engine"
            );
            let seen = String::from_utf8_lossy(&captured.lock().unwrap()).into_owned();
            assert!(seen.contains("/engine-proof"), "{seen}");
            assert!(
                !seen.contains(token),
                "the bearer reached the squatter: {seen}"
            );
            assert!(
                !seen.to_ascii_lowercase().contains("authorization"),
                "{seen}"
            );
        }
    }

    #[test]
    fn explicit_upstream_is_adopt_only_and_never_a_spawn_target() {
        let _lock = crate::ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let old = std::env::var_os("RYU_RESEARCH_UPSTREAM");
        std::env::set_var("RYU_RESEARCH_UPSTREAM", "https://engine.example.test/base");
        let target = EngineTarget::from_env().unwrap();
        assert!(target.adopt_only);
        assert_eq!(target.host, "engine.example.test");
        match old {
            Some(value) => std::env::set_var("RYU_RESEARCH_UPSTREAM", value),
            None => std::env::remove_var("RYU_RESEARCH_UPSTREAM"),
        }
    }
}
