//! Self-contained lifecycle for the embedded Python autoresearch engine.
//!
//! Production builds carry an allowlisted engine payload in the Rust binary. On
//! first use it is published atomically into an immutable, digest-addressed
//! directory below RYU_DIR; concurrent processes either publish the same payload
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
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::{ResearchHost, RESEARCH_PORT};

const READY_TIMEOUT: Duration = Duration::from_secs(20);
const READY_POLL: Duration = Duration::from_millis(125);
const DIGEST_MARKER: &str = ".embedded-digest";

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
            None => RESEARCH_PORT,
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
            return Ok(());
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
        .unwrap_or_else(|| crate::default_ryu_dir().join("research-workspaces"))
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
    materialize_embedded(&crate::default_ryu_dir())
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
    let store = data_dir.join("research").join("engine");
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

async fn authenticated_ready(client: &reqwest::Client, base: &str, token: &str) -> bool {
    client
        .get(format!("{base}/experiments"))
        .bearer_auth(token)
        .send()
        .await
        .is_ok_and(|response| response.status().is_success())
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
                "RYU_DIR",
                "RESEARCH_DIR",
                "RESEARCH_WORKSPACES",
                "RYU_RESEARCH_UPSTREAM",
                "RYU_RESEARCH_ENGINE_TOKEN",
                "RESEARCH_PORT",
            ]
            .into_iter()
            .map(|name| (name, std::env::var_os(name)))
            .collect();
            std::env::set_var("RYU_DIR", data_dir);
            std::env::remove_var("RESEARCH_DIR");
            std::env::set_var("RESEARCH_WORKSPACES", data_dir.join("research-workspaces"));
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
    fn empty_ryu_dir_materializes_the_complete_payload_idempotently() {
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
    async fn empty_ryu_dir_starts_and_second_supervisor_adopts_when_python_exists() {
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
