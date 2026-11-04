use std::collections::HashSet;
use std::fs;
use std::process::Stdio;
use std::sync::Arc;

use super::materialize::{payload_digest, safe_embedded_path, valid_materialization};
use super::probe::verify_engine_proof;
use super::spawn::python_candidates;
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

/// Modules an embedded engine file imports relatively (`from . import a, b`
/// and `from .a import x`), as `ryu_research/<module>.py` paths.
fn relative_imports(source: &str) -> Vec<String> {
    let mut modules = Vec::new();
    for line in source.lines().map(str::trim_start) {
        let Some(rest) = line.strip_prefix("from .") else {
            continue;
        };
        let names = match rest.strip_prefix(" import ") {
            Some(listed) => listed.split(',').map(str::trim).collect::<Vec<_>>(),
            None => vec![rest.split_whitespace().next().unwrap_or_default()],
        };
        for name in names {
            let module = name.split_whitespace().next().unwrap_or_default();
            let module = module.trim_matches(|ch| ch == '(' || ch == ')');
            if !module.is_empty() {
                modules.push(format!("ryu_research/{}.py", module.replace('.', "/")));
            }
        }
    }
    modules
}

#[test]
fn relative_imports_parse_both_import_forms() {
    assert_eq!(
        relative_imports("from . import campaigns, sandbox\nfrom .sealed_errors import body\n"),
        [
            "ryu_research/campaigns.py",
            "ryu_research/sandbox.py",
            "ryu_research/sealed_errors.py"
        ]
    );
    assert!(relative_imports("from __future__ import annotations\nimport os\n").is_empty());
}

/// Every module an embedded engine file imports must itself be embedded:
/// otherwise the materialized engine dies on import and the supervisor only
/// sees a readiness timeout (as when `sealed_errors.py` was left out).
#[test]
fn embedded_engine_imports_only_embedded_modules() {
    let embedded: HashSet<&str> = EMBEDDED_FILES.iter().map(|file| file.path).collect();
    let mut missing = Vec::new();
    for file in EMBEDDED_FILES
        .iter()
        .filter(|file| file.path.starts_with("ryu_research/") && file.path.ends_with(".py"))
    {
        let source = std::str::from_utf8(file.bytes).expect("engine sources are UTF-8");
        for module in relative_imports(source) {
            if !embedded.contains(module.as_str()) {
                missing.push(format!("{} imports {module}", file.path));
            }
        }
    }
    assert!(missing.is_empty(), "not embedded: {missing:?}");
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
