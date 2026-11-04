//! Spawning the Python engine with a cleared, allowlisted environment.

use std::ffi::OsString;
use std::io::ErrorKind;
use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, Result};
use tokio::process::{Child, Command};

use super::{engine_seal_env, EngineTarget};

pub(super) async fn spawn_python(
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

pub(super) fn python_candidates(code_dir: &Path) -> Vec<OsString> {
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
