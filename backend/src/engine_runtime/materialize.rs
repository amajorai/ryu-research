//! Publishing the embedded engine payload and the engine bearer token on disk.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{bail, Context, Result};
use rand::{rngs::OsRng, RngCore};
use sha2::{Digest, Sha256};

use super::{nonempty_env, workspaces_dir, DIGEST_MARKER, EMBEDDED_FILES};

pub(super) fn payload_digest() -> String {
    let mut hasher = Sha256::new();
    for file in EMBEDDED_FILES {
        hasher.update((file.path.len() as u64).to_le_bytes());
        hasher.update(file.path.as_bytes());
        hasher.update((file.bytes.len() as u64).to_le_bytes());
        hasher.update(file.bytes);
    }
    hex::encode(hasher.finalize())
}

pub(super) fn safe_embedded_path(path: &str) -> Result<PathBuf> {
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

pub(super) fn valid_materialization(path: &Path, digest: &str) -> bool {
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

pub(super) fn create_private_dir(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

pub(super) fn ensure_engine_token() -> Result<String> {
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

pub(super) fn random_hex(byte_count: usize) -> String {
    let mut bytes = vec![0_u8; byte_count];
    OsRng.fill_bytes(&mut bytes);
    hex::encode(bytes)
}
