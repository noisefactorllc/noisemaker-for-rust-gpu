//! Helpers shared by the differential tests: the reference checkout, Node, and
//! scratch directories.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;

/// The repository root (two levels above this crate).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("crate lives in crates/<name>")
        .to_path_buf()
}

/// `$NM_REFERENCE_ROOT` when set and `node` runs; otherwise `None` after
/// explaining on stderr why the differential check does not run.
pub fn reference_env(test: &str) -> Option<PathBuf> {
    let Some(root) = std::env::var_os("NM_REFERENCE_ROOT") else {
        eprintln!("{test}: NM_REFERENCE_ROOT is not set; skipping the differential check");
        return None;
    };
    match Command::new("node").arg("--version").output() {
        Ok(out) if out.status.success() => Some(PathBuf::from(root)),
        _ => {
            eprintln!("{test}: node is not available; skipping the differential check");
            None
        }
    }
}

/// Runs `node tools/reference-host.mjs <args>` and returns its stdout.
pub fn reference_host(reference: &Path, args: &[&str]) -> Vec<u8> {
    let out = Command::new("node")
        .arg(repo_root().join("tools").join("reference-host.mjs"))
        .args(args)
        .env("NM_REFERENCE_ROOT", reference)
        .output()
        .expect("running node");
    assert!(
        out.status.success(),
        "reference-host {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// A fresh scratch directory under the target directory.
pub fn scratch_dir(name: &str) -> PathBuf {
    let base = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!("noisemaker-host-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
