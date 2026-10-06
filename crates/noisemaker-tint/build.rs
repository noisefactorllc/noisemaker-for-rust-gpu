//! Builds Tint from the Dawn revision pinned in `dawn.json` with Dawn's own
//! CMake build (configured by `cmake/CMakeLists.txt` for the WGSL reader and
//! the MSL writer only), together with the C shim in `shim/`, and links the
//! result into one static library.
//!
//! Sources, all verified against the pin:
//!
//! * Dawn: `NM_DAWN_SOURCE`, a Dawn git checkout at exactly the pinned commit
//!   with no modified tracked file (or a tree this script extracted, which
//!   carries its marker file); otherwise the pinned commit is fetched
//!   (`git fetch --depth 1 <repository> <commit>` into a temporary bare
//!   repository; git checks the hash of every object it receives, so the
//!   fetched commit id proves the tree), and the paths the Tint build reads
//!   (`dawn.paths`) are extracted into the cache with `git archive`.
//! * The third-party directories Dawn's CMake reads (`dependencies`): fetched
//!   the same way, each commit checked against the gitlink the Dawn commit
//!   records at its path; a populated, clean checkout at that path inside
//!   `NM_DAWN_SOURCE` is used as is.
//! * CMake: `CMAKE` or `cmake` on `PATH` when at least `cmake.minimum`;
//!   otherwise the pinned release for the host is downloaded into the cache
//!   and checked against its SHA-256.
//!
//! `NM_TINT_CACHE` overrides the cache directory (default: the user cache
//! directory, `noisemaker-tint`). The native build itself lives in `OUT_DIR`
//! and is incremental.

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::Value;

const MARKER: &str = ".noisemaker-tint-commit";

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    for file in [
        "build.rs",
        "dawn.json",
        "cmake/CMakeLists.txt",
        "shim/nm_tint.cc",
        "shim/nm_tint.h",
    ] {
        println!("cargo:rerun-if-changed={}", manifest.join(file).display());
    }
    for var in [
        "NM_DAWN_SOURCE",
        "NM_TINT_CACHE",
        "CMAKE",
        "MACOSX_DEPLOYMENT_TARGET",
        "CC",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }

    let pin: Value = serde_json::from_str(
        &fs::read_to_string(manifest.join("dawn.json")).expect("reading dawn.json"),
    )
    .expect("dawn.json is JSON");
    let dawn_commit = str_at(&pin, &["dawn", "commit"]);
    let dawn_repository = str_at(&pin, &["dawn", "repository"]);
    let dawn_paths: Vec<String> = pin["dawn"]["paths"]
        .as_array()
        .expect("dawn.paths")
        .iter()
        .map(|p| p.as_str().expect("dawn.paths entries").to_owned())
        .collect();
    println!("cargo:rustc-env=NM_TINT_DAWN_COMMIT={dawn_commit}");

    let cache = cache_dir();
    fs::create_dir_all(&cache)
        .unwrap_or_else(|e| panic!("creating the cache {}: {e}", cache.display()));

    let user_source = env::var_os("NM_DAWN_SOURCE").map(PathBuf::from);
    let dawn_source = match &user_source {
        Some(dir) => {
            verify_user_dawn(dir, &dawn_commit);
            dir.clone()
        }
        None => fetch_dawn(&cache, &dawn_repository, &dawn_commit, &dawn_paths, &pin),
    };

    let mut cmake_defines = vec![
        ("NM_DAWN_SOURCE_DIR".to_owned(), path_str(&dawn_source)),
        ("NM_DAWN_COMMIT".to_owned(), dawn_commit.clone()),
    ];
    for dep in pin["dependencies"].as_array().expect("dependencies") {
        let path = dep["path"].as_str().expect("dependency path");
        let commit = dep["commit"].as_str().expect("dependency commit");
        let repository = dep["repository"].as_str().expect("dependency repository");
        let variable = dep["cmake_variable"]
            .as_str()
            .expect("dependency cmake_variable");
        if let Some(dir) = user_source.as_ref().filter(|d| d.join(".git").exists()) {
            // The checkout's own gitlink must name the pinned commit.
            let recorded = gitlink(dir, "HEAD", path);
            if recorded != commit {
                panic!("NM_DAWN_SOURCE records {path} at {recorded}; dawn.json pins {commit}");
            }
        }
        let populated = user_source
            .as_ref()
            .map(|dir| dir.join(path))
            .filter(|dir| clean_checkout_at(dir, commit));
        let dir = match populated {
            Some(dir) => dir,
            None => fetch_dependency(&cache, path, repository, commit),
        };
        cmake_defines.push((variable.to_owned(), path_str(&dir)));
    }

    let cmake = find_cmake(&pin, &cache);
    let build_dir = out_dir.join("tint-build");
    configure_and_build(&cmake, &manifest, &build_dir, &cmake_defines);
    link(&build_dir, &out_dir);
}

fn str_at(value: &Value, keys: &[&str]) -> String {
    let mut v = value;
    for k in keys {
        v = &v[*k];
    }
    v.as_str()
        .unwrap_or_else(|| panic!("dawn.json: {} is not a string", keys.join(".")))
        .to_owned()
}

fn path_str(path: &Path) -> String {
    path.to_str()
        .unwrap_or_else(|| panic!("non-UTF-8 path {}", path.display()))
        .to_owned()
}

fn cache_dir() -> PathBuf {
    if let Some(dir) = env::var_os("NM_TINT_CACHE") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(dir).join("noisemaker-tint");
    }
    if cfg!(windows) {
        if let Some(dir) = env::var_os("LOCALAPPDATA") {
            return PathBuf::from(dir).join("noisemaker-tint");
        }
    } else if let Some(home) = env::var_os("HOME") {
        let home = PathBuf::from(home);
        return if cfg!(target_os = "macos") {
            home.join("Library/Caches/noisemaker-tint")
        } else {
            home.join(".cache/noisemaker-tint")
        };
    }
    panic!("no cache directory: set NM_TINT_CACHE");
}

// ---------------------------------------------------------------------------
// processes

fn describe(cmd: &Command) -> String {
    let mut s = cmd.get_program().to_string_lossy().into_owned();
    for a in cmd.get_args() {
        s.push(' ');
        s.push_str(&a.to_string_lossy());
    }
    s
}

/// Run `cmd`, returning its stdout; panic with its stderr on failure.
fn run(cmd: &mut Command) -> String {
    let output = cmd
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|e| panic!("running `{}`: {e}", describe(cmd)));
    if !output.status.success() {
        panic!(
            "`{}` failed ({}):\n{}{}",
            describe(cmd),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// Run `cmd` with inherited output (long builds stream into cargo's log).
fn run_streaming(cmd: &mut Command) {
    let status = cmd
        .stdin(Stdio::null())
        .status()
        .unwrap_or_else(|e| panic!("running `{}`: {e}", describe(cmd)));
    if !status.success() {
        panic!("`{}` failed ({status})", describe(cmd));
    }
}

fn git(dir: &Path) -> Command {
    let mut cmd = Command::new("git");
    cmd.arg("-C").arg(dir);
    cmd
}

// ---------------------------------------------------------------------------
// sources

/// The commit `treeish` records at `path` (a gitlink).
fn gitlink(repo: &Path, treeish: &str, path: &str) -> String {
    let line = run(git(repo).args(["ls-tree", treeish, "--", path]));
    let mut fields = line.split_whitespace();
    match (fields.next(), fields.next(), fields.next()) {
        (Some("160000"), Some("commit"), Some(id)) => id.to_owned(),
        _ => panic!(
            "{} does not record a gitlink at {path} in {treeish}: {line:?}",
            repo.display()
        ),
    }
}

/// `dir` is a git checkout at `commit` with no modified tracked file.
fn clean_checkout_at(dir: &Path, commit: &str) -> bool {
    if !dir.join(".git").exists() {
        return false;
    }
    let head = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(["rev-parse", "HEAD"])
        .output();
    let Ok(head) = head else { return false };
    if String::from_utf8_lossy(&head.stdout).trim() != commit {
        return false;
    }
    run(git(dir).args([
        "status",
        "--porcelain",
        "--untracked-files=no",
        "--ignore-submodules=all",
    ]))
    .trim()
    .is_empty()
}

fn marker_matches(dir: &Path, commit: &str) -> bool {
    fs::read_to_string(dir.join(MARKER)).is_ok_and(|m| m.trim() == commit)
}

fn verify_user_dawn(dir: &Path, commit: &str) {
    if marker_matches(dir, commit) {
        return;
    }
    if !dir.join(".git").exists() {
        panic!(
            "NM_DAWN_SOURCE={} is neither a git checkout nor a tree extracted by this build \
             (no {MARKER} naming {commit})",
            dir.display()
        );
    }
    let head = run(git(dir).args(["rev-parse", "HEAD"]));
    if head.trim() != commit {
        panic!(
            "NM_DAWN_SOURCE={} is at {}; dawn.json pins {commit}",
            dir.display(),
            head.trim()
        );
    }
    if !clean_checkout_at(dir, commit) {
        panic!(
            "NM_DAWN_SOURCE={} has modified tracked files; the pinned Dawn is built unmodified",
            dir.display()
        );
    }
}

/// Fetch `commit` of `repository` into the bare repository `bare` and check
/// that the fetched object is that commit.
fn fetch_commit(bare: &Path, repository: &str, commit: &str) {
    if !bare.join("HEAD").exists() {
        fs::create_dir_all(bare).unwrap();
        run(git(bare).args(["init", "--quiet", "--bare"]));
    }
    let have = Command::new("git")
        .arg("-C")
        .arg(bare)
        .args(["cat-file", "-e", &format!("{commit}^{{commit}}")])
        .status()
        .is_ok_and(|s| s.success());
    if !have {
        eprintln!("noisemaker-tint: fetching {repository} at {commit}");
        run(git(bare).args([
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--no-tags",
            repository,
            commit,
        ]));
    }
    let id = run(git(bare).args(["rev-parse", "--verify", &format!("{commit}^{{commit}}")]));
    if id.trim() != commit {
        panic!("{repository}: fetched {} instead of {commit}", id.trim());
    }
}

/// Extract `paths` (everything when empty) of `commit` from `bare` into
/// `dest`, atomically, with the marker file.
fn extract(bare: &Path, commit: &str, paths: &[String], dest: &Path) {
    let tmp = dest.with_extension(format!("tmp-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let mut archive = git(bare);
    archive
        .args(["archive", "--format=tar", commit])
        .args(paths);
    let mut archive_child = archive
        .stdout(Stdio::piped())
        .stdin(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("running `{}`: {e}", describe(&archive)));
    let tar_status = Command::new("tar")
        .arg("-x")
        .arg("-f")
        .arg("-")
        .arg("-C")
        .arg(&tmp)
        .stdin(archive_child.stdout.take().unwrap())
        .status()
        .expect("running tar");
    let archive_status = archive_child.wait().unwrap();
    if !archive_status.success() || !tar_status.success() {
        panic!("extracting {commit} from {} failed", bare.display());
    }
    fs::write(tmp.join(MARKER), format!("{commit}\n")).unwrap();
    if fs::rename(&tmp, dest).is_err() {
        // Another build extracted the same commit first.
        let _ = fs::remove_dir_all(&tmp);
        if !marker_matches(dest, commit) {
            panic!("could not move {} into place", dest.display());
        }
    }
}

fn fetch_dawn(
    cache: &Path,
    repository: &str,
    commit: &str,
    paths: &[String],
    pin: &Value,
) -> PathBuf {
    let dest = cache.join(format!("dawn-{commit}"));
    if marker_matches(&dest, commit) {
        return dest;
    }
    let bare = scratch_repository(cache, "dawn");
    fetch_commit(&bare, repository, commit);
    for dep in pin["dependencies"].as_array().expect("dependencies") {
        let path = dep["path"].as_str().unwrap();
        let pinned = dep["commit"].as_str().unwrap();
        let recorded = gitlink(&bare, commit, path);
        if recorded != pinned {
            panic!("Dawn {commit} records {path} at {recorded}; dawn.json pins {pinned}");
        }
    }
    extract(&bare, commit, paths, &dest);
    let _ = fs::remove_dir_all(&bare);
    dest
}

/// A bare repository of this build process alone (concurrent builds never
/// share one), removed once its commit is extracted.
fn scratch_repository(cache: &Path, name: &str) -> PathBuf {
    let bare = cache
        .join("git")
        .join(format!("{name}-{}.git", std::process::id()));
    let _ = fs::remove_dir_all(&bare);
    bare
}

fn fetch_dependency(cache: &Path, path: &str, repository: &str, commit: &str) -> PathBuf {
    let name = path.replace('/', "_");
    let dest = cache.join(format!("{name}-{commit}"));
    if marker_matches(&dest, commit) {
        return dest;
    }
    let bare = scratch_repository(cache, &name);
    fetch_commit(&bare, repository, commit);
    extract(&bare, commit, &[], &dest);
    let _ = fs::remove_dir_all(&bare);
    dest
}

// ---------------------------------------------------------------------------
// CMake

fn version_of(cmake: &Path) -> Option<(u32, u32)> {
    let out = Command::new(cmake).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let version = text.lines().next()?.strip_prefix("cmake version ")?;
    let mut parts = version.split('.');
    Some((
        parts.next()?.parse().ok()?,
        parts.next()?.trim().parse().ok()?,
    ))
}

fn find_cmake(pin: &Value, cache: &Path) -> PathBuf {
    let minimum = str_at(pin, &["cmake", "minimum"]);
    let (major, minor) = minimum
        .split_once('.')
        .expect("cmake.minimum is MAJOR.MINOR");
    let minimum = (major.parse::<u32>().unwrap(), minor.parse::<u32>().unwrap());
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(c) = env::var_os("CMAKE") {
        candidates.push(PathBuf::from(c));
    }
    candidates.push(PathBuf::from("cmake"));
    for candidate in candidates {
        if version_of(&candidate).is_some_and(|v| v >= minimum) {
            return candidate;
        }
    }

    let host = env::var("HOST").unwrap();
    let platform = if host.contains("apple-darwin") {
        "macos"
    } else if host.starts_with("x86_64") && host.contains("linux") {
        "linux-x86_64"
    } else if host.starts_with("aarch64") && host.contains("linux") {
        "linux-aarch64"
    } else {
        panic!(
            "CMake >= {}.{} is required: install it or set CMAKE",
            minimum.0, minimum.1
        );
    };
    let download = &pin["cmake"]["downloads"][platform];
    let version = str_at(pin, &["cmake", "version"]);
    let url = download["url"].as_str().expect("cmake download url");
    let sha256 = download["sha256"].as_str().expect("cmake download sha256");
    let binary = download["binary"].as_str().expect("cmake download binary");
    let dest = cache.join(format!("cmake-{version}-{platform}"));
    let cmake = dest.join(binary);
    if marker_matches(&dest, sha256) && cmake.exists() {
        return cmake;
    }
    eprintln!("noisemaker-tint: downloading CMake {version} ({url})");
    let tmp = dest.with_extension(format!("tmp-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    fs::create_dir_all(&tmp).unwrap();
    let archive = tmp.join("cmake.tar.gz");
    run(Command::new("curl")
        .args([
            "--fail",
            "--silent",
            "--show-error",
            "--location",
            "--output",
        ])
        .arg(&archive)
        .arg(url));
    let mut bytes = Vec::new();
    fs::File::open(&archive)
        .unwrap()
        .read_to_end(&mut bytes)
        .unwrap();
    let digest = hex(&sha256_digest(&bytes));
    if digest != sha256 {
        panic!("{url}: SHA-256 {digest}, dawn.json pins {sha256}");
    }
    run(Command::new("tar")
        .arg("-xzf")
        .arg(&archive)
        .arg("-C")
        .arg(&tmp));
    fs::remove_file(&archive).unwrap();
    fs::write(tmp.join(MARKER), format!("{sha256}\n")).unwrap();
    if fs::rename(&tmp, &dest).is_err() {
        let _ = fs::remove_dir_all(&tmp);
    }
    if !cmake.exists() {
        panic!("{} is missing after extracting {url}", cmake.display());
    }
    cmake
}

fn configure_and_build(
    cmake: &Path,
    manifest: &Path,
    build_dir: &Path,
    defines: &[(String, String)],
) {
    let target = env::var("TARGET").unwrap();
    let mut configure = Command::new(cmake);
    configure
        .arg("-S")
        .arg(manifest.join("cmake"))
        .arg("-B")
        .arg(build_dir)
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-Wno-dev");
    if !build_dir.join("CMakeCache.txt").exists() {
        let ninja = Command::new("ninja")
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success());
        if ninja {
            configure.args(["-G", "Ninja"]);
        } else if cfg!(unix) {
            configure.args(["-G", "Unix Makefiles"]);
        }
    }
    if target.contains("apple-darwin") {
        // The deployment target rustc links for (its default when unset).
        let deployment = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| {
            if target.starts_with("aarch64") {
                "11.0"
            } else {
                "10.12"
            }
            .to_owned()
        });
        let arch = if target.starts_with("aarch64") {
            "arm64"
        } else {
            "x86_64"
        };
        configure
            .arg(format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={deployment}"))
            .arg(format!("-DCMAKE_OSX_ARCHITECTURES={arch}"));
    }
    for (name, value) in defines {
        configure.arg(format!("-D{name}={value}"));
    }
    run(&mut configure);

    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "4".into());
    run_streaming(Command::new(cmake).arg("--build").arg(build_dir).args([
        "--target",
        "nm_tint_shim",
        "--config",
        "Release",
        "--parallel",
        &jobs,
    ]));
}

// ---------------------------------------------------------------------------
// linking

fn collect_archives(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_archives(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

fn link(build_dir: &Path, out_dir: &Path) {
    let target = env::var("TARGET").unwrap();
    let lib_dir = out_dir.join("lib");
    fs::create_dir_all(&lib_dir).unwrap();
    let windows = target.contains("windows");
    let mut archives = Vec::new();
    collect_archives(build_dir, if windows { "lib" } else { "a" }, &mut archives);
    archives.sort();
    if !archives.iter().any(|a| {
        a.file_stem()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.ends_with("nm_tint_shim"))
    }) {
        panic!("the CMake build produced no nm_tint_shim library");
    }
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if windows {
        // MSVC's linker resolves symbols across libraries in any order.
        for archive in &archives {
            let name = archive.file_stem().unwrap().to_str().unwrap();
            fs::copy(archive, lib_dir.join(archive.file_name().unwrap())).unwrap();
            println!("cargo:rustc-link-lib=static={name}");
        }
        return;
    }
    // One archive, so that no linker depends on the order of Tint's libraries.
    let merged = lib_dir.join("libnm_tint.a");
    let _ = fs::remove_file(&merged);
    if target.contains("apple") {
        run(Command::new("libtool")
            .args(["-static", "-no_warning_for_no_symbols", "-o"])
            .arg(&merged)
            .args(&archives));
    } else {
        let mut script = format!("create {}\n", merged.display());
        for archive in &archives {
            script.push_str(&format!("addlib {}\n", archive.display()));
        }
        script.push_str("save\nend\n");
        let mut ar = Command::new(env::var("AR").unwrap_or_else(|_| "ar".into()));
        ar.arg("-M").stdin(Stdio::piped());
        let mut child = ar.spawn().expect("running ar");
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), script.as_bytes()).unwrap();
        drop(child.stdin.take());
        if !child.wait().unwrap().success() {
            panic!("ar -M failed merging the Tint libraries");
        }
    }
    println!("cargo:rustc-link-lib=static=nm_tint");
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4), for the pinned CMake download.

fn sha256_digest(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for block in message.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*word);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [
                t1.wrapping_add(t2),
                v[0],
                v[1],
                v[2],
                v[3].wrapping_add(t1),
                v[4],
                v[5],
                v[6],
            ];
        }
        for (hi, vi) in h.iter_mut().zip(v) {
            *hi = hi.wrapping_add(vi);
        }
    }
    let mut out = [0u8; 32];
    for (chunk, word) in out.as_chunks_mut::<4>().0.iter_mut().zip(h) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
