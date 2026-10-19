use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "nm-cli-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        std::fs::write(
            path.join("mesh.dsl"),
            "search render\nmeshLoader().meshRender().write(o0)\nrender(o0)\n",
        )
        .unwrap();
        Self(path)
    }

    fn command(&self, mode: &str, obj: Option<&Path>) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_nm-render"));
        cmd.current_dir(&self.0).arg(mode);
        if mode == "batch" {
            let mut fixture = serde_json::json!({
                "dsl": "mesh.dsl", "out": "out.png", "size": 16, "frames": 1
            });
            if let Some(obj) = obj {
                fixture["obj"] = serde_json::json!(obj);
            }
            std::fs::write(
                self.0.join("batch.json"),
                serde_json::to_vec(&vec![fixture]).unwrap(),
            )
            .unwrap();
            cmd.arg("batch.json");
        } else {
            cmd.args(["--dsl", "mesh.dsl", "--size", "16", "--frames", "1"]);
            if mode == "animate" {
                cmd.args(["--out-dir", "frames"]);
            } else {
                cmd.args(["--out", "out.png"]);
            }
            if let Some(obj) = obj {
                cmd.arg("--obj").arg(obj);
            }
        }
        cmd
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn succeeded(output: &Output) {
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn explicit_obj_read_errors_fail_every_render_entrypoint() {
    for mode in ["render", "animate", "batch"] {
        for name in ["missing.obj", "directory.obj"] {
            let scratch = Scratch::new();
            if name == "directory.obj" {
                std::fs::create_dir(scratch.0.join(name)).unwrap();
            }
            let out = scratch
                .command(mode, Some(Path::new(name)))
                .output()
                .unwrap();
            assert!(!out.status.success(), "{mode} accepted {name}");
            assert!(String::from_utf8_lossy(&out.stderr).contains(name));
            assert!(!scratch.0.join("out.png").exists());
            assert!(!scratch.0.join("frames/frame_00000.png").exists());
        }
    }
}

#[test]
fn optional_sidecar_and_explicit_valid_obj_render() {
    for mode in ["render", "animate", "batch"] {
        for explicit in [false, true] {
            let scratch = Scratch::new();
            let obj = Path::new("triangle.obj");
            std::fs::write(
                scratch.0.join(obj),
                "v -1 -1 0\nv 1 -1 0\nv 0 1 0\nf 1 2 3\n",
            )
            .unwrap();
            let out = scratch
                .command(mode, explicit.then_some(obj))
                .output()
                .unwrap();
            succeeded(&out);
            let image = if mode == "animate" {
                "frames/frame_00000.png"
            } else {
                "out.png"
            };
            assert!(std::fs::metadata(scratch.0.join(image)).unwrap().len() > 0);
        }
    }
}

#[test]
fn animation_rejects_existing_sequence_without_changing_user_files() {
    let scratch = Scratch::new();
    let frames = scratch.0.join("frames");
    std::fs::create_dir(&frames).unwrap();
    std::fs::write(frames.join("keep.txt"), b"user data").unwrap();
    std::fs::write(frames.join("frame_custom.png"), b"unrelated image").unwrap();
    let mut first = Command::new(env!("CARGO_BIN_EXE_nm-render"));
    first.current_dir(&scratch.0).args([
        "animate",
        "--dsl",
        "mesh.dsl",
        "--size",
        "16",
        "--frames",
        "4",
        "--out-dir",
        "frames",
    ]);
    succeeded(&first.output().unwrap());
    let before: Vec<_> = (0..4)
        .map(|i| std::fs::read(frames.join(format!("frame_{i:05}.png"))).unwrap())
        .collect();
    let out = scratch.command("animate", None).output().unwrap();
    assert!(
        !out.status.success(),
        "animation reused an existing sequence"
    );
    assert!(String::from_utf8_lossy(&out.stderr).contains("frame_"));
    for (i, bytes) in before.iter().enumerate() {
        assert_eq!(
            *bytes,
            std::fs::read(frames.join(format!("frame_{i:05}.png"))).unwrap()
        );
    }
    assert_eq!(
        std::fs::read(frames.join("keep.txt")).unwrap(),
        b"user data"
    );
    assert_eq!(
        std::fs::read(frames.join("frame_custom.png")).unwrap(),
        b"unrelated image"
    );
}

#[test]
fn zero_frame_animation_fails_without_creating_output() {
    let scratch = Scratch::new();
    let out = Command::new(env!("CARGO_BIN_EXE_nm-render"))
        .current_dir(&scratch.0)
        .args([
            "animate",
            "--dsl",
            "mesh.dsl",
            "--out-dir",
            "frames",
            "--frames",
            "0",
            "--size",
            "16",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(!scratch.0.join("frames").exists());
}

/// Runs `render` on the repository's synth_media program (the asymmetric
/// test card) and returns the written PNG's RGBA rows.
fn render_test_card(scratch: &Scratch, out: &str, orientation: Option<&str>) -> Vec<u8> {
    let program =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../parity/coverage/synth_media.dsl");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nm-render"));
    cmd.current_dir(&scratch.0)
        .arg("render")
        .arg("--dsl")
        .arg(&program)
        .args(["--out", out, "--size", "64", "--frames", "1"]);
    if let Some(orientation) = orientation {
        cmd.args(["--orientation", orientation]);
    }
    succeeded(&cmd.output().unwrap());
    noisemaker_gpu::png_io::read_png_rgba8(&scratch.0.join(out))
        .unwrap()
        .data
}

#[test]
fn render_writes_the_image_upright_by_default() {
    let scratch = Scratch::new();
    let default = render_test_card(&scratch, "default.png", None);
    let presented = render_test_card(&scratch, "presented.png", Some("presented"));
    let texture = render_test_card(&scratch, "texture.png", Some("texture"));
    assert_eq!(default, presented);
    let row = 64 * 4;
    let flipped: Vec<u8> = texture.chunks(row).rev().flatten().copied().collect();
    assert_eq!(default, flipped);
    assert_ne!(
        default, texture,
        "the test card must not be vertically symmetric"
    );
}

#[test]
fn render_rejects_values_that_cannot_give_the_image() {
    let cases: [(&[&str], &str); 6] = [
        (&["--frames", "0"], "--frames must be at least 1"),
        (
            &["--run-seconds", "1", "--sample-every", "0"],
            "--sample-every must be a positive",
        ),
        (
            &["--run-frames", "4", "--sample-every-frames", "0"],
            "--sample-every-frames must be at least 1",
        ),
        (&["--size", "0"], "--size must be 1 to "),
        (&["--size", "70000"], "--size must be 1 to "),
        (
            &["--width", "16", "--height", "0"],
            "--height must be 1 to ",
        ),
    ];
    for (options, message) in cases {
        let scratch = Scratch::new();
        let out = Command::new(env!("CARGO_BIN_EXE_nm-render"))
            .current_dir(&scratch.0)
            .args(["render", "--dsl", "mesh.dsl", "--out", "out.png"])
            .args(options)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert_eq!(out.status.code(), Some(2), "{options:?}: {stderr}");
        assert!(stderr.contains(message), "{options:?}: {stderr}");
        let written: Vec<_> = std::fs::read_dir(&scratch.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .filter(|name| name.ends_with(".png"))
            .collect();
        assert!(written.is_empty(), "{options:?} wrote {written:?}");
    }
}
