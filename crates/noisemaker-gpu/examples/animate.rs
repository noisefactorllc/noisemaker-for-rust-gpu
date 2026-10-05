//! Render a Polymorphic DSL program over its loop to a numbered PNG sequence.
//!
//! ```text
//! cargo run --release -p noisemaker-for-rust-gpu --example animate -- \
//!     PROGRAM.dsl|'DSL SOURCE' OUT_DIR [--size N] [--fps F] [--loop-seconds S] [--frames N]
//! ```
//!
//! The renderer's loop is `loopDuration` seconds long (10 by default) and
//! every frame renders at a normalized loop time in 0..1. At `--fps` frames
//! per second, frame `i` is `i / fps` seconds into the loop, so it renders at
//! `((i / fps) % loop) / loop` — the time the reference's render loop
//! (`CanvasRenderer._renderLoop`) gives a frame drawn at that moment. One loop
//! (`fps * loop` frames) is written by default, as `OUT_DIR/frame_00000.png`,
//! `frame_00001.png`, ..., each in the orientation a canvas shows it. Encode
//! them with, e.g., `ffmpeg -framerate 30 -i OUT_DIR/frame_%05d.png out.mp4`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::dsl::error_formatter::format_compile_error;
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions};
use noisemaker_gpu::png_io::write_png_rgba8;
use noisemaker_gpu::{GpuDevice, Orientation, RenderError};

const USAGE: &str = "usage: animate PROGRAM.dsl|'DSL SOURCE' OUT_DIR [--size N] [--fps F] [--loop-seconds S] [--frames N]";

struct Args {
    program: String,
    out_dir: PathBuf,
    size: u32,
    fps: f64,
    loop_seconds: f64,
    frames: Option<u32>,
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let (mut size, mut fps, mut loop_seconds, mut frames) = (512, 30.0, 10.0, None);
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--size" => {
                size = value("--size")?
                    .parse()
                    .map_err(|e| format!("--size: {e}"))?
            }
            "--fps" => fps = value("--fps")?.parse().map_err(|e| format!("--fps: {e}"))?,
            "--loop-seconds" => {
                loop_seconds = value("--loop-seconds")?
                    .parse()
                    .map_err(|e| format!("--loop-seconds: {e}"))?
            }
            "--frames" => {
                frames = Some(
                    value("--frames")?
                        .parse()
                        .map_err(|e| format!("--frames: {e}"))?,
                )
            }
            "-h" | "--help" => return Err(String::new()),
            _ => positional.push(arg),
        }
    }
    if !(fps > 0.0 && loop_seconds > 0.0) {
        return Err("--fps and --loop-seconds must be positive".into());
    }
    let [program, out_dir]: [String; 2] = positional
        .try_into()
        .map_err(|_| "expected a program and an output directory".to_owned())?;
    Ok(Args {
        program,
        out_dir: PathBuf::from(out_dir),
        size,
        fps,
        loop_seconds,
        frames,
    })
}

/// A path to an existing file is read; anything else is DSL source.
fn program_source(program: &str) -> Result<String, String> {
    let path = Path::new(program);
    if path.is_file() {
        std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
    } else {
        Ok(program.to_owned())
    }
}

fn run(args: &Args, source: &str) -> Result<u32, RenderError> {
    let device = GpuDevice::create(&Default::default()).map_err(RenderError::Js)?;
    let renderer = CanvasRenderer::new(
        &device,
        CanvasRendererOptions {
            width: args.size,
            height: args.size,
            ..Default::default()
        },
    );
    let mut host = DemoHost::new(renderer, DemoHostOptions::default());
    host.rebuild_pipeline_from_dsl(source, true)?;
    host.settle()?;

    let renderer = host.renderer_mut();
    renderer.set_loop_duration(args.loop_seconds);
    let frames = args
        .frames
        .unwrap_or((args.fps * args.loop_seconds).round() as u32);
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| RenderError::Js(format!("{}: {e}", args.out_dir.display())))?;
    // The first frame starts the clock: its deltaTime is 0.
    renderer.sync_time(0.0);
    for i in 0..frames {
        let elapsed = f64::from(i) / args.fps;
        let time = (elapsed % args.loop_seconds) / args.loop_seconds;
        renderer.render(time)?;
        let pixels = renderer.read_output()?.oriented(Orientation::Presented);
        let path = args.out_dir.join(format!("frame_{i:05}.png"));
        write_png_rgba8(&path, pixels.width, pixels.height, &pixels.data)
            .map_err(RenderError::Js)?;
    }
    renderer.dispose()?;
    Ok(frames)
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("animate: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let source = match program_source(&args.program) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("animate: {e}");
            return ExitCode::from(2);
        }
    };
    match run(&args, &source) {
        Ok(frames) => {
            println!("wrote {frames} frames to {}", args.out_dir.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            match e.dsl_error() {
                Some(error) => eprintln!("{}", format_compile_error(&source, error)),
                None => eprintln!("animate: {e}"),
            }
            ExitCode::from(1)
        }
    }
}
