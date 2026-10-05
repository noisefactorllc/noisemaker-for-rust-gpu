//! Render a Polymorphic DSL program to a PNG through the host API.
//!
//! ```text
//! cargo run --release -p noisemaker-for-rust-gpu --example render_dsl -- \
//!     PROGRAM.dsl|'DSL SOURCE' [OUT.png] [--size N] [--time T] [--frames N] [--media IMAGE.png]
//! ```
//!
//! The program is loaded as the reference demo page loads it
//! ([`DemoHost`]: compiled by a [`CanvasRenderer`], its parameters applied
//! through ProgramState, its host inputs — text canvases, meshes, overlays and
//! the media image — produced), rendered `--frames` times at the normalized
//! loop time `--time`, and the render surface is written in the orientation
//! a canvas shows it. A program that does not compile is reported with the
//! DSL error formatter and exit status 1.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::rc::Rc;

use noisemaker_gpu::demo::{DemoHost, DemoHostOptions};
use noisemaker_gpu::dsl::error_formatter::format_compile_error;
use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions};
use noisemaker_gpu::png_io::{read_png_rgba8, write_png_rgba8};
use noisemaker_gpu::{GpuDevice, Orientation, RenderError};

const USAGE: &str = "usage: render_dsl PROGRAM.dsl|'DSL SOURCE' [OUT.png] [--size N] [--time T] [--frames N] [--media IMAGE.png]";

struct Args {
    program: String,
    out: PathBuf,
    size: u32,
    time: f64,
    frames: u32,
    media: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut positional = Vec::new();
    let mut args = Args {
        program: String::new(),
        out: PathBuf::from("render.png"),
        size: 512,
        time: 0.25,
        frames: 8,
        media: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| it.next().ok_or_else(|| format!("{name} needs a value"));
        match arg.as_str() {
            "--size" => {
                args.size = value("--size")?
                    .parse()
                    .map_err(|e| format!("--size: {e}"))?
            }
            "--time" => {
                args.time = value("--time")?
                    .parse()
                    .map_err(|e| format!("--time: {e}"))?
            }
            "--frames" => {
                args.frames = value("--frames")?
                    .parse()
                    .map_err(|e| format!("--frames: {e}"))?
            }
            "--media" => args.media = Some(PathBuf::from(value("--media")?)),
            "-h" | "--help" => return Err(String::new()),
            _ => positional.push(arg),
        }
    }
    let mut positional = positional.into_iter();
    args.program = positional.next().ok_or("no program given")?;
    if let Some(out) = positional.next() {
        args.out = PathBuf::from(out);
    }
    if let Some(extra) = positional.next() {
        return Err(format!("unexpected argument '{extra}'"));
    }
    Ok(args)
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

fn run(args: &Args, source: &str) -> Result<(), RenderError> {
    let device = GpuDevice::create(&Default::default()).map_err(RenderError::Js)?;
    let renderer = CanvasRenderer::new(
        &device,
        CanvasRendererOptions {
            width: args.size,
            height: args.size,
            ..Default::default()
        },
    );
    let default_media = match &args.media {
        Some(path) => Some(Rc::new(read_png_rgba8(path).map_err(RenderError::Js)?)),
        None => None,
    };
    let mut host = DemoHost::new(
        renderer,
        DemoHostOptions {
            default_media,
            ..Default::default()
        },
    );
    // rebuildPipelineFromDsl: compile, load ProgramState, initialize the
    // controls (which write every parameter), apply the step values.
    host.rebuild_pipeline_from_dsl(source, true)?;
    // Finish what the page does asynchronously: media, text, meshes, overlays.
    host.settle()?;

    let renderer = host.renderer_mut();
    renderer.sync_time(args.time);
    for _ in 0..args.frames {
        renderer.render(args.time)?;
    }
    let pixels = renderer.read_output()?.oriented(Orientation::Presented);
    write_png_rgba8(&args.out, pixels.width, pixels.height, &pixels.data)
        .map_err(RenderError::Js)?;
    renderer.dispose()
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(args) => args,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("render_dsl: {e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let source = match program_source(&args.program) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("render_dsl: {e}");
            return ExitCode::from(2);
        }
    };
    match run(&args, &source) {
        Ok(()) => {
            println!("wrote {}", args.out.display());
            ExitCode::SUCCESS
        }
        Err(e) => {
            match e.dsl_error() {
                Some(error) => eprintln!("{}", format_compile_error(&source, error)),
                None => eprintln!("render_dsl: {e}"),
            }
            ExitCode::from(1)
        }
    }
}
