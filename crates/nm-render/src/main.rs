//! nm-render — render Noisemaker DSL programs on the GPU, browse the effect
//! catalog, and dump the frontend's stages for the parity gates.
//!
//! `render` renders one program with the golden protocol of
//! `parity/batch-golden.mjs`; `animate` renders a DSL program over its loop to
//! a numbered PNG sequence (and an mp4 through `ffmpeg`); `batch` renders a
//! manifest of fixtures in one process and device (a fresh pipeline per
//! fixture); `effects` and `effect` list the catalog and one effect's
//! parameters; `dump` writes one frontend stage per program as JSON lines.
//!
//! A `--dsl` program runs through the demo host (`noisemaker_gpu::demo`):
//! compiled, parameters applied as the demo page applies them, host inputs
//! (media, text, overlays, meshes) produced natively. The demo's default
//! media image comes from `--media`, or is the demo's own test card, embedded
//! in the catalog (`noisemaker_effects::test_card_png`). A program that does
//! not compile is reported with `formatDslError` and a nonzero exit. A DSL
//! with a Portable sidecar (`<name>.portable.json`, or `--portable FILE`)
//! registers that user effect first, as `CanvasRenderer.registerPortableEffect`
//! does; one with a MIDI sidecar (`<name>.midi.json`) receives those MIDI
//! messages after it loads.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use noisemaker_dsl::js::number_to_string;
use noisemaker_dsl::registry::is_starter_effect;
use noisemaker_dsl::{Object, Registry, Stage, Value};
use noisemaker_gpu::Orientation;
use noisemaker_gpu::protocol::ParamOverride;

const EXAMPLES: &str = "\
Examples:
  nm-render effects --namespace synth
  nm-render effect synth/noise
  nm-render render --dsl program.dsl --out frame.png --width 1280 --height 720 \\
      --param step_0.octaves=4 --orientation presented
  nm-render animate --dsl program.dsl --out-dir frames --fps 30 --mp4 loop.mp4

Exit status: 0 on success, 1 when a program fails to compile or render, 2 for
usage and input errors.";

#[derive(Parser)]
#[command(
    name = "nm-render",
    version,
    about = "Render Noisemaker DSL programs on the GPU",
    long_about = "Render Noisemaker DSL programs on the GPU (wgpu), with the reference \
                  engine's frontend, runtime and demo-page host behavior.",
    after_help = EXAMPLES
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// The row order of written PNGs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OrientationArg {
    /// Texture row 0 first: readPixels order, the parity protocol's (the
    /// image appears upside down)
    Texture,
    /// As the reference's present() shows the frame on a canvas
    Presented,
}

impl From<OrientationArg> for Orientation {
    fn from(o: OrientationArg) -> Orientation {
        match o {
            OrientationArg::Texture => Orientation::Texture,
            OrientationArg::Presented => Orientation::Presented,
        }
    }
}

#[derive(Subcommand)]
enum Command {
    /// Render one program to a PNG with the golden protocol
    ///
    /// The program is loaded as the demo page loads it (a DSL) or as a fresh
    /// pipeline (a graph), host inputs are produced (or uploaded from PNGs),
    /// the pipeline is reset to a fresh state, `render(time)` runs FRAMES
    /// times, and the render surface is read back and written as an RGBA8
    /// PNG.
    Render {
        /// A reference graph (`<name>.graph.json` from parity/batch-golden.mjs)
        #[arg(long, conflicts_with = "dsl", required_unless_present = "dsl")]
        graph: Option<PathBuf>,
        /// A DSL program, loaded through the demo host
        #[arg(long)]
        dsl: Option<PathBuf>,
        /// Output PNG
        #[arg(long)]
        out: PathBuf,
        /// Square size in pixels
        #[arg(long, default_value_t = 256, conflicts_with_all = ["width", "height"])]
        size: u32,
        /// Width in pixels (with --height)
        #[arg(long, requires = "height")]
        width: Option<u32>,
        /// Height in pixels (with --width)
        #[arg(long, requires = "width")]
        height: Option<u32>,
        /// Normalized loop time (0..1) of every frame
        #[arg(long, default_value_t = 0.25)]
        time: f64,
        /// Frames to render
        #[arg(long, default_value_t = 8)]
        frames: u32,
        /// Set a parameter as the demo's control does (ProgramState.setValue),
        /// STEP_N.NAME=VALUE; VALUE is JSON when it parses as JSON (2.5, true,
        /// [1, 0, 0, 1], "3"), a string otherwise (repeatable; needs --dsl)
        #[arg(long = "param", value_name = "step_N.NAME=VALUE")]
        params: Vec<String>,
        /// Row order of the PNG
        #[arg(long, value_enum, default_value_t = OrientationArg::Texture)]
        orientation: OrientationArg,
        /// Host texture overrides, ID=PNG (repeatable): texture contents
        /// uploaded in place of the natively produced input
        #[arg(long = "host-texture", value_name = "ID=PNG")]
        host_textures: Vec<String>,
        /// The OBJ loaded into mesh0 (default for --dsl: the .obj next to it)
        #[arg(long)]
        obj: Option<PathBuf>,
        /// The demo's default media image (default: the demo's test card,
        /// embedded)
        #[arg(long)]
        media: Option<PathBuf>,
        /// Write the graph the fixture rendered as JSON
        #[arg(long)]
        graph_out: Option<PathBuf>,
        /// A Portable effect definition registered before the program loads
        /// (default for --dsl: the .portable.json next to it, its WGSL and
        /// GLSL in <name>.<program>.wgsl and .glsl files beside it)
        #[arg(long, requires = "dsl")]
        portable: Option<PathBuf>,
        /// Timed mode: seconds to run, stepping render(((frame + 1) / 600) % 1)
        #[arg(long, requires = "sample_every")]
        run_seconds: Option<f64>,
        /// Timed mode: seconds between samples (OUT_STEM.tSEC.png)
        #[arg(long)]
        sample_every: Option<f64>,
        /// Timed mode: total frames to run (stepping as --run-seconds does)
        #[arg(long)]
        run_frames: Option<u64>,
        /// Timed mode: a sample every N frames (OUT_STEM.fFRAMES.png)
        #[arg(long, value_name = "N")]
        sample_every_frames: Option<u64>,
        /// Timed mode: a sample after each of these frame counts
        /// (OUT_STEM.fFRAMES.png), e.g. 1,2,3,10
        #[arg(long, value_name = "LIST", value_delimiter = ',')]
        sample_frames: Vec<u64>,
        /// Read a backend texture (or a global surface: its current read
        /// texture) back as raw little-endian float32 RGBA at every sample,
        /// OUT_STEM[.LABEL].ID.bin (repeatable)
        #[arg(long = "dump-texture", value_name = "ID")]
        dump_textures: Vec<String>,
        /// Also snapshot the --dump-texture textures after every executed
        /// pass of each sampled frame: OUT_STEM[.LABEL].pNNN.ID.bin, the pass
        /// list in OUT_STEM[.LABEL].passes.json
        #[arg(long, requires = "dump_textures")]
        dump_passes: bool,
    },
    /// Render a DSL program over its loop to numbered PNG frames
    ///
    /// The program is loaded through the demo host as `render --dsl` loads
    /// it, then frame I renders at the normalized loop time
    /// ((I / FPS) % LOOP_SECONDS) / LOOP_SECONDS, as the page's render loop
    /// advances it, and is written to OUT_DIR/frame_IIIII.png. With --mp4 the
    /// frames are then encoded by ffmpeg (H.264, yuv420p).
    Animate {
        /// The DSL program
        #[arg(long)]
        dsl: PathBuf,
        /// Directory for the frames (created when missing)
        #[arg(long)]
        out_dir: PathBuf,
        /// Frames per second
        #[arg(long, default_value_t = 30.0)]
        fps: f64,
        /// Loop duration in seconds (CanvasRenderer.loopDuration)
        #[arg(long, default_value_t = 10.0)]
        loop_seconds: f64,
        /// Frames to render (default: one loop, FPS x LOOP_SECONDS)
        #[arg(long)]
        frames: Option<u32>,
        /// Square size in pixels
        #[arg(long, default_value_t = 512, conflicts_with_all = ["width", "height"])]
        size: u32,
        /// Width in pixels (with --height)
        #[arg(long, requires = "height")]
        width: Option<u32>,
        /// Height in pixels (with --width)
        #[arg(long, requires = "width")]
        height: Option<u32>,
        /// Set a parameter, STEP_N.NAME=VALUE (repeatable; see `render --help`)
        #[arg(long = "param", value_name = "step_N.NAME=VALUE")]
        params: Vec<String>,
        /// Row order of the frames
        #[arg(long, value_enum, default_value_t = OrientationArg::Presented)]
        orientation: OrientationArg,
        /// Host texture overrides, ID=PNG (repeatable)
        #[arg(long = "host-texture", value_name = "ID=PNG")]
        host_textures: Vec<String>,
        /// The OBJ loaded into mesh0 (default: the .obj next to the DSL)
        #[arg(long)]
        obj: Option<PathBuf>,
        /// A Portable effect definition registered before the program loads
        /// (default: the .portable.json next to the DSL)
        #[arg(long)]
        portable: Option<PathBuf>,
        /// The demo's default media image (default: the demo's test card,
        /// embedded)
        #[arg(long)]
        media: Option<PathBuf>,
        /// Also encode the frames to this mp4 with ffmpeg
        #[arg(long)]
        mp4: Option<PathBuf>,
        /// The ffmpeg executable
        #[arg(long, default_value = "ffmpeg", requires = "mp4")]
        ffmpeg: PathBuf,
    },
    /// List the effects of the catalog
    ///
    /// One line per effect: its id (NAMESPACE/NAME), the DSL call, its kind
    /// (`starter`: begins a chain; `filter`: transforms its input) and its
    /// description.
    Effects {
        /// Only this namespace (synth, filter, mixer, ...)
        #[arg(long)]
        namespace: Option<String>,
        /// Print a JSON array instead
        #[arg(long)]
        json: bool,
    },
    /// Show one effect's parameters: types, defaults, ranges and choices
    Effect {
        /// The effect: NAMESPACE/NAME (synth/noise), NAMESPACE.NAME, or the
        /// bare DSL call name when it is unique (noise)
        id: String,
        /// Print the definition's parameters as JSON instead
        #[arg(long)]
        json: bool,
    },
    /// Dump one frontend stage for each program as JSON lines
    ///
    /// Records are {"program", "stage", "result"|"error"}, the format of
    /// tools/reference-oracle.mjs.
    Dump {
        /// tokens | ast | validated | expanded | graph
        stage: String,
        /// Output file (default: standard output)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Run only this stage, on the previous stage's output read from a
        /// reference JSON-lines dump (checks one stage in isolation)
        #[arg(long)]
        isolated: Option<PathBuf>,
        /// DSL program files
        #[arg(required = true)]
        programs: Vec<PathBuf>,
    },
    /// Render every fixture of a JSON manifest in one process
    ///
    /// The manifest is an array of {"graph"|"dsl", "out", "size", "time",
    /// "frames", "hostTextures": {id: png}, "obj", "portable", "graphOut",
    /// "runSeconds", "sampleEvery", "runFrames", "sampleEveryFrames",
    /// "sampleFrames": [n], "dumpTextures": [id], "dumpPasses", "midi"} (the
    /// `render` options of the same names; "midi" defaults to the DSL's
    /// .midi.json sidecar). Failures are reported per fixture; the
    /// exit status is nonzero if any fixture failed.
    Batch {
        /// The manifest
        manifest: PathBuf,
        /// The demo's default media image (default: the demo's test card,
        /// embedded)
        #[arg(long)]
        media: Option<PathBuf>,
    },
}

/// A command's failure: `Usage` for bad arguments or unreadable input (exit
/// 2), `Failed` for a program that does not compile or render (exit 1; the
/// details are already printed).
enum Failure {
    Usage(String),
    Failed,
}

impl From<String> for Failure {
    fn from(message: String) -> Failure {
        Failure::Usage(message)
    }
}

fn program_name(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.strip_suffix(".dsl").unwrap_or(n).to_owned())
        .unwrap_or_default()
}

fn read_jsonl(path: &Path) -> Result<std::collections::HashMap<String, Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = std::collections::HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let rec = Value::from_json(line).map_err(|e| format!("{}: {e}", path.display()))?;
        let name = rec.get("program").as_str().unwrap_or_default().to_owned();
        out.insert(name, rec);
    }
    Ok(out)
}

fn dump(
    stage: &str,
    out: Option<PathBuf>,
    isolated: Option<PathBuf>,
    programs: Vec<PathBuf>,
) -> Result<(), String> {
    let stage = Stage::from_name(stage).ok_or_else(|| format!("unknown stage '{stage}'"))?;
    let registry = Registry::with_catalog();
    let previous = match &isolated {
        Some(path) => {
            let prev = stage.previous().ok_or_else(|| {
                format!("stage {} has no previous stage to start from", stage.name())
            })?;
            Some((prev, read_jsonl(path)?))
        }
        None => None,
    };
    let mut lines = String::new();
    for path in &programs {
        let name = program_name(path);
        let result = match &previous {
            Some((prev, records)) => match records.get(&name) {
                Some(rec) if rec.as_object().is_some_and(|o| o.contains_key("error")) => {
                    Err(noisemaker_dsl::JsError::Thrown(rec.get("error").clone()))
                }
                Some(rec) => {
                    let src = std::fs::read_to_string(path)
                        .map_err(|e| format!("{}: {e}", path.display()))?;
                    noisemaker_dsl::run_stage_from(
                        stage,
                        *prev,
                        rec.get("result").clone(),
                        &src,
                        &registry,
                    )
                }
                None => Err(noisemaker_dsl::JsError::error(format!(
                    "no {} record for {name}",
                    prev.name()
                ))),
            },
            None => {
                let src = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                noisemaker_dsl::run_stage(stage, &src, &registry)
            }
        };
        let mut rec = noisemaker_dsl::Object::new();
        rec.insert("program", Value::from(name));
        rec.insert("stage", Value::from(stage.name()));
        match result {
            Ok(v) => rec.insert("result", v),
            Err(e) => rec.insert("error", e.to_value()),
        };
        lines.push_str(&Value::Object(rec).to_json().unwrap());
        lines.push('\n');
    }
    match out {
        Some(path) => std::fs::write(&path, lines).map_err(|e| format!("{}: {e}", path.display())),
        None => std::io::stdout()
            .write_all(lines.as_bytes())
            .map_err(|e| e.to_string()),
    }
}

fn device() -> Result<noisemaker_gpu::GpuDevice, String> {
    noisemaker_gpu::GpuDevice::create(&Default::default())
}

fn report_fixture(
    name: &str,
    result: &Result<noisemaker_gpu::protocol::FixtureReport, String>,
) -> bool {
    match result {
        Ok(report) => {
            if report.dropped_bindings > 0 {
                eprintln!(
                    "nm-render: {name}: warning: {} bind-group entries outside the auto layout were dropped",
                    report.dropped_bindings
                );
            }
            for reason in &report.tint_fallbacks {
                eprintln!("nm-render: {name}: warning: compiled with naga, not Tint: {reason}");
            }
            if !report.device_errors.is_empty() {
                for e in &report.device_errors {
                    eprintln!("nm-render: {name}: device error: {e}");
                }
                eprintln!(
                    "nm-render: {name}: FAILED: {} device error(s)",
                    report.device_errors.len()
                );
                return false;
            }
            for out in &report.outputs {
                println!("{name}: wrote {}", out.display());
            }
            true
        }
        Err(e) => {
            eprintln!("nm-render: {name}: FAILED: {e}");
            false
        }
    }
}

/// The shared context with the demo's default media image: `--media`, else
/// the embedded test card.
fn context(media: Option<PathBuf>) -> Result<noisemaker_gpu::protocol::ProtocolContext, String> {
    noisemaker_gpu::protocol::ProtocolContext::new(media.as_deref())
}

fn parse_host_textures(args: Vec<String>) -> Result<Vec<(String, PathBuf)>, String> {
    args.into_iter()
        .map(|h| {
            let (id, path) = h
                .split_once('=')
                .ok_or_else(|| format!("--host-texture expects ID=PNG, got '{h}'"))?;
            Ok((id.to_owned(), PathBuf::from(path)))
        })
        .collect()
}

fn parse_params(args: &[String]) -> Result<Vec<ParamOverride>, String> {
    args.iter()
        .map(|p| ParamOverride::parse(p).map_err(|e| format!("--param: {e}")))
        .collect()
}

/// Compile the DSL program at `path` with the frontend; a program that does
/// not compile is reported with `formatDslError` (syntax errors with their
/// source context) or the demo's compilation-error text.
fn check_dsl(
    path: &Path,
    portable: Option<&Path>,
    registry: &std::rc::Rc<Registry>,
) -> Result<(), Failure> {
    let source = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let registry = noisemaker_gpu::protocol::registry_for_dsl(registry, path, portable)?;
    match noisemaker_dsl::compiler::compile_graph(&source, &registry, &Default::default()) {
        Ok(_) => Ok(()),
        Err(error) => {
            eprintln!(
                "nm-render: {} does not compile:\n{}",
                path.display(),
                noisemaker_dsl::error_formatter::format_compile_error(&source, &error)
            );
            Err(Failure::Failed)
        }
    }
}

struct RenderArgs {
    graph: Option<PathBuf>,
    dsl: Option<PathBuf>,
    out: PathBuf,
    size: u32,
    width: Option<u32>,
    height: Option<u32>,
    time: f64,
    frames: u32,
    params: Vec<String>,
    orientation: OrientationArg,
    host_textures: Vec<String>,
    obj: Option<PathBuf>,
    media: Option<PathBuf>,
    graph_out: Option<PathBuf>,
    portable: Option<PathBuf>,
    run_seconds: Option<f64>,
    sample_every: Option<f64>,
    run_frames: Option<u64>,
    sample_every_frames: Option<u64>,
    sample_frames: Vec<u64>,
    dump_textures: Vec<String>,
    dump_passes: bool,
}

fn render(args: RenderArgs) -> Result<(), Failure> {
    if !args.params.is_empty() && args.dsl.is_none() {
        return Err(Failure::Usage("--param needs a --dsl program".into()));
    }
    let params = parse_params(&args.params)?;
    let host_textures = parse_host_textures(args.host_textures)?;
    let context = context(args.media)?;
    let source = match (args.graph, args.dsl) {
        (Some(g), _) => noisemaker_gpu::protocol::GraphSource::GraphFile(g),
        (None, Some(d)) => {
            check_dsl(&d, args.portable.as_deref(), &context.registry)?;
            noisemaker_gpu::protocol::GraphSource::DslFile(d)
        }
        (None, None) => return Err(Failure::Usage("one of --graph or --dsl is required".into())),
    };
    let spec = noisemaker_gpu::protocol::FixtureSpec {
        source,
        out: args.out.clone(),
        width: args.width.unwrap_or(args.size),
        height: args.height.unwrap_or(args.size),
        time: args.time,
        frames: args.frames,
        host_textures,
        run_seconds: args.run_seconds.unwrap_or(0.0),
        sample_every: args.sample_every.unwrap_or(5.0),
        run_frames: args.run_frames.unwrap_or(0),
        sample_every_frames: args.sample_every_frames.unwrap_or(0),
        sample_frames: args.sample_frames,
        dump_textures: args.dump_textures,
        dump_passes: args.dump_passes,
        obj: args.obj,
        graph_out: args.graph_out,
        params,
        orientation: args.orientation.into(),
        portable: args.portable,
        midi: None,
    };
    let device = device()?;
    let result = noisemaker_gpu::protocol::run_fixture(&device, &spec, &context);
    let name = args
        .out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_owned();
    if report_fixture(&name, &result) {
        Ok(())
    } else {
        Err(Failure::Failed)
    }
}

struct AnimateArgs {
    dsl: PathBuf,
    out_dir: PathBuf,
    fps: f64,
    loop_seconds: f64,
    frames: Option<u32>,
    size: u32,
    width: Option<u32>,
    height: Option<u32>,
    params: Vec<String>,
    orientation: OrientationArg,
    host_textures: Vec<String>,
    obj: Option<PathBuf>,
    portable: Option<PathBuf>,
    media: Option<PathBuf>,
    mp4: Option<PathBuf>,
    ffmpeg: PathBuf,
}

fn animate(args: AnimateArgs) -> Result<(), Failure> {
    use noisemaker_gpu::protocol::{AnimationSpec, frame_path, run_animation};
    if !(args.fps > 0.0 && args.fps.is_finite()) {
        return Err(Failure::Usage(format!(
            "--fps must be positive, got {}",
            args.fps
        )));
    }
    if !(args.loop_seconds > 0.0 && args.loop_seconds.is_finite()) {
        return Err(Failure::Usage(format!(
            "--loop-seconds must be positive, got {}",
            args.loop_seconds
        )));
    }
    let frames = match args.frames {
        Some(n) => n,
        None => (args.fps * args.loop_seconds).round().max(1.0) as u32,
    };
    let params = parse_params(&args.params)?;
    let host_textures = parse_host_textures(args.host_textures)?;
    let context = context(args.media)?;
    check_dsl(&args.dsl, args.portable.as_deref(), &context.registry)?;
    let spec = AnimationSpec {
        dsl: args.dsl.clone(),
        out_dir: args.out_dir.clone(),
        width: args.width.unwrap_or(args.size),
        height: args.height.unwrap_or(args.size),
        fps: args.fps,
        loop_seconds: args.loop_seconds,
        frames,
        params,
        host_textures,
        obj: args.obj,
        orientation: args.orientation.into(),
        portable: args.portable,
    };
    let device = device()?;
    let started = std::time::Instant::now();
    let report = match run_animation(&device, &spec, &context) {
        Ok(report) => report,
        Err(e) => {
            eprintln!("nm-render: {}: FAILED: {e}", args.dsl.display());
            return Err(Failure::Failed);
        }
    };
    if !report.device_errors.is_empty() {
        for e in &report.device_errors {
            eprintln!("nm-render: {}: device error: {e}", args.dsl.display());
        }
        eprintln!(
            "nm-render: {}: FAILED: {} device error(s)",
            args.dsl.display(),
            report.device_errors.len()
        );
        return Err(Failure::Failed);
    }
    println!(
        "wrote {} frames to {} ({:.1}s)",
        report.frames.len(),
        args.out_dir.display(),
        started.elapsed().as_secs_f64()
    );
    let Some(mp4) = args.mp4 else {
        return Ok(());
    };
    // Literal arguments, no shell: the frame pattern is ffmpeg's own.
    let pattern = frame_path(&args.out_dir, 0)
        .with_file_name("frame_%05d.png")
        .into_os_string();
    let status = std::process::Command::new(&args.ffmpeg)
        .arg("-hide_banner")
        .arg("-loglevel")
        .arg("error")
        .arg("-y")
        .arg("-framerate")
        .arg(number_to_string(args.fps))
        .arg("-i")
        .arg(&pattern)
        // yuv420p needs even dimensions.
        .arg("-vf")
        .arg("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        .arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg(&mp4)
        .status();
    match status {
        Ok(s) if s.success() => {
            println!("wrote {}", mp4.display());
            Ok(())
        }
        Ok(s) => {
            eprintln!(
                "nm-render: {} exited with {s}; the frames are in {}",
                args.ffmpeg.display(),
                args.out_dir.display()
            );
            Err(Failure::Failed)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "nm-render: no mp4: {} not found (install ffmpeg or pass --ffmpeg PATH); the frames are in {}",
                args.ffmpeg.display(),
                args.out_dir.display()
            );
            Err(Failure::Failed)
        }
        Err(e) => {
            eprintln!("nm-render: {}: {e}", args.ffmpeg.display());
            Err(Failure::Failed)
        }
    }
}

/// One catalog effect with its definition.
struct CatalogEffect {
    source: &'static noisemaker_effects::EffectSource,
    def: Value,
}

impl CatalogEffect {
    fn func(&self) -> String {
        self.def
            .get("func")
            .as_str()
            .unwrap_or(self.source.name)
            .to_owned()
    }

    fn kind(&self) -> &'static str {
        if is_starter_effect(&self.def) {
            "starter"
        } else {
            "filter"
        }
    }

    fn description(&self) -> String {
        self.def
            .get("description")
            .as_str()
            .unwrap_or_default()
            .to_owned()
    }

    fn summary(&self) -> Value {
        let mut o = Object::new();
        o.insert("id", Value::from(self.source.id()));
        o.insert("namespace", Value::from(self.source.namespace));
        o.insert("name", Value::from(self.source.name));
        o.insert("func", Value::from(self.func()));
        o.insert("kind", Value::from(self.kind()));
        o.insert("description", Value::from(self.description()));
        o.insert("tags", self.def.get("tags").clone());
        Value::Object(o)
    }
}

fn catalog() -> Result<Vec<CatalogEffect>, String> {
    noisemaker_effects::EFFECTS
        .iter()
        .map(|source| {
            Value::from_json(source.definition_json)
                .map(|def| CatalogEffect { source, def })
                .map_err(|e| format!("{}: {e}", source.id()))
        })
        .collect()
}

fn effects(namespace: Option<String>, json: bool) -> Result<(), Failure> {
    let all = catalog()?;
    if let Some(ns) = &namespace
        && !all.iter().any(|e| e.source.namespace == ns)
    {
        let mut namespaces: Vec<&str> = all.iter().map(|e| e.source.namespace).collect();
        namespaces.dedup();
        return Err(Failure::Usage(format!(
            "no namespace '{ns}' (namespaces: {})",
            namespaces.join(", ")
        )));
    }
    let selected: Vec<&CatalogEffect> = all
        .iter()
        .filter(|e| {
            namespace
                .as_deref()
                .is_none_or(|ns| e.source.namespace == ns)
        })
        .collect();
    if json {
        let list = Value::Array(selected.iter().map(|e| e.summary()).collect());
        return print_out(&(list.to_json_pretty().unwrap_or_default() + "\n"));
    }
    let rows: Vec<[String; 4]> = selected
        .iter()
        .map(|e| {
            [
                e.source.id(),
                format!("{}()", e.func()),
                e.kind().to_owned(),
                e.description(),
            ]
        })
        .collect();
    print_out(&table(&["EFFECT", "CALL", "KIND", "DESCRIPTION"], &rows))
}

/// Write `text` to standard output; a closed pipe (`nm-render effects | head`)
/// ends the output quietly.
fn print_out(text: &str) -> Result<(), Failure> {
    match std::io::stdout().lock().write_all(text.as_bytes()) {
        Err(e) if e.kind() != std::io::ErrorKind::BrokenPipe => {
            Err(Failure::Usage(format!("standard output: {e}")))
        }
        _ => Ok(()),
    }
}

/// Rows under a header, columns padded to their widths (the last column
/// unpadded).
fn table<const N: usize>(header: &[&str; N], rows: &[[String; N]]) -> String {
    let mut widths = header.map(|h| h.chars().count());
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let mut out = String::new();
    let mut line = |cells: &[&str]| {
        let mut text = String::new();
        for (i, cell) in cells.iter().enumerate() {
            if i + 1 == cells.len() {
                text.push_str(cell);
            } else {
                text.push_str(&format!("{cell:<width$}  ", width = widths[i]));
            }
        }
        out.push_str(text.trim_end());
        out.push('\n');
    };
    line(header);
    for row in rows {
        line(&row.each_ref().map(String::as_str));
    }
    out
}

/// The catalog effect `query` names: `ns/name`, `ns.name`, `ns.func`, or a
/// unique bare name or call.
fn find_effect<'a>(all: &'a [CatalogEffect], query: &str) -> Result<&'a CatalogEffect, String> {
    let qualified: Vec<&CatalogEffect> = all
        .iter()
        .filter(|e| {
            let ns = e.source.namespace;
            query == e.source.id()
                || query == format!("{ns}.{}", e.source.name)
                || query == format!("{ns}.{}", e.func())
        })
        .collect();
    if let [one] = qualified.as_slice() {
        return Ok(one);
    }
    let bare: Vec<&CatalogEffect> = all
        .iter()
        .filter(|e| query == e.source.name || query == e.func())
        .collect();
    match bare.as_slice() {
        [one] => Ok(one),
        [] => {
            let lower = query.to_lowercase();
            let similar: Vec<String> = all
                .iter()
                .filter(|e| {
                    e.source.name.to_lowercase().contains(&lower)
                        || e.func().to_lowercase().contains(&lower)
                })
                .map(|e| e.source.id())
                .take(8)
                .collect();
            if similar.is_empty() {
                Err(format!(
                    "no effect '{query}' (see `nm-render effects` for the catalog)"
                ))
            } else {
                Err(format!(
                    "no effect '{query}'; similar: {}",
                    similar.join(", ")
                ))
            }
        }
        many => Err(format!(
            "'{query}' is ambiguous: {}; use NAMESPACE/NAME",
            many.iter()
                .map(|e| e.source.id())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// A value as DSL-literal text (JSON for strings, arrays and objects).
fn value_text(v: &Value) -> String {
    match v {
        Value::Number(n) => number_to_string(*n),
        Value::Undefined => String::new(),
        other => other.to_json().unwrap_or_default(),
    }
}

fn effect(query: &str, json: bool) -> Result<(), Failure> {
    let all = catalog()?;
    let e = find_effect(&all, query)?;
    let globals = e
        .def
        .get("globals")
        .as_object()
        .cloned()
        .unwrap_or_default();
    if json {
        let mut o = e.summary().as_object().cloned().unwrap_or_default();
        o.insert("parameters", Value::Object(globals));
        return print_out(&(Value::Object(o).to_json_pretty().unwrap_or_default() + "\n"));
    }
    let mut out = format!("{}  {}()  {}\n", e.source.id(), e.func(), e.kind());
    let description = e.description();
    if !description.is_empty() {
        out.push_str(&description);
        out.push('\n');
    }
    if let Some(tags) = e.def.get("tags").as_array().filter(|t| !t.is_empty()) {
        let tags: Vec<&str> = tags.iter().filter_map(Value::as_str).collect();
        out.push_str(&format!("tags: {}\n", tags.join(", ")));
    }
    out.push('\n');
    if globals.is_empty() {
        out.push_str("no parameters\n");
        return print_out(&out);
    }
    let mut rows: Vec<[String; 5]> = Vec::new();
    for (name, spec) in globals.iter() {
        let get = |k: &str| spec.get(k);
        let range = match (get("min"), get("max")) {
            (Value::Undefined, Value::Undefined) => String::new(),
            (min, max) => format!("{}..{}", value_text(min), value_text(max)),
        };
        let mut notes: Vec<String> = Vec::new();
        let step = get("step");
        if !step.is_undefined() {
            notes.push(format!("step {}", value_text(step)));
        }
        if let Some(choices) = get("choices").as_object() {
            // `null` entries are the dropdown's group headings.
            let list: Vec<String> = choices
                .iter()
                .filter(|(_, v)| !v.is_nullish())
                .map(|(k, v)| format!("{k}={}", value_text(v)))
                .collect();
            notes.push(format!("choices: {}", list.join(", ")));
        }
        let enum_path = get("enum");
        if !enum_path.is_undefined() {
            notes.push(format!("enum {}", value_text(enum_path).replace('"', "")));
        }
        let define = get("define");
        if define.is_truthy() {
            notes.push(format!(
                "compile-time ({})",
                value_text(define).replace('"', "")
            ));
        }
        let ui = get("ui");
        if ui.get("hidden") == &Value::Bool(true) {
            notes.push("hidden".into());
        }
        if let Some(label) = ui.get("label").as_str()
            && !label.eq_ignore_ascii_case(name)
        {
            notes.push(format!("label \"{label}\""));
        }
        rows.push([
            name.clone(),
            value_text(get("type")).replace('"', ""),
            value_text(get("default")),
            range,
            notes.join("; "),
        ]);
    }
    out.push_str(&table(
        &["PARAMETER", "TYPE", "DEFAULT", "RANGE", "NOTES"],
        &rows,
    ));
    out.push_str(
        "\nSet one with: nm-render render --dsl PROGRAM.dsl --param step_N.NAME=VALUE (N: the step's index in the program)\n",
    );
    print_out(&out)
}

fn manifest_spec(entry: &Value) -> Result<(String, noisemaker_gpu::protocol::FixtureSpec), String> {
    use noisemaker_gpu::protocol::{FixtureSpec, GraphSource};
    let out = entry
        .get("out")
        .as_str()
        .ok_or("manifest entry has no \"out\"")?
        .to_owned();
    let source = if let Some(g) = entry.get("graph").as_str() {
        GraphSource::GraphFile(PathBuf::from(g))
    } else if let Some(d) = entry.get("dsl").as_str() {
        GraphSource::DslFile(PathBuf::from(d))
    } else {
        return Err(format!("{out}: manifest entry needs \"graph\" or \"dsl\""));
    };
    let name = match &source {
        GraphSource::GraphFile(p) | GraphSource::DslFile(p) => {
            let n = program_name(p);
            n.strip_suffix(".graph.json")
                .map(str::to_owned)
                .unwrap_or(n)
        }
        GraphSource::Graph(_) => out.clone(),
    };
    let num = |key: &str, default: f64| entry.get(key).as_f64().unwrap_or(default);
    let path = |key: &str| entry.get(key).as_str().map(PathBuf::from);
    let size = num("size", 256.0);
    let frame_list = |key: &str| -> Result<Vec<u64>, String> {
        match entry.get(key) {
            Value::Undefined => Ok(Vec::new()),
            Value::Array(items) => items
                .iter()
                .map(|v| {
                    v.as_f64()
                        .filter(|n| *n >= 1.0 && n.fract() == 0.0)
                        .map(|n| n as u64)
                        .ok_or_else(|| format!("{name}: {key} must hold frame counts"))
                })
                .collect(),
            _ => Err(format!("{name}: {key} must be an array")),
        }
    };
    let sample_frames = frame_list("sampleFrames")?;
    let dump_textures = match entry.get("dumpTextures") {
        Value::Undefined => Vec::new(),
        Value::Array(items) => items
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("{name}: dumpTextures must hold texture ids"))
            })
            .collect::<Result<_, _>>()?,
        _ => return Err(format!("{name}: dumpTextures must be an array")),
    };
    let mut hosts = Vec::new();
    if let Some(map) = entry.get("hostTextures").as_object() {
        for (id, path) in map.iter() {
            let path = path
                .as_str()
                .ok_or_else(|| format!("{name}: hostTextures.{id} must be a path"))?;
            hosts.push((id.clone(), PathBuf::from(path)));
        }
    }
    Ok((
        name,
        FixtureSpec {
            source,
            out: PathBuf::from(out),
            width: num("width", size) as u32,
            height: num("height", size) as u32,
            time: num("time", 0.25),
            frames: num("frames", 8.0) as u32,
            host_textures: hosts,
            run_seconds: num("runSeconds", 0.0),
            sample_every: num("sampleEvery", 5.0),
            run_frames: num("runFrames", 0.0) as u64,
            sample_every_frames: num("sampleEveryFrames", 0.0) as u64,
            sample_frames,
            dump_textures,
            dump_passes: entry.get("dumpPasses").is_truthy(),
            obj: path("obj"),
            portable: path("portable"),
            midi: path("midi"),
            graph_out: path("graphOut"),
            ..FixtureSpec::default()
        },
    ))
}

fn batch(manifest: PathBuf, media: Option<PathBuf>) -> Result<(), Failure> {
    let text =
        std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let entries = Value::from_json(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let entries = entries
        .as_array()
        .ok_or_else(|| format!("{}: the manifest must be a JSON array", manifest.display()))?
        .clone();
    let context = context(media)?;
    let device = device()?;
    let (mut ok, mut failed) = (0usize, 0usize);
    let started = std::time::Instant::now();
    for (i, entry) in entries.iter().enumerate() {
        let (name, spec) = match manifest_spec(entry) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("nm-render: entry {i}: FAILED: {e}");
                failed += 1;
                continue;
            }
        };
        let t0 = std::time::Instant::now();
        let result = noisemaker_gpu::protocol::run_fixture(&device, &spec, &context);
        if report_fixture(&name, &result) {
            ok += 1;
        } else {
            failed += 1;
        }
        eprintln!(
            "nm-render: {name}: {:.0} ms",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
    eprintln!(
        "nm-render batch: ok={ok} failed={failed} total={} ({:.1}s)",
        entries.len(),
        started.elapsed().as_secs_f64()
    );
    if failed == 0 {
        Ok(())
    } else {
        Err(Failure::Failed)
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Dump {
            stage,
            out,
            isolated,
            programs,
        } => dump(&stage, out, isolated, programs).map_err(Failure::Usage),
        Command::Render {
            graph,
            dsl,
            out,
            size,
            width,
            height,
            time,
            frames,
            params,
            orientation,
            host_textures,
            obj,
            media,
            graph_out,
            portable,
            run_seconds,
            sample_every,
            run_frames,
            sample_every_frames,
            sample_frames,
            dump_textures,
            dump_passes,
        } => render(RenderArgs {
            graph,
            dsl,
            out,
            size,
            width,
            height,
            time,
            frames,
            params,
            orientation,
            host_textures,
            obj,
            media,
            graph_out,
            portable,
            run_seconds,
            sample_every,
            run_frames,
            sample_every_frames,
            sample_frames,
            dump_textures,
            dump_passes,
        }),
        Command::Animate {
            dsl,
            out_dir,
            fps,
            loop_seconds,
            frames,
            size,
            width,
            height,
            params,
            orientation,
            host_textures,
            obj,
            portable,
            media,
            mp4,
            ffmpeg,
        } => animate(AnimateArgs {
            dsl,
            out_dir,
            fps,
            loop_seconds,
            frames,
            size,
            width,
            height,
            params,
            orientation,
            host_textures,
            obj,
            portable,
            media,
            mp4,
            ffmpeg,
        }),
        Command::Effects { namespace, json } => effects(namespace, json),
        Command::Effect { id, json } => effect(&id, json),
        Command::Batch { manifest, media } => batch(manifest, media),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Failed) => ExitCode::from(1),
        Err(Failure::Usage(e)) => {
            eprintln!("nm-render: {e}");
            ExitCode::from(2)
        }
    }
}
