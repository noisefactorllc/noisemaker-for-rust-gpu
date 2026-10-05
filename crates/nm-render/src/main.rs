//! nm-render — render Noisemaker DSL programs on the GPU and dump the frontend's
//! stages for the parity gates.
//!
//! `render` renders one fixture with the golden protocol of
//! `parity/batch-golden.mjs`; `batch` renders a manifest of fixtures in one
//! process and device (a fresh pipeline per fixture).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use noisemaker_dsl::{Registry, Stage, Value};

#[derive(Parser)]
#[command(
    name = "nm-render",
    version,
    about = "Render Noisemaker DSL programs on the GPU"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Dump one frontend stage for each program as JSON lines
    /// ({"program", "stage", "result"|"error"}), the format of
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
    /// Render one program with the golden protocol: a fresh pipeline at the
    /// size, host textures uploaded, `render(time)` FRAMES times, then the render
    /// surface read back and written as an RGBA8 PNG (top row first).
    Render {
        /// A reference graph (`<name>.graph.json` from parity/batch-golden.mjs)
        #[arg(long, conflicts_with = "dsl", required_unless_present = "dsl")]
        graph: Option<PathBuf>,
        /// A DSL program compiled by the Rust frontend
        #[arg(long)]
        dsl: Option<PathBuf>,
        /// Output PNG
        #[arg(long)]
        out: PathBuf,
        /// Square size in pixels
        #[arg(long, default_value_t = 256, conflicts_with_all = ["width", "height"])]
        size: u32,
        #[arg(long, requires = "height")]
        width: Option<u32>,
        #[arg(long, requires = "width")]
        height: Option<u32>,
        /// Normalized loop time of every frame
        #[arg(long, default_value_t = 0.25)]
        time: f64,
        /// Frames to render
        #[arg(long, default_value_t = 8)]
        frames: u32,
        /// Host texture uploads, ID=PNG (repeatable)
        #[arg(long = "host-texture", value_name = "ID=PNG")]
        host_textures: Vec<String>,
        /// Timed mode: seconds to run, stepping render(((frame + 1) / 600) % 1)
        #[arg(long, requires = "sample_every")]
        run_seconds: Option<f64>,
        /// Timed mode: seconds between samples (<out stem>.t<sec>.png)
        #[arg(long)]
        sample_every: Option<f64>,
    },
    /// Render every fixture of a JSON manifest in one process: an array of
    /// {"graph"|"dsl", "out", "size", "time", "frames", "hostTextures": {id: png},
    /// "runSeconds", "sampleEvery"}. Failures are reported per fixture; the exit
    /// status is nonzero if any fixture failed.
    Batch {
        /// The manifest
        manifest: PathBuf,
    },
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

#[allow(clippy::too_many_arguments)]
fn render(
    graph: Option<PathBuf>,
    dsl: Option<PathBuf>,
    out: PathBuf,
    size: u32,
    width: Option<u32>,
    height: Option<u32>,
    time: f64,
    frames: u32,
    host_textures: Vec<String>,
    run_seconds: Option<f64>,
    sample_every: Option<f64>,
) -> Result<bool, String> {
    let source = match (graph, dsl) {
        (Some(g), _) => noisemaker_gpu::protocol::GraphSource::GraphFile(g),
        (None, Some(d)) => noisemaker_gpu::protocol::GraphSource::DslFile(d),
        (None, None) => return Err("one of --graph or --dsl is required".into()),
    };
    let mut hosts = Vec::new();
    for h in host_textures {
        let (id, path) = h
            .split_once('=')
            .ok_or_else(|| format!("--host-texture expects ID=PNG, got '{h}'"))?;
        hosts.push((id.to_owned(), PathBuf::from(path)));
    }
    let spec = noisemaker_gpu::protocol::FixtureSpec {
        source,
        out: out.clone(),
        width: width.unwrap_or(size),
        height: height.unwrap_or(size),
        time,
        frames,
        host_textures: hosts,
        run_seconds: run_seconds.unwrap_or(0.0),
        sample_every: sample_every.unwrap_or(5.0),
    };
    let device = device()?;
    let result = noisemaker_gpu::protocol::run_fixture(&device, &spec, None);
    let name = out
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
        .to_owned();
    Ok(report_fixture(&name, &result))
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
    let size = num("size", 256.0);
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
        },
    ))
}

fn batch(manifest: PathBuf) -> Result<bool, String> {
    let text =
        std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let entries = Value::from_json(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let entries = entries
        .as_array()
        .ok_or_else(|| format!("{}: the manifest must be a JSON array", manifest.display()))?
        .clone();
    let device = device()?;
    let mut registry: Option<Registry> = None;
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
        if matches!(
            spec.source,
            noisemaker_gpu::protocol::GraphSource::DslFile(_)
        ) && registry.is_none()
        {
            registry = Some(Registry::with_catalog());
        }
        let t0 = std::time::Instant::now();
        let result = noisemaker_gpu::protocol::run_fixture(&device, &spec, registry.as_ref());
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
    Ok(failed == 0)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Dump {
            stage,
            out,
            isolated,
            programs,
        } => dump(&stage, out, isolated, programs).map(|()| true),
        Command::Render {
            graph,
            dsl,
            out,
            size,
            width,
            height,
            time,
            frames,
            host_textures,
            run_seconds,
            sample_every,
        } => render(
            graph,
            dsl,
            out,
            size,
            width,
            height,
            time,
            frames,
            host_textures,
            run_seconds,
            sample_every,
        ),
        Command::Batch { manifest } => batch(manifest),
    };
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("nm-render: {e}");
            ExitCode::from(2)
        }
    }
}
