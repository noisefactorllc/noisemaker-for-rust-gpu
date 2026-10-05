//! The golden protocol of `parity/batch-golden.mjs`, for candidate renders.
//!
//! A fixture renders on a fresh pipeline (every pipeline-written texture zero,
//! surfaces in read/write orientation, frame index and clock zero, no global
//! uniforms) sized `width x height`, with the host textures the minter saved
//! uploaded through `updateTextureFromSource` (`flipY: false`: the saved PNGs are
//! texture contents). It then renders `frames` frames at the pinned normalized
//! `time` and reads back the render surface's read texture with `readPixels`, or,
//! in timed mode, steps `render(((frame + 1) / 600) % 1)` and saves a sample every
//! `sample_every` seconds (60 frames per second).

use std::path::{Path, PathBuf};

use noisemaker_dsl::Value;

use crate::backend::GpuDevice;
use crate::graph::Graph;
use crate::png_io::{read_png_rgba8, write_png_rgba8};
use crate::{Renderer, RendererOptions};

/// Where a fixture's graph comes from.
#[derive(Debug, Clone)]
pub enum GraphSource {
    /// A reference graph (`<name>.graph.json`).
    GraphFile(PathBuf),
    /// DSL source compiled by the Rust frontend.
    DslFile(PathBuf),
    /// An already-built graph.
    Graph(Box<Graph>),
}

/// One fixture to render.
#[derive(Debug, Clone)]
pub struct FixtureSpec {
    pub source: GraphSource,
    /// The output PNG (timed samples go to `<out stem>.t<sec>.png`).
    pub out: PathBuf,
    pub width: u32,
    pub height: u32,
    pub time: f64,
    pub frames: u32,
    /// `(textureId, png path)` host inputs.
    pub host_textures: Vec<(String, PathBuf)>,
    /// Timed mode: total seconds (0 disables).
    pub run_seconds: f64,
    /// Timed mode: seconds between samples.
    pub sample_every: f64,
}

impl Default for FixtureSpec {
    fn default() -> Self {
        FixtureSpec {
            source: GraphSource::Graph(Box::default()),
            out: PathBuf::new(),
            width: 256,
            height: 256,
            time: 0.25,
            frames: 8,
            host_textures: Vec::new(),
            run_seconds: 0.0,
            sample_every: 5.0,
        }
    }
}

/// What a fixture run produced.
#[derive(Debug, Clone, Default)]
pub struct FixtureReport {
    /// The PNGs written.
    pub outputs: Vec<PathBuf>,
    /// Device (validation) errors observed while rendering.
    pub device_errors: Vec<String>,
    /// Bind-group entries dropped because the auto layout lacks their binding.
    pub dropped_bindings: usize,
    /// Pipeline and backend diagnostics.
    pub diagnostics: Vec<Value>,
}

/// Compile DSL source with the Rust frontend into a graph.
pub fn compile_dsl(source: &str, registry: &noisemaker_dsl::Registry) -> Result<Graph, String> {
    let value = noisemaker_dsl::compiler::compile_graph(
        source,
        registry,
        &noisemaker_dsl::compiler::CompileOptions::default(),
    )
    .map_err(|e| format!("DSL compile failed: {e}"))?;
    Graph::from_value(&value)
}

/// `<out without .png>.t<sec>.png`.
pub fn timed_sample_path(out: &Path, seconds: f64) -> PathBuf {
    let text = out.to_string_lossy();
    let stem = text.strip_suffix(".png").unwrap_or(&text);
    PathBuf::from(format!(
        "{stem}.t{}.png",
        noisemaker_dsl::js::number_to_string(seconds)
    ))
}

/// Render one fixture on `device`.
pub fn run_fixture(
    device: &GpuDevice,
    spec: &FixtureSpec,
    registry: Option<&noisemaker_dsl::Registry>,
) -> Result<FixtureReport, String> {
    let graph = match &spec.source {
        GraphSource::GraphFile(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Graph::from_reference_json(&text).map_err(|e| format!("{}: {e}", path.display()))?
        }
        GraphSource::DslFile(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let owned;
            let registry = match registry {
                Some(r) => r,
                None => {
                    owned = noisemaker_dsl::Registry::with_catalog();
                    &owned
                }
            };
            compile_dsl(&text, registry)?
        }
        GraphSource::Graph(graph) => (**graph).clone(),
    };

    let mut renderer = Renderer::new(
        device,
        graph,
        spec.width,
        spec.height,
        RendererOptions::default(),
    )
    .map_err(|e| format!("pipeline init failed: {e}"))?;
    for (id, path) in &spec.host_textures {
        let image = read_png_rgba8(path)?;
        renderer
            .update_texture_from_rgba8(id, image.width, image.height, &image.data, false)
            .map_err(|e| format!("host texture {id}: {e}"))?;
    }

    let mut report = FixtureReport::default();
    let result = (|| -> Result<(), String> {
        if spec.run_seconds > 0.0 {
            let every_frames = (spec.sample_every * 60.0).round().max(1.0) as u64;
            let samples = ((spec.run_seconds * 60.0) / every_frames as f64)
                .floor()
                .max(1.0) as u64;
            for s in 0..samples {
                let start = s * every_frames;
                for i in 0..every_frames {
                    let t = ((start + i + 1) as f64 / 600.0) % 1.0;
                    renderer
                        .render(t)
                        .map_err(|e| format!("render failed: {e}"))?;
                }
                let seconds = (s + 1) as f64 * spec.sample_every;
                let pixels = renderer
                    .read_output()
                    .map_err(|e| format!("readback failed: {e}"))?;
                let path = timed_sample_path(&spec.out, seconds);
                write_png_rgba8(&path, pixels.width, pixels.height, &pixels.data)?;
                report.outputs.push(path);
            }
        } else {
            for _ in 0..spec.frames {
                renderer
                    .render(spec.time)
                    .map_err(|e| format!("render failed: {e}"))?;
            }
            let pixels = renderer
                .read_output()
                .map_err(|e| format!("readback failed: {e}"))?;
            write_png_rgba8(&spec.out, pixels.width, pixels.height, &pixels.data)?;
            report.outputs.push(spec.out.clone());
        }
        Ok(())
    })();

    let pipeline = renderer.pipeline_mut();
    pipeline.backend.collect_device_errors();
    report.device_errors = pipeline.backend.device_error_log.clone();
    report.dropped_bindings = pipeline.backend.dropped_binding_count;
    report.diagnostics = pipeline
        .diagnostics
        .records
        .iter()
        .chain(pipeline.backend.diagnostics.records.iter())
        .cloned()
        .collect();
    let _ = renderer.dispose();
    result?;
    Ok(report)
}
