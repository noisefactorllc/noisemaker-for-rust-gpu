//! The golden protocol of `parity/batch-golden.mjs`, for candidate renders.
//!
//! The minter renders each fixture in the reference demo page: it loads the
//! DSL through the page (compile, ProgramState, the controls' parameter
//! writes, `applyStepParameterValues`), loads the meshes a mesh fixture needs
//! (each `externalMesh` step's first built-in mesh, then the fixture's `.obj`
//! sidecar into `mesh0`), waits for the host inputs (media, text) and the
//! asyncInit overlays, resets the pipeline to a fresh state (every
//! pipeline-written texture cleared, surfaces back to read/write orientation,
//! frame index and clock zeroed, global uniforms emptied), renders `frames`
//! frames at the pinned normalized `time` and reads back the render surface's
//! read texture. Timed fixtures step `render(((frame + 1) / 600) % 1)` and
//! save a sample every `sample_every` seconds (60 frames per second).
//!
//! A DSL fixture ([`GraphSource::DslFile`]) runs that whole protocol here:
//! [`DemoHost`] is the page, and the host inputs are produced natively (the
//! demo's default media image, text canvases, overlays, meshes). A graph
//! fixture ([`GraphSource::GraphFile`]: the graph the page rendered, saved by
//! the minter) renders on a fresh pipeline with the meshes loaded the same way
//! and the host textures given as PNGs. Either way, `host_textures` replace
//! textures with PNGs (texture contents, uploaded with `flipY: false`) — the
//! minter's captures, to grade with exactly the reference's host inputs.
//!
//! Outside the parity protocol, a DSL fixture can also take parameter
//! overrides ([`ParamOverride`], set as the demo's controls set them) and be
//! written in the orientation a canvas shows ([`Orientation::Presented`]);
//! [`run_animation`] renders a DSL program over its loop to a PNG sequence.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use noisemaker_dsl::{Registry, Value};
use noisemaker_host::obj::{PackedMesh, pack_mesh, parse_obj};
use noisemaker_host::text::TextFonts;

use crate::backend::GpuDevice;
use crate::demo::{DEFAULT_MEDIA_PATH, DemoHost, DemoHostOptions};
use crate::graph::{Graph, pass};
use crate::host::{CanvasRenderer, CanvasRendererOptions};
use crate::jsre::JsRegex;
use crate::jsv::to_js_string;
use crate::pipeline::Pipeline;
use crate::png_io::{Rgba8Image, read_png_rgba8, write_png_rgba8};
use crate::present::Orientation;
use crate::{Renderer, RendererOptions};

/// A parameter override of a DSL fixture: the value a control sets on one
/// step's parameter, applied through [`DemoHost::set_control_value`]
/// (ProgramState's `setValue`, which validates and coerces it, then the
/// page's control-change handling).
#[derive(Debug, Clone, PartialEq)]
pub struct ParamOverride {
    /// The step key (`step_<N>`, the step's index in the program).
    pub step: String,
    /// The parameter name (a key of the effect's `globals`).
    pub name: String,
    /// The value.
    pub value: Value,
}

impl ParamOverride {
    /// Parse `step_N.name=value`. The value is read as JSON when it is JSON
    /// (`2.5`, `true`, `[1, 0, 0, 1]`, `"3"`) and as a string otherwise
    /// (`hello world`, `noise.simplex`).
    pub fn parse(text: &str) -> Result<ParamOverride, String> {
        let (target, value) = text
            .split_once('=')
            .ok_or_else(|| format!("expected step_N.name=value, got '{text}'"))?;
        let (step, name) = target
            .split_once('.')
            .filter(|(step, name)| {
                !name.is_empty()
                    && step
                        .strip_prefix("step_")
                        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
            })
            .ok_or_else(|| format!("expected step_N.name=value, got '{text}'"))?;
        let value = Value::from_json(value).unwrap_or_else(|_| Value::from(value));
        Ok(ParamOverride {
            step: step.to_owned(),
            name: name.to_owned(),
            value,
        })
    }
}

impl std::str::FromStr for ParamOverride {
    type Err = String;

    fn from_str(text: &str) -> Result<ParamOverride, String> {
        ParamOverride::parse(text)
    }
}

/// Where a fixture's graph comes from.
#[derive(Debug, Clone)]
pub enum GraphSource {
    /// A reference graph (`<name>.graph.json`).
    GraphFile(PathBuf),
    /// DSL source, run through the demo host.
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
    /// `(textureId, png path)` host texture overrides.
    pub host_textures: Vec<(String, PathBuf)>,
    /// Timed mode: total seconds (0 disables).
    pub run_seconds: f64,
    /// Timed mode: seconds between samples.
    pub sample_every: f64,
    /// The fixture's OBJ sidecar, loaded into `mesh0`; `None` for a DSL file
    /// means `<dsl without .dsl>.obj` when it exists.
    pub obj: Option<PathBuf>,
    /// Write the graph the fixture rendered (JSON, as the minter writes
    /// `<name>.graph.json`) here.
    pub graph_out: Option<PathBuf>,
    /// Parameter overrides, applied after the DSL loads (DSL fixtures only).
    pub params: Vec<ParamOverride>,
    /// The row order of the written PNGs (the protocol's is
    /// [`Orientation::Texture`]).
    pub orientation: Orientation,
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
            obj: None,
            graph_out: None,
            params: Vec::new(),
            orientation: Orientation::Texture,
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
    /// The host textures kept through the fresh-state reset.
    pub host_textures: Vec<String>,
}

/// What fixtures of one run share: the DSL registries, the text fonts and the
/// demo's default media image.
pub struct ProtocolContext {
    pub registry: Rc<Registry>,
    pub fonts: Rc<TextFonts>,
    /// The demo's default media image (`img/testcard.png`); `None` renders
    /// media steps without media, as the page does when the image fails to
    /// load.
    pub default_media: Option<Rc<Rgba8Image>>,
}

impl ProtocolContext {
    /// A context with the embedded catalog, the default fonts and the media
    /// image at `media` (when given).
    pub fn new(media: Option<&Path>) -> Result<ProtocolContext, String> {
        let default_media = match media {
            Some(path) => Some(Rc::new(read_png_rgba8(path)?)),
            None => None,
        };
        Ok(ProtocolContext {
            registry: Rc::new(Registry::with_catalog()),
            fonts: Rc::new(TextFonts::default()),
            default_media,
        })
    }

    /// The demo's default media image in a reference checkout.
    pub fn reference_media_path(reference_root: &Path) -> PathBuf {
        reference_root.join(DEFAULT_MEDIA_PATH)
    }
}

/// Compile DSL source with the Rust frontend into a graph.
pub fn compile_dsl(source: &str, registry: &Registry) -> Result<Graph, String> {
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

/// Host-supplied texture ids (media and text steps).
const EXTERNAL_TEXTURE_ID: &str = r"^[A-Za-z][A-Za-z0-9]*_step_\d+$";
/// asyncInit overlay ids.
const ASYNC_OVERLAY_ID: &str = r"^node_\d+_[A-Za-z][A-Za-z0-9]*$";
/// Mesh texture inputs.
const MESH_TEXTURE_INPUT: &str = r"^global_mesh\d+_(positions|normals|uvs)";

/// `externalIds`: the host-supplied texture ids the passes read.
pub fn external_texture_ids(graph: &Graph) -> Vec<String> {
    let re = JsRegex::new(EXTERNAL_TEXTURE_ID, "");
    let mut ids = Vec::new();
    for p in &graph.passes {
        if let Some(inputs) = pass::inputs(p) {
            for id in inputs.values() {
                if let Some(id) = id.as_str()
                    && re.test(id)
                    && !ids.iter().any(|i| i == id)
                {
                    ids.push(id.to_owned());
                }
            }
        }
    }
    ids
}

/// `asyncOverlayIds()`: inputs no pass writes that exist as textures and
/// belong to a node whose asyncInit started.
pub fn async_overlay_ids(pipeline: &Pipeline) -> Vec<String> {
    let mut nodes: Vec<String> = Vec::new();
    for p in &pipeline.graph.passes {
        let key = pass::get(p, "effectKey");
        let node = pass::get(p, "nodeId");
        if key.is_truthy()
            && node.is_truthy()
            && pipeline
                .effects()
                .get(&to_js_string(key))
                .is_some_and(|e| e.async_init.is_some())
        {
            let node = to_js_string(node);
            if !nodes.contains(&node) {
                nodes.push(node);
            }
        }
    }
    let written: HashSet<String> = pipeline
        .graph
        .passes
        .iter()
        .filter_map(pass::outputs)
        .flat_map(|o| o.values().map(to_js_string))
        .collect();
    let re = JsRegex::new(ASYNC_OVERLAY_ID, "");
    let mut found: Vec<String> = Vec::new();
    for p in &pipeline.graph.passes {
        let Some(inputs) = pass::inputs(p) else {
            continue;
        };
        for id in inputs.values() {
            let id = to_js_string(id);
            if written.contains(&id) || !pipeline.backend.textures.contains_key(&id) {
                continue;
            }
            for node in &nodes {
                if id.starts_with(&format!("{node}_")) && !found.contains(&id) {
                    found.push(id.clone());
                }
            }
        }
    }
    found.retain(|id| re.test(id));
    found
}

/// `meshPlan(graph, dslPath)` of the minter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshPlan {
    /// The first built-in mesh of each `externalMesh` step: `(meshId, catalog
    /// path)`.
    pub builtins: Vec<(String, String)>,
    /// The fixture's OBJ sidecar text (read as Node reads it: UTF-8).
    pub obj_text: Option<String>,
}

/// `meshPlan(graph, dslPath)`: `None` when the graph reads no mesh and there
/// is no sidecar.
pub fn mesh_plan(graph: &Graph, obj: Option<&Path>) -> Option<MeshPlan> {
    let obj_text = obj
        .filter(|p| p.exists())
        .and_then(|p| std::fs::read(p).ok())
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned());
    let mesh_input = JsRegex::new(MESH_TEXTURE_INPUT, "");
    let reads_mesh = graph.passes.iter().any(|p| {
        pass::inputs(p).is_some_and(|inputs| {
            inputs
                .values()
                .any(|id| id.as_str().is_some_and(|id| mesh_input.test(id)))
        })
    });
    if !reads_mesh && obj_text.is_none() {
        return None;
    }
    let mut builtins = Vec::new();
    let mut seen = HashSet::new();
    for p in &graph.passes {
        let key = pass::get(p, "effectKey");
        if !key.is_truthy() {
            continue;
        }
        let key = to_js_string(key);
        if !seen.insert(format!("{}|{key}", to_js_string(pass::get(p, "stepIndex")))) {
            continue;
        }
        let Some((ns, func)) = key.split_once('.') else {
            continue;
        };
        let Some(def) = crate::effects::catalog_definition(ns, func) else {
            continue;
        };
        let external = def.get("externalMesh");
        let builtin = def.get("builtinMeshes");
        if !external.is_truthy() || !builtin.is_truthy() {
            continue;
        }
        if let Some((_, first)) = builtin.as_object().and_then(|b| b.iter().next())
            && first.is_truthy()
        {
            builtins.push((to_js_string(external), to_js_string(first)));
        }
    }
    Some(MeshPlan { builtins, obj_text })
}

impl MeshPlan {
    /// The packed meshes `applyMeshPlan` loads, in load order: the built-ins,
    /// then the sidecar into `mesh0` (an empty mesh when there is neither).
    pub fn loads(&self) -> Result<Vec<(String, PackedMesh)>, String> {
        let mut loads = Vec::new();
        for (mesh_id, path) in &self.builtins {
            let mesh = noisemaker_host::obj::builtin_mesh(path)
                .ok_or_else(|| format!("mesh load failed: {path} is not a catalog mesh"))?;
            loads.push((mesh_id.clone(), pack_mesh(&mesh)));
        }
        match &self.obj_text {
            Some(text) => loads.push(("mesh0".to_owned(), pack_mesh(&parse_obj(text)))),
            None if self.builtins.is_empty() => {
                loads.push(("mesh0".to_owned(), pack_mesh(&parse_obj(""))))
            }
            None => {}
        }
        Ok(loads)
    }
}

/// The fresh-state reset: clear every texture except the kept host inputs
/// (`keep`, mesh data, the MIDI note grid, external textures) and the 3D and
/// cube textures; put double-buffered surfaces back in read/write
/// orientation; zero the frame index and clock; empty the global uniforms.
pub fn reset_fresh_state(pipeline: &mut Pipeline, keep: &HashSet<String>) {
    let mesh = JsRegex::new(r"^global_mesh\d+_(positions|normals|uvs)$", "");
    let ids: Vec<String> = pipeline.backend.textures.keys().cloned().collect();
    for id in ids {
        let Some(tex) = pipeline.backend.textures.get(&id).cloned() else {
            continue;
        };
        let kept = keep.contains(&id) || mesh.test(&id) || id == "midiNoteGrid" || tex.is_external;
        if kept || tex.is_3d || tex.cube {
            continue;
        }
        pipeline.backend.clear_texture(&id);
    }
    let names: Vec<String> = pipeline.surfaces.keys().cloned().collect();
    for name in names {
        let read = format!("global_{name}_read");
        let write = format!("global_{name}_write");
        if pipeline.backend.textures.contains_key(&read)
            && pipeline.backend.textures.contains_key(&write)
            && let Some(surface) = pipeline.surfaces.get_mut(&name)
        {
            surface.read = Some(read);
            surface.write = Some(write);
        }
    }
    pipeline.global_uniforms = noisemaker_dsl::Object::new();
    pipeline.frame_index = 0.0;
    pipeline.last_time = 0.0;
}

/// Render the protocol's frames (or timed samples) on `pipeline`, writing
/// the PNGs.
fn render_frames(
    pipeline: &mut Pipeline,
    spec: &FixtureSpec,
    report: &mut FixtureReport,
) -> Result<(), String> {
    let read_output = |pipeline: &mut Pipeline| {
        let name = pipeline
            .graph
            .render_surface_name()
            .unwrap_or("o0")
            .to_owned();
        let surface = pipeline
            .surfaces
            .get(&name)
            .ok_or_else(|| format!("readback failed: no render surface {name}"))?;
        let id = surface
            .read
            .clone()
            .ok_or_else(|| format!("readback failed: render surface {name} has no read texture"))?;
        pipeline
            .backend
            .read_pixels(&id)
            .map_err(|e| format!("readback failed: {e}"))
    };
    if spec.run_seconds > 0.0 {
        let every_frames = (spec.sample_every * 60.0).round().max(1.0) as u64;
        let samples = ((spec.run_seconds * 60.0) / every_frames as f64)
            .floor()
            .max(1.0) as u64;
        for s in 0..samples {
            let start = s * every_frames;
            for i in 0..every_frames {
                let t = ((start + i + 1) as f64 / 600.0) % 1.0;
                pipeline
                    .render(t)
                    .map_err(|e| format!("render failed: {e}"))?;
            }
            let seconds = (s + 1) as f64 * spec.sample_every;
            let pixels = read_output(pipeline)?.oriented(spec.orientation);
            let path = timed_sample_path(&spec.out, seconds);
            write_png_rgba8(&path, pixels.width, pixels.height, &pixels.data)?;
            report.outputs.push(path);
        }
    } else {
        // __noisemakerSetPausedTime(time): syncTime, so the first frame's
        // deltaTime is 0.
        pipeline.sync_time(spec.time);
        for _ in 0..spec.frames {
            pipeline
                .render(spec.time)
                .map_err(|e| format!("render failed: {e}"))?;
        }
        let pixels = read_output(pipeline)?.oriented(spec.orientation);
        write_png_rgba8(&spec.out, pixels.width, pixels.height, &pixels.data)?;
        report.outputs.push(spec.out.clone());
    }
    Ok(())
}

fn collect_report(pipeline: &mut Pipeline, report: &mut FixtureReport) {
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
}

/// Upload PNG host textures (texture contents: `flipY: false`).
fn upload_host_textures(
    pipeline: &mut Pipeline,
    textures: &[(String, PathBuf)],
) -> Result<(), String> {
    for (id, path) in textures {
        let image = read_png_rgba8(path)?;
        pipeline
            .backend
            .update_texture_from_rgba8(id, image.width, image.height, &image.data, false)
            .map_err(|e| format!("host texture {id}: {e}"))?;
    }
    Ok(())
}

/// Write the pipeline's graph to `spec.graph_out` (program specs without
/// their shader sources, like the minter's graph).
fn write_graph(pipeline: &Pipeline, spec: &FixtureSpec) -> Result<(), String> {
    let Some(path) = &spec.graph_out else {
        return Ok(());
    };
    let mut graph = pipeline.graph.clone();
    for program in graph.programs.values_mut() {
        if let Value::Object(o) = program {
            for key in ["glsl", "fragment", "vertex"] {
                o.remove(key);
            }
        }
    }
    let text = graph
        .to_value()
        .to_json_pretty()
        .ok_or_else(|| format!("{}: graph is not serializable", path.display()))?;
    std::fs::write(path, text + "\n").map_err(|e| format!("{}: {e}", path.display()))
}

/// The sidecar a DSL fixture loads into `mesh0`: the given OBJ, else the
/// `.obj` next to the DSL file.
fn sidecar(spec: &FixtureSpec) -> Option<PathBuf> {
    match &spec.source {
        GraphSource::DslFile(path) => dsl_sidecar(path, spec.obj.as_deref()),
        _ => spec.obj.clone(),
    }
}

fn dsl_sidecar(dsl: &Path, obj: Option<&Path>) -> Option<PathBuf> {
    Some(obj.map_or_else(|| dsl.with_extension("obj"), Path::to_path_buf))
}

/// A demo host on `device` with the context's registries, fonts and media.
fn demo_host(device: &GpuDevice, width: u32, height: u32, context: &ProtocolContext) -> DemoHost {
    let renderer = CanvasRenderer::new(
        device,
        CanvasRendererOptions {
            width,
            height,
            registry: Some(context.registry.clone()),
            ..Default::default()
        },
    );
    DemoHost::new(
        renderer,
        DemoHostOptions {
            default_media: context.default_media.clone(),
            text_fonts: context.fonts.clone(),
            ..Default::default()
        },
    )
}

/// Apply parameter overrides as control changes, after checking that each
/// names a step of the program and a parameter of its effect.
fn apply_param_overrides(host: &mut DemoHost, params: &[ParamOverride]) -> Result<(), String> {
    for p in params {
        let state = host.program_state();
        let Some(def) = state.get_effect_def(&p.step) else {
            let steps = state.get_step_keys().join(", ");
            return Err(format!(
                "--param {}.{}: the program has no step {} (its steps: {steps})",
                p.step, p.name, p.step
            ));
        };
        let globals = def.get("globals");
        if !globals.as_object().is_some_and(|g| g.contains_key(&p.name)) {
            let names: Vec<String> = globals
                .as_object()
                .map(|g| g.keys().cloned().collect())
                .unwrap_or_default();
            let effect = state
                .effect_entry(&p.step)
                .map(|e| e.id())
                .unwrap_or_else(|| p.step.clone());
            return Err(format!(
                "--param {}.{}: {effect} has no parameter {} (its parameters: {})",
                p.step,
                p.name,
                p.name,
                names.join(", ")
            ));
        }
        host.set_control_value(&p.step, &p.name, p.value.clone())
            .map_err(|e| format!("--param {}.{}: {e}", p.step, p.name))?;
    }
    if !params.is_empty() {
        host.settle().map_err(|e| format!("host inputs: {e}"))?;
    }
    Ok(())
}

/// Load DSL source into `host` as the minter loads a fixture: the program
/// through the page (compile, ProgramState, controls, step values), its host
/// inputs settled, the parameter overrides applied, the host texture
/// overrides (a media step's texture is its media source; anything else is
/// uploaded as texture contents) and the meshes loaded (each `externalMesh`
/// step's first built-in mesh, then `obj` into `mesh0`). Returns the raw
/// (non-media) texture overrides it uploaded.
fn load_dsl(
    host: &mut DemoHost,
    text: &str,
    params: &[ParamOverride],
    host_textures: &[(String, PathBuf)],
    obj: Option<&Path>,
) -> Result<Vec<(String, PathBuf)>, String> {
    host.rebuild_pipeline_from_dsl(text, true)
        .map_err(|e| format!("DSL compile failed: {e}"))?;
    host.settle().map_err(|e| format!("host inputs: {e}"))?;
    apply_param_overrides(host, params)?;
    // Host texture overrides: a media step's texture is its media source
    // (shown as the page shows a loaded file: its imageSize follows);
    // anything else is uploaded as texture contents.
    let mut raw_overrides = Vec::new();
    for (id, png) in host_textures {
        let step = host
            .media_inputs()
            .iter()
            .find(|(_, m)| m.texture_id == *id)
            .map(|(step, _)| *step);
        match step {
            Some(step) => {
                let image = read_png_rgba8(png)?;
                host.set_media_image(step, Rc::new(image))
                    .map_err(|e| format!("host texture {id}: {e}"))?;
            }
            None => raw_overrides.push((id.clone(), png.clone())),
        }
    }
    let missing_media: Vec<String> = host
        .media_inputs()
        .values()
        .filter(|m| m.source.is_none())
        .map(|m| m.texture_id.clone())
        .collect();
    if !missing_media.is_empty() {
        eprintln!(
            "nm-render: warning: no media image for {} (the page's failed load)",
            missing_media.join(", ")
        );
    }
    let renderer = host.renderer_mut();
    let graph = renderer
        .pipeline()
        .ok_or("no pipeline after the DSL load")?
        .graph
        .clone();
    // applyMeshPlan: the minter loads through the page's renderer.
    if let Some(plan) = mesh_plan(&graph, obj) {
        let mut results = Vec::new();
        for (mesh_id, path) in &plan.builtins {
            results.push(renderer.load_builtin_mesh(path, mesh_id));
        }
        match &plan.obj_text {
            Some(text) => results.push(renderer.load_obj_from_string(text, "mesh0")),
            None if plan.builtins.is_empty() => {
                results.push(renderer.load_obj_from_string("", "mesh0"))
            }
            None => {}
        }
        let failed: Vec<String> = results
            .iter()
            .filter(|r| !r.success)
            .map(|r| r.error.clone().unwrap_or_default())
            .collect();
        if !failed.is_empty() {
            return Err(format!("mesh load failed: {}", failed.join("; ")));
        }
    }
    let pipeline = renderer.pipeline_mut().ok_or("no pipeline")?;
    upload_host_textures(pipeline, &raw_overrides)?;
    Ok(raw_overrides)
}

/// Run a DSL fixture through the demo host.
fn run_dsl_fixture(
    device: &GpuDevice,
    spec: &FixtureSpec,
    path: &Path,
    context: &ProtocolContext,
) -> Result<FixtureReport, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut host = demo_host(device, spec.width, spec.height, context);
    let mut report = FixtureReport::default();
    let result = (|| -> Result<(), String> {
        load_dsl(
            &mut host,
            &text,
            &spec.params,
            &spec.host_textures,
            sidecar(spec).as_deref(),
        )?;
        let pipeline = host.renderer_mut().pipeline_mut().ok_or("no pipeline")?;
        let mut keep: Vec<String> = external_texture_ids(&pipeline.graph);
        for id in async_overlay_ids(pipeline) {
            if !keep.contains(&id) {
                keep.push(id);
            }
        }
        for (id, _) in &spec.host_textures {
            if !keep.contains(id) {
                keep.push(id.clone());
            }
        }
        report.host_textures = keep.clone();
        write_graph(pipeline, spec)?;
        reset_fresh_state(pipeline, &keep.into_iter().collect());
        render_frames(pipeline, spec, &mut report)
    })();
    if let Some(pipeline) = host.renderer_mut().pipeline_mut() {
        collect_report(pipeline, &mut report);
    }
    let _ = host.renderer_mut().dispose();
    result?;
    Ok(report)
}

/// Render one fixture on `device`.
pub fn run_fixture(
    device: &GpuDevice,
    spec: &FixtureSpec,
    context: &ProtocolContext,
) -> Result<FixtureReport, String> {
    if !spec.params.is_empty() && !matches!(spec.source, GraphSource::DslFile(_)) {
        return Err("parameter overrides need a DSL fixture".into());
    }
    let graph = match &spec.source {
        GraphSource::DslFile(path) => return run_dsl_fixture(device, spec, path, context),
        GraphSource::GraphFile(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            Graph::from_reference_json(&text).map_err(|e| format!("{}: {e}", path.display()))?
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
    let mut report = FixtureReport::default();
    let result = (|| -> Result<(), String> {
        let pipeline = renderer.pipeline_mut();
        if let Some(plan) = mesh_plan(&pipeline.graph, sidecar(spec).as_deref()) {
            for (mesh_id, mesh) in plan.loads()? {
                pipeline.backend.upload_mesh_data(
                    &mesh_id,
                    &mesh.position_data,
                    &mesh.normal_data,
                    &mesh.uv_data,
                    mesh.width as u32,
                    mesh.height as u32,
                    mesh.vertex_count as u32,
                );
            }
        }
        upload_host_textures(pipeline, &spec.host_textures)?;
        report.host_textures = spec
            .host_textures
            .iter()
            .map(|(id, _)| id.clone())
            .collect();
        write_graph(pipeline, spec)?;
        render_frames(pipeline, spec, &mut report)
    })();
    collect_report(renderer.pipeline_mut(), &mut report);
    let _ = renderer.dispose();
    result?;
    Ok(report)
}

/// One animation: a DSL program loaded as [`run_fixture`] loads a DSL fixture
/// (demo host, host inputs, parameter overrides, meshes), then rendered over
/// its loop the way the page's render loop advances it, one PNG per frame.
#[derive(Debug, Clone)]
pub struct AnimationSpec {
    /// The DSL program.
    pub dsl: PathBuf,
    /// The directory the frames are written to (created when missing):
    /// `frame_00000.png`, `frame_00001.png`, ... ([`frame_path`]).
    pub out_dir: PathBuf,
    pub width: u32,
    pub height: u32,
    /// Frames per second.
    pub fps: f64,
    /// The loop duration in seconds (`CanvasRenderer.loopDuration`): frame
    /// `i` renders at the normalized loop time `((i / fps) % loop) / loop`.
    pub loop_seconds: f64,
    /// Frames to render (one loop is `fps * loop_seconds`).
    pub frames: u32,
    /// Parameter overrides.
    pub params: Vec<ParamOverride>,
    /// `(textureId, png path)` host texture overrides.
    pub host_textures: Vec<(String, PathBuf)>,
    /// The OBJ loaded into `mesh0` (`None`: the `.obj` next to the DSL when
    /// it exists).
    pub obj: Option<PathBuf>,
    /// The row order of the frames (a canvas's by default).
    pub orientation: Orientation,
}

impl Default for AnimationSpec {
    fn default() -> Self {
        AnimationSpec {
            dsl: PathBuf::new(),
            out_dir: PathBuf::new(),
            width: 512,
            height: 512,
            fps: 30.0,
            loop_seconds: 10.0,
            frames: 300,
            params: Vec::new(),
            host_textures: Vec::new(),
            obj: None,
            orientation: Orientation::Presented,
        }
    }
}

/// What an animation run produced.
#[derive(Debug, Clone, Default)]
pub struct AnimationReport {
    /// The frames written, in order.
    pub frames: Vec<PathBuf>,
    /// Device (validation) errors observed while rendering.
    pub device_errors: Vec<String>,
}

/// The path of frame `index` in `dir`: `frame_<index, 5 digits>.png`.
pub fn frame_path(dir: &Path, index: u32) -> PathBuf {
    dir.join(format!("frame_{index:05}.png"))
}

/// The normalized loop time of frame `index` at `fps` frames per second over
/// a loop of `loop_seconds` (`(elapsed % loopDuration) / loopDuration`).
pub fn frame_time(index: u32, fps: f64, loop_seconds: f64) -> f64 {
    let elapsed = f64::from(index) / fps;
    (elapsed % loop_seconds) / loop_seconds
}

/// Render an animation on `device`.
pub fn run_animation(
    device: &GpuDevice,
    spec: &AnimationSpec,
    context: &ProtocolContext,
) -> Result<AnimationReport, String> {
    if !(spec.fps > 0.0 && spec.fps.is_finite()) {
        return Err(format!("fps must be positive, got {}", spec.fps));
    }
    if !(spec.loop_seconds > 0.0 && spec.loop_seconds.is_finite()) {
        return Err(format!(
            "the loop duration must be positive, got {}",
            spec.loop_seconds
        ));
    }
    let text =
        std::fs::read_to_string(&spec.dsl).map_err(|e| format!("{}: {e}", spec.dsl.display()))?;
    std::fs::create_dir_all(&spec.out_dir)
        .map_err(|e| format!("{}: {e}", spec.out_dir.display()))?;
    let mut host = demo_host(device, spec.width, spec.height, context);
    let mut report = AnimationReport::default();
    let result = (|| -> Result<(), String> {
        load_dsl(
            &mut host,
            &text,
            &spec.params,
            &spec.host_textures,
            dsl_sidecar(&spec.dsl, spec.obj.as_deref()).as_deref(),
        )?;
        let renderer = host.renderer_mut();
        renderer.set_loop_duration(spec.loop_seconds);
        renderer.sync_time(frame_time(0, spec.fps, spec.loop_seconds));
        for i in 0..spec.frames {
            renderer
                .render(frame_time(i, spec.fps, spec.loop_seconds))
                .map_err(|e| format!("render failed: {e}"))?;
            let pixels = renderer
                .read_output()
                .map_err(|e| format!("readback failed: {e}"))?
                .oriented(spec.orientation);
            let path = frame_path(&spec.out_dir, i);
            write_png_rgba8(&path, pixels.width, pixels.height, &pixels.data)?;
            report.frames.push(path);
        }
        Ok(())
    })();
    if let Some(pipeline) = host.renderer_mut().pipeline_mut() {
        pipeline.backend.collect_device_errors();
        report.device_errors = pipeline.backend.device_error_log.clone();
    }
    let _ = host.renderer_mut().dispose();
    result?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_overrides_parse_json_or_text() {
        let p = ParamOverride::parse("step_0.scale=2.5").unwrap();
        assert_eq!(
            (p.step.as_str(), p.name.as_str(), p.value),
            ("step_0", "scale", Value::Number(2.5))
        );
        let p: ParamOverride = "step_12.color=[1, 0, 0, 1]".parse().unwrap();
        assert_eq!(p.value, Value::from_json("[1, 0, 0, 1]").unwrap());
        let p = ParamOverride::parse("step_1.text=Hello = World").unwrap();
        assert_eq!(p.value, Value::from("Hello = World"));
        let p = ParamOverride::parse("step_1.text=\"3\"").unwrap();
        assert_eq!(p.value, Value::from("3"));
        for bad in [
            "step_0.scale",
            "step0.scale=1",
            "step_.scale=1",
            "step_0.=1",
            "scale=1",
        ] {
            assert!(ParamOverride::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn animation_frames_follow_the_loop() {
        assert_eq!(frame_time(0, 30.0, 10.0), 0.0);
        assert_eq!(frame_time(150, 30.0, 10.0), 0.5);
        assert_eq!(frame_time(300, 30.0, 10.0), 0.0);
        assert_eq!(
            frame_path(Path::new("out"), 7),
            Path::new("out").join("frame_00007.png")
        );
    }
}
