//! The pipeline executor (port of `runtime/pipeline.js`'s `Pipeline`).
//!
//! It owns the graph and the backend, allocates the global surfaces (`o0..o7`,
//! `geo0..7`, `vol0..7`, the `mesh0..7` triplets and every `global_*` texture the
//! graph names) and the graph's textures, and renders frames: global uniforms,
//! automation, per-frame ping-pong bindings, pass conditions and repeats, mip
//! regeneration, presentation and the end-of-frame buffer swap.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;
use noisemaker_dsl::js::{math_round, parse_float};
use noisemaker_dsl::{Object, Value};

use crate::automation::{
    AudioRequirements, AutomationContext, ExternalState, is_automation_value, js_max, js_min,
    resolve_uniform_value, visit_audio_requirements,
};
use crate::backend::{Capabilities, FrameState, PixelData, WebGpuBackend};
use crate::diagnostics::{DiagnosticCollector, codes};
use crate::error::RenderError;
use crate::graph::{Graph, pass};
use crate::hooks::{AsyncInitContext, EffectLifecycle, EffectRegistry, UpdateContext};
use crate::jsre::JsRegex;
use crate::jsv::{interpolate, strict_equals, to_js_string, to_number};
use crate::preflight::{PreflightReport, mrt_format_bytes, preflight_effect};
use crate::sink::{CUBE_FACE_BASES, CanvasSink, SinkDescriptor, SinkManager};

/// Pipeline construction options.
#[derive(Clone, Default)]
pub struct PipelineOptions {
    /// `options.texturePooling`: share one backend texture among the members of
    /// each physical group of `graph.allocations` (default off).
    pub texture_pooling: bool,
    /// Effects with native lifecycle / asyncInit behavior.
    pub effects: EffectRegistry,
}

/// A global surface: double-buffered (`read`/`write`) or a mesh triplet.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Surface {
    pub read: Option<String>,
    pub write: Option<String>,
    pub current_frame: Option<f64>,
    pub positions: Option<String>,
    pub normals: Option<String>,
    pub uvs: Option<String>,
    pub width: Option<f64>,
    pub height: Option<f64>,
}

/// `getResourcePlan()`.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourcePlan {
    pub pooling: bool,
    pub allocations: IndexMap<String, Value>,
    pub shared_textures: Vec<Vec<String>>,
    pub textures: Vec<ResourceRecord>,
}

/// One materialized texture of a [`ResourcePlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceRecord {
    pub id: String,
    pub width: f64,
    pub height: f64,
    pub format: Value,
    pub virtual_textures: Vec<String>,
}

/// The per-pass viewport cache (`_viewportSpecSource` / `_viewportBox`).
#[derive(Debug, Clone, Default)]
struct ViewportCache {
    spec_source: Option<Value>,
    /// `_viewportBox === null`: an already-numeric viewport passes through.
    numeric: bool,
    /// `viewportResolved === spec` (set by the numeric branch).
    resolved_is_spec: bool,
}

/// A pending debounced async regeneration.
struct PendingRegen {
    due: std::time::Instant,
    params: Object,
}

/// The pipeline executor.
pub struct Pipeline {
    pub graph: Graph,
    pub backend: WebGpuBackend,
    pub texture_pooling: bool,
    texture_aliases: IndexMap<String, String>,
    pub sink_manager: SinkManager,
    sink_descriptor: SinkDescriptor,
    disposed: bool,
    /// Structured diagnostics (`ERR_DIMENSION_FALLBACK`).
    pub diagnostics: DiagnosticCollector,
    warned_dimension_fallbacks: HashSet<String>,
    pub frame_index: f64,
    pub last_time: f64,
    /// Global surfaces by name (a `Map`: insertion ordered).
    pub surfaces: IndexMap<String, Surface>,
    mip_targets: Vec<String>,
    pub global_uniforms: Object,
    pub width: f64,
    pub height: f64,
    frame_read_textures: IndexMap<String, Option<String>>,
    frame_write_textures: IndexMap<String, Option<String>>,
    pub animation_duration: f64,
    /// `_oscillatorPassProxy`: reused for every pass with automated uniforms.
    oscillator_pass_proxy: Object,
    /// `_resolvedUniforms`: swaps with the proxy's uniforms on every proxy use.
    resolved_uniforms: Object,
    pub last_pass_count: usize,
    pub is_compiling: bool,
    tile_offset: Option<[f64; 2]>,
    full_resolution: Option<[f64; 2]>,
    render_scale: Option<f64>,
    pub external_state: ExternalState,
    effects: EffectRegistry,
    async_renders: HashMap<String, Rc<RefCell<bool>>>,
    async_debounce: IndexMap<String, PendingRegen>,
    async_param_cache: HashMap<String, Object>,
    lifecycle_effects: IndexMap<String, Rc<RefCell<dyn EffectLifecycle>>>,
    init_lifecycle_done: HashSet<String>,
    runtime_uniforms: IndexMap<String, Object>,
    has_runtime_uniforms: bool,
    warned_volume_clamps: HashSet<String>,
    needs_midi_note_grid: bool,
    empty_note_grid: Option<Vec<f32>>,
    viewport_cache: HashMap<usize, ViewportCache>,
    /// The texture id presented by the last frame.
    pub last_presented: Option<String>,
}

const SURFACE_USAGE: &str = r#"["render","sample","copySrc","copyDst","storage"]"#;

fn surface_usage() -> Value {
    Value::from_json(SURFACE_USAGE).unwrap()
}

/// `Math.pow(base, exponent)` (`1 ** ±Infinity` is NaN in JavaScript).
fn js_pow(base: f64, exponent: f64) -> f64 {
    if exponent.is_infinite() && base.abs() == 1.0 {
        return f64::NAN;
    }
    base.powf(exponent)
}

/// `isStateSurface(name)` of `swapBuffers`: surfaces whose final bindings persist
/// across frames instead of swapping.
pub fn is_state_surface(name: &str) -> bool {
    if matches!(name, "xyz" | "vel" | "rgba" | "trail") {
        return true;
    }
    if name.ends_with("_xyz")
        || name.ends_with("_vel")
        || name.ends_with("_rgba")
        || name.ends_with("_trail")
    {
        return true;
    }
    if name.contains("state") || name.contains("State") {
        return true;
    }
    JsRegex::new(r"^(xyz|vel|rgba|points_trail)_node_\d+$", "").test(name)
}

/// The pipeline's `parseGlobalName`: `global_<name>` only.
pub fn parse_global_name(tex_id: &Value) -> Option<String> {
    tex_id.as_str()?.strip_prefix("global_").map(str::to_owned)
}

impl Pipeline {
    /// `new Pipeline(graph, backend, options)`.
    pub fn new(graph: Graph, backend: WebGpuBackend, options: PipelineOptions) -> Pipeline {
        let mut sink_manager = SinkManager::default();
        sink_manager
            .add(Box::new(CanvasSink::default()))
            .expect("a new sink manager accepts sinks");
        let mut proxy = Object::new();
        proxy.insert("uniforms", Value::object());
        Pipeline {
            graph,
            backend,
            texture_pooling: options.texture_pooling,
            texture_aliases: IndexMap::new(),
            sink_manager,
            sink_descriptor: SinkDescriptor::default(),
            disposed: false,
            diagnostics: DiagnosticCollector::default(),
            warned_dimension_fallbacks: HashSet::new(),
            frame_index: 0.0,
            last_time: 0.0,
            surfaces: IndexMap::new(),
            mip_targets: Vec::new(),
            global_uniforms: Object::new(),
            width: 0.0,
            height: 0.0,
            frame_read_textures: IndexMap::new(),
            frame_write_textures: IndexMap::new(),
            animation_duration: 10.0,
            oscillator_pass_proxy: proxy,
            resolved_uniforms: Object::new(),
            last_pass_count: 0,
            is_compiling: false,
            tile_offset: None,
            full_resolution: None,
            render_scale: None,
            external_state: ExternalState::default(),
            effects: options.effects,
            async_renders: HashMap::new(),
            async_debounce: IndexMap::new(),
            async_param_cache: HashMap::new(),
            lifecycle_effects: IndexMap::new(),
            init_lifecycle_done: HashSet::new(),
            runtime_uniforms: IndexMap::new(),
            has_runtime_uniforms: false,
            warned_volume_clamps: HashSet::new(),
            needs_midi_note_grid: false,
            empty_note_grid: None,
            viewport_cache: HashMap::new(),
            last_presented: None,
        }
    }

    /// `getCapabilities()`.
    pub fn get_capabilities(&self) -> Capabilities {
        self.backend.capabilities.clone()
    }

    /// `setAnimationDuration(seconds)`.
    pub fn set_animation_duration(&mut self, seconds: f64) {
        self.animation_duration = seconds;
    }

    /// `setMidiState(midiState)`.
    pub fn set_midi_state(&mut self, midi: Option<Box<dyn crate::automation::MidiSource>>) {
        self.external_state.midi = midi;
    }

    /// `setAudioState(audioState)`.
    pub fn set_audio_state(&mut self, audio: Option<Box<dyn crate::automation::AudioSource>>) {
        self.external_state.audio = audio;
    }

    /// `shouldDeferRender()`.
    pub fn should_defer_render(&mut self) -> bool {
        self.sink_manager.should_defer_render()
    }

    /// `init(width, height)`: initialize the backend, compile every program, size.
    pub fn init(&mut self, width: f64, height: f64) -> Result<(), RenderError> {
        self.backend.init();
        self.compile_programs()?;
        self.resize(width, height)
    }

    /// `compilePrograms()`: compile each distinct program of the graph's passes.
    pub fn compile_programs(&mut self) -> Result<(), RenderError> {
        self.is_compiling = true;
        let result = (|| {
            let mut compiled: Vec<Value> = Vec::new();
            for i in 0..self.graph.passes.len() {
                let program = pass::get(&self.graph.passes[i], "program").clone();
                if compiled.iter().any(|p| p == &program) {
                    continue;
                }
                let Some(spec) = self.resolve_program_spec(&self.graph.passes[i]).cloned() else {
                    return Err(RenderError::thrown(
                        "ERR_PROGRAM_SPEC_MISSING",
                        &[
                            ("program", program),
                            ("pass", pass::get(&self.graph.passes[i], "id").clone()),
                        ],
                    ));
                };
                self.backend
                    .compile_program(&to_js_string(&program), &spec)?;
                compiled.push(program);
            }
            Ok(())
        })();
        self.is_compiling = false;
        result
    }

    /// `resolveProgramSpec(pass)`.
    pub fn resolve_program_spec(&self, pass: &Object) -> Option<&Value> {
        self.graph
            .programs
            .get(&pass::program(pass))
            .filter(|spec| spec.is_truthy())
    }

    /// `resize(width, height)`: (re)create surfaces and textures for the new size.
    pub fn resize(&mut self, width: f64, height: f64) -> Result<(), RenderError> {
        self.width = width;
        self.height = height;
        self.sink_descriptor.width = width;
        self.sink_descriptor.height = height;
        self.sink_manager.configure(&self.sink_descriptor);
        self.create_surfaces()?;
        self.refresh_mip_targets();
        let defaults = self.collect_default_uniforms();
        self.recreate_textures(&defaults)?;
        self.init_async_effects()?;
        self.init_lifecycle_effects();
        Ok(())
    }

    /// `initLifecycleEffects()`.
    pub fn init_lifecycle_effects(&mut self) {
        self.lifecycle_effects.clear();
        let mut seen = HashSet::new();
        for p in &self.graph.passes {
            let key = pass::get(p, "effectKey");
            if !key.is_truthy() {
                continue;
            }
            let key = to_js_string(key);
            if !seen.insert(key.clone()) {
                continue;
            }
            let Some(lifecycle) = self.effects.get(&key).and_then(|e| e.lifecycle.clone()) else {
                continue;
            };
            let (has_init, has_update, has_destroy) = {
                let l = lifecycle.borrow();
                (l.has_on_init(), l.has_on_update(), l.has_on_destroy())
            };
            if !has_init && !has_update && !has_destroy {
                continue;
            }
            self.lifecycle_effects
                .insert(key.clone(), lifecycle.clone());
            if has_init && !self.init_lifecycle_done.contains(&key) {
                self.init_lifecycle_done.insert(key);
                lifecycle.borrow_mut().on_init();
            }
        }
    }

    /// `_invokeUpdateHooks(time, deltaTime)`.
    fn invoke_update_hooks(&mut self, time: f64, delta: f64) {
        self.has_runtime_uniforms = false;
        self.runtime_uniforms.clear();
        for (key, lifecycle) in &self.lifecycle_effects {
            if !lifecycle.borrow().has_on_update() {
                continue;
            }
            let context = UpdateContext {
                time,
                delta,
                uniforms: &self.global_uniforms,
            };
            if let Some(uniforms) = lifecycle.borrow_mut().on_update(&context) {
                self.runtime_uniforms.insert(key.clone(), uniforms);
                self.has_runtime_uniforms = true;
            }
        }
    }

    /// `_withRuntimeUniforms(pass, runtimeUniforms)`: hook uniforms fill only keys
    /// the pass does not resolve.
    fn with_runtime_uniforms(pass: &Object, runtime: &Object) -> Object {
        let mut merged = pass::uniforms(pass).cloned().unwrap_or_default();
        for (key, value) in runtime.iter() {
            if merged.get_or_undefined(key).is_undefined() {
                merged.insert(key.clone(), value.clone());
            }
        }
        let mut out = pass.clone();
        out.insert("uniforms", Value::Object(merged));
        out
    }

    /// `initAsyncEffects()`: start every asyncInit effect of the graph.
    pub fn init_async_effects(&mut self) -> Result<(), RenderError> {
        for cancelled in self.async_renders.values() {
            *cancelled.borrow_mut() = true;
        }
        self.async_renders.clear();
        self.async_debounce.clear();
        self.async_param_cache.clear();
        let mut seen = HashSet::new();
        let mut starts = Vec::new();
        for p in &self.graph.passes {
            let key = pass::get(p, "effectKey");
            let node = pass::get(p, "nodeId");
            if !key.is_truthy() || !node.is_truthy() {
                continue;
            }
            let node = to_js_string(node);
            if !seen.insert(node.clone()) {
                continue;
            }
            let key = to_js_string(key);
            if self
                .effects
                .get(&key)
                .and_then(|e| e.async_init.as_ref())
                .is_none()
            {
                continue;
            }
            starts.push((node, key));
        }
        for (node, key) in starts {
            let params = self.global_uniforms.clone();
            self.start_async_init(&node, &key, params)?;
        }
        Ok(())
    }

    /// `checkAsyncRegen(nodeId, effectKey, stepValues)`: schedule a debounced
    /// regeneration when a scalar parameter of an asyncInit effect changed.
    pub fn check_async_regen(&mut self, node_id: &str, effect_key: &str, step_values: &Object) {
        let Some(effect) = self.effects.get(effect_key) else {
            return;
        };
        if effect.async_init.is_none() {
            return;
        }
        let globals = effect.globals.clone();
        let cache = self
            .async_param_cache
            .entry(node_id.to_owned())
            .or_default();
        let mut changed = false;
        for (name, value) in step_values.iter() {
            if name == "alpha" || name.starts_with('_') || value.is_nullish() {
                continue;
            }
            if matches!(value, Value::Object(_) | Value::Array(_)) {
                continue;
            }
            if !globals.get_or_undefined(name).is_truthy() {
                continue;
            }
            if !strict_equals(cache.get_or_undefined(name), value) {
                changed = true;
                cache.insert(name.clone(), value.clone());
            }
        }
        if changed {
            self.async_debounce.insert(
                node_id.to_owned(),
                PendingRegen {
                    due: std::time::Instant::now() + std::time::Duration::from_millis(300),
                    params: step_values.clone(),
                },
            );
        }
    }

    /// Fire debounced regenerations whose 300 ms timer has elapsed (the
    /// reference's `setTimeout` callbacks).
    pub fn run_due_async_regens(&mut self) -> Result<(), RenderError> {
        let now = std::time::Instant::now();
        let due: Vec<String> = self
            .async_debounce
            .iter()
            .filter(|(_, p)| p.due <= now)
            .map(|(k, _)| k.clone())
            .collect();
        for node in due {
            let Some(pending) = self.async_debounce.shift_remove(&node) else {
                continue;
            };
            let key = self
                .graph
                .passes
                .iter()
                .find(|p| pass::get(p, "nodeId").as_str() == Some(node.as_str()))
                .map(|p| to_js_string(pass::get(p, "effectKey")));
            if let Some(key) = key {
                self.start_async_init(&node, &key, pending.params)?;
            }
        }
        Ok(())
    }

    /// `_startAsyncInit(nodeId, effectDef)`.
    fn start_async_init(
        &mut self,
        node_id: &str,
        effect_key: &str,
        params: Object,
    ) -> Result<(), RenderError> {
        if let Some(previous) = self.async_renders.get(node_id) {
            *previous.borrow_mut() = true;
        }
        let cancelled = Rc::new(RefCell::new(false));
        self.async_renders
            .insert(node_id.to_owned(), cancelled.clone());
        let Some(effect) = self
            .effects
            .get(effect_key)
            .and_then(|e| e.async_init.clone())
        else {
            return Ok(());
        };
        struct Context<'a> {
            backend: &'a mut WebGpuBackend,
            node_id: &'a str,
            width: f64,
            height: f64,
            params: Object,
            cancelled: Rc<RefCell<bool>>,
            error: Option<RenderError>,
        }
        impl AsyncInitContext for Context<'_> {
            fn update_texture(&mut self, tex_name: &str, width: u32, height: u32, rgba: &[u8]) {
                if *self.cancelled.borrow() {
                    return;
                }
                let id = format!("{}_{tex_name}", self.node_id);
                if let Err(e) = self
                    .backend
                    .update_texture_from_rgba8(&id, width, height, rgba, true)
                {
                    self.error.get_or_insert(e);
                }
            }
            fn width(&self) -> f64 {
                self.width
            }
            fn height(&self) -> f64 {
                self.height
            }
            fn params(&self) -> &Object {
                &self.params
            }
            fn is_cancelled(&self) -> bool {
                *self.cancelled.borrow()
            }
        }
        let mut context = Context {
            backend: &mut self.backend,
            node_id,
            width: self.width,
            height: self.height,
            params,
            cancelled,
            error: None,
        };
        // asyncInit failures are logged by the reference, never thrown.
        let _ = effect.async_init(&mut context);
        if let Some(e) = context.error {
            return Err(e);
        }
        Ok(())
    }

    /// `collectDefaultUniforms()`: every pass's uniforms merged in pass order.
    pub fn collect_default_uniforms(&self) -> Object {
        let mut uniforms = Object::new();
        for p in &self.graph.passes {
            if let Some(u) = pass::uniforms(p) {
                uniforms.assign(u);
            }
        }
        uniforms
    }

    /// `getAudioInputRequirements()`.
    pub fn get_audio_input_requirements(&self) -> AudioRequirements {
        let mut out = AudioRequirements::default();
        for p in &self.graph.passes {
            let key = pass::get(p, "effectKey");
            if key.is_truthy()
                && EffectRegistry::catalog_tags(&to_js_string(key))
                    .iter()
                    .any(|t| t == "audio")
            {
                out.needs_legacy = true;
            }
            visit_audio_requirements(pass::get(p, "uniforms"), &mut out);
        }
        out
    }

    /// `isVolumeSizeUniform(name)`.
    pub fn is_volume_size_uniform(name: &str) -> bool {
        name == "volumeSize"
            || name.starts_with("volumeSize_chain_")
            || name.starts_with("volumeSize_node_")
    }

    /// `clampVolumeSize(value)`: snap a volume size down (power of two from 16) so
    /// its `size x size²` atlas fits the device's maximum texture size.
    pub fn clamp_volume_size(&mut self, value: f64) -> f64 {
        let max = self.backend.capabilities.max_texture_size as f64;
        if max == 0.0 || value * value <= max {
            return value;
        }
        let mut clamped = 16.0;
        while (clamped * 2.0) * (clamped * 2.0) <= max && clamped * 2.0 < value {
            clamped *= 2.0;
        }
        let key = format!(
            "{}->{}",
            interpolate(&Value::Number(value)),
            interpolate(&Value::Number(clamped))
        );
        self.warned_volume_clamps.insert(key);
        clamped
    }

    /// `clampGraphVolumeSizes()`.
    pub fn clamp_graph_volume_sizes(&mut self) {
        for i in 0..self.graph.passes.len() {
            let Some(uniforms) = pass::uniforms(&self.graph.passes[i]) else {
                continue;
            };
            let keys: Vec<String> = uniforms.keys().cloned().collect();
            for key in keys {
                if !Self::is_volume_size_uniform(&key) {
                    continue;
                }
                let Some(Value::Number(value)) = pass::uniforms(&self.graph.passes[i])
                    .and_then(|u| u.get(&key))
                    .cloned()
                else {
                    continue;
                };
                let clamped = self.clamp_volume_size(value);
                if clamped != value {
                    pass::uniforms_mut(&mut self.graph.passes[i])
                        .unwrap()
                        .insert(key, Value::Number(clamped));
                }
            }
        }
    }

    /// `applyMrtFormatBudget()`: demote trailing rgba32f attachments of MRT passes
    /// over the device's color-attachment byte budget to rgba16f.
    pub fn apply_mrt_format_budget(&mut self) {
        let budget = self.backend.capabilities.max_color_bytes_per_sample;
        if budget == 0 {
            return;
        }
        let Some(textures) = self.graph.textures.as_mut() else {
            return;
        };
        for p in &self.graph.passes {
            let Some(outputs) = pass::outputs(p) else {
                continue;
            };
            let ids: Vec<String> = outputs.values().map(to_js_string).collect();
            if ids.len() <= 1 {
                continue;
            }
            let mut total: u32 = ids
                .iter()
                .map(|id| {
                    mrt_format_bytes(
                        textures
                            .get(id)
                            .map(|s| s.get_or_undefined("format"))
                            .unwrap_or(&Value::Undefined),
                    )
                })
                .sum();
            if total <= budget {
                continue;
            }
            for id in ids.iter().rev() {
                if total <= budget {
                    break;
                }
                let Some(spec) = textures.get_mut(id) else {
                    continue;
                };
                if matches!(
                    spec.get_or_undefined("format").as_str(),
                    Some("rgba32f" | "rgba32float")
                ) {
                    spec.insert("format", Value::from("rgba16f"));
                    total -= 8;
                }
            }
        }
    }

    /// `recreateTexturePreserving(texId, spec)`: recreate a 2D texture, resampling
    /// a persistent texture's contents into the replacement.
    pub fn recreate_texture_preserving(
        &mut self,
        tex_id: &str,
        spec: &Object,
    ) -> Result<(), RenderError> {
        let existing = self.backend.textures.get(tex_id).cloned();
        let mut preserve_id = None;
        if let Some(existing) = existing
            && !existing.is_3d
            && existing.persistent == Some(true)
        {
            let id = format!("{tex_id}__preserve_tmp");
            let mut tmp = Object::new();
            tmp.insert("width", Value::Number(existing.width));
            tmp.insert("height", Value::Number(existing.height));
            tmp.insert("format", existing.format.clone());
            tmp.insert(
                "usage",
                Value::from_json(r#"["sample","copySrc","copyDst"]"#).unwrap(),
            );
            self.backend.create_texture(&id, &tmp)?;
            self.backend.copy_texture(tex_id, &id);
            preserve_id = Some(id);
        }
        self.backend.destroy_texture(tex_id);
        self.backend.create_texture(tex_id, spec)?;
        if let Some(id) = preserve_id {
            self.backend.copy_texture(&id, tex_id);
            self.backend.destroy_texture(&id);
        }
        Ok(())
    }

    /// `refreshMipTargets()`.
    pub fn refresh_mip_targets(&mut self) {
        self.mip_targets.clear();
        let Some(textures) = &self.graph.textures else {
            return;
        };
        for (tex_id, spec) in textures {
            if !spec.get_or_undefined("mipmaps").is_truthy() {
                continue;
            }
            if tex_id.starts_with("global_") {
                if let Some(name) = parse_global_name(&Value::from(tex_id.as_str()))
                    && let Some(surface) = self.surfaces.get(&name)
                {
                    self.mip_targets.extend(surface.read.clone());
                    self.mip_targets.extend(surface.write.clone());
                }
            } else {
                self.mip_targets.push(tex_id.clone());
            }
        }
    }

    fn surface_spec(
        width: f64,
        height: f64,
        format: &Value,
        mipmaps: bool,
        persistent: bool,
    ) -> Object {
        let mut spec = Object::new();
        spec.insert("width", Value::Number(width));
        spec.insert("height", Value::Number(height));
        spec.insert("format", format.clone());
        spec.insert("usage", surface_usage());
        spec.insert("mipmaps", Value::Bool(mipmaps));
        spec.insert("persistent", Value::Bool(persistent));
        spec
    }

    /// `createSurfaces()`: the global surfaces (`o0..o7`, `geo0..7`, `vol0..7` and
    /// every `global_*` id the passes name) and the `mesh0..7` triplets.
    pub fn create_surfaces(&mut self) -> Result<(), RenderError> {
        self.clamp_graph_volume_sizes();
        self.apply_mrt_format_budget();

        let mut names: IndexMap<String, ()> = IndexMap::new();
        for prefix in ["o", "geo", "vol"] {
            for i in 0..8 {
                names.insert(format!("{prefix}{i}"), ());
            }
        }
        let defaults = self.collect_default_uniforms();
        let mesh_pattern = JsRegex::new(r"^mesh\d+_(positions|normals|uvs)$", "");
        for p in &self.graph.passes {
            for key in ["inputs", "outputs"] {
                if let Some(ids) = pass::get(p, key).as_object() {
                    for id in ids.values() {
                        if let Some(name) = parse_global_name(id)
                            && !mesh_pattern.test(&name)
                        {
                            names.insert(name, ());
                        }
                    }
                }
            }
        }
        self.needs_midi_note_grid = self.graph.passes.iter().any(|p| {
            pass::inputs(p).is_some_and(|inputs| {
                inputs
                    .values()
                    .any(|id| id.as_str() == Some("midiNoteGrid"))
            })
        });

        let volume = JsRegex::new(r"^vol[0-7]$", "");
        for name in names.keys() {
            let is_volume = volume.test(name);
            let mut surface_width = if is_volume { 64.0 } else { self.width };
            let mut surface_height = if is_volume { 4096.0 } else { self.height };
            let mut surface_format = Value::from("rgba16f");
            let tex_spec = self
                .graph
                .textures
                .as_ref()
                .and_then(|t| t.get(&format!("global_{name}")))
                .cloned();
            if let Some(spec) = &tex_spec {
                surface_width =
                    self.resolve_dimension(spec.get_or_undefined("width"), self.width, &defaults)?;
                surface_height = self.resolve_dimension(
                    spec.get_or_undefined("height"),
                    self.height,
                    &defaults,
                )?;
                let format = spec.get_or_undefined("format");
                if format.is_truthy() {
                    surface_format = format.clone();
                }
            }
            let mipmaps = tex_spec
                .as_ref()
                .is_some_and(|s| matches!(s.get_or_undefined("mipmaps"), Value::Bool(true)));
            let persistent = tex_spec
                .as_ref()
                .is_some_and(|s| matches!(s.get_or_undefined("persistent"), Value::Bool(true)));
            let spec = Self::surface_spec(
                surface_width,
                surface_height,
                &surface_format,
                mipmaps,
                persistent,
            );
            let read_id = format!("global_{name}_read");
            let write_id = format!("global_{name}_write");

            if let Some(old) = self.surfaces.get(name).cloned() {
                let matches = |id: &Option<String>| {
                    id.as_ref()
                        .and_then(|id| self.backend.textures.get(id))
                        .is_some_and(|t| {
                            t.width == surface_width
                                && t.height == surface_height
                                && strict_equals(&t.format, &surface_format)
                        })
                };
                if matches(&old.read) && matches(&old.write) {
                    continue;
                }
                self.recreate_texture_preserving(&read_id, &spec)?;
                self.recreate_texture_preserving(&write_id, &spec)?;
                continue;
            }
            self.backend.create_texture(&read_id, &spec)?;
            self.backend.create_texture(&write_id, &spec)?;
            self.surfaces.insert(
                name.clone(),
                Surface {
                    read: Some(read_id),
                    write: Some(write_id),
                    current_frame: Some(0.0),
                    ..Default::default()
                },
            );
        }

        for i in 0..8 {
            let name = format!("mesh{i}");
            if self.surfaces.contains_key(&name) {
                continue;
            }
            let mut spec = Object::new();
            spec.insert("width", Value::Number(256.0));
            spec.insert("height", Value::Number(256.0));
            spec.insert("format", Value::from("rgba32f"));
            spec.insert("usage", surface_usage());
            let positions = format!("global_{name}_positions");
            let normals = format!("global_{name}_normals");
            let uvs = format!("global_{name}_uvs");
            self.backend.create_texture(&positions, &spec)?;
            self.backend.create_texture(&normals, &spec)?;
            self.backend.create_texture(&uvs, &spec)?;
            self.surfaces.insert(
                name,
                Surface {
                    positions: Some(positions),
                    normals: Some(normals),
                    uvs: Some(uvs),
                    width: Some(256.0),
                    height: Some(256.0),
                    ..Default::default()
                },
            );
        }
        Ok(())
    }

    /// `isDynamicDimension(spec)`: only plain numbers are fixed.
    pub fn is_dynamic_dimension(spec: &Value) -> bool {
        !matches!(spec, Value::Number(_))
    }

    /// `recreateTextures(uniforms)`: (re)allocate every graph texture whose resolved
    /// size or format changed.
    pub fn recreate_textures(&mut self, uniforms: &Object) -> Result<(), RenderError> {
        let Some(textures) = self.graph.textures.clone() else {
            return Ok(());
        };
        let previous = std::mem::take(&mut self.texture_aliases);
        self.texture_aliases = if self.texture_pooling {
            self.build_texture_pooling_plan()
        } else {
            IndexMap::new()
        };
        self.release_regrouped_textures(&previous);

        for (tex_id, spec) in &textures {
            let is_global = tex_id.starts_with("global");
            if is_global
                && !Self::is_dynamic_dimension(spec.get_or_undefined("width"))
                && !Self::is_dynamic_dimension(spec.get_or_undefined("height"))
            {
                continue;
            }
            let width =
                self.resolve_dimension(spec.get_or_undefined("width"), self.width, uniforms)?;
            let height =
                self.resolve_dimension(spec.get_or_undefined("height"), self.height, uniforms)?;
            if is_global {
                let Some(name) = parse_global_name(&Value::from(tex_id.as_str())) else {
                    continue;
                };
                let Some(surface) = self.surfaces.get(&name).cloned() else {
                    continue;
                };
                let format_value = spec.get_or_undefined("format");
                let expected = if format_value.is_truthy() {
                    format_value.clone()
                } else {
                    Value::from("rgba16f")
                };
                let matches = |id: &Option<String>| {
                    id.as_ref()
                        .and_then(|id| self.backend.textures.get(id))
                        .is_some_and(|t| {
                            t.width == width
                                && t.height == height
                                && strict_equals(&t.format, &expected)
                        })
                };
                if matches(&surface.read) && matches(&surface.write) {
                    continue;
                }
                let surface_spec = Self::surface_spec(
                    width,
                    height,
                    &expected,
                    matches!(spec.get_or_undefined("mipmaps"), Value::Bool(true)),
                    matches!(spec.get_or_undefined("persistent"), Value::Bool(true)),
                );
                let read = surface.read.clone().unwrap_or_else(|| "undefined".into());
                let write = surface.write.clone().unwrap_or_else(|| "undefined".into());
                self.recreate_texture_preserving(&read, &surface_spec)?;
                self.recreate_texture_preserving(&write, &surface_spec)?;
            } else {
                if let Some(storage) = self.texture_aliases.get(tex_id)
                    && storage != tex_id
                {
                    continue;
                }
                let is_3d = spec.get_or_undefined("is3D").is_truthy();
                if let Some(existing) = self.backend.textures.get(tex_id).cloned()
                    && existing.width == width
                    && existing.height == height
                    && strict_equals(&existing.format, spec.get_or_undefined("format"))
                {
                    if !is_3d {
                        continue;
                    }
                    let depth =
                        self.resolve_dimension(spec.get_or_undefined("depth"), width, uniforms)?;
                    if existing.depth == Some(depth) {
                        continue;
                    }
                }
                let mut new_spec = spec.clone();
                new_spec.insert("width", Value::Number(width));
                new_spec.insert("height", Value::Number(height));
                if is_3d {
                    let depth =
                        self.resolve_dimension(spec.get_or_undefined("depth"), width, uniforms)?;
                    new_spec.insert("depth", Value::Number(depth));
                    self.backend.destroy_texture(tex_id);
                    self.backend.create_texture_3d(tex_id, &new_spec)?;
                } else {
                    self.recreate_texture_preserving(tex_id, &new_spec)?;
                }
            }
        }
        self.apply_texture_aliases();
        self.refresh_mip_targets();
        Ok(())
    }

    /// `buildTexturePoolingPlan()`: virtual id → storage id for the poolable
    /// members of each physical allocation group.
    pub fn build_texture_pooling_plan(&self) -> IndexMap<String, String> {
        let mut aliases = IndexMap::new();
        let (Some(allocations), Some(textures)) = (&self.graph.allocations, &self.graph.textures)
        else {
            return aliases;
        };
        let mut first_touch_is_write: HashMap<String, bool> = HashMap::new();
        let mut self_sampled: HashSet<String> = HashSet::new();
        let mut partially_written: HashSet<String> = HashSet::new();
        for p in &self.graph.passes {
            let inputs: Vec<String> = pass::inputs(p)
                .map(|i| i.values().map(to_js_string).collect())
                .unwrap_or_default();
            let mut input_set: Vec<String> = Vec::new();
            for id in inputs {
                if !input_set.contains(&id) {
                    input_set.push(id);
                }
            }
            let outputs: Vec<String> = pass::outputs(p)
                .map(|o| o.values().map(to_js_string).collect())
                .unwrap_or_default();
            // Scatter draws and blending keep the destination's previous contents
            // observable, and so does a non-clearing viewport pass.
            let partial = pass::get(p, "drawMode").is_truthy()
                || pass::get(p, "blend").is_truthy()
                || (pass::get(p, "viewport").is_truthy() && !pass::get(p, "clear").is_truthy());
            if partial {
                partially_written.extend(outputs.iter().cloned());
            }
            for id in &outputs {
                first_touch_is_write.entry(id.clone()).or_insert(true);
                if input_set.contains(id) {
                    self_sampled.insert(id.clone());
                }
            }
            for id in &input_set {
                first_touch_is_write.entry(id.clone()).or_insert(false);
            }
        }
        let mut groups: IndexMap<String, Vec<String>> = IndexMap::new();
        for (tex_id, physical) in allocations {
            if !physical.is_truthy() || !textures.contains_key(tex_id) {
                continue;
            }
            if tex_id.starts_with("global") {
                continue;
            }
            if first_touch_is_write.get(tex_id) == Some(&false) {
                continue;
            }
            if self_sampled.contains(tex_id) || partially_written.contains(tex_id) {
                continue;
            }
            groups
                .entry(to_js_string(physical))
                .or_default()
                .push(tex_id.clone());
        }
        for members in groups.values() {
            if members.len() < 2 {
                continue;
            }
            let specs: Vec<Option<&Object>> = members.iter().map(|id| textures.get(id)).collect();
            if specs.iter().any(|s| match s {
                None => true,
                Some(s) => {
                    matches!(s.get_or_undefined("persistent"), Value::Bool(true))
                        || matches!(s.get_or_undefined("mipmaps"), Value::Bool(true))
                        || matches!(s.get_or_undefined("is3D"), Value::Bool(true))
                }
            }) {
                continue;
            }
            let signature = |s: &Object| {
                Value::Array(vec![
                    s.get_or_undefined("width").clone(),
                    s.get_or_undefined("height").clone(),
                    s.get_or_undefined("format").clone(),
                ])
                .to_json()
            };
            let first = signature(specs[0].unwrap());
            if !specs.iter().all(|s| signature(s.unwrap()) == first) {
                continue;
            }
            let storage = members[0].clone();
            for member in members {
                aliases.insert(member.clone(), storage.clone());
            }
        }
        aliases
    }

    /// `releaseRegroupedTextures(previousAliases, nextAliases)`.
    fn release_regrouped_textures(&mut self, previous: &IndexMap<String, String>) {
        if previous.is_empty() {
            return;
        }
        let mut groups: IndexMap<String, Vec<String>> = IndexMap::new();
        for (member, storage) in previous {
            groups
                .entry(storage.clone())
                .or_default()
                .push(member.clone());
        }
        for (storage, members) in groups {
            let unchanged = members
                .iter()
                .all(|m| self.texture_aliases.get(m) == Some(&storage));
            if unchanged {
                continue;
            }
            for member in members {
                if self.backend.textures.contains_key(&member) {
                    self.backend.destroy_texture(&member);
                }
            }
        }
    }

    /// `applyTextureAliases()`: point each pooled secondary member at its group's
    /// storage record.
    fn apply_texture_aliases(&mut self) {
        let aliases = self.texture_aliases.clone();
        for (member, storage) in aliases {
            if member == storage {
                continue;
            }
            let Some(record) = self.backend.textures.get(&storage).cloned() else {
                continue;
            };
            if let Some(existing) = self.backend.textures.get(&member).cloned()
                && !Rc::ptr_eq(&existing, &record)
            {
                self.backend.destroy_texture(&member);
            }
            self.backend.textures.insert(member, record);
        }
    }

    /// `getResourcePlan()`.
    pub fn get_resource_plan(&self) -> ResourcePlan {
        let allocations = self.graph.allocations.clone().unwrap_or_default();
        let Some(textures) = &self.graph.textures else {
            return ResourcePlan {
                pooling: self.texture_pooling,
                allocations: IndexMap::new(),
                shared_textures: Vec::new(),
                textures: Vec::new(),
            };
        };
        let mut records: Vec<(Rc<crate::backend::TextureRecord>, String, Vec<String>)> = Vec::new();
        for tex_id in textures.keys() {
            if tex_id.starts_with("global") {
                continue;
            }
            let Some(record) = self.backend.textures.get(tex_id) else {
                continue;
            };
            match records.iter_mut().find(|(r, _, _)| Rc::ptr_eq(r, record)) {
                Some((_, _, members)) => members.push(tex_id.clone()),
                None => records.push((record.clone(), tex_id.clone(), vec![tex_id.clone()])),
            }
        }
        let mut out_records = Vec::new();
        let mut shared = Vec::new();
        for (_, id, members) in records {
            let record = self.backend.textures.get(&id);
            out_records.push(ResourceRecord {
                id: id.clone(),
                width: record.map(|r| r.width).unwrap_or(f64::NAN),
                height: record.map(|r| r.height).unwrap_or(f64::NAN),
                format: record.map(|r| r.format.clone()).unwrap_or(Value::Undefined),
                virtual_textures: members.clone(),
            });
            if members.len() > 1 {
                shared.push(members);
            }
        }
        ResourcePlan {
            pooling: self.texture_pooling,
            allocations,
            shared_textures: shared,
            textures: out_records,
        }
    }

    /// `updateParameterTextures(uniforms)`.
    pub fn update_parameter_textures(&mut self, uniforms: &Object) -> Result<(), RenderError> {
        self.recreate_textures(uniforms)?;
        self.refresh_mip_targets();
        Ok(())
    }

    /// `setTileRegion({offset, fullResolution, renderScale})`.
    pub fn set_tile_region(
        &mut self,
        offset: [f64; 2],
        full_resolution: [f64; 2],
        render_scale: Option<f64>,
    ) {
        self.tile_offset = Some(offset);
        self.full_resolution = Some(full_resolution);
        self.render_scale = Some(render_scale.unwrap_or(1.0));
    }

    /// `clearTileRegion()`.
    pub fn clear_tile_region(&mut self) {
        self.tile_offset = None;
        self.full_resolution = None;
        self.render_scale = None;
    }

    /// `renderCubemap({size, outputSurface, time})`: one render per cube face
    /// (+X, -X, +Y, -Y, +Z, -Z) with `cubeBasis` set to the face basis.
    pub fn render_cubemap(
        &mut self,
        size: f64,
        output_surface: &str,
        time: f64,
    ) -> Result<Vec<PixelData>, RenderError> {
        let (prev_w, prev_h) = (self.width, self.height);
        if self.width != size || self.height != size {
            self.resize(size, size)?;
        }
        let mut faces = Vec::with_capacity(6);
        for basis in CUBE_FACE_BASES.iter() {
            self.set_uniform(
                "cubeBasis",
                Value::Array(basis.iter().map(|v| Value::Number(*v)).collect()),
            )?;
            self.render(time)?;
            let Some(surface) = self.surfaces.get(output_surface) else {
                return Err(RenderError::Js(format!(
                    "Error: renderCubemap: output surface \"{output_surface}\" not found — the composition must write its cubemap-renderer result to it (e.g. .renderCubemapSurface().write({output_surface}))"
                )));
            };
            let read = surface.read.clone().unwrap_or_default();
            faces.push(self.backend.read_pixels(&read)?);
        }
        if prev_w != size || prev_h != size {
            self.resize(prev_w, prev_h)?;
        }
        Ok(faces)
    }

    /// `setUniform(name, value)`: set a global uniform, propagate it to every pass
    /// that carries it (and its `_node_N`/`_chain_N` variants for unscoped names),
    /// expand palette presets, and resize textures that depend on it.
    pub fn set_uniform(&mut self, name: &str, value: Value) -> Result<(), RenderError> {
        let mut value = value;
        if (name == "stateSize" || name.starts_with("stateSize_node_"))
            && let Value::Number(n) = value
        {
            let max_state = if self.backend.capabilities.max_state_size != 0 {
                self.backend.capabilities.max_state_size as f64
            } else {
                2048.0
            };
            if n > max_state {
                value = Value::Number(max_state);
            }
        }
        if Self::is_volume_size_uniform(name)
            && let Value::Number(n) = value
        {
            value = Value::Number(self.clamp_volume_size(n));
        }

        let old_value = self.global_uniforms.get_or_undefined(name).clone();
        self.global_uniforms.insert(name, value.clone());

        if name == "palette"
            && let Value::Number(index) = value
            && let Some(expanded) = noisemaker_dsl::palette::expand_palette(index)
                .map_err(|e| RenderError::Js(e.to_string()))?
        {
            for (u_name, u_value) in expanded {
                self.set_uniform(&u_name, u_value)?;
            }
            return Ok(());
        }

        let is_scoped = JsRegex::new(r"_node_\d+$", "").test(name)
            || JsRegex::new(r"_chain_\d+$", "").test(name);
        for p in self.graph.passes.iter_mut() {
            let Some(uniforms) = pass::uniforms_mut(p) else {
                continue;
            };
            if uniforms.contains_key(name) && !is_automation_value(uniforms.get_or_undefined(name))
            {
                uniforms.insert(name, value.clone());
            }
            if !is_scoped {
                let node_prefix = format!("{name}_node_");
                let chain_prefix = format!("{name}_chain_");
                let keys: Vec<String> = uniforms.keys().cloned().collect();
                for key in keys {
                    if (key.starts_with(&node_prefix) || key.starts_with(&chain_prefix))
                        && !is_automation_value(uniforms.get_or_undefined(&key))
                    {
                        uniforms.insert(key.clone(), value.clone());
                        self.global_uniforms.insert(key, value.clone());
                    }
                }
            }
        }

        if !strict_equals(&old_value, &value)
            && let Some(textures) = &self.graph.textures
        {
            let affects = textures.values().any(|spec| {
                let w = spec.get_or_undefined("width");
                let h = spec.get_or_undefined("height");
                let d = spec.get_or_undefined("depth");
                Self::dimension_references_param(w, name)
                    || Self::dimension_references_param(h, name)
                    || (d.is_truthy() && Self::dimension_references_param(d, name))
                    || Self::dimension_references_scoped_param(w, name)
                    || Self::dimension_references_scoped_param(h, name)
                    || (d.is_truthy() && Self::dimension_references_scoped_param(d, name))
            });
            if affects {
                let mut merged = self.global_uniforms.clone();
                merged.assign(&self.collect_default_uniforms());
                self.update_parameter_textures(&merged)?;
            }
        }
        Ok(())
    }

    /// `broadcastChainScopedParam(sourcePass, uniformName, scopedName)`.
    pub fn broadcast_chain_scoped_param(
        &mut self,
        source: usize,
        uniform_name: &str,
        scoped_name: &str,
    ) {
        let Some(source_uniforms) = pass::uniforms(&self.graph.passes[source]) else {
            return;
        };
        let mut value = source_uniforms.get_or_undefined(uniform_name).clone();
        if uniform_name == "volumeSize"
            && let Value::Number(n) = value
        {
            let clamped = self.clamp_volume_size(n);
            if clamped != n {
                value = Value::Number(clamped);
                let uniforms = pass::uniforms_mut(&mut self.graph.passes[source]).unwrap();
                uniforms.insert(uniform_name, value.clone());
                if uniforms.contains_key(scoped_name) {
                    uniforms.insert(scoped_name, value.clone());
                }
            }
        }
        for (i, other) in self.graph.passes.iter_mut().enumerate() {
            if i == source {
                continue;
            }
            let inherits = pass::get(other, "inheritsVolumeSize").is_truthy();
            let Some(uniforms) = pass::uniforms_mut(other) else {
                continue;
            };
            if !uniforms.contains_key(scoped_name) {
                continue;
            }
            uniforms.insert(scoped_name, value.clone());
            if uniform_name == "volumeSize" && inherits && uniforms.contains_key(uniform_name) {
                uniforms.insert(uniform_name, value.clone());
            }
        }
    }

    /// `dimensionReferencesParam(spec, paramName)`.
    pub fn dimension_references_param(spec: &Value, name: &str) -> bool {
        matches!(spec, Value::Object(_) | Value::Array(_))
            && (spec.get("param").as_str() == Some(name)
                || spec.get("screenDivide").as_str() == Some(name))
    }

    /// `dimensionReferencesScopedParam(spec, paramName)`.
    pub fn dimension_references_scoped_param(spec: &Value, name: &str) -> bool {
        if !matches!(spec, Value::Object(_) | Value::Array(_)) {
            return false;
        }
        let reference = match (spec.get("param"), spec.get("screenDivide")) {
            (Value::String(s), _) => s.as_str(),
            (_, Value::String(s)) => s.as_str(),
            _ => return false,
        };
        reference.starts_with(&format!("{name}_node_"))
            || reference.starts_with(&format!("{name}_chain_"))
    }

    /// `resolveDimension(spec, screenSize, uniforms)`: numbers, `'screen'`-like
    /// keywords, percentages, `{param, paramDefault, multiply, power, default}`,
    /// `{screenDivide, default}` and `{scale, clamp}`; anything else falls back to
    /// the screen size with an `ERR_DIMENSION_FALLBACK` diagnostic.
    pub fn resolve_dimension(
        &mut self,
        spec: &Value,
        screen: f64,
        uniforms: &Object,
    ) -> Result<f64, RenderError> {
        if let Value::Number(n) = spec {
            return Ok(js_max(1.0, n.floor()));
        }
        if matches!(
            spec.as_str(),
            Some("screen" | "auto" | "input" | "resolution")
        ) {
            return Ok(screen);
        }
        if let Some(s) = spec.as_str()
            && s.ends_with('%')
        {
            let percent = parse_float(s);
            return Ok(js_max(1.0, (screen * percent / 100.0).floor()));
        }
        if spec.is_null() {
            return Err(RenderError::type_error(
                "Cannot read properties of null (reading 'param')",
            ));
        }
        if matches!(spec, Value::Object(_) | Value::Array(_)) {
            let param = spec.get("param");
            if !param.is_undefined() {
                let has_transform =
                    !spec.get("power").is_undefined() || !spec.get("multiply").is_undefined();
                let param_default = if spec.get("paramDefault").is_nullish() {
                    Value::Number(64.0)
                } else {
                    spec.get("paramDefault").clone()
                };
                let key = to_js_string(param);
                let looked_up = uniforms.get_or_undefined(&key);
                let value = if looked_up.is_nullish() {
                    param_default
                } else {
                    looked_up.clone()
                };
                let mut value = value;
                if !spec.get("multiply").is_undefined() {
                    value = Value::Number(to_number(&value) * to_number(spec.get("multiply")));
                }
                if !spec.get("power").is_undefined() {
                    value = Value::Number(js_pow(to_number(&value), to_number(spec.get("power"))));
                }
                if has_transform && looked_up.is_undefined() && !spec.get("default").is_undefined()
                {
                    value = spec.get("default").clone();
                }
                return Ok(js_max(1.0, to_number(&value).floor()));
            }
            let screen_divide = spec.get("screenDivide");
            if !screen_divide.is_undefined() {
                let looked_up = uniforms.get_or_undefined(&to_js_string(screen_divide));
                let divisor = if !looked_up.is_nullish() {
                    looked_up.clone()
                } else if !spec.get("default").is_nullish() {
                    spec.get("default").clone()
                } else {
                    Value::Number(1.0)
                };
                return Ok(js_max(1.0, math_round(screen / to_number(&divisor))));
            }
            let scale = spec.get("scale");
            if !scale.is_undefined() {
                let mut computed = (screen * to_number(scale)).floor();
                let clamp = spec.get("clamp");
                if clamp.is_truthy() {
                    if !clamp.get("min").is_undefined() {
                        computed = js_max(to_number(clamp.get("min")), computed);
                    }
                    if !clamp.get("max").is_undefined() {
                        computed = js_min(to_number(clamp.get("max")), computed);
                    }
                }
                return Ok(js_max(1.0, computed));
            }
        }
        if !spec.is_nullish() {
            let key = if matches!(spec, Value::Object(_) | Value::Array(_)) {
                spec.to_json().unwrap_or_else(|| "[unserializable]".into())
            } else {
                to_js_string(spec)
            };
            if self.warned_dimension_fallbacks.insert(key.clone()) {
                let mut record = Object::new();
                record.insert("code", Value::from(codes::DIMENSION_FALLBACK));
                record.insert("backend", Value::from(self.backend.get_name()));
                record.insert("stage", Value::from("dimension"));
                record.insert("spec", Value::from(key));
                record.insert("fallback", Value::from("screen"));
                self.diagnostics.add(Value::Object(record));
            }
        }
        Ok(screen)
    }

    /// `syncTime(time)`.
    pub fn sync_time(&mut self, time: f64) {
        self.last_time = time;
    }

    /// `render(time)`: execute one frame at normalized time `time`.
    pub fn render(&mut self, time: f64) -> Result<(), RenderError> {
        if self.is_compiling {
            return Ok(());
        }
        self.run_due_async_regens()?;
        let mut delta = if self.last_time > 0.0 {
            time - self.last_time
        } else {
            0.0
        };
        if delta < 0.0 {
            delta = 1.0 / 60.0 / 10.0;
        }
        self.last_time = time;
        self.update_global_uniforms(time, delta);
        self.invoke_update_hooks(time, delta);

        self.frame_read_textures.clear();
        self.frame_write_textures.clear();
        for (name, surface) in &self.surfaces {
            self.frame_read_textures
                .insert(name.clone(), surface.read.clone());
            self.frame_write_textures
                .insert(name.clone(), surface.write.clone());
        }

        self.backend.begin_frame();
        let context = AutomationContext::now();
        let mut pass_count = 0usize;
        for i in 0..self.graph.passes.len() {
            if let Err(e) = self.render_pass(i, time, &context, &mut pass_count) {
                self.backend.abandon_frame();
                return Err(e);
            }
        }

        if !self.mip_targets.is_empty() {
            let targets = self.mip_targets.clone();
            self.backend.generate_mipmaps(&targets);
        }
        self.backend.end_frame();

        if let Some(name) = self.graph.render_surface_name().map(str::to_owned)
            && let Some(surface) = self.surfaces.get(&name)
        {
            let present = match self.frame_read_textures.get(&name) {
                Some(Some(id)) => Some(id.clone()),
                _ => surface.read.clone(),
            };
            if let Some(present) = present {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs_f64() * 1000.0)
                    .unwrap_or(0.0);
                self.sink_manager
                    .submit(&mut self.backend, &present, timestamp);
                self.last_presented = Some(present);
            }
        }
        self.swap_buffers();
        self.last_pass_count = pass_count;
        self.frame_index += 1.0;
        Ok(())
    }

    /// One iteration of `render()`'s pass loop.
    fn render_pass(
        &mut self,
        index: usize,
        time: f64,
        context: &AutomationContext,
        pass_count: &mut usize,
    ) -> Result<(), RenderError> {
        let has_viewport = !pass::get(&self.graph.passes[index], "viewport").is_undefined();
        if has_viewport {
            self.resolve_pass_viewport(index, false)?;
        }
        let uses_proxy = self.resolve_pass_uniforms(index, time, context);
        if has_viewport && uses_proxy {
            self.resolve_pass_viewport(index, true)?;
        } else if has_viewport {
            self.resolve_pass_viewport(index, false)?;
        }
        let mut current: Object = if uses_proxy {
            self.oscillator_pass_proxy.clone()
        } else {
            self.graph.passes[index].clone()
        };
        if self.has_runtime_uniforms && !current.get_or_undefined("effectKey").is_undefined() {
            let key = to_js_string(current.get_or_undefined("effectKey"));
            if let Some(runtime) = self.runtime_uniforms.get(&key) {
                current = Self::with_runtime_uniforms(&current, runtime);
            }
        }
        if self.should_skip_pass(&current) {
            return Ok(());
        }
        let repeat = self.resolve_repeat_count(&current);
        let mut iter = 0.0;
        while iter < repeat {
            let state = self.get_frame_state();
            self.backend.execute_pass(&current, &state)?;
            *pass_count += 1;
            self.update_frame_surface_bindings(&current, &state);
            if repeat > 1.0 {
                self.adopt_iteration_bindings(&current);
            }
            iter += 1.0;
        }
        Ok(())
    }

    /// `updateGlobalUniforms(time, deltaTime)`.
    pub fn update_global_uniforms(&mut self, time: f64, delta: f64) {
        let aspect = self.width / self.height;
        let g = &mut self.global_uniforms;
        g.insert("time", Value::Number(time));
        g.insert("deltaTime", Value::Number(delta));
        g.insert("frame", Value::Number(self.frame_index));
        let pair = |a: f64, b: f64| Value::Array(vec![Value::Number(a), Value::Number(b)]);
        let set_pair = |g: &mut Object, key: &str, a: f64, b: f64| {
            let existing = g.get_or_undefined(key);
            if !existing.is_truthy() {
                g.insert(key, pair(a, b));
            } else if let Some(Value::Array(items)) = g.get_mut(key) {
                if items.is_empty() {
                    items.push(Value::Number(a));
                } else {
                    items[0] = Value::Number(a);
                }
                if items.len() < 2 {
                    items.push(Value::Number(b));
                } else {
                    items[1] = Value::Number(b);
                }
            }
        };
        set_pair(g, "resolution", self.width, self.height);
        if !g.get_or_undefined("tileOffset").is_truthy() {
            g.insert("tileOffset", pair(0.0, 0.0));
        }
        if !g.get_or_undefined("fullResolution").is_truthy() {
            g.insert("fullResolution", pair(self.width, self.height));
        }
        let (tx, ty) = self.tile_offset.map(|t| (t[0], t[1])).unwrap_or((0.0, 0.0));
        set_pair(g, "tileOffset", tx, ty);
        if let Some(full) = self.full_resolution {
            set_pair(g, "fullResolution", full[0], full[1]);
            let full_aspect = full[0] / full[1];
            g.insert("aspect", Value::Number(full_aspect));
            g.insert("aspectRatio", Value::Number(full_aspect));
        } else {
            set_pair(g, "fullResolution", self.width, self.height);
            g.insert("aspect", Value::Number(aspect));
            g.insert("aspectRatio", Value::Number(aspect));
        }
        let scale = match self.render_scale {
            Some(s) if s != 0.0 && !s.is_nan() => s,
            _ => 1.0,
        };
        g.insert("renderScale", Value::Number(scale));

        // Typed arrays (Float32Array) are objects, not arrays, to the packers.
        let typed_array = |data: &[f32]| {
            Value::Object(
                data.iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), Value::Number(*v as f64)))
                    .collect(),
            )
        };
        if let Some(audio) = &self.external_state.audio {
            if let Some(w) = audio.waveform() {
                g.insert("audioWaveform", typed_array(w));
            }
            if let Some(s) = audio.spectrum() {
                g.insert("audioSpectrum", typed_array(s));
            }
        }
        if let Some(midi) = self.external_state.midi.as_mut() {
            midi.update_note_grid();
            let grid = midi.note_grid().to_vec();
            let clock = midi.clock_count();
            self.backend
                .upload_data_texture("midiNoteGrid", &grid, 128, 16);
            self.global_uniforms
                .insert("midiClockCount", Value::Number(clock));
        } else if self.needs_midi_note_grid {
            let grid = self
                .empty_note_grid
                .get_or_insert_with(|| vec![0.0; 128 * 16 * 4])
                .clone();
            self.backend
                .upload_data_texture("midiNoteGrid", &grid, 128, 16);
        }
        let clock = self.global_uniforms.get_or_undefined("midiClockCount");
        let clock = if clock.is_truthy() {
            clock.clone()
        } else {
            Value::Number(0.0)
        };
        self.global_uniforms.insert("midiClockCount", clock);
    }

    /// `resolvePassUniforms(pass, time)`: evaluate automated uniforms; when any
    /// is automated, the reused proxy (with a fixed subset of the pass's fields
    /// and the resolved uniforms) stands in for the pass. Returns whether the
    /// proxy is used.
    fn resolve_pass_uniforms(
        &mut self,
        index: usize,
        time: f64,
        context: &AutomationContext,
    ) -> bool {
        let original = &self.graph.passes[index];
        let Some(uniforms) = pass::uniforms(original) else {
            return false;
        };
        if !pass::get(original, "uniforms").is_truthy() {
            return false;
        }
        // Clear (to undefined, keeping the keys) and refill.
        for value in self.resolved_uniforms.values_mut() {
            *value = Value::Undefined;
        }
        let specs = pass::get(original, "uniformSpecs");
        let mut has_oscillators = false;
        for (name, value) in uniforms.iter() {
            let spec = specs.get(name);
            match resolve_uniform_value(value, time, spec, &self.external_state, context) {
                Some(resolved) => {
                    self.resolved_uniforms.insert(name.clone(), resolved);
                    has_oscillators = true;
                }
                None => {
                    self.resolved_uniforms.insert(name.clone(), value.clone());
                    // `resolved !== value` is true for NaN.
                    if matches!(value, Value::Number(n) if n.is_nan()) {
                        has_oscillators = true;
                    }
                }
            }
        }
        if !has_oscillators {
            return false;
        }
        let proxy = &mut self.oscillator_pass_proxy;
        for key in [
            "id",
            "program",
            "inputs",
            "outputs",
            "clear",
            "blend",
            "drawMode",
            "count",
            "repeat",
            "conditions",
            "viewport",
            "viewportResolved",
            "drawBuffers",
            "storageTextures",
            "samplerTypes",
            "entryPoint",
        ] {
            proxy.insert(key, original.get_or_undefined(key).clone());
        }
        let proxy_uniforms = match proxy.get_or_undefined("uniforms") {
            Value::Object(o) => o.clone(),
            _ => Object::new(),
        };
        proxy.insert(
            "uniforms",
            Value::Object(std::mem::take(&mut self.resolved_uniforms)),
        );
        self.resolved_uniforms = proxy_uniforms;
        true
    }

    /// `shouldSkipPass(pass)`: `skipIf` (any match skips) and `runIf` (any
    /// mismatch skips) conditions on pass or global uniforms.
    pub fn should_skip_pass(&self, p: &Object) -> bool {
        let conditions = pass::get(p, "conditions");
        if !conditions.is_truthy() {
            return false;
        }
        let uniforms = pass::get(p, "uniforms");
        let lookup = |name: &Value| -> Value {
            let key = to_js_string(name);
            let v = uniforms.get(&key);
            if v.is_nullish() {
                self.global_uniforms.get_or_undefined(&key).clone()
            } else {
                v.clone()
            }
        };
        if let Some(skip_if) = conditions.get("skipIf").as_array()
            && conditions.get("skipIf").is_truthy()
        {
            for condition in skip_if {
                if strict_equals(&lookup(condition.get("uniform")), condition.get("equals")) {
                    return true;
                }
            }
        }
        if let Some(run_if) = conditions.get("runIf").as_array()
            && conditions.get("runIf").is_truthy()
        {
            for condition in run_if {
                if !strict_equals(&lookup(condition.get("uniform")), condition.get("equals")) {
                    return true;
                }
            }
        }
        false
    }

    /// `resolveRepeatCount(pass)` (NaN yields zero iterations, as the reference's
    /// `iter < NaN` loop does).
    pub fn resolve_repeat_count(&self, p: &Object) -> f64 {
        let repeat = pass::get(p, "repeat");
        if !repeat.is_truthy() {
            return 1.0;
        }
        if let Value::Number(n) = repeat {
            return js_max(1.0, n.floor());
        }
        if let Value::String(name) = repeat {
            let global = self.global_uniforms.get_or_undefined(name);
            let value = if global.is_nullish() {
                pass::get(p, "uniforms").get(name).clone()
            } else {
                global.clone()
            };
            if let Value::Number(n) = value {
                return js_max(1.0, n.floor());
            }
        }
        1.0
    }

    /// `resolvePassViewport(pass, cacheHolder)`: resolve an authored viewport spec
    /// to `{x, y, w, h}` on `viewportResolved` (of the original pass, and of the
    /// proxy when `on_proxy`).
    fn resolve_pass_viewport(&mut self, index: usize, on_proxy: bool) -> Result<(), RenderError> {
        let spec = pass::get(&self.graph.passes[index], "viewport").clone();
        if !matches!(spec, Value::Object(_) | Value::Array(_)) {
            return Ok(());
        }
        let mut cache = self.viewport_cache.remove(&index).unwrap_or_default();
        let result = (|| {
            if cache.spec_source.as_ref() != Some(&spec) {
                cache.spec_source = Some(spec.clone());
                let numeric = ["x", "y", "w", "h"]
                    .iter()
                    .all(|k| matches!(spec.get(k), Value::Number(_)));
                if numeric {
                    cache.numeric = true;
                    cache.resolved_is_spec = true;
                    self.graph.passes[index].insert("viewportResolved", spec.clone());
                    if on_proxy {
                        self.oscillator_pass_proxy
                            .insert("viewportResolved", spec.clone());
                    }
                    return Ok(());
                }
                cache.numeric = false;
            }
            if cache.resolved_is_spec {
                if on_proxy {
                    self.oscillator_pass_proxy
                        .insert("viewportResolved", spec.clone());
                }
                return Ok(());
            }
            let uniforms = if on_proxy {
                pass::uniforms(&self.oscillator_pass_proxy)
                    .cloned()
                    .unwrap_or_default()
            } else {
                pass::uniforms(&self.graph.passes[index])
                    .cloned()
                    .unwrap_or_default()
            };
            let or_zero = |v: &Value| {
                if v.is_nullish() {
                    Value::Number(0.0)
                } else {
                    v.clone()
                }
            };
            let x = self.resolve_dimension(&or_zero(spec.get("x")), self.width, &uniforms)?;
            let y = self.resolve_dimension(&or_zero(spec.get("y")), self.height, &uniforms)?;
            let width_spec = if spec.get("w").is_nullish() {
                spec.get("width")
            } else {
                spec.get("w")
            };
            let height_spec = if spec.get("h").is_nullish() {
                spec.get("height")
            } else {
                spec.get("h")
            };
            let w = if width_spec.is_undefined() {
                self.width
            } else {
                self.resolve_dimension(width_spec, self.width, &uniforms)?
            };
            let h = if height_spec.is_undefined() {
                self.height
            } else {
                self.resolve_dimension(height_spec, self.height, &uniforms)?
            };
            let mut box_ = Object::new();
            box_.insert("x", Value::Number(x));
            box_.insert("y", Value::Number(y));
            box_.insert("w", Value::Number(w));
            box_.insert("h", Value::Number(h));
            self.graph.passes[index].insert("viewportResolved", Value::Object(box_.clone()));
            if on_proxy {
                self.oscillator_pass_proxy
                    .insert("viewportResolved", Value::Object(box_));
            }
            Ok(())
        })();
        self.viewport_cache.insert(index, cache);
        result
    }

    /// `adoptIterationBindings(pass)`: mirror a repeated pass's frame-local
    /// ping-pong bindings into the cross-frame surface records.
    fn adopt_iteration_bindings(&mut self, p: &Object) {
        let Some(outputs) = pass::outputs(p) else {
            return;
        };
        for output in outputs.values() {
            if output.as_str().is_none() {
                continue;
            }
            let Some(name) = parse_global_name(output) else {
                continue;
            };
            let Some(surface) = self.surfaces.get_mut(&name) else {
                continue;
            };
            if let Some(read) = self.frame_read_textures.get(&name)
                && read.is_some()
            {
                surface.read = read.clone();
            }
            if let Some(write) = self.frame_write_textures.get(&name)
                && write.is_some()
            {
                surface.write = write.clone();
            }
        }
    }

    /// `swapBuffers()`: state surfaces keep the frame's final bindings; display
    /// surfaces swap.
    fn swap_buffers(&mut self) {
        let frame = self.frame_index;
        for (name, surface) in self.surfaces.iter_mut() {
            surface.current_frame = Some(frame);
            if is_state_surface(name) {
                let final_read = self.frame_read_textures.get(name).cloned().flatten();
                let final_write = self.frame_write_textures.get(name).cloned().flatten();
                if let (Some(r), Some(w)) = (final_read, final_write)
                    && !r.is_empty()
                    && !w.is_empty()
                {
                    surface.read = Some(r);
                    surface.write = Some(w);
                }
            } else {
                std::mem::swap(&mut surface.read, &mut surface.write);
            }
        }
    }

    /// `getFrameState()`.
    pub fn get_frame_state(&self) -> FrameState {
        let mut state = FrameState {
            frame_index: self.frame_index,
            time: self.last_time,
            global_uniforms: self.global_uniforms.clone(),
            render_surface: self.graph.render_surface.clone(),
            screen_width: self.width,
            screen_height: self.height,
            ..Default::default()
        };
        for (name, surface) in &self.surfaces {
            let read = match self.frame_read_textures.get(name) {
                Some(Some(id)) => Some(id.clone()),
                _ => surface.read.clone(),
            };
            if let Some(read) = read
                && let Some(tex) = self.backend.textures.get(&read)
            {
                state.surfaces.insert(name.clone(), tex.clone());
            }
            let write = match self.frame_write_textures.get(name) {
                Some(Some(id)) => Some(id.clone()),
                _ => surface.write.clone(),
            };
            state.write_surfaces.insert(
                name.clone(),
                write.map(Value::from).unwrap_or(Value::Undefined),
            );
        }
        state
    }

    /// `getOutput(surfaceName)`: the read texture record of a surface (default: the
    /// render surface).
    pub fn get_output(
        &self,
        surface_name: Option<&str>,
    ) -> Option<Rc<crate::backend::TextureRecord>> {
        let name = surface_name
            .map(str::to_owned)
            .or_else(|| self.graph.render_surface_name().map(str::to_owned))?;
        let surface = self.surfaces.get(&name)?;
        self.backend.textures.get(surface.read.as_ref()?).cloned()
    }

    /// `clearSurface(surfaceName)`: clear both halves of a surface.
    pub fn clear_surface(&mut self, surface_name: &str) {
        let Some(surface) = self.surfaces.get(surface_name).cloned() else {
            return;
        };
        if let Some(read) = &surface.read {
            self.backend.clear_texture(read);
        }
        if let Some(write) = &surface.write {
            self.backend.clear_texture(write);
        }
    }

    /// `updateFrameSurfaceBindings(pass, state)`: after a pass writes a global
    /// surface, later passes read the fresh texture and write the other one.
    fn update_frame_surface_bindings(&mut self, p: &Object, state: &FrameState) {
        let Some(outputs) = pass::outputs(p) else {
            return;
        };
        for output in outputs.values() {
            if output.as_str().is_none() {
                continue;
            }
            let Some(name) = parse_global_name(output) else {
                continue;
            };
            let Some(write_id) = state.write_surface(&name) else {
                continue;
            };
            let current_read = self.frame_read_textures.get(&name).cloned().flatten();
            self.frame_read_textures
                .insert(name.clone(), Some(write_id));
            if let Some(read) = current_read
                && !read.is_empty()
            {
                self.frame_write_textures.insert(name, Some(read));
            }
        }
    }

    /// `preflight(capabilities)`.
    pub fn preflight(&self, capabilities: Option<&Capabilities>) -> PreflightReport {
        let caps = capabilities
            .cloned()
            .unwrap_or_else(|| self.get_capabilities());
        let mut shaders: IndexMap<String, Value> = IndexMap::new();
        for p in &self.graph.passes {
            if !pass::get(p, "program").is_truthy() {
                continue;
            }
            if let Some(spec) = self.resolve_program_spec(p) {
                shaders.insert(pass::program(p), spec.clone());
            }
        }
        preflight_effect(
            &self.graph.passes,
            self.graph.textures.as_ref(),
            &caps,
            if shaders.is_empty() {
                None
            } else {
                Some(&shaders)
            },
        )
    }

    /// `dispose()`: run `onDestroy` hooks, cancel async work, close sinks and
    /// release every backend resource.
    pub fn dispose(&mut self) -> Result<(), RenderError> {
        if self.disposed {
            return Ok(());
        }
        self.disposed = true;
        let mut first_error: Option<RenderError> = None;
        for lifecycle in self.lifecycle_effects.values() {
            if !lifecycle.borrow().has_on_destroy() {
                continue;
            }
            if let Err(e) = lifecycle.borrow_mut().on_destroy() {
                first_error.get_or_insert(RenderError::Js(e));
            }
        }
        self.lifecycle_effects.clear();
        for cancelled in self.async_renders.values() {
            *cancelled.borrow_mut() = true;
        }
        self.async_renders.clear();
        self.async_debounce.clear();
        if let Err(e) = self.sink_manager.close() {
            first_error.get_or_insert(RenderError::Js(e));
        }
        let ids: Vec<String> = self.backend.textures.keys().cloned().collect();
        for id in ids {
            self.backend.destroy_texture(&id);
        }
        self.backend.destroy(true);
        self.surfaces.clear();
        self.graph = Graph::default();
        self.frame_read_textures.clear();
        self.frame_write_textures.clear();
        self.global_uniforms = Object::new();
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
