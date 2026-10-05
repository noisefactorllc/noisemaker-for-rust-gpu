//! The host layer: a port of the reference's `CanvasRenderer`
//! (`shaders/src/renderer/canvas.js`) without its DOM duties (canvas
//! element, `requestAnimationFrame`, WebGL context loss, lazy effect loading).
//!
//! A [`CanvasRenderer`] owns the effect registries, the device and the live
//! [`Pipeline`]. It compiles DSL ([`CanvasRenderer::compile`]: the first
//! program creates the pipeline; later programs hot-swap the graph into it,
//! falling back to a fresh pipeline when that fails), renders frames at a
//! normalized loop time, applies effect parameters to pass uniforms
//! ([`CanvasRenderer::apply_parameter_values`],
//! [`CanvasRenderer::apply_step_parameter_values`]), uploads host media and
//! meshes, renders cubemaps, registers output sinks, creates frame export
//! queues and carries the MIDI and audio states into every pipeline it
//! creates. It implements [`ProgramHost`], so a
//! [`noisemaker_dsl::program_state::ProgramState`] can drive it as the
//! reference's ProgramState drives a CanvasRenderer.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use noisemaker_dsl::compiler::{CompileOptions as DslCompileOptions, compile_graph};
use noisemaker_dsl::palette::expand_palette_value;
use noisemaker_dsl::program_state::{
    ProgramHost, convert_parameter_for_uniform, resolve_enum_value, write_uniform_aliases,
};
use noisemaker_dsl::registry::EffectEntry;
use noisemaker_dsl::{JsError, Object, Registry, Value};
use noisemaker_host::obj::{PackedMesh, decode_obj_text, pack_mesh, parse_obj};

use crate::automation::{AudioState, MidiState, SharedAudioState, SharedMidiState};
use crate::backend::{Capabilities, GpuDevice, PixelData, WebGpuBackend};
use crate::effects::{MEDIA_EFFECT_KEY, MediaLifecycle, native_effects_with};
use crate::error::RenderError;
use crate::frame_export::{
    FrameExportError, FrameExportOptions, FrameExportQueue, WebGpuFrameExportAdapter,
};
use crate::graph::{Graph, pass};
use crate::hooks::EffectRegistry;
use crate::jsv::{strict_equals, to_js_string};
use crate::pipeline::{Pipeline, PipelineOptions};
use crate::sink::{Sink, SinkId};

/// Width and height of the mesh textures (`_packCacheAndUploadMesh`).
pub const MESH_TEXTURE_SIZE: u32 = 256;

/// Options of [`CanvasRenderer::new`].
pub struct CanvasRendererOptions {
    /// Render width (`options.width`, default 1024).
    pub width: u32,
    /// Render height (`options.height`, default 1024).
    pub height: u32,
    /// The DSL registries (`registerEffect`, ops, enums); default: the
    /// embedded catalog, every effect loaded.
    pub registry: Option<Rc<Registry>>,
    /// The native effect hooks; default: [`crate::effects::native_effects`]
    /// (with the renderer's synth/media hook object).
    pub effects: Option<EffectRegistry>,
    /// The pipeline's `texturePooling` option.
    pub texture_pooling: bool,
}

impl Default for CanvasRendererOptions {
    fn default() -> Self {
        CanvasRendererOptions {
            width: 1024,
            height: 1024,
            registry: None,
            effects: None,
            texture_pooling: false,
        }
    }
}

/// Options of [`CanvasRenderer::compile`].
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// `options.shaderOverrides`: per-step shader overrides keyed by step index.
    pub shader_overrides: Object,
}

/// One entry of `buildUniformBindings`: a pass uniform fed by a parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UniformBinding {
    pub pass_index: usize,
    pub uniform_name: String,
}

/// `getMediaSteps()` entry: a step's external texture binding.
#[derive(Debug, Clone, PartialEq)]
pub struct MediaStep {
    /// `'<externalTexture>_step_N'`.
    pub texture_id: String,
    /// The pass input the texture binds to (`imageTex`).
    pub uniform: String,
    pub step_index: f64,
    /// The effect key (`synth.media`).
    pub effect: String,
}

/// The result of a mesh load (`{success, vertexCount, error?}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeshLoadResult {
    pub success: bool,
    pub vertex_count: usize,
    pub error: Option<String>,
}

impl MeshLoadResult {
    fn failed(error: impl Into<String>) -> MeshLoadResult {
        MeshLoadResult {
            success: false,
            vertex_count: 0,
            error: Some(error.into()),
        }
    }
}

/// Options of [`CanvasRenderer::update_texture_from_source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TextureUpdateOptions {
    /// `flipY` (default `true`).
    pub flip_y: bool,
}

impl Default for TextureUpdateOptions {
    fn default() -> Self {
        TextureUpdateOptions { flip_y: true }
    }
}

/// `isAutomationControlled(value)`: an oscillator, MIDI or audio binding (or
/// a `_varRef` to one) that UI values must not overwrite.
pub fn is_automation_controlled(value: &Value) -> bool {
    if !value.is_truthy() || !matches!(value, Value::Object(_) | Value::Array(_)) {
        return false;
    }
    if value.get("_varRef").is_truthy() {
        return true;
    }
    let ty = value.get("type");
    let ty = if ty.is_truthy() {
        ty.clone()
    } else {
        value.get("_ast").get("type").clone()
    };
    matches!(ty.as_str(), Some("Oscillator" | "Midi" | "Audio"))
}

/// `Array.isArray(value) ? value.slice() : value`.
fn slice_copy(value: &Value) -> Value {
    value.clone()
}

fn js_error(e: JsError) -> RenderError {
    match e {
        JsError::Error { name, message } => RenderError::Js(format!("{name}: {message}")),
        JsError::Thrown(v) => RenderError::Thrown(v),
    }
}

fn render_to_js(e: RenderError) -> JsError {
    match e {
        RenderError::Thrown(v) => JsError::Thrown(v),
        other => JsError::error(other.to_string()),
    }
}

/// `writeUniformAliases(pass, paramName, uniformName, value)` on a graph pass.
fn write_pass_uniform_aliases(
    pass: &mut Object,
    param_name: &str,
    uniform_name: &str,
    value: &Value,
) -> Result<bool, RenderError> {
    if !pass.get_or_undefined("uniformAliases").is_truthy() {
        return Ok(false);
    }
    let mut wrapped = Value::Object(std::mem::take(pass));
    let result = write_uniform_aliases(&mut wrapped, param_name, &Value::from(uniform_name), value);
    if let Value::Object(o) = wrapped {
        *pass = o;
    }
    result.map_err(js_error)
}

/// The reference CanvasRenderer, without its DOM duties.
pub struct CanvasRenderer {
    device: GpuDevice,
    registry: Rc<Registry>,
    enums: Rc<Value>,
    effects: EffectRegistry,
    media: Rc<RefCell<MediaLifecycle>>,
    texture_pooling: bool,
    pipeline: Option<Pipeline>,
    width: u32,
    height: u32,
    current_dsl: String,
    loop_duration: f64,
    loop_start: Instant,
    frame_count: u64,
    deferred_frame_count: u64,
    last_pass_count: usize,
    last_render_time: Duration,
    uniform_bindings: IndexMap<String, Vec<UniformBinding>>,
    midi_state: Option<SharedMidiState>,
    audio_state: Option<SharedAudioState>,
    /// `_meshCache`: packed meshes re-uploaded to every new pipeline.
    mesh_cache: IndexMap<String, PackedMesh>,
    /// The pipeline's passes as ProgramState sees them (`graph_passes`),
    /// moved out of the graph while a ProgramState operation runs and moved
    /// back before the pipeline is used.
    program_passes: Option<Vec<Value>>,
}

impl CanvasRenderer {
    /// `new CanvasRenderer(options)` on `device`.
    pub fn new(device: &GpuDevice, options: CanvasRendererOptions) -> CanvasRenderer {
        let registry = options
            .registry
            .unwrap_or_else(|| Rc::new(Registry::with_catalog()));
        let media = Rc::new(RefCell::new(MediaLifecycle::default()));
        let effects = options
            .effects
            .unwrap_or_else(|| native_effects_with(media.clone()));
        CanvasRenderer {
            device: device.clone(),
            enums: Rc::new(Value::Object(registry.enums.clone())),
            registry,
            effects,
            media,
            texture_pooling: options.texture_pooling,
            pipeline: None,
            width: options.width,
            height: options.height,
            current_dsl: String::new(),
            loop_duration: 10.0,
            loop_start: Instant::now(),
            frame_count: 0,
            deferred_frame_count: 0,
            last_pass_count: 0,
            last_render_time: Duration::ZERO,
            uniform_bindings: IndexMap::new(),
            midi_state: None,
            audio_state: None,
            mesh_cache: IndexMap::new(),
            program_passes: None,
        }
    }

    // ---------------------------------------------------------------- state

    /// Move ProgramState's view of the passes back into the pipeline graph.
    fn flush_program_passes(&mut self) {
        if let Some(view) = self.program_passes.take()
            && let Some(pipeline) = self.pipeline.as_mut()
        {
            pipeline.graph.passes = view
                .into_iter()
                .map(|p| match p {
                    Value::Object(o) => o,
                    _ => Object::new(),
                })
                .collect();
        }
    }

    /// `renderer.pipeline`.
    pub fn pipeline(&mut self) -> Option<&Pipeline> {
        self.flush_program_passes();
        self.pipeline.as_ref()
    }

    /// `renderer.pipeline` (mutable).
    pub fn pipeline_mut(&mut self) -> Option<&mut Pipeline> {
        self.flush_program_passes();
        self.pipeline.as_mut()
    }

    /// The live pipeline's backend.
    pub fn backend(&self) -> Option<&WebGpuBackend> {
        self.pipeline.as_ref().map(|p| &p.backend)
    }

    /// The DSL registries.
    pub fn registry(&self) -> &Rc<Registry> {
        &self.registry
    }

    /// The native effect hooks.
    pub fn effects(&self) -> &EffectRegistry {
        &self.effects
    }

    /// `renderer.enums`.
    pub fn enums(&self) -> &Rc<Value> {
        &self.enums
    }

    /// The device.
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// `currentDsl`.
    pub fn current_dsl(&self) -> &str {
        &self.current_dsl
    }

    /// `currentDsl = dsl`.
    pub fn set_current_dsl(&mut self, dsl: impl Into<String>) {
        self.current_dsl = dsl.into();
    }

    /// The render width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The render height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// `frameCount`: frames rendered since the last compile.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// `deferredFrameCount`: loop frames skipped because a sink deferred.
    pub fn deferred_frame_count(&self) -> u64 {
        self.deferred_frame_count
    }

    /// `lastPassCount`.
    pub fn last_pass_count(&self) -> usize {
        self.last_pass_count
    }

    /// `lastRenderTime` of the last loop frame.
    pub fn last_render_time(&self) -> Duration {
        self.last_render_time
    }

    /// `loopDuration` in seconds.
    pub fn loop_duration(&self) -> f64 {
        self.loop_duration
    }

    /// `lastTime`: the last normalized time the pipeline rendered.
    pub fn last_time(&self) -> f64 {
        match &self.pipeline {
            Some(p) if p.last_time != 0.0 && !p.last_time.is_nan() => p.last_time,
            _ => 0.0,
        }
    }

    /// `capabilities`: the pipeline's, or the reference defaults without one.
    pub fn capabilities(&self) -> Capabilities {
        match &self.pipeline {
            Some(p) => p.get_capabilities(),
            None => Capabilities {
                is_mobile: false,
                float_blend: true,
                float_linear: true,
                color_buffer_float: true,
                max_draw_buffers: 8,
                max_texture_size: 4096,
                max_color_bytes_per_sample: 64,
                max_state_size: 2048,
            },
        }
    }

    // ------------------------------------------------------------ configuration

    /// `setLoopDuration(duration)`: also restarts the loop.
    pub fn set_loop_duration(&mut self, seconds: f64) {
        self.loop_duration = seconds;
        self.loop_start = Instant::now();
    }

    /// `setUniform(name, value)`.
    pub fn set_uniform(&mut self, name: &str, value: Value) -> Result<(), RenderError> {
        match self.pipeline_mut() {
            Some(p) => p.set_uniform(name, value),
            None => Ok(()),
        }
    }

    /// `setTileRegion({offset, fullResolution, renderScale})`.
    pub fn set_tile_region(
        &mut self,
        offset: [f64; 2],
        full_resolution: [f64; 2],
        render_scale: Option<f64>,
    ) {
        if let Some(p) = self.pipeline_mut() {
            p.set_tile_region(offset, full_resolution, render_scale);
        }
    }

    /// `clearTileRegion()`.
    pub fn clear_tile_region(&mut self) {
        if let Some(p) = self.pipeline_mut() {
            p.clear_tile_region();
        }
    }

    /// `renderCubemap({size, outputSurface, time})`: six faces +X, -X, +Y,
    /// -Y, +Z, -Z (empty without a pipeline).
    pub fn render_cubemap(
        &mut self,
        size: u32,
        output_surface: &str,
        time: f64,
    ) -> Result<Vec<PixelData>, RenderError> {
        match self.pipeline_mut() {
            Some(p) => p.render_cubemap(size as f64, output_surface, time),
            None => Ok(Vec::new()),
        }
    }

    /// `resize(width, height)`.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        self.width = width;
        self.height = height;
        match self.pipeline_mut() {
            Some(p) => p.resize(width as f64, height as f64),
            None => Ok(()),
        }
    }

    /// `setMidiState(midiState)`: the state `midi()` reads (a new one when
    /// `None`), carried into every pipeline this renderer creates.
    pub fn set_midi_state(&mut self, state: Option<SharedMidiState>) -> SharedMidiState {
        let state = state.unwrap_or_else(|| Rc::new(RefCell::new(MidiState::new())));
        self.midi_state = Some(state.clone());
        if let Some(p) = self.pipeline.as_mut() {
            p.set_midi_state(Some(state.clone()));
        }
        state
    }

    /// `midiState`.
    pub fn midi_state(&self) -> Option<&SharedMidiState> {
        self.midi_state.as_ref()
    }

    /// `setAudioState(audioState)`: the state `audio()` reads (a new one when
    /// `None`).
    pub fn set_audio_state(&mut self, state: Option<SharedAudioState>) -> SharedAudioState {
        let state = state.unwrap_or_else(|| Rc::new(RefCell::new(AudioState::new())));
        self.audio_state = Some(state.clone());
        if let Some(p) = self.pipeline.as_mut() {
            p.set_audio_state(Some(state.clone()));
        }
        state
    }

    /// `audioState`.
    pub fn audio_state(&self) -> Option<&SharedAudioState> {
        self.audio_state.as_ref()
    }

    /// synth/media's `setMediaDimensions(width, height)` (the dimensions its
    /// `onUpdate` reports where a pass does not set `imageSize`).
    pub fn set_media_dimensions(&mut self, width: f64, height: f64) {
        if let Some(hooks) = self.effects.get(MEDIA_EFFECT_KEY)
            && let Some(lifecycle) = &hooks.lifecycle
            && let Some(media) = lifecycle
                .borrow_mut()
                .as_any_mut()
                .downcast_mut::<MediaLifecycle>()
        {
            media.set_media_dimensions(width, height);
            return;
        }
        self.media.borrow_mut().set_media_dimensions(width, height);
    }

    // ------------------------------------------------------------- render loop

    /// `syncTime(normalizedTime)`.
    pub fn sync_time(&mut self, normalized_time: f64) {
        if let Some(p) = self.pipeline.as_mut() {
            p.sync_time(normalized_time);
        }
    }

    /// `render(normalizedTime)`: one frame at a loop time in 0..1.
    pub fn render(&mut self, normalized_time: f64) -> Result<(), RenderError> {
        self.flush_program_passes();
        let Some(p) = self.pipeline.as_mut() else {
            return Ok(());
        };
        p.render(normalized_time)?;
        self.last_pass_count = p.last_pass_count;
        self.frame_count += 1;
        Ok(())
    }

    /// The normalized loop time at `now` (`(elapsed % loopDuration) /
    /// loopDuration` since the loop started).
    pub fn normalized_time_at(&self, now: Instant) -> f64 {
        let elapsed = now.saturating_duration_since(self.loop_start).as_secs_f64();
        (elapsed % self.loop_duration) / self.loop_duration
    }

    /// One iteration of the render loop (`_renderLoop`) at `now`: skipped
    /// (counted as deferred) while a sink defers, otherwise a frame at the
    /// loop time.
    pub fn tick(&mut self, now: Instant) -> Result<(), RenderError> {
        self.flush_program_passes();
        let Some(p) = self.pipeline.as_mut() else {
            return Ok(());
        };
        if p.should_defer_render() {
            self.deferred_frame_count += 1;
            return Ok(());
        }
        let start = Instant::now();
        let t = self.normalized_time_at(now);
        let p = self.pipeline.as_mut().expect("checked above");
        p.render(t)?;
        self.last_render_time = start.elapsed();
        self.last_pass_count = p.last_pass_count;
        self.frame_count += 1;
        Ok(())
    }

    // ------------------------------------------------------------ lifecycle

    /// `dispose()`: dispose the pipeline (the renderer can compile again).
    pub fn dispose(&mut self) -> Result<(), RenderError> {
        self.flush_program_passes();
        self.uniform_bindings.clear();
        match self.pipeline.take() {
            Some(mut p) => p.dispose(),
            None => Ok(()),
        }
    }

    /// `createRuntime(dsl, options)`: compile and create a pipeline.
    fn create_runtime(&self, dsl: &str, options: &CompileOptions) -> Result<Pipeline, RenderError> {
        let graph = self.compile_graph(dsl, options)?;
        let mut pipeline = Pipeline::new(
            graph,
            self.device.backend(),
            PipelineOptions {
                texture_pooling: self.texture_pooling,
                effects: self.effects.clone(),
            },
        );
        pipeline.init(self.width as f64, self.height as f64)?;
        Ok(pipeline)
    }

    /// `compileGraph(dsl, {shaderOverrides})`.
    fn compile_graph(&self, dsl: &str, options: &CompileOptions) -> Result<Graph, RenderError> {
        let value = compile_graph(
            dsl,
            &self.registry,
            &DslCompileOptions {
                shader_overrides: options.shader_overrides.clone(),
            },
        )
        .map_err(js_error)?;
        Graph::from_value(&value).map_err(RenderError::Js)
    }

    /// `recompile(pipeline, newSource, options)` (runtime/compiler.js): swap
    /// the new graph into the pipeline, recreate its surfaces and textures,
    /// restart its asyncInit and lifecycle effects. `None` when any step
    /// fails (the reference logs and returns null).
    fn recompile(
        &self,
        pipeline: &mut Pipeline,
        dsl: &str,
        options: &CompileOptions,
    ) -> Option<()> {
        let result = (|| -> Result<(), RenderError> {
            let graph = self.compile_graph(dsl, options)?;
            pipeline.poll_async_effects()?;
            pipeline.swap_graph(graph);
            pipeline.create_surfaces()?;
            let defaults = pipeline.collect_default_uniforms();
            pipeline.recreate_textures(&defaults)?;
            pipeline.init_async_effects()?;
            pipeline.init_lifecycle_effects();
            Ok(())
        })();
        match result {
            Ok(()) => Some(()),
            Err(e) => {
                eprintln!("Recompilation failed: {e}");
                None
            }
        }
    }

    /// `compile(dsl, {shaderOverrides})`: the first program creates the
    /// pipeline (with this renderer's MIDI and audio states); later programs
    /// recompile into it, or replace it with a fresh pipeline when the
    /// recompile fails. Resets the frame count and uniform bindings and
    /// re-uploads cached meshes.
    pub fn compile(&mut self, dsl: &str, options: &CompileOptions) -> Result<(), RenderError> {
        self.flush_program_passes();
        self.current_dsl = dsl.to_owned();
        match self.pipeline.take() {
            None => {
                let mut pipeline = self.create_runtime(dsl, options)?;
                pipeline.set_midi_state(self.midi_state.clone());
                pipeline.set_audio_state(self.audio_state.clone());
                self.pipeline = Some(pipeline);
            }
            Some(mut pipeline) => {
                pipeline.is_compiling = true;
                if self.recompile(&mut pipeline, dsl, options).is_none() {
                    pipeline.is_compiling = false;
                    match self.create_runtime(dsl, options) {
                        Ok(mut replacement) => {
                            replacement.set_midi_state(self.midi_state.clone());
                            replacement.set_audio_state(self.audio_state.clone());
                            if let Err(e) = pipeline.dispose() {
                                eprintln!("Failed to dispose previous pipeline: {e}");
                            }
                            self.pipeline = Some(replacement);
                        }
                        Err(e) => {
                            self.pipeline = Some(pipeline);
                            return Err(e);
                        }
                    }
                } else {
                    let compiled = pipeline.compile_programs();
                    self.pipeline = Some(pipeline);
                    compiled?;
                }
            }
        }
        self.frame_count = 0;
        self.uniform_bindings.clear();
        self.reupload_cached_meshes();
        Ok(())
    }

    /// `_reuploadCachedMeshes()`.
    fn reupload_cached_meshes(&mut self) {
        let Some(p) = self.pipeline.as_mut() else {
            return;
        };
        for (mesh_id, mesh) in &self.mesh_cache {
            p.backend.upload_mesh_data(
                mesh_id,
                &mesh.position_data,
                &mesh.normal_data,
                &mesh.uv_data,
                mesh.width as u32,
                mesh.height as u32,
                mesh.vertex_count as u32,
            );
        }
    }

    // ---------------------------------------------------------------- sinks

    /// `addSink(sink)`: register an output sink on the active pipeline.
    pub fn add_sink(&mut self, sink: Box<dyn Sink>) -> Result<SinkId, RenderError> {
        let Some(p) = self.pipeline.as_mut() else {
            return Err(RenderError::Js(
                "Error: CanvasRenderer has no active pipeline; compile before adding a sink".into(),
            ));
        };
        p.sink_manager.add(sink).map_err(RenderError::Js)
    }

    /// The removal function `addSink` returns.
    pub fn remove_sink(&mut self, id: SinkId) -> Result<(), RenderError> {
        match self.pipeline.as_mut() {
            Some(p) => p.sink_manager.remove(id).map_err(RenderError::Js),
            None => Ok(()),
        }
    }

    /// `createFrameExportQueue(options)` on the active pipeline's backend.
    /// Enqueue with the pipeline's backend as the source
    /// (`queue.enqueue(renderer.backend().unwrap(), ...)`).
    pub fn create_frame_export_queue(
        &self,
        options: FrameExportOptions,
    ) -> Result<FrameExportQueue<WebGpuFrameExportAdapter>, FrameExportError> {
        let Some(p) = &self.pipeline else {
            return Err(FrameExportError(
                "Error: CanvasRenderer has no active pipeline; compile before creating a frame export queue"
                    .into(),
            ));
        };
        crate::frame_export::create_frame_export_queue(&p.backend, options)
    }

    // ---------------------------------------------------------- host textures

    /// `updateTextureFromSource(texId, source, {flipY})` for an RGBA8 image
    /// (`width * height * 4` bytes, top row first, straight alpha). Returns
    /// the source dimensions, `(0, 0)` without a pipeline.
    pub fn update_texture_from_source(
        &mut self,
        texture_id: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
        options: TextureUpdateOptions,
    ) -> Result<(u32, u32), RenderError> {
        let Some(p) = self.pipeline.as_mut() else {
            eprintln!("[updateTextureFromSource] Pipeline not ready");
            return Ok((0, 0));
        };
        p.backend
            .update_texture_from_rgba8(texture_id, width, height, rgba, options.flip_y)
    }

    /// `getMediaSteps()`: one entry per step bound to an external texture.
    pub fn get_media_steps(&mut self) -> Vec<MediaStep> {
        let Some(p) = self.pipeline() else {
            return Vec::new();
        };
        p.graph
            .media_steps
            .as_array()
            .map(|steps| {
                steps
                    .iter()
                    .map(|s| MediaStep {
                        texture_id: to_js_string(s.get("textureId")),
                        uniform: to_js_string(s.get("uniform")),
                        step_index: s.get("stepIndex").as_f64().unwrap_or(f64::NAN),
                        effect: to_js_string(s.get("effect")),
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    // ----------------------------------------------------------------- meshes

    /// `_packCacheAndUploadMesh(meshId, meshData)`.
    fn pack_cache_and_upload_mesh(&mut self, mesh_id: &str, packed: PackedMesh) -> MeshLoadResult {
        let p = self.pipeline.as_mut().expect("callers check the pipeline");
        let (_, vertex_count) = p.backend.upload_mesh_data(
            mesh_id,
            &packed.position_data,
            &packed.normal_data,
            &packed.uv_data,
            packed.width as u32,
            packed.height as u32,
            packed.vertex_count as u32,
        );
        self.mesh_cache.insert(mesh_id.to_owned(), packed);
        MeshLoadResult {
            success: true,
            vertex_count: vertex_count as usize,
            error: None,
        }
    }

    /// `loadOBJFromString(objText, meshId)`: parse, pack into the 256x256
    /// mesh textures, cache (for later pipelines) and upload.
    pub fn load_obj_from_string(&mut self, obj_text: &str, mesh_id: &str) -> MeshLoadResult {
        if self.pipeline.is_none() {
            eprintln!("[loadOBJFromString] Pipeline not ready");
            return MeshLoadResult::failed("Pipeline not ready");
        }
        let packed = pack_mesh(&parse_obj(obj_text));
        self.pack_cache_and_upload_mesh(mesh_id, packed)
    }

    /// `loadOBJFromURL(url, meshId)` for a file path: read (decoded as a
    /// fetch decodes it), parse, pack, cache and upload.
    pub fn load_obj_from_path(&mut self, path: impl AsRef<Path>, mesh_id: &str) -> MeshLoadResult {
        if self.pipeline.is_none() {
            eprintln!("[loadOBJFromURL] Pipeline not ready");
            return MeshLoadResult::failed("Pipeline not ready");
        }
        let path = path.as_ref();
        match std::fs::read(path) {
            Ok(bytes) => {
                let packed = pack_mesh(&parse_obj(&decode_obj_text(&bytes)));
                self.pack_cache_and_upload_mesh(mesh_id, packed)
            }
            Err(e) => {
                eprintln!("[Canvas] Failed to load OBJ: {e}");
                MeshLoadResult::failed(format!("Failed to load OBJ: {}: {e}", path.display()))
            }
        }
    }

    /// `loadOBJFromURL(`${basePath}/${path}`, meshId)` for a catalog mesh
    /// (`share/meshes/<name>.obj`, or a built-in name such as `sphere`).
    pub fn load_builtin_mesh(&mut self, name_or_path: &str, mesh_id: &str) -> MeshLoadResult {
        if self.pipeline.is_none() {
            eprintln!("[loadOBJFromURL] Pipeline not ready");
            return MeshLoadResult::failed("Pipeline not ready");
        }
        match noisemaker_host::obj::builtin_mesh(name_or_path) {
            Some(mesh) => self.pack_cache_and_upload_mesh(mesh_id, pack_mesh(&mesh)),
            None => MeshLoadResult::failed(format!(
                "Failed to load OBJ: {name_or_path} is not a catalog mesh"
            )),
        }
    }

    // --------------------------------------------------------------- parameters

    /// `resolveEnumValue(path)`.
    pub fn resolve_enum_value(&self, path: &Value) -> Value {
        resolve_enum_value(path, &self.enums)
    }

    /// `convertParameterForUniform(value, spec)`.
    pub fn convert_parameter_for_uniform(
        &self,
        value: &Value,
        spec: &Value,
    ) -> Result<Value, RenderError> {
        convert_parameter_for_uniform(value, spec, &self.enums).map_err(js_error)
    }

    /// `_isEffectPass(pass, effect)`.
    fn is_effect_pass(pass: &Object, effect: &EffectEntry) -> bool {
        let func = pass::get(pass, "effectFunc");
        let func = if func.is_truthy() {
            func
        } else {
            pass::get(pass, "effectKey")
        };
        if !func.is_truthy() || !strict_equals(func, effect.def.get("func")) {
            return false;
        }
        let target = effect.def.get("namespace");
        let target = if target.is_truthy() {
            target.clone()
        } else if effect.namespace.is_empty() {
            Value::Null
        } else {
            Value::from(effect.namespace.as_str())
        };
        let pass_namespace = pass::get(pass, "effectNamespace");
        !(target.is_truthy()
            && pass_namespace.is_truthy()
            && !strict_equals(pass_namespace, &target))
    }

    /// `buildUniformBindings(effect)`: which pass uniforms each of the effect's
    /// parameters feeds (directly, or through a pass's renamed shader
    /// uniform).
    pub fn build_uniform_bindings(&mut self, effect: &EffectEntry) {
        self.flush_program_passes();
        self.uniform_bindings.clear();
        let Some(p) = &self.pipeline else {
            return;
        };
        let globals = effect.def.get("globals");
        let Some(globals) = globals.as_object() else {
            return;
        };
        // paramName -> shader variable names, from the definition's pass
        // `uniforms` bridges ({shaderVarName: paramName}).
        let mut bridges: IndexMap<String, Vec<String>> = IndexMap::new();
        if let Some(passes) = effect.def.get("passes").as_array() {
            for def_pass in passes {
                let Some(uniforms) = def_pass.get("uniforms").as_object() else {
                    continue;
                };
                for (shader_name, param_ref) in uniforms.iter() {
                    let param_ref = to_js_string(param_ref);
                    if *shader_name == param_ref {
                        continue;
                    }
                    let entry = bridges.entry(param_ref).or_default();
                    if !entry.contains(shader_name) {
                        entry.push(shader_name.clone());
                    }
                }
            }
        }
        let mut bindings: IndexMap<String, Vec<UniformBinding>> = IndexMap::new();
        for (index, pass) in p.graph.passes.iter().enumerate() {
            if !Self::is_effect_pass(pass, effect) {
                continue;
            }
            for (param_name, spec) in globals.iter() {
                if spec.get("type").as_str() == Some("surface") {
                    continue;
                }
                let Some(uniforms) = pass::uniforms(pass) else {
                    continue;
                };
                let uniform = spec.get("uniform");
                let uniform_name = if uniform.is_truthy() {
                    to_js_string(uniform)
                } else {
                    param_name.clone()
                };
                if uniforms.contains_key(&uniform_name) {
                    bindings
                        .entry(param_name.clone())
                        .or_default()
                        .push(UniformBinding {
                            pass_index: index,
                            uniform_name,
                        });
                    continue;
                }
                if let Some(shader_names) = bridges.get(param_name) {
                    for shader_name in shader_names {
                        if uniforms.contains_key(shader_name) {
                            bindings
                                .entry(param_name.clone())
                                .or_default()
                                .push(UniformBinding {
                                    pass_index: index,
                                    uniform_name: shader_name.clone(),
                                });
                        }
                    }
                }
            }
        }
        self.uniform_bindings = bindings;
    }

    /// The bindings `buildUniformBindings` built.
    pub fn uniform_bindings(&self) -> &IndexMap<String, Vec<UniformBinding>> {
        &self.uniform_bindings
    }

    /// `applyParameterValues(effect, parameterValues)`: write the effect's
    /// parameter values to the pass uniforms they feed (automation-controlled
    /// values excluded), propagating chain-scoped parameters.
    pub fn apply_parameter_values(
        &mut self,
        effect: &EffectEntry,
        parameter_values: &Object,
    ) -> Result<(), RenderError> {
        self.flush_program_passes();
        if self.pipeline.is_none() {
            return Ok(());
        }
        if self.uniform_bindings.is_empty() {
            self.build_uniform_bindings(effect);
        }
        let globals = effect
            .def
            .get("globals")
            .as_object()
            .cloned()
            .unwrap_or_default();
        let mut scoped_param_changed = false;
        for (param_name, spec) in globals.iter() {
            if spec.get("type").as_str() == Some("surface") {
                continue;
            }
            let Some(bindings) = self.uniform_bindings.get(param_name).cloned() else {
                continue;
            };
            if bindings.is_empty() {
                continue;
            }
            let current = parameter_values.get_or_undefined(param_name);
            if current.is_undefined() || is_automation_controlled(current) {
                continue;
            }
            let converted = self.convert_parameter_for_uniform(current, spec)?;
            let uniform = spec.get("uniform");
            let own_uniform = if uniform.is_truthy() {
                to_js_string(uniform)
            } else {
                param_name.clone()
            };
            let p = self.pipeline.as_mut().expect("checked above");
            for pass in p.graph.passes.iter_mut() {
                if !Self::is_effect_pass(pass, effect) {
                    continue;
                }
                write_pass_uniform_aliases(pass, param_name, &own_uniform, &converted)?;
            }
            for binding in &bindings {
                let Some(pass) = p.graph.passes.get_mut(binding.pass_index) else {
                    continue;
                };
                let inherits = pass::get(pass, "inheritsVolumeSize").is_truthy();
                let Some(uniforms) = pass::uniforms_mut(pass) else {
                    continue;
                };
                if binding.uniform_name == "volumeSize" && inherits {
                    continue;
                }
                uniforms.insert(binding.uniform_name.clone(), slice_copy(&converted));
                let scoped = pass::get(pass, "scopedParams")
                    .get(&binding.uniform_name)
                    .clone();
                if scoped.is_truthy() {
                    let scoped_name = to_js_string(&scoped);
                    let value = pass::uniforms(pass)
                        .map(|u| u.get_or_undefined(&binding.uniform_name).clone())
                        .unwrap_or_default();
                    pass::uniforms_mut(pass)
                        .expect("checked above")
                        .insert(scoped_name.clone(), value);
                    scoped_param_changed = true;
                    p.broadcast_chain_scoped_param(
                        binding.pass_index,
                        &binding.uniform_name,
                        &scoped_name,
                    );
                }
            }
        }
        if scoped_param_changed {
            let p = self.pipeline.as_mut().expect("checked above");
            let defaults = p.collect_default_uniforms();
            p.recreate_textures(&defaults)?;
        }
        Ok(())
    }

    /// `applyStepParameterValues(stepParameterValues)`: write each step's
    /// values (`{step_N: {param: value}}`) to its own passes' uniforms, so
    /// instances of one effect keep separate values; then expand palettes,
    /// schedule overlay regeneration (`checkAsyncRegen`) and re-resolve
    /// textures sized by changed chain-scoped parameters.
    pub fn apply_step_parameter_values(
        &mut self,
        step_parameter_values: &Object,
    ) -> Result<(), RenderError> {
        self.flush_program_passes();
        let registry = self.registry.clone();
        let enums = self.enums.clone();
        let Some(p) = self.pipeline.as_mut() else {
            return Ok(());
        };
        let mut scoped_param_changed = false;
        for index in 0..p.graph.passes.len() {
            let pass = &p.graph.passes[index];
            let step_index = pass::get(pass, "stepIndex");
            if step_index.is_undefined() {
                continue;
            }
            let step_key = format!("step_{}", to_js_string(step_index));
            let step_params = step_parameter_values.get_or_undefined(&step_key);
            let Some(step_params) = step_params.as_object().filter(|_| step_params.is_truthy())
            else {
                continue;
            };
            let effect_key = pass::get(pass, "effectKey").clone();
            let effect_def = if effect_key.is_truthy() {
                registry.get_effect(&to_js_string(&effect_key))
            } else {
                None
            };
            let Some(effect_def) = effect_def else {
                continue;
            };
            let globals = effect_def.def.get("globals");
            let Some(globals) = globals.as_object().filter(|_| globals.is_truthy()) else {
                continue;
            };

            // Uniforms a colorModeUniform controls are set by the expander.
            let color_mode_controlled: Vec<String> = globals
                .values()
                .filter_map(|spec| {
                    let u = spec.get("colorModeUniform");
                    u.is_truthy().then(|| to_js_string(u))
                })
                .collect();

            let mut palette_expansion: Option<Object> = None;
            for (param_name, value) in step_params.iter() {
                if param_name == "_skip" {
                    continue;
                }
                if is_automation_controlled(value) {
                    continue;
                }
                let spec = globals.get_or_undefined(param_name);
                if !spec.is_truthy() || spec.get("type").as_str() == Some("surface") {
                    continue;
                }
                let uniform = spec.get("uniform");
                let uniform_name = if uniform.is_truthy() {
                    to_js_string(uniform)
                } else {
                    param_name.clone()
                };
                if color_mode_controlled.contains(&uniform_name) {
                    continue;
                }
                let pass = &mut p.graph.passes[index];
                if !pass::get(pass, "uniforms").is_truthy() {
                    continue;
                }
                let has_uniform =
                    pass::uniforms(pass).is_some_and(|u| u.contains_key(&uniform_name));
                if !has_uniform {
                    let converted =
                        convert_parameter_for_uniform(value, spec, &enums).map_err(js_error)?;
                    write_pass_uniform_aliases(pass, param_name, &uniform_name, &converted)?;
                    continue;
                }
                if uniform_name == "volumeSize" && pass::get(pass, "inheritsVolumeSize").is_truthy()
                {
                    continue;
                }
                let converted =
                    convert_parameter_for_uniform(value, spec, &enums).map_err(js_error)?;
                if let Some(uniforms) = pass::uniforms_mut(pass) {
                    uniforms.insert(uniform_name.clone(), slice_copy(&converted));
                }
                write_pass_uniform_aliases(pass, param_name, &uniform_name, &converted)?;

                let scoped = pass::get(pass, "scopedParams").get(&uniform_name).clone();
                if pass::get(pass, "scopedParams").is_truthy() && scoped.is_truthy() {
                    let scoped_name = to_js_string(&scoped);
                    let current = pass::uniforms(pass)
                        .map(|u| u.get_or_undefined(&uniform_name).clone())
                        .unwrap_or_default();
                    if let Some(uniforms) = pass::uniforms_mut(pass) {
                        uniforms.insert(scoped_name.clone(), current);
                    }
                    scoped_param_changed = true;
                    p.broadcast_chain_scoped_param(index, &uniform_name, &scoped_name);
                }

                if spec.get("type").as_str() == Some("palette") {
                    palette_expansion = expand_palette_value(&converted).map_err(js_error)?;
                }
            }

            if let Some(expansion) = palette_expansion {
                let pass = &mut p.graph.passes[index];
                if let Some(uniforms) = pass::uniforms_mut(pass) {
                    for (u_name, u_value) in expansion.iter() {
                        if uniforms.contains_key(u_name) {
                            uniforms.insert(u_name.clone(), slice_copy(u_value));
                        }
                    }
                }
            }

            // A changed non-alpha param of an asyncInit effect re-renders its
            // overlay (debounced; unchanged params are no-ops).
            let node_id = pass::get(&p.graph.passes[index], "nodeId").clone();
            if node_id.is_truthy() {
                p.check_async_regen(
                    &to_js_string(&node_id),
                    &to_js_string(&effect_key),
                    step_params,
                );
            }
        }
        if scoped_param_changed {
            let defaults = p.collect_default_uniforms();
            p.recreate_textures(&defaults)?;
        }
        Ok(())
    }
}

impl ProgramHost for CanvasRenderer {
    fn current_dsl(&self) -> String {
        self.current_dsl.clone()
    }

    fn enums(&self) -> Rc<Value> {
        self.enums.clone()
    }

    fn graph_passes(&mut self) -> Option<&mut Vec<Value>> {
        if self.program_passes.is_none() {
            let p = self.pipeline.as_mut()?;
            let passes = std::mem::take(&mut p.graph.passes);
            self.program_passes = Some(passes.into_iter().map(Value::Object).collect());
        }
        self.program_passes.as_mut()
    }

    fn broadcast_chain_scoped_param(
        &mut self,
        pass_index: usize,
        uniform_name: &str,
        scoped_name: &str,
    ) -> Result<(), JsError> {
        self.flush_program_passes();
        if let Some(p) = self.pipeline.as_mut() {
            p.broadcast_chain_scoped_param(pass_index, uniform_name, scoped_name);
        }
        Ok(())
    }

    fn check_async_regen(
        &mut self,
        node_id: &Value,
        effect_key: &Value,
        step_values: &Object,
    ) -> Result<(), JsError> {
        // Touches the pipeline's overlay cache only, never the passes.
        if let Some(p) = self.pipeline.as_mut() {
            p.check_async_regen(
                &to_js_string(node_id),
                &to_js_string(effect_key),
                step_values,
            );
        }
        Ok(())
    }

    fn collect_default_uniforms(&mut self) -> Result<Object, JsError> {
        self.flush_program_passes();
        Ok(self
            .pipeline
            .as_ref()
            .map(|p| p.collect_default_uniforms())
            .unwrap_or_default())
    }

    fn recreate_textures(&mut self, uniforms: Object) -> Result<(), JsError> {
        self.flush_program_passes();
        match self.pipeline.as_mut() {
            Some(p) => p.recreate_textures(&uniforms).map_err(render_to_js),
            None => Ok(()),
        }
    }

    fn set_uniform(&mut self, name: &str, value: &Value) -> Result<(), JsError> {
        self.flush_program_passes();
        match self.pipeline.as_mut() {
            Some(p) => p.set_uniform(name, value.clone()).map_err(render_to_js),
            None => Ok(()),
        }
    }
}

/// Cube export (`renderer/cubeExport.js`): assemble the six faces of
/// [`CanvasRenderer::render_cubemap`] into export layouts.
pub mod cube_export {
    use crate::backend::PixelData;

    const FACE_NAMES: [&str; 6] = ["px", "nx", "py", "ny", "pz", "nz"];

    /// `faceFileNames()`: `px.png` ... `nz.png`.
    pub fn face_file_names() -> Vec<String> {
        FACE_NAMES.iter().map(|n| format!("{n}.png")).collect()
    }

    /// `CROSS_CELL`: the horizontal-cross grid cell (column, row) of each face
    /// (+X, -X, +Y, -Y, +Z, -Z).
    pub const CROSS_CELL: [(usize, usize); 6] = [(0, 1), (2, 1), (1, 0), (1, 2), (1, 1), (3, 1)];

    /// `crossLayout(faces)`: a 4x3 horizontal cross of square faces.
    pub fn cross_layout(faces: &[PixelData]) -> PixelData {
        let size = faces[0].width as usize;
        let (w, h) = (size * 4, size * 3);
        let mut data = vec![0u8; w * h * 4];
        for (f, &(cx, cy)) in CROSS_CELL.iter().enumerate() {
            let Some(face) = faces.get(f) else {
                continue;
            };
            let (ox, oy) = (cx * size, cy * size);
            for y in 0..size {
                let dst = ((oy + y) * w + ox) * 4;
                let src = y * size * 4;
                data[dst..dst + size * 4].copy_from_slice(&face.data[src..src + size * 4]);
            }
        }
        PixelData {
            width: w as u32,
            height: h as u32,
            data,
        }
    }
}
