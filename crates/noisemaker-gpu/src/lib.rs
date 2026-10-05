//! GPU renderer for the Noisemaker effect catalog.
//!
//! A port of the reference engine's pipeline executor (`runtime/pipeline.js`) and
//! WebGPU backend (`runtime/backends/webgpu.js`) onto wgpu, rendering the reference
//! WGSL programs unmodified.
//!
//! ```no_run
//! use noisemaker_gpu::{GpuDevice, Graph, Renderer, RendererOptions};
//!
//! let device = GpuDevice::create(&Default::default()).unwrap();
//! let graph = Graph::from_json(&std::fs::read_to_string("noise.graph.json").unwrap()).unwrap();
//! let mut renderer = Renderer::new(&device, graph, 256, 256, RendererOptions::default()).unwrap();
//! for _ in 0..8 {
//!     renderer.render(0.25).unwrap();
//! }
//! let pixels = renderer.read_output().unwrap();
//! assert_eq!(pixels.data.len(), 256 * 256 * 4);
//! ```
//!
//! The host layer ([`host::CanvasRenderer`], a port of the reference's
//! `CanvasRenderer`) compiles DSL into a live pipeline and hot-swaps later
//! programs into it, applies effect parameters, uploads host media and meshes,
//! and carries MIDI and audio state; the catalog's native effect hooks
//! ([`effects`]: synth/media's lifecycle, the fibers/scratches/strayHair
//! overlays) run through it. [`demo::DemoHost`] adds what the reference demo
//! page does between a program and the renderer (ProgramState, the controls'
//! parameter writes, media, text and mesh inputs).
//!
//! ```no_run
//! use noisemaker_gpu::GpuDevice;
//! use noisemaker_gpu::host::{CanvasRenderer, CanvasRendererOptions, CompileOptions};
//!
//! let device = GpuDevice::create(&Default::default()).unwrap();
//! let mut renderer = CanvasRenderer::new(&device, CanvasRendererOptions::default());
//! renderer
//!     .compile("search synth\nnoise().write(o0)\nrender(o0)", &CompileOptions::default())
//!     .unwrap();
//! renderer.render(0.25).unwrap();
//! ```
//!
//! [`present::Presenter`] shows frames in a window (or on any wgpu target) the
//! way the reference's `present()` shows them on its canvas, and
//! [`present::Orientation`] gives read-back pixels that orientation. A
//! program that does not compile fails with [`RenderError::Dsl`], which
//! [`dsl::error_formatter::format_compile_error`] renders with its source
//! context. The examples run the host API end to end: `render_dsl` (a
//! program to a PNG), `animate` (a program over its loop to a PNG sequence)
//! and `viewer` (a live window that recompiles the program when its file
//! changes).

pub use noisemaker_dsl as dsl;
pub use noisemaker_host as host_inputs;
pub use noisemaker_input as input;

pub mod automation;
pub mod backend;
pub mod demo;
pub mod diagnostics;
pub mod effects;
pub mod error;
pub mod frame_export;
pub mod graph;
pub mod hooks;
pub mod host;
pub mod jsre;
pub mod jsv;
pub mod lowering;
pub mod pipeline;
pub mod png_io;
pub mod preflight;
pub mod present;
pub mod protocol;
pub mod reflect;
pub mod sink;
pub mod uniforms;
pub mod wgsl;

pub use automation::{ExternalState, SharedAudioState, SharedMidiState};
pub use backend::{Capabilities, DeviceOptions, GpuDevice, PixelData, WebGpuBackend};
pub use error::RenderError;
pub use frame_export::{FrameExportQueue, WebGpuFrameExportAdapter};
pub use graph::Graph;
pub use hooks::{EffectHooks, EffectRegistry};
pub use host::CanvasRenderer;
pub use noisemaker_dsl::{Object, Value};
pub use pipeline::{Pipeline, PipelineOptions};
pub use present::{Orientation, Presenter};

/// Options for [`Renderer::new`].
#[derive(Clone, Default)]
pub struct RendererOptions {
    /// Share pooled textures per `graph.allocations` (reference `texturePooling`).
    pub texture_pooling: bool,
    /// Native effect hooks (lifecycle, asyncInit).
    pub effects: EffectRegistry,
}

/// A pipeline on a WebGPU backend: `createPipeline(graph, {preferWebGPU: true})`.
pub struct Renderer {
    pipeline: Pipeline,
}

impl Renderer {
    /// Create the backend on `device`, compile the graph's programs and size every
    /// surface and texture for `width x height` (`createPipeline` + `init`).
    pub fn new(
        device: &GpuDevice,
        graph: Graph,
        width: u32,
        height: u32,
        options: RendererOptions,
    ) -> Result<Renderer, RenderError> {
        let backend = device.backend();
        let mut pipeline = Pipeline::new(
            graph,
            backend,
            PipelineOptions {
                texture_pooling: options.texture_pooling,
                effects: options.effects,
            },
        );
        pipeline.init(width as f64, height as f64)?;
        Ok(Renderer { pipeline })
    }

    /// `pipeline.resize(width, height)`.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), RenderError> {
        self.pipeline.resize(width as f64, height as f64)
    }

    /// `pipeline.render(time)`: one frame at normalized loop time `time` (0..1).
    pub fn render(&mut self, time: f64) -> Result<(), RenderError> {
        self.pipeline.render(time)
    }

    /// `backend.readPixels(textureId)`: RGBA8, top row first.
    pub fn read_pixels(&mut self, texture_id: &str) -> Result<PixelData, RenderError> {
        self.pipeline.backend.read_pixels(texture_id)
    }

    /// The render surface's current read texture (what the reference presents
    /// after the frame's buffer swap), read back as RGBA8.
    pub fn read_output(&mut self) -> Result<PixelData, RenderError> {
        let name = self
            .pipeline
            .graph
            .render_surface_name()
            .unwrap_or("o0")
            .to_owned();
        let surface = self
            .pipeline
            .surfaces
            .get(&name)
            .ok_or_else(|| RenderError::Js(format!("Error: no render surface {name}")))?;
        let id = surface.read.clone().ok_or_else(|| {
            RenderError::Js(format!("Error: render surface {name} has no read texture"))
        })?;
        self.read_pixels(&id)
    }

    /// `pipeline.setUniform(name, value)`.
    pub fn set_uniform(&mut self, name: &str, value: Value) -> Result<(), RenderError> {
        self.pipeline.set_uniform(name, value)
    }

    /// `backend.updateTextureFromSource(id, image, {flipY})` for an RGBA8 image.
    pub fn update_texture_from_rgba8(
        &mut self,
        id: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
        flip_y: bool,
    ) -> Result<(u32, u32), RenderError> {
        self.pipeline
            .backend
            .update_texture_from_rgba8(id, width, height, rgba, flip_y)
    }

    /// `backend.uploadMeshData(meshId, positions, normals, uvs, width, height, vertexCount)`.
    #[allow(clippy::too_many_arguments)]
    pub fn upload_mesh_data(
        &mut self,
        mesh_id: &str,
        positions: &[f32],
        normals: &[f32],
        uvs: &[f32],
        width: u32,
        height: u32,
        vertex_count: u32,
    ) -> (bool, u32) {
        self.pipeline.backend.upload_mesh_data(
            mesh_id,
            positions,
            normals,
            uvs,
            width,
            height,
            vertex_count,
        )
    }

    /// `backend.uploadDataTexture(id, data, width, height)`.
    pub fn upload_data_texture(&mut self, id: &str, data: &[f32], width: u32, height: u32) {
        self.pipeline
            .backend
            .upload_data_texture(id, data, width, height)
    }

    /// The device errors (validation, out-of-memory, internal) observed so far.
    pub fn device_errors(&mut self) -> usize {
        self.pipeline.backend.device_errors()
    }

    /// The underlying pipeline.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// The underlying pipeline (mutable).
    pub fn pipeline_mut(&mut self) -> &mut Pipeline {
        &mut self.pipeline
    }

    /// `pipeline.dispose()`.
    pub fn dispose(mut self) -> Result<(), RenderError> {
        self.pipeline.dispose()
    }
}
