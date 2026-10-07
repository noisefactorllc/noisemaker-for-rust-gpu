//! Port of the reference WebGPU backend (`runtime/backends/webgpu.js`) on wgpu.
//!
//! The structure follows the reference: one command encoder per frame
//! (`beginFrame`/`endFrame`), passes encoded into it, uniform buffers taken from a
//! pool that `beginFrame` refills, and the operations the reference submits on the
//! queue immediately (texture copies, clears, mip generation, read-backs) submitted
//! immediately here too, so every queue ordering matches.
//!
//! Device validation errors never panic: the backend installs an uncaptured-error
//! handler (the reference's `uncapturederror` listener) that records each error as
//! an `ERR_DEVICE_VALIDATION` diagnostic; callers check
//! [`WebGpuBackend::device_errors`].

mod bind;
mod passes;
mod probe;
mod programs;
mod shaders;
mod textures;
#[cfg(target_vendor = "apple")]
mod tint;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;
use noisemaker_dsl::{Object, Value};

use crate::diagnostics::{DiagnosticCollector, codes};
use crate::error::RenderError;
use crate::uniforms::PackScratch;

pub use probe::{FloatPixels, TextureSnapshot};
pub use programs::{ComputeProgram, Program, ProgramKind, RenderProgram};
pub use shaders::{DeviceShader, ShaderCompiler};
pub use textures::{PixelData, TextureRecord};

/// `DEFAULT_VERTEX_SHADER_WGSL` (`runtime/default-shaders.js`).
pub const DEFAULT_VERTEX_SHADER_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let pos = positions[vertexIndex];

    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}
"#;

/// `DEFAULT_VERTEX_ENTRY_POINT`.
pub const DEFAULT_VERTEX_ENTRY_POINT: &str = "vs_main";
/// `DEFAULT_FRAGMENT_ENTRY_POINT`.
pub const DEFAULT_FRAGMENT_ENTRY_POINT: &str = "main";

/// Device capabilities (`backend.capabilities`).
#[derive(Debug, Clone, PartialEq)]
pub struct Capabilities {
    pub is_mobile: bool,
    pub float_blend: bool,
    pub float_linear: bool,
    pub color_buffer_float: bool,
    pub max_draw_buffers: u32,
    pub max_texture_size: u32,
    pub max_color_bytes_per_sample: u32,
    pub max_state_size: u32,
}

impl Default for Capabilities {
    /// The `Backend` base-class defaults (before `init`).
    fn default() -> Self {
        Capabilities {
            is_mobile: false,
            float_blend: true,
            float_linear: true,
            color_buffer_float: true,
            max_draw_buffers: 8,
            max_texture_size: 4096,
            max_color_bytes_per_sample: 64,
            max_state_size: 2048,
        }
    }
}

/// The per-frame state the pipeline hands to the backend (`getFrameState()`).
#[derive(Clone, Default)]
pub struct FrameState {
    pub frame_index: f64,
    pub time: f64,
    /// `state.globalUniforms`.
    pub global_uniforms: Object,
    /// `state.surfaces`: surface name → texture record of its current read texture.
    pub surfaces: IndexMap<String, Rc<TextureRecord>>,
    /// `state.writeSurfaces`: surface name → current write texture id.
    pub write_surfaces: IndexMap<String, Value>,
    /// `state.graph.renderSurface`.
    pub render_surface: Value,
    pub screen_width: f64,
    pub screen_height: f64,
}

impl FrameState {
    /// `state.writeSurfaces?.[name]` when truthy.
    pub fn write_surface(&self, name: &str) -> Option<String> {
        self.write_surfaces
            .get(name)
            .filter(|v| v.is_truthy())
            .map(crate::jsv::to_js_string)
    }
}

/// The WebGPU backend.
pub struct WebGpuBackend {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// `this.textures`: texture id → record (a `Map`: insertion ordered).
    pub textures: IndexMap<String, Rc<TextureRecord>>,
    /// `this.programs`.
    pub programs: HashMap<String, Rc<Program>>,
    pub capabilities: Capabilities,
    /// Structured diagnostics (device validation errors, missing render targets).
    pub diagnostics: DiagnosticCollector,
    /// `_warnedMissingRenderTargets`: the `kind|output|pass` keys recorded.
    warned_missing_render_targets: HashSet<String>,
    samplers: HashMap<String, wgpu::Sampler>,
    storage_buffers: HashMap<String, wgpu::Buffer>,
    command_encoder: Option<wgpu::CommandEncoder>,
    default_vertex_module: Option<DeviceShader>,
    depth_texture: Option<wgpu::Texture>,
    depth_texture_size: (u32, u32),
    uniform_buffer_pool: Vec<wgpu::Buffer>,
    active_uniform_buffers: Vec<wgpu::Buffer>,
    /// `_mergedUniforms`: reused across calls; keys are cleared to `undefined`, never
    /// deleted, so key order is the order of first insertion over the backend's life.
    merged_uniforms: Object,
    merged_uniform_keys: Vec<String>,
    pack_scratch: PackScratch,
    dummy_texture_view: Option<wgpu::TextureView>,
    resample_module: Option<DeviceShader>,
    resample_pipelines: HashMap<String, wgpu::RenderPipeline>,
    buffer_to_texture_pipelines: HashMap<String, wgpu::RenderPipeline>,
    storage_textures: HashMap<String, (wgpu::Texture, wgpu::TextureView)>,
    device_errors: Arc<Mutex<Vec<String>>>,
    device_error_count: usize,
    /// Every device error observed, in order.
    pub device_error_log: Vec<String>,
    /// Bind-group entries dropped because the auto layout lacks their binding.
    pub dropped_binding_count: usize,
    /// How WGSL becomes device code ([`shaders`]).
    compiler: shaders::Compiler,
}

/// `GPUTextureUsage` flags (same values as wgpu's).
fn usage_flag(name: &str) -> wgpu::TextureUsages {
    match name {
        "render" => wgpu::TextureUsages::RENDER_ATTACHMENT,
        "sample" => wgpu::TextureUsages::TEXTURE_BINDING,
        "storage" => wgpu::TextureUsages::STORAGE_BINDING,
        "copySrc" => wgpu::TextureUsages::COPY_SRC,
        "copyDst" => wgpu::TextureUsages::COPY_DST,
        _ => wgpu::TextureUsages::empty(),
    }
}

/// A WebGPU texture format name → wgpu format.
pub fn wgpu_format(name: &str) -> Option<wgpu::TextureFormat> {
    use wgpu::TextureFormat as F;
    Some(match name {
        "r8unorm" => F::R8Unorm,
        "r8snorm" => F::R8Snorm,
        "r8uint" => F::R8Uint,
        "r8sint" => F::R8Sint,
        "r16uint" => F::R16Uint,
        "r16sint" => F::R16Sint,
        "r16float" => F::R16Float,
        "r16unorm" => F::R16Unorm,
        "r16snorm" => F::R16Snorm,
        "rg8unorm" => F::Rg8Unorm,
        "rg8snorm" => F::Rg8Snorm,
        "rg8uint" => F::Rg8Uint,
        "rg8sint" => F::Rg8Sint,
        "r32uint" => F::R32Uint,
        "r32sint" => F::R32Sint,
        "r32float" => F::R32Float,
        "rg16uint" => F::Rg16Uint,
        "rg16sint" => F::Rg16Sint,
        "rg16float" => F::Rg16Float,
        "rg16unorm" => F::Rg16Unorm,
        "rg16snorm" => F::Rg16Snorm,
        "rgba8unorm" => F::Rgba8Unorm,
        "rgba8unorm-srgb" => F::Rgba8UnormSrgb,
        "rgba8snorm" => F::Rgba8Snorm,
        "rgba8uint" => F::Rgba8Uint,
        "rgba8sint" => F::Rgba8Sint,
        "bgra8unorm" => F::Bgra8Unorm,
        "bgra8unorm-srgb" => F::Bgra8UnormSrgb,
        "rgb9e5ufloat" => F::Rgb9e5Ufloat,
        "rgb10a2uint" => F::Rgb10a2Uint,
        "rgb10a2unorm" => F::Rgb10a2Unorm,
        "rg11b10ufloat" => F::Rg11b10Ufloat,
        "rg32uint" => F::Rg32Uint,
        "rg32sint" => F::Rg32Sint,
        "rg32float" => F::Rg32Float,
        "rgba16uint" => F::Rgba16Uint,
        "rgba16sint" => F::Rgba16Sint,
        "rgba16float" => F::Rgba16Float,
        "rgba16unorm" => F::Rgba16Unorm,
        "rgba16snorm" => F::Rgba16Snorm,
        "rgba32uint" => F::Rgba32Uint,
        "rgba32sint" => F::Rgba32Sint,
        "rgba32float" => F::Rgba32Float,
        "stencil8" => F::Stencil8,
        "depth16unorm" => F::Depth16Unorm,
        "depth24plus" => F::Depth24Plus,
        "depth24plus-stencil8" => F::Depth24PlusStencil8,
        "depth32float" => F::Depth32Float,
        "depth32float-stencil8" => F::Depth32FloatStencil8,
        _ => return None,
    })
}

/// `resolveFormat`: the internal short names to WebGPU names; anything else passes
/// through; a missing format is `rgba8unorm`.
pub fn resolve_format(format: &Value) -> Value {
    let key = crate::jsv::to_js_string(format);
    let mapped = match key.as_str() {
        "rgba8" => Some("rgba8unorm"),
        "rgba16f" => Some("rgba16float"),
        "rgba32f" => Some("rgba32float"),
        "r8" => Some("r8unorm"),
        "r16f" => Some("r16float"),
        "r32f" => Some("r32float"),
        "rg8" => Some("rg8unorm"),
        "rg16f" => Some("rg16float"),
        "rg32f" => Some("rg32float"),
        "rgba8unorm" | "rgba16float" | "rgba32float" | "r8unorm" | "r16float" | "r32float"
        | "bgra8unorm" => Some(key.as_str()),
        _ => None,
    };
    match mapped {
        Some(m) => Value::from(m),
        None if format.is_truthy() => format.clone(),
        None => Value::from("rgba8unorm"),
    }
}

/// The wgpu format for a resolved WebGPU format value; non-enum values throw the
/// `TypeError` WebIDL raises.
pub fn gpu_format_of(resolved: &Value) -> Result<wgpu::TextureFormat, RenderError> {
    resolved
        .as_str()
        .and_then(wgpu_format)
        .ok_or_else(|| {
            RenderError::type_error(format!(
                "Failed to read the 'format' property: The provided value '{}' is not a valid enum value of type GPUTextureFormat.",
                crate::jsv::to_js_string(resolved)
            ))
        })
}

/// `resolveUsage`.
pub fn resolve_usage(usage: &Value) -> wgpu::TextureUsages {
    let mut flags = wgpu::TextureUsages::empty();
    match usage {
        Value::Array(items) => {
            for u in items {
                if let Some(s) = u.as_str() {
                    flags |= usage_flag(s);
                }
            }
        }
        Value::String(s) => {
            for c in s.chars() {
                flags |= usage_flag(&c.to_string());
            }
        }
        _ => {}
    }
    flags
}

/// The WebGPU name of a wgpu format (the inverse of [`wgpu_format`]).
pub fn format_name(format: wgpu::TextureFormat) -> &'static str {
    use wgpu::TextureFormat as F;
    match format {
        F::R8Unorm => "r8unorm",
        F::R8Snorm => "r8snorm",
        F::R8Uint => "r8uint",
        F::R8Sint => "r8sint",
        F::R16Uint => "r16uint",
        F::R16Sint => "r16sint",
        F::R16Float => "r16float",
        F::Rg8Unorm => "rg8unorm",
        F::R32Uint => "r32uint",
        F::R32Sint => "r32sint",
        F::R32Float => "r32float",
        F::Rg16Float => "rg16float",
        F::Rgba8Unorm => "rgba8unorm",
        F::Rgba8UnormSrgb => "rgba8unorm-srgb",
        F::Bgra8Unorm => "bgra8unorm",
        F::Rg32Float => "rg32float",
        F::Rgba16Float => "rgba16float",
        F::Rgba32Float => "rgba32float",
        F::Depth24Plus => "depth24plus",
        _ => "unknown",
    }
}

/// Options for [`GpuDevice::create`].
#[derive(Debug, Clone)]
pub struct DeviceOptions {
    /// Adapter power preference (the reference takes the browser's default adapter).
    pub power_preference: wgpu::PowerPreference,
    /// The shader compiler. `None`: `NM_SHADER_COMPILER` when set, else
    /// [`ShaderCompiler::Tint`] on Metal (the compiler of the reference's
    /// Chromium) and [`ShaderCompiler::Naga`] on every other backend.
    pub shader_compiler: Option<ShaderCompiler>,
}

impl Default for DeviceOptions {
    fn default() -> Self {
        DeviceOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            shader_compiler: None,
        }
    }
}

/// A device created the way `createPipeline` creates one for the WebGPU backend.
#[derive(Clone)]
pub struct GpuDevice {
    /// The instance the adapter came from (windowed hosts create their
    /// surfaces on it).
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// The compiler of the device's shaders.
    pub shader_compiler: ShaderCompiler,
}

impl GpuDevice {
    /// `true` when the device runs on Metal, where the reference compiles WGSL
    /// with Tint's MSL writer.
    pub fn is_metal(&self) -> bool {
        self.adapter.get_info().backend == wgpu::Backend::Metal
    }

    /// A WebGPU backend on this device, compiling shaders with the device's
    /// [`ShaderCompiler`] (on the naga path, with the shader lowering of its
    /// platform).
    pub fn backend(&self) -> WebGpuBackend {
        let mut backend = WebGpuBackend::new(self.device.clone(), self.queue.clone());
        backend.compiler = match self.shader_compiler {
            ShaderCompiler::Naga => shaders::Compiler::Naga {
                lowering: self.is_metal(),
            },
            #[cfg(target_vendor = "apple")]
            ShaderCompiler::Tint => shaders::Compiler::Tint(Box::new(tint::TintState::new(
                tint::metal_gpu(&self.adapter, &self.device),
            ))),
            #[cfg(not(target_vendor = "apple"))]
            ShaderCompiler::Tint => unreachable!("GpuDevice::create selects Tint only on Metal"),
        };
        backend
    }

    /// `createPipeline`'s WebGPU device: request an adapter, enable
    /// `float32-filterable` when the adapter has it, and raise
    /// `maxColorAttachmentBytesPerSample` to `min(adapter limit, 128)`. The
    /// Tint compiler also needs wgpu's passthrough shaders and immediates
    /// (see [`ShaderCompiler`]); a Metal adapter without them is an error.
    pub fn create(options: &DeviceOptions) -> Result<GpuDevice, String> {
        Self::create_on(wgpu::Instance::default(), None, options)
    }

    /// [`GpuDevice::create`] on `instance` with an adapter that can present to
    /// `surface` (a window surface created on the same instance), the device a
    /// windowed host renders and presents with.
    pub fn create_for_surface(
        instance: wgpu::Instance,
        surface: &wgpu::Surface<'_>,
        options: &DeviceOptions,
    ) -> Result<GpuDevice, String> {
        Self::create_on(instance, Some(surface), options)
    }

    fn create_on(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
        options: &DeviceOptions,
    ) -> Result<GpuDevice, String> {
        pollster::block_on(async {
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: options.power_preference,
                    force_fallback_adapter: false,
                    compatible_surface: surface,
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("no WebGPU adapter: {e}"))?;
            let metal = adapter.get_info().backend == wgpu::Backend::Metal;
            let shader_compiler = match options.shader_compiler.or(ShaderCompiler::from_env()?) {
                Some(ShaderCompiler::Tint) if !metal || cfg!(not(target_vendor = "apple")) => {
                    return Err(format!(
                        "the Tint shader compiler needs a Metal adapter; this one is {:?}",
                        adapter.get_info().backend
                    ));
                }
                Some(choice) => choice,
                None if metal && cfg!(target_vendor = "apple") => ShaderCompiler::Tint,
                None => ShaderCompiler::Naga,
            };
            let mut required_features = wgpu::Features::empty();
            if adapter
                .features()
                .contains(wgpu::Features::FLOAT32_FILTERABLE)
            {
                required_features |= wgpu::Features::FLOAT32_FILTERABLE;
            }
            let mut required_limits = wgpu::Limits {
                max_color_attachment_bytes_per_sample: adapter
                    .limits()
                    .max_color_attachment_bytes_per_sample
                    .min(128),
                ..wgpu::Limits::default()
            };
            if shader_compiler == ShaderCompiler::Tint {
                let needed = wgpu::Features::PASSTHROUGH_SHADERS | wgpu::Features::IMMEDIATES;
                let missing = needed - adapter.features();
                if !missing.is_empty() {
                    return Err(format!(
                        "the Tint shader compiler needs wgpu's {missing:?} on this Metal adapter \
                         ({}); set NM_SHADER_COMPILER=naga to compile with naga instead",
                        adapter.get_info().name
                    ));
                }
                required_features |= needed;
                required_limits.max_immediate_size = adapter.limits().max_immediate_size;
            }
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("noisemaker"),
                    required_features,
                    required_limits,
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("requestDevice failed: {e}"))?;
            Ok(GpuDevice {
                instance,
                adapter,
                device,
                queue,
                shader_compiler,
            })
        })
    }
}

impl WebGpuBackend {
    /// `new WebGPUBackend(device, null)`: a headless backend (no canvas context).
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> WebGpuBackend {
        let device_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = device_errors.clone();
        device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
            let detail = match &error {
                wgpu::Error::Validation { description, .. } => description.clone(),
                other => other.to_string(),
            };
            sink.lock().unwrap().push(detail);
        }));
        WebGpuBackend {
            device,
            queue,
            textures: IndexMap::new(),
            programs: HashMap::new(),
            capabilities: Capabilities::default(),
            diagnostics: DiagnosticCollector::default(),
            warned_missing_render_targets: HashSet::new(),
            samplers: HashMap::new(),
            storage_buffers: HashMap::new(),
            command_encoder: None,
            default_vertex_module: None,
            depth_texture: None,
            depth_texture_size: (0, 0),
            uniform_buffer_pool: Vec::new(),
            active_uniform_buffers: Vec::new(),
            merged_uniforms: Object::new(),
            merged_uniform_keys: Vec::new(),
            pack_scratch: PackScratch::default(),
            dummy_texture_view: None,
            resample_module: None,
            resample_pipelines: HashMap::new(),
            buffer_to_texture_pipelines: HashMap::new(),
            storage_textures: HashMap::new(),
            device_errors,
            device_error_count: 0,
            device_error_log: Vec::new(),
            dropped_binding_count: 0,
            compiler: shaders::Compiler::Naga { lowering: false },
        }
    }

    /// Compile with naga, with the shader lowering that mirrors the
    /// reference's Tint MSL output when `enabled` (for devices on the Metal
    /// backend; see [`crate::lowering`]).
    pub fn set_tint_msl_lowering(&mut self, enabled: bool) {
        self.compiler = shaders::Compiler::Naga { lowering: enabled };
    }

    /// The compiler of this backend's shaders.
    pub fn shader_compiler(&self) -> ShaderCompiler {
        match self.compiler {
            shaders::Compiler::Naga { .. } => ShaderCompiler::Naga,
            shaders::Compiler::Tint(_) => ShaderCompiler::Tint,
        }
    }

    /// The pipelines the Tint compiler could not create, each created on the
    /// naga path instead (where wgpu reports why it is invalid), with the
    /// reason.
    pub fn tint_fallback_log(&self) -> Vec<String> {
        match &self.compiler {
            shaders::Compiler::Tint(state) => state.fallbacks(),
            shaders::Compiler::Naga { .. } => Vec::new(),
        }
    }

    /// `getName()`.
    pub fn get_name(&self) -> &'static str {
        "WebGPU"
    }

    /// `init()`: capabilities, the four samplers and the 1x1 transparent dummy texture.
    pub fn init(&mut self) {
        let limits = self.device.limits();
        let is_mobile = false;
        self.capabilities = Capabilities {
            is_mobile,
            float_blend: true,
            float_linear: true,
            color_buffer_float: true,
            max_draw_buffers: 8,
            max_texture_size: if limits.max_texture_dimension_2d != 0 {
                limits.max_texture_dimension_2d
            } else {
                8192
            },
            max_color_bytes_per_sample: if limits.max_color_attachment_bytes_per_sample != 0 {
                limits.max_color_attachment_bytes_per_sample
            } else {
                32
            },
            max_state_size: if is_mobile { 512 } else { 2048 },
        };

        let sampler =
            |min_mag: wgpu::FilterMode, mip: wgpu::MipmapFilterMode, address: wgpu::AddressMode| {
                self.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: None,
                    address_mode_u: address,
                    address_mode_v: address,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: min_mag,
                    min_filter: min_mag,
                    mipmap_filter: mip,
                    ..Default::default()
                })
            };
        let default = sampler(
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
        );
        let nearest = sampler(
            wgpu::FilterMode::Nearest,
            wgpu::MipmapFilterMode::Nearest,
            wgpu::AddressMode::ClampToEdge,
        );
        let repeat = sampler(
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Nearest,
            wgpu::AddressMode::Repeat,
        );
        let mipmap = sampler(
            wgpu::FilterMode::Linear,
            wgpu::MipmapFilterMode::Linear,
            wgpu::AddressMode::ClampToEdge,
        );
        self.samplers.insert("default".into(), default);
        self.samplers.insert("nearest".into(), nearest);
        self.samplers.insert("repeat".into(), repeat);
        self.samplers.insert("mipmap".into(), mipmap);

        let dummy = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &dummy,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[0, 0, 0, 0],
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.dummy_texture_view = Some(dummy.create_view(&Default::default()));
        self.collect_device_errors();
    }

    /// `_recordMissingRenderTarget(kind, outputId, passId)`: record a missing
    /// render target as a structured diagnostic, matching the WebGL2 backend,
    /// once per `kind|output|pass` so per-frame rendering cannot grow it.
    pub(crate) fn record_missing_render_target(
        &mut self,
        kind: &str,
        output_id: Value,
        pass_id: Value,
    ) {
        let key = format!(
            "{kind}|{}|{}",
            crate::jsv::to_js_string(&output_id),
            crate::jsv::to_js_string(&pass_id)
        );
        if !self.warned_missing_render_targets.insert(key) {
            return;
        }
        let mut record = Object::new();
        record.insert("code", Value::from(codes::MISSING_RENDER_TARGET));
        record.insert("backend", Value::from("webgpu"));
        record.insert("stage", Value::from("render"));
        record.insert("kind", Value::from(kind));
        record.insert("pass", pass_id);
        record.insert("output", output_id);
        self.diagnostics.add(Value::Object(record));
    }

    /// Move device errors reported since the last call into the diagnostics
    /// (`ERR_DEVICE_VALIDATION`, as the reference's `uncapturederror` listener
    /// records them) and return them.
    pub fn collect_device_errors(&mut self) -> Vec<String> {
        let drained: Vec<String> = std::mem::take(&mut *self.device_errors.lock().unwrap());
        for detail in &drained {
            self.device_error_count += 1;
            self.device_error_log.push(detail.clone());
            let mut record = Object::new();
            record.insert("code", Value::from(codes::DEVICE_VALIDATION));
            record.insert("backend", Value::from("webgpu"));
            record.insert("stage", Value::from("device-validation"));
            record.insert("detail", Value::from(detail.as_str()));
            self.diagnostics.add(Value::Object(record));
        }
        drained
    }

    /// The number of device (validation, out-of-memory, internal) errors observed
    /// over the backend's lifetime.
    pub fn device_errors(&mut self) -> usize {
        self.collect_device_errors();
        self.device_error_count
    }

    /// `beginFrame()`: return the frame's uniform buffers to the pool and open the
    /// frame's command encoder.
    pub fn begin_frame(&mut self) {
        while let Some(buffer) = self.active_uniform_buffers.pop() {
            self.uniform_buffer_pool.push(buffer);
        }
        self.command_encoder = Some(self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            },
        ));
    }

    /// `endFrame()`: submit the frame's commands.
    pub fn end_frame(&mut self) {
        if let Some(encoder) = self.command_encoder.take() {
            self.queue.submit([encoder.finish()]);
        }
        self.collect_device_errors();
    }

    /// Drop an unfinished frame (the reference leaves it unsubmitted when a pass
    /// throws; the next `beginFrame` replaces it).
    pub fn abandon_frame(&mut self) {
        self.command_encoder = None;
    }

    /// `present(textureId)`: a headless backend has no canvas context.
    pub fn present(&mut self, _texture_id: &str) {}

    /// `destroy(options)`.
    pub fn destroy(&mut self, skip_textures: bool) {
        if !skip_textures {
            let ids: Vec<String> = self.textures.keys().cloned().collect();
            for id in ids {
                self.destroy_texture(&id);
            }
            self.textures.clear();
        }
        if let Some(depth) = self.depth_texture.take() {
            depth.destroy();
            self.depth_texture_size = (0, 0);
        }
        self.programs.clear();
        self.samplers.clear();
        for buffer in self.uniform_buffer_pool.drain(..) {
            buffer.destroy();
        }
        for buffer in self.active_uniform_buffers.drain(..) {
            buffer.destroy();
        }
    }

    /// Wait for all submitted GPU work.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// `parseGlobalName` of the backend: `global_name` and `globalName` forms.
    pub fn parse_global_name(tex_id: &Value) -> Option<String> {
        let tex_id = tex_id.as_str()?;
        if let Some(rest) = tex_id.strip_prefix("global_") {
            return Some(rest.to_owned());
        }
        if tex_id.starts_with("global") && tex_id.encode_utf16().count() > 6 {
            let suffix = &tex_id[6..];
            let mut chars = suffix.chars();
            let first = chars.next()?;
            if first.is_ascii_uppercase() || first.is_ascii_digit() {
                return Some(format!("{}{}", first.to_lowercase(), chars.as_str()));
            }
        }
        None
    }

    /// `this.samplers.get(name)`: `default`, `nearest`, `repeat` or `mipmap`
    /// (`None` before `init`).
    pub fn sampler_for(&self, name: &str) -> Option<&wgpu::Sampler> {
        self.samplers.get(name)
    }

    fn sampler(&self, name: &str) -> &wgpu::Sampler {
        self.samplers
            .get(name)
            .or_else(|| self.samplers.get("default"))
            .expect("backend samplers are created by init()")
    }
}
