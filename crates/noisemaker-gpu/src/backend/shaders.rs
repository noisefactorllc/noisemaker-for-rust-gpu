//! Shader compilation for the device.
//!
//! The reference compiles every WGSL module with Chromium's Dawn: on Metal,
//! Tint translates each pipeline stage to MSL and Metal compiles that MSL in
//! relaxed math mode (`#pragma METAL fp math_mode(relaxed)`, which Dawn
//! prepends) without `preserveInvariance`. wgpu instead translates WGSL with
//! naga and compiles with fast math and `preserveInvariance` on, and Metal
//! then fuses and associates floating-point arithmetic differently
//! ([`crate::lowering`] documents the measured differences). So on Metal this
//! module compiles the way Chromium does: [`noisemaker_tint`] (Tint built from
//! Chromium's Dawn revision) generates each stage's MSL with the options of
//! Dawn's Metal backend, and wgpu compiles that MSL as a passthrough module,
//! which wgpu-hal compiles with Metal's default compile options (no
//! `preserveInvariance`), the math mode coming from Dawn's pragma.
//!
//! [`ShaderCompiler::Naga`] keeps wgpu's own WGSL path, after
//! [`crate::lowering`] on Metal: the compiler of every other backend, and on
//! Metal an explicit opt-out (`NM_SHADER_COMPILER=naga`, or
//! [`super::DeviceOptions::shader_compiler`]) for A/B comparisons. The Tint
//! path itself is `backend/tint.rs`, built for Apple targets only.
//!
//! # Pipeline layouts
//!
//! The reference creates its pipelines with `layout: 'auto'`, as the naga
//! path does with `layout: None`. wgpu cannot derive a layout from a
//! passthrough module, so the Tint path derives it with wgpu-core's own
//! derivation: `wgpu_core::validation::Interface::check_stage` over the
//! pipeline's stages fills the entry maps from which wgpu-core creates an
//! implicit layout (trailing empty groups dropped, entries sorted by
//! binding), and the bind group layouts and pipeline layout are created from
//! exactly those entries. The
//! same stage checks, plus the color-target check wgpu-core applies to an
//! implicit layout, reject what wgpu rejects; such a pipeline, or one Tint
//! cannot translate, is created on the naga path instead, which reports the
//! error wgpu reports (each such fallback is logged in
//! [`super::WebGpuBackend::tint_fallback_log`]).
//!
//! # The Metal argument table
//!
//! Tint's binding remapper gets the Metal indices wgpu-hal's Metal backend
//! assigns to a pipeline layout (`wgpu-hal` `metal::Device::
//! create_pipeline_layout`): per stage, the immediate block takes buffer 0
//! when the layout has immediates; then, group by group, every bind group
//! layout entry visible to the stage takes the next buffer (uniform and
//! storage buffers), texture (sampled and storage textures) or sampler
//! index, in binding order. wgpu-hal binds the immediate block and every
//! bind group at those indices, so the generated MSL differs from Chromium's
//! only in these indices (Dawn numbers its argument table its own way) and
//! in the entry point's name (Chromium appends the page origin's bytes).
//!
//! # Immediates
//!
//! Dawn's immediate block holds, after any user immediates, the depth range
//! a written `frag_depth` is clamped to (`RenderImmediates::clampFragDepth`,
//! two floats) and the byte sizes of the stage's storage buffers, from which
//! Tint computes `arrayLength` and the robustness clamps of runtime-sized
//! arrays. Here the block is wgpu's immediate data of the pipeline layout
//! (`Features::IMMEDIATES`): `[min depth, max depth]` when the fragment stage
//! writes `frag_depth`, then one `u32` size per storage-buffer entry of the
//! layout, numbered across the layout rather than per stage (wgpu shares one
//! immediate range between stages; Dawn indexes the sizes by Metal buffer
//! index). The passes set it ([`ImmediateBlock::data`], with the sizes of the
//! buffers they bind) right after setting such a pipeline.

use std::rc::Rc;

use crate::reflect::ShaderReflection;

#[cfg(target_vendor = "apple")]
pub(crate) use super::tint::{TintShader, TintState};
#[cfg(not(target_vendor = "apple"))]
pub(crate) use unavailable::{TintShader, TintState};

/// The compiler that turns WGSL into the device's shader code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderCompiler {
    /// wgpu compiles the WGSL with naga (on Metal after
    /// [`crate::lowering`]).
    Naga,
    /// Tint at Chromium's Dawn revision generates MSL as Dawn's Metal
    /// backend does and wgpu compiles it as a passthrough module (Metal
    /// only).
    Tint,
}

impl ShaderCompiler {
    /// `NM_SHADER_COMPILER`: `naga` or `tint` (unset or empty: no choice).
    pub fn from_env() -> Result<Option<ShaderCompiler>, String> {
        match std::env::var("NM_SHADER_COMPILER") {
            Ok(v) if v.is_empty() => Ok(None),
            Ok(v) => match v.as_str() {
                "naga" => Ok(Some(ShaderCompiler::Naga)),
                "tint" => Ok(Some(ShaderCompiler::Tint)),
                other => Err(format!(
                    "NM_SHADER_COMPILER={other}: expected \"naga\" or \"tint\""
                )),
            },
            Err(_) => Ok(None),
        }
    }
}

/// The compiler state of a backend.
pub(crate) enum Compiler {
    /// wgpu's WGSL path, with [`crate::lowering`] when `lowering`.
    Naga { lowering: bool },
    /// Tint + Metal passthrough (never constructed where the Tint path is
    /// not built).
    #[cfg_attr(not(target_vendor = "apple"), allow(dead_code))]
    Tint(Box<TintState>),
}

/// A WGSL module compiled for the device.
#[derive(Clone)]
pub enum DeviceShader {
    /// A wgpu (naga) module.
    Naga(wgpu::ShaderModule),
    /// WGSL that Tint translates per pipeline stage.
    Tint(Rc<TintShader>),
}

/// The immediate block of a Tint-path pipeline (see the module
/// documentation).
pub(crate) struct ImmediateBlock {
    /// `[min depth, max depth]` first (the fragment stage writes
    /// `frag_depth`).
    pub(crate) frag_depth: bool,
    /// The storage-buffer entries `(group, binding)` whose byte sizes follow,
    /// in index order.
    pub(crate) storage: Vec<(u32, u32)>,
}

impl ImmediateBlock {
    /// The byte offset of the storage-buffer sizes.
    pub(crate) fn sizes_offset(&self) -> u32 {
        if self.frag_depth { 8 } else { 0 }
    }

    /// The block's size in bytes.
    pub(crate) fn size(&self) -> u32 {
        self.sizes_offset() + 4 * self.storage.len() as u32
    }

    /// The block's bytes, with `buffer_size(group, binding)` the size in
    /// bytes of the buffer bound at a storage entry.
    pub(crate) fn data(&self, buffer_size: impl Fn(u32, u32) -> u64) -> Vec<u8> {
        let mut data = Vec::with_capacity(self.size() as usize);
        if self.frag_depth {
            // The viewport's depth range (the backends set 0..1).
            data.extend_from_slice(&0.0f32.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        for &(group, binding) in &self.storage {
            let size = buffer_size(group, binding).min(u64::from(u32::MAX)) as u32;
            data.extend_from_slice(&size.to_le_bytes());
        }
        data
    }
}

impl super::WebGpuBackend {
    /// Compile `source` (already validated into `reflection`) for the device.
    pub(super) fn device_shader(
        &self,
        id: &str,
        source: &str,
        reflection: &ShaderReflection,
    ) -> DeviceShader {
        match &self.compiler {
            Compiler::Tint(state) => {
                DeviceShader::Tint(Rc::new(state.shader(&self.device, id, source, reflection)))
            }
            Compiler::Naga { lowering } => {
                DeviceShader::Naga(self.naga_module(id, source, *lowering))
            }
        }
    }

    /// The wgpu (naga) module of `source`, after [`crate::lowering`] when
    /// `lowering`.
    pub(super) fn naga_module(&self, id: &str, source: &str, lowering: bool) -> wgpu::ShaderModule {
        let lowered = lowering
            .then(|| crate::lowering::lower_for_tint_msl(source))
            .flatten()
            .filter(|l| ShaderReflection::parse(&l.source).is_ok());
        let (device_source, self_bounded) = match lowered {
            Some(l) => (l.source, l.self_bounded),
            None => (source.to_owned(), false),
        };
        let descriptor = wgpu::ShaderModuleDescriptor {
            label: Some(id),
            source: wgpu::ShaderSource::Wgsl(device_source.into()),
        };
        if self_bounded {
            // SAFETY: every loop of the lowered WGSL either passes Tint's
            // finiteness analysis (a constant-bounded, unit-step integer
            // index) or carries Tint's own loop counter, so none can run
            // unbounded; every other runtime check stays on.
            unsafe {
                self.device.create_shader_module_trusted(
                    descriptor,
                    wgpu::ShaderRuntimeChecks {
                        force_loop_bounding: false,
                        ..wgpu::ShaderRuntimeChecks::checked()
                    },
                )
            }
        } else {
            self.device.create_shader_module(descriptor)
        }
    }

    /// The naga module of a shader (for a Tint shader, its fallback module).
    fn naga_of(&self, shader: &DeviceShader) -> wgpu::ShaderModule {
        match shader {
            DeviceShader::Naga(m) => m.clone(),
            DeviceShader::Tint(t) => t.fallback_module(self),
        }
    }

    /// Create a render pipeline with the layout the reference's
    /// `layout: 'auto'` derives (see the module documentation).
    pub(super) fn create_render_pipeline_from(
        &self,
        label: &str,
        vertex: (&DeviceShader, &str),
        fragment: Option<(&DeviceShader, &str)>,
        targets: &[Option<wgpu::ColorTargetState>],
        primitive: wgpu::PrimitiveState,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        if let Compiler::Tint(state) = &self.compiler
            && let DeviceShader::Tint(vs) = vertex.0
        {
            let fs = match fragment {
                Some((DeviceShader::Tint(fs), entry)) => Some(Some((fs.as_ref(), entry))),
                Some(_) => None,
                None => Some(None),
            };
            if let Some(fs) = fs {
                match state.render_pipeline(
                    self,
                    label,
                    (vs, vertex.1),
                    fs,
                    targets,
                    primitive,
                    depth_stencil.clone(),
                ) {
                    Ok(pipeline) => return pipeline,
                    Err(reason) => state.record_fallback(reason),
                }
            }
        }
        let vertex_module = self.naga_of(vertex.0);
        let fragment_module = fragment.map(|(s, _)| self.naga_of(s));
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &vertex_module,
                    entry_point: Some(vertex.1),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive,
                depth_stencil,
                multisample: Default::default(),
                fragment: fragment_module.as_ref().map(|m| wgpu::FragmentState {
                    module: m,
                    entry_point: fragment.map(|(_, e)| e),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
    }

    /// Create a compute pipeline with the layout the reference's
    /// `layout: 'auto'` derives (see the module documentation).
    pub(super) fn create_compute_pipeline_from(
        &self,
        label: &str,
        shader: &DeviceShader,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        if let (Compiler::Tint(state), DeviceShader::Tint(cs)) = (&self.compiler, shader) {
            match state.compute_pipeline(self, label, cs, entry_point) {
                Ok(pipeline) => return pipeline,
                Err(reason) => state.record_fallback(reason),
            }
        }
        let module = self.naga_of(shader);
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    /// The immediate block a Tint-path render pipeline reads, if any.
    pub(super) fn render_immediate_block(
        &self,
        pipeline: &wgpu::RenderPipeline,
    ) -> Option<Rc<ImmediateBlock>> {
        match &self.compiler {
            Compiler::Tint(state) => state.render_immediates(pipeline),
            Compiler::Naga { .. } => None,
        }
    }

    /// The immediate block a Tint-path compute pipeline reads, if any.
    pub(super) fn compute_immediate_block(
        &self,
        pipeline: &wgpu::ComputePipeline,
    ) -> Option<Rc<ImmediateBlock>> {
        match &self.compiler {
            Compiler::Tint(state) => state.compute_immediates(pipeline),
            Compiler::Naga { .. } => None,
        }
    }
}

/// The Tint path's types where it is not built (non-Apple targets): no value
/// of them exists, so [`Compiler::Tint`] and [`DeviceShader::Tint`] never
/// occur.
#[cfg(not(target_vendor = "apple"))]
mod unavailable {
    use std::convert::Infallible;
    use std::rc::Rc;

    use super::ImmediateBlock;
    use crate::backend::WebGpuBackend;
    use crate::reflect::ShaderReflection;

    pub(crate) struct TintState(Infallible);
    pub struct TintShader(Infallible);

    impl TintState {
        pub(crate) fn shader(
            &self,
            _: &wgpu::Device,
            _: &str,
            _: &str,
            _: &ShaderReflection,
        ) -> TintShader {
            match self.0 {}
        }

        #[allow(clippy::too_many_arguments)]
        pub(crate) fn render_pipeline(
            &self,
            _: &WebGpuBackend,
            _: &str,
            _: (&TintShader, &str),
            _: Option<(&TintShader, &str)>,
            _: &[Option<wgpu::ColorTargetState>],
            _: wgpu::PrimitiveState,
            _: Option<wgpu::DepthStencilState>,
        ) -> Result<wgpu::RenderPipeline, String> {
            match self.0 {}
        }

        pub(crate) fn compute_pipeline(
            &self,
            _: &WebGpuBackend,
            _: &str,
            _: &TintShader,
            _: &str,
        ) -> Result<wgpu::ComputePipeline, String> {
            match self.0 {}
        }

        pub(crate) fn record_fallback(&self, _: String) {
            match self.0 {}
        }

        pub(crate) fn fallbacks(&self) -> Vec<String> {
            match self.0 {}
        }

        pub(crate) fn render_immediates(
            &self,
            _: &wgpu::RenderPipeline,
        ) -> Option<Rc<ImmediateBlock>> {
            match self.0 {}
        }

        pub(crate) fn compute_immediates(
            &self,
            _: &wgpu::ComputePipeline,
        ) -> Option<Rc<ImmediateBlock>> {
            match self.0 {}
        }
    }

    impl TintShader {
        pub(crate) fn fallback_module(&self, _: &WebGpuBackend) -> wgpu::ShaderModule {
            match self.0 {}
        }
    }
}
