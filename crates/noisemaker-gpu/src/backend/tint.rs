//! The Tint path of [`super::shaders`] (Apple targets): pipeline layouts
//! derived as wgpu-core derives `layout: None`, Dawn's MSL options with
//! wgpu-hal's Metal argument table, and the passthrough modules. The module
//! documentation of [`super::shaders`] describes the contract.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use noisemaker_tint as tint;
use wgpu::wgc::validation::{BindingLayoutSource, Interface, ShaderStageForValidation, StageIo};

use super::WebGpuBackend;
use super::shaders::ImmediateBlock;
use crate::reflect::ShaderReflection;

/// Tint-path state: the GPU's Dawn toggles and the caches of derived
/// layouts, stage modules and immediate blocks. Shaders are identified by
/// their source text, so the caches hold one entry per distinct source,
/// however often a graph is recompiled. The immediate blocks are kept for the
/// few pipelines that have one (storage buffers or a written `frag_depth`).
pub(crate) struct TintState {
    gpu: tint::dawn::MetalGpu,
    shader_ids: RefCell<HashMap<String, u64>>,
    layouts: RefCell<HashMap<String, Rc<TintLayout>>>,
    modules: RefCell<HashMap<String, wgpu::ShaderModule>>,
    render_immediates: RefCell<HashMap<wgpu::RenderPipeline, Rc<ImmediateBlock>>>,
    compute_immediates: RefCell<HashMap<wgpu::ComputePipeline, Rc<ImmediateBlock>>>,
    fallbacks: RefCell<Vec<String>>,
}

/// A WGSL module on the Tint path: its source, its naga reflection (for the
/// layout derivation) and the naga module of a fallback pipeline.
pub struct TintShader {
    id: u64,
    label: String,
    source: String,
    reflection: ShaderReflection,
    interface: Interface,
    naga: RefCell<Option<wgpu::ShaderModule>>,
}

/// The pipeline layout of a Tint-path pipeline.
struct TintLayout {
    /// Bind group layout entries by group (wgpu-core's implicit layout).
    groups: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
    pipeline_layout: wgpu::PipelineLayout,
    immediates: Option<Rc<ImmediateBlock>>,
}

/// The derived bind group entries of a pipeline (wgpu-core's implicit
/// layout: trailing empty groups dropped, entries sorted by binding).
fn derived_groups(source: BindingLayoutSource) -> Vec<Vec<wgpu::BindGroupLayoutEntry>> {
    let BindingLayoutSource::Derived(maps) = source else {
        unreachable!("the layout source is derived");
    };
    let mut groups: Vec<Vec<wgpu::BindGroupLayoutEntry>> = maps
        .iter()
        .map(|map| {
            let mut entries: Vec<wgpu::BindGroupLayoutEntry> = map.values().copied().collect();
            entries.sort_by_key(|e| e.binding);
            entries
        })
        .collect();
    while groups.last().is_some_and(Vec::is_empty) {
        groups.pop();
    }
    groups
}

/// `NM_TINT_DUMP_MSL=DIR` (a development aid, like Dawn's `dump_shaders`
/// toggle): write each generated stage's MSL to
/// `DIR/<label>.<stage>.<entry point>[.points].msl`, the label's characters
/// outside `[A-Za-z0-9_-]` replaced by `_`.
fn dump_msl(label: &str, stage: tint::Stage, entry_point: &str, point_list: bool, msl: &str) {
    let Some(dir) = std::env::var_os("NM_TINT_DUMP_MSL") else {
        return;
    };
    let clean: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let stage = match stage {
        tint::Stage::Vertex => "vertex",
        tint::Stage::Fragment => "fragment",
        tint::Stage::Compute => "compute",
    };
    let points = if point_list { ".points" } else { "" };
    let path =
        std::path::Path::new(&dir).join(format!("{clean}.{stage}.{entry_point}{points}.msl"));
    // A development aid: a failed write only loses the dump.
    let _ = std::fs::create_dir_all(&dir).and_then(|()| std::fs::write(path, msl));
}

fn stage_bit(stage: tint::Stage) -> wgpu::ShaderStages {
    match stage {
        tint::Stage::Vertex => wgpu::ShaderStages::VERTEX,
        tint::Stage::Fragment => wgpu::ShaderStages::FRAGMENT,
        tint::Stage::Compute => wgpu::ShaderStages::COMPUTE,
    }
}

/// The targets blend with a factor that reads the second blend source.
fn uses_dual_source(targets: &[Option<wgpu::ColorTargetState>]) -> bool {
    targets.iter().flatten().any(|t| {
        t.blend.is_some_and(|b| {
            [b.color, b.alpha].iter().any(|c| {
                c.src_factor.ref_second_blend_source() || c.dst_factor.ref_second_blend_source()
            })
        })
    })
}

/// The properties of the device's Metal GPU that select Dawn's shader
/// toggles: the vendor (Dawn reads the PCI vendor id; the adapter name
/// carries it) and the Apple9 GPU family.
pub(crate) fn metal_gpu(adapter: &wgpu::Adapter, device: &wgpu::Device) -> tint::dawn::MetalGpu {
    use objc2_metal::MTLDevice as _;
    let name = adapter.get_info().name.to_ascii_lowercase();
    let vendor = if name.contains("apple") {
        tint::dawn::Vendor::Apple
    } else if name.contains("intel") {
        tint::dawn::Vendor::Intel
    } else if name.contains("amd") || name.contains("radeon") {
        tint::dawn::Vendor::Amd
    } else {
        tint::dawn::Vendor::Other
    };
    // SAFETY: the Metal device is only queried, not retained or modified.
    let apple9 = unsafe { device.as_hal::<wgpu::hal::api::Metal>() }.is_some_and(|hal| {
        hal.raw_device()
            .supportsFamily(objc2_metal::MTLGPUFamily::Apple9)
    });
    tint::dawn::MetalGpu { vendor, apple9 }
}

impl TintShader {
    /// The naga module a fallback pipeline uses (after
    /// [`crate::lowering`], as the naga path compiles on Metal).
    pub(crate) fn fallback_module(&self, backend: &WebGpuBackend) -> wgpu::ShaderModule {
        self.naga
            .borrow_mut()
            .get_or_insert_with(|| backend.naga_module(&self.label, &self.source, true))
            .clone()
    }
}

impl TintState {
    pub(crate) fn new(gpu: tint::dawn::MetalGpu) -> TintState {
        TintState {
            gpu,
            shader_ids: RefCell::new(HashMap::new()),
            layouts: RefCell::new(HashMap::new()),
            modules: RefCell::new(HashMap::new()),
            render_immediates: RefCell::new(HashMap::new()),
            compute_immediates: RefCell::new(HashMap::new()),
            fallbacks: RefCell::new(Vec::new()),
        }
    }

    /// A Tint-path shader of `source` (already validated into `reflection`).
    pub(crate) fn shader(
        &self,
        device: &wgpu::Device,
        label: &str,
        source: &str,
        reflection: &ShaderReflection,
    ) -> TintShader {
        let id = {
            let mut ids = self.shader_ids.borrow_mut();
            let next = ids.len() as u64;
            *ids.entry(source.to_owned()).or_insert(next)
        };
        TintShader {
            id,
            label: label.to_owned(),
            source: source.to_owned(),
            interface: Interface::new(reflection.module(), reflection.info(), device.limits()),
            reflection: reflection.clone(),
            naga: RefCell::new(None),
        }
    }

    pub(crate) fn record_fallback(&self, reason: String) {
        self.fallbacks.borrow_mut().push(reason);
    }

    pub(crate) fn fallbacks(&self) -> Vec<String> {
        self.fallbacks.borrow().clone()
    }

    pub(crate) fn render_immediates(
        &self,
        pipeline: &wgpu::RenderPipeline,
    ) -> Option<Rc<ImmediateBlock>> {
        self.render_immediates.borrow().get(pipeline).cloned()
    }

    pub(crate) fn compute_immediates(
        &self,
        pipeline: &wgpu::ComputePipeline,
    ) -> Option<Rc<ImmediateBlock>> {
        self.compute_immediates.borrow().get(pipeline).cloned()
    }

    /// A render pipeline of Tint's MSL, or why it cannot be created (the
    /// caller then creates it on the naga path).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn render_pipeline(
        &self,
        backend: &WebGpuBackend,
        label: &str,
        (vs, vs_entry): (&TintShader, &str),
        fragment: Option<(&TintShader, &str)>,
        targets: &[Option<wgpu::ColorTargetState>],
        primitive: wgpu::PrimitiveState,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> Result<wgpu::RenderPipeline, String> {
        let device = &backend.device;
        let limits = device.limits();
        let mut source = BindingLayoutSource::new_derived(&limits);
        let mut sizes = naga::FastHashMap::default();
        let io = vs
            .interface
            .check_stage(
                &mut source,
                &mut sizes,
                vs_entry,
                ShaderStageForValidation::Vertex {
                    topology: primitive.topology,
                    compare_function: depth_stencil.as_ref().and_then(|d| d.depth_compare),
                },
                StageIo::default(),
                Some(primitive.topology),
            )
            .map_err(|e| format!("{label}: vertex stage `{vs_entry}`: {e}"))?;
        if let Some((fs, fs_entry)) = fragment {
            let io = fs
                .interface
                .check_stage(
                    &mut source,
                    &mut sizes,
                    fs_entry,
                    ShaderStageForValidation::Fragment {
                        dual_source_blending: uses_dual_source(targets),
                        has_depth_attachment: depth_stencil.is_some(),
                    },
                    io,
                    Some(primitive.topology),
                )
                .map_err(|e| format!("{label}: fragment stage `{fs_entry}`: {e}"))?;
            for (location, output) in io.varyings.iter() {
                if let Some(Some(target)) = targets.get(*location as usize) {
                    wgpu::wgc::validation::check_texture_format(target.format, &output.ty)
                        .map_err(|_| {
                            format!("{label}: color target {location}: incompatible format")
                        })?;
                }
            }
        }
        let frag_depth = fragment.is_some_and(|(fs, entry)| fs.reflection.writes_frag_depth(entry));
        let layout_key = format!(
            "r|{}:{vs_entry}|{}",
            vs.id,
            fragment.map_or(String::new(), |(fs, e)| format!("{}:{e}", fs.id))
        );
        let layout = self.layout(
            device,
            &layout_key,
            label,
            derived_groups(source),
            frag_depth,
        );

        let point_list = primitive.topology == wgpu::PrimitiveTopology::PointList;
        let vertex_module = self.stage_module(
            device,
            &layout_key,
            &layout,
            vs,
            vs_entry,
            tint::Stage::Vertex,
            point_list,
        )?;
        let fragment_module = match fragment {
            Some((fs, entry)) => Some(self.stage_module(
                device,
                &layout_key,
                &layout,
                fs,
                entry,
                tint::Stage::Fragment,
                false,
            )?),
            None => None,
        };
        let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vertex_module,
                entry_point: Some(tint::dawn::ENTRY_POINT),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive,
            depth_stencil,
            multisample: Default::default(),
            fragment: fragment_module.as_ref().map(|m| wgpu::FragmentState {
                module: m,
                entry_point: Some(tint::dawn::ENTRY_POINT),
                compilation_options: Default::default(),
                targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        if let Some(error) = pollster::block_on(guard.pop()) {
            return Err(format!("{label}: Metal passthrough pipeline: {error}"));
        }
        if let Some(block) = &layout.immediates {
            self.render_immediates
                .borrow_mut()
                .insert(pipeline.clone(), block.clone());
        }
        Ok(pipeline)
    }

    /// A compute pipeline of Tint's MSL, or why it cannot be created (the
    /// caller then creates it on the naga path).
    pub(crate) fn compute_pipeline(
        &self,
        backend: &WebGpuBackend,
        label: &str,
        cs: &TintShader,
        entry_point: &str,
    ) -> Result<wgpu::ComputePipeline, String> {
        let device = &backend.device;
        let limits = device.limits();
        let mut source = BindingLayoutSource::new_derived(&limits);
        let mut sizes = naga::FastHashMap::default();
        cs.interface
            .check_stage(
                &mut source,
                &mut sizes,
                entry_point,
                ShaderStageForValidation::Compute,
                StageIo::default(),
                None,
            )
            .map_err(|e| format!("{label}: compute stage `{entry_point}`: {e}"))?;
        let layout_key = format!("c|{}:{entry_point}", cs.id);
        let layout = self.layout(device, &layout_key, label, derived_groups(source), false);
        let module = self.stage_module(
            device,
            &layout_key,
            &layout,
            cs,
            entry_point,
            tint::Stage::Compute,
            false,
        )?;
        let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&layout.pipeline_layout),
            module: &module,
            entry_point: Some(tint::dawn::ENTRY_POINT),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = pollster::block_on(guard.pop()) {
            return Err(format!("{label}: Metal passthrough pipeline: {error}"));
        }
        if let Some(block) = &layout.immediates {
            self.compute_immediates
                .borrow_mut()
                .insert(pipeline.clone(), block.clone());
        }
        Ok(pipeline)
    }

    /// The (cached) layout of a pipeline whose derived entries are `groups`.
    fn layout(
        &self,
        device: &wgpu::Device,
        key: &str,
        label: &str,
        groups: Vec<Vec<wgpu::BindGroupLayoutEntry>>,
        frag_depth: bool,
    ) -> Rc<TintLayout> {
        if let Some(layout) = self.layouts.borrow().get(key)
            && layout.groups == groups
        {
            return layout.clone();
        }
        let storage: Vec<(u32, u32)> = groups
            .iter()
            .enumerate()
            .flat_map(|(group, entries)| {
                entries.iter().filter_map(move |e| match e.ty {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    } => Some((group as u32, e.binding)),
                    _ => None,
                })
            })
            .collect();
        let immediates = (frag_depth || !storage.is_empty()).then(|| {
            Rc::new(ImmediateBlock {
                frag_depth,
                storage,
            })
        });
        let bind_group_layouts: Vec<Option<wgpu::BindGroupLayout>> = groups
            .iter()
            .map(|entries| {
                (!entries.is_empty()).then(|| {
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(label),
                        entries,
                    })
                })
            })
            .collect();
        let refs: Vec<Option<&wgpu::BindGroupLayout>> =
            bind_group_layouts.iter().map(Option::as_ref).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &refs,
            immediate_size: immediates.as_ref().map_or(0, |b| b.size()),
        });
        let layout = Rc::new(TintLayout {
            groups,
            pipeline_layout,
            immediates,
        });
        self.layouts
            .borrow_mut()
            .insert(key.to_owned(), layout.clone());
        layout
    }

    /// Dawn's MSL options for one stage of a pipeline with `layout`, with
    /// the Metal indices wgpu-hal assigns (see the documentation of
    /// [`super::shaders`]).
    fn stage_options(
        &self,
        layout: &TintLayout,
        stage: tint::Stage,
        entry_point: &str,
        point_list: bool,
    ) -> Result<tint::MslOptions, String> {
        let mut options =
            tint::dawn::options(stage, entry_point, &self.gpu, point_list, 0xFFFF_FFFF);
        let visible = stage_bit(stage);
        let mut buffers = u32::from(layout.immediates.is_some());
        let (mut textures, mut samplers) = (0u32, 0u32);
        let next = |counter: &mut u32| {
            let slot = *counter;
            *counter += 1;
            slot
        };
        for (group, entries) in layout.groups.iter().enumerate() {
            for entry in entries {
                if !entry.visibility.contains(visible) {
                    continue;
                }
                if entry.count.is_some() {
                    return Err(format!("binding array at @binding({})", entry.binding));
                }
                let (class, slot) = match entry.ty {
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        ..
                    } => (tint::ResourceClass::Uniform, next(&mut buffers)),
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { .. },
                        ..
                    } => (tint::ResourceClass::Storage, next(&mut buffers)),
                    wgpu::BindingType::Sampler(_) => {
                        (tint::ResourceClass::Sampler, next(&mut samplers))
                    }
                    wgpu::BindingType::Texture { .. } => {
                        (tint::ResourceClass::Texture, next(&mut textures))
                    }
                    wgpu::BindingType::StorageTexture { .. } => {
                        (tint::ResourceClass::StorageTexture, next(&mut textures))
                    }
                    other => return Err(format!("unsupported binding type {other:?}")),
                };
                options.bindings.push(tint::Binding {
                    group: group as u32,
                    binding: entry.binding,
                    class,
                    slot,
                });
                if class == tint::ResourceClass::Storage
                    && let Some(block) = &layout.immediates
                {
                    let index = block
                        .storage
                        .iter()
                        .position(|&s| s == (group as u32, entry.binding))
                        .expect("every storage entry has a size slot");
                    options.buffer_sizes.push(tint::BufferSize {
                        group: group as u32,
                        binding: entry.binding,
                        index: index as u32,
                    });
                }
            }
        }
        if let Some(block) = &layout.immediates {
            options.immediate_slot = Some(0);
            if !options.buffer_sizes.is_empty() {
                options.buffer_sizes_offset = Some(block.sizes_offset());
            }
            if block.frag_depth {
                options.depth_range_offsets = Some((0, 4));
            }
        }
        Ok(options)
    }

    /// The (cached) passthrough module of one pipeline stage.
    #[allow(clippy::too_many_arguments)]
    fn stage_module(
        &self,
        device: &wgpu::Device,
        layout_key: &str,
        layout: &TintLayout,
        shader: &TintShader,
        entry_point: &str,
        stage: tint::Stage,
        point_list: bool,
    ) -> Result<wgpu::ShaderModule, String> {
        let key = format!(
            "{layout_key}|{stage:?}|{}:{entry_point}|{point_list}",
            shader.id
        );
        if let Some(module) = self.modules.borrow().get(&key) {
            return Ok(module.clone());
        }
        let options = self.stage_options(layout, stage, entry_point, point_list)?;
        let msl = tint::wgsl_to_msl(&shader.source, &options)
            .map_err(|e| format!("{}: Tint ({stage:?} `{entry_point}`): {e}", shader.label))?;
        if msl.needs_storage_buffer_sizes && layout.immediates.is_none() {
            return Err(format!(
                "{}: Tint reads storage-buffer sizes the layout has no immediates for",
                shader.label
            ));
        }
        dump_msl(&shader.label, stage, entry_point, point_list, &msl.source);
        let [x, y, z] = msl.workgroup_size;
        let entry_points = [wgpu::PassthroughShaderEntryPoint {
            name: Cow::Borrowed(tint::dawn::ENTRY_POINT),
            workgroup_size: (x, y, z),
        }];
        let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        // SAFETY: the MSL is Tint's translation of WGSL that naga validated
        // (and Tint parsed and resolved), with Dawn's robustness transform:
        // every resource access is bounds-checked and every loop bounded as
        // in Chromium. Its resource bindings follow the Metal argument table
        // wgpu-hal builds for `layout`, and `workgroup_size` is the entry
        // point's own.
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some(&shader.label),
                entry_points: Cow::Borrowed(&entry_points),
                msl: Some(Cow::Owned(msl.source)),
                ..Default::default()
            })
        };
        if let Some(error) = pollster::block_on(guard.pop()) {
            return Err(format!(
                "{}: Metal ({stage:?} `{entry_point}`): {error}",
                shader.label
            ));
        }
        self.modules.borrow_mut().insert(key, module.clone());
        Ok(module)
    }
}
