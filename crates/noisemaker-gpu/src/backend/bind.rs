//! Bind groups and uniform buffers of the WebGPU backend (`createBindGroup`,
//! `createUniformBuffer`, `createSingleUniformBuffer`, storage buffers and
//! textures, the legacy bind group, the uniform buffer pool).

use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use noisemaker_dsl::{Object, Value};

use super::passes::BindTarget;
use super::{FrameState, Program, TextureRecord, WebGpuBackend};
use crate::error::RenderError;
use crate::graph::pass;
use crate::jsre::JsRegex;
use crate::jsv::to_js_string;
use crate::reflect::Stage;
use crate::uniforms::{default_single_uniform, pack_uniforms_with_layout, single_uniform_data};
use crate::wgsl::{BindingKind, ShaderBinding};

/// The resource of one bind-group entry before it is turned into a wgpu entry.
enum EntryResource {
    View(Rc<TextureRecord>, ViewKind),
    Dummy,
    Sampler(String),
    Buffer(wgpu::Buffer),
    StorageView(wgpu::TextureView),
}

#[derive(Clone, Copy)]
enum ViewKind {
    /// `tex.view`.
    Default,
}

/// `/^global_mesh\d+_(?:positions|normals|uvs)$/`.
fn is_mesh_data_id(id: &str) -> bool {
    JsRegex::new(r"^global_mesh\d+_(?:positions|normals|uvs)$", "").test(id)
}

/// `texId.replace(/_chain_\d+$/, '')`.
pub fn strip_chain_scope(id: &str) -> String {
    JsRegex::new(r"_chain_\d+$", "").replace_first(id, "")
}

impl WebGpuBackend {
    /// `this.textures.get(texId) || this.textures.get(unscopedTexId)`.
    fn texture_or_unscoped(&self, tex_id: &Value) -> Option<Rc<TextureRecord>> {
        let key = to_js_string(tex_id);
        if let Some(t) = self.textures.get(&key) {
            return Some(t.clone());
        }
        if tex_id.as_str().is_some() {
            return self.textures.get(&strip_chain_scope(&key)).cloned();
        }
        None
    }

    /// `createBindGroup(pass, program, state, pipeline)`.
    pub(super) fn create_bind_group(
        &mut self,
        pass: &Object,
        program: &Rc<Program>,
        state: &FrameState,
        target: Option<BindTarget<'_>>,
    ) -> Result<wgpu::BindGroup, RenderError> {
        let mut bindings: Vec<ShaderBinding> = program.bindings.clone();
        let entry_point = pass::get(pass, "entryPoint");

        // Multi-entry-point compute shaders bind only what the pass names.
        if entry_point.is_truthy() && program.is_compute {
            let mut needed: BTreeSet<String> = BTreeSet::new();
            if let Some(inputs) = pass::get(pass, "inputs").as_object() {
                needed.extend(inputs.keys().cloned());
            }
            if let Some(outputs) = pass::get(pass, "outputs").as_object() {
                needed.extend(outputs.keys().cloned());
            }
            needed.insert("params".into());
            for b in &bindings {
                if b.kind == BindingKind::Storage {
                    needed.insert(b.name.clone());
                }
            }
            bindings.retain(|b| needed.contains(&b.name));
        }

        let pass_uniforms = pass::uniforms(pass);
        let get_uniform = |name: &str| -> Value {
            if let Some(u) = pass_uniforms
                && u.contains_key(name)
            {
                return u.get_or_undefined(name).clone();
            }
            if state.global_uniforms.contains_key(name) {
                return state.global_uniforms.get_or_undefined(name).clone();
            }
            Value::Undefined
        };

        // Input name → texture view.
        let mut texture_map: HashMap<String, Rc<TextureRecord>> = HashMap::new();
        let inputs_value = pass::get(pass, "inputs");
        if let Some(inputs) = inputs_value.as_object() {
            for (input_name, tex_id) in inputs.iter() {
                let surface_name = Self::parse_global_name(tex_id);
                let unscoped_mesh_id = tex_id.as_str().map(strip_chain_scope);
                let is_mesh_data = unscoped_mesh_id.as_deref().is_some_and(is_mesh_data_id);
                let record = if is_mesh_data {
                    self.textures
                        .get(&to_js_string(tex_id))
                        .or_else(|| self.textures.get(unscoped_mesh_id.as_deref().unwrap()))
                        .cloned()
                } else if let Some(surface) = surface_name {
                    match state.surfaces.get(&surface) {
                        Some(s) => Some(s.clone()),
                        None => self.texture_or_unscoped(tex_id),
                    }
                } else {
                    self.textures.get(&to_js_string(tex_id)).cloned()
                };
                if let Some(record) = record {
                    texture_map.insert(input_name.clone(), record.clone());
                    if input_name == "inputTex" {
                        texture_map.insert("tex0".into(), record.clone());
                        texture_map.insert("inputColor".into(), record);
                    }
                }
            }
        }

        // Surface inputs sample NEAREST (WebGL2 parity); a pass reading an
        // external upload samples LINEAR, one reading a mipmapped texture uses
        // the mipmap sampler.
        let input_records: Vec<Option<Rc<TextureRecord>>> = match inputs_value.as_object() {
            Some(inputs) => inputs
                .values()
                .map(|id| self.texture_or_unscoped(id))
                .collect(),
            None => Vec::new(),
        };
        let input_sampler_default = if input_records.iter().flatten().any(|t| t.is_external) {
            "default"
        } else if input_records
            .iter()
            .flatten()
            .any(|t| t.mipmaps == Some(true))
        {
            "mipmap"
        } else {
            "nearest"
        };

        let input_keys: Vec<String> = inputs_value
            .as_object()
            .map(|o| o.keys().cloned().collect())
            .unwrap_or_default();
        let sampler_types = pass::get(pass, "samplerTypes");

        let mut entries: Vec<(u32, EntryResource)> = Vec::new();
        for binding in &bindings {
            if binding.group != 0 {
                continue;
            }
            match binding.kind {
                BindingKind::Texture => {
                    let mut view = texture_map.get(&binding.name).cloned();
                    if view.is_none() && binding.name.starts_with("tex") {
                        let idx = noisemaker_dsl::js::parse_int(&binding.name[3..], 10);
                        if !idx.is_nan() && idx < input_keys.len() as f64 {
                            // `idx` may be negative (`tex-1`): no such key.
                            if idx >= 0.0 {
                                view = texture_map.get(&input_keys[idx as usize]).cloned();
                            }
                        }
                    }
                    entries.push((
                        binding.binding,
                        match view {
                            Some(record) => EntryResource::View(record, ViewKind::Default),
                            None => EntryResource::Dummy,
                        },
                    ));
                }
                BindingKind::Sampler => {
                    let explicit = sampler_types.get(&binding.name);
                    let kind = if explicit.is_truthy() {
                        to_js_string(explicit)
                    } else {
                        input_sampler_default.to_owned()
                    };
                    entries.push((binding.binding, EntryResource::Sampler(kind)));
                }
                BindingKind::Uniform => {
                    let decl = binding.type_decl.as_str();
                    let is_struct = !decl.is_empty()
                        && !decl.contains('<')
                        && decl != "f32"
                        && decl != "i32"
                        && decl != "u32"
                        && decl != "bool"
                        && !decl.starts_with("vec")
                        && !decl.starts_with("mat");
                    if is_struct {
                        if let Some(buffer) =
                            self.create_uniform_buffer(pass, state, Some(program))?
                        {
                            entries.push((binding.binding, EntryResource::Buffer(buffer)));
                        }
                    } else {
                        let mut value = get_uniform(&binding.name);
                        if !matches!(value, Value::Number(_) | Value::Bool(_) | Value::Array(_)) {
                            value = default_single_uniform(decl);
                        }
                        if let Some(buffer) = self.create_single_uniform_buffer(&value, decl) {
                            self.active_uniform_buffers.push(buffer.clone());
                            entries.push((binding.binding, EntryResource::Buffer(buffer)));
                        }
                    }
                }
                BindingKind::Storage => {
                    let buffer = self.create_storage_buffer(binding, state);
                    entries.push((binding.binding, EntryResource::Buffer(buffer)));
                }
                BindingKind::StorageTexture => {
                    if let Some(view) = self.create_storage_texture_view(binding, pass, state) {
                        entries.push((binding.binding, EntryResource::StorageView(view)));
                    }
                }
                BindingKind::Unknown => {}
            }
        }

        let (layout, layout_bindings) = match &target {
            Some(BindTarget::Render(p)) => (
                p.get_bind_group_layout(0),
                program
                    .render()
                    .map(|r| r.layout_bindings.clone())
                    .unwrap_or_default(),
            ),
            Some(BindTarget::Compute(p, ep)) => (
                p.get_bind_group_layout(0),
                program
                    .compute()
                    .and_then(|c| c.reflection.used_bindings(Stage::Compute, ep, 0))
                    .unwrap_or_default(),
            ),
            None => match &program.kind {
                super::ProgramKind::Render(r) => (
                    r.pipeline.get_bind_group_layout(0),
                    r.layout_bindings.clone(),
                ),
                super::ProgramKind::Compute(c) => (
                    c.pipeline.get_bind_group_layout(0),
                    c.reflection
                        .used_bindings(Stage::Compute, &c.entry_point, 0)
                        .unwrap_or_default(),
                ),
            },
        };

        if bindings.is_empty() {
            if !program.source_has_bindings {
                return self.create_legacy_bind_group(pass, program, state);
            }
            return Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&program.id),
                layout: &layout,
                entries: &[],
            }));
        }
        Ok(self.create_bind_group_from_entries(&program.id, &layout, &layout_bindings, entries))
    }

    /// `createBindGroupFromEntries(layout, entries)`. The reference drops an entry
    /// the auto layout does not contain and retries; here the layout's binding set
    /// is known from reflection, so such entries are dropped up front.
    fn create_bind_group_from_entries(
        &mut self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        layout_bindings: &BTreeSet<u32>,
        entries: Vec<(u32, EntryResource)>,
    ) -> wgpu::BindGroup {
        let mut kept: Vec<(u32, EntryResource)> = Vec::with_capacity(entries.len());
        for (binding, resource) in entries {
            if layout_bindings.contains(&binding) {
                kept.push((binding, resource));
            } else {
                let mut record = Object::new();
                record.insert("code", Value::from("ERR_BINDING_NOT_IN_LAYOUT"));
                record.insert("backend", Value::from("webgpu"));
                record.insert("stage", Value::from("bind"));
                record.insert("program", Value::from(label));
                record.insert("bindingIndex", Value::from(binding));
                self.diagnostics.add(Value::Object(record));
                self.dropped_binding_count += 1;
            }
        }
        let dummy = self.dummy_texture_view.clone();
        let wgpu_entries: Vec<wgpu::BindGroupEntry> = kept
            .iter()
            .map(|(binding, resource)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: match resource {
                    EntryResource::View(record, ViewKind::Default) => {
                        wgpu::BindingResource::TextureView(&record.view)
                    }
                    EntryResource::Dummy => wgpu::BindingResource::TextureView(
                        dummy.as_ref().expect("init() creates the dummy texture"),
                    ),
                    EntryResource::Sampler(kind) => {
                        wgpu::BindingResource::Sampler(self.sampler(kind))
                    }
                    EntryResource::Buffer(buffer) => buffer.as_entire_binding(),
                    EntryResource::StorageView(view) => wgpu::BindingResource::TextureView(view),
                },
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &wgpu_entries,
        })
    }

    /// `getBufferFromPool(requiredSize)`: the first pooled buffer at least that big.
    fn get_buffer_from_pool(&mut self, required: u64) -> Option<wgpu::Buffer> {
        let index = self
            .uniform_buffer_pool
            .iter()
            .position(|b| b.size() >= required)?;
        Some(self.uniform_buffer_pool.remove(index))
    }

    fn pooled_uniform_buffer(&mut self, size: u64) -> wgpu::Buffer {
        match self.get_buffer_from_pool(size) {
            Some(buffer) => buffer,
            None => self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("uniforms"),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        }
    }

    /// `createSingleUniformBuffer(value, typeDecl)`.
    fn create_single_uniform_buffer(
        &mut self,
        value: &Value,
        type_decl: &str,
    ) -> Option<wgpu::Buffer> {
        let data = single_uniform_data(value, type_decl)?;
        let size = (data.len() as u64).max(16);
        let buffer = self.pooled_uniform_buffer(size);
        if !data.is_empty() {
            self.queue.write_buffer(&buffer, 0, &data);
        }
        Some(buffer)
    }

    /// `createUniformBuffer(pass, state, program)`: merge the pass uniforms and the
    /// global uniforms into the persistent merge object, pack them (with the
    /// program's layout when it has one), and upload into a pooled buffer at
    /// least `max(data, declared struct size, 16)` bytes big.
    fn create_uniform_buffer(
        &mut self,
        pass: &Object,
        state: &FrameState,
        program: Option<&Rc<Program>>,
    ) -> Result<Option<wgpu::Buffer>, RenderError> {
        for key in self.merged_uniform_keys.drain(..) {
            self.merged_uniforms.insert(key, Value::Undefined);
        }
        let pass_uniforms = pass::uniforms(pass);
        if let Some(uniforms) = pass_uniforms {
            for (key, val) in uniforms.iter() {
                if !val.is_undefined() {
                    self.merged_uniforms.insert(key.clone(), val.clone());
                    self.merged_uniform_keys.push(key.clone());
                }
            }
        }
        for (key, val) in state.global_uniforms.iter() {
            if pass_uniforms.is_some_and(|u| u.contains_key(key)) {
                continue;
            }
            if !val.is_undefined() {
                if self.merged_uniforms.get_or_undefined(key).is_undefined() {
                    self.merged_uniform_keys.push(key.clone());
                }
                self.merged_uniforms.insert(key.clone(), val.clone());
            }
        }
        if self.merged_uniform_keys.is_empty() {
            return Ok(None);
        }
        let layout = program.and_then(|p| p.packed_uniform_layout.as_ref());
        let data = match layout {
            Some(layout) => pack_uniforms_with_layout(&self.merged_uniforms, layout)?,
            None => self.pack_scratch.pack_uniforms(&self.merged_uniforms)?,
        };
        let declared = program.map(|p| p.declared_uniform_buffer_size).unwrap_or(0);
        let size = (data.len() as u64).max(declared).max(16);
        let buffer = self.pooled_uniform_buffer(size);
        if !data.is_empty() {
            self.queue.write_buffer(&buffer, 0, &data);
        }
        self.active_uniform_buffers.push(buffer.clone());
        Ok(Some(buffer))
    }

    /// `createStorageBuffer(binding, pass, state)`: one buffer per binding name for
    /// the backend's lifetime, sized for the screen.
    fn create_storage_buffer(
        &mut self,
        binding: &ShaderBinding,
        state: &FrameState,
    ) -> wgpu::Buffer {
        if let Some(buffer) = self.storage_buffers.get(&binding.name) {
            return buffer.clone();
        }
        let name = binding.name.as_str();
        let width = if state.screen_width != 0.0 {
            state.screen_width
        } else {
            1280.0
        };
        let height = if state.screen_height != 0.0 {
            state.screen_height
        } else {
            720.0
        };
        let mut byte_size = if name == "output_buffer" || name == "outputBuffer" {
            width * height * 4.0 * 4.0
        } else if name == "stats_buffer" {
            let workgroups = (width / 8.0).ceil() * (height / 8.0).ceil();
            (2.0 + workgroups * 2.0) * 4.0
        } else if name.contains("downsample") {
            (width / 4.0).ceil() * (height / 4.0).ceil() * 4.0 * 4.0
        } else {
            width * height * 4.0 * 4.0
        };
        byte_size = byte_size.max(256.0);
        byte_size = (byte_size / 256.0).ceil() * 256.0;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(name),
            size: byte_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.storage_buffers
            .insert(binding.name.clone(), buffer.clone());
        buffer
    }

    /// `createStorageTextureView(binding, pass, state)`.
    fn create_storage_texture_view(
        &mut self,
        binding: &ShaderBinding,
        pass: &Object,
        state: &FrameState,
    ) -> Option<wgpu::TextureView> {
        let mapping = pass::get(pass, "storageTextures");
        let texture_id = mapping.get(&binding.name);
        if !texture_id.is_truthy() {
            if binding.name == "output_texture" {
                return Some(self.get_output_storage_view(state));
            }
            return None;
        }
        let texture_id = to_js_string(texture_id);
        if texture_id == "outputTex" {
            return Some(self.get_output_storage_view(state));
        }
        if JsRegex::new(r"^o[0-7]$", "").test(&texture_id)
            && let Some(write) = state.write_surface(&texture_id)
            && let Some(texture) = self.textures.get(&write)
        {
            return Some(texture.render_or_view().clone());
        }
        if let Some(surface) = Self::parse_global_name(&Value::from(texture_id.as_str()))
            && let Some(write) = state.write_surface(&surface)
            && let Some(texture) = self.textures.get(&write)
        {
            return Some(texture.render_or_view().clone());
        }
        self.textures
            .get(&texture_id)
            .map(|texture| texture.render_or_view().clone())
    }

    /// `getOutputStorageView(state)`: the render surface's write texture, else a
    /// screen-sized fallback storage texture.
    fn get_output_storage_view(&mut self, state: &FrameState) -> wgpu::TextureView {
        if let Some(render_surface) = state.render_surface.as_str().filter(|s| !s.is_empty())
            && let Some(write) = state.write_surface(render_surface)
            && let Some(texture) = self.textures.get(&write)
        {
            return texture.render_or_view().clone();
        }
        let render_surface = match &state.render_surface {
            Value::Undefined | Value::Null => Value::Null,
            name => name.clone(),
        };
        self.record_missing_render_target("storage-surface", render_surface, Value::Null);
        let width = if state.screen_width != 0.0 {
            state.screen_width
        } else {
            1280.0
        } as u32;
        let height = if state.screen_height != 0.0 {
            state.screen_height
        } else {
            720.0
        } as u32;
        let key = format!("outputStorage_{width}x{height}");
        if let Some((_, view)) = self.storage_textures.get(&key) {
            return view.clone();
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&key),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        self.storage_textures.insert(key, (texture, view.clone()));
        view
    }

    /// `createLegacyBindGroup(pass, program, state)`: texture/sampler pairs for every
    /// input, then the uniform buffer, at consecutive bindings, against the
    /// program's compile-time pipeline layout.
    fn create_legacy_bind_group(
        &mut self,
        pass: &Object,
        program: &Rc<Program>,
        state: &FrameState,
    ) -> Result<wgpu::BindGroup, RenderError> {
        let mut entries: Vec<(u32, EntryResource)> = Vec::new();
        let mut binding = 0u32;
        let sampler_types = pass::get(pass, "samplerTypes");
        if let Some(inputs) = pass::get(pass, "inputs").as_object() {
            for (sampler_name, tex_id) in inputs.iter() {
                let view = if tex_id.as_str() == Some("none") {
                    None
                } else if let Some(surface) = Self::parse_global_name(tex_id) {
                    state.surfaces.get(&surface).cloned()
                } else {
                    self.textures.get(&to_js_string(tex_id)).cloned()
                };
                entries.push((
                    binding,
                    match view {
                        Some(record) => EntryResource::View(record, ViewKind::Default),
                        None => EntryResource::Dummy,
                    },
                ));
                binding += 1;
                let rec = self.texture_or_unscoped(tex_id);
                let legacy_default = match &rec {
                    Some(r) if r.is_external => "default",
                    Some(r) if r.is_3d => {
                        if r.filter.as_str() == Some("linear") {
                            "default"
                        } else {
                            "nearest"
                        }
                    }
                    Some(r) if r.mipmaps == Some(true) => "mipmap",
                    _ => "nearest",
                };
                let explicit = sampler_types.get(sampler_name);
                let kind = if explicit.is_truthy() {
                    to_js_string(explicit)
                } else {
                    legacy_default.to_owned()
                };
                entries.push((binding, EntryResource::Sampler(kind)));
                binding += 1;
            }
        }
        if let Some(buffer) = self.create_uniform_buffer(pass, state, None)? {
            entries.push((binding, EntryResource::Buffer(buffer)));
        }
        let (layout, layout_bindings) = match &program.kind {
            super::ProgramKind::Render(r) => (
                r.pipeline.get_bind_group_layout(0),
                r.layout_bindings.clone(),
            ),
            super::ProgramKind::Compute(c) => (
                c.pipeline.get_bind_group_layout(0),
                c.reflection
                    .used_bindings(Stage::Compute, &c.entry_point, 0)
                    .unwrap_or_default(),
            ),
        };
        let _ = layout_bindings;
        // The legacy path hands every entry to the layout (no retry filter).
        let all: BTreeSet<u32> = entries.iter().map(|(b, _)| *b).collect();
        Ok(self.create_bind_group_from_entries(&program.id, &layout, &all, entries))
    }
}
