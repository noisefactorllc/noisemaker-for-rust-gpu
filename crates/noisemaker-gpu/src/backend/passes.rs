//! Pass execution of the WebGPU backend (`executePass` and its render, MRT,
//! 3D-mesh and compute paths, pipeline caches, draw counts and dispatch sizes).

use std::rc::Rc;

use noisemaker_dsl::{Object, Value};

use super::programs::topology_of;
use super::{
    DEFAULT_FRAGMENT_ENTRY_POINT, FrameState, Program, TextureRecord, WebGpuBackend, gpu_format_of,
};
use crate::error::RenderError;
use crate::graph::pass;
use crate::jsre::JsRegex;
use crate::jsv::{enforce_range_u32, to_js_string, to_number};

/// A resolved viewport `{x, y, w, h}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viewport {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// `passEncoder.draw(count, ...)` / `dispatchWorkgroups(...)` argument conversion
/// (`[EnforceRange] GPUSize32`).
fn gpu_size32(v: &Value, what: &str) -> Result<u32, RenderError> {
    enforce_range_u32(v)
        .map_err(|e| RenderError::type_error(format!("Failed to execute '{what}': {e}")))
}

/// A blend factor name (`'one'`, `'one-minus-src-alpha'`, ...) → wgpu factor.
fn blend_factor(name: &str) -> Result<wgpu::BlendFactor, RenderError> {
    use wgpu::BlendFactor as B;
    Ok(match name {
        "zero" => B::Zero,
        "one" => B::One,
        "src" => B::Src,
        "one-minus-src" => B::OneMinusSrc,
        "src-alpha" => B::SrcAlpha,
        "one-minus-src-alpha" => B::OneMinusSrcAlpha,
        "dst" => B::Dst,
        "one-minus-dst" => B::OneMinusDst,
        "dst-alpha" => B::DstAlpha,
        "one-minus-dst-alpha" => B::OneMinusDstAlpha,
        "src-alpha-saturated" => B::SrcAlphaSaturated,
        "constant" => B::Constant,
        "one-minus-constant" => B::OneMinusConstant,
        "src1" => B::Src1,
        "one-minus-src1" => B::OneMinusSrc1,
        "src1-alpha" => B::Src1Alpha,
        "one-minus-src1-alpha" => B::OneMinusSrc1Alpha,
        other => {
            return Err(RenderError::type_error(format!(
                "The provided value '{other}' is not a valid enum value of type GPUBlendFactor."
            )));
        }
    })
}

impl WebGpuBackend {
    /// `resolveBlendState(blend)`: none for a falsy blend; `[src, dst]` arrays map
    /// WebGL-style names (`ONE_MINUS_SRC_ALPHA`) to WebGPU names; anything else is
    /// additive (`one`, `one`).
    pub fn resolve_blend_state(blend: &Value) -> Result<Option<wgpu::BlendState>, RenderError> {
        if !blend.is_truthy() {
            return Ok(None);
        }
        let (src, dst) = if let Value::Array(items) = blend {
            let to_factor = |f: &Value| match f {
                Value::String(s) => {
                    let mapped = s.to_lowercase().replace('_', "-");
                    if mapped.is_empty() {
                        None
                    } else {
                        Some(mapped)
                    }
                }
                _ => None,
            };
            let src = items
                .first()
                .and_then(to_factor)
                .unwrap_or_else(|| "one".to_owned());
            let dst = items
                .get(1)
                .and_then(to_factor)
                .unwrap_or_else(|| "one".to_owned());
            (src, dst)
        } else {
            ("one".to_owned(), "one".to_owned())
        };
        let component = wgpu::BlendComponent {
            src_factor: blend_factor(&src)?,
            dst_factor: blend_factor(&dst)?,
            operation: wgpu::BlendOperation::Add,
        };
        Ok(Some(wgpu::BlendState {
            color: component,
            alpha: component,
        }))
    }

    fn blend_key(blend: &Value) -> String {
        if blend.is_truthy() {
            blend.to_json().unwrap_or_else(|| "undefined".into())
        } else {
            "noblend".into()
        }
    }

    /// `getPipelineKey({blend, topology, format})`.
    pub fn get_pipeline_key(blend: &Value, topology: &Value, format: &str) -> String {
        let topo = if topology.is_truthy() {
            to_js_string(topology)
        } else {
            "triangle-list".into()
        };
        let format = if format.is_empty() {
            "rgba16float"
        } else {
            format
        };
        format!("{topo}|{}|{format}", Self::blend_key(blend))
    }

    fn create_render_pipeline_for(
        &self,
        program: &Program,
        label: &str,
        targets: &[Option<wgpu::ColorTargetState>],
        primitive: wgpu::PrimitiveState,
        depth_stencil: Option<wgpu::DepthStencilState>,
    ) -> wgpu::RenderPipeline {
        let render = program.render().expect("render program");
        self.create_render_pipeline_from(
            label,
            (&render.vertex_module, &render.vertex_entry_point),
            Some((
                &render.fragment_module,
                if render.fragment_entry_point.is_empty() {
                    DEFAULT_FRAGMENT_ENTRY_POINT
                } else {
                    &render.fragment_entry_point
                },
            )),
            targets,
            primitive,
            depth_stencil,
        )
    }

    /// The byte size of the storage buffer `program` binds at
    /// `(group, binding)` (0 when none is bound).
    fn program_storage_size(&self, program: &Program, group: u32, binding: u32) -> u64 {
        program
            .bindings
            .iter()
            .find(|b| {
                b.group == group
                    && b.binding == binding
                    && b.kind == crate::wgsl::BindingKind::Storage
            })
            .and_then(|b| self.storage_buffers.get(&b.name))
            .map_or(0, wgpu::Buffer::size)
    }

    /// The immediate data a Tint-path render pipeline of `program` reads
    /// ([`super::shaders`]), if any.
    fn render_immediates_for(
        &self,
        pipeline: &wgpu::RenderPipeline,
        program: &Program,
    ) -> Option<Vec<u8>> {
        self.render_immediate_block(pipeline)
            .map(|block| block.data(|g, b| self.program_storage_size(program, g, b)))
    }

    /// `resolveRenderPipeline(program, {blend, topology, format})`.
    fn resolve_render_pipeline(
        &self,
        program: &Program,
        blend: &Value,
        topology: &str,
        format: &str,
    ) -> Result<wgpu::RenderPipeline, RenderError> {
        let render = program.render().expect("render program");
        let key = Self::get_pipeline_key(blend, &Value::from(topology), format);
        if let Some(p) = render.pipeline_cache.borrow().get(&key) {
            return Ok(p.clone());
        }
        let target_format = if !format.is_empty() {
            Value::from(format)
        } else if render.output_format.is_truthy() {
            render.output_format.clone()
        } else {
            Value::from("rgba16float")
        };
        let pipeline = self.create_render_pipeline_for(
            program,
            &program.id,
            &[Some(wgpu::ColorTargetState {
                format: gpu_format_of(&target_format)?,
                blend: Self::resolve_blend_state(blend)?,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            wgpu::PrimitiveState {
                topology: topology_of(if topology.is_empty() {
                    "triangle-list"
                } else {
                    topology
                })?,
                ..Default::default()
            },
            None,
        );
        render
            .pipeline_cache
            .borrow_mut()
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// `resolveMRTRenderPipeline(program, {blend, topology, formats})`.
    fn resolve_mrt_render_pipeline(
        &self,
        program: &Program,
        blend: &Value,
        topology: &str,
        formats: &[Value],
    ) -> Result<wgpu::RenderPipeline, RenderError> {
        let render = program.render().expect("render program");
        let topology = if topology.is_empty() {
            "triangle-list"
        } else {
            topology
        };
        let joined = formats
            .iter()
            .map(to_js_string)
            .collect::<Vec<_>>()
            .join("_");
        let key = format!("mrt_{topology}_{joined}_{}", Self::blend_key(blend));
        if let Some(p) = render.pipeline_cache.borrow().get(&key) {
            return Ok(p.clone());
        }
        let blend_state = Self::resolve_blend_state(blend)?;
        let targets = formats
            .iter()
            .map(|f| {
                Ok(Some(wgpu::ColorTargetState {
                    format: gpu_format_of(f)?,
                    blend: blend_state,
                    write_mask: wgpu::ColorWrites::ALL,
                }))
            })
            .collect::<Result<Vec<_>, RenderError>>()?;
        let pipeline = self.create_render_pipeline_for(
            program,
            &program.id,
            &targets,
            wgpu::PrimitiveState {
                topology: topology_of(topology)?,
                ..Default::default()
            },
            None,
        );
        render
            .pipeline_cache
            .borrow_mut()
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// `resolve3DRenderPipeline(program, {blend, format})`: depth-tested
    /// (`depth24plus`, `less`), back-face culled, clockwise front faces.
    fn resolve_3d_render_pipeline(
        &self,
        program: &Program,
        blend: &Value,
        format: &str,
    ) -> Result<wgpu::RenderPipeline, RenderError> {
        let render = program.render().expect("render program");
        let key_format = if format.is_empty() {
            "rgba16float"
        } else {
            format
        };
        let key = format!("3d|{key_format}|{}", Self::blend_key(blend));
        if let Some(p) = render.pipeline_cache.borrow().get(&key) {
            return Ok(p.clone());
        }
        let target_format = if !format.is_empty() {
            Value::from(format)
        } else if render.output_format.is_truthy() {
            render.output_format.clone()
        } else {
            Value::from("rgba16float")
        };
        let pipeline = self.create_render_pipeline_for(
            program,
            &program.id,
            &[Some(wgpu::ColorTargetState {
                format: gpu_format_of(&target_format)?,
                blend: Self::resolve_blend_state(blend)?,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Cw,
                ..Default::default()
            },
            Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
        );
        render
            .pipeline_cache
            .borrow_mut()
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// `getDepthTexture(width, height)`.
    fn get_depth_texture(&mut self, width: u32, height: u32) -> wgpu::Texture {
        if self.depth_texture.is_none() || self.depth_texture_size != (width, height) {
            if let Some(old) = self.depth_texture.take() {
                old.destroy();
            }
            self.depth_texture = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth24Plus,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }));
            self.depth_texture_size = (width, height);
        }
        self.depth_texture.clone().unwrap()
    }

    /// `executePass(pass, state)`.
    pub fn execute_pass(&mut self, pass: &Object, state: &FrameState) -> Result<(), RenderError> {
        let program_id = pass::program(pass);
        let Some(program) = self.programs.get(&program_id).cloned() else {
            return Err(RenderError::thrown(
                "ERR_PROGRAM_NOT_FOUND",
                &[
                    ("pass", pass::get(pass, "id").clone()),
                    ("program", pass::get(pass, "program").clone()),
                ],
            ));
        };
        let result = if program.is_compute {
            self.execute_compute_pass(pass, &program, state)
        } else {
            self.execute_render_pass(pass, &program, state)
        };
        self.collect_device_errors();
        result
    }

    /// Resolve an output id to the current write texture of its global surface.
    fn resolve_output_id(output_id: &Value, state: &FrameState) -> Value {
        if let Some(surface) = Self::parse_global_name(output_id)
            && let Some(write) = state.write_surface(&surface)
        {
            return Value::from(write);
        }
        output_id.clone()
    }

    /// `this.textures.get(id) || state.surfaces?.[id]`.
    fn texture_or_surface(&self, id: &Value, state: &FrameState) -> Option<Rc<TextureRecord>> {
        let key = to_js_string(id);
        self.textures
            .get(&key)
            .cloned()
            .or_else(|| state.surfaces.get(&key).cloned())
    }

    /// `executeRenderPass(pass, program, state)`.
    fn execute_render_pass(
        &mut self,
        pass: &Object,
        program: &Rc<Program>,
        state: &FrameState,
    ) -> Result<(), RenderError> {
        let outputs_value = pass::get(pass, "outputs");
        let output_keys: Vec<String> = match outputs_value {
            Value::Object(o) => o.keys().cloned().collect(),
            Value::Array(a) => (0..a.len()).map(|i| i.to_string()).collect(),
            _ => Vec::new(),
        };
        let is_mrt = to_number(pass::get(pass, "drawBuffers")) > 1.0 || output_keys.len() > 1;
        if is_mrt {
            return self.execute_mrt_render_pass(pass, program, state, &output_keys);
        }

        let Some(outputs) = outputs_value.as_object() else {
            return Err(RenderError::type_error(format!(
                "Cannot read properties of {} (reading 'color')",
                if outputs_value.is_null() {
                    "null"
                } else {
                    "undefined"
                }
            )));
        };
        let color = outputs.get_or_undefined("color");
        let output_value = if color.is_truthy() {
            color.clone()
        } else {
            outputs.values().next().cloned().unwrap_or(Value::Undefined)
        };
        let output_id = Self::resolve_output_id(&output_value, state);
        let Some(output_tex) = self.texture_or_surface(&output_id, state) else {
            return Err(RenderError::thrown(
                "ERR_TEXTURE_NOT_FOUND",
                &[
                    ("pass", pass::get(pass, "id").clone()),
                    ("texture", output_id.clone()),
                ],
            ));
        };
        let viewport = Self::resolve_viewport(pass, &output_tex);
        let load = if pass::get(pass, "clear").is_truthy() {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };
        let resolved_format = match &output_tex.gpu_format {
            Some(f) if !f.is_empty() => f.clone(),
            _ if output_tex.format.is_truthy() => to_js_string(&output_tex.format),
            _ => to_js_string(&program.render().unwrap().output_format),
        };
        let draw_mode = pass::draw_mode(pass);
        let blend = pass::get(pass, "blend");
        let is_3d = draw_mode == Some("triangles");

        let (pipeline, depth_view) = if is_3d {
            let width = enforce_range_u32(&Value::Number(output_tex.width))
                .map_err(RenderError::type_error)?;
            let height = enforce_range_u32(&Value::Number(output_tex.height))
                .map_err(RenderError::type_error)?;
            let depth = self.get_depth_texture(width, height);
            let view = depth.create_view(&Default::default());
            (
                self.resolve_3d_render_pipeline(program, blend, &resolved_format)?,
                Some(view),
            )
        } else {
            let topology = if draw_mode == Some("points") {
                "point-list"
            } else {
                "triangle-list"
            };
            (
                self.resolve_render_pipeline(program, blend, topology, &resolved_format)?,
                None,
            )
        };

        let bind_group =
            self.create_bind_group(pass, program, state, Some(BindTarget::Render(&pipeline)))?;
        let immediates = self.render_immediates_for(&pipeline, program);

        let vertex_count = match draw_mode {
            Some("points") => gpu_size32(
                &self.resolve_point_count(pass, state, &output_id, Some(&output_tex)),
                "draw",
            )?,
            Some("billboards") => {
                let count = to_number(&self.resolve_point_count(
                    pass,
                    state,
                    &output_id,
                    Some(&output_tex),
                ));
                gpu_size32(&Value::Number(count * 6.0), "draw")?
            }
            Some("triangles") => gpu_size32(&self.resolve_mesh_vertex_count(pass, state), "draw")?,
            _ => 3,
        };

        let encoder = self.command_encoder.as_mut().ok_or_else(|| {
            RenderError::type_error("Cannot read properties of null (reading 'beginRenderPass')")
        })?;
        let target_view = output_tex.render_or_view();
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&program.id),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth_view.as_ref().map(|view| {
                wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_pipeline(&pipeline);
        if let Some(data) = &immediates {
            render_pass.set_immediates(0, data);
        }
        render_pass.set_bind_group(0, &bind_group, &[]);
        if let Some(vp) = viewport {
            render_pass.set_viewport(vp.x as f32, vp.y as f32, vp.w as f32, vp.h as f32, 0.0, 1.0);
        }
        render_pass.draw(0..vertex_count, 0..1);
        Ok(())
    }

    /// `executeMRTRenderPass(pass, program, state, outputKeys)`.
    fn execute_mrt_render_pass(
        &mut self,
        pass: &Object,
        program: &Rc<Program>,
        state: &FrameState,
        output_keys: &[String],
    ) -> Result<(), RenderError> {
        let outputs = pass::get(pass, "outputs");
        let mut targets: Vec<Rc<TextureRecord>> = Vec::new();
        let mut formats: Vec<Value> = Vec::new();
        for key in output_keys {
            let output_id = Self::resolve_output_id(outputs.get(key), state);
            let Some(tex) = self.texture_or_surface(&output_id, state) else {
                continue;
            };
            formats.push(tex.resolved_gpu_format());
            targets.push(tex);
        }
        if targets.is_empty() {
            return Err(RenderError::thrown(
                "ERR_NO_MRT_OUTPUTS",
                &[("pass", pass::get(pass, "id").clone())],
            ));
        }
        let draw_mode = pass::draw_mode(pass);
        let topology = if draw_mode == Some("points") {
            "point-list"
        } else {
            "triangle-list"
        };
        let pipeline = self.resolve_mrt_render_pipeline(
            program,
            pass::get(pass, "blend"),
            topology,
            &formats,
        )?;
        let viewport_tex = targets[0].clone();
        let load = if pass::get(pass, "clear").is_truthy() {
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT)
        } else {
            wgpu::LoadOp::Load
        };

        let bind_group =
            self.create_bind_group(pass, program, state, Some(BindTarget::Render(&pipeline)))?;
        let immediates = self.render_immediates_for(&pipeline, program);
        let vertex_count = match draw_mode {
            Some("points") => gpu_size32(
                &self.resolve_point_count(pass, state, &Value::Null, Some(&viewport_tex)),
                "draw",
            )?,
            Some("billboards") => {
                let count = to_number(&self.resolve_point_count(
                    pass,
                    state,
                    &Value::Null,
                    Some(&viewport_tex),
                ));
                gpu_size32(&Value::Number(count * 6.0), "draw")?
            }
            Some("triangles") => gpu_size32(&self.resolve_mesh_vertex_count(pass, state), "draw")?,
            _ => 3,
        };

        let encoder = self.command_encoder.as_mut().ok_or_else(|| {
            RenderError::type_error("Cannot read properties of null (reading 'beginRenderPass')")
        })?;
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = targets
            .iter()
            .map(|t| {
                Some(wgpu::RenderPassColorAttachment {
                    view: t.render_or_view(),
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(&program.id),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        render_pass.set_viewport(
            0.0,
            0.0,
            viewport_tex.width as f32,
            viewport_tex.height as f32,
            0.0,
            1.0,
        );
        render_pass.set_pipeline(&pipeline);
        if let Some(data) = &immediates {
            render_pass.set_immediates(0, data);
        }
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..vertex_count, 0..1);
        Ok(())
    }

    /// `resolveViewport(pass, tex)`: the texture's full size, else the pass's
    /// resolved (or raw) viewport.
    pub fn resolve_viewport(pass: &Object, tex: &TextureRecord) -> Option<Viewport> {
        if tex.has_size() {
            return Some(Viewport {
                x: 0.0,
                y: 0.0,
                w: tex.width,
                h: tex.height,
            });
        }
        let resolved = pass::get(pass, "viewportResolved");
        let viewport = if resolved.is_truthy() {
            resolved
        } else {
            pass::get(pass, "viewport")
        };
        if viewport.is_truthy() {
            return Some(Viewport {
                x: to_number(viewport.get("x")),
                y: to_number(viewport.get("y")),
                w: to_number(viewport.get("w")),
                h: to_number(viewport.get("h")),
            });
        }
        None
    }

    /// `resolvePointCount(pass, state, outputId, outputTex)`.
    pub fn resolve_point_count(
        &self,
        pass: &Object,
        state: &FrameState,
        output_id: &Value,
        output_tex: Option<&Rc<TextureRecord>>,
    ) -> Value {
        let count_value = pass::get(pass, "count");
        let mut count = if count_value.is_truthy() {
            count_value.clone()
        } else {
            Value::Number(1000.0)
        };
        let keyword = count.as_str().map(str::to_owned);
        if matches!(keyword.as_deref(), Some("auto" | "screen" | "input")) {
            let inputs = pass::get(pass, "inputs");
            let ref_tex: Option<Rc<TextureRecord>> =
                if keyword.as_deref() == Some("input") && inputs.is_truthy() {
                    let xyz = inputs.get("xyzTex");
                    let state_input = if xyz.is_truthy() {
                        xyz
                    } else {
                        inputs.get("inputTex")
                    };
                    if state_input.is_truthy() {
                        match Self::parse_global_name(state_input) {
                            Some(surface) => state.surfaces.get(&surface).cloned(),
                            None => self.textures.get(&to_js_string(state_input)).cloned(),
                        }
                    } else {
                        None
                    }
                } else {
                    output_tex
                        .cloned()
                        .or_else(|| self.textures.get(&to_js_string(output_id)).cloned())
                };
            if let Some(tex) = ref_tex
                && tex.has_size()
            {
                count = Value::Number(tex.width * tex.height);
            }
        }
        count
    }

    /// `resolveMeshVertexCount(pass, state)`.
    pub fn resolve_mesh_vertex_count(&self, pass: &Object, state: &FrameState) -> Value {
        let count_value = pass::get(pass, "count");
        let mut count = if count_value.is_truthy() {
            count_value.clone()
        } else {
            Value::Number(3.0)
        };
        let count_uniform = pass::get(pass, "countUniform");
        if count_uniform.is_truthy() {
            let name = to_js_string(count_uniform);
            let mut value = pass::uniforms(pass)
                .map(|u| u.get_or_undefined(&name).clone())
                .unwrap_or(Value::Undefined);
            if value.is_undefined() {
                value = state.global_uniforms.get_or_undefined(&name).clone();
            }
            if let Value::Number(n) = value
                && n > 0.0
            {
                return Value::Number(n);
            }
        }
        let is_auto = matches!(count.as_str(), Some("auto" | "input"))
            || match &count {
                Value::String(_) => false,
                other => to_number(other) <= 0.0,
            };
        if is_auto {
            let mut ref_tex: Option<Rc<TextureRecord>> = None;
            if let Some(inputs) = pass::inputs(pass) {
                let positions = inputs.get_or_undefined("meshPositions");
                let mesh_input = if positions.is_truthy() {
                    positions
                } else {
                    inputs.get_or_undefined("inputTex")
                };
                if mesh_input.is_truthy() {
                    let id = to_js_string(mesh_input);
                    ref_tex = self.textures.get(&id).cloned();
                    if ref_tex.is_none() {
                        let unscoped = JsRegex::new(r"_chain_\d+$", "").replace_first(&id, "");
                        if unscoped != id {
                            ref_tex = self.textures.get(&unscoped).cloned();
                        }
                    }
                    if ref_tex.is_none()
                        && let Some(surface) = Self::parse_global_name(mesh_input)
                    {
                        ref_tex = state.surfaces.get(&surface).cloned();
                    }
                }
            }
            count = match ref_tex {
                Some(tex) if tex.has_size() => Value::Number(tex.width * tex.height),
                _ => Value::Number(3.0),
            };
        }
        count
    }

    /// `getComputePipeline(program, entryPoint)`.
    fn get_compute_pipeline(
        &self,
        program: &Program,
        entry_point: &Value,
    ) -> wgpu::ComputePipeline {
        let compute = program.compute().expect("compute program");
        let target = if entry_point.is_truthy() {
            to_js_string(entry_point)
        } else if !compute.entry_point.is_empty() {
            compute.entry_point.clone()
        } else {
            "main".into()
        };
        if let Some(p) = compute.pipelines.borrow().get(&target) {
            return p.clone();
        }
        let pipeline = self.create_compute_pipeline_from(&program.id, &compute.module, &target);
        compute
            .pipelines
            .borrow_mut()
            .insert(target, pipeline.clone());
        pipeline
    }

    /// The entry point a compute pass runs (`pass.entryPoint || program.entryPoint || 'main'`).
    pub(super) fn compute_entry_point(program: &Program, entry_point: &Value) -> String {
        let compute = program.compute().expect("compute program");
        if entry_point.is_truthy() {
            to_js_string(entry_point)
        } else if !compute.entry_point.is_empty() {
            compute.entry_point.clone()
        } else {
            "main".into()
        }
    }

    /// `executeComputePass(pass, program, state)`.
    fn execute_compute_pass(
        &mut self,
        pass: &Object,
        program: &Rc<Program>,
        state: &FrameState,
    ) -> Result<(), RenderError> {
        let entry_point = pass::get(pass, "entryPoint");
        let pipeline = self.get_compute_pipeline(program, entry_point);
        let bind_group = self.create_bind_group(
            pass,
            program,
            state,
            Some(BindTarget::Compute(
                &pipeline,
                Self::compute_entry_point(program, entry_point),
            )),
        )?;
        let immediates = self
            .compute_immediate_block(&pipeline)
            .map(|block| block.data(|g, b| self.program_storage_size(program, g, b)));
        let workgroups = self.resolve_workgroups(pass, state)?;
        let x = gpu_size32(workgroups.get("0"), "dispatchWorkgroups")?;
        let optional = |v: &Value| -> Result<u32, RenderError> {
            if v.is_undefined() {
                Ok(1)
            } else {
                gpu_size32(v, "dispatchWorkgroups")
            }
        };
        let y = optional(workgroups.get("1"))?;
        let z = optional(workgroups.get("2"))?;
        {
            let encoder = self.command_encoder.as_mut().ok_or_else(|| {
                RenderError::type_error(
                    "Cannot read properties of null (reading 'beginComputePass')",
                )
            })?;
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&program.id),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            if let Some(data) = &immediates {
                compute_pass.set_immediates(0, data);
            }
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(x, y, z);
        }
        let output_buffer_binding = program
            .bindings
            .iter()
            .find(|b| {
                (b.name == "output_buffer" || b.name == "outputBuffer")
                    && b.kind == crate::wgsl::BindingKind::Storage
            })
            .map(|b| b.name.clone());
        if let Some(buffer_name) = output_buffer_binding
            && let Some(outputs) = pass::get(pass, "outputs").as_object()
        {
            let color = outputs.get_or_undefined("color");
            let frag = outputs.get_or_undefined("fragColor");
            let output_id = if color.is_truthy() {
                color.clone()
            } else if frag.is_truthy() {
                frag.clone()
            } else {
                outputs.values().next().cloned().unwrap_or(Value::Undefined)
            };
            if output_id.is_truthy() {
                self.copy_buffer_to_texture(state, &output_id, &buffer_name)?;
            }
        }
        Ok(())
    }

    /// `resolveWorkgroups(pass, state)`.
    pub fn resolve_workgroups(
        &self,
        pass: &Object,
        state: &FrameState,
    ) -> Result<Value, RenderError> {
        let workgroups = pass::get(pass, "workgroups");
        if workgroups.is_truthy() {
            return Ok(workgroups.clone());
        }
        let size = pass::get(pass, "size");
        if size.is_truthy() {
            let or = |v: &Value, fallback: &Value| {
                if v.is_undefined() {
                    fallback.clone()
                } else {
                    v.clone()
                }
            };
            let x = or(size.get("x"), size.get("width"));
            let y = or(size.get("y"), size.get("height"));
            let depth = size.get("depth");
            let z = or(
                size.get("z"),
                if depth.is_truthy() {
                    depth
                } else {
                    &Value::Number(1.0)
                },
            );
            if x.is_truthy() && y.is_truthy() {
                return Ok(Value::Array(vec![x, y, z]));
            }
        }
        let outputs = pass::get(pass, "outputs");
        let color = outputs.get("color");
        let output_id = if color.is_truthy() {
            color.clone()
        } else {
            outputs
                .as_object()
                .and_then(|o| o.values().next().cloned())
                .unwrap_or(Value::Undefined)
        };
        let output = if output_id.is_truthy() {
            self.textures.get(&to_js_string(&output_id)).cloned()
        } else {
            None
        };
        let dispatch = |w: f64, h: f64| {
            Value::Array(vec![
                Value::Number((w / 8.0).ceil()),
                Value::Number((h / 8.0).ceil()),
                Value::Number(1.0),
            ])
        };
        if let Some(output) = output {
            return Ok(dispatch(output.width, output.height));
        }
        if state.screen_width != 0.0 && state.screen_height != 0.0 {
            return Ok(dispatch(state.screen_width, state.screen_height));
        }
        Err(RenderError::thrown(
            "ERR_COMPUTE_DISPATCH_UNRESOLVED",
            &[
                ("pass", pass::get(pass, "id").clone()),
                (
                    "detail",
                    Value::from("Compute dispatch dimensions could not be inferred"),
                ),
            ],
        ))
    }

    /// `getBufferToTextureRenderPipeline(format)`.
    fn get_buffer_to_texture_pipeline(
        &mut self,
        format: &Value,
    ) -> Result<wgpu::RenderPipeline, RenderError> {
        let key = format!("bufferToTexture_{}", to_js_string(format));
        if let Some(p) = self.buffer_to_texture_pipelines.get(&key) {
            return Ok(p.clone());
        }
        let module = self.builtin_shader("bufferToTexture", BUFFER_TO_TEXTURE_WGSL);
        let pipeline = self.create_render_pipeline_from(
            "bufferToTexture",
            (&module, "vs_main"),
            Some((&module, "fs_main")),
            &[Some(wgpu::ColorTargetState {
                format: gpu_format_of(format)?,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            wgpu::PrimitiveState::default(),
            None,
        );
        self.buffer_to_texture_pipelines
            .insert(key, pipeline.clone());
        Ok(pipeline)
    }

    /// `copyBufferToTexture(state, outputId, bufferName)`: draw the storage buffer's
    /// RGBA floats into the output texture inside the frame's encoder.
    fn copy_buffer_to_texture(
        &mut self,
        state: &FrameState,
        output_id: &Value,
        buffer_name: &str,
    ) -> Result<(), RenderError> {
        let Some(output_buffer) = self.storage_buffers.get(buffer_name).cloned() else {
            return Ok(());
        };
        let mut output_tex: Option<Rc<TextureRecord>> = None;
        if let Some(surface) = Self::parse_global_name(output_id)
            && let Some(write) = state.write_surface(&surface)
        {
            output_tex = self.textures.get(&write).cloned();
        }
        if output_tex.is_none()
            && output_id.as_str() == Some("outputTex")
            && let Some(render_surface) = state.render_surface.as_str().filter(|s| !s.is_empty())
            && let Some(write) = state.write_surface(render_surface)
        {
            output_tex = self.textures.get(&write).cloned();
        }
        if output_tex.is_none() {
            output_tex = self.textures.get(&to_js_string(output_id)).cloned();
        }
        let Some(output_tex) = output_tex else {
            return Ok(());
        };
        let width = if state.screen_width != 0.0 {
            state.screen_width
        } else {
            output_tex.width
        };
        let height = if state.screen_height != 0.0 {
            state.screen_height
        } else {
            output_tex.height
        };
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bufferToTexture params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let words = [
            noisemaker_dsl::js::to_uint32(width),
            noisemaker_dsl::js::to_uint32(height),
            0u32,
            0u32,
        ];
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        self.queue.write_buffer(&params, 0, &bytes);
        let format = match &output_tex.gpu_format {
            Some(f) if !f.is_empty() => Value::from(f.as_str()),
            _ => Value::from("rgba8unorm"),
        };
        let pipeline = self.get_buffer_to_texture_pipeline(&format)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
        let output_buffer_size = output_buffer.size();
        let immediates = self.render_immediate_block(&pipeline).map(|block| {
            block.data(|_, binding| if binding == 0 { output_buffer_size } else { 0 })
        });
        {
            let encoder = self.command_encoder.as_mut().ok_or_else(|| {
                RenderError::type_error(
                    "Cannot read properties of null (reading 'beginRenderPass')",
                )
            })?;
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("copyBufferToTexture"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_tex.view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&pipeline);
            if let Some(data) = &immediates {
                render_pass.set_immediates(0, data);
            }
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        self.active_uniform_buffers.push(params);
        Ok(())
    }
}

/// The pipeline a bind group is created for: its auto layout (group 0) and the
/// set of bindings that layout contains.
pub(super) enum BindTarget<'a> {
    Render(&'a wgpu::RenderPipeline),
    Compute(&'a wgpu::ComputePipeline, String),
}

/// The copy shader of `getBufferToTextureRenderPipeline`.
pub const BUFFER_TO_TEXTURE_WGSL: &str = r#"
            struct BufferToTextureParams {
                width: u32,
                height: u32,
                _pad0: u32,
                _pad1: u32,
            }

            @group(0) @binding(0) var<storage, read> input_buffer: array<f32>;
            @group(0) @binding(1) var<uniform> params: BufferToTextureParams;

            struct VertexOutput {
                @builtin(position) position: vec4<f32>,
            }

            @vertex
            fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                // Fullscreen triangle
                var pos = array<vec2<f32>, 3>(
                    vec2<f32>(-1.0, -1.0),
                    vec2<f32>(3.0, -1.0),
                    vec2<f32>(-1.0, 3.0)
                );

                var output: VertexOutput;
                output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
                return output;
            }

            @fragment
            fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
                // Use fragment position directly (in pixels)
                let x = u32(input.position.x);
                let y = u32(input.position.y);

                if (x >= params.width || y >= params.height) {
                    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
                }

                let pixel_idx = y * params.width + x;
                let base = pixel_idx * 4u;

                return vec4<f32>(
                    input_buffer[base + 0u],
                    input_buffer[base + 1u],
                    input_buffer[base + 2u],
                    input_buffer[base + 3u]
                );
            }
        "#;
