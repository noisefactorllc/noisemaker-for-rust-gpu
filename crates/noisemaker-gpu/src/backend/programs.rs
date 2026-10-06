//! Program compilation of the WebGPU backend (`compileProgram`,
//! `compileRenderProgram`, `compileComputeProgram`).

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use indexmap::IndexMap;
use noisemaker_dsl::Value;

use super::shaders::{Compiler, DeviceShader};
use super::{
    DEFAULT_FRAGMENT_ENTRY_POINT, DEFAULT_VERTEX_ENTRY_POINT, DEFAULT_VERTEX_SHADER_WGSL,
    WebGpuBackend, gpu_format_of, resolve_format,
};
use crate::diagnostics::codes;
use crate::error::{RenderError, ShaderDiagnostic};
use crate::jsre::{JsRegex, group};
use crate::reflect::{CompilationMessage, ShaderReflection, Stage};
use crate::wgsl::{
    ShaderBinding, UniformLayout, detect_entry_points, has_shader_bindings, inject_defines,
    normalize_spec_layout, parse_declared_uniform_buffer_size, parse_entry_point_bindings,
    parse_packed_uniform_layout, parse_shader_bindings,
};

/// A compiled program (`programInfo`).
pub struct Program {
    pub id: String,
    pub is_compute: bool,
    /// `bindings`: the parsed (and dead-filtered) binding declarations.
    pub bindings: Vec<ShaderBinding>,
    /// `_sourceHasBindings`.
    pub source_has_bindings: bool,
    /// `packedUniformLayout`: `spec.uniformLayout` or the parsed layout.
    pub packed_uniform_layout: Option<UniformLayout>,
    /// `declaredUniformBufferSize`.
    pub declared_uniform_buffer_size: u64,
    pub kind: ProgramKind,
}

/// The render- or compute-specific part of a program.
pub enum ProgramKind {
    Render(RenderProgram),
    Compute(ComputeProgram),
}

/// A render program: its modules, entry points and pipeline cache.
pub struct RenderProgram {
    pub vertex_module: DeviceShader,
    pub fragment_module: DeviceShader,
    pub vertex_entry_point: String,
    pub fragment_entry_point: String,
    /// `outputFormat` (resolved WebGPU format name).
    pub output_format: Value,
    /// `pipeline`: the pipeline created at compile time.
    pub pipeline: wgpu::RenderPipeline,
    /// `pipelineCache`: key → pipeline.
    pub pipeline_cache: RefCell<HashMap<String, wgpu::RenderPipeline>>,
    /// The group-0 binding indices of every auto layout of this program (the
    /// vertex and fragment entry points' statically used resources).
    pub layout_bindings: BTreeSet<u32>,
}

/// A compute program: its module, entry points and per-entry-point pipelines.
pub struct ComputeProgram {
    pub module: DeviceShader,
    pub reflection: Box<ShaderReflection>,
    /// `pipeline`: the default entry point's pipeline.
    pub pipeline: wgpu::ComputePipeline,
    /// `pipelines`: entry point → pipeline.
    pub pipelines: RefCell<HashMap<String, wgpu::ComputePipeline>>,
    /// `entryPoint`: the default entry point.
    pub entry_point: String,
    /// `entryPoints`: every `@compute` entry point.
    pub entry_points: Vec<String>,
    /// `entryPointBindings`.
    pub entry_point_bindings: IndexMap<String, Vec<u32>>,
}

impl Program {
    pub fn render(&self) -> Option<&RenderProgram> {
        match &self.kind {
            ProgramKind::Render(r) => Some(r),
            ProgramKind::Compute(_) => None,
        }
    }

    pub fn compute(&self) -> Option<&ComputeProgram> {
        match &self.kind {
            ProgramKind::Compute(c) => Some(c),
            ProgramKind::Render(_) => None,
        }
    }
}

fn compile_error(id: &str, messages: Vec<CompilationMessage>) -> RenderError {
    let detail = messages
        .iter()
        .map(|m| {
            format!(
                "Line {}: {}",
                m.line
                    .map(|l| l.to_string())
                    .unwrap_or_else(|| "undefined".into()),
                m.message
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    RenderError::Shader(Box::new(ShaderDiagnostic {
        code: codes::COMPILE.into(),
        backend: "webgpu".into(),
        stage: "compile".into(),
        program: Some(id.to_owned()),
        detail,
        messages,
    }))
}

/// `spec[key]` when it is a truthy string.
fn spec_str<'a>(spec: &'a Value, key: &str) -> Option<&'a str> {
    let v = spec.get(key);
    if v.is_truthy() { v.as_str() } else { None }
}

/// A WebGPU topology name → wgpu topology.
pub fn topology_of(name: &str) -> Result<wgpu::PrimitiveTopology, RenderError> {
    Ok(match name {
        "point-list" => wgpu::PrimitiveTopology::PointList,
        "line-list" => wgpu::PrimitiveTopology::LineList,
        "line-strip" => wgpu::PrimitiveTopology::LineStrip,
        "triangle-list" => wgpu::PrimitiveTopology::TriangleList,
        "triangle-strip" => wgpu::PrimitiveTopology::TriangleStrip,
        other => {
            return Err(RenderError::type_error(format!(
                "The provided value '{other}' is not a valid enum value of type GPUPrimitiveTopology."
            )));
        }
    })
}

impl WebGpuBackend {
    /// Parse and validate `source` and compile it for the device
    /// ([`super::shaders`]: Tint on Metal, else naga after the device's
    /// shader-compiler parity lowering, [`crate::lowering`]).
    fn device_module(
        &self,
        id: &str,
        source: &str,
    ) -> Result<(DeviceShader, ShaderReflection), RenderError> {
        let reflection = ShaderReflection::parse(source).map_err(|m| compile_error(id, m))?;
        let shader = self.device_shader(id, source, &reflection);
        Ok((shader, reflection))
    }

    /// A shader of the backend itself (the reference's default vertex,
    /// resample and buffer-to-texture shaders): Tint-compiled like every
    /// program on the Tint path, a plain wgpu module on the naga path. WGSL
    /// that does not validate (the reference's resample shader, which Dawn
    /// rejects too) stays a wgpu module, so wgpu reports its error as before.
    pub(super) fn builtin_shader(&self, label: &str, source: &str) -> DeviceShader {
        if let Compiler::Tint(_) = &self.compiler
            && let Ok(reflection) = ShaderReflection::parse(source)
        {
            return self.device_shader(label, source, &reflection);
        }
        DeviceShader::Naga(
            self.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }),
        )
    }

    /// `resolveWGSLSource(spec)`: `wgsl`, else `source`, else a non-GLSL `fragment`.
    pub fn resolve_wgsl_source(spec: &Value) -> Option<String> {
        if let Some(s) = spec_str(spec, "wgsl") {
            return Some(s.to_owned());
        }
        if spec.get("wgsl").is_truthy() {
            return Some(crate::jsv::to_js_string(spec.get("wgsl")));
        }
        if spec.get("source").is_truthy() {
            return Some(crate::jsv::to_js_string(spec.get("source")));
        }
        if let Some(fragment) = spec_str(spec, "fragment")
            && !fragment.contains("#version")
        {
            return Some(fragment.to_owned());
        }
        None
    }

    /// `getDefaultVertexModule()`.
    pub(super) fn get_default_vertex_module(&mut self) -> DeviceShader {
        if self.default_vertex_module.is_none() {
            self.default_vertex_module =
                Some(self.builtin_shader("default vertex", DEFAULT_VERTEX_SHADER_WGSL));
        }
        self.default_vertex_module.clone().unwrap()
    }

    /// `compileProgram(id, spec)`: inject defines, detect entry points, and compile
    /// a compute program (a `@compute` source without `@fragment`) or a render
    /// program.
    pub fn compile_program(&mut self, id: &str, spec: &Value) -> Result<Rc<Program>, RenderError> {
        let Some(source) = Self::resolve_wgsl_source(spec) else {
            let keys = spec
                .as_object()
                .map(|o| o.keys().cloned().collect::<Vec<_>>().join(", "))
                .unwrap_or_default();
            return Err(RenderError::Shader(Box::new(ShaderDiagnostic {
                code: codes::NO_SOURCE.into(),
                backend: "webgpu".into(),
                stage: "missing-source".into(),
                program: Some(id.to_owned()),
                detail: format!(
                    "No WGSL shader source found for program '{id}'. Available keys: {keys}"
                ),
                messages: Vec::new(),
            })));
        };
        let defines = spec.get("defines");
        let processed = if defines.is_truthy() {
            inject_defines(&source, defines)
        } else {
            source.clone()
        };
        let has_compute = JsRegex::new(r"@compute\s", "").test(&processed);
        let has_fragment = JsRegex::new(r"@fragment\s", "").test(&processed);
        let detected = detect_entry_points(&processed);
        let entry = |detected: &Option<String>, key: &str| -> Option<String> {
            detected
                .clone()
                .or_else(|| spec_str(spec, key).map(str::to_owned))
        };
        let enhanced = EnhancedSpec {
            fragment_entry_point: entry(&detected.fragment, "fragmentEntryPoint"),
            vertex_entry_point: entry(&detected.vertex, "vertexEntryPoint"),
            compute_entry_point: entry(&detected.compute, "computeEntryPoint"),
        };
        let program = if has_compute && !has_fragment {
            self.compile_compute_program(id, &processed, spec, &enhanced)?
        } else {
            self.compile_render_program(id, &processed, spec, &enhanced)?
        };
        let program = Rc::new(program);
        self.programs.insert(id.to_owned(), program.clone());
        self.collect_device_errors();
        Ok(program)
    }

    fn packed_layout(spec: &Value, source: &str) -> Option<UniformLayout> {
        let layout = spec.get("uniformLayout");
        if layout.is_truthy() {
            Some(normalize_spec_layout(layout))
        } else {
            parse_packed_uniform_layout(source)
        }
    }

    fn compile_compute_program(
        &mut self,
        id: &str,
        source: &str,
        spec: &Value,
        enhanced: &EnhancedSpec,
    ) -> Result<Program, RenderError> {
        let (module, reflection) = self.device_module(id, source)?;
        let bindings = parse_shader_bindings(source);
        let entry_points: Vec<String> = JsRegex::new(r"@compute[^f]*fn\s+(\w+)", "g")
            .exec_all(source)
            .map(|c| group(&c, 1).unwrap().to_owned())
            .collect();
        let entry_point_bindings = parse_entry_point_bindings(source, &bindings);
        let default_entry_point = enhanced
            .compute_entry_point
            .clone()
            .or_else(|| entry_points.first().cloned())
            .unwrap_or_else(|| "main".to_owned());
        let pipeline = self.create_compute_pipeline_from(id, &module, &default_entry_point);
        let mut pipelines = HashMap::new();
        pipelines.insert(default_entry_point.clone(), pipeline.clone());
        Ok(Program {
            id: id.to_owned(),
            is_compute: true,
            source_has_bindings: has_shader_bindings(source),
            packed_uniform_layout: Self::packed_layout(spec, source),
            declared_uniform_buffer_size: parse_declared_uniform_buffer_size(source),
            bindings,
            kind: ProgramKind::Compute(ComputeProgram {
                module,
                reflection: Box::new(reflection),
                pipeline,
                pipelines: RefCell::new(pipelines),
                entry_point: default_entry_point,
                entry_points,
                entry_point_bindings,
            }),
        })
    }

    fn compile_render_program(
        &mut self,
        id: &str,
        source: &str,
        spec: &Value,
        enhanced: &EnhancedSpec,
    ) -> Result<Program, RenderError> {
        let mut bindings = parse_shader_bindings(source);
        let mut source_has_bindings = has_shader_bindings(source);
        let has_vertex = JsRegex::new(r"@vertex\s", "").test(source);

        let (main_module, main_reflection) = self.device_module(id, source)?;

        let vertex_source = spec_str(spec, "vertexWGSL").or_else(|| spec_str(spec, "vertexWgsl"));
        let (vertex_module, vertex_entry_point, vertex_bindings_used) =
            if let Some(vertex_source) = vertex_source {
                let (module, reflection) = self.device_module(id, vertex_source)?;
                let entry_point = enhanced
                    .vertex_entry_point
                    .clone()
                    .unwrap_or_else(|| DEFAULT_VERTEX_ENTRY_POINT.to_owned());
                let vertex_bindings = parse_shader_bindings(vertex_source);
                source_has_bindings = source_has_bindings || has_shader_bindings(vertex_source);
                if !vertex_bindings.is_empty() {
                    let existing: Vec<(u32, u32)> =
                        bindings.iter().map(|b| (b.group, b.binding)).collect();
                    for vb in vertex_bindings {
                        if !existing.contains(&(vb.group, vb.binding)) {
                            bindings.push(vb);
                        }
                    }
                    bindings.sort_by(|a, b| a.group.cmp(&b.group).then(a.binding.cmp(&b.binding)));
                }
                let used = reflection
                    .used_bindings(Stage::Vertex, &entry_point, 0)
                    .unwrap_or_default();
                (module, entry_point, used)
            } else if has_vertex {
                let entry_point = enhanced
                    .vertex_entry_point
                    .clone()
                    .unwrap_or_else(|| DEFAULT_VERTEX_ENTRY_POINT.to_owned());
                let used = main_reflection
                    .used_bindings(Stage::Vertex, &entry_point, 0)
                    .unwrap_or_default();
                (main_module.clone(), entry_point, used)
            } else {
                (
                    self.get_default_vertex_module(),
                    DEFAULT_VERTEX_ENTRY_POINT.to_owned(),
                    BTreeSet::new(),
                )
            };

        let fragment_entry_point = enhanced
            .fragment_entry_point
            .clone()
            .or_else(|| spec_str(spec, "entryPoint").map(str::to_owned))
            .unwrap_or_else(|| DEFAULT_FRAGMENT_ENTRY_POINT.to_owned());
        let output_format_spec = spec.get("outputFormat");
        let default_output = Value::from("rgba16float");
        let output_format = resolve_format(if output_format_spec.is_truthy() {
            output_format_spec
        } else {
            &default_output
        });
        let mut layout_bindings = vertex_bindings_used;
        if let Some(used) = main_reflection.used_bindings(Stage::Fragment, &fragment_entry_point, 0)
        {
            layout_bindings.extend(used);
        }

        let topology_value = spec.get("topology");
        let topology = topology_of(
            topology_value
                .as_str()
                .filter(|_| topology_value.is_truthy())
                .unwrap_or("triangle-list"),
        )?;
        let blend = Self::resolve_blend_state(spec.get("blend"))?;
        let format = gpu_format_of(&output_format)?;
        let pipeline = self.create_render_pipeline_from(
            id,
            (&vertex_module, &vertex_entry_point),
            Some((&main_module, &fragment_entry_point)),
            &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            wgpu::PrimitiveState {
                topology,
                ..Default::default()
            },
            None,
        );
        let mut cache = HashMap::new();
        let initial_key = Self::get_pipeline_key(
            spec.get("blend"),
            spec.get("topology"),
            output_format.as_str().unwrap_or("rgba16float"),
        );
        cache.insert(initial_key, pipeline.clone());

        Ok(Program {
            id: id.to_owned(),
            is_compute: false,
            bindings,
            source_has_bindings,
            packed_uniform_layout: Self::packed_layout(spec, source),
            declared_uniform_buffer_size: parse_declared_uniform_buffer_size(source),
            kind: ProgramKind::Render(RenderProgram {
                vertex_module,
                fragment_module: main_module,
                vertex_entry_point,
                fragment_entry_point,
                output_format,
                pipeline,
                pipeline_cache: RefCell::new(cache),
                layout_bindings,
            }),
        })
    }
}

/// The entry points of `enhancedSpec`: detected in the define-injected source,
/// falling back to the spec's own.
pub(super) struct EnhancedSpec {
    pub fragment_entry_point: Option<String>,
    pub vertex_entry_point: Option<String>,
    pub compute_entry_point: Option<String>,
}
