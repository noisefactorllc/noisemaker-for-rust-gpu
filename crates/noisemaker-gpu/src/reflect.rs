//! Shader reflection with naga.
//!
//! The reference creates pipelines with `layout: 'auto'` and builds bind-group
//! entries from regex-parsed bindings. wgpu derives the same auto layout from the
//! entry points' statically used resources; this module computes that exact set
//! (the rule of `wgpu_core::validation::Interface::new`: a global variable with a
//! binding and a non-empty use in the entry point's function info) so the backend
//! knows which parsed entries the layout contains. Parsing and validating here also
//! turns WGSL errors into compile diagnostics instead of wgpu panics.

use std::collections::BTreeSet;

use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};

/// A parsed and validated WGSL module.
#[derive(Clone)]
pub struct ShaderReflection {
    module: naga::Module,
    info: ModuleInfo,
}

/// The stage of an entry point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
}

impl Stage {
    fn naga(self) -> naga::ShaderStage {
        match self {
            Stage::Vertex => naga::ShaderStage::Vertex,
            Stage::Fragment => naga::ShaderStage::Fragment,
            Stage::Compute => naga::ShaderStage::Compute,
        }
    }
}

/// One compiler message, in the shape of `getCompilationInfo()` entries.
#[derive(Debug, Clone, PartialEq)]
pub struct CompilationMessage {
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub message: String,
}

impl ShaderReflection {
    /// Parse and validate `source`; on failure return the compiler messages.
    pub fn parse(source: &str) -> Result<ShaderReflection, Vec<CompilationMessage>> {
        let module = naga::front::wgsl::parse_str(source).map_err(|e| {
            let location = e.location(source);
            vec![CompilationMessage {
                line: location.map(|l| l.line_number),
                column: location.map(|l| l.line_position),
                message: e.emit_to_string(source),
            }]
        })?;
        let info = Validator::new(ValidationFlags::all(), Capabilities::all())
            .validate(&module)
            .map_err(|e| {
                let location = e.location(source);
                vec![CompilationMessage {
                    line: location.map(|l| l.line_number),
                    column: location.map(|l| l.line_position),
                    message: e.emit_to_string(source),
                }]
            })?;
        Ok(ShaderReflection { module, info })
    }

    /// The naga module.
    pub fn module(&self) -> &naga::Module {
        &self.module
    }

    /// The module's validation info.
    pub fn info(&self) -> &ModuleInfo {
        &self.info
    }

    /// `true` when the fragment entry point `name` writes
    /// `@builtin(frag_depth)`.
    pub fn writes_frag_depth(&self, name: &str) -> bool {
        let Some(ep) = self
            .module
            .entry_points
            .iter()
            .find(|ep| ep.stage == naga::ShaderStage::Fragment && ep.name == name)
        else {
            return false;
        };
        let Some(result) = &ep.function.result else {
            return false;
        };
        let is_depth = |b: &Option<naga::Binding>| {
            matches!(b, Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth)))
        };
        if is_depth(&result.binding) {
            return true;
        }
        match &self.module.types[result.ty].inner {
            naga::TypeInner::Struct { members, .. } => members.iter().any(|m| is_depth(&m.binding)),
            _ => false,
        }
    }

    /// `true` when the module has an entry point `name` for `stage`.
    pub fn has_entry_point(&self, stage: Stage, name: &str) -> bool {
        self.module
            .entry_points
            .iter()
            .any(|ep| ep.stage == stage.naga() && ep.name == name)
    }

    /// The `@binding` indices of `@group(group)` that entry point `name` uses,
    /// or `None` when the module has no such entry point.
    pub fn used_bindings(&self, stage: Stage, name: &str, group: u32) -> Option<BTreeSet<u32>> {
        let index = self
            .module
            .entry_points
            .iter()
            .position(|ep| ep.stage == stage.naga() && ep.name == name)?;
        let info = self.info.get_entry_point(index);
        let mut used = BTreeSet::new();
        for (handle, var) in self.module.global_variables.iter() {
            if let Some(binding) = &var.binding
                && binding.group == group
                && !info[handle].is_empty()
            {
                used.insert(binding.binding);
            }
        }
        Some(used)
    }
}

/// The scalar/vector suffix of an f32 type (`f32`, `vec2f`, `vec3f`, `vec4f`).
fn f32_suffix(inner: &naga::TypeInner) -> Option<&'static str> {
    use naga::{ScalarKind, TypeInner, VectorSize};
    match *inner {
        TypeInner::Scalar(s) if s.kind == ScalarKind::Float && s.width == 4 => Some("f32"),
        TypeInner::Vector { size, scalar }
            if scalar.kind == ScalarKind::Float && scalar.width == 4 =>
        {
            Some(match size {
                VectorSize::Bi => "vec2f",
                VectorSize::Tri => "vec3f",
                VectorSize::Quad => "vec4f",
            })
        }
        _ => None,
    }
}

/// `true` when `expr` is a constant whose every component is an integral value.
fn is_integral_constant(
    module: &naga::Module,
    arena: &naga::Arena<naga::Expression>,
    expr: naga::Handle<naga::Expression>,
) -> bool {
    use naga::{Expression, Literal};
    match &arena[expr] {
        Expression::Literal(Literal::F32(v)) => v.is_finite() && v.fract() == 0.0,
        Expression::Literal(Literal::AbstractFloat(v)) => v.is_finite() && v.fract() == 0.0,
        Expression::Literal(Literal::F16(v)) => {
            let v = f32::from(*v);
            v.is_finite() && v.fract() == 0.0
        }
        Expression::ZeroValue(_) => true,
        Expression::Splat { value, .. } => is_integral_constant(module, arena, *value),
        Expression::Compose { components, .. } => components
            .iter()
            .all(|c| is_integral_constant(module, arena, *c)),
        Expression::Constant(c) => {
            let init = module.constants[*c].init;
            is_integral_constant(module, &module.global_expressions, init)
        }
        _ => false,
    }
}

impl ShaderReflection {
    /// The `pow(x, y)` call sites of the module whose exponent is not an integral
    /// constant, as `(byte offset of the "pow" identifier, f32 type suffix)`.
    pub fn pow_sites(&self, source: &str) -> Vec<(usize, &'static str)> {
        let mut sites = Vec::new();
        let mut scan = |function: &naga::Function, info: &naga::valid::FunctionInfo| {
            for (handle, expr) in function.expressions.iter() {
                let naga::Expression::Math {
                    fun: naga::MathFunction::Pow,
                    arg1: Some(exponent),
                    ..
                } = expr
                else {
                    continue;
                };
                if is_integral_constant(&self.module, &function.expressions, *exponent) {
                    continue;
                }
                let inner = info[handle].ty.inner_with(&self.module.types);
                let Some(suffix) = f32_suffix(inner) else {
                    continue;
                };
                let Some(range) = function.expressions.get_span(handle).to_range() else {
                    continue;
                };
                let text = &source[range.start.min(source.len())..];
                let rest = text.strip_prefix("pow").map(|r| r.trim_start());
                if rest.is_some_and(|r| r.starts_with('(')) {
                    sites.push((range.start, suffix));
                }
            }
        };
        for (handle, function) in self.module.functions.iter() {
            scan(function, &self.info[handle]);
        }
        for (index, ep) in self.module.entry_points.iter().enumerate() {
            scan(&ep.function, self.info.get_entry_point(index));
        }
        sites.sort_unstable();
        sites.dedup();
        sites
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_layout_bindings_follow_static_use() {
        let src = r#"
            @group(0) @binding(0) var a: texture_2d<f32>;
            @group(0) @binding(1) var s: sampler;
            @group(0) @binding(2) var<uniform> u: vec4<f32>;
            fn helper() -> vec4<f32> { return u; }
            @fragment fn main() -> @location(0) vec4<f32> { return helper(); }
            @fragment fn other() -> @location(0) vec4<f32> { return textureSample(a, s, vec2f(0.5)); }
        "#;
        let r = ShaderReflection::parse(src).unwrap();
        assert_eq!(
            r.used_bindings(Stage::Fragment, "main", 0).unwrap(),
            BTreeSet::from([2])
        );
        assert_eq!(
            r.used_bindings(Stage::Fragment, "other", 0).unwrap(),
            BTreeSet::from([0, 1])
        );
        assert!(r.used_bindings(Stage::Vertex, "main", 0).is_none());
    }

    #[test]
    fn invalid_wgsl_reports_a_location() {
        let err = ShaderReflection::parse("fn main( {").err().unwrap();
        assert_eq!(err.len(), 1);
        assert!(err[0].line.is_some());
    }
}
