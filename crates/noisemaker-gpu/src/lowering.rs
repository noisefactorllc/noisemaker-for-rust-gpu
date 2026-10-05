//! Shader lowering that mirrors the reference's shader compiler on Metal.
//!
//! The reference renders through Dawn, whose Tint compiler translates WGSL to
//! MSL; wgpu translates the same WGSL with naga. Where the two translations
//! differ in a way that changes rendered pixels, this module rewrites the WGSL
//! handed to wgpu so naga emits the arithmetic Tint emits:
//!
//! * `pow(x, y)`: Tint's MSL writer emits `powr(x, y)`, naga's emits `pow(x, y)`.
//!   Under Metal's math mode the two compile to different instruction
//!   sequences, except for integral constant exponents, which the Metal
//!   compiler strength-reduces identically for both. `powr` compiles to
//!   `exp2(y * log2(x))`, so every other call site is rewritten to a helper
//!   computing exactly that (verified bit-identical to `powr` on this
//!   hardware). The reference's own analyses (bindings, uniform layouts, entry
//!   points) keep reading the unmodified source.

use std::collections::BTreeSet;

use crate::reflect::ShaderReflection;

/// Rewrite the non-integral-exponent `pow` calls of `source` (already parsed
/// into `reflection`) into `powr`-equivalent helpers. Returns `None` when the
/// source has no such call.
pub fn lower_pow_to_powr(source: &str, reflection: &ShaderReflection) -> Option<String> {
    let sites = reflection.pow_sites(source);
    if sites.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(source.len() + 512);
    let mut last = 0usize;
    let mut used = BTreeSet::new();
    for (start, suffix) in sites {
        if start < last {
            continue;
        }
        out.push_str(&source[last..start]);
        out.push_str("nm_rt_powr_");
        out.push_str(suffix);
        last = start + "pow".len();
        used.insert(suffix);
    }
    out.push_str(&source[last..]);
    for suffix in used {
        let ty = match suffix {
            "f32" => "f32",
            "vec2f" => "vec2<f32>",
            "vec3f" => "vec3<f32>",
            _ => "vec4<f32>",
        };
        out.push_str(&format!(
            "\nfn nm_rt_powr_{suffix}(x: {ty}, y: {ty}) -> {ty} {{ return exp2(y * log2(x)); }}\n"
        ));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_non_integral_exponents_only() {
        let src = "@fragment fn main(@location(0) v: vec2<f32>) -> @location(0) vec4<f32> {\n\
                   let a = pow(v.x, 2.4);\n\
                   let b = pow(v, vec2<f32>(3.0));\n\
                   let c = pow (v, v);\n\
                   return vec4<f32>(a, b.x, c.y, pow(v.y, 5.0));\n}";
        let reflection = ShaderReflection::parse(src).unwrap();
        let lowered = lower_pow_to_powr(src, &reflection).unwrap();
        assert!(lowered.contains("let a = nm_rt_powr_f32(v.x, 2.4);"));
        assert!(lowered.contains("let b = pow(v, vec2<f32>(3.0));"));
        assert!(lowered.contains("let c = nm_rt_powr_vec2f (v, v);"));
        assert!(lowered.contains("pow(v.y, 5.0)"));
        assert!(lowered.contains("fn nm_rt_powr_vec2f(x: vec2<f32>, y: vec2<f32>) -> vec2<f32>"));
        ShaderReflection::parse(&lowered).unwrap();
    }

    #[test]
    fn no_rewrite_without_dynamic_pow() {
        let src =
            "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(pow(2.0, 3.0)); }";
        let reflection = ShaderReflection::parse(src).unwrap();
        assert!(lower_pow_to_powr(src, &reflection).is_none());
    }
}
