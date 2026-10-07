//! Effect preflight (port of `runtime/preflight.js`): static analysis of an effect
//! graph against device capabilities — per-backend authorability, predicted MRT
//! format demotions and predicted `maxTextureSize` clamps.

use indexmap::IndexMap;
use noisemaker_dsl::{Object, Value};

use crate::backend::Capabilities;

/// `mrtFormatBytes(format)`: the color-attachment byte cost of a format.
pub fn mrt_format_bytes(format: &Value) -> u32 {
    match format.as_str() {
        Some("rgba32f") | Some("rgba32float") => 16,
        Some("rgba8") | Some("rgba8unorm") => 4,
        Some("r32f") | Some("r32float") => 4,
        Some("r16f") | Some("r16float") => 2,
        Some("r8") | Some("r8unorm") => 1,
        _ => 8,
    }
}

fn is_glsl_source(text: &Value) -> bool {
    matches!(text, Value::String(s) if s.contains("#version"))
}

// Each predicate selects the source exactly as its backend's compileProgram
// does, then asks whether that source is in the backend's language: a backend
// that selects source in the other language fails to compile it.

fn is_wgsl_bucket(bucket: &Value) -> bool {
    if !bucket.is_truthy() {
        return false;
    }
    // WebGPU resolveWGSLSource(): wgsl, then source, then a non-GLSL fragment.
    if bucket.get("wgsl").is_truthy() {
        return true;
    }
    if bucket.get("source").is_truthy() {
        return !is_glsl_source(bucket.get("source"));
    }
    bucket.get("fragment").is_truthy() && !is_glsl_source(bucket.get("fragment"))
}

fn is_glsl_bucket(bucket: &Value) -> bool {
    if !bucket.is_truthy() {
        return false;
    }
    // WebGL2 compileProgram(): source, then glsl, then fragment. A vertex
    // shader alone is not a program source.
    if bucket.get("source").is_truthy() {
        return is_glsl_source(bucket.get("source"));
    }
    bucket.get("glsl").is_truthy() || bucket.get("fragment").is_truthy()
}

/// One backend's verdict.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BackendVerdict {
    pub authorable: bool,
    pub reasons: Vec<String>,
}

/// A predicted MRT demotion.
#[derive(Debug, Clone, PartialEq)]
pub struct FormatChange {
    pub texture: Value,
    pub pass: Value,
    pub from: Value,
    pub to: String,
    pub budget: u32,
}

/// A predicted `maxTextureSize` clamp.
#[derive(Debug, Clone, PartialEq)]
pub struct Clamp {
    pub texture: String,
    pub field: String,
    pub requested: f64,
    pub limit: u32,
}

/// `preflightEffect` result.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PreflightReport {
    pub webgl2: BackendVerdict,
    pub webgpu: BackendVerdict,
    pub format_changes: Vec<FormatChange>,
    pub clamps: Vec<Clamp>,
}

/// `preflightEffect(definition, capabilities, shaders)` over a pass list, texture
/// specs and per-program shader buckets (`None`: source availability unknown).
pub fn preflight_effect(
    passes: &[Object],
    textures: Option<&IndexMap<String, Object>>,
    caps: &Capabilities,
    shaders: Option<&IndexMap<String, Value>>,
) -> PreflightReport {
    let has_shader_info = shaders.is_some_and(|s| !s.is_empty());
    let mut webgl2 = Vec::new();
    let mut webgpu = Vec::new();
    let mut format_changes = Vec::new();
    let mut clamps = Vec::new();
    let budget = caps.max_color_bytes_per_sample;
    for pass in passes {
        let program = pass.get_or_undefined("program");
        if !program.is_truthy() {
            continue;
        }
        let program_name = crate::jsv::to_js_string(program);
        let name = pass.get_or_undefined("name");
        let label = if name.is_truthy() {
            crate::jsv::to_js_string(name)
        } else {
            program_name.clone()
        };
        if has_shader_info {
            let bucket = shaders
                .unwrap()
                .get(&program_name)
                .cloned()
                .unwrap_or(Value::Undefined);
            if !is_glsl_bucket(&bucket) {
                webgl2.push(format!(
                    "program '{program_name}' has no GLSL source (pass '{label}')"
                ));
            }
            if !is_wgsl_bucket(&bucket) {
                webgpu.push(format!(
                    "program '{program_name}' has no WGSL source (pass '{label}')"
                ));
            }
        }
        let outputs = pass.get_or_undefined("outputs");
        let output_count = outputs.as_object().map(|o| o.len()).unwrap_or(0);
        if output_count as u32 > caps.max_draw_buffers {
            let reason = format!(
                "pass '{label}' writes {output_count} color attachments, device allows {}",
                caps.max_draw_buffers
            );
            webgl2.push(reason.clone());
            webgpu.push(reason);
        }
        if budget != 0 && output_count > 1 {
            let entries: Vec<(Value, Option<&Object>)> = outputs
                .as_object()
                .unwrap()
                .values()
                .map(|id| {
                    let spec = textures.and_then(|t| t.get(&crate::jsv::to_js_string(id)));
                    (id.clone(), spec)
                })
                .collect();
            let mut total: u32 = entries
                .iter()
                .map(|(_, spec)| {
                    mrt_format_bytes(
                        spec.map(|s| s.get_or_undefined("format"))
                            .unwrap_or(&Value::Undefined),
                    )
                })
                .sum();
            if total > budget {
                for (tex_id, spec) in entries.iter().rev() {
                    if total <= budget {
                        break;
                    }
                    let Some(spec) = spec else { continue };
                    let format = spec.get_or_undefined("format");
                    if matches!(format.as_str(), Some("rgba32f" | "rgba32float")) {
                        let pass_label = if name.is_truthy() {
                            name.clone()
                        } else if pass.get_or_undefined("id").is_truthy() {
                            pass.get_or_undefined("id").clone()
                        } else {
                            program.clone()
                        };
                        format_changes.push(FormatChange {
                            texture: tex_id.clone(),
                            pass: pass_label,
                            from: format.clone(),
                            to: "rgba16f".into(),
                            budget,
                        });
                        total -= 8;
                    }
                }
            }
        }
    }
    if let Some(textures) = textures {
        for (tex_id, spec) in textures {
            for field in ["width", "height", "depth"] {
                if let Value::Number(n) = spec.get_or_undefined(field)
                    && *n > caps.max_texture_size as f64
                {
                    clamps.push(Clamp {
                        texture: tex_id.clone(),
                        field: field.into(),
                        requested: *n,
                        limit: caps.max_texture_size,
                    });
                }
            }
        }
    }
    PreflightReport {
        webgl2: BackendVerdict {
            authorable: webgl2.is_empty(),
            reasons: webgl2,
        },
        webgpu: BackendVerdict {
            authorable: webgpu.is_empty(),
            reasons: webgpu,
        },
        format_changes,
        clamps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bucket(json: &str) -> Value {
        Value::from_json(json).unwrap()
    }

    #[test]
    fn single_channel_formats_cost_their_own_bytes() {
        for (format, bytes) in [
            ("rgba32float", 16),
            ("rgba8unorm", 4),
            ("r32f", 4),
            ("r32float", 4),
            ("r16f", 2),
            ("r16float", 2),
            ("r8", 1),
            ("r8unorm", 1),
            ("rgba16float", 8),
        ] {
            assert_eq!(mrt_format_bytes(&Value::from(format)), bytes, "{format}");
        }
    }

    #[test]
    fn sources_are_judged_by_the_language_each_backend_selects() {
        let glsl = r##""#version 300 es\nvoid main(){}""##;
        // A vertex shader alone is not a WebGL2 program source.
        assert!(!is_glsl_bucket(&bucket(r#"{"vertex":"v"}"#)));
        // Generic source is selected first by both backends, then judged.
        let generic_glsl = bucket(&format!(r#"{{"source":{glsl},"fragment":"f"}}"#));
        assert!(is_glsl_bucket(&generic_glsl));
        assert!(!is_wgsl_bucket(&generic_glsl));
        let generic_wgsl = bucket(r#"{"source":"@fragment fn main() {}","glsl":"g"}"#);
        assert!(!is_glsl_bucket(&generic_wgsl));
        assert!(is_wgsl_bucket(&generic_wgsl));
        // A GLSL fragment does not make a WGSL bucket; any fragment is GLSL's.
        let fragment = bucket(&format!(r#"{{"fragment":{glsl}}}"#));
        assert!(is_glsl_bucket(&fragment));
        assert!(!is_wgsl_bucket(&fragment));
    }
}
