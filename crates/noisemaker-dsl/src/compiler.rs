//! Port of `runtime/compiler.js`: `compileGraph` ties the frontend together —
//! lex, parse and validate the DSL source, expand the plans into render passes,
//! allocate pooled textures and assemble the executable render graph.

use crate::error::JsError;
use crate::expander::{
    ExpandOptions, is_str, js_entries, js_string, js_values, member, starts_with,
};
use crate::js::integer_to_string_radix;
use crate::registry::Registry;
use crate::resources::allocate_resources;
use crate::value::{Object, Value};

/// Compilation options (`options` of `compileGraph`).
#[derive(Debug, Clone, Default)]
pub struct CompileOptions {
    /// `options.shaderOverrides`: per-step shader overrides keyed by step index,
    /// e.g. `{ "0": { "main": { "glsl": "...", "wgsl": "..." } } }`.
    pub shader_overrides: Object,
}

/// `compileGraph(source, options)`: compile DSL source into an executable graph
/// `{ id, source, passes, programs, allocations, textures, renderSurface,
/// mediaSteps }` (the reference's `compiledAt` timestamp is omitted). Program
/// specs keep their shader sources. Throws what the reference throws: the
/// frontend's errors, `{ code: 'ERR_COMPILATION_FAILED', diagnostics }`,
/// `{ code: 'ERR_EXPANSION_FAILED', errors }`, and the reference's own
/// TypeErrors on some inputs.
pub fn compile_graph(
    source: &str,
    registry: &Registry,
    options: &CompileOptions,
) -> Result<Value, JsError> {
    // Stage 1: parse and validate the DSL (`compile(source)` of lang/index.js).
    let tokens = crate::lexer::lex(source)?;
    let ast = crate::parser::parse_with_registry(&tokens, registry)?;
    let compilation_result = crate::validator::validate(&ast, registry)?;
    compile_graph_from_validated(source, &compilation_result, registry, options)
}

/// `compileGraph` from its stage-1 result: `compilation_result` is the
/// `validate(parse(lex(source)))` output for `source`. The diagnostics check,
/// expansion, allocation and graph assembly are `compileGraph`'s.
pub fn compile_graph_from_validated(
    source: &str,
    compilation_result: &Value,
    registry: &Registry,
    options: &CompileOptions,
) -> Result<Value, JsError> {
    check_diagnostics(compilation_result)?;

    // Stage 2: expand the logical graph into render passes.
    let expansion = crate::expander::expand(
        compilation_result,
        registry,
        &ExpandOptions {
            shader_overrides: options.shader_overrides.clone(),
        },
    )?;
    if !expansion.errors.is_empty() {
        let mut thrown = Object::new();
        thrown.insert("code", Value::from("ERR_EXPANSION_FAILED"));
        thrown.insert("errors", Value::Array(expansion.errors));
        return Err(JsError::Thrown(Value::Object(thrown)));
    }

    // Stage 3: allocate resources (texture pooling).
    let allocations = allocate_resources(&expansion.passes)?;

    // Stage 4: build the execution graph.
    let mut graph = Object::new();
    graph.insert("id", Value::String(hash_source(source)));
    graph.insert("source", Value::from(source));
    let textures = extract_texture_specs(&expansion.passes, options, &expansion.texture_specs)?;
    graph.insert("passes", Value::Array(expansion.passes));
    graph.insert("programs", Value::Object(expansion.programs));
    graph.insert("allocations", Value::Object(allocations));
    graph.insert("textures", Value::Object(textures));
    // Which surface to present to the screen (e.g. 'o0', 'o2').
    graph.insert("renderSurface", expansion.render_surface);
    // Per-step external texture bindings (e.g. imageTex_step_0).
    graph.insert("mediaSteps", Value::Array(expansion.media_steps));
    Ok(Value::Object(graph))
}

/// `compileGraph`'s handling of the validator's diagnostics: warnings are logged
/// (`console.warn('[noisemaker] CODE: message')`, to standard error), and any
/// error throws `{ code: 'ERR_COMPILATION_FAILED', diagnostics }` with every
/// diagnostic.
fn check_diagnostics(compilation_result: &Value) -> Result<(), JsError> {
    let diagnostics = member(compilation_result, "diagnostics")?;
    // `compilationResult.diagnostics?.length > 0`, with JavaScript's coercion of
    // whatever `length` is.
    let length = match diagnostics {
        Value::Array(a) => a.len() as f64,
        Value::String(s) => s.encode_utf16().count() as f64,
        Value::Object(o) => match o.get("length") {
            Some(length) => crate::palette::to_number(length),
            None => f64::NAN,
        },
        _ => f64::NAN,
    };
    if length.is_nan() || length <= 0.0 {
        return Ok(());
    }
    let Value::Array(list) = diagnostics else {
        return Err(JsError::type_error(
            "compilationResult.diagnostics.filter is not a function",
        ));
    };
    let mut warnings = Vec::new();
    for d in list {
        if is_str(member(d, "severity")?, "warning") {
            warnings.push(d);
        }
    }
    for w in warnings {
        eprintln!(
            "[noisemaker] {}: {}",
            js_string(member(w, "code")?),
            js_string(member(w, "message")?)
        );
    }
    let mut errors = Vec::new();
    for d in list {
        if is_str(member(d, "severity")?, "error") {
            errors.push(d);
        }
    }
    if !errors.is_empty() {
        let mut thrown = Object::new();
        thrown.insert("code", Value::from("ERR_COMPILATION_FAILED"));
        thrown.insert("diagnostics", diagnostics.clone());
        return Err(JsError::Thrown(Value::Object(thrown)));
    }
    Ok(())
}

/// The usage flags of a 2D texture.
fn usage_2d() -> Value {
    Value::Array(
        ["render", "sample", "copySrc", "copyDst"]
            .into_iter()
            .map(Value::from)
            .collect(),
    )
}

/// `extractTextureSpecs(passes, options, textureSpecs)`: the textures to
/// allocate, as an ordered object (the reference's `Map`): every effect-defined
/// texture spec first (dimension specs preserved, `'screen'` by default), then
/// every non-global pass output not yet defined, at screen size.
pub fn extract_texture_specs(
    passes: &[Value],
    _options: &CompileOptions,
    texture_specs: &Object,
) -> Result<Object, JsError> {
    let mut textures = Object::new();

    // Effect-defined texture specs (including global_ textures).
    for (tex_id, effect_spec) in texture_specs.iter() {
        let or_default = |key: &str, default: &str| -> Value {
            let v = effect_spec.get(key);
            if v.is_truthy() {
                v.clone()
            } else {
                Value::from(default)
            }
        };
        let mut spec = Object::new();
        spec.insert("width", or_default("width", "screen"));
        spec.insert("height", or_default("height", "screen"));
        spec.insert("format", or_default("format", "rgba16f"));
        // copyDst lets chain handoffs and external uploads write the texture.
        spec.insert("usage", usage_2d());
        if effect_spec.get("is3D").is_truthy() {
            let depth = effect_spec.get("depth");
            let width = effect_spec.get("width");
            let depth = if depth.is_truthy() {
                depth.clone()
            } else if width.is_truthy() {
                width.clone()
            } else {
                Value::Number(64.0)
            };
            spec.insert("depth", depth);
            spec.insert("is3D", Value::Bool(true));
            spec.insert(
                "usage",
                Value::Array(
                    ["storage", "sample", "copySrc", "copyDst"]
                        .into_iter()
                        .map(Value::from)
                        .collect(),
                ),
            );
            // Definition-level filtering policy of 3D textures ('nearest' | 'linear').
            let filter = effect_spec.get("filter");
            if filter.is_truthy() {
                spec.insert("filter", filter.clone());
            }
        } else {
            // 2D allocation policies: a mip chain regenerated after each frame
            // that renders the texture, and contents preserved across recreation.
            let mipmaps = effect_spec.get("mipmaps");
            if !mipmaps.is_undefined() {
                spec.insert("mipmaps", mipmaps.clone());
            }
            let persistent = effect_spec.get("persistent");
            if !persistent.is_undefined() {
                spec.insert("persistent", persistent.clone());
            }
        }
        textures.insert(tex_id.clone(), Value::Object(spec));
    }

    // Pass outputs not already defined (global_ textures are surfaces).
    for pass in passes {
        let outputs = member(pass, "outputs")?;
        if !outputs.is_truthy() {
            continue;
        }
        for tex_id in js_values(outputs) {
            if starts_with(&tex_id, "texId", "global_")? {
                continue;
            }
            let key = js_string(&tex_id);
            if textures.contains_key(&key) {
                continue;
            }
            let mut spec = Object::new();
            spec.insert("width", Value::from("screen"));
            spec.insert("height", Value::from("screen"));
            spec.insert("format", Value::from("rgba16f"));
            spec.insert("usage", usage_2d());
            textures.insert(key, Value::Object(spec));
        }
    }
    Ok(textures)
}

/// `hashSource(source)`: the 32-bit string hash of the source (over UTF-16 code
/// units) in base 36.
pub fn hash_source(source: &str) -> String {
    let mut hash: i32 = 0;
    for unit in source.encode_utf16() {
        // ((hash << 5) - hash) + char, truncated to a 32-bit integer.
        hash = hash
            .wrapping_shl(5)
            .wrapping_sub(hash)
            .wrapping_add(i32::from(unit));
    }
    integer_to_string_radix(i64::from(hash), 36)
}

/// `formatError(err)`: a compilation error as readable text (`recompile` logs it).
pub fn format_error(err: &JsError) -> String {
    let err = match err {
        // An Error instance: only `message` is an own enumerable-looking member
        // the formatter reads; JSON.stringify of an Error is "{}".
        JsError::Error { message, .. } => {
            return if message.is_empty() {
                "{}".into()
            } else {
                message.clone()
            };
        }
        JsError::Thrown(v) => v,
    };
    let code = err.get("code");
    if is_str(code, "ERR_COMPILATION_FAILED")
        && let Value::Array(diagnostics) = err.get("diagnostics")
    {
        let text = diagnostics
            .iter()
            .filter(|d| is_str(d.get("severity"), "error"))
            .map(|d| {
                let message = d.get("message");
                let mut msg = if message.is_truthy() {
                    js_string(message)
                } else {
                    "Unknown error".into()
                };
                let location = d.get("location");
                if location.is_truthy() {
                    msg.push_str(&format!(
                        " (line {}, col {})",
                        js_string(location.get("line")),
                        js_string(location.get("column"))
                    ));
                }
                msg
            })
            .collect::<Vec<_>>()
            .join("; ");
        return if text.is_empty() {
            "Unknown compilation error".into()
        } else {
            text
        };
    }
    if is_str(code, "ERR_EXPANSION_FAILED")
        && let Value::Array(errors) = err.get("errors")
    {
        return errors
            .iter()
            .map(|e| {
                let message = e.get("message");
                if message.is_truthy() {
                    js_string(message)
                } else {
                    js_string(e)
                }
            })
            .collect::<Vec<_>>()
            .join("; ");
    }
    if is_str(code, "ERR_SHADER_COMPILE") {
        let detail = err.get("detail");
        return if detail.is_truthy() {
            js_string(detail)
        } else {
            "Shader compile error".into()
        };
    }
    let message = err.get("message");
    if message.is_truthy() {
        return js_string(message);
    }
    let detail = err.get("detail");
    if detail.is_truthy() {
        return js_string(detail);
    }
    match err {
        Value::Object(_) | Value::Array(_) | Value::Null => {
            err.to_json().unwrap_or_else(|| "undefined".into())
        }
        other => js_string(other),
    }
}

/// The oracle's `normalizeGraph(graph)`: program specs without their shader
/// source texts (`glsl`, `wgsl`, `vertex`, `fragment`).
pub fn normalize_graph(graph: &Value) -> Value {
    const SHADER_SOURCE_KEYS: [&str; 4] = ["glsl", "wgsl", "vertex", "fragment"];
    let mut programs = Object::new();
    let empty = Value::object();
    let source = if graph.get("programs").is_truthy() {
        graph.get("programs")
    } else {
        &empty
    };
    for (id, spec) in js_entries(source) {
        let out: Object = js_entries(&spec)
            .into_iter()
            .filter(|(k, _)| !SHADER_SOURCE_KEYS.contains(&k.as_str()))
            .collect();
        programs.insert(id, Value::Object(out));
    }
    let mut out = match graph {
        Value::Object(o) => o.clone(),
        _ => Object::new(),
    };
    out.remove("compiledAt");
    out.insert("programs", Value::Object(programs));
    Value::Object(out)
}

/// `compileGraph(src)` normalized as the oracle dumps it (Maps as objects,
/// program specs without shader source texts, no `compiledAt`).
pub fn dump_graph(src: &str, registry: &Registry) -> Result<Value, JsError> {
    compile_graph(src, registry, &CompileOptions::default()).map(|g| normalize_graph(&g))
}

/// The graph dump of `compileGraph(src)` built from an already validated program
/// (`validated` is the reference's `validate(parse(lex(src)))` output).
pub fn dump_graph_from_validated(
    src: &str,
    validated: &Value,
    registry: &Registry,
) -> Result<Value, JsError> {
    compile_graph_from_validated(src, validated, registry, &CompileOptions::default())
        .map(|g| normalize_graph(&g))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_matches_reference_examples() {
        assert_eq!(hash_source(""), "0");
        assert_eq!(
            hash_source(
                "search synth, filter\nnoise(seed: 1, scaleX: 50, scaleY: 50).adjust().write(o0)\nrender(o0)\n"
            ),
            "-5u2z87"
        );
    }

    #[test]
    fn format_error_messages() {
        let err = JsError::Thrown(crate::js!({
            "code": "ERR_EXPANSION_FAILED",
            "errors": [{"message": "Effect 'x' not found"}, {"message": "b"}]
        }));
        assert_eq!(format_error(&err), "Effect 'x' not found; b");
        let err = JsError::Thrown(crate::js!({
            "code": "ERR_COMPILATION_FAILED",
            "diagnostics": [
                {"severity": "warning", "message": "w"},
                {"severity": "error", "message": "bad", "location": {"line": 2, "column": 5}}
            ]
        }));
        assert_eq!(format_error(&err), "bad (line 2, col 5)");
        assert_eq!(format_error(&JsError::type_error("oops")), "oops");
    }

    /// `hashSource` and `formatError` against the reference's own functions
    /// (evaluated from `runtime/compiler.js`). Runs when `NM_REFERENCE_ROOT`
    /// points at the reference checkout and `node` is on PATH.
    #[test]
    fn hash_and_format_error_match_reference() {
        let Ok(root) = std::env::var("NM_REFERENCE_ROOT") else {
            eprintln!("skipping compiler reference parity: NM_REFERENCE_ROOT is not set");
            return;
        };
        let sources = [
            "",
            "a",
            "search synth\nnoise().write(o0)\nrender(o0)\n",
            "😀 emoji \u{1F600} and \u{e9}\u{301}",
            "\u{FFFF}\u{10000}",
            &"long line ".repeat(500),
        ];
        let errors = [
            r#"{"code":"ERR_COMPILATION_FAILED","diagnostics":[{"severity":"warning","message":"w"},{"severity":"error","message":"bad","location":{"line":2,"column":5}},{"severity":"error"},{"severity":"error","message":7}]}"#,
            r#"{"code":"ERR_COMPILATION_FAILED","diagnostics":[{"severity":"warning","message":"w"}]}"#,
            r#"{"code":"ERR_COMPILATION_FAILED","diagnostics":"x","message":"m"}"#,
            r#"{"code":"ERR_EXPANSION_FAILED","errors":[{"message":"Effect 'x' not found"},{"step":1},{"message":""}]}"#,
            r#"{"code":"ERR_SHADER_COMPILE","detail":"line 3"}"#,
            r#"{"code":"ERR_SHADER_COMPILE"}"#,
            r#"{"detail":"d"}"#,
            r#"{"other":[1,{"x":null}]}"#,
            r#"[1,2]"#,
            r#""thrown string""#,
            r#"42"#,
        ];
        let script = format!(
            r#"import {{ readFileSync }} from 'node:fs'
const src = readFileSync({path}, 'utf8').replace(/^import .*$/gm, '').replace(/^export /gm, '')
const {{ hashSource, formatError }} = new Function('compile', 'expand', 'allocateResources', 'createPipeline', src + '\nreturn {{ hashSource, formatError }}')()
const sources = {sources}
const errors = {errors}.map(t => JSON.parse(t))
errors.push(new TypeError('type message'), new SyntaxError(''))
process.stdout.write(JSON.stringify({{ hashes: sources.map(hashSource), formatted: errors.map(formatError) }}))"#,
            path =
                serde_json::to_string(&format!("{root}/shaders/src/runtime/compiler.js")).unwrap(),
            sources = serde_json::to_string(&sources).unwrap(),
            errors = serde_json::to_string(&errors).unwrap(),
        );
        let output = std::process::Command::new("node")
            .args(["--input-type=module", "-e", &script])
            .output()
            .expect("node is on PATH");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reference = Value::from_json(&String::from_utf8(output.stdout).unwrap()).unwrap();
        for (i, source) in sources.iter().enumerate() {
            assert_eq!(
                reference.get("hashes").at(i).as_str(),
                Some(hash_source(source).as_str()),
                "hashSource({source:?})"
            );
        }
        let mut thrown: Vec<JsError> = errors
            .iter()
            .map(|t| JsError::Thrown(Value::from_json(t).unwrap()))
            .collect();
        thrown.push(JsError::type_error("type message"));
        thrown.push(JsError::syntax(""));
        for (i, err) in thrown.iter().enumerate() {
            assert_eq!(
                reference.get("formatted").at(i).as_str(),
                Some(format_error(err).as_str()),
                "formatError({err:?})"
            );
        }
    }
}
