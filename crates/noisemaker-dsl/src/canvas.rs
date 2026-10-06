//! The free helpers `renderer/canvas.js` exports next to `CanvasRenderer`:
//! parameter cloning, DSL identifiers and enum names, and the classification
//! of effect definitions the demo UI and the host use (starter, mixer, 3D
//! generator or processor, volume and geometry parameters).
//!
//! The reference helpers take an effect record (`{namespace, name,
//! instance}`); these take the definition instance (`instance`, an
//! [`crate::registry::EffectEntry`]'s `def`). Definitions are read with
//! JavaScript property semantics; a malformed definition (a `null` parameter
//! spec, a non-array `passes`), where the reference would throw a
//! `TypeError` from inside the helper, reads as having no such member.

use crate::js::is_js_whitespace;
use crate::unparser::jsv::{entries, get_opt, json_stringify, values};
use crate::value::Value;

/// Known 3D generator effects (`KNOWN_3D_GENERATORS`): they initialize their
/// own volumes.
pub const KNOWN_3D_GENERATORS: &[&str] = &[
    "noise3d",
    "cell3d",
    "shape3d",
    "fractal3d",
    "flythrough3d",
    "cellularAutomata3d",
    "reactionDiffusion3d",
    "heightmap3d",
];

/// Known 3D processor effects (`KNOWN_3D_PROCESSORS`): they modify volumes and
/// need `inputTex3d`.
pub const KNOWN_3D_PROCESSORS: &[&str] = &[
    "flow3d",
    "palette3d",
    "render3d",
    "renderLit3d",
    "renderCubemap3d",
    "renderCubemapSurface",
    "renderLandscape3d",
];

/// The pass inputs that make an effect a chain filter (`isStarterEffect`).
const STARTER_PIPELINE_INPUTS: &[&str] = &[
    "inputTex",
    "inputTex3d",
    "o0",
    "o1",
    "o2",
    "o3",
    "o4",
    "o5",
    "o6",
    "o7",
];

/// `cloneParamValue(value)`: arrays are copied, objects deep-copied through
/// `JSON.parse(JSON.stringify(value))` (so `undefined` members and functions
/// are dropped and non-finite numbers become `null`), anything else is
/// returned as is.
pub fn clone_param_value(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.clone()),
        Value::Object(_) => json_stringify(value)
            .and_then(|text| Value::from_json(&text).ok())
            .unwrap_or_else(|| value.clone()),
        other => other.clone(),
    }
}

/// `isValidIdentifier(name)`: `/^[a-zA-Z_][a-zA-Z0-9_]*$/`, a name the DSL
/// accepts unquoted.
pub fn is_valid_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `sanitizeEnumName(name)`: whitespace runs removed with the next character
/// upper-cased (`"Cell Scale"` → `"CellScale"`), characters outside
/// `[a-zA-Z0-9_]` dropped; `None` when the result is not an identifier.
/// Whitespace is the regular-expression class `\s`.
pub fn sanitize_enum_name(name: &str) -> Option<String> {
    // name.replace(/\s+(.)/g, (_, c) => c.toUpperCase()).replace(/\s+/g, '')
    let chars: Vec<char> = name.chars().collect();
    let mut result = String::new();
    let mut i = 0;
    while i < chars.len() {
        if is_js_whitespace(chars[i]) {
            let mut j = i;
            while j < chars.len() && is_js_whitespace(chars[j]) {
                j += 1;
            }
            if j < chars.len() {
                result.extend(chars[j].to_uppercase());
                i = j + 1;
            } else {
                i = j;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }
    let result: String = result
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    is_valid_identifier(&result).then_some(result)
}

/// `effect.instance.globals.tex` (optional chaining).
fn tex_spec(def: &Value) -> Value {
    get_opt(&get_opt(def, "globals"), "tex")
}

/// `hasTexSurfaceParam(effect)`: a `tex` parameter of type `surface` (mixer
/// effects).
pub fn has_tex_surface_param(def: &Value) -> bool {
    get_opt(&tex_spec(def), "type").as_str() == Some("surface")
}

/// `hasExplicitTexParam(effect)`: a `surface` `tex` parameter whose default is
/// not `inputTex`, so it needs an explicit surface instead of the chain input.
pub fn has_explicit_tex_param(def: &Value) -> bool {
    let tex = tex_spec(def);
    get_opt(&tex, "type").as_str() == Some("surface")
        && get_opt(&tex, "default").as_str() != Some("inputTex")
}

/// The definition's passes (`instance.passes || []`).
fn passes(def: &Value) -> Vec<Value> {
    match get_opt(def, "passes") {
        Value::Array(passes) => passes,
        _ => Vec::new(),
    }
}

/// `needsInputTex3d(effect)`: a pass reads `inputTex3d` (3D consumers).
pub fn needs_input_tex3d(def: &Value) -> bool {
    passes(def).iter().any(|pass| {
        let inputs = get_opt(pass, "inputs");
        inputs.is_truthy()
            && values(&inputs)
                .iter()
                .any(|v| v.as_str() == Some("inputTex3d"))
    })
}

/// `instance.outputTex3d && list.includes(instance.func)`. The reference
/// returns the falsy `outputTex3d` operand itself when the effect has none;
/// callers use the result as a boolean.
fn is_known_3d(def: &Value, list: &[&str]) -> bool {
    get_opt(def, "outputTex3d").is_truthy()
        && get_opt(def, "func")
            .as_str()
            .is_some_and(|f| list.contains(&f))
}

/// `is3dGenerator(effect)`: a known 3D generator with a 3D output.
pub fn is_3d_generator(def: &Value) -> bool {
    is_known_3d(def, KNOWN_3D_GENERATORS)
}

/// `is3dProcessor(effect)`: a known 3D processor with a 3D output.
pub fn is_3d_processor(def: &Value) -> bool {
    is_known_3d(def, KNOWN_3D_PROCESSORS)
}

/// The result of `getVolGeoParams`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VolGeoParams {
    /// The first parameter of type `volume`.
    pub vol_param: Option<String>,
    /// The first parameter of type `geometry`.
    pub geo_param: Option<String>,
}

impl VolGeoParams {
    /// `{volParam, geoParam}` (`null` when absent).
    pub fn to_value(&self) -> Value {
        let mut o = crate::value::Object::new();
        let opt = |p: &Option<String>| p.as_deref().map_or(Value::Null, Value::from);
        o.insert("volParam", opt(&self.vol_param));
        o.insert("geoParam", opt(&self.geo_param));
        Value::Object(o)
    }
}

/// `getVolGeoParams(effect)`: the first `volume` and `geometry` parameters.
pub fn get_vol_geo_params(def: &Value) -> VolGeoParams {
    let globals = get_opt(def, "globals");
    let mut out = VolGeoParams::default();
    if !globals.is_truthy() {
        return out;
    }
    for (key, spec) in entries(&globals) {
        let ty = get_opt(&spec, "type");
        if ty.as_str() == Some("volume") && out.vol_param.is_none() {
            out.vol_param = Some(key.clone());
        }
        if ty.as_str() == Some("geometry") && out.geo_param.is_none() {
            out.geo_param = Some(key);
        }
    }
    out
}

/// `isStarterEffect(effect)`: no pass reads a pipeline input (`inputTex`,
/// `inputTex3d`, `o0`..`o7`), so the effect can begin a chain. An effect
/// without passes is a starter.
pub fn is_starter_effect(def: &Value) -> bool {
    let passes = passes(def);
    if passes.is_empty() {
        return true;
    }
    !passes.iter().any(|pass| match pass.get("inputs") {
        Value::Object(inputs) => inputs.values().any(|v| {
            v.as_str()
                .is_some_and(|s| STARTER_PIPELINE_INPUTS.contains(&s))
        }),
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize() {
        assert_eq!(
            sanitize_enum_name("Cell Scale").as_deref(),
            Some("CellScale")
        );
        assert_eq!(sanitize_enum_name("a  b c").as_deref(), Some("aBC"));
        assert_eq!(sanitize_enum_name("3d"), None);
        assert_eq!(sanitize_enum_name("x-y"), Some("xy".into()));
        // `\s` includes U+FEFF and excludes U+0085.
        assert_eq!(sanitize_enum_name("a\u{feff}b").as_deref(), Some("aB"));
        assert_eq!(sanitize_enum_name("a\u{85}b").as_deref(), Some("ab"));
        assert_eq!(
            sanitize_enum_name("soft ßtep").as_deref(),
            Some("softSStep")
        );
    }

    #[test]
    fn clone_param() {
        let v = Value::from_json(r#"{"a": [1, 2], "b": {"c": "d"}}"#).unwrap();
        assert_eq!(clone_param_value(&v), v);
        let mut o = crate::value::Object::new();
        o.insert("u", Value::Undefined);
        o.insert("n", Value::Number(f64::NAN));
        let cloned = clone_param_value(&Value::Object(o));
        assert!(!cloned.as_object().unwrap().contains_key("u"));
        assert!(cloned.get("n").is_null());
    }
}
