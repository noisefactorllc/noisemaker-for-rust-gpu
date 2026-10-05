//! Port of `runtime/expander.js`: expands the logical graph (the validator's
//! plans) into the render graph — render passes, shader programs, texture specs,
//! the render surface and the media steps.
//!
//! The port follows the reference function by function and statement by
//! statement. JavaScript data semantics are kept with [`Value`]: the reference's
//! `Map`s iterate in insertion order ([`IndexMap`]), its plain objects in
//! JavaScript property order ([`Object`]), members may hold `undefined`, and the
//! TypeErrors the reference raises on some inputs (`plan.chain is not iterable`,
//! property reads of `null`/`undefined`, `startsWith` on a non-string, a
//! non-integral palette index) are raised at the same point with the same message.

use std::borrow::Cow;

use indexmap::IndexMap;

use crate::error::JsError;
use crate::js::{
    is_finite_number, parse_int, same_value_zero, strict_equals, value_to_property_key,
};
use crate::palette::expand_palette;
use crate::registry::Registry;
use crate::value::{Object, Value};

// --- JavaScript semantics ------------------------------------------------------

static UNDEFINED: Value = Value::Undefined;

/// `String(value)`: the conversion template literals and property keys apply.
pub(crate) fn js_string(value: &Value) -> String {
    value_to_property_key(value)
}

/// The TypeError a member read of `null`/`undefined` raises.
fn read_error(base: &Value, key: &str) -> JsError {
    let what = if base.is_null() { "null" } else { "undefined" };
    JsError::type_error(format!(
        "Cannot read properties of {what} (reading '{key}')"
    ))
}

/// `base.key` (a non-optional member read): reading a member of `null` or
/// `undefined` throws; other primitives read `undefined`.
pub(crate) fn member<'a>(base: &'a Value, key: &str) -> Result<&'a Value, JsError> {
    if base.is_nullish() {
        return Err(read_error(base, key));
    }
    Ok(base.get(key))
}

/// `base?.key`.
fn opt_member<'a>(base: &'a Value, key: &str) -> &'a Value {
    if base.is_nullish() {
        &UNDEFINED
    } else {
        base.get(key)
    }
}

/// `value.startsWith(prefix)`, with the TypeErrors the reference raises when
/// `value` (the source expression `expr`) is not a string.
pub(crate) fn starts_with(value: &Value, expr: &str, prefix: &str) -> Result<bool, JsError> {
    match value {
        Value::String(s) => Ok(s.starts_with(prefix)),
        Value::Undefined | Value::Null => Err(read_error(value, "startsWith")),
        _ => Err(JsError::type_error(format!(
            "{expr}.startsWith is not a function"
        ))),
    }
}

/// `value === "s"`.
pub(crate) fn is_str(value: &Value, s: &str) -> bool {
    matches!(value, Value::String(v) if v == s)
}

/// `set.has(value)` for a `Set` kept as a vector.
fn set_has(set: &[Value], value: &Value) -> bool {
    set.iter().any(|v| same_value_zero(v, value))
}

/// `set.add(value)` for a `Set` kept as a vector.
fn set_add(set: &mut Vec<Value>, value: Value) {
    if !set_has(set, &value) {
        set.push(value);
    }
}

/// `a || b`.
fn or<'a>(a: &'a Value, b: &'a Value) -> &'a Value {
    if a.is_truthy() { a } else { b }
}

/// `a || "fallback"`.
fn or_str(a: &Value, fallback: &str) -> Value {
    if a.is_truthy() {
        a.clone()
    } else {
        Value::from(fallback)
    }
}

/// `value !== null && typeof value === 'object'`.
fn is_object_type(value: &Value) -> bool {
    matches!(value, Value::Object(_) | Value::Array(_))
}

/// The UTF-16 code units of a string as one-unit strings (string indexing).
fn code_unit_strings(s: &str) -> Vec<Value> {
    s.encode_utf16()
        .map(|u| Value::String(String::from_utf16_lossy(&[u])))
        .collect()
}

/// `Object.entries(value)`, `Object.values(value)` and `Object.keys(value)`
/// for the (non-nullish) values the reference enumerates.
pub(crate) use crate::unparser::jsv::{
    entries as js_entries, keys as js_keys, values as js_values,
};

/// `{ ...value }`.
fn spread(value: &Value) -> Object {
    match value {
        Value::Object(o) => o.clone(),
        Value::Array(_) | Value::String(_) => js_entries(value).into_iter().collect(),
        _ => Object::new(),
    }
}

/// `Object.prototype.hasOwnProperty.call(value, key)`.
fn has_own(value: &Value, key: &str) -> bool {
    match value {
        Value::Object(o) => o.contains_key(key),
        Value::Array(a) => {
            key == "length"
                || key
                    .parse::<usize>()
                    .is_ok_and(|i| i < a.len() && i.to_string() == key)
        }
        Value::String(s) => {
            key == "length"
                || key
                    .parse::<usize>()
                    .is_ok_and(|i| i < s.encode_utf16().count() && i.to_string() == key)
        }
        _ => false,
    }
}

/// `key in value` for an object-type `value`.
fn has_property(value: &Value, key: &str) -> bool {
    has_own(value, key)
}

/// `Array.prototype.sort()` order: UTF-16 code unit order.
fn js_default_compare(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

/// The collation weights of `String.prototype.localeCompare` (ICU root
/// collation, V8's default) for one character: `None` for a completely
/// ignorable character, else `(primary, tertiary)`. ASCII follows the root
/// collation order (whitespace, punctuation and symbols, digits, then letters
/// with lowercase before uppercase at the tertiary level); characters beyond
/// ASCII, which no catalog define name uses, order by code point after ASCII.
fn collation_weights(c: char) -> Option<(u32, u32)> {
    const ORDER: &str = "\t\n\u{b}\u{c}\r _-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$0123456789";
    let code = c as u32;
    if code < 0x80 {
        if c.is_ascii_control() && !('\t'..='\r').contains(&c) {
            return None;
        }
        if c.is_ascii_alphabetic() {
            let base = ORDER.chars().count() as u32;
            let letter = (c.to_ascii_lowercase() as u32) - ('a' as u32);
            return Some((base + letter, u32::from(c.is_ascii_uppercase())));
        }
        let rank = ORDER
            .chars()
            .position(|o| o == c)
            .expect("ASCII order is complete");
        return Some((rank as u32, 0));
    }
    Some((0x100 + code, 0))
}

/// `a.localeCompare(b)` (see [`collation_weights`]).
fn locale_compare(a: &str, b: &str) -> std::cmp::Ordering {
    let weights =
        |s: &str| -> Vec<(u32, u32)> { s.chars().filter_map(collation_weights).collect() };
    let (wa, wb) = (weights(a), weights(b));
    let primary = |w: &[(u32, u32)]| w.iter().map(|p| p.0).collect::<Vec<_>>();
    let tertiary = |w: &[(u32, u32)]| w.iter().map(|p| p.1).collect::<Vec<_>>();
    primary(&wa)
        .cmp(&primary(&wb))
        .then_with(|| tertiary(&wa).cmp(&tertiary(&wb)))
}

/// `for (const x of value)`: the iterated elements, or the TypeError the
/// reference raises for a non-iterable `value` (the source expression `expr`).
fn iterate<'a>(value: &'a Value, expr: &str) -> Result<Cow<'a, [Value]>, JsError> {
    match value {
        Value::Array(a) => Ok(Cow::Borrowed(a.as_slice())),
        Value::String(s) => Ok(Cow::Owned(
            s.chars().map(|c| Value::String(c.to_string())).collect(),
        )),
        _ => Err(JsError::type_error(format!("{expr} is not iterable"))),
    }
}

// --- Patterns --------------------------------------------------------------------

/// `SURFACE_REF_PATTERN.test(name)`: `/^(?:o|vol|geo|xyz|vel|rgba)[0-7]$/`.
fn is_surface_ref(name: &str) -> bool {
    ["o", "vol", "geo", "xyz", "vel", "rgba"]
        .iter()
        .any(|prefix| {
            name.strip_prefix(prefix).is_some_and(|rest| {
                rest.len() == 1 && rest.as_bytes()[0].is_ascii_digit() && rest.as_bytes()[0] <= b'7'
            })
        })
}

/// `/^global_(xyz|vel|rgba|points_trail|life_data)$/.test(name)`.
fn is_particle_global(name: &str) -> bool {
    name.strip_prefix("global_")
        .is_some_and(|rest| matches!(rest, "xyz" | "vel" | "rgba" | "points_trail" | "life_data"))
}

/// `TEXTURE_ARG_KINDS`.
const TEXTURE_ARG_KINDS: &[&str] = &[
    "temp", "output", "source", "feedback", "vol", "geo", "xyz", "vel", "rgba", "pipeline",
];

/// `isTextureArg(arg)`.
fn is_texture_arg(arg: &Value) -> bool {
    is_object_type(arg)
        && matches!(arg.get("kind"), Value::String(kind) if TEXTURE_ARG_KINDS.contains(&kind.as_str()))
}

// --- Helpers of the reference module -------------------------------------------------

/// The pipeline lanes a chain carries from step to step: `currentInput`,
/// `currentInput3d`, `currentInputGeo`, `currentInputXyz`, `currentInputVel`,
/// `currentInputRgba`.
#[derive(Debug, Clone)]
struct Lanes {
    input: Value,
    input3d: Value,
    geo: Value,
    xyz: Value,
    vel: Value,
    rgba: Value,
}

impl Lanes {
    fn new() -> Self {
        Lanes {
            input: Value::Null,
            input3d: Value::Null,
            geo: Value::Null,
            xyz: Value::Null,
            vel: Value::Null,
            rgba: Value::Null,
        }
    }
}

type TextureMap = IndexMap<String, Value>;

/// `textureMap.get(key)`.
fn texture_map_get(texture_map: &TextureMap, key: &str) -> Value {
    texture_map.get(key).cloned().unwrap_or(Value::Undefined)
}

/// `registerPassthrough(nodeId, textureMap, currentInput, ...)`: register
/// passthrough outputs for a node that doesn't generate passes.
fn register_passthrough(node_id: &str, texture_map: &mut TextureMap, lanes: &Lanes) {
    if lanes.input.is_truthy() {
        texture_map.insert(format!("{node_id}_out"), lanes.input.clone());
    }
    if lanes.input3d.is_truthy() {
        texture_map.insert(format!("{node_id}_out3d"), lanes.input3d.clone());
    }
    if lanes.geo.is_truthy() {
        texture_map.insert(format!("{node_id}_outGeo"), lanes.geo.clone());
    }
    if lanes.xyz.is_truthy() {
        texture_map.insert(format!("{node_id}_outXyz"), lanes.xyz.clone());
    }
    if lanes.vel.is_truthy() {
        texture_map.insert(format!("{node_id}_outVel"), lanes.vel.clone());
    }
    if lanes.rgba.is_truthy() {
        texture_map.insert(format!("{node_id}_outRgba"), lanes.rgba.clone());
    }
}

/// The GLSL fragment source of the blit program (`ensureBlitProgram`).
pub const BLIT_FRAGMENT: &str = "#version 300 es
            precision highp float;
            in vec2 v_texCoord;
            uniform sampler2D src;
            out vec4 fragColor;
            void main() {
                fragColor = texture(src, v_texCoord);
            }";

/// The WGSL source of the blit program (`ensureBlitProgram`).
pub const BLIT_WGSL: &str = "
            struct FragmentInput {
                @builtin(position) position: vec4<f32>,
                @location(0) uv: vec2<f32>,
            }

            @group(0) @binding(0) var src: texture_2d<f32>;
            @group(0) @binding(1) var srcSampler: sampler;

            @fragment
            fn main(in: FragmentInput) -> @location(0) vec4<f32> {
                let uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
                return textureSample(src, srcSampler, uv);
            }
        ";

/// `ensureBlitProgram(programs)`: ensure the blit copy program is registered.
fn ensure_blit_program(programs: &mut Object) {
    if programs.get("blit").is_some_and(Value::is_truthy) {
        return;
    }
    let mut blit = Object::new();
    blit.insert("fragment", Value::from(BLIT_FRAGMENT));
    blit.insert("wgsl", Value::from(BLIT_WGSL));
    blit.insert("fragmentEntryPoint", Value::from("main"));
    programs.insert("blit", Value::Object(blit));
}

/// `resolveGlobalSurfaceRef(name)`: resolve a surface ref string to a `global_`
/// prefixed name.
fn resolve_global_surface_ref(name: &str) -> String {
    if name == "none" {
        return "none".into();
    }
    if name.starts_with("global_") {
        return name.into();
    }
    if is_surface_ref(name) {
        return format!("global_{name}");
    }
    name.into()
}

/// `resolveEnum(path)` of `expand`: walk the standard enums (`stdEnums` of
/// `lang/std_enums.js`, never the merged effect enums) and return the leaf's
/// `value`, or `None` (the reference's `null`).
fn resolve_enum(std_enums: &Value, path: &str) -> Option<Value> {
    let mut node = std_enums;
    for part in path.split('.') {
        // `node && node[part]`: every intermediate value must be truthy. Only
        // own members of plain objects can lead to a leaf (a walk into the
        // prototype chain or a primitive never ends at an object with a `value`).
        match node {
            Value::Object(o) => match o.get(part) {
                Some(next) if next.is_truthy() => node = next,
                _ => return None,
            },
            _ => return None,
        }
    }
    match node {
        Value::Object(o) => match o.get("value") {
            Some(v) if !v.is_undefined() => Some(v.clone()),
            _ => None,
        },
        _ => None,
    }
}

/// `if (def.type === 'member' && typeof val === 'string') { const resolved =
/// resolveEnum(val); if (resolved !== null) val = resolved }`.
fn resolve_member(std_enums: &Value, def: &Value, val: Value) -> Result<Value, JsError> {
    if is_str(member(def, "type")?, "member")
        && let Value::String(path) = &val
        && let Some(resolved) = resolve_enum(std_enums, path)
        && !resolved.is_null()
    {
        return Ok(resolved);
    }
    Ok(val)
}

/// `scopeParticleTex(texName)`: scope particle textures to the current pipeline.
fn scope_particle_tex(tex_name: &str, particle_pipeline_id: Option<&str>) -> String {
    let Some(pipeline_id) = particle_pipeline_id else {
        return tex_name.into();
    };
    if is_particle_global(tex_name) {
        return format!("{tex_name}_{pipeline_id}");
    }
    tex_name.into()
}

/// `scopeChainTex(texName)`: particle scoping takes priority; every other
/// `global_` texture is scoped to the chain.
fn scope_chain_tex(
    tex_name: &str,
    particle_pipeline_id: Option<&str>,
    chain_scope_id: &str,
) -> String {
    let particle_result = scope_particle_tex(tex_name, particle_pipeline_id);
    if particle_result != tex_name {
        return particle_result;
    }
    if tex_name.starts_with("global_") {
        return format!("{tex_name}_{chain_scope_id}");
    }
    tex_name.into()
}

/// `dimReferencesParam(dim)`.
fn dim_references_param(dim: &Value) -> bool {
    is_object_type(dim)
        && (!dim.get("param").is_undefined() || !dim.get("screenDivide").is_undefined())
}

/// `scopedParamMap`: original sizing param -> scoped param, a `Map` keyed by the
/// param value.
type ScopedParamMap = Vec<(Value, String)>;

fn scoped_param_map_set(map: &mut ScopedParamMap, key: Value, scoped: String) {
    match map.iter_mut().find(|(k, _)| same_value_zero(k, &key)) {
        Some(slot) => slot.1 = scoped,
        None => map.push((key, scoped)),
    }
}

/// The scope a texture's sizing params are scoped to (`scopeDimSpec`'s closure).
struct DimScope<'a> {
    tex_name: &'a str,
    particle_pipeline_id: Option<&'a str>,
    scope_suffix: &'a str,
    volume_size_param: &'a str,
}

/// `scopeDimSpec(dimSpec)`: scope a dimension spec's param reference to this
/// pipeline/chain and track the mapping for uniform propagation.
fn scope_dim_spec(
    dim_spec: &Value,
    scope: &DimScope<'_>,
    scoped_param_map: &mut ScopedParamMap,
) -> Result<Value, JsError> {
    if is_object_type(dim_spec) || dim_spec.is_null() {
        // `typeof dimSpec === 'object' && dimSpec.param !== undefined` (reads
        // `null.param` for a null dimension, as the reference does).
        let original_param = member(dim_spec, "param")?;
        if !original_param.is_undefined() {
            let dimension_scope = if is_str(original_param, "stateSize")
                && scope.particle_pipeline_id.is_some()
                && !scope.tex_name.starts_with("global_")
            {
                scope.particle_pipeline_id.unwrap_or_default()
            } else {
                scope.scope_suffix
            };
            let scoped_param = if is_str(original_param, "volumeSize") {
                scope.volume_size_param.to_owned()
            } else {
                format!("{}_{dimension_scope}", js_string(original_param))
            };
            scoped_param_map_set(
                scoped_param_map,
                original_param.clone(),
                scoped_param.clone(),
            );
            let mut out = spread(dim_spec);
            out.insert("param", Value::String(scoped_param));
            return Ok(Value::Object(out));
        }
        let original_param = member(dim_spec, "screenDivide")?;
        if !original_param.is_undefined() {
            let scoped_param = format!("{}_{}", js_string(original_param), scope.scope_suffix);
            scoped_param_map_set(
                scoped_param_map,
                original_param.clone(),
                scoped_param.clone(),
            );
            let mut out = spread(dim_spec);
            out.insert("screenDivide", Value::String(scoped_param));
            return Ok(Value::Object(out));
        }
    }
    Ok(dim_spec.clone())
}

/// A volume exported by `write3d`: `{ param, value }` (the writer chain's
/// scoped `volumeSize` param and its value).
#[derive(Debug, Clone)]
struct WrittenVolume {
    param: String,
    value: Value,
}

/// A volume read by `read3d`: `{ surface, writer }`.
#[derive(Debug, Clone)]
struct ReadVolume {
    surface: Value,
    writer: Option<WrittenVolume>,
}

/// `map.get(key)` for a `Map` whose keys are all strings: any other key misses.
fn get_by_value<'a, T>(map: &'a IndexMap<String, T>, key: &Value) -> Option<&'a T> {
    match key {
        Value::String(k) => map.get(k),
        _ => None,
    }
}

// --- expand ----------------------------------------------------------------------

/// Expansion options (`options` of `expand`).
#[derive(Debug, Clone, Default)]
pub struct ExpandOptions {
    /// `options.shaderOverrides`: per-step shader overrides keyed by step index,
    /// e.g. `{ "0": { "main": { "glsl": "...", "wgsl": "..." } } }`. An override
    /// replaces the effect's `shaders` for that step.
    pub shader_overrides: Object,
}

/// The result of `expand`: `{ passes, errors, programs, textureSpecs,
/// renderSurface, mediaSteps }`.
#[derive(Debug, Clone, PartialEq)]
pub struct Expansion {
    /// Render passes in execution order (pass objects as the reference builds them).
    pub passes: Vec<Value>,
    /// `{ message, step? }` expansion errors.
    pub errors: Vec<Value>,
    /// Program id -> program spec (shader sources, `uniformLayout`, `defines`).
    pub programs: Object,
    /// Virtual texture id -> effect-defined texture spec.
    pub texture_specs: Object,
    /// The surface to present (`'o0'`, ...), or `null`.
    pub render_surface: Value,
    /// `{ textureId, uniform, stepIndex, effect }` per media texture.
    pub media_steps: Vec<Value>,
}

impl Expansion {
    /// The expansion as the reference's return value.
    pub fn to_value(&self) -> Value {
        let mut out = Object::new();
        out.insert("passes", Value::Array(self.passes.clone()));
        out.insert("errors", Value::Array(self.errors.clone()));
        out.insert("programs", Value::Object(self.programs.clone()));
        out.insert("textureSpecs", Value::Object(self.texture_specs.clone()));
        out.insert("renderSurface", self.render_surface.clone());
        out.insert("mediaSteps", Value::Array(self.media_steps.clone()));
        Value::Object(out)
    }
}

/// `expand(compilationResult)` serialized as the oracle dumps it (the `expanded`
/// stage), with no options.
pub fn expand_to_value(compilation: &Value, registry: &Registry) -> Result<Value, JsError> {
    expand(compilation, registry, &ExpandOptions::default()).map(|e| e.to_value())
}

/// The per-pass inputs of `expand` that stay fixed for one step.
struct StepContext<'a> {
    plan: &'a Value,
    step: &'a Value,
    is_last_step: bool,
    effect_name: &'a Value,
    effect_def: &'a Value,
    node_id: &'a str,
    chain_scope_id: &'a str,
    particle_pipeline_id: Option<&'a str>,
    std_enums: &'a Value,
}

/// `expand(compilationResult, options)`: expand the logical graph (plans) into a
/// render graph (passes).
pub fn expand(
    compilation_result: &Value,
    registry: &Registry,
    options: &ExpandOptions,
) -> Result<Expansion, JsError> {
    let shader_overrides = &options.shader_overrides;
    let std_enums = Value::Object(registry.std_enums());
    let mut passes: Vec<Value> = Vec::new();
    let mut errors: Vec<Value> = Vec::new();
    let mut programs = Object::new();
    let mut texture_specs = Object::new(); // nodeId_texName -> { width, height, format, is3D?, depth? }
    let mut texture_map: TextureMap = IndexMap::new(); // logical_id -> virtual_texture_id
    let mut written_volumes: IndexMap<String, WrittenVolume> = IndexMap::new();
    let mut read_volumes: IndexMap<String, ReadVolume> = IndexMap::new();
    let mut exported_textures: IndexMap<String, Value> = IndexMap::new();
    let mut media_steps: Vec<Value> = Vec::new();
    let mut media_step_ids: Vec<String> = Vec::new();
    let mut last_written_surface = Value::Null;

    // 1. Expand each plan into passes.
    let plans = iterate(
        member(compilation_result, "plans")?,
        "compilationResult.plans",
    )?;
    for (plan_index, plan) in plans.iter().enumerate() {
        let mut lanes = Lanes::new();
        // The last inline write target ({kind, name}), to avoid a redundant final blit.
        let mut last_inline_write_target: Option<(Value, Value)> = None;
        // The current particle pipeline scope (the node that created the particle textures).
        let mut current_particle_pipeline_id: Option<String> = None;
        // Pipeline uniforms accumulate from upstream effects for downstream consumption.
        let mut pipeline_uniforms = Object::new();
        // `compilationResult.plans.indexOf(plan)`: plans are distinct objects.
        let chain_scope_id = format!("chain_{plan_index}");
        let volume_size_param = format!("volumeSize_{chain_scope_id}");

        let chain_value = member(plan, "chain")?;
        let chain = iterate(chain_value, "plan.chain")?;
        let chain_len = chain.len();
        for (step_position, step) in chain.iter().enumerate() {
            let builtin = member(step, "builtin")?.is_truthy();
            let op = member(step, "op")?;

            // Builtin read: sets the current input.
            if builtin && is_str(op, "_read") {
                let tex = opt_member(member(step, "args")?, "tex");
                if tex.is_truthy() && is_str(member(tex, "kind")?, "output") {
                    lanes.input =
                        Value::String(format!("global_{}", js_string(member(tex, "name")?)));
                }
                // Register the read output so subsequent steps can find it via step.from.
                let node_id = format!("node_{}", js_string(member(step, "temp")?));
                texture_map.insert(format!("{node_id}_out"), lanes.input.clone());
                continue;
            }
            if builtin && is_str(op, "_read3d") {
                let args = member(step, "args")?;
                let tex3d = opt_member(args, "tex3d");
                let geo = opt_member(args, "geo");
                if tex3d.is_truthy() {
                    // VolRef (vol0-vol7) or a plain name.
                    if is_str(member(tex3d, "kind")?, "vol")
                        || is_str(member(tex3d, "type")?, "VolRef")
                    {
                        lanes.input3d =
                            Value::String(format!("global_{}", js_string(member(tex3d, "name")?)));
                    } else {
                        lanes.input3d = or(member(tex3d, "name")?, tex3d).clone();
                    }
                }
                if geo.is_truthy() {
                    // GeoRef (geo0-geo7) or a plain name.
                    if is_str(member(geo, "kind")?, "geo") || is_str(member(geo, "type")?, "GeoRef")
                    {
                        lanes.geo =
                            Value::String(format!("global_{}", js_string(member(geo, "name")?)));
                    } else {
                        lanes.geo = or(member(geo, "name")?, geo).clone();
                    }
                }
                // The producer scope is resolved after all plans have been
                // expanded: readers may precede writers to consume the previous frame.
                let volume = get_by_value(&written_volumes, &lanes.input3d).cloned();
                if lanes.input3d.is_truthy() {
                    // Preserve the writer visible at this read.
                    let size = match &volume {
                        Some(v) if !v.value.is_nullish() => v.value.clone(),
                        _ => Value::Number(64.0),
                    };
                    read_volumes.insert(
                        volume_size_param.clone(),
                        ReadVolume {
                            surface: lanes.input3d.clone(),
                            writer: volume,
                        },
                    );
                    pipeline_uniforms.insert("volumeSize", size.clone());
                    pipeline_uniforms.insert(volume_size_param.clone(), size);
                }
                // Register the read3d output so subsequent steps can find it via step.from.
                let node_id = format!("node_{}", js_string(member(step, "temp")?));
                if lanes.input3d.is_truthy() {
                    texture_map.insert(format!("{node_id}_out3d"), lanes.input3d.clone());
                }
                if lanes.geo.is_truthy() {
                    texture_map.insert(format!("{node_id}_outGeo"), lanes.geo.clone());
                }
                continue;
            }

            // Builtin write: output to a surface AND pass through (write() is chainable).
            if builtin && is_str(op, "_write") {
                let tex = opt_member(member(step, "args")?, "tex");
                if tex.is_truthy() && lanes.input.is_truthy() {
                    // Skip the blit if the target is "none" - just pass through.
                    let tex_name = member(tex, "name")?;
                    if !is_str(tex_name, "none") {
                        let target_surface = format!("global_{}", js_string(tex_name));
                        // Only blit if the current input is not already the target surface.
                        if !is_str(&lanes.input, &target_surface) {
                            let node_id = format!("node_{}", js_string(member(step, "temp")?));
                            let mut blit = Object::new();
                            blit.insert("id", Value::String(format!("{node_id}_write_blit")));
                            blit.insert("program", Value::from("blit"));
                            blit.insert("type", Value::from("render"));
                            let mut inputs = Object::new();
                            inputs.insert("src", lanes.input.clone());
                            blit.insert("inputs", Value::Object(inputs));
                            let mut outputs = Object::new();
                            outputs.insert("color", Value::String(target_surface));
                            blit.insert("outputs", Value::Object(outputs));
                            blit.insert("uniforms", Value::object());
                            blit.insert("nodeId", Value::String(node_id));
                            blit.insert("stepIndex", member(step, "temp")?.clone());
                            passes.push(Value::Object(blit));

                            ensure_blit_program(&mut programs);

                            // Track the last written surface for the render directive.
                            last_written_surface = tex_name.clone();
                            // Track this inline write target to skip a redundant final blit.
                            last_inline_write_target =
                                Some((member(tex, "kind")?.clone(), tex_name.clone()));
                        }
                    }
                    // Pass through: the output of write() is the texture that was written.
                    let node_id = format!("node_{}", js_string(member(step, "temp")?));
                    texture_map.insert(format!("{node_id}_out"), lanes.input.clone());
                }
                continue;
            }

            // Builtin write3d: write the 3D volume and geometry to global surfaces (chainable).
            if builtin && is_str(op, "_write3d") {
                let args = member(step, "args")?;
                let tex3d = opt_member(args, "tex3d");
                let geo = opt_member(args, "geo");
                let node_id = format!("node_{}", js_string(member(step, "temp")?));

                // Blit the 3D volume to the target global surface (skip "none").
                if tex3d.is_truthy()
                    && !is_str(member(tex3d, "name")?, "none")
                    && lanes.input3d.is_truthy()
                {
                    let target_vol = format!("global_{}", js_string(member(tex3d, "name")?));
                    exported_textures.insert(target_vol.clone(), lanes.input3d.clone());
                    let source_key = js_string(&lanes.input3d);
                    if let Some(spec) = texture_specs.get(&source_key).filter(|s| s.is_truthy()) {
                        let copy = spread(spec);
                        texture_specs.insert(target_vol.clone(), Value::Object(copy));
                    }
                    if let Some(volume_size) = pipeline_uniforms.get("volumeSize")
                        && !volume_size.is_undefined()
                    {
                        written_volumes.insert(
                            target_vol.clone(),
                            WrittenVolume {
                                param: volume_size_param.clone(),
                                value: volume_size.clone(),
                            },
                        );
                    }
                    // Only blit if the current input is not already the target.
                    if !is_str(&lanes.input3d, &target_vol) {
                        let mut blit = Object::new();
                        blit.insert("id", Value::String(format!("{node_id}_write3d_vol_blit")));
                        blit.insert("program", Value::from("blit"));
                        blit.insert("type", Value::from("render"));
                        let mut inputs = Object::new();
                        inputs.insert("src", lanes.input3d.clone());
                        blit.insert("inputs", Value::Object(inputs));
                        let mut outputs = Object::new();
                        outputs.insert("color", Value::String(target_vol));
                        blit.insert("outputs", Value::Object(outputs));
                        blit.insert("uniforms", Value::object());
                        blit.insert("nodeId", Value::String(node_id.clone()));
                        blit.insert("stepIndex", member(step, "temp")?.clone());
                        passes.push(Value::Object(blit));

                        ensure_blit_program(&mut programs);
                    }
                }

                // Blit the geometry buffer to the target global surface (skip "none").
                if geo.is_truthy() && !is_str(member(geo, "name")?, "none") && lanes.geo.is_truthy()
                {
                    let target_geo = format!("global_{}", js_string(member(geo, "name")?));
                    exported_textures.insert(target_geo.clone(), lanes.geo.clone());
                    let source_key = js_string(&lanes.geo);
                    if let Some(spec) = texture_specs.get(&source_key).filter(|s| s.is_truthy()) {
                        let copy = spread(spec);
                        texture_specs.insert(target_geo.clone(), Value::Object(copy));
                    }
                    // Only blit if the current input is not already the target.
                    if !is_str(&lanes.geo, &target_geo) {
                        let mut blit = Object::new();
                        blit.insert("id", Value::String(format!("{node_id}_write3d_geo_blit")));
                        blit.insert("program", Value::from("blit"));
                        blit.insert("type", Value::from("render"));
                        let mut inputs = Object::new();
                        inputs.insert("src", lanes.geo.clone());
                        blit.insert("inputs", Value::Object(inputs));
                        let mut outputs = Object::new();
                        outputs.insert("color", Value::String(target_geo));
                        blit.insert("outputs", Value::Object(outputs));
                        blit.insert("uniforms", Value::object());
                        blit.insert("nodeId", Value::String(node_id.clone()));
                        blit.insert("stepIndex", member(step, "temp")?.clone());
                        passes.push(Value::Object(blit));
                    }
                }

                // Pass through: the outputs of write3d() are the textures that were written.
                texture_map.insert(format!("{node_id}_out"), lanes.input.clone());
                texture_map.insert(format!("{node_id}_out3d"), lanes.input3d.clone());
                texture_map.insert(format!("{node_id}_outGeo"), lanes.geo.clone());
                continue;
            }

            // Subchain begin/end markers are metadata nodes that pass through.
            if builtin && (is_str(op, "_subchain_begin") || is_str(op, "_subchain_end")) {
                let node_id = format!("node_{}", js_string(member(step, "temp")?));
                register_passthrough(&node_id, &mut texture_map, &lanes);
                continue;
            }

            // Any non-write step continues the chain after an inline write.
            last_inline_write_target = None;

            // `_skip`: pass the current input through unchanged.
            if is_true(opt_member(member(step, "args")?, "_skip")) {
                let node_id = format!("node_{}", js_string(member(step, "temp")?));
                register_passthrough(&node_id, &mut texture_map, &lanes);
                continue;
            }

            let effect_name = op;
            let effect = match effect_name {
                Value::String(name) => registry.get_effect(name),
                _ => None,
            };
            let Some(effect) = effect else {
                let mut error = Object::new();
                error.insert(
                    "message",
                    Value::String(format!("Effect '{}' not found", js_string(effect_name))),
                );
                error.insert("step", step.clone());
                errors.push(Value::Object(error));
                continue;
            };
            let effect_def = &effect.def;

            // A unique id for this effect instance.
            let node_id = format!("node_{}", js_string(member(step, "temp")?));

            // Scoped params of this node's textures ('stateSize' -> 'stateSize_node_1').
            let mut scoped_param_map: ScopedParamMap = Vec::new();

            // An effect that declares global_xyz in its textures CREATES the particle
            // textures and starts a new particle pipeline scope.
            let textures = member(effect_def, "textures")?;
            let creates_particle_textures =
                textures.is_truthy() && member(textures, "global_xyz")?.is_truthy();
            if creates_particle_textures {
                current_particle_pipeline_id = Some(node_id.clone());
                lanes.xyz = Value::Null;
                lanes.vel = Value::Null;
                lanes.rgba = Value::Null;
            }
            let particle_pipeline_id = current_particle_pipeline_id.as_deref();

            // Compile-time defines: globals declared with `define: 'MACRO_NAME'`.
            let globals = member(effect_def, "globals")?;
            let step_args = member(step, "args")?;
            let mut compile_time_defines = Object::new();
            if globals.is_truthy() {
                let mut sorted_global_names = js_keys(globals);
                sorted_global_names.sort_by(|a, b| js_default_compare(a, b));
                for global_name in &sorted_global_names {
                    let def = globals.get(global_name);
                    if !def.is_truthy() || !member(def, "define")?.is_truthy() {
                        continue;
                    }
                    let mut value = member(def, "default")?.clone();
                    if step_args.is_truthy() && has_own(step_args, global_name) {
                        let arg_val = step_args.get(global_name);
                        value = if is_object_type(arg_val) && has_property(arg_val, "value") {
                            arg_val.get("value").clone()
                        } else {
                            arg_val.clone()
                        };
                    }
                    value = resolve_member(&std_enums, def, value)?;
                    if !value.is_nullish() {
                        compile_time_defines.insert(js_string(member(def, "define")?), value);
                    }
                }
            }
            // The deterministic program cache-key suffix of the defines.
            let program_define_suffix: String = compile_time_defines
                .iter()
                .map(|(k, v)| format!("__{k}_{}", js_string(v)))
                .collect();

            // Collect programs - per-step shader overrides first (keyed by step.temp).
            let step_overrides =
                shader_overrides.get_or_undefined(&js_string(member(step, "temp")?));
            let shaders_source = or(step_overrides, member(effect_def, "shaders")?);
            if shaders_source.is_truthy() {
                for (prog_name, shaders) in js_entries(shaders_source) {
                    // nodeId prefix for ALL programs, define suffix per variant.
                    let unique_prog_name = format!("{node_id}_{prog_name}{program_define_suffix}");
                    if !programs
                        .get(&unique_prog_name)
                        .is_some_and(Value::is_truthy)
                    {
                        // Per-program layouts (uniformLayouts) take precedence over uniformLayout.
                        let program_layout = or(
                            opt_member(member(effect_def, "uniformLayouts")?, &prog_name),
                            member(effect_def, "uniformLayout")?,
                        )
                        .clone();
                        let mut spec = spread(&shaders);
                        spec.insert("uniformLayout", program_layout);
                        spec.insert("defines", Value::Object(compile_time_defines.clone()));
                        programs.insert(unique_prog_name, Value::Object(spec));
                    }
                }
            }

            // Collect texture specs. global_ textures are shared within a chain
            // (particle textures within a particle pipeline); others are node-local.
            if textures.is_truthy() {
                for (tex_name, spec) in js_entries(textures) {
                    let is_particle_tex = is_particle_global(&tex_name);
                    let should_scope_as_particle =
                        is_particle_tex && particle_pipeline_id.is_some();
                    let should_scope_as_chain = tex_name.starts_with("global_") && !is_particle_tex;
                    let virtual_tex_id = if tex_name.starts_with("global_") {
                        if should_scope_as_particle {
                            format!("{tex_name}_{}", particle_pipeline_id.unwrap_or_default())
                        } else {
                            // Chain-scope all non-particle global textures.
                            format!("{tex_name}_{chain_scope_id}")
                        }
                    } else {
                        // Node-local texture - add node prefix.
                        format!("{node_id}_{tex_name}")
                    };

                    // Scoped textures scope the params their width/height reference.
                    let spec_width = member(&spec, "width")?;
                    let spec_height = member(&spec, "height")?;
                    let has_param_ref =
                        dim_references_param(spec_width) || dim_references_param(spec_height);
                    let mut resolved_spec = spread(&spec);
                    let should_scope_params = should_scope_as_particle
                        || should_scope_as_chain
                        || (particle_pipeline_id.is_some() && !tex_name.starts_with("global_"))
                        || has_param_ref;
                    if should_scope_params {
                        let scope_suffix = if should_scope_as_particle {
                            particle_pipeline_id.unwrap_or_default()
                        } else {
                            chain_scope_id.as_str()
                        };
                        let scope = DimScope {
                            tex_name: &tex_name,
                            particle_pipeline_id,
                            scope_suffix,
                            volume_size_param: &volume_size_param,
                        };
                        let width = scope_dim_spec(spec_width, &scope, &mut scoped_param_map)?;
                        resolved_spec.insert("width", width);
                        let height = scope_dim_spec(spec_height, &scope, &mut scoped_param_map)?;
                        resolved_spec.insert("height", height);
                    }
                    texture_specs.insert(virtual_tex_id, Value::Object(resolved_spec));
                }
            }

            // Collect 3D texture specs (same naming convention).
            let textures3d = member(effect_def, "textures3d")?;
            if textures3d.is_truthy() {
                for (tex_name, spec) in js_entries(textures3d) {
                    let virtual_tex_id = if tex_name.starts_with("global_") {
                        scope_chain_tex(&tex_name, particle_pipeline_id, &chain_scope_id)
                    } else {
                        format!("{node_id}_{tex_name}")
                    };
                    let mut resolved = spread(&spec);
                    resolved.insert("is3D", Value::Bool(true));
                    texture_specs.insert(virtual_tex_id, Value::Object(resolved));
                }
            }

            // Resolve inputs: step.from refers to a previous temp output (null: generator).
            let from = member(step, "from")?;
            if !from.is_null() {
                lanes.input =
                    texture_map_get(&texture_map, &format!("node_{}_out", js_string(from)));
            }

            // Process globals BEFORE the passes so downstream effects can use
            // uniforms like volumeSize set by upstream 3D generators; only set
            // defaults that are not already set from upstream.
            if globals.is_truthy() {
                for (global_name, def) in js_entries(globals) {
                    let uniform = member(&def, "uniform")?;
                    if uniform.is_truthy() && !member(&def, "default")?.is_undefined() {
                        let key = js_string(uniform);
                        // Skip if already set from upstream (preserve pipeline inheritance).
                        if !pipeline_uniforms.get_or_undefined(&key).is_undefined() {
                            continue;
                        }
                        let val =
                            resolve_member(&std_enums, &def, member(&def, "default")?.clone())?;
                        pipeline_uniforms.insert(key, val);
                    }

                    // Surface globals with colorModeUniform: colorMode from the default
                    // when the surface param is not given.
                    let color_mode_uniform = member(&def, "colorModeUniform")?;
                    if is_str(member(&def, "type")?, "surface")
                        && color_mode_uniform.is_truthy()
                        && (!step_args.is_truthy() || !has_own(step_args, &global_name))
                    {
                        // 'none' means colorMode=0, anything else colorMode=1.
                        let is_none = is_str(member(&def, "default")?, "none");
                        pipeline_uniforms.insert(
                            js_string(color_mode_uniform),
                            Value::Number(if is_none { 0.0 } else { 1.0 }),
                        );
                    }
                }
            }

            // Uniforms controlled by a surface's colorModeUniform.
            let mut color_mode_controlled_uniforms: Vec<Value> = Vec::new();

            // FIRST PASS: surface args set their colorModeUniform.
            if step_args.is_truthy() {
                for (arg_name, arg) in js_entries(step_args) {
                    if is_texture_arg(&arg) {
                        let global_def = opt_member(globals, &arg_name);
                        let color_mode_uniform = opt_member(global_def, "colorModeUniform");
                        if color_mode_uniform.is_truthy() {
                            // colorMode: 0 if the surface is 'none', 1 otherwise.
                            let is_none = is_str(arg.get("name"), "none");
                            pipeline_uniforms.insert(
                                js_string(color_mode_uniform),
                                Value::Number(if is_none { 0.0 } else { 1.0 }),
                            );
                            set_add(
                                &mut color_mode_controlled_uniforms,
                                color_mode_uniform.clone(),
                            );
                        }
                    }
                }
            }

            // SECOND PASS: non-surface args.
            if step_args.is_truthy() {
                for (arg_name, arg) in js_entries(step_args) {
                    let is_object_arg = is_object_type(&arg);
                    if is_texture_arg(&arg) {
                        continue;
                    }
                    let uniform_name = arg_uniform_name(globals, &arg_name)?;
                    if set_has(&color_mode_controlled_uniforms, &uniform_name) {
                        continue;
                    }
                    // A 3D input from upstream dictates volumeSize.
                    if is_str(&uniform_name, "volumeSize")
                        && lanes.input3d.is_truthy()
                        && !pipeline_uniforms
                            .get_or_undefined("volumeSize")
                            .is_undefined()
                    {
                        continue;
                    }
                    let resolved_value = if is_object_arg && !arg.get("value").is_undefined() {
                        arg.get("value").clone()
                    } else {
                        arg.clone()
                    };
                    pipeline_uniforms.insert(js_string(&uniform_name), resolved_value);
                }
            }

            // Expand passes.
            let passes_value = member(effect_def, "passes")?;
            let empty = Value::Array(Vec::new());
            let effect_passes_value = or(passes_value, &empty);
            let effect_passes = iterate(effect_passes_value, "effectPasses")?;
            let mut conditional_uniforms: Vec<Value> = Vec::new();
            for pass_def in effect_passes.iter() {
                let conditions = opt_member(member(pass_def, "conditions")?, "runIf");
                let skip = opt_member(member(pass_def, "conditions")?, "skipIf");
                let run_if = iterate(or(conditions, &empty), "((intermediate value) || [])")?;
                let skip_if = iterate(or(skip, &empty), "((intermediate value) || [])")?;
                for condition in run_if.iter().chain(skip_if.iter()) {
                    set_add(
                        &mut conditional_uniforms,
                        member(condition, "uniform")?.clone(),
                    );
                }
            }

            let ctx = StepContext {
                plan,
                step,
                is_last_step: step_position + 1 == chain_len,
                effect_name,
                effect_def,
                node_id: &node_id,
                chain_scope_id: &chain_scope_id,
                particle_pipeline_id,
                std_enums: &std_enums,
            };
            // `for (let i = 0; i < effectPasses.length; i++)` indexes the passes
            // (a string indexes UTF-16 code units, unlike the `for...of` above).
            let indexed_passes = match effect_passes_value {
                Value::String(s) => Cow::Owned(code_unit_strings(s)),
                _ => effect_passes.clone(),
            };
            for (i, pass_def) in indexed_passes.iter().enumerate() {
                let pass = expand_pass(
                    &ctx,
                    pass_def,
                    i,
                    indexed_passes.len(),
                    &program_define_suffix,
                    &compile_time_defines,
                    &conditional_uniforms,
                    &scoped_param_map,
                    &lanes,
                    &mut pipeline_uniforms,
                    &mut programs,
                    &mut texture_map,
                    &mut media_steps,
                    &mut media_step_ids,
                    &mut last_written_surface,
                )?;
                passes.push(pass);
            }

            // Update currentInput for the next step in the chain.
            lanes.input = texture_map_get(&texture_map, &format!("{node_id}_out"));

            // An explicit outputTex lets an effect pass through the 2D chain.
            let output_tex = member(effect_def, "outputTex")?;
            if output_tex.is_truthy() && !lanes.input.is_truthy() {
                if is_str(output_tex, "inputTex") {
                    // Restore the previous node's output.
                    if !from.is_null() {
                        let prev_output =
                            texture_map_get(&texture_map, &format!("node_{}_out", js_string(from)));
                        if prev_output.is_truthy() {
                            texture_map.insert(format!("{node_id}_out"), prev_output.clone());
                            lanes.input = prev_output;
                        }
                    }
                } else {
                    // Map an internal texture to the 2D pipeline.
                    let virtual_tex_id = if starts_with(output_tex, "internalTexName", "global_")? {
                        scope_chain_tex(
                            &js_string(output_tex),
                            particle_pipeline_id,
                            &chain_scope_id,
                        )
                    } else {
                        format!("{node_id}_{}", js_string(output_tex))
                    };
                    texture_map.insert(
                        format!("{node_id}_out"),
                        Value::String(virtual_tex_id.clone()),
                    );
                    lanes.input = Value::String(virtual_tex_id);
                }
            }

            // Update the pipeline lanes this node produced.
            let out3d = texture_map_get(&texture_map, &format!("{node_id}_out3d"));
            if out3d.is_truthy() {
                lanes.input3d = out3d.clone();
            }
            let out_xyz = texture_map_get(&texture_map, &format!("{node_id}_outXyz"));
            if out_xyz.is_truthy() {
                lanes.xyz = out_xyz.clone();
            }
            let out_vel = texture_map_get(&texture_map, &format!("{node_id}_outVel"));
            if out_vel.is_truthy() {
                lanes.vel = out_vel.clone();
            }
            let out_rgba = texture_map_get(&texture_map, &format!("{node_id}_outRgba"));
            if out_rgba.is_truthy() {
                lanes.rgba = out_rgba.clone();
            }

            // An explicit outputTex3d exposes an internal texture as the 3D output.
            let output_tex3d = member(effect_def, "outputTex3d")?;
            if output_tex3d.is_truthy() && !out3d.is_truthy() {
                if is_str(output_tex3d, "inputTex3d") {
                    // Passes through the input 3D texture (possibly modified in place).
                    if lanes.input3d.is_truthy() {
                        texture_map.insert(format!("{node_id}_out3d"), lanes.input3d.clone());
                    }
                } else {
                    let virtual_tex_id = if starts_with(output_tex3d, "internalTexName", "global_")?
                    {
                        scope_chain_tex(
                            &js_string(output_tex3d),
                            particle_pipeline_id,
                            &chain_scope_id,
                        )
                    } else {
                        format!("{node_id}_{}", js_string(output_tex3d))
                    };
                    texture_map.insert(
                        format!("{node_id}_out3d"),
                        Value::String(virtual_tex_id.clone()),
                    );
                    lanes.input3d = Value::String(virtual_tex_id);
                }
            }

            // An explicit outputGeo exposes a geometry buffer (normals + depth).
            let output_geo = member(effect_def, "outputGeo")?;
            if output_geo.is_truthy() {
                if is_str(output_geo, "inputGeo") {
                    if lanes.geo.is_truthy() {
                        texture_map.insert(format!("{node_id}_outGeo"), lanes.geo.clone());
                    }
                } else {
                    let virtual_geo_id = format!("{node_id}_{}", js_string(output_geo));
                    texture_map.insert(
                        format!("{node_id}_outGeo"),
                        Value::String(virtual_geo_id.clone()),
                    );
                    lanes.geo = Value::String(virtual_geo_id);
                }
            }

            // Explicit agent state outputs.
            for (key, passthrough, out_suffix, produced, lane) in [
                ("outputXyz", "inputXyz", "outXyz", &out_xyz, &mut lanes.xyz),
                ("outputVel", "inputVel", "outVel", &out_vel, &mut lanes.vel),
                (
                    "outputRgba",
                    "inputRgba",
                    "outRgba",
                    &out_rgba,
                    &mut lanes.rgba,
                ),
            ] {
                let tex_name = member(effect_def, key)?;
                if !tex_name.is_truthy() || produced.is_truthy() {
                    continue;
                }
                if is_str(tex_name, passthrough) {
                    if lane.is_truthy() {
                        texture_map.insert(format!("{node_id}_{out_suffix}"), lane.clone());
                    }
                } else {
                    let virtual_id = if starts_with(tex_name, "texName", "global_")? {
                        scope_chain_tex(&js_string(tex_name), particle_pipeline_id, &chain_scope_id)
                    } else {
                        format!("{node_id}_{}", js_string(tex_name))
                    };
                    texture_map.insert(
                        format!("{node_id}_{out_suffix}"),
                        Value::String(virtual_id.clone()),
                    );
                    *lane = Value::String(virtual_id);
                }
            }
        }

        // The final output of the chain (.write(o0)).
        let write = member(plan, "write")?;
        if write.is_truthy() && lanes.input.is_truthy() {
            let out_name = write_target_name(write)?;
            last_written_surface = out_name.clone();

            // Skip the final blit if the last step was an inline write to the same surface.
            let already_written = last_inline_write_target
                .as_ref()
                .is_some_and(|(kind, name)| {
                    is_str(kind, "output") && strict_equals(name, &out_name)
                });
            if already_written {
                continue;
            }

            let target_surface = format!("global_{}", js_string(&out_name));
            // Only blit if the current input is not already the target surface.
            if !is_str(&lanes.input, &target_surface) {
                let mut blit = Object::new();
                blit.insert(
                    "id",
                    Value::String(format!("final_blit_{}", js_string(&out_name))),
                );
                blit.insert("program", Value::from("blit"));
                blit.insert("type", Value::from("render"));
                let mut inputs = Object::new();
                inputs.insert("src", lanes.input.clone());
                blit.insert("inputs", Value::Object(inputs));
                let mut outputs = Object::new();
                outputs.insert("color", Value::String(target_surface));
                blit.insert("outputs", Value::Object(outputs));
                blit.insert("uniforms", Value::object());
                passes.push(Value::Object(blit));
            }
        }
    }

    // Follow volume handoffs after expansion so ordering and re-export do not
    // change atlas dimensions. Cycles without a producer retain their defaults.
    let mut resolved_volumes: IndexMap<String, WrittenVolume> = IndexMap::new();
    for param in read_volumes.keys() {
        let mut visited = Vec::new();
        if let Some(source) = resolve_volume(param, &read_volumes, &written_volumes, &mut visited) {
            resolved_volumes.insert(param.clone(), source);
        }
    }
    let export_ids: Vec<String> = exported_textures.keys().cloned().collect();
    for id in &export_ids {
        let mut visited = Vec::new();
        resolve_export(
            &Value::String(id.clone()),
            &exported_textures,
            &mut texture_specs,
            &mut visited,
        );
    }
    for (_, spec) in texture_specs.iter_mut() {
        for axis in ["width", "height", "depth"] {
            let param = opt_member(spec.get(axis), "param");
            if let Some(source) = get_by_value(&resolved_volumes, param) {
                let mut dim = spread(spec.get(axis));
                dim.insert("param", Value::String(source.param.clone()));
                spec.set(axis, Value::Object(dim));
            }
        }
    }
    for pass in passes.iter_mut() {
        for (param, source) in resolved_volumes.iter() {
            let Some(Value::Object(uniforms)) = pass.get_mut("uniforms") else {
                continue;
            };
            if !uniforms.contains_key(param) {
                continue;
            }
            uniforms.remove(param);
            uniforms.insert(source.param.clone(), source.value.clone());
            uniforms.insert("volumeSize", source.value.clone());
            if let Some(Value::Object(scoped)) = pass.get_mut("scopedParams")
                && scoped.get("volumeSize").is_some_and(|v| is_str(v, param))
            {
                scoped.insert("volumeSize", Value::String(source.param.clone()));
            }
        }
    }

    // The render surface: an explicit render() directive, else the last surface
    // written; an error when neither exists.
    let render = member(compilation_result, "render")?;
    let render_surface = if render.is_truthy() {
        render.clone()
    } else if last_written_surface.is_truthy() {
        last_written_surface
    } else {
        let mut error = Object::new();
        error.insert(
            "message",
            Value::from(
                "No render surface specified and no write() found - add render(oN) or write(oN)",
            ),
        );
        errors.push(Value::Object(error));
        Value::Null
    };

    Ok(Expansion {
        passes,
        errors,
        programs,
        texture_specs,
        render_surface,
        media_steps,
    })
}

/// `value === true` (`step.args?._skip === true`).
fn is_true(value: &Value) -> bool {
    matches!(value, Value::Bool(true))
}

/// `typeof plan.write === 'object' ? plan.write.name : plan.write`.
fn write_target_name(write: &Value) -> Result<Value, JsError> {
    if is_object_type(write) || write.is_null() {
        Ok(member(write, "name")?.clone())
    } else {
        Ok(write.clone())
    }
}

/// The surface a chain writes to for `feedback`/`selfTex` inputs and for the
/// last pass of the last step: `${prefix}_${outName}` with prefix `feedback` for
/// feedback targets, `global` otherwise.
fn write_target_surface(write: &Value) -> Result<(Value, String), JsError> {
    let out_name = write_target_name(write)?;
    let out_kind = or_str(member(write, "kind")?, "output");
    let prefix = if is_str(&out_kind, "feedback") {
        "feedback"
    } else {
        "global"
    };
    let surface = format!("{prefix}_{}", js_string(&out_name));
    Ok((out_name, surface))
}

/// The uniform an arg sets: `effectDef.globals[argName].uniform` when the global
/// declares one, else the arg name.
fn arg_uniform_name(globals: &Value, arg_name: &str) -> Result<Value, JsError> {
    if globals.is_truthy() {
        let global = globals.get(arg_name);
        if global.is_truthy() {
            let uniform = member(global, "uniform")?;
            if uniform.is_truthy() {
                return Ok(uniform.clone());
            }
        }
    }
    Ok(Value::String(arg_name.to_owned()))
}

/// One pass of an effect: the body of `expand`'s `for (let i = 0; i <
/// effectPasses.length; i++)` loop.
#[allow(clippy::too_many_arguments)]
fn expand_pass(
    ctx: &StepContext<'_>,
    pass_def: &Value,
    i: usize,
    pass_count: usize,
    program_define_suffix: &str,
    compile_time_defines: &Object,
    conditional_uniforms: &[Value],
    scoped_param_map: &ScopedParamMap,
    lanes: &Lanes,
    pipeline_uniforms: &mut Object,
    programs: &mut Object,
    texture_map: &mut TextureMap,
    media_steps: &mut Vec<Value>,
    media_step_ids: &mut Vec<String>,
    last_written_surface: &mut Value,
) -> Result<Value, JsError> {
    let node_id = ctx.node_id;
    let effect_def = ctx.effect_def;
    let step = ctx.step;
    let step_args = member(step, "args")?;
    let globals = member(effect_def, "globals")?;
    let pass_id = format!("{node_id}_pass_{i}");

    // nodeId-prefixed program name with the compile-time-define suffix.
    let mut program_name = format!(
        "{node_id}_{}{program_define_suffix}",
        js_string(member(pass_def, "program")?)
    );
    let pass_defines = member(pass_def, "defines")?;
    if pass_defines.is_truthy() {
        // Conditional passes select precompiled variants.
        let base_program = programs.get(&program_name).cloned();
        let mut define_entries = js_entries(pass_defines);
        define_entries.sort_by(|(a, _), (b, _)| locale_compare(a, b));
        let pass_define_suffix: String = define_entries
            .iter()
            .map(|(key, value)| format!("__{key}_{}", js_string(value)))
            .collect();
        program_name.push_str(&pass_define_suffix);
        if let Some(base_program) = base_program.filter(Value::is_truthy)
            && !programs.get(&program_name).is_some_and(Value::is_truthy)
        {
            let mut variant = spread(&base_program);
            let mut defines = compile_time_defines.clone();
            defines.assign(&spread(pass_defines));
            variant.insert("defines", Value::Object(defines));
            programs.insert(program_name.clone(), Value::Object(variant));
        }
    }

    let mut pass = Object::new();
    pass.insert("id", Value::String(pass_id));
    pass.insert("program", Value::String(program_name));
    for key in [
        "entryPoint",
        "drawMode",
        "drawBuffers",
        "count",
        "countUniform",
        "repeat",
        "blend",
        "conditions",
        "workgroups",
        "storageBuffers",
        "storageTextures",
        "name",
        "type",
        "clear",
        "viewport",
        "samplerTypes",
    ] {
        pass.insert(key, member(pass_def, key)?.clone());
    }
    pass.insert("inputs", Value::object());
    pass.insert("outputs", Value::object());
    pass.insert("uniforms", Value::object());

    // Metadata mapping passes back to their effect definitions.
    pass.insert("effectKey", ctx.effect_name.clone());
    pass.insert(
        "effectFunc",
        or(member(effect_def, "func")?, ctx.effect_name).clone(),
    );
    let namespace = member(effect_def, "namespace")?;
    pass.insert(
        "effectNamespace",
        if namespace.is_truthy() {
            namespace.clone()
        } else {
            Value::Null
        },
    );
    pass.insert("nodeId", Value::String(node_id.to_owned()));
    pass.insert("stepIndex", member(step, "temp")?.clone());

    // Consumer passes that inherit volumeSize from upstream.
    if lanes.input3d.is_truthy()
        && !pipeline_uniforms
            .get_or_undefined("volumeSize")
            .is_undefined()
    {
        pass.insert("inheritsVolumeSize", Value::Bool(true));
    }

    // Start with the pipeline uniforms inherited from upstream effects.
    let mut uniforms = pipeline_uniforms.clone();

    // Defaults only where not already set from upstream.
    if globals.is_truthy() {
        for def in js_values(globals) {
            let uniform = member(&def, "uniform")?;
            if uniform.is_truthy() && !member(&def, "default")?.is_undefined() {
                let key = js_string(uniform);
                if !uniforms.get_or_undefined(&key).is_undefined() {
                    continue;
                }
                let val = resolve_member(ctx.std_enums, &def, member(&def, "default")?.clone())?;
                uniforms.insert(key.clone(), val.clone());
                pipeline_uniforms.insert(key, val);
            }
        }
    }

    // uniformSpecs: consumer ranges for percentage-based automation scaling.
    if globals.is_truthy() {
        let mut uniform_specs = Object::new();
        for (arg_name, def) in js_entries(globals) {
            let uniform_name =
                or(member(&def, "uniform")?, &Value::String(arg_name.clone())).clone();
            let def_type = member(&def, "type")?;
            let choices = member(&def, "choices")?;
            if (is_str(def_type, "float") || is_str(def_type, "int")) && !choices.is_truthy() {
                let mut range = Object::new();
                let min = member(&def, "min")?;
                let max = member(&def, "max")?;
                range.insert(
                    "min",
                    if min.is_nullish() {
                        Value::Number(0.0)
                    } else {
                        min.clone()
                    },
                );
                range.insert(
                    "max",
                    if max.is_nullish() {
                        Value::Number(100.0)
                    } else {
                        max.clone()
                    },
                );
                uniform_specs.insert(js_string(&uniform_name), Value::Object(range));
            } else if is_str(def_type, "int")
                && choices.is_truthy()
                && set_has(conditional_uniforms, &uniform_name)
            {
                // A conditional selector uses the same integer in every shader
                // pass and in CPU-side pass selection.
                let mut spec = Object::new();
                spec.insert("type", Value::from("int"));
                let min = member(&def, "min")?;
                let max = member(&def, "max")?;
                if is_finite_number(min) && is_finite_number(max) {
                    spec.insert("min", min.clone());
                    spec.insert("max", max.clone());
                }
                uniform_specs.insert(js_string(&uniform_name), Value::Object(spec));
            }
        }
        pass.insert("uniformSpecs", Value::Object(uniform_specs));
    }

    // Map uniforms from the step args.
    if step_args.is_truthy() {
        for (arg_name, arg) in js_entries(step_args) {
            let is_object_arg = is_object_type(&arg);
            // Texture arguments are handled as inputs.
            if is_texture_arg(&arg) {
                continue;
            }
            let uniform_name = arg_uniform_name(globals, &arg_name)?;
            // Skip colorMode uniforms controlled by a surface's colorModeUniform.
            if globals.is_truthy() {
                let mut is_controlled = false;
                for global_def in js_values(globals) {
                    if strict_equals(member(&global_def, "colorModeUniform")?, &uniform_name) {
                        is_controlled = true;
                        break;
                    }
                }
                if is_controlled {
                    continue;
                }
            }
            // A 3D input from upstream dictates volumeSize.
            if is_str(&uniform_name, "volumeSize")
                && lanes.input3d.is_truthy()
                && !pipeline_uniforms
                    .get_or_undefined("volumeSize")
                    .is_undefined()
            {
                continue;
            }
            let resolved_value = if is_object_arg && !arg.get("value").is_undefined() {
                arg.get("value").clone()
            } else {
                arg.clone()
            };
            let key = js_string(&uniform_name);
            uniforms.insert(key.clone(), resolved_value.clone());
            pipeline_uniforms.insert(key, resolved_value);
        }
    }

    // Pass-level uniforms: `uniforms: { uniformName: "globalParamName" | number }`.
    let pass_uniforms = member(pass_def, "uniforms")?;
    if pass_uniforms.is_truthy() {
        let mut uniform_aliases: Option<Object> = None;
        for (uniform_name, global_ref) in js_entries(pass_uniforms) {
            // Constants specialize draws that share a program.
            if let Value::Number(_) = global_ref {
                uniforms.insert(uniform_name, global_ref);
                continue;
            }
            // A renamed mapping lets runtime parameter updates reach this uniform.
            if !is_str(&global_ref, &uniform_name) {
                uniform_aliases
                    .get_or_insert_with(Object::new)
                    .insert(uniform_name.clone(), global_ref.clone());
            }
            let by_uniform = pipeline_uniforms.get_or_undefined(&uniform_name);
            let by_ref = pipeline_uniforms.get_or_undefined(&js_string(&global_ref));
            if !by_uniform.is_undefined() {
                uniforms.insert(uniform_name, by_uniform.clone());
            } else if !by_ref.is_undefined() {
                // Stored under the param name (e.g. "mix") not the shader var ("mixAmt").
                uniforms.insert(uniform_name, by_ref.clone());
            } else if globals.is_truthy() && globals.get(&js_string(&global_ref)).is_truthy() {
                // The global param's default.
                let global_def = globals.get(&js_string(&global_ref));
                let default = member(global_def, "default")?;
                if !default.is_undefined() {
                    let val = resolve_member(ctx.std_enums, global_def, default.clone())?;
                    uniforms.insert(uniform_name, val);
                }
            }
        }
        if let Some(aliases) = uniform_aliases {
            pass.insert("uniformAliases", Value::Object(aliases));
        }
    }

    // Expand a classicNoisedeck palette index into the dependent
    // paletteOffset/Amp/Freq/Phase/Mode uniforms so the first frame renders the
    // selected palette.
    if globals.is_truthy() {
        for (arg_name, global_def) in js_entries(globals) {
            if !is_str(member(&global_def, "type")?, "palette") {
                continue;
            }
            let uniform_name = js_string(or(
                member(&global_def, "uniform")?,
                &Value::String(arg_name),
            ));
            let Value::Number(index) = uniforms.get_or_undefined(&uniform_name) else {
                continue;
            };
            let Some(expanded) = expand_palette(*index)? else {
                continue;
            };
            for (u_name, u_value) in expanded.iter() {
                if uniforms.contains_key(u_name) {
                    uniforms.insert(u_name.clone(), u_value.clone());
                    pipeline_uniforms.insert(u_name.clone(), u_value.clone());
                }
            }
        }
    }

    // Map inputs.
    let mut inputs = Object::new();
    let pass_inputs = member(pass_def, "inputs")?;
    if pass_inputs.is_truthy() {
        for (uniform_name, tex_ref) in js_entries(pass_inputs) {
            if let Some(binding) = map_input(
                ctx,
                &tex_ref,
                &uniform_name,
                lanes,
                texture_map,
                media_steps,
                media_step_ids,
            )? {
                inputs.insert(uniform_name, binding);
            }
        }
    }

    // Map outputs.
    let mut outputs = Object::new();
    let pass_outputs = member(pass_def, "outputs")?;
    if pass_outputs.is_truthy() {
        for (attachment, tex_ref) in js_entries(pass_outputs) {
            let virtual_tex = if is_str(&tex_ref, "outputTex") {
                // The main 2D output: the last pass of the last step writes the
                // chain's target surface directly.
                let is_last_pass = i + 1 == pass_count;
                let write = member(ctx.plan, "write")?;
                let virtual_tex = if ctx.is_last_step && is_last_pass && write.is_truthy() {
                    let (out_name, surface) = write_target_surface(write)?;
                    // The last written surface determines the render surface.
                    *last_written_surface = out_name;
                    surface
                } else {
                    format!("{node_id}_out")
                };
                let tex = Value::String(virtual_tex.clone());
                texture_map.insert(virtual_tex.clone(), tex.clone());
                texture_map.insert(format!("{node_id}_out"), tex.clone());
                tex
            } else if is_str(&tex_ref, "outputTex3d") {
                let tex = Value::String(format!("{node_id}_out3d"));
                texture_map.insert(format!("{node_id}_out3d"), tex.clone());
                tex
            } else if is_str(&tex_ref, "outputXyz") {
                let tex = Value::String(format!("{node_id}_outXyz"));
                texture_map.insert(format!("{node_id}_outXyz"), tex.clone());
                tex
            } else if is_str(&tex_ref, "outputVel") {
                let tex = Value::String(format!("{node_id}_outVel"));
                texture_map.insert(format!("{node_id}_outVel"), tex.clone());
                tex
            } else if is_str(&tex_ref, "outputRgba") {
                let tex = Value::String(format!("{node_id}_outRgba"));
                texture_map.insert(format!("{node_id}_outRgba"), tex.clone());
                tex
            } else if is_str(&tex_ref, "inputTex3d") {
                // Write back to the 3D texture we received.
                or_str(&lanes.input3d, &format!("{node_id}_inputTex3d"))
            } else if is_str(&tex_ref, "inputGeo") {
                or_str(&lanes.geo, &format!("{node_id}_inputGeo"))
            } else if is_str(&tex_ref, "inputXyz") {
                or_str(&lanes.xyz, &format!("{node_id}_inputXyz"))
            } else if is_str(&tex_ref, "inputVel") {
                or_str(&lanes.vel, &format!("{node_id}_inputVel"))
            } else if is_str(&tex_ref, "inputRgba") {
                or_str(&lanes.rgba, &format!("{node_id}_inputRgba"))
            } else if starts_with(&tex_ref, "texRef", "global_")? {
                Value::String(scope_chain_tex(
                    &js_string(&tex_ref),
                    ctx.particle_pipeline_id,
                    ctx.chain_scope_id,
                ))
            } else if starts_with(&tex_ref, "texRef", "feedback_")? {
                tex_ref.clone()
            } else {
                // Node-local texture - add node prefix.
                Value::String(format!("{node_id}_{}", js_string(&tex_ref)))
            };
            outputs.insert(attachment, virtual_tex);
        }
    }

    // Propagate scoped param uniforms for texture sizing: the scoped name
    // carries the value so resolveDimension() can find it.
    for (original_param, scoped_param) in scoped_param_map {
        let value = uniforms
            .get_or_undefined(&js_string(original_param))
            .clone();
        if !value.is_undefined() {
            uniforms.insert(scoped_param.clone(), value.clone());
            pipeline_uniforms.insert(scoped_param.clone(), value);
        }
    }

    pass.insert("inputs", Value::Object(inputs));
    pass.insert("outputs", Value::Object(outputs));
    pass.insert("uniforms", Value::Object(uniforms));

    // Scoped param mappings for runtime updates (applyStepParameterValues).
    if !scoped_param_map.is_empty() {
        let scoped: Object = scoped_param_map
            .iter()
            .map(|(k, v)| (js_string(k), Value::String(v.clone())))
            .collect();
        pass.insert("scopedParams", Value::Object(scoped));
    }

    Ok(Value::Object(pass))
}

/// One entry of the "Map Inputs" loop: the binding of the pass input
/// `uniformName` to the texture `texRef`, or `None` when the input stays unbound.
fn map_input(
    ctx: &StepContext<'_>,
    tex_ref: &Value,
    uniform_name: &str,
    lanes: &Lanes,
    texture_map: &TextureMap,
    media_steps: &mut Vec<Value>,
    media_step_ids: &mut Vec<String>,
) -> Result<Option<Value>, JsError> {
    let node_id = ctx.node_id;
    let effect_def = ctx.effect_def;
    let step = ctx.step;
    let step_args = member(step, "args")?;
    let globals = member(effect_def, "globals")?;

    // Standard and legacy pipeline inputs (2D).
    let is_pipeline_input = is_str(tex_ref, "inputTex")
        || (starts_with(tex_ref, "texRef", "o")? && {
            let Value::String(s) = tex_ref else {
                unreachable!()
            };
            !parse_int(&s[1..], 0).is_nan()
        });

    if is_pipeline_input {
        return Ok(Some(or(&lanes.input, tex_ref).clone()));
    }
    if is_str(tex_ref, "inputTex3d") {
        // 3D pipeline input from the previous node.
        return Ok(Some(or(&lanes.input3d, tex_ref).clone()));
    }
    if is_str(tex_ref, "inputGeo") {
        // Geometry buffer pipeline input.
        return Ok(Some(or(&lanes.geo, tex_ref).clone()));
    }
    if is_str(tex_ref, "inputXyz") {
        return Ok(Some(or(&lanes.xyz, tex_ref).clone()));
    }
    if is_str(tex_ref, "inputVel") {
        return Ok(Some(or(&lanes.vel, tex_ref).clone()));
    }
    if is_str(tex_ref, "inputRgba") {
        return Ok(Some(or(&lanes.rgba, tex_ref).clone()));
    }
    if is_str(tex_ref, "noise") {
        return Ok(Some(Value::from("global_noise")));
    }
    if is_str(tex_ref, "midiNoteGrid") {
        return Ok(Some(Value::from("midiNoteGrid")));
    }
    if is_str(tex_ref, "feedback") || is_str(tex_ref, "selfTex") {
        // Read from the surface this chain writes to.
        let write = member(ctx.plan, "write")?;
        if write.is_truthy() {
            let (_, surface) = write_target_surface(write)?;
            return Ok(Some(Value::String(surface)));
        }
        // No explicit write target.
        return Ok(Some(or_str(&lanes.input, "global_inputTex")));
    }
    let external_texture = member(effect_def, "externalTexture")?;
    if external_texture.is_truthy() && strict_equals(tex_ref, external_texture) {
        // External texture input (camera/video): a per-step texture id.
        let tex_id = format!(
            "{}_step_{}",
            js_string(tex_ref),
            js_string(member(step, "temp")?)
        );
        // Record the binding so hosts can enumerate media texture ids.
        if !media_step_ids.contains(&tex_id) {
            media_step_ids.push(tex_id.clone());
            let mut media_step = Object::new();
            media_step.insert("textureId", Value::String(tex_id.clone()));
            media_step.insert("uniform", Value::String(uniform_name.to_owned()));
            media_step.insert("stepIndex", member(step, "temp")?.clone());
            media_step.insert("effect", ctx.effect_name.clone());
            media_steps.push(Value::Object(media_step));
        }
        return Ok(Some(Value::String(tex_id)));
    }
    let tex_ref_key = js_string(tex_ref);
    if step_args.is_truthy() && has_own(step_args, &tex_ref_key) {
        // A reference to an argument (e.g. blend(tex: ...)).
        let arg = step_args.get(&tex_ref_key);
        // Null/undefined arguments are intentionally unbound inputs.
        if arg.is_nullish() {
            return Ok(None);
        }
        let kind = arg.get("kind");
        let name = arg.get("name");
        if is_str(kind, "temp") {
            let key = format!("node_{}_out", js_string(arg.get("index")));
            return Ok(Some(texture_map_get(texture_map, &key)));
        }
        if is_str(kind, "pipeline") && (is_str(name, "inputTex") || is_str(name, "inputColor")) {
            return Ok(Some(or(&lanes.input, name).clone()));
        }
        if ["output", "source", "vol", "geo", "xyz", "vel", "rgba"]
            .iter()
            .any(|k| is_str(kind, k))
        {
            return Ok(Some(if is_str(name, "none") {
                Value::from("none")
            } else {
                Value::String(format!("global_{}", js_string(name)))
            }));
        }
        if let Value::String(s) = arg {
            // "none" binds to the blank/default texture.
            return Ok(Some(Value::String(resolve_global_surface_ref(s))));
        }
        return Ok(None);
    }
    if globals.is_truthy() {
        let global = globals.get(&tex_ref_key);
        if global.is_truthy() && !member(global, "default")?.is_undefined() {
            // A parameter with a default value.
            let default_val = member(global, "default")?;
            if is_str(default_val, "none") {
                return Ok(Some(Value::from("none")));
            }
            if is_str(default_val, "inputTex") || is_str(default_val, "inputColor") {
                return Ok(Some(or(&lanes.input, default_val).clone()));
            }
            if is_surface_ref(&js_string(default_val)) {
                return Ok(Some(Value::String(format!(
                    "global_{}",
                    js_string(default_val)
                ))));
            }
            if starts_with(default_val, "defaultVal", "global_")? {
                return Ok(Some(Value::String(scope_chain_tex(
                    &js_string(default_val),
                    ctx.particle_pipeline_id,
                    ctx.chain_scope_id,
                ))));
            }
            return Ok(Some(default_val.clone()));
        }
    }
    if starts_with(tex_ref, "texRef", "global_")? {
        // Explicit global reference - scope to chain/particle.
        return Ok(Some(Value::String(scope_chain_tex(
            &tex_ref_key,
            ctx.particle_pipeline_id,
            ctx.chain_scope_id,
        ))));
    }
    if is_str(tex_ref, "outputTex") {
        // This node's main output (e.g. in feedback passes).
        return Ok(Some(Value::String(format!("{node_id}_out"))));
    }
    // Internal texture or explicit reference - node-prefix it.
    Ok(Some(Value::String(format!("{node_id}_{tex_ref_key}"))))
}

/// `resolveVolume(param, visited)`: follow a reader's volume to the chain that
/// produced it.
fn resolve_volume(
    param: &str,
    read_volumes: &IndexMap<String, ReadVolume>,
    written_volumes: &IndexMap<String, WrittenVolume>,
    visited: &mut Vec<String>,
) -> Option<WrittenVolume> {
    if visited.iter().any(|v| v == param) {
        return None;
    }
    visited.push(param.to_owned());
    let read = read_volumes.get(param);
    let writer = read
        .and_then(|r| r.writer.clone())
        .or_else(|| read.and_then(|r| get_by_value(written_volumes, &r.surface).cloned()));
    let writer = writer?;
    if writer.param == param {
        return None;
    }
    resolve_volume(&writer.param, read_volumes, written_volumes, visited).or(Some(writer))
}

/// `resolveExport(id, visited)`: an exported volume/geometry surface takes the
/// spec of the texture it was exported from.
fn resolve_export(
    id: &Value,
    exported_textures: &IndexMap<String, Value>,
    texture_specs: &mut Object,
    visited: &mut Vec<Value>,
) -> Value {
    let key = js_string(id);
    if set_has(visited, id) {
        return texture_specs.get_or_undefined(&key).clone();
    }
    visited.push(id.clone());
    let source = get_by_value(exported_textures, id)
        .cloned()
        .unwrap_or(Value::Undefined);
    if !source.is_truthy() || strict_equals(&source, id) {
        return texture_specs.get_or_undefined(&key).clone();
    }
    let spec = resolve_export(&source, exported_textures, texture_specs, visited);
    if spec.is_truthy() {
        texture_specs.insert(key.clone(), Value::Object(spread(&spec)));
    }
    texture_specs.get_or_undefined(&key).clone()
}

#[cfg(test)]
mod tests;
