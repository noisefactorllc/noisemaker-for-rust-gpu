//! Port of `lang/unparser.js`: regenerate DSL source from a compiled program.
//!
//! [`unparse`] serializes the validator's output (`{plans, vars, render,
//! searchNamespaces, ...}`) back to DSL text, applying per-step parameter
//! overrides; [`apply_parameter_updates`] is the compile → override → unparse
//! convenience; [`format_value`], [`unparse_call`] and [`unparse_chain`] are
//! the reference's other exports, and [`format_let_expr`] (module-private in the
//! reference) is public here as well.
//!
//! The reference functions return arbitrary JavaScript values in places (a
//! `palette`-typed parameter formats to the raw value, a `geometry` parameter
//! without a default to `undefined`), and its callers compare results with `===`
//! and stringify them with template literals or `Array.prototype.join`. The port
//! keeps those semantics: formatting helpers return a [`Value`], and every
//! stringification goes through [`jsv`]'s `ToString`/`join` rules. Errors the
//! reference throws on malformed input are returned as [`JsError`]s with V8's
//! messages.

pub mod jsv;

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::JsError;
use crate::registry::Registry;
use crate::value::{Object, Value};

use crate::js::{is_js_whitespace, math_round};
use jsv::{
    DepthGuard, cannot_read, entries, get, get_opt, get_v, is_finite_number, is_object_like,
    iterate, join, keys, member, not_a_function, number_to_hex, object_member, pad_start,
    quote_json_string, quote_json_utf16, repeat, same_value_zero, set_plain, spread_into,
    strict_equals, to_number, to_property_key, to_string, utf16, values,
};

/// Oscillator type number → `oscKind` member (`oscKindNames`).
const OSC_KIND_NAMES: &[&str] = &[
    "sine", "tri", "saw", "sawInv", "square", "noise1d", "noise2d",
];

/// MIDI mode number → `midiMode` member (`midiModeNames`).
const MIDI_MODE_NAMES: &[&str] = &[
    "noteChange",
    "gateNote",
    "gateVelocity",
    "triggerNote",
    "velocity",
    "cc",
    "cc14",
    "nrpn",
    "pitchBend",
    "pressure",
    "polyPressure",
];

/// Audio band number → `audioBand` member (`audioBandNames`).
const AUDIO_BAND_NAMES: &[&str] = &["low", "mid", "high", "vol", "raw"];

/// `options.customFormatter(value, spec)`: a `null`/`undefined` result falls
/// through to the built-in formatting.
pub type CustomFormatter<'a> = Rc<dyn Fn(&Value, &Value) -> Result<Value, JsError> + 'a>;

/// `options.getEffectDef(op, namespace)`: the effect definition of a step (a
/// falsy result means none).
pub type EffectDefLookup<'a> = Rc<dyn Fn(&Value, &Value) -> Result<Value, JsError> + 'a>;

/// `options.formatTemp(index)`: the DSL of the nested chain that produces the
/// temporary surface `index`.
pub type FormatTemp<'a> = Rc<dyn Fn(&Value) -> Result<Value, JsError> + 'a>;

/// The options object of `unparse`, `unparseCall` and `formatValue`.
///
/// Value-typed members hold the option as the reference reads it (`undefined`
/// when absent); they are reference-counted so the `{ ...options, key }` copies
/// the reference makes per call stay cheap.
#[derive(Clone, Default)]
pub struct UnparseOptions<'a> {
    /// `customFormatter`.
    pub custom_formatter: Option<CustomFormatter<'a>>,
    /// `enums`: the enum tree `formatValue` resolves `spec.enum` paths in.
    /// `unparse` substitutes the standard enums when this is falsy.
    pub enums: Rc<Value>,
    /// `getEffectDef`.
    pub get_effect_def: Option<EffectDefLookup<'a>>,
    /// `omitSearchDirective` (truthy: no `search` line).
    pub omit_search_directive: Rc<Value>,
    /// `multilineKwargs` (anything but `false` enables multi-line calls).
    pub multiline_kwargs: Rc<Value>,
    /// `indent` (the chain indent of a call; non-finite reads as 0).
    pub indent: Rc<Value>,
    /// `specs`: parameter specs of the call being formatted (effect globals).
    pub specs: Rc<Value>,
    /// `schemaSpecs`: op-schema specs that only feed default suppression.
    pub schema_specs: Rc<Value>,
    /// `formatTemp`.
    pub format_temp: Option<FormatTemp<'a>>,
}

impl std::fmt::Debug for UnparseOptions<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnparseOptions")
            .field("custom_formatter", &self.custom_formatter.is_some())
            .field("enums", &self.enums.is_truthy())
            .field("get_effect_def", &self.get_effect_def.is_some())
            .field("omit_search_directive", &self.omit_search_directive)
            .field("multiline_kwargs", &self.multiline_kwargs)
            .field("indent", &self.indent)
            .field("format_temp", &self.format_temp.is_some())
            .finish()
    }
}

/// `oscKindNames` / `midiModeNames` / `audioBandNames` as a JavaScript array, so
/// that indexing with an arbitrary value behaves like `names[value]`.
fn names_array(names: &[&str]) -> Value {
    Value::Array(names.iter().map(|n| Value::from(*n)).collect())
}

/// `names[key] || fallback`.
fn name_or(names: &[&str], key: &Value, fallback: &str) -> Result<Value, JsError> {
    let v = get_v(&names_array(names), key)?;
    Ok(if v.is_truthy() { v } else { fallback.into() })
}

/// `a || b`.
fn or(a: Value, b: impl FnOnce() -> Value) -> Value {
    if a.is_truthy() { a } else { b() }
}

/// `x > 0` for a JavaScript value (`ToNumber` comparison; `NaN` is never greater).
fn greater_than_zero(x: &Value) -> Result<bool, JsError> {
    Ok(to_number(x)? > 0.0)
}

/// `s.<method>` where `s` must be a string: the string, or the `TypeError` the
/// reference throws when `s` is not one.
fn str_method<'v>(s: &'v Value, expr: &str, method: &str) -> Result<&'v str, JsError> {
    match s {
        Value::String(text) => Ok(text),
        Value::Undefined | Value::Null => Err(cannot_read(s, method)),
        _ => Err(not_a_function(&format!("{expr}.{method}"))),
    }
}

/// `path.join(sep)` where `path` must be an array.
fn array_join(path: &Value, sep: &str, expr: &str) -> Result<String, JsError> {
    match path {
        Value::Array(items) => join(items, sep),
        Value::Undefined | Value::Null => Err(cannot_read(path, "join")),
        _ => Err(not_a_function(&format!("{expr}.join"))),
    }
}

/// The value an `Array.prototype.join` pushes for an element (`null` and
/// `undefined` become empty strings); `lines.push(comment)` relies on it.
fn join_elem(v: &Value) -> Result<String, JsError> {
    if v.is_nullish() {
        Ok(String::new())
    } else {
        to_string(v)
    }
}

/// `x.length > 0` for an optional collection.
fn has_length(x: &Value) -> Result<bool, JsError> {
    greater_than_zero(&member(x, "length"))
}

/// `formatEnumName(name)`: drop an `Enum` suffix.
fn format_enum_name(name: &str) -> String {
    name.strip_suffix("Enum").unwrap_or(name).to_owned()
}

/// `formatLosslessNumber(value)`: a finite number without rounding and without
/// exponent notation (which the DSL's numeric literal grammar lacks).
pub fn format_lossless_number(value: f64) -> String {
    let text = crate::js::number_to_string(value);
    if !text.contains(['e', 'E']) {
        return text;
    }
    let lower = text.to_lowercase();
    let (coefficient, exponent_text) = lower.split_once('e').expect("exponent notation");
    let exponent = crate::js::string_to_number(exponent_text);
    let negative = coefficient.starts_with('-');
    let unsigned = if negative {
        &coefficient[1..]
    } else {
        coefficient
    };
    let (integer_part, fraction_part) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{integer_part}{fraction_part}");
    let decimal_index = integer_part.len() as f64 + exponent;
    let sign = if negative { "-" } else { "" };
    if decimal_index <= 0.0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-decimal_index) as usize));
    }
    let decimal_index = decimal_index as usize;
    if decimal_index >= digits.len() {
        return format!("{sign}{digits}{}", "0".repeat(decimal_index - digits.len()));
    }
    format!(
        "{sign}{}.{}",
        &digits[..decimal_index],
        &digits[decimal_index..]
    )
}

/// `/^[a-zA-Z_][a-zA-Z0-9_]*$/`.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// `/^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$/`.
fn is_enum_path(s: &str) -> bool {
    let segments: Vec<&str> = s.split('.').collect();
    segments.len() > 1 && segments.iter().all(|seg| is_identifier(seg))
}

/// `(n) => Math.max(0, Math.min(255, Math.round(n * 255))).toString(16).padStart(2, '0')`.
fn to_hex_clamped(n: &Value) -> Result<String, JsError> {
    let rounded = math_round(to_number(n)? * 255.0);
    // Math.min/Math.max propagate NaN.
    let clamped = if rounded.is_nan() {
        f64::NAN
    } else {
        rounded.clamp(0.0, 255.0)
    };
    Ok(pad_start(&number_to_hex(clamped), 2, '0'))
}

/// `(n) => Math.round(n * 255).toString(16).padStart(2, '0')` (the unclamped
/// `vec4` hex form).
fn to_hex_unclamped(n: &Value) -> Result<String, JsError> {
    let rounded = math_round(to_number(n)? * 255.0);
    Ok(pad_start(&number_to_hex(rounded), 2, '0'))
}

fn hex_color(items: &[Value], count: usize, clamped: bool) -> Result<Value, JsError> {
    let mut out = String::from("#");
    for i in 0..count {
        let n = items.get(i).cloned().unwrap_or(Value::Undefined);
        out.push_str(&if clamped {
            to_hex_clamped(&n)?
        } else {
            to_hex_unclamped(&n)?
        });
    }
    Ok(out.into())
}

/// `items.map(v => formatValue(v, null, options)).join(', ')`.
fn format_list(items: &[Value], options: &UnparseOptions<'_>) -> Result<String, JsError> {
    let mut formatted = Vec::with_capacity(items.len());
    for v in items {
        formatted.push(format_value(v, &Value::Null, options, &Value::Undefined)?);
    }
    join(&formatted, ", ")
}

/// `decodeJsonStringLiteralContent(raw)` (`lang/stringLiterals.js`) as UTF-16
/// code units: `JSON.parse('"' + raw + '"')`, or — when that throws — a decode of
/// the small escape set the lexer accepts in single-quoted strings that keeps
/// unknown escapes verbatim.
fn decode_json_string_literal_content(raw: &Value) -> Result<Vec<u16>, JsError> {
    if let Ok(text) = to_string(raw)
        && let Some(units) = jsv::json_parse_string_body(&utf16(&text))
    {
        return Ok(units);
    }
    fn escape(next: &[u16]) -> Option<&'static str> {
        let s = String::from_utf16(next).ok()?;
        Some(match s.as_str() {
            "'" => "'",
            "\"" => "\"",
            "\\" => "\\",
            "n" => "\n",
            "r" => "\r",
            "t" => "\t",
            "b" => "\u{8}",
            "f" => "\u{c}",
            "v" => "\u{b}",
            "0" => "\0",
            _ => return None,
        })
    }
    let backslash = b'\\' as u16;
    let mut decoded: Vec<u16> = Vec::new();
    match raw {
        Value::String(s) => {
            let units = utf16(s);
            let mut i = 0;
            while i < units.len() {
                if units[i] != backslash || i + 1 >= units.len() {
                    decoded.push(units[i]);
                    i += 1;
                    continue;
                }
                i += 1;
                let next = &units[i..i + 1];
                match escape(next) {
                    Some(e) => decoded.extend(e.encode_utf16()),
                    None => {
                        decoded.push(backslash);
                        decoded.extend_from_slice(next);
                    }
                }
                i += 1;
            }
        }
        _ => {
            // Array-likes iterate their elements (`raw[i]`, `raw.length`).
            let length = to_number(&member(raw, "length"))?;
            let mut i = 0.0;
            while i < length {
                let item = get_v(raw, &Value::Number(i))?;
                if !strict_equals(&item, &Value::from("\\")) || i + 1.0 >= length {
                    decoded.extend(to_string(&item)?.encode_utf16());
                    i += 1.0;
                    continue;
                }
                i += 1.0;
                let next = get_v(raw, &Value::Number(i))?;
                let key = to_property_key(&next)?;
                match escape(&utf16(&key)) {
                    Some(e) if key.encode_utf16().count() == 1 => decoded.extend(e.encode_utf16()),
                    _ => {
                        decoded.push(backslash);
                        decoded.extend(to_string(&next)?.encode_utf16());
                    }
                }
                i += 1.0;
            }
        }
    }
    Ok(decoded)
}

/// `JSON.stringify(decodeJsonStringLiteralContent(raw))`.
fn stringify_decoded(raw: &Value) -> Result<String, JsError> {
    Ok(quote_json_utf16(&decode_json_string_literal_content(raw)?))
}

/// `node && node.type === 'Number' ? node.value : undefined`.
fn num_val(node: &Value) -> Value {
    if node.is_truthy() && strict_equals(&get_opt(node, "type"), &"Number".into()) {
        get_opt(node, "value")
    } else {
        Value::Undefined
    }
}

/// `x.path[x.path.length - 1]`.
fn last_path_segment(path: &Value) -> Result<Value, JsError> {
    let index = to_number(&get(path, "length")?)? - 1.0;
    get_v(path, &Value::Number(index))
}

fn is_type(v: &Value, t: &str) -> bool {
    strict_equals(&get_opt(v, "type"), &Value::from(t))
}

/// `formatOscillator(osc)`: an oscillator value as `osc(...)`.
fn format_oscillator(osc: &Value) -> Result<Value, JsError> {
    let ast = get_opt(osc, "_ast");
    if is_type(&ast, "Oscillator") {
        return format_let_expr(&ast, &UnparseOptions::default());
    }
    let type_name = name_or(OSC_KIND_NAMES, &get(osc, "oscType")?, "sine")?;
    let mut parts = vec![format!("type: oscKind.{}", to_string(&type_name)?)];

    // Only include non-default values
    let min = get(osc, "min")?;
    if !strict_equals(&min, &Value::Number(0.0)) {
        parts.push(format!("min: {}", to_string(&min)?));
    }
    let max = get(osc, "max")?;
    if !strict_equals(&max, &Value::Number(1.0)) {
        parts.push(format!("max: {}", to_string(&max)?));
    }
    let speed = get(osc, "speed")?;
    if !strict_equals(&speed, &Value::Number(1.0)) {
        parts.push(format!("speed: {}", to_string(&speed)?));
    }
    let offset = get(osc, "offset")?;
    if !strict_equals(&offset, &Value::Number(0.0)) {
        parts.push(format!("offset: {}", to_string(&offset)?));
    }
    let seed = get(osc, "seed")?;
    // Only include seed for noise type
    if !strict_equals(&seed, &Value::Number(1.0))
        && strict_equals(&get(osc, "oscType")?, &Value::Number(5.0))
    {
        parts.push(format!("seed: {}", to_string(&seed)?));
    }
    Ok(format!("osc({})", parts.join(", ")).into())
}

/// `formatMidi(midi)`: a MIDI value as `midi(...)`.
fn format_midi(midi: &Value) -> Result<Value, JsError> {
    let channel = get(midi, "channel")?;
    let zone = get(midi, "zone")?;
    if (channel.is_truthy() && is_object_like(&channel))
        || (zone.is_truthy() && is_object_like(&zone))
    {
        return format_let_expr(midi, &UnparseOptions::default());
    }
    let ast = get_opt(midi, "_ast");
    if is_type(&ast, "Midi") {
        return format_let_expr(&ast, &UnparseOptions::default());
    }
    let mut parts = if !zone.is_undefined() {
        let lower = strict_equals(&zone, &Value::Number(0.0));
        vec![format!(
            "zone: midiZone.{}",
            if lower { "lower" } else { "upper" }
        )]
    } else {
        vec![format!("channel: {}", to_string(&channel)?)]
    };
    let members = get(midi, "members")?;
    if !members.is_undefined() {
        parts.push(format!("members: {}", to_string(&members)?));
    }

    // Only include non-default values
    let mode_name = name_or(MIDI_MODE_NAMES, &get(midi, "mode")?, "velocity")?;
    if !strict_equals(&mode_name, &"velocity".into()) {
        parts.push(format!("mode: midiMode.{}", to_string(&mode_name)?));
    }
    let cc = get(midi, "cc")?;
    if !cc.is_undefined() {
        parts.push(format!("cc: {}", to_string(&cc)?));
    }
    let nrpn = get(midi, "nrpn")?;
    if !nrpn.is_undefined() {
        parts.push(format!("nrpn: {}", to_string(&nrpn)?));
    }
    let min = get(midi, "min")?;
    if !strict_equals(&min, &Value::Number(0.0)) {
        parts.push(format!("min: {}", to_string(&min)?));
    }
    let max = get(midi, "max")?;
    if !strict_equals(&max, &Value::Number(1.0)) {
        parts.push(format!("max: {}", to_string(&max)?));
    }
    let sensitivity = get(midi, "sensitivity")?;
    if !strict_equals(&sensitivity, &Value::Number(1.0)) {
        parts.push(format!("sensitivity: {}", to_string(&sensitivity)?));
    }
    if let Value::String(name) = get(midi, "name")?
        && !name.is_empty()
    {
        parts.push(format!("name: {}", quote_json_string(&name)));
    }
    if let Value::String(id) = get(midi, "id")?
        && !id.is_empty()
    {
        parts.push(format!("id: {}", quote_json_string(&id)));
    }
    Ok(format!("midi({})", parts.join(", ")).into())
}

/// `formatAudio(audio)`: an audio value as `audio(...)`.
fn format_audio(audio: &Value) -> Result<Value, JsError> {
    let band = get(audio, "band")?;
    if band.is_truthy() && is_object_like(&band) {
        return format_let_expr(audio, &UnparseOptions::default());
    }
    let ast = get_opt(audio, "_ast");
    if is_type(&ast, "Audio") {
        return format_let_expr(&ast, &UnparseOptions::default());
    }
    let band_name = name_or(AUDIO_BAND_NAMES, &band, "low")?;
    let mut parts = vec![format!("band: audioBand.{}", to_string(&band_name)?)];

    // Only include non-default values
    let min = get(audio, "min")?;
    if !strict_equals(&min, &Value::Number(0.0)) {
        parts.push(format!("min: {}", to_string(&min)?));
    }
    let max = get(audio, "max")?;
    if !strict_equals(&max, &Value::Number(1.0)) {
        parts.push(format!("max: {}", to_string(&max)?));
    }
    let channel = get(audio, "channel")?;
    if channel.is_integer() && channel.as_f64().is_some_and(|c| c >= 1.0) {
        parts.push(format!("channel: {}", to_string(&channel)?));
    }
    if let Value::String(name) = get(audio, "name")?
        && !name.is_empty()
    {
        parts.push(format!("name: {}", quote_json_string(&name)));
    }
    if let Value::String(id) = get(audio, "id")?
        && !id.is_empty()
    {
        parts.push(format!("id: {}", quote_json_string(&id)));
    }
    Ok(format!("audio({})", parts.join(", ")).into())
}

/// The `enums` walk of `formatValue`: `node = node && node[part] ? node[part] : null`
/// over every segment of the enum path. Own members are walked by reference;
/// inherited and primitive members (rare) continue on owned copies.
fn walk_enum_path<'v>(enums: &'v Value, parts: &[&str]) -> std::borrow::Cow<'v, Value> {
    use std::borrow::Cow;
    let mut node = enums;
    for (i, part) in parts.iter().enumerate() {
        if !node.is_truthy() {
            return Cow::Owned(Value::Null);
        }
        let own = match node {
            Value::Object(o) => o.get(part),
            Value::Array(a) if crate::value::is_array_index(part) => {
                part.parse::<usize>().ok().and_then(|i| a.get(i))
            }
            _ => None,
        };
        match own {
            Some(next) => {
                if !next.is_truthy() {
                    return Cow::Owned(Value::Null);
                }
                node = next;
            }
            None => {
                let mut owned = member(node, part);
                if !owned.is_truthy() {
                    return Cow::Owned(Value::Null);
                }
                for part in &parts[i + 1..] {
                    let next = member(&owned, part);
                    if !next.is_truthy() {
                        return Cow::Owned(Value::Null);
                    }
                    owned = next;
                }
                return Cow::Owned(owned);
            }
        }
    }
    Cow::Borrowed(node)
}

/// `formatValue(value, spec, options, sourceForm)`: one parameter value as DSL.
///
/// `spec` is the parameter spec (`null`/`undefined` when unknown); `source_form`
/// is the validator's `argSources` tag (`'array'` round-trips `[…]` literals).
pub fn format_value(
    value: &Value,
    spec: &Value,
    options: &UnparseOptions<'_>,
    source_form: &Value,
) -> Result<Value, JsError> {
    let _depth = DepthGuard::enter()?;
    // Temporary surfaces represent nested effect chains, not named surfaces.
    if is_kind_temp(value)
        && let Some(format_temp) = &options.format_temp
    {
        return format_temp(&get_opt(value, "index"));
    }
    let empty_enums = Value::Object(Object::new());
    let enums: &Value = if options.enums.is_undefined() {
        &empty_enums
    } else {
        &options.enums
    };

    // Try custom formatter first if provided
    if let Some(custom_formatter) = &options.custom_formatter {
        let custom = custom_formatter(value, spec)?;
        if !custom.is_nullish() {
            return Ok(custom);
        }
    }

    if value.is_nullish() {
        return Ok("null".into());
    }

    let spec_type = get_opt(spec, "type");
    let is_lossless_vector = strict_equals(&spec_type, &"vec4".into())
        && strict_equals(&get_opt(&get_opt(spec, "ui"), "format"), &"vector".into())
        && matches!(value, Value::Array(a) if a.len() == 4 && a.iter().all(is_finite_number));
    if is_lossless_vector {
        let Value::Array(items) = value else {
            unreachable!()
        };
        let parts: Vec<String> = items
            .iter()
            .map(|v| format_lossless_number(v.as_f64().unwrap_or(f64::NAN)))
            .collect();
        return Ok(format!("[{}]", parts.join(", ")).into());
    }

    // Round-trip array literal source form: when the validator tagged an arg as
    // having come from a `[…]` source, emit it back as `[…]`.
    if strict_equals(source_form, &"array".into())
        && let Value::Array(items) = value
    {
        return Ok(format!("[{}]", format_list(items, options)?).into());
    }

    // Handle variable reference marker - output just the variable name
    if is_object_like(value) {
        let var_ref = get_opt(value, "_varRef");
        if var_ref.is_truthy() {
            return Ok(var_ref);
        }
    }

    let ast = get_opt(value, "_ast");
    if is_type(&ast, "Ident") {
        return get(&ast, "name");
    }

    if let Value::Bool(b) = value {
        return Ok(if *b { "true" } else { "false" }.into());
    }

    // Handle inline choices - look up enum name from numeric value
    let choices = get_opt(spec, "choices");
    if choices.is_truthy() && matches!(value, Value::Number(_)) {
        for (name, val) in entries(&choices) {
            if name.ends_with(':') {
                continue; // skip group labels
            }
            if strict_equals(&val, value) {
                return Ok(format_enum_name(&name).into());
            }
        }
    }

    // Handle global enum reference (e.g., spec.enum = "palette")
    let enum_spec = get_opt(spec, "enum");
    if enum_spec.is_truthy() && matches!(value, Value::Number(_)) {
        let enum_path = str_method(&enum_spec, "enumPath", "split")?;
        let parts: Vec<&str> = enum_path.split('.').collect();
        let node = walk_enum_path(enums, &parts);
        if is_object_like(&node) {
            for (name, val) in entries(&node) {
                let num_val = if is_object_like(&val) && jsv::has_property(&val, "value") {
                    member(&val, "value")
                } else {
                    val
                };
                if strict_equals(&num_val, value) {
                    return Ok(name.into());
                }
            }
        }
    }

    // Handle enum string values — strip namespace prefix (e.g., "oscType.sine" → "sine")
    if enum_spec.is_truthy()
        && let Value::String(s) = value
    {
        let prefix = format!("{}.", to_string(&enum_spec)?);
        if let Some(rest) = s.strip_prefix(&prefix) {
            return Ok(rest.into());
        }
    }

    // Handle surface, volume and geometry types
    if let Some(formatted) = format_reference_type(value, spec, &spec_type)? {
        return Ok(formatted);
    }

    // Handle member type (enum path already formatted) and palette type
    if strict_equals(&spec_type, &"member".into()) || strict_equals(&spec_type, &"palette".into()) {
        return Ok(value.clone());
    }

    // Handle automation objects (Oscillator, Midi, Audio)
    if is_object_like(value) {
        if is_type(value, "Oscillator")
            || strict_equals(&get_opt(value, "oscillator"), &true.into())
        {
            return format_oscillator(value);
        }
        if is_type(value, "Midi") {
            return format_midi(value);
        }
        if is_type(value, "Audio") {
            return format_audio(value);
        }
    }

    if let Value::Number(n) = value {
        // Emit the shortest representation that reparses to the same value.
        return Ok(format_lossless_number(*n).into());
    }

    if let Value::String(s) = value {
        // Colors (hex strings like #ffffff) must NOT be quoted even though they're strings
        if s.starts_with('#') {
            return Ok(value.clone());
        }
        let needs_quoting =
            strict_equals(&spec_type, &"string".into()) || (!is_identifier(s) && !is_enum_path(s));
        if needs_quoting {
            // Use triple-quotes for multi-line strings
            if s.contains('\n') {
                return Ok(format!("\"\"\"{s}\"\"\"").into());
            }
            let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
            return Ok(format!("\"{escaped}\"").into());
        }
        return Ok(value.clone());
    }

    // Handle both regular arrays and typed arrays
    if let Value::Array(arr) = value {
        return format_array(arr, &spec_type, options);
    }

    if let Value::Object(_) = value {
        return format_object(value, spec, options, &ast);
    }

    Ok(to_string(value)?.into())
}

/// `formatValue`'s surface, volume and geometry branches (`None` for other
/// parameter types).
#[inline(never)]
fn format_reference_type(
    value: &Value,
    spec: &Value,
    spec_type: &Value,
) -> Result<Option<Value>, JsError> {
    // Handle surface type
    if strict_equals(spec_type, &"surface".into()) {
        // Handle object surface references (e.g., {kind: 'output', name: 'o1'})
        if is_object_like(value) {
            let name = get_opt(value, "name");
            if name.is_truthy() {
                if strict_equals(&name, &"none".into()) {
                    return Ok(Some("none".into()));
                }
                return Ok(Some(format!("read({})", to_string(&name)?).into()));
            }
        }
        return Ok(Some(match value {
            Value::String(s) if !s.is_empty() => {
                if s == "none" {
                    "none".into()
                } else if s.contains('(') {
                    value.clone()
                } else {
                    format!("read({s})").into()
                }
            }
            _ => {
                let default_surface = or(get_opt(spec, "default"), || "inputTex".into());
                if strict_equals(&default_surface, &"none".into()) {
                    "none".into()
                } else {
                    format!("read({})", to_string(&default_surface)?).into()
                }
            }
        }));
    }
    // Handle volume and geometry types: a name, the value, or the default
    let volume = strict_equals(spec_type, &"volume".into());
    if volume || strict_equals(spec_type, &"geometry".into()) {
        if is_object_like(value) {
            let name = get_opt(value, "name");
            if name.is_truthy() {
                return Ok(Some(name));
            }
        }
        return Ok(Some(match value {
            Value::String(s) if !s.is_empty() => value.clone(),
            _ if volume => or(get_opt(spec, "default"), || "vol0".into()),
            _ => get_opt(spec, "default"),
        }));
    }
    Ok(None)
}

/// `formatValue`'s array branch: `vec2(...)`/`vec3(...)` or a hex color.
#[inline(never)]
fn format_array(
    arr: &[Value],
    spec_type: &Value,
    options: &UnparseOptions<'_>,
) -> Result<Value, JsError> {
    let all_numbers = arr.iter().all(|v| matches!(v, Value::Number(_)));
    let is_type_named = |t: &str| strict_equals(spec_type, &Value::from(t));
    // Handle vec2 explicitly if spec says so
    if is_type_named("vec2") && arr.len() == 2 && all_numbers {
        return Ok(format!("vec2({})", format_list(arr, options)?).into());
    }
    // Check if this is a color param (should format as hex regardless of vec3/vec4 type)
    let is_color_control = is_type_named("color");
    // Handle vec3 explicitly if spec says so
    if is_type_named("vec3") && arr.len() == 3 && all_numbers {
        if is_color_control {
            return hex_color(arr, 3, true);
        }
        return Ok(format!("vec3({})", format_list(arr, options)?).into());
    }
    // Handle vec4 explicitly if spec says so - format as hex color
    if is_type_named("vec4") && arr.len() == 4 && all_numbers {
        return hex_color(arr, 4, false);
    }
    // Infer type from array length for numeric arrays
    if all_numbers {
        if arr.len() == 2 {
            return Ok(format!("vec2({})", format_list(arr, options)?).into());
        }
        if arr.len() == 3 {
            if is_color_control {
                return hex_color(arr, 3, true);
            }
            return Ok(format!("vec3({})", format_list(arr, options)?).into());
        }
        if arr.len() == 4 {
            return hex_color(arr, 4, true);
        }
    }
    // Fallback for other arrays - this should not happen in valid DSL
    if is_color_control && arr.len() >= 3 {
        return hex_color(arr, 3, true);
    }
    let head = &arr[..arr.len().min(3)];
    Ok(format!("vec3({})", format_list(head, options)?).into())
}

/// `value?.kind === 'temp'`.
fn is_kind_temp(value: &Value) -> bool {
    strict_equals(&get_opt(value, "kind"), &"temp".into())
}

/// The plain-object branches of `formatValue` (AST nodes, surface references and
/// the array-like safety net).
fn format_object(
    value: &Value,
    spec: &Value,
    options: &UnparseOptions<'_>,
    ast: &Value,
) -> Result<Value, JsError> {
    let ast_type_is = |t: &str| ast.is_truthy() && is_type(ast, t);
    // Handle String AST node
    if is_type(value, "String") {
        let s = get(value, "value")?;
        let s = str_method(&s, "value.value", "replace")?;
        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
        return Ok(format!("\"{escaped}\"").into());
    }
    // Handle oscillator configuration
    if is_type(value, "Oscillator") {
        return format_oscillator(value);
    }
    // Handle oscillator AST from _ast property
    if ast_type_is("Oscillator") {
        return format_oscillator(value);
    }
    // Handle MIDI configuration
    if is_type(value, "Midi")
        && (matches!(get_opt(value, "channel"), Value::Number(_))
            || matches!(get_opt(value, "zone"), Value::Number(_)))
    {
        return format_midi(value);
    }
    // Handle MIDI AST from _ast property
    if ast_type_is("Midi") {
        return format_midi(value);
    }
    // Handle Audio configuration
    if is_type(value, "Audio") && matches!(get_opt(value, "band"), Value::Number(_)) {
        return format_audio(value);
    }
    // Handle Audio AST from _ast property
    if ast_type_is("Audio") {
        return format_audio(value);
    }
    // Handle special AST node types: raw Oscillator, Midi and Audio AST nodes
    if is_type(value, "Oscillator") {
        return format_raw_oscillator_node(value);
    }
    if is_type(value, "Midi") {
        return format_raw_midi_node(value, options);
    }
    if is_type(value, "Audio") {
        return format_raw_audio_node(value);
    }
    // Handle Read node (pipeline built-in)
    if is_type(value, "Read") {
        let surface = get_opt(value, "surface");
        let surface_name = or(get_opt(&surface, "name"), || surface.clone());
        return Ok(format!("read({})", to_string(&surface_name)?).into());
    }
    // Handle Read3D node (pipeline built-in)
    if is_type(value, "Read3D") {
        let tex3d = get_opt(value, "tex3d");
        let tex3d_name = or(get_opt(&tex3d, "name"), || tex3d.clone());
        let geo = get_opt(value, "geo");
        if geo.is_truthy() {
            let geo_name = or(get_opt(&geo, "name"), || geo.clone());
            return Ok(format!(
                "read3d({}, {})",
                to_string(&tex3d_name)?,
                to_string(&geo_name)?
            )
            .into());
        }
        return Ok(format!("read3d({})", to_string(&tex3d_name)?).into());
    }
    for t in ["OutputRef", "SourceRef", "VolRef", "GeoRef"] {
        if is_type(value, t) {
            return Ok(get_opt(value, "name"));
        }
    }
    if is_type(value, "Member") {
        return Ok(array_join(&get_opt(value, "path"), ".", "value.path")?.into());
    }
    if is_type(value, "Number") {
        return format_value(&get_opt(value, "value"), spec, options, &Value::Undefined);
    }
    if is_type(value, "Boolean") {
        return Ok(if get_opt(value, "value").is_truthy() {
            "true"
        } else {
            "false"
        }
        .into());
    }
    // Surface reference - wrap in read() if spec indicates surface type
    let kind = get_opt(value, "kind");
    if ["output", "feedback", "source"]
        .iter()
        .any(|k| strict_equals(&kind, &Value::from(*k)))
    {
        if spec.is_truthy() && strict_equals(&get_opt(spec, "type"), &"surface".into()) {
            return Ok(format!("read({})", to_string(&get_opt(value, "name"))?).into());
        }
        return Ok(get_opt(value, "name"));
    }

    // SAFETY: Never let arrays become raw comma-separated strings.
    if let Some(formatted) = format_array_like(value, spec)? {
        return Ok(formatted);
    }

    Ok(to_string(value)?.into())
}

/// `formatValue`'s branch for a raw `Oscillator` AST node (sub-fields are AST
/// nodes). Kept out of line: debug frames hold every branch's temporaries.
#[inline(never)]
fn format_raw_oscillator_node(value: &Value) -> Result<Value, JsError> {
    let type_path = get_opt(value, "oscType");
    let mut type_name = Value::from("sine");
    if type_path.is_truthy()
        && is_type(&type_path, "Member")
        && get_opt(&type_path, "path").is_truthy()
    {
        type_name = last_path_segment(&get_opt(&type_path, "path"))?;
    } else if type_path.is_truthy() && is_type(&type_path, "Ident") {
        type_name = get_opt(&type_path, "name");
    }
    let mut parts = vec![format!("type: oscKind.{}", to_string(&type_name)?)];
    for (field, default) in [
        ("min", 0.0),
        ("max", 1.0),
        ("speed", 1.0),
        ("offset", 0.0),
        ("seed", 1.0),
    ] {
        let node = get_opt(value, field);
        if node.is_truthy()
            && is_type(&node, "Number")
            && !strict_equals(&get_opt(&node, "value"), &Value::Number(default))
        {
            parts.push(format!("{field}: {}", to_string(&get_opt(&node, "value"))?));
        }
    }
    Ok(format!("osc({})", parts.join(", ")).into())
}

/// `formatValue`'s branch for a raw `Midi` AST node.
#[inline(never)]
fn format_raw_midi_node(value: &Value, options: &UnparseOptions<'_>) -> Result<Value, JsError> {
    let mut parts = Vec::new();
    let channel = get_opt(value, "channel");
    if channel.is_truthy() && is_type(&channel, "Number") {
        parts.push(format!(
            "channel: {}",
            to_string(&get_opt(&channel, "value"))?
        ));
    }
    let mode_path = get_opt(value, "mode");
    let mut mode_name = Value::from("velocity");
    if mode_path.is_truthy()
        && is_type(&mode_path, "Member")
        && get_opt(&mode_path, "path").is_truthy()
    {
        mode_name = last_path_segment(&get_opt(&mode_path, "path"))?;
    } else if mode_path.is_truthy() && is_type(&mode_path, "Ident") {
        mode_name = get_opt(&mode_path, "name");
    }
    if !strict_equals(&mode_name, &"velocity".into()) {
        parts.push(format!("mode: midiMode.{}", to_string(&mode_name)?));
    }
    for field in ["cc", "nrpn", "zone", "members"] {
        let node = get_opt(value, field);
        if !node.is_undefined() {
            parts.push(format!(
                "{field}: {}",
                to_string(&format_let_expr(&node, options)?)?
            ));
        }
    }
    for (field, default) in [("min", 0.0), ("max", 1.0), ("sensitivity", 1.0)] {
        let node = get_opt(value, field);
        if node.is_truthy()
            && is_type(&node, "Number")
            && !strict_equals(&get_opt(&node, "value"), &Value::Number(default))
        {
            parts.push(format!("{field}: {}", to_string(&get_opt(&node, "value"))?));
        }
    }
    Ok(format!("midi({})", parts.join(", ")).into())
}

/// `formatValue`'s branch for a raw `Audio` AST node.
#[inline(never)]
fn format_raw_audio_node(value: &Value) -> Result<Value, JsError> {
    let band_path = get_opt(value, "band");
    let mut band_str = String::from("audioBand.low");
    if band_path.is_truthy()
        && is_type(&band_path, "Member")
        && get_opt(&band_path, "path").is_truthy()
    {
        band_str = format!(
            "audioBand.{}",
            to_string(&last_path_segment(&get_opt(&band_path, "path"))?)?
        );
    } else if band_path.is_truthy() && is_type(&band_path, "Ident") {
        band_str = to_string(&get_opt(&band_path, "name"))?;
    } else if band_path.is_truthy() && is_type(&band_path, "Number") {
        band_str = to_string(&get_opt(&band_path, "value"))?;
    }
    let mut parts = vec![format!("band: {band_str}")];
    for (field, default) in [("min", 0.0), ("max", 1.0)] {
        let node = get_opt(value, field);
        if node.is_truthy()
            && is_type(&node, "Number")
            && !strict_equals(&get_opt(&node, "value"), &Value::Number(default))
        {
            parts.push(format!("{field}: {}", to_string(&get_opt(&node, "value"))?));
        }
    }
    let channel = get_opt(value, "channel");
    if is_type(&channel, "Number") {
        parts.push(format!(
            "channel: {}",
            to_string(&get_opt(&channel, "value"))?
        ));
    }
    for field in ["name", "id"] {
        let node = get_opt(value, field);
        if is_type(&node, "String") {
            parts.push(format!(
                "{field}: {}",
                stringify_decoded(&get_opt(&node, "value"))?
            ));
        }
    }
    Ok(format!("audio({})", parts.join(", ")).into())
}

/// `formatValue`'s safety net for array-like objects (a numeric `length`):
/// `Array.from(value)` formatted as `vec2`/`vec3`/hex when it holds 2-4 numbers.
#[inline(never)]
fn format_array_like(value: &Value, spec: &Value) -> Result<Option<Value>, JsError> {
    let Value::Number(len) = get_opt(value, "length") else {
        return Ok(None);
    };
    // Array.from(arrayLike): ToLength(length) elements.
    let n = if len.is_nan() || len <= 0.0 {
        0.0
    } else {
        len.trunc().min(9_007_199_254_740_991.0)
    };
    if n > 4_294_967_295.0 {
        return Err(JsError::Error {
            name: "RangeError".into(),
            message: "Invalid array length".into(),
        });
    }
    if !(2.0..=4.0).contains(&n) {
        return Ok(None);
    }
    let arr: Vec<Value> = (0..n as usize)
        .map(|i| member(value, &i.to_string()))
        .collect();
    if !arr.iter().all(|v| matches!(v, Value::Number(_))) {
        return Ok(None);
    }
    let is_color_control = strict_equals(&get_opt(spec, "type"), &"color".into());
    Ok(Some(match arr.len() {
        2 => format!("vec2({})", join(&arr, ", ")?).into(),
        3 if is_color_control => hex_color(&arr, 3, true)?,
        3 => format!("vec3({})", join(&arr, ", ")?).into(),
        // 4 elements - hex color
        _ => hex_color(&arr, 4, true)?,
    }))
}

/// `unparseCall(call, options)`: one `{name, kwargs, args}` call as DSL.
///
/// Keyword arguments equal (after formatting) to their spec's default are
/// omitted; calls with more than two arguments are written one argument per
/// line unless `options.multilineKwargs === false`.
pub fn unparse_call(call: &Value, options: &UnparseOptions<'_>) -> Result<String, JsError> {
    let name = get(call, "name")?;
    let mut parts: Vec<Value> = Vec::new();
    let specs: Value = if options.specs.is_truthy() {
        (*options.specs).clone()
    } else {
        Value::Object(Object::new())
    };
    let multiline_kwargs = !strict_equals(&options.multiline_kwargs, &false.into());
    let base_indent = if is_finite_number(&options.indent) {
        options.indent.as_f64().unwrap_or(0.0)
    } else {
        0.0
    };
    let parent_indent = repeat(" ", f64::max(0.0, base_indent))?;
    let child_indent = repeat(" ", f64::max(0.0, base_indent + 2.0))?;

    // Handle kwargs (named arguments)
    let kwargs = get(call, "kwargs")?;
    let kwarg_count = if kwargs.is_truthy() {
        keys(&kwargs).len()
    } else {
        0
    };
    if kwarg_count > 0 {
        for (key, value) in entries(&kwargs) {
            // Skip _skip: false
            if key == "_skip" && strict_equals(&value, &false.into()) {
                continue;
            }
            // Get spec from options if available
            let spec = or(member(&specs, &key), || Value::Null);
            // Without an effect definition, fall back to the op schema the
            // validator used to fill defaults (default suppression only).
            let default_spec = if spec.is_truthy() {
                spec.clone()
            } else {
                or(get_opt(&options.schema_specs, &key), || Value::Null)
            };
            // Round-trip the source form for this key (argSources sidecar).
            let source_form = get_opt(&get(call, "argSources")?, &key);

            // Check against default value
            if default_spec.is_truthy() {
                let default = member(&default_spec, "default");
                if !default.is_undefined() {
                    let formatted_value = format_value(&value, &spec, options, &source_form)?;
                    let formatted_default =
                        format_value(&default, &spec, options, &Value::Undefined)?;
                    // For surface params, 'none' must always be explicit when set
                    // (so the expander binds the blank texture) — unless the default IS 'none'
                    let none = Value::from("none");
                    let is_explicit_none =
                        strict_equals(&get_opt(&spec, "type"), &"surface".into())
                            && strict_equals(&formatted_value, &none)
                            && !strict_equals(&formatted_default, &none);
                    if strict_equals(&formatted_value, &formatted_default) && !is_explicit_none {
                        continue;
                    }
                }
            }
            let formatted = format_value(&value, &spec, options, &source_form)?;
            parts.push(format!("{key}: {}", to_string(&formatted)?).into());
        }
    }

    // Handle positional args
    let args = get(call, "args")?;
    if args.is_truthy() && has_length(&args)? {
        for arg in iterate(&args, "call.args")? {
            parts.push(format_value(
                &arg,
                &Value::Null,
                options,
                &Value::Undefined,
            )?);
        }
    }

    // Use multiline formatting only if more than 2 kwargs; 1-2 params stay inline
    let has_kwargs = kwarg_count > 0 && !parts.is_empty();
    let name = to_string(&name)?;
    if multiline_kwargs && has_kwargs && parts.len() > 2 {
        let mut lines = Vec::with_capacity(parts.len());
        for p in &parts {
            lines.push(format!("{child_indent}{}", to_string(p)?));
        }
        return Ok(format!("{name}(\n{}\n{parent_indent})", lines.join(",\n")));
    }
    Ok(format!("{name}({})", join(&parts, ", ")?))
}

/// `unparseChain(chain, options)`: calls joined as a method chain.
pub fn unparse_chain(chain: &Value, options: &UnparseOptions<'_>) -> Result<String, JsError> {
    let items = match chain {
        Value::Array(items) => items,
        Value::Undefined | Value::Null => return Err(cannot_read(chain, "map")),
        _ => return Err(not_a_function("chain.map")),
    };
    let mut parts = Vec::with_capacity(items.len());
    for (idx, call) in items.iter().enumerate() {
        let mut call_options = options.clone();
        call_options.indent = Rc::new(Value::Number(if idx == 0 { 0.0 } else { 2.0 }));
        parts.push(unparse_call(call, &call_options)?);
    }
    // Join with line break after closing paren, 2-space indent on next line
    Ok(parts.join("\n  ."))
}

/// `formatLetExpr(expr, options)`: a `let` right-hand side (an AST expression
/// node) back to DSL source.
pub fn format_let_expr(expr: &Value, options: &UnparseOptions<'_>) -> Result<Value, JsError> {
    let _depth = DepthGuard::enter()?;
    if !expr.is_truthy() {
        return Ok("null".into());
    }
    let var_ref = get_opt(expr, "_varRef");
    if var_ref.is_truthy() {
        return Ok(var_ref);
    }
    let expr_type = get_opt(expr, "type");
    let Value::String(expr_type) = expr_type else {
        return format_value(expr, &Value::Null, options, &Value::Undefined);
    };
    match expr_type.as_str() {
        "Number" => Ok(to_string(&get_opt(expr, "value"))?.into()),
        "String" => Ok(match jsv::json_stringify(&get_opt(expr, "value")) {
            Some(s) => s.into(),
            None => Value::Undefined,
        }),
        "Boolean" => Ok(if get_opt(expr, "value").is_truthy() {
            "true"
        } else {
            "false"
        }
        .into()),
        "Ident" => Ok(get_opt(expr, "name")),
        "Member" => Ok(array_join(&get_opt(expr, "path"), ".", "expr.path")?.into()),
        "Oscillator" => format_let_oscillator(expr, options),
        "Midi" => format_let_midi(expr, options),
        "Audio" => format_let_audio(expr, options),
        "Call" => Ok(unparse_call(expr, options)?.into()),
        "Chain" => Ok(unparse_chain(&get_opt(expr, "chain"), options)?.into()),
        "Func" => Ok(format!(
            "() => {}",
            to_string(&format_let_expr(&get_opt(expr, "body"), options)?)?
        )
        .into()),
        _ => format_value(expr, &Value::Null, options, &Value::Undefined),
    }
}

/// `parts.push(\`${name}: ${formatLetExpr(node, options)}\`)` unless `node` is
/// absent or a `Number` node equal to `default` (`pushIfNonDefault`).
fn push_if_non_default(
    parts: &mut Vec<String>,
    name: &str,
    node: &Value,
    default: Option<f64>,
    options: &UnparseOptions<'_>,
) -> Result<(), JsError> {
    if !node.is_truthy() {
        return Ok(());
    }
    let v = num_val(node);
    if let Some(default) = default
        && strict_equals(&v, &Value::Number(default))
    {
        return Ok(());
    }
    parts.push(format!(
        "{name}: {}",
        to_string(&format_let_expr(node, options)?)?
    ));
    Ok(())
}

/// `formatLetExpr` for a raw `Oscillator` AST node (sub-fields are AST nodes).
fn format_let_oscillator(expr: &Value, options: &UnparseOptions<'_>) -> Result<Value, JsError> {
    let osc_type = get_opt(expr, "oscType");
    let mut type_str = Value::from("oscKind.sine");
    let type_value = get_opt(&osc_type, "value");
    if is_type(&osc_type, "Member") && get(&osc_type, "path")?.is_truthy() {
        let segment = last_path_segment(&get_opt(&osc_type, "path"))?;
        type_str = format!("oscKind.{}", to_string(&segment)?).into();
    } else if is_type(&osc_type, "Ident") {
        type_str = get(&osc_type, "name")?; // variable reference, no prefix
    } else if is_type(&osc_type, "Number")
        && type_value.is_integer()
        && type_value
            .as_f64()
            .is_some_and(|v| (0.0..=5.0).contains(&v))
    {
        let index = type_value.as_f64().unwrap_or(0.0) as usize;
        type_str = format!("oscKind.{}", OSC_KIND_NAMES[index]).into();
    }
    let mut parts = vec![format!("type: {}", to_string(&type_str)?)];
    push_if_non_default(&mut parts, "min", &get_opt(expr, "min"), Some(0.0), options)?;
    push_if_non_default(&mut parts, "max", &get_opt(expr, "max"), Some(1.0), options)?;
    push_if_non_default(
        &mut parts,
        "speed",
        &get_opt(expr, "speed"),
        Some(1.0),
        options,
    )?;
    push_if_non_default(
        &mut parts,
        "offset",
        &get_opt(expr, "offset"),
        Some(0.0),
        options,
    )?;
    push_if_non_default(
        &mut parts,
        "seed",
        &get_opt(expr, "seed"),
        Some(1.0),
        options,
    )?;
    Ok(format!("osc({})", parts.join(", ")).into())
}

/// `formatLetExpr` for a raw `Midi` AST node.
fn format_let_midi(expr: &Value, options: &UnparseOptions<'_>) -> Result<Value, JsError> {
    let mut parts = Vec::new();
    let channel = get_opt(expr, "channel");
    if channel.is_truthy() {
        parts.push(format!(
            "channel: {}",
            to_string(&format_let_expr(&channel, options)?)?
        ));
    }
    let mode = get_opt(expr, "mode");
    let mut mode_str = Value::Null;
    if is_type(&mode, "Member") && get(&mode, "path")?.is_truthy() {
        let name = last_path_segment(&get_opt(&mode, "path"))?;
        if !strict_equals(&name, &"velocity".into()) {
            mode_str = format!("midiMode.{}", to_string(&name)?).into();
        }
    } else if is_type(&mode, "Ident") {
        mode_str = get(&mode, "name")?; // variable reference, no prefix
    } else if is_type(&mode, "Number")
        && !strict_equals(&get_opt(&mode, "value"), &Value::Number(4.0))
    {
        mode_str = to_string(&get_opt(&mode, "value"))?.into();
    }
    if mode_str.is_truthy() {
        parts.push(format!("mode: {}", to_string(&mode_str)?));
    }
    push_if_non_default(&mut parts, "min", &get_opt(expr, "min"), Some(0.0), options)?;
    push_if_non_default(&mut parts, "max", &get_opt(expr, "max"), Some(1.0), options)?;
    push_if_non_default(
        &mut parts,
        "sensitivity",
        &get_opt(expr, "sensitivity"),
        Some(1.0),
        options,
    )?;
    for field in ["cc", "nrpn", "zone", "members"] {
        push_if_non_default(&mut parts, field, &get_opt(expr, field), None, options)?;
    }
    // MIDI identity accepts either DSL quote style. Normalize the decoded value
    // through JSON escaping so every result reparses.
    for field in ["name", "id"] {
        let node = get_opt(expr, field);
        if is_type(&node, "String") {
            parts.push(format!(
                "{field}: {}",
                stringify_decoded(&get_opt(&node, "value"))?
            ));
        }
    }
    Ok(format!("midi({})", parts.join(", ")).into())
}

/// `formatLetExpr` for a raw `Audio` AST node.
fn format_let_audio(expr: &Value, options: &UnparseOptions<'_>) -> Result<Value, JsError> {
    let format_audio_field = |node: &Value| -> Result<Value, JsError> {
        if is_type(node, "String") {
            Ok(stringify_decoded(&get_opt(node, "value"))?.into())
        } else {
            format_let_expr(node, options)
        }
    };
    let band = get_opt(expr, "band");
    let band_str = if band.is_truthy() {
        format_audio_field(&band)?
    } else {
        "audioBand.low".into()
    };
    let mut parts = vec![format!("band: {}", to_string(&band_str)?)];
    for (field, default) in [("min", Some(0.0)), ("max", Some(1.0)), ("channel", None)] {
        let node = get_opt(expr, field);
        if !node.is_truthy() {
            continue;
        }
        let numeric = num_val(&node);
        if let Some(default) = default
            && strict_equals(&numeric, &Value::Number(default))
        {
            continue;
        }
        parts.push(format!(
            "{field}: {}",
            to_string(&format_audio_field(&node)?)?
        ));
    }
    for field in ["name", "id"] {
        let node = get_opt(expr, field);
        if is_type(&node, "String") {
            parts.push(format!(
                "{field}: {}",
                to_string(&format_audio_field(&node)?)?
            ));
        }
    }
    Ok(format!("audio({})", parts.join(", ")).into())
}

/// A `stepsByTemp` entry: the step and its override object.
#[derive(Clone)]
struct StepEntry {
    step: Value,
    override_: Value,
}

/// `new Map()` keyed by JavaScript values (SameValueZero), in insertion order.
type TempMap = Vec<(Value, StepEntry)>;

fn map_get<'m, T>(map: &'m [(Value, T)], key: &Value) -> Option<&'m T> {
    map.iter()
        .find(|(k, _)| same_value_zero(k, key))
        .map(|(_, v)| v)
}

/// `map.set(key, value)`: an existing key keeps its position.
fn map_set<T>(map: &mut Vec<(Value, T)>, key: Value, value: T) {
    match map.iter_mut().find(|(k, _)| same_value_zero(k, &key)) {
        Some(slot) => slot.1 = value,
        None => map.push((key, value)),
    }
}

fn set_has(set: &[Value], key: &Value) -> bool {
    set.iter().any(|k| same_value_zero(k, key))
}

/// `collectDependencies(index, collected)`: the temp producer `index` and every
/// producer it reads (its `from` predecessor and temp-valued arguments).
///
/// The reference recurses; only the reachable set is observable (`collected` is
/// read for membership), and no step of the walk can throw, so the port walks
/// iteratively and needs no stack proportional to the producer chain's length.
fn collect_dependencies(steps_by_temp: &TempMap, index: &Value, collected: &mut Vec<Value>) {
    let mut pending = vec![index.clone()];
    while let Some(index) = pending.pop() {
        if set_has(collected, &index) {
            continue;
        }
        let Some(entry) = map_get(steps_by_temp, &index) else {
            continue;
        };
        collected.push(index);
        // Visit order mirrors the recursion: `from` first, then argument temps.
        let mut next = Vec::new();
        let from = member(&entry.step, "from");
        if !from.is_nullish() {
            next.push(from);
        }
        let mut merged = Object::new();
        spread_into(&mut merged, &member(&entry.step, "args"));
        spread_into(&mut merged, &entry.override_);
        for value in merged.values() {
            if is_kind_temp(value) {
                next.push(get_opt(value, "index"));
            }
        }
        pending.extend(next.into_iter().rev());
    }
}

/// One element of a chain being rebuilt (`{ code, leadingComments?, isSubchainBegin?,
/// isSubchainEnd? }`).
struct ChainElement {
    code: String,
    leading_comments: Option<Value>,
    is_subchain_begin: bool,
    is_subchain_end: bool,
}

/// `joinChainWithComments(chain)`: chain elements joined with their comments,
/// two-space continuation indent (four inside subchain blocks).
fn join_chain_with_comments(chain: &[ChainElement]) -> Result<String, JsError> {
    let mut parts: Vec<String> = Vec::new();
    let mut in_subchain = false;
    for (i, elem) in chain.iter().enumerate() {
        let is_first = i == 0;
        let base_indent = if in_subchain { "    " } else { "  " };

        // Emit leading comments for this element
        if let Some(comments) = &elem.leading_comments
            && comments.is_truthy()
            && has_length(comments)?
        {
            for comment in iterate(comments, "elem.leadingComments")? {
                if is_first {
                    // Comments before first element go on their own line
                    parts.push(join_elem(&comment)?);
                } else {
                    // Comments before chained elements get indented
                    parts.push(format!("{base_indent}{}", to_string(&comment)?));
                }
            }
        }

        if elem.is_subchain_begin {
            if is_first {
                parts.push(elem.code.clone());
            } else {
                parts.push(format!("  .{}", elem.code));
            }
            in_subchain = true;
            continue;
        }
        if elem.is_subchain_end {
            parts.push(format!("  {}", elem.code));
            in_subchain = false;
            continue;
        }
        if is_first {
            parts.push(elem.code.clone());
        } else if in_subchain {
            // Inside subchain: 4-space indent with dot
            parts.push(format!("    .{}", elem.code));
        } else {
            parts.push(format!("  .{}", elem.code));
        }
    }
    Ok(parts.join("\n"))
}

/// `step.leadingComments` when present and non-empty.
fn leading_comments_of(step: &Value) -> Result<Option<Value>, JsError> {
    let comments = get(step, "leadingComments")?;
    if comments.is_truthy() && has_length(&comments)? {
        Ok(Some(comments))
    } else {
        Ok(None)
    }
}

/// `x?.name || x` (surface references written as names).
fn name_or_self(x: &Value) -> Value {
    or(get_opt(x, "name"), || x.clone())
}

/// `overrides[index]` (throws for a `null` overrides object, as the reference does).
fn override_at(overrides: &Value, index: f64) -> Result<Value, JsError> {
    get_v(overrides, &Value::Number(index))
}

/// `unparse(compiled, overrides, options)`: the complete DSL source of a compiled
/// program.
///
/// `overrides` maps a global step index (counting every step of every plan,
/// builtins included) to parameter values that replace the step's arguments;
/// `options.getEffectDef` supplies effect definitions whose `globals` filter and
/// format the parameters. The registry provides the op schemas (`ops`) used
/// for default suppression when no definition is available, and the standard
/// enums used when `options.enums` is not given.
pub fn unparse(
    compiled: &Value,
    overrides: &Value,
    options: &UnparseOptions<'_>,
    registry: &Registry,
) -> Result<String, JsError> {
    let _depth = DepthGuard::enter()?;
    let empty_overrides = Value::Object(Object::new());
    let overrides = if overrides.is_undefined() {
        &empty_overrides
    } else {
        overrides
    };
    // Ensure enum definitions are available for formatting
    let mut options = options.clone();
    if !options.enums.is_truthy() {
        options.enums = Rc::new(Value::Object(registry.std_enums()));
    }
    let options = options;
    let mut lines: Vec<String> = Vec::new();
    let get_effect_def = options.get_effect_def.clone();
    let search_namespaces = or(get(compiled, "searchNamespaces")?, || {
        Value::Array(Vec::new())
    });

    // Add search directive if present (with two line breaks after)
    if greater_than_zero(&member(&search_namespaces, "length"))?
        && !options.omit_search_directive.is_truthy()
    {
        lines.push(format!(
            "search {}",
            array_join(&search_namespaces, ", ", "searchNamespaces")?
        ));
        lines.push(String::new()); // First blank line after search
    }

    // Emit let declarations
    let vars = get(compiled, "vars")?;
    if vars.is_truthy() && has_length(&vars)? {
        for v in iterate(&vars, "compiled.vars")? {
            if let Some(comments) = leading_comments_of(&v)? {
                for c in iterate(&comments, "v.leadingComments")? {
                    lines.push(join_elem(&c)?);
                }
            }
            let expr_str = format_let_expr(&get(&v, "expr")?, &options)?;
            lines.push(format!(
                "let {} = {}",
                to_string(&get(&v, "name")?)?,
                to_string(&expr_str)?
            ));
        }
        lines.push(String::new()); // blank line separator after let block
    }

    // Track global step index across all plans
    let mut global_step_index = 0.0;

    let plans = or(get(compiled, "plans")?, || Value::Array(Vec::new()));
    let plan_count = to_number(&member(&plans, "length"))?;
    let mut plan_index = 0.0;
    while plan_index < plan_count {
        let plan = get_v(&plans, &Value::Number(plan_index))?;
        let chain = get(&plan, "chain")?;
        if !chain.is_truthy() || strict_equals(&member(&chain, "length"), &Value::Number(0.0)) {
            plan_index += 1.0;
            continue;
        }

        // The compiler flattens inline surface producers into the plan. Rebuild
        // those dependency chains as arguments, retaining each producer's edits.
        let steps = match &chain {
            Value::Array(steps) => steps.clone(),
            _ => return Err(not_a_function("plan.chain.map")),
        };
        let mut steps_by_temp: TempMap = Vec::new();
        for (index, step) in steps.iter().enumerate() {
            let temp = get(step, "temp")?;
            let override_ = or(
                override_at(overrides, global_step_index + index as f64)?,
                || Value::Object(Object::new()),
            );
            map_set(
                &mut steps_by_temp,
                temp,
                StepEntry {
                    step: step.clone(),
                    override_,
                },
            );
        }
        let mut inline_temps: Vec<Value> = Vec::new();
        for (_, entry) in &steps_by_temp {
            let args = or(get(&entry.step, "args")?, || Value::Object(Object::new()));
            let mut all = values(&args);
            all.extend(values(&entry.override_));
            for value in &all {
                if is_kind_temp(value) {
                    collect_dependencies(
                        &steps_by_temp,
                        &get_opt(value, "index"),
                        &mut inline_temps,
                    );
                }
            }
        }
        let steps_by_temp = Rc::new(steps_by_temp);
        let inline_code: Rc<RefCell<Vec<(Value, Value)>>> = Rc::default();
        let format_temp: FormatTemp<'_> = {
            let steps_by_temp = steps_by_temp.clone();
            let inline_code = inline_code.clone();
            let options = options.clone();
            let search_namespaces = search_namespaces.clone();
            Rc::new(move |index: &Value| -> Result<Value, JsError> {
                if let Some(code) = map_get(&inline_code.borrow(), index) {
                    return Ok(code.clone());
                }
                let mut dependencies = Vec::new();
                collect_dependencies(&steps_by_temp, index, &mut dependencies);
                let mut chain = Vec::new();
                let mut nested_overrides = Object::new();
                for (temp, entry) in steps_by_temp.iter() {
                    if !set_has(&dependencies, temp) {
                        continue;
                    }
                    nested_overrides.insert(chain.len().to_string(), entry.override_.clone());
                    chain.push(entry.step.clone());
                }
                let mut plan = Object::new();
                plan.insert("chain", Value::Array(chain));
                let mut nested = Object::new();
                nested.insert("searchNamespaces", search_namespaces.clone());
                nested.insert("plans", Value::Array(vec![Value::Object(plan)]));
                let mut nested_options = options.clone();
                nested_options.multiline_kwargs = Rc::new(Value::Bool(false));
                nested_options.omit_search_directive = Rc::new(Value::Bool(true));
                let code = unparse(
                    &Value::Object(nested),
                    &Value::Object(nested_overrides),
                    &nested_options,
                    registry,
                )?;
                let code = Value::String(code);
                map_set(&mut inline_code.borrow_mut(), index.clone(), code.clone());
                Ok(code)
            })
        };
        let mut plan_options = options.clone();
        plan_options.format_temp = Some(format_temp);

        // Emit plan-level leading comments
        if let Some(comments) = leading_comments_of(&plan)? {
            for comment in iterate(&comments, "plan.leadingComments")? {
                lines.push(join_elem(&comment)?);
            }
        }

        // Build chains from steps, tracking comments; read() starts a new chain.
        let mut chains: Vec<Vec<ChainElement>> = Vec::new();
        let mut current_chain: Vec<ChainElement> = Vec::new();
        let mut in_subchain = false;

        for step in &steps {
            if set_has(&inline_temps, &get(step, "temp")?) {
                global_step_index += 1.0;
                continue;
            }
            let make_chain_element = |code: String| -> Result<ChainElement, JsError> {
                Ok(ChainElement {
                    code,
                    leading_comments: leading_comments_of(step)?,
                    is_subchain_begin: false,
                    is_subchain_end: false,
                })
            };
            let builtin = get(step, "builtin")?.is_truthy();
            let op = get(step, "op")?;
            let op_is = |name: &str| builtin && strict_equals(&op, &Value::from(name));
            let args = get(step, "args")?;

            // Handle builtin read operations - always starts a new chain
            if op_is("_read") || op_is("_read3d") {
                if !current_chain.is_empty() {
                    chains.push(std::mem::take(&mut current_chain));
                }
                // Check for _skip: overrides take precedence over step.args
                let step_override = override_at(overrides, global_step_index)?;
                let has_override = !get_opt(&step_override, "_skip").is_undefined();
                let is_skipped = if has_override {
                    strict_equals(&get(&step_override, "_skip")?, &true.into())
                } else {
                    strict_equals(&get_opt(&args, "_skip"), &true.into())
                };
                // Use positional form when no _skip, keyword form when _skip is present
                let code = if op_is("_read") {
                    let tex_name = to_string(&name_or_self(&get_opt(&args, "tex")))?;
                    if is_skipped {
                        format!("read(surface: {tex_name}, _skip: true)")
                    } else {
                        format!("read({tex_name})")
                    }
                } else {
                    let tex3d = to_string(&name_or_self(&get_opt(&args, "tex3d")))?;
                    let geo = to_string(&name_or_self(&get_opt(&args, "geo")))?;
                    if is_skipped {
                        format!("read3d(tex3d: {tex3d}, geo: {geo}, _skip: true)")
                    } else {
                        format!("read3d({tex3d}, {geo})")
                    }
                };
                current_chain.push(make_chain_element(code)?);
                global_step_index += 1.0;
                continue;
            }
            // Handle builtin write operations (mid-chain writes)
            if op_is("_write") {
                let tex_name = to_string(&name_or_self(&get_opt(&args, "tex")))?;
                current_chain.push(make_chain_element(format!("write({tex_name})"))?);
                global_step_index += 1.0;
                continue;
            }
            // Handle builtin write3d operations (mid-chain write3d)
            if op_is("_write3d") {
                let tex3d = to_string(&name_or_self(&get_opt(&args, "tex3d")))?;
                let geo = to_string(&name_or_self(&get_opt(&args, "geo")))?;
                current_chain.push(make_chain_element(format!("write3d({tex3d}, {geo})"))?);
                global_step_index += 1.0;
                continue;
            }

            // Handle subchain begin marker - starts a subchain block
            if op_is("_subchain_begin") {
                let name = get_opt(&args, "name");
                let id = get_opt(&args, "id");
                let mut parts = Vec::new();
                if name.is_truthy() {
                    parts.push(format!("name: \"{}\"", to_string(&name)?));
                }
                if id.is_truthy() {
                    parts.push(format!("id: \"{}\"", to_string(&id)?));
                }
                current_chain.push(ChainElement {
                    code: format!("subchain({}) {{", parts.join(", ")),
                    leading_comments: leading_comments_of(step)?,
                    is_subchain_begin: true,
                    is_subchain_end: false,
                });
                in_subchain = true;
                global_step_index += 1.0;
                continue;
            }

            // Handle subchain end marker - ends a subchain block
            if op_is("_subchain_end") {
                current_chain.push(ChainElement {
                    code: "}".into(),
                    leading_comments: None,
                    is_subchain_begin: false,
                    is_subchain_end: true,
                });
                in_subchain = false;
                global_step_index += 1.0;
                continue;
            }

            // Check for parameter overrides for this step
            let step_overrides = or(override_at(overrides, global_step_index)?, || {
                Value::Object(Object::new())
            });

            // Get effect definition if callback provided
            let mut effect_def = Value::Null;
            if let Some(get_effect_def) = &get_effect_def {
                let ns = get_opt(step, "namespace");
                let namespace = or(get_opt(&ns, "namespace"), || {
                    or(get_opt(&ns, "resolved"), || Value::Null)
                });
                effect_def = get_effect_def(&op, &namespace)?;
            }

            // Determine the call name - strip namespace prefix if it's in search namespaces
            let mut call_name = op.clone();
            let step_namespace = get(step, "namespace")?;
            let ns_info = or(get_opt(&step_namespace, "call"), || step_namespace.clone());
            let is_from_override = strict_equals(&get_opt(&ns_info, "fromOverride"), &true.into());
            let from_namespace = if is_from_override {
                or(get_opt(&ns_info, "resolved"), || {
                    or(get_opt(&ns_info, "name"), || {
                        get_opt(&step_namespace, "resolved")
                    })
                })
            } else {
                Value::Null
            };
            for ns in iterate(&search_namespaces, "searchNamespaces")? {
                let prefix = format!("{}.", to_string(&ns)?);
                let name = str_method(&call_name, "callName", "startsWith")?;
                if let Some(rest) = name.strip_prefix(&prefix) {
                    call_name = rest.into();
                    break;
                }
            }
            // For from() overrides, strip the override namespace prefix from the call name
            if is_from_override && from_namespace.is_truthy() {
                let from_prefix = format!("{}.", to_string(&from_namespace)?);
                let name = str_method(&call_name, "callName", "startsWith")?;
                if let Some(rest) = name.strip_prefix(&from_prefix) {
                    call_name = rest.into();
                }
            }

            let mut kwargs = Object::new();
            // Start with original args (already keyed by param names)
            if args.is_truthy() {
                for (key, value) in entries(&args) {
                    // Skip internal properties
                    if key == "from" || key == "temp" {
                        continue;
                    }
                    // Skip _skip: false (only include when true)
                    if key == "_skip" && !strict_equals(&value, &true.into()) {
                        continue;
                    }
                    // Handle surface references
                    let kind = get_opt(&value, "kind");
                    if is_object_like(&value)
                        && kind.is_truthy()
                        && !strict_equals(&kind, &"temp".into())
                    {
                        set_plain(&mut kwargs, &key, get_opt(&value, "name"));
                    } else {
                        set_plain(&mut kwargs, &key, value);
                    }
                }
            }

            // Build specs map from effect definition
            let specs = or(get_opt(&effect_def, "globals"), || {
                Value::Object(Object::new())
            });
            // Without an effect definition, recover the op schema the validator
            // used to fill default values into step args.
            let mut schema_specs = Object::new();
            if !effect_def.is_truthy() {
                let op_spec = object_member(&registry.ops, &to_property_key(&op)?);
                let op_args = get_opt(&op_spec, "args");
                if op_args.is_truthy() {
                    for arg in iterate(&op_args, "opSpec.args")? {
                        let arg_name = get_opt(&arg, "name");
                        if arg_name.is_truthy() {
                            let key = to_property_key(&arg_name)?;
                            if member(&specs, &key).is_undefined() {
                                set_plain(&mut schema_specs, &key, arg.clone());
                            }
                        }
                    }
                }
            }

            // Apply overrides - with an effect definition, only keys defined in
            // its globals; internal _ prefixed args (e.g. _skip) always.
            for (key, value) in entries(&step_overrides) {
                if key.starts_with('_') {
                    set_plain(&mut kwargs, &key, value);
                } else if effect_def.is_truthy() {
                    if !member(&specs, &key).is_undefined() {
                        set_plain(&mut kwargs, &key, value);
                    }
                } else {
                    set_plain(&mut kwargs, &key, value);
                }
            }

            // volumeSize on render*3d consumers is inherited from the upstream 3D
            // producer; never emit the consumer's own copy.
            if !object_member(&kwargs, "volumeSize").is_undefined()
                && strict_equals(
                    &get_opt(&get_opt(&get_opt(&specs, "volumeSize"), "ui"), "control"),
                    &false.into(),
                )
            {
                kwargs.remove("volumeSize");
            }

            let mut call = Object::new();
            call.insert("name", call_name);
            call.insert("kwargs", Value::Object(kwargs));
            call.insert("args", Value::Array(Vec::new()));
            let arg_sources = get(step, "argSources")?;
            if arg_sources.is_truthy() {
                call.insert("argSources", arg_sources);
            }

            // Calculate indent: 4 spaces inside subchain, 2 outside; 0 for first element
            let call_indent = if current_chain.is_empty() {
                0.0
            } else if in_subchain {
                4.0
            } else {
                2.0
            };
            let mut call_options = plan_options.clone();
            call_options.specs = Rc::new(specs);
            call_options.schema_specs = Rc::new(Value::Object(schema_specs));
            call_options.indent = Rc::new(Value::Number(call_indent));
            let mut call_code = unparse_call(&Value::Object(call), &call_options)?;
            // Wrap in from(namespace, call) for cross-namespace references
            if is_from_override && from_namespace.is_truthy() {
                call_code = format!("from({}, {call_code})", to_string(&from_namespace)?);
            }
            current_chain.push(make_chain_element(call_code)?);
            global_step_index += 1.0;
        }

        // Flush final chain
        if !current_chain.is_empty() {
            chains.push(current_chain);
        }

        let mut joined = Vec::with_capacity(chains.len());
        for chain in &chains {
            joined.push(join_chain_with_comments(chain)?);
        }
        let mut line = joined.join("\n\n");

        // Check if chain already ends with a _write step (chainable writes are inline)
        let last_step = steps.last().cloned().unwrap_or(Value::Undefined);
        let last_is = |name: &str| {
            last_step.is_truthy()
                && get_opt(&last_step, "builtin").is_truthy()
                && strict_equals(&get_opt(&last_step, "op"), &Value::from(name))
        };
        let chain_ends_with_write = last_is("_write");
        let chain_ends_with_write3d = last_is("_write3d");

        // Add write directive only if chain doesn't already end with _write
        let write = get(&plan, "write")?;
        if write.is_truthy() && !chain_ends_with_write {
            let write_name = match &write {
                Value::String(_) => write.clone(),
                _ => get(&write, "name")?,
            };
            line.push_str(&format!("\n  .write({})", to_string(&write_name)?));
        }
        // Add write3d directive only if chain doesn't already end with _write3d
        let write3d = get(&plan, "write3d")?;
        if write3d.is_truthy() && !chain_ends_with_write3d {
            let tex3d = to_string(&name_or_self(&get_opt(&write3d, "tex3d")))?;
            let geo = to_string(&name_or_self(&get_opt(&write3d, "geo")))?;
            line.push_str(&format!("\n  .write3d({tex3d}, {geo})"));
        }

        lines.push(line);

        // Blank line between chain statements, not after the last plan
        if plan_index < plan_count - 1.0 {
            lines.push(String::new());
        }
        plan_index += 1.0;
    }

    // Add render directive if present (surface name: o0-o7)
    let render = get(compiled, "render")?;
    if render.is_truthy() {
        lines.push(String::new());
        lines.push(format!("render({})", to_string(&render)?));
    }

    // Add trailing comments if present
    let trailing = get(compiled, "trailingComments")?;
    if trailing.is_truthy() && has_length(&trailing)? {
        for comment in iterate(&trailing, "compiled.trailingComments")? {
            lines.push(join_elem(&comment)?);
        }
    }

    Ok(lines.join("\n"))
}

/// JavaScript LineTerminator (what `.` does not match and where `^`/`$` anchor in
/// multiline mode).
fn is_line_terminator(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')
}

/// `originalDsl.match(/^search\s+(\S.*?)$/m)?.[1]`.
fn match_search_directive(src: &str) -> Option<String> {
    let chars: Vec<char> = src.chars().collect();
    let pattern: Vec<char> = "search".chars().collect();
    let mut starts = vec![0];
    for (i, c) in chars.iter().enumerate() {
        if is_line_terminator(*c) {
            starts.push(i + 1);
        }
    }
    for start in starts {
        if chars.len() < start + pattern.len() || chars[start..start + pattern.len()] != pattern[..]
        {
            continue;
        }
        let mut i = start + pattern.len();
        let ws_start = i;
        while i < chars.len() && is_js_whitespace(chars[i]) {
            i += 1;
        }
        // `\s+` needs one whitespace; `\S` then needs a following non-space.
        if i == ws_start || i >= chars.len() {
            continue;
        }
        let mut j = i + 1;
        while j < chars.len() && !is_line_terminator(chars[j]) {
            j += 1;
        }
        return Some(chars[i..j].iter().collect());
    }
    None
}

/// `s.split(/\s*,\s*/)`.
fn split_comma_list(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut out = Vec::new();
    let mut piece_start = 0;
    let mut p = 0;
    while p < chars.len() {
        // A match starting exactly at p: greedy whitespace, then a comma.
        let mut q = p;
        while q < chars.len() && is_js_whitespace(chars[q]) {
            q += 1;
        }
        if q < chars.len() && chars[q] == ',' {
            let mut end = q + 1;
            while end < chars.len() && is_js_whitespace(chars[end]) {
                end += 1;
            }
            out.push(chars[piece_start..p].iter().collect());
            piece_start = end;
            p = end;
        } else {
            p += 1;
        }
    }
    out.push(chars[piece_start..].iter().collect());
    out
}

/// `applyParameterUpdates(originalDsl, compileFn, parameterUpdates)`: compile
/// the source, take its `search` namespaces from the source text, and regenerate
/// it with the overrides applied. A compile result without `plans` returns the
/// source unchanged.
pub fn apply_parameter_updates(
    original_dsl: &str,
    compile_fn: impl FnOnce(&str) -> Result<Value, JsError>,
    parameter_updates: &Value,
    registry: &Registry,
) -> Result<String, JsError> {
    // Parse the original DSL
    let mut compiled = compile_fn(original_dsl)?;
    if !compiled.is_truthy() || !get(&compiled, "plans")?.is_truthy() {
        return Ok(original_dsl.to_owned());
    }

    // Extract search namespaces from original source
    if let Some(list) = match_search_directive(original_dsl)
        && let Value::Object(o) = &mut compiled
    {
        let namespaces = split_comma_list(&list)
            .into_iter()
            .map(Value::from)
            .collect();
        set_plain(o, "searchNamespaces", Value::Array(namespaces));
    }

    // Generate new source with overrides
    unparse(
        &compiled,
        parameter_updates,
        &UnparseOptions::default(),
        registry,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::js;

    #[test]
    fn lossless_numbers_have_no_exponent() {
        for (x, s) in [
            (1.7778, "1.7778"),
            (0.0004, "0.0004"),
            (1e-7, "0.0000001"),
            (1.5e-7, "0.00000015"),
            (-2.5e-9, "-0.0000000025"),
            (1e21, "1000000000000000000000"),
            (1.2345e22, "12345000000000000000000"),
            (-1e21, "-1000000000000000000000"),
            (0.0, "0"),
            (-0.0, "0"),
            (5e-324, &format!("0.{}5", "0".repeat(323))),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (0.8000000780001997, "0.8000000780001997"),
        ] {
            assert_eq!(format_lossless_number(x), s, "{x}");
        }
    }

    #[test]
    fn hex_colors_round_and_clamp() {
        let o = UnparseOptions::default();
        let f = |v: Value, spec: Value| format_value(&v, &spec, &o, &Value::Undefined).unwrap();
        assert_eq!(
            f(js!([0.2, 0.4, 0.6, 0.8]), js!({"type": "color"})),
            Value::from("#336699cc")
        );
        assert_eq!(
            f(
                Value::from_json("[1.5, -1, 0.5]").unwrap(),
                js!({"type": "color"})
            ),
            Value::from("#ff0080")
        );
        // The vec4 hex form does not clamp.
        assert_eq!(
            f(
                Value::from_json("[2, -0.5, 0, 1]").unwrap(),
                js!({"type": "vec4"})
            ),
            Value::from("#1fe-7f00ff")
        );
        assert_eq!(
            f(
                js!([
                    0.8000000780001997,
                    0.4500000195000499,
                    0.09999996099990016,
                    0.4500000195000499
                ]),
                js!({"type": "vec4"})
            ),
            Value::from("#cc731973")
        );
        assert_eq!(
            f(
                Value::from_json("[0.8000000780001997, -0.45, 1.1, 0.0000001]").unwrap(),
                js!({"type": "vec4", "ui": {"format": "vector"}})
            ),
            Value::from("[0.8000000780001997, -0.45, 1.1, 0.0000001]")
        );
    }

    #[test]
    fn strings_quote_only_when_needed() {
        let o = UnparseOptions::default();
        let f = |v: &str, spec: Value| {
            format_value(&Value::from(v), &spec, &o, &Value::Undefined).unwrap()
        };
        assert_eq!(f("perlin", Value::Null), Value::from("perlin"));
        assert_eq!(f("oscKind.sine", Value::Null), Value::from("oscKind.sine"));
        assert_eq!(f("0vjd", Value::Null), Value::from("\"0vjd\""));
        assert_eq!(
            f("a \"b\" \\ c", Value::Null),
            Value::from("\"a \\\"b\\\" \\\\ c\"")
        );
        assert_eq!(
            f("two\nlines", Value::Null),
            Value::from("\"\"\"two\nlines\"\"\"")
        );
        assert_eq!(
            f("#ff0000", js!({"type": "string"})),
            Value::from("#ff0000")
        );
        assert_eq!(
            f("plain", js!({"type": "string"})),
            Value::from("\"plain\"")
        );
    }

    #[test]
    fn json_escapes_of_decoded_identities() {
        // Valid JSON escapes decode before re-escaping.
        assert_eq!(
            stringify_decoded(&Value::from(r#"a \"q\" \\ b"#)).unwrap(),
            r#""a \"q\" \\ b""#
        );
        // Single-quote escapes take the fallback path; unknown escapes stay verbatim.
        assert_eq!(
            stringify_decoded(&Value::from(r"it\'s \x")).unwrap(),
            r#""it's \\x""#
        );
        // Lone surrogate escapes survive as escapes.
        assert_eq!(
            stringify_decoded(&Value::from(r"\ud800")).unwrap(),
            r#""\ud800""#
        );
    }

    #[test]
    fn search_directive_extraction() {
        assert_eq!(
            match_search_directive("search synth, filter\nnoise()").as_deref(),
            Some("synth, filter")
        );
        assert_eq!(
            match_search_directive("// x\nsearch  a ,b , c // tail\n").as_deref(),
            Some("a ,b , c // tail")
        );
        assert_eq!(
            match_search_directive("search\n\nsynth\nx").as_deref(),
            Some("synth")
        );
        assert_eq!(match_search_directive("search   \n"), None);
        assert_eq!(match_search_directive("  search synth"), None);
        assert_eq!(split_comma_list("a , , b"), vec!["a", "", "b"]);
        assert_eq!(split_comma_list("synth, filter "), vec!["synth", "filter "]);
        assert_eq!(split_comma_list("a,"), vec!["a", ""]);
    }

    #[test]
    fn runaway_recursion_is_a_range_error() {
        // `formatMidi` hands a runtime config with an object channel back to
        // `formatLetExpr`, which falls back to `formatValue`: the reference
        // recurses until the stack overflows.
        let value = js!({"_ast": {"type": "Midi"}, "channel": {"x": 1}});
        let err = format_value(
            &value,
            &Value::Null,
            &UnparseOptions::default(),
            &Value::Undefined,
        )
        .unwrap_err();
        assert_eq!(err, jsv::stack_overflow());
        // Nesting far beyond any program's stays within the guard's budget.
        let mut nested = Value::from(1.0);
        for _ in 0..24 {
            nested = Value::Array(vec![nested]);
        }
        let formatted = format_value(
            &nested,
            &Value::Null,
            &UnparseOptions::default(),
            &"array".into(),
        );
        assert!(formatted.is_ok());
    }

    #[test]
    fn basic_unparse() {
        let registry = Registry::new();
        let compiled = js!({
            "searchNamespaces": ["synth", "filter"],
            "plans": [{
                "chain": [
                    {"op": "synth.noise", "args": {"scale": 3}},
                    {"op": "filter.warp", "args": {"displacement": 0.5, "octaves": 3, "freq": 2}}
                ],
                "write": {"kind": "output", "name": "o0"}
            }]
        });
        let out = unparse(&compiled, &js!({}), &UnparseOptions::default(), &registry).unwrap();
        assert_eq!(
            out,
            "search synth, filter\n\nnoise(scale: 3)\n  .warp(\n    displacement: 0.5,\n    octaves: 3,\n    freq: 2\n  )\n  .write(o0)"
        );
    }
}
