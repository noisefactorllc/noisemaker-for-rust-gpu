//! JavaScript value semantics shared by the DSL tooling ports (unparser,
//! transform, error formatter, effect validator).
//!
//! The reference tooling manipulates plain JavaScript values with the language's
//! own coercions: template literals call `ToString`, `Array.prototype.join` turns
//! `null`/`undefined` elements into empty strings, `===` compares objects by
//! identity, property reads see inherited prototype members, `Object.entries`
//! enumerates strings by code unit. These helpers reproduce those rules over
//! [`Value`], including the `TypeError`/`RangeError` messages V8 throws.
//!
//! The number semantics and the basic comparisons (`===`, SameValueZero,
//! `Number.isFinite`) are [`crate::js`](mod@crate::js)'s, re-exported here; [`to_string`] and
//! [`to_number`] differ from [`crate::js::value_to_property_key`] and
//! [`crate::js::to_number`] on purpose: they run `ToPrimitive` on plain
//! objects whose own `toString`/`valueOf` members the tooling can see, and
//! throw where V8 throws.
//!
//! Object identity: a [`Value`] owns its members, so two objects never alias.
//! `===` between two objects (or arrays, or functions) is therefore `false`, which
//! is what the reference observes for values that do not share a reference.

use crate::error::JsError;
pub use crate::js::{is_finite_number, same_value_zero, strict_equals};
use crate::js::{number_to_string, string_to_number};
use crate::value::{Object, Value, is_array_index};

/// Inherited members of `Object.prototype` (own properties of every plain object's
/// prototype).
const OBJECT_PROTO: &[&str] = &[
    "constructor",
    "__defineGetter__",
    "__defineSetter__",
    "hasOwnProperty",
    "__lookupGetter__",
    "__lookupSetter__",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toString",
    "valueOf",
    "__proto__",
    "toLocaleString",
];

/// Methods of `Array.prototype`.
const ARRAY_PROTO: &[&str] = &[
    "constructor",
    "at",
    "concat",
    "copyWithin",
    "fill",
    "find",
    "findIndex",
    "findLast",
    "findLastIndex",
    "lastIndexOf",
    "pop",
    "push",
    "reverse",
    "shift",
    "unshift",
    "slice",
    "sort",
    "splice",
    "includes",
    "indexOf",
    "join",
    "keys",
    "entries",
    "values",
    "forEach",
    "filter",
    "flat",
    "flatMap",
    "map",
    "every",
    "some",
    "reduce",
    "reduceRight",
    "toLocaleString",
    "toString",
    "toReversed",
    "toSorted",
    "toSpliced",
    "with",
];

/// Methods of `String.prototype`.
const STRING_PROTO: &[&str] = &[
    "constructor",
    "anchor",
    "at",
    "big",
    "blink",
    "bold",
    "charAt",
    "charCodeAt",
    "codePointAt",
    "concat",
    "endsWith",
    "fontcolor",
    "fontsize",
    "fixed",
    "includes",
    "indexOf",
    "isWellFormed",
    "italics",
    "lastIndexOf",
    "link",
    "localeCompare",
    "match",
    "matchAll",
    "normalize",
    "padEnd",
    "padStart",
    "repeat",
    "replace",
    "replaceAll",
    "search",
    "slice",
    "small",
    "split",
    "strike",
    "sub",
    "substr",
    "substring",
    "sup",
    "startsWith",
    "toString",
    "toWellFormed",
    "trim",
    "trimStart",
    "trimLeft",
    "trimEnd",
    "trimRight",
    "toLocaleLowerCase",
    "toLocaleUpperCase",
    "toLowerCase",
    "toUpperCase",
    "valueOf",
];

/// Methods of `Number.prototype`.
const NUMBER_PROTO: &[&str] = &[
    "constructor",
    "toExponential",
    "toFixed",
    "toPrecision",
    "toString",
    "valueOf",
    "toLocaleString",
];

/// Methods of `Boolean.prototype`.
const BOOLEAN_PROTO: &[&str] = &["constructor", "toString", "valueOf"];

/// Methods of `Function.prototype`.
const FUNCTION_PROTO: &[&str] = &["constructor", "apply", "bind", "call", "toString"];

/// The largest string V8 builds (`String::kMaxLength` on 64-bit hosts).
pub const MAX_STRING_LENGTH: usize = (1 << 29) - 24;

/// A built-in function value as `Function.prototype.toString` prints it.
fn native_function(name: &str) -> Value {
    Value::Function(format!("function {name}() {{ [native code] }}"))
}

/// The inherited member `key` of a primitive or object whose prototype chain is
/// `protos` followed by `Object.prototype`.
fn inherited(key: &str, constructor: &str, protos: &[&[&str]]) -> Value {
    if key == "constructor" {
        return native_function(constructor);
    }
    for proto in protos {
        if proto.contains(&key) {
            return native_function(key);
        }
    }
    if key == "__proto__" {
        // `Object.prototype` (an object without enumerable members).
        return Value::Object(Object::new());
    }
    if OBJECT_PROTO.contains(&key) {
        return native_function(key);
    }
    Value::Undefined
}

/// UTF-16 code units of `s`.
pub fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

/// `s.length`.
fn str_length(s: &str) -> usize {
    s.encode_utf16().count()
}

/// The one-code-unit string `s[index]` (a lone surrogate becomes U+FFFD, which a
/// Rust string cannot hold).
fn code_unit_string(units: &[u16], index: usize) -> String {
    String::from_utf16_lossy(&units[index..index + 1])
}

/// `TypeError: Cannot read properties of <base> (reading '<key>')`.
pub fn cannot_read(base: &Value, key: &str) -> JsError {
    let what = if base.is_null() { "null" } else { "undefined" };
    JsError::type_error(format!(
        "Cannot read properties of {what} (reading '{key}')"
    ))
}

/// `TypeError: <expr> is not a function`.
pub fn not_a_function(expr: &str) -> JsError {
    JsError::type_error(format!("{expr} is not a function"))
}

/// `TypeError: <expr> is not iterable`.
fn not_iterable(expr: &str) -> JsError {
    JsError::type_error(format!("{expr} is not iterable"))
}

/// `RangeError: Maximum call stack size exceeded`.
pub fn stack_overflow() -> JsError {
    JsError::Error {
        name: "RangeError".into(),
        message: "Maximum call stack size exceeded".into(),
    }
}

/// `base[key]` for a non-nullish base: own members first, then the prototype
/// chain of the base's type. Nullish bases read as `undefined` here; use [`get`]
/// where the reference would throw.
pub fn member(base: &Value, key: &str) -> Value {
    match base {
        Value::Undefined | Value::Null => Value::Undefined,
        Value::Object(o) => match o.get(key) {
            Some(v) => v.clone(),
            None => inherited(key, "Object", &[]),
        },
        Value::Array(a) => {
            if key == "length" {
                return Value::Number(a.len() as f64);
            }
            if is_array_index(key) {
                let i: usize = key.parse().unwrap_or(usize::MAX);
                return a.get(i).cloned().unwrap_or(Value::Undefined);
            }
            if key == "__proto__" {
                // `Array.prototype` is itself an (empty) array.
                return Value::Array(Vec::new());
            }
            inherited(key, "Array", &[ARRAY_PROTO])
        }
        Value::String(s) => {
            if key == "length" {
                return Value::Number(str_length(s) as f64);
            }
            if is_array_index(key) {
                let units = utf16(s);
                let i: usize = key.parse().unwrap_or(usize::MAX);
                return if i < units.len() {
                    Value::String(code_unit_string(&units, i))
                } else {
                    Value::Undefined
                };
            }
            inherited(key, "String", &[STRING_PROTO])
        }
        Value::Number(_) => inherited(key, "Number", &[NUMBER_PROTO]),
        Value::Bool(_) => inherited(key, "Boolean", &[BOOLEAN_PROTO]),
        Value::Function(_) => match key {
            "prototype" | "name" | "length" => Value::Undefined,
            _ => inherited(key, "Function", &[FUNCTION_PROTO]),
        },
    }
}

/// `obj[key]` for a plain object: its own member, else the `Object.prototype`
/// member of that name.
pub fn object_member(obj: &Object, key: &str) -> Value {
    match obj.get(key) {
        Some(v) => v.clone(),
        None => inherited(key, "Object", &[]),
    }
}

/// `base[key]` (throws for nullish bases).
pub fn get(base: &Value, key: &str) -> Result<Value, JsError> {
    if base.is_nullish() {
        return Err(cannot_read(base, key));
    }
    Ok(member(base, key))
}

/// `base?.[key]`.
pub fn get_opt(base: &Value, key: &str) -> Value {
    member(base, key)
}

/// `base[keyValue]` with `ToPropertyKey(keyValue)` (throws for nullish bases).
pub fn get_v(base: &Value, key: &Value) -> Result<Value, JsError> {
    let k = to_property_key(key)?;
    get(base, &k)
}

/// `key in obj` for an object operand (own members or inherited ones).
pub fn has_property(obj: &Value, key: &str) -> bool {
    match obj {
        Value::Object(o) => o.contains_key(key) || OBJECT_PROTO.contains(&key),
        Value::Array(a) => {
            key == "length"
                || (is_array_index(key) && key.parse::<usize>().is_ok_and(|i| i < a.len()))
                || ARRAY_PROTO.contains(&key)
                || OBJECT_PROTO.contains(&key)
        }
        Value::Function(_) => {
            matches!(key, "prototype" | "name" | "length")
                || FUNCTION_PROTO.contains(&key)
                || OBJECT_PROTO.contains(&key)
        }
        _ => false,
    }
}

/// `ToPropertyKey(v)`.
pub fn to_property_key(v: &Value) -> Result<String, JsError> {
    to_string(v)
}

/// `ToString(v)` (`String(v)`, template literal substitution).
pub fn to_string(v: &Value) -> Result<String, JsError> {
    Ok(match v {
        Value::Undefined => "undefined".into(),
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => number_to_string(*n),
        Value::String(s) => s.clone(),
        Value::Array(a) => join(a, ",")?,
        Value::Object(o) => object_to_string(o)?,
        Value::Function(src) => src.clone(),
    })
}

/// `OrdinaryToPrimitive(obj, string)` for a plain object: an own non-callable
/// `toString` falls through to `valueOf`, whose inherited version returns the
/// object itself (not a primitive).
fn object_to_string(o: &Object) -> Result<String, JsError> {
    match o.get("toString") {
        None | Some(Value::Function(_)) => Ok("[object Object]".into()),
        Some(_) => match o.get("valueOf") {
            Some(Value::Function(_)) => Ok("[object Object]".into()),
            _ => Err(JsError::type_error(
                "Cannot convert object to primitive value",
            )),
        },
    }
}

/// `ToNumber(v)`.
pub fn to_number(v: &Value) -> Result<f64, JsError> {
    Ok(match v {
        Value::Undefined => f64::NAN,
        Value::Null => 0.0,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Number(n) => *n,
        Value::String(s) => string_to_number(s),
        Value::Array(_) | Value::Object(_) | Value::Function(_) => string_to_number(&to_string(v)?),
    })
}

/// `Array.prototype.join(sep)`: `null`/`undefined` elements become empty strings.
pub fn join(values: &[Value], sep: &str) -> Result<String, JsError> {
    let mut out = String::new();
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        if !v.is_nullish() {
            out.push_str(&to_string(v)?);
        }
    }
    Ok(out)
}

/// `typeof v === 'object' && v !== null` (arrays included).
pub fn is_object_like(v: &Value) -> bool {
    matches!(v, Value::Object(_) | Value::Array(_))
}

/// `Object.entries(v)` (own enumerable string-keyed members; strings enumerate
/// their code units, arrays their indices; other primitives have none).
pub fn entries(v: &Value) -> Vec<(String, Value)> {
    match v {
        Value::Object(o) => o.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect(),
        Value::String(s) => {
            let units = utf16(s);
            (0..units.len())
                .map(|i| (i.to_string(), Value::String(code_unit_string(&units, i))))
                .collect()
        }
        _ => Vec::new(),
    }
}

/// `Object.keys(v)`.
pub fn keys(v: &Value) -> Vec<String> {
    entries(v).into_iter().map(|(k, _)| k).collect()
}

/// `Object.values(v)`.
pub fn values(v: &Value) -> Vec<Value> {
    entries(v).into_iter().map(|(_, v)| v).collect()
}

/// `Object.entries(v)` where the reference throws for nullish operands.
pub fn entries_strict(v: &Value) -> Result<Vec<(String, Value)>, JsError> {
    if v.is_nullish() {
        return Err(JsError::type_error(
            "Cannot convert undefined or null to object",
        ));
    }
    Ok(entries(v))
}

/// `obj[key] = value` on a plain object. Assigning `__proto__` replaces the
/// prototype instead of creating an own member, so it adds nothing enumerable.
pub fn set_plain(obj: &mut Object, key: &str, value: Value) {
    if key == "__proto__" {
        return;
    }
    obj.insert(key.to_owned(), value);
}

/// `{ ...target, ...source }` (own enumerable members of `source` copied in).
pub fn spread_into(target: &mut Object, source: &Value) {
    for (k, v) in entries(source) {
        // Spread defines properties, so `__proto__` becomes an own member here.
        target.insert(k, v);
    }
}

/// The values a `for...of` loop visits (`<expr> is not iterable` otherwise).
/// Strings iterate by code point.
pub fn iterate(v: &Value, expr: &str) -> Result<Vec<Value>, JsError> {
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        _ => Err(not_iterable(expr)),
    }
}

/// The values a `for...of` loop over a composite expression (`(a || [])`)
/// visits; V8 then names the value instead of the expression.
pub fn iterate_anon(v: &Value) -> Result<Vec<Value>, JsError> {
    let suffix = "is not iterable (cannot read property Symbol(Symbol.iterator))";
    match v {
        Value::Array(a) => Ok(a.clone()),
        Value::String(s) => Ok(s.chars().map(|c| Value::String(c.to_string())).collect()),
        Value::Number(n) => Err(JsError::type_error(format!(
            "number {} {suffix}",
            number_to_string(*n)
        ))),
        Value::Bool(b) => Err(JsError::type_error(format!("boolean {b} {suffix}"))),
        Value::Function(_) => Err(JsError::type_error(format!("function {suffix}"))),
        Value::Object(_) | Value::Undefined | Value::Null => {
            Err(JsError::type_error(format!("object {suffix}")))
        }
    }
}

/// `a in b`: own or inherited members of an object operand; a primitive right
/// operand throws as in V8.
pub fn in_operator(key: &Value, obj: &Value) -> Result<bool, JsError> {
    let k = to_property_key(key)?;
    match obj {
        Value::Object(_) | Value::Array(_) | Value::Function(_) => Ok(has_property(obj, &k)),
        _ => Err(JsError::type_error(format!(
            "Cannot use 'in' operator to search for '{k}' in {}",
            to_string(obj)?
        ))),
    }
}

/// `String.prototype.repeat(count)` for a non-negative count.
pub fn repeat(s: &str, count: f64) -> Result<String, JsError> {
    let n = if count.is_nan() { 0.0 } else { count.trunc() };
    if n < 0.0 || n.is_infinite() {
        return Err(JsError::Error {
            name: "RangeError".into(),
            message: format!("Invalid count value: {}", number_to_string(count)),
        });
    }
    let n = n as usize;
    if n.saturating_mul(str_length(s)) > MAX_STRING_LENGTH {
        return Err(JsError::Error {
            name: "RangeError".into(),
            message: "Invalid string length".into(),
        });
    }
    Ok(s.repeat(n))
}

/// `String.prototype.padStart(width, fill)` for a one-code-unit fill.
pub fn pad_start(s: &str, width: usize, fill: char) -> String {
    let len = str_length(s);
    if len >= width {
        return s.to_owned();
    }
    let mut out: String = std::iter::repeat_n(fill, width - len).collect();
    out.push_str(s);
    out
}

/// `Number.prototype.toString(16)` for an integral or non-finite number (the
/// results of `Math.round`). Exact: every step of the radix-16 conversion is a
/// power-of-two operation.
pub fn number_to_hex(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    if x == 0.0 {
        return "0".into();
    }
    if x < 0.0 {
        return format!("-{}", number_to_hex(-x));
    }
    // x = mantissa × 2^exp with an integral mantissa.
    let bits = x.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mut mantissa, mut exp) = if raw_exp == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), raw_exp - 1075)
    };
    while exp < 0 && mantissa & 1 == 0 {
        mantissa >>= 1;
        exp += 1;
    }
    if exp < 0 {
        // Not integral: the integer part followed by the fraction digits.
        let int = (x.trunc()) as u64;
        let mut out = format!("{int:x}");
        let mut f = x.fract();
        if f > 0.0 {
            out.push('.');
            while f > 0.0 {
                f *= 16.0;
                let d = f.trunc() as u32;
                out.push(std::char::from_digit(d, 16).unwrap());
                f -= d as f64;
            }
        }
        return out;
    }
    // Binary digits of mantissa followed by `exp` zero bits, grouped by four.
    let mut binary = format!("{mantissa:b}");
    binary.extend(std::iter::repeat_n('0', exp as usize));
    let pad = (4 - binary.len() % 4) % 4;
    let binary = format!("{}{binary}", "0".repeat(pad));
    let mut out = String::with_capacity(binary.len() / 4);
    for chunk in binary.as_bytes().chunks(4) {
        let nibble = chunk
            .iter()
            .fold(0u32, |acc, b| acc * 2 + (b - b'0') as u32);
        out.push(std::char::from_digit(nibble, 16).unwrap());
    }
    let trimmed = out.trim_start_matches('0');
    if trimmed.is_empty() {
        "0".into()
    } else {
        trimmed.to_owned()
    }
}

/// `QuoteJSONString(s)` (the string form `JSON.stringify` emits).
pub fn quote_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        push_json_char(&mut out, c);
    }
    out.push('"');
    out
}

fn push_json_char(out: &mut String, c: char) {
    match c {
        '\u{8}' => out.push_str("\\b"),
        '\t' => out.push_str("\\t"),
        '\n' => out.push_str("\\n"),
        '\u{c}' => out.push_str("\\f"),
        '\r' => out.push_str("\\r"),
        '"' => out.push_str("\\\""),
        '\\' => out.push_str("\\\\"),
        c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
        c => out.push(c),
    }
}

/// `QuoteJSONString` over UTF-16 code units: lone surrogates are escaped as
/// `\udXXX`, as the well-formed `JSON.stringify` does.
pub fn quote_json_utf16(units: &[u16]) -> String {
    let mut out = String::with_capacity(units.len() + 2);
    out.push('"');
    for r in char::decode_utf16(units.iter().copied()) {
        match r {
            Ok(c) => push_json_char(&mut out, c),
            Err(e) => out.push_str(&format!("\\u{:04x}", e.unpaired_surrogate())),
        }
    }
    out.push('"');
    out
}

/// `JSON.stringify(v)` (compact). `None` where `JSON.stringify` returns
/// `undefined` (`undefined` and function values).
pub fn json_stringify(v: &Value) -> Option<String> {
    let mut out = String::new();
    if write_json(v, &mut out) {
        Some(out)
    } else {
        None
    }
}

fn write_json(v: &Value, out: &mut String) -> bool {
    match v {
        Value::Undefined | Value::Function(_) => return false,
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_finite() {
                out.push_str(&number_to_string(*n));
            } else {
                out.push_str("null");
            }
        }
        Value::String(s) => out.push_str(&quote_json_string(s)),
        Value::Array(a) => {
            out.push('[');
            for (i, e) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                if !write_json(e, out) {
                    out.push_str("null");
                }
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            let mut first = true;
            for (k, e) in o.iter() {
                if matches!(e, Value::Undefined | Value::Function(_)) {
                    continue;
                }
                if !first {
                    out.push(',');
                }
                first = false;
                out.push_str(&quote_json_string(k));
                out.push(':');
                write_json(e, out);
            }
            out.push('}');
        }
    }
    true
}

/// `JSON.parse('"' + raw + '"')` over UTF-16 code units: `None` where `JSON.parse`
/// throws (an unescaped quote or control character, an invalid escape).
pub fn json_parse_string_body(raw: &[u16]) -> Option<Vec<u16>> {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;
    while i < raw.len() {
        let u = raw[i];
        if u == b'"' as u16 || u < 0x20 {
            return None;
        }
        if u != b'\\' as u16 {
            out.push(u);
            i += 1;
            continue;
        }
        let next = *raw.get(i + 1)?;
        i += 2;
        let decoded = match next {
            0x22 => 0x22,
            0x5c => 0x5c,
            0x2f => 0x2f,
            0x62 => 0x08,
            0x66 => 0x0c,
            0x6e => 0x0a,
            0x72 => 0x0d,
            0x74 => 0x09,
            0x75 => {
                let hex = raw.get(i..i + 4)?;
                let mut code = 0u16;
                for &h in hex {
                    let d = char::from_u32(h as u32)?.to_digit(16)?;
                    code = code * 16 + d as u16;
                }
                i += 4;
                code
            }
            _ => return None,
        };
        out.push(decoded);
    }
    Some(out)
}

/// A stack guard for the mutually recursive formatting paths. The reference
/// recurses without bound on some malformed inputs and dies with
/// `RangeError: Maximum call stack size exceeded`; the port reports the same
/// error instead of exhausting the native stack.
///
/// The guard measures the stack the recursion has used below its outermost
/// guarded call and refuses to go deeper than [`STACK_BUDGET`] bytes, so the
/// limit holds for any frame size (debug or release builds) on any thread whose
/// stack has that much room left. Well-formed programs nest a few levels deep.
pub struct DepthGuard;

thread_local! {
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static BASE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Stack the guarded recursion may use (half of the 2 MiB that Rust gives
/// spawned threads by default).
pub const STACK_BUDGET: usize = 1 << 20;

impl DepthGuard {
    #[inline(never)]
    pub fn enter() -> Result<DepthGuard, JsError> {
        let marker = 0u8;
        let sp = std::hint::black_box(std::ptr::addr_of!(marker)) as usize;
        DEPTH.with(|depth| {
            if depth.get() == 0 {
                BASE.with(|base| base.set(sp));
            }
            let base = BASE.with(|base| base.get());
            if base.abs_diff(sp) > STACK_BUDGET {
                return Err(stack_overflow());
            }
            depth.set(depth.get() + 1);
            Ok(DepthGuard)
        })
    }
}

impl Drop for DepthGuard {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::js;

    #[test]
    fn hex_matches_number_to_string_16() {
        for (x, s) in [
            (0.0, "0"),
            (-0.0, "0"),
            (255.0, "ff"),
            (-127.0, "-7f"),
            (510.0, "1fe"),
            (1e20, "56bc75e2d63100000"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "Infinity"),
            (2f64.powi(60), "1000000000000000"),
            (0.5, "0.8"),
        ] {
            assert_eq!(number_to_hex(x), s, "{x}");
        }
    }

    #[test]
    fn to_string_and_join() {
        let arr = js!([1, null, "a", [2, 3], undefined]);
        assert_eq!(to_string(&arr).unwrap(), "1,,a,2,3,");
        assert_eq!(to_string(&js!({"a": 1})).unwrap(), "[object Object]");
        assert!(to_string(&js!({"toString": 5})).is_err());
        assert_eq!(to_string(&Value::Number(1e21)).unwrap(), "1e+21");
    }

    #[test]
    fn inherited_members() {
        assert!(matches!(
            member(&js!({}), "constructor"),
            Value::Function(_)
        ));
        assert_eq!(member(&js!([1, 2, 3]), "length"), Value::Number(3.0));
        assert_eq!(member(&Value::from("abc"), "1"), Value::from("b"));
        assert!(member(&js!({}), "nope").is_undefined());
        assert_eq!(
            get(&Value::Undefined, "join").unwrap_err(),
            JsError::type_error("Cannot read properties of undefined (reading 'join')")
        );
    }

    #[test]
    fn json_semantics() {
        assert_eq!(
            json_stringify(&js!({"a": undefined, "b": [undefined, 1e21], "c": "q\"\u{1}"}))
                .unwrap(),
            r#"{"b":[null,1e+21],"c":"q\"\u0001"}"#
        );
        assert_eq!(json_stringify(&Value::Undefined), None);
        let units = utf16(r"a\u00e9\n\ud83d\ude00");
        let decoded = json_parse_string_body(&units).unwrap();
        assert_eq!(String::from_utf16(&decoded).unwrap(), "aé\n😀");
        assert!(json_parse_string_body(&utf16(r"\'")).is_none());
        assert!(json_parse_string_body(&utf16("a\"b")).is_none());
        assert_eq!(quote_json_utf16(&[0xd800, 0x61]), "\"\\ud800a\"");
    }

    #[test]
    fn equality() {
        assert!(strict_equals(&Value::Number(0.0), &Value::Number(-0.0)));
        assert!(!strict_equals(
            &Value::Number(f64::NAN),
            &Value::Number(f64::NAN)
        ));
        assert!(same_value_zero(
            &Value::Number(f64::NAN),
            &Value::Number(f64::NAN)
        ));
        assert!(!strict_equals(&js!({}), &js!({})));
        let _ = js::number_to_string(1.0);
    }

    #[test]
    fn repeat_limits() {
        assert_eq!(repeat(" ", 2.7).unwrap(), "  ");
        assert!(repeat(" ", 1e10).is_err());
    }
}
