//! A JavaScript value model for automation descriptors.
//!
//! The reference evaluates `osc()`, `midi()` and `audio()` descriptors that are
//! plain JavaScript objects, and its checks observe JavaScript semantics:
//! strict equality with literals, truthiness, `Number.isFinite` and
//! `Number.isInteger`, loose relational comparisons (`config.mode >= 5`),
//! property-key lookups (`this.channels[n]`), `??`, and object identity (the
//! recursion stack). [`JsValue`] carries exactly what those checks need.
//!
//! Objects and arrays are shared ([`Arc`]) and initialised once
//! ([`OnceLock`]), so descriptor graphs can share nodes and even form cycles,
//! as JavaScript object graphs can; identity is the allocation.

use std::fmt;
use std::sync::{Arc, OnceLock};

use indexmap::IndexMap;
use serde_json::Value as Json;

/// A JavaScript value.
#[derive(Clone, Default)]
pub enum JsValue {
    /// `undefined` (also an absent property).
    #[default]
    Undefined,
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A number.
    Number(f64),
    /// A string.
    String(Arc<str>),
    /// A plain object.
    Object(Arc<JsObject>),
    /// An array.
    Array(Arc<JsArray>),
    /// A function (its source text); DSL arrow functions compile to these.
    Function(Arc<str>),
}

/// A plain object: own enumerable string-keyed properties in insertion order.
#[derive(Default)]
pub struct JsObject {
    props: OnceLock<IndexMap<String, JsValue>>,
}

/// An array.
#[derive(Default)]
pub struct JsArray {
    items: OnceLock<Vec<JsValue>>,
}

static UNDEFINED: JsValue = JsValue::Undefined;

impl JsObject {
    /// An object whose properties are set later with [`JsObject::init`].
    pub fn uninit() -> Arc<JsObject> {
        Arc::new(JsObject::default())
    }

    /// Sets the properties of an object created with [`JsObject::uninit`];
    /// returns `false` when they were already set.
    pub fn init(&self, props: IndexMap<String, JsValue>) -> bool {
        self.props.set(props).is_ok()
    }

    /// Own property `key` (`undefined` when absent).
    pub fn get(&self, key: &str) -> &JsValue {
        self.props
            .get()
            .and_then(|props| props.get(key))
            .unwrap_or(&UNDEFINED)
    }

    /// Own properties in insertion order.
    pub fn props(&self) -> impl Iterator<Item = (&str, &JsValue)> {
        self.props
            .get()
            .into_iter()
            .flat_map(|props| props.iter().map(|(k, v)| (k.as_str(), v)))
    }

    /// `Object.values(obj)`: integer-like keys in ascending order first, then
    /// the other keys in insertion order.
    pub fn values_in_property_order(&self) -> Vec<&JsValue> {
        let Some(props) = self.props.get() else {
            return Vec::new();
        };
        let mut indexed: Vec<(u32, &JsValue)> = props
            .iter()
            .filter_map(|(k, v)| array_index(k).map(|i| (i, v)))
            .collect();
        indexed.sort_by_key(|(i, _)| *i);
        let mut out: Vec<&JsValue> = indexed.into_iter().map(|(_, v)| v).collect();
        out.extend(
            props
                .iter()
                .filter(|(k, _)| array_index(k).is_none())
                .map(|(_, v)| v),
        );
        out
    }
}

impl JsArray {
    /// An array whose items are set later with [`JsArray::init`].
    pub fn uninit() -> Arc<JsArray> {
        Arc::new(JsArray::default())
    }

    /// Sets the items of an array created with [`JsArray::uninit`].
    pub fn init(&self, items: Vec<JsValue>) -> bool {
        self.items.set(items).is_ok()
    }

    /// The items.
    pub fn items(&self) -> &[JsValue] {
        self.items.get().map_or(&[], |items| items.as_slice())
    }
}

/// A canonical array index (`"0"`, `"17"`, never `"01"`), which property
/// enumeration orders first.
fn array_index(key: &str) -> Option<u32> {
    if key.is_empty()
        || (key.len() > 1 && key.starts_with('0'))
        || !key.bytes().all(|b| b.is_ascii_digit())
    {
        return None;
    }
    key.parse::<u32>().ok().filter(|&i| i != u32::MAX)
}

impl JsValue {
    /// A number.
    pub fn number(x: f64) -> Self {
        JsValue::Number(x)
    }

    /// A string.
    pub fn string(s: &str) -> Self {
        JsValue::String(Arc::from(s))
    }

    /// An object with the given properties.
    pub fn object<K: Into<String>>(props: impl IntoIterator<Item = (K, JsValue)>) -> Self {
        let object = JsObject::uninit();
        object.init(props.into_iter().map(|(k, v)| (k.into(), v)).collect());
        JsValue::Object(object)
    }

    /// An array.
    pub fn array(items: impl IntoIterator<Item = JsValue>) -> Self {
        let array = JsArray::uninit();
        array.init(items.into_iter().collect());
        JsValue::Array(array)
    }

    /// `value.key` for an object; `undefined` for everything else (property
    /// reads on primitives and arrays find no descriptor fields).
    pub fn get(&self, key: &str) -> &JsValue {
        match self {
            JsValue::Object(object) => object.get(key),
            _ => &UNDEFINED,
        }
    }

    /// `value === undefined`.
    pub fn is_undefined(&self) -> bool {
        matches!(self, JsValue::Undefined)
    }

    /// `value === null || value === undefined` (the `??` test).
    pub fn is_nullish(&self) -> bool {
        matches!(self, JsValue::Undefined | JsValue::Null)
    }

    /// `typeof value === 'object' && value !== null` (arrays included).
    pub fn is_object_like(&self) -> bool {
        matches!(self, JsValue::Object(_) | JsValue::Array(_))
    }

    /// `Boolean(value)`.
    pub fn truthy(&self) -> bool {
        match self {
            JsValue::Undefined | JsValue::Null => false,
            JsValue::Bool(b) => *b,
            JsValue::Number(n) => !(*n == 0.0 || n.is_nan()),
            JsValue::String(s) => !s.is_empty(),
            JsValue::Object(_) | JsValue::Array(_) | JsValue::Function(_) => true,
        }
    }

    /// The number, when the value is one.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            JsValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// The string, when the value is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// `Number.isFinite(value)`.
    pub fn is_finite_number(&self) -> bool {
        self.as_number().is_some_and(f64::is_finite)
    }

    /// `Number.isInteger(value)`.
    pub fn is_integer(&self) -> bool {
        self.as_number().is_some_and(crate::jsmath::is_integer)
    }

    /// `value === n` for a number literal `n`.
    pub fn strict_eq_number(&self, n: f64) -> bool {
        self.as_number() == Some(n)
    }

    /// `value === s` for a string literal `s`.
    pub fn strict_eq_str(&self, s: &str) -> bool {
        self.as_str() == Some(s)
    }

    /// Identity of an object or array (`Set.has` uses it).
    pub fn identity(&self) -> Option<usize> {
        match self {
            JsValue::Object(object) => Some(Arc::as_ptr(object) as *const () as usize),
            JsValue::Array(array) => Some(Arc::as_ptr(array) as *const () as usize),
            _ => None,
        }
    }

    /// `ToNumber(ToPrimitive(value, number))`, as relational comparisons with
    /// a number apply it.
    pub fn to_number(&self) -> f64 {
        match self {
            JsValue::Undefined => f64::NAN,
            JsValue::Null => 0.0,
            JsValue::Bool(b) => f64::from(u8::from(*b)),
            JsValue::Number(n) => *n,
            JsValue::String(s) => string_to_number(s),
            // OrdinaryToPrimitive: "[object Object]" or the source text.
            JsValue::Object(_) | JsValue::Function(_) => f64::NAN,
            JsValue::Array(_) => match self.array_join(0) {
                Some(text) => string_to_number(&text),
                None => f64::NAN,
            },
        }
    }

    /// `String(array)` for arrays whose join is needed numerically: elements
    /// joined with commas, `null`/`undefined` as empty. Returns `None` when an
    /// element has no simple text (objects, functions, non-integral numbers in
    /// multi-element arrays), which never parses as a number anyway.
    fn array_join(&self, depth: usize) -> Option<String> {
        let JsValue::Array(array) = self else {
            return None;
        };
        if depth > 64 {
            // V8 joins a cyclic reference as the empty string.
            return Some(String::new());
        }
        let items = array.items();
        if items.len() > 1 {
            // A comma never parses as a number.
            return Some(",".to_string());
        }
        match items.first() {
            None | Some(JsValue::Undefined) | Some(JsValue::Null) => Some(String::new()),
            Some(JsValue::String(s)) => Some(s.to_string()),
            Some(JsValue::Number(n)) => Some(number_text(*n)),
            Some(JsValue::Bool(b)) => Some(b.to_string()),
            Some(item @ JsValue::Array(_)) => item.array_join(depth + 1),
            Some(JsValue::Object(_)) => Some("[object Object]".to_string()),
            Some(JsValue::Function(source)) => Some(source.to_string()),
        }
    }

    /// The channel number `this.channels[value]` finds in a `MidiState`
    /// (`ToPropertyKey(value)` is one of `"1"`..`"16"`), or `None`.
    pub fn channel_key(&self) -> Option<u8> {
        let key = match self {
            JsValue::Number(n) => {
                return (crate::jsmath::is_integer(*n) && (1.0..=16.0).contains(n))
                    .then_some(*n as u8);
            }
            JsValue::String(s) => s.to_string(),
            JsValue::Array(_) => self.array_join(0)?,
            _ => return None,
        };
        match key.as_str() {
            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "10" | "11" | "12" | "13"
            | "14" | "15" | "16" => key.parse().ok(),
            _ => None,
        }
    }

    /// Converts JSON into a value. Conventions: `{"$num": "NaN" | "Infinity" |
    /// "-Infinity" | "-0" | "<decimal>"}` for numbers JSON cannot carry (or
    /// carry exactly through a fast float parser), `{"$undefined":
    /// true}`, `{"$function": "source"}`, and object identity with `{"$id":
    /// name, ...}` / `{"$ref": name}` (references may point to enclosing
    /// objects, forming cycles).
    pub fn from_json(json: &Json) -> Result<JsValue, String> {
        let mut ids: IndexMap<String, JsValue> = IndexMap::new();
        collect_ids(json, &mut ids)?;
        let value = build(json, &ids)?;
        Ok(value)
    }
}

/// JavaScript `Number.prototype.toString()` for the values array joins need:
/// integers print as integers; anything else gets the shortest round-trip
/// text (which parses back to the same number).
fn number_text(n: f64) -> String {
    if n.is_nan() {
        "NaN".to_string()
    } else if n.is_infinite() {
        if n > 0.0 { "Infinity" } else { "-Infinity" }.to_string()
    } else if n == 0.0 {
        "0".to_string()
    } else if n == n.trunc() && n.abs() < 1e21 {
        format!("{n:.0}")
    } else {
        format!("{n:e}")
    }
}

/// JavaScript `StringToNumber`: whitespace-trimmed decimal literals (with
/// optional sign, fraction and exponent), `Infinity`, `0x`/`0o`/`0b`
/// integers; empty is 0; anything else is NaN.
pub fn string_to_number(s: &str) -> f64 {
    let trimmed = s.trim_matches(is_js_whitespace);
    if trimmed.is_empty() {
        return 0.0;
    }
    let bytes = trimmed.as_bytes();
    if bytes.len() > 2 && bytes[0] == b'0' {
        let radix = match bytes[1] {
            b'x' | b'X' => Some(16),
            b'o' | b'O' => Some(8),
            b'b' | b'B' => Some(2),
            _ => None,
        };
        if let Some(radix) = radix {
            return parse_radix(&trimmed[2..], radix);
        }
    }
    let (sign, unsigned) = match bytes[0] {
        b'+' => (1.0, &trimmed[1..]),
        b'-' => (-1.0, &trimmed[1..]),
        _ => (1.0, trimmed),
    };
    if unsigned == "Infinity" {
        return sign * f64::INFINITY;
    }
    if !is_decimal_literal(unsigned) {
        return f64::NAN;
    }
    match unsigned.parse::<f64>() {
        Ok(value) => sign * value,
        Err(_) => f64::NAN,
    }
}

fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// `StrUnsignedDecimalLiteral`: digits [. digits] or . digits, then an
/// optional exponent.
fn is_decimal_literal(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let int_digits = i - int_start;
    let mut frac_digits = 0;
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        frac_digits = i - frac_start;
    }
    if int_digits == 0 && frac_digits == 0 {
        return false;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        i += 1;
        if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == b.len()
}

/// A `0x`/`0o`/`0b` literal's value, rounded to the nearest double like the
/// mathematical value of the digits.
fn parse_radix(digits: &str, radix: u32) -> f64 {
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return f64::NAN;
    }
    let bits_per_digit = match radix {
        16 => 4,
        8 => 3,
        _ => 1,
    };
    // Exact binary expansion, then round-half-even to 53 significant bits.
    let mut bits: Vec<u8> = Vec::new();
    for c in digits.chars() {
        let d = c.to_digit(radix).expect("checked digit");
        for shift in (0..bits_per_digit).rev() {
            bits.push(((d >> shift) & 1) as u8);
        }
    }
    let first = match bits.iter().position(|&b| b == 1) {
        Some(first) => first,
        None => return 0.0,
    };
    let bits = &bits[first..];
    if bits.len() <= 53 {
        return bits.iter().fold(0.0, |acc, &b| acc * 2.0 + f64::from(b));
    }
    let mut mantissa: u64 = bits[..53]
        .iter()
        .fold(0u64, |acc, &b| (acc << 1) | u64::from(b));
    let round = bits[53] == 1;
    let sticky = bits[54..].contains(&1);
    if round && (sticky || mantissa & 1 == 1) {
        mantissa += 1;
    }
    let exponent = (bits.len() - 53) as i32;
    (mantissa as f64) * 2f64.powi(exponent)
}

fn collect_ids(json: &Json, ids: &mut IndexMap<String, JsValue>) -> Result<(), String> {
    match json {
        Json::Object(map) => {
            if let Some(id) = map.get("$id") {
                let id = id.as_str().ok_or("$id must be a string")?.to_string();
                let value = if map.contains_key("$array") {
                    JsValue::Array(JsArray::uninit())
                } else {
                    JsValue::Object(JsObject::uninit())
                };
                if ids.insert(id.clone(), value).is_some() {
                    return Err(format!("duplicate $id {id}"));
                }
            }
            for value in map.values() {
                collect_ids(value, ids)?;
            }
        }
        Json::Array(items) => {
            for item in items {
                collect_ids(item, ids)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn build(json: &Json, ids: &IndexMap<String, JsValue>) -> Result<JsValue, String> {
    Ok(match json {
        Json::Null => JsValue::Null,
        Json::Bool(b) => JsValue::Bool(*b),
        Json::Number(n) => JsValue::Number(n.as_f64().ok_or("number out of range")?),
        Json::String(s) => JsValue::string(s),
        Json::Array(items) => JsValue::array(
            items
                .iter()
                .map(|i| build(i, ids))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Json::Object(map) => {
            if let Some(reference) = map.get("$ref") {
                let id = reference.as_str().ok_or("$ref must be a string")?;
                return ids
                    .get(id)
                    .cloned()
                    .ok_or_else(|| format!("unknown $ref {id}"));
            }
            if let Some(tag) = map.get("$num") {
                return Ok(JsValue::Number(match tag.as_str() {
                    Some("NaN") => f64::NAN,
                    Some("Infinity") => f64::INFINITY,
                    Some("-Infinity") => f64::NEG_INFINITY,
                    Some("-0") => -0.0,
                    Some(decimal) => decimal
                        .parse::<f64>()
                        .map_err(|_| format!("bad $num {tag}"))?,
                    None => return Err(format!("bad $num {tag}")),
                }));
            }
            if map.contains_key("$undefined") {
                return Ok(JsValue::Undefined);
            }
            if let Some(source) = map.get("$function") {
                return Ok(JsValue::Function(Arc::from(source.as_str().unwrap_or(""))));
            }
            if let Some(id) = map.get("$id").and_then(Json::as_str) {
                let target = ids[id].clone();
                match &target {
                    JsValue::Array(array) => {
                        let items = map
                            .get("$array")
                            .and_then(Json::as_array)
                            .ok_or("$array must be an array")?;
                        array.init(
                            items
                                .iter()
                                .map(|i| build(i, ids))
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                    }
                    JsValue::Object(object) => {
                        object.init(build_props(map, ids)?);
                    }
                    _ => unreachable!(),
                }
                return Ok(target);
            }
            let object = JsObject::uninit();
            object.init(build_props(map, ids)?);
            JsValue::Object(object)
        }
    })
}

fn build_props(
    map: &serde_json::Map<String, Json>,
    ids: &IndexMap<String, JsValue>,
) -> Result<IndexMap<String, JsValue>, String> {
    let mut props = IndexMap::new();
    for (key, value) in map {
        if key == "$id" {
            continue;
        }
        props.insert(key.clone(), build(value, ids)?);
    }
    Ok(props)
}

impl fmt::Debug for JsValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_depth(f, 0)
    }
}

impl JsValue {
    fn fmt_depth(&self, f: &mut fmt::Formatter<'_>, depth: usize) -> fmt::Result {
        if depth > 8 {
            return write!(f, "…");
        }
        match self {
            JsValue::Undefined => write!(f, "undefined"),
            JsValue::Null => write!(f, "null"),
            JsValue::Bool(b) => write!(f, "{b}"),
            JsValue::Number(n) => write!(f, "{n}"),
            JsValue::String(s) => write!(f, "{s:?}"),
            JsValue::Function(_) => write!(f, "[function]"),
            JsValue::Array(array) => {
                write!(f, "[")?;
                for (i, item) in array.items().iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    item.fmt_depth(f, depth + 1)?;
                }
                write!(f, "]")
            }
            JsValue::Object(object) => {
                write!(f, "{{")?;
                for (i, (k, v)) in object.props().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: ")?;
                    v.fmt_depth(f, depth + 1)?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_to_number_follows_the_spec() {
        assert_eq!(string_to_number(""), 0.0);
        assert_eq!(string_to_number("  12 \n"), 12.0);
        assert_eq!(string_to_number("-1.5e3"), -1500.0);
        assert_eq!(string_to_number(".5"), 0.5);
        assert_eq!(string_to_number("5."), 5.0);
        assert_eq!(string_to_number("0x1F"), 31.0);
        assert_eq!(string_to_number("0b101"), 5.0);
        assert_eq!(string_to_number("0o17"), 15.0);
        assert_eq!(string_to_number("-Infinity"), f64::NEG_INFINITY);
        for nan in [
            "1e", "abc", "1_000", "-0x10", "inf", "NaN", "0x", ".", "+-1", "1 2",
        ] {
            assert!(string_to_number(nan).is_nan(), "{nan}");
        }
        assert_eq!(string_to_number("0x20000000000001"), 9007199254740992.0);
        assert_eq!(string_to_number("0x20000000000003"), 9007199254740996.0);
    }

    #[test]
    fn to_number_of_arrays_joins() {
        let arr = |items: Vec<JsValue>| JsValue::array(items);
        assert_eq!(arr(vec![]).to_number(), 0.0);
        assert_eq!(arr(vec![JsValue::Number(3.0)]).to_number(), 3.0);
        assert_eq!(arr(vec![JsValue::string(" 7 ")]).to_number(), 7.0);
        assert!(
            arr(vec![JsValue::Number(1.0), JsValue::Number(2.0)])
                .to_number()
                .is_nan()
        );
        assert_eq!(arr(vec![arr(vec![JsValue::Number(4.5)])]).to_number(), 4.5);
        assert_eq!(arr(vec![JsValue::Null]).to_number(), 0.0);
        assert!(arr(vec![JsValue::Bool(true)]).to_number().is_nan());
    }

    #[test]
    fn channel_keys_are_property_keys() {
        assert_eq!(JsValue::Number(5.0).channel_key(), Some(5));
        assert_eq!(JsValue::Number(1.5).channel_key(), None);
        assert_eq!(JsValue::Number(-0.0).channel_key(), None);
        assert_eq!(JsValue::string("16").channel_key(), Some(16));
        assert_eq!(JsValue::string("05").channel_key(), None);
        assert_eq!(
            JsValue::array([JsValue::Number(3.0)]).channel_key(),
            Some(3)
        );
        assert_eq!(JsValue::Bool(true).channel_key(), None);
    }

    #[test]
    fn json_round_trip_supports_identity_and_cycles() {
        let json: Json = serde_json::from_str(
            r#"{"$id": "a", "type": "Oscillator", "speed": {"$ref": "a"}, "min": {"$num": "-0"}, "shared": [{"$id": "b"}, {"$ref": "b"}]}"#,
        )
        .unwrap();
        let value = JsValue::from_json(&json).unwrap();
        assert_eq!(value.get("speed").identity(), value.identity());
        assert!(value.get("min").as_number().unwrap().is_sign_negative());
        let JsValue::Array(shared) = value.get("shared") else {
            panic!()
        };
        assert_eq!(shared.items()[0].identity(), shared.items()[1].identity());
    }

    #[test]
    fn object_values_order_integer_keys_first() {
        let object = JsValue::object([
            ("b", JsValue::Number(1.0)),
            ("2", JsValue::Number(2.0)),
            ("a", JsValue::Number(3.0)),
            ("0", JsValue::Number(4.0)),
        ]);
        let JsValue::Object(object) = object else {
            panic!()
        };
        let values: Vec<f64> = object
            .values_in_property_order()
            .iter()
            .map(|v| v.as_number().unwrap())
            .collect();
        assert_eq!(values, vec![4.0, 2.0, 1.0, 3.0]);
    }
}
