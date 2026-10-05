//! A dynamic value with JavaScript data semantics.
//!
//! The reference frontend manipulates plain JavaScript objects. Porting it with
//! the same data model keeps every stage diffable against the reference dumps:
//!
//! * numbers are IEEE doubles (`Number::isInteger` is a property of the value);
//! * `undefined` and `null` are distinct, and an object member may hold `undefined`
//!   (it is still a key, as `in` and `Object.keys` see it);
//! * object members iterate in JavaScript property order: array-index keys first in
//!   ascending numeric order, then the other keys in insertion order;
//! * serialization follows `JSON.stringify`: `undefined` members are omitted,
//!   `undefined` array elements and non-finite numbers become `null`, and functions
//!   (represented by [`Value::Function`]) are omitted like `undefined`.

use std::fmt;

use indexmap::IndexMap;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};

/// An ordered object with JavaScript property-order semantics.
///
/// Members are enumerable unless defined with [`Object::define_hidden`], which
/// models `Object.defineProperty(obj, key, {value, writable: true, enumerable:
/// false})`: a hidden member reads, writes and answers `in` like any member but is
/// left out of `Object.keys`/`entries`, spreads ([`Object::assign`]) and
/// serialization.
#[derive(Clone, Default, PartialEq)]
pub struct Object {
    map: IndexMap<String, Value>,
    hidden: Option<Box<IndexMap<String, Value>>>,
}

/// A JavaScript-like dynamic value.
#[derive(Clone, Default, PartialEq)]
pub enum Value {
    #[default]
    Undefined,
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Object),
    /// A JavaScript function value. The reference stores closures in a few places
    /// (DSL arrow functions compile to `{fn: state => ...}`); they never serialize,
    /// and the source text is kept for diagnostics.
    Function(String),
}

/// `true` when `key` is a canonical array index (`"0"`, `"17"`, never `"01"`),
/// which JavaScript orders before string keys.
pub fn is_array_index(key: &str) -> bool {
    if key.is_empty() || key.len() > 10 {
        return false;
    }
    if key == "0" {
        return true;
    }
    let bytes = key.as_bytes();
    if bytes[0] == b'0' || !bytes.iter().all(u8::is_ascii_digit) {
        return false;
    }
    key.parse::<u64>().is_ok_and(|n| n < u32::MAX as u64)
}

impl Object {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// The member value, or `None` when the key is absent (an absent key reads as
    /// `undefined` in JavaScript; see [`Object::get_or_undefined`]).
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map
            .get(key)
            .or_else(|| self.hidden.as_ref().and_then(|h| h.get(key)))
    }

    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        if self.map.contains_key(key) {
            return self.map.get_mut(key);
        }
        self.hidden.as_mut().and_then(|h| h.get_mut(key))
    }

    /// `obj[key]` in JavaScript: absent keys read as `undefined`.
    pub fn get_or_undefined(&self, key: &str) -> &Value {
        self.get(key).unwrap_or(&UNDEFINED)
    }

    /// `key in obj`.
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key) || self.hidden.as_ref().is_some_and(|h| h.contains_key(key))
    }

    /// `Object.defineProperty(obj, key, {value, writable: true, enumerable: false})`:
    /// define (or redefine) `key` as a non-enumerable member.
    pub fn define_hidden(&mut self, key: impl Into<String>, value: Value) {
        let key = key.into();
        self.map.shift_remove(&key);
        self.hidden.get_or_insert_with(Default::default).insert(key, value);
    }

    /// `true` when `key` is a non-enumerable member.
    pub fn is_hidden(&self, key: &str) -> bool {
        self.hidden.as_ref().is_some_and(|h| h.contains_key(key))
    }

    /// The non-enumerable members, in definition order
    /// (`Object.getOwnPropertyNames` minus `Object.keys`).
    pub fn hidden_members(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.hidden.iter().flat_map(|h| h.iter())
    }

    /// `obj[key] = value`, keeping JavaScript property order: an existing key keeps
    /// its position, a new array-index key is placed among the other index keys in
    /// ascending order, and a new string key is appended.
    pub fn insert(&mut self, key: impl Into<String>, value: Value) -> Option<Value> {
        let key = key.into();
        if let Some(slot) = self.map.get_mut(&key) {
            return Some(std::mem::replace(slot, value));
        }
        if let Some(slot) = self.hidden.as_mut().and_then(|h| h.get_mut(&key)) {
            return Some(std::mem::replace(slot, value));
        }
        if is_array_index(&key) {
            let n: u64 = key.parse().unwrap();
            let pos = self
                .map
                .keys()
                .position(|k| !is_array_index(k) || k.parse::<u64>().unwrap() > n)
                .unwrap_or(self.map.len());
            self.map.shift_insert(pos, key, value);
        } else {
            self.map.insert(key, value);
        }
        None
    }

    /// `delete obj[key]`, preserving the order of the remaining members.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.map
            .shift_remove(key)
            .or_else(|| self.hidden.as_mut().and_then(|h| h.shift_remove(key)))
    }

    /// Members in JavaScript property order (`Object.entries`).
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&String, &Value)> + ExactSizeIterator {
        self.map.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut Value)> {
        self.map.iter_mut()
    }

    /// `Object.keys(obj)`.
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &String> + ExactSizeIterator {
        self.map.keys()
    }

    /// `Object.values(obj)`.
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &Value> + ExactSizeIterator {
        self.map.values()
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut Value> {
        self.map.values_mut()
    }

    /// `obj[key] = obj[key] || {}`: the member object, created empty when the
    /// member is absent or not an object.
    pub fn object_entry(&mut self, key: &str) -> &mut Object {
        if !matches!(self.map.get(key), Some(Value::Object(_))) {
            self.insert(key.to_owned(), Value::object());
        }
        match self.map.get_mut(key) {
            Some(Value::Object(o)) => o,
            _ => unreachable!(),
        }
    }

    /// `{ ...a, ...b }` style merge: members of `other` assigned onto a copy of self.
    pub fn assign(&mut self, other: &Object) {
        for (k, v) in other.iter() {
            self.insert(k.clone(), v.clone());
        }
    }
}

impl FromIterator<(String, Value)> for Object {
    fn from_iter<T: IntoIterator<Item = (String, Value)>>(iter: T) -> Self {
        let mut obj = Object::new();
        for (k, v) in iter {
            obj.insert(k, v);
        }
        obj
    }
}

impl<'a> IntoIterator for &'a Object {
    type Item = (&'a String, &'a Value);
    type IntoIter = indexmap::map::Iter<'a, String, Value>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.iter()
    }
}

impl IntoIterator for Object {
    type Item = (String, Value);
    type IntoIter = indexmap::map::IntoIter<String, Value>;
    fn into_iter(self) -> Self::IntoIter {
        self.map.into_iter()
    }
}

static UNDEFINED: Value = Value::Undefined;

impl Value {
    pub fn object() -> Value {
        Value::Object(Object::new())
    }

    pub fn array() -> Value {
        Value::Array(Vec::new())
    }

    pub fn is_undefined(&self) -> bool {
        matches!(self, Value::Undefined)
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// `value === undefined || value === null` (`value == null` in JavaScript).
    pub fn is_nullish(&self) -> bool {
        matches!(self, Value::Undefined | Value::Null)
    }

    /// JavaScript truthiness (`!!value`).
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Undefined | Value::Null => false,
            Value::Bool(b) => *b,
            Value::Number(n) => *n != 0.0 && !n.is_nan(),
            Value::String(s) => !s.is_empty(),
            Value::Array(_) | Value::Object(_) | Value::Function(_) => true,
        }
    }

    /// `typeof value`.
    pub fn type_of(&self) -> &'static str {
        match self {
            Value::Undefined => "undefined",
            Value::Null | Value::Array(_) | Value::Object(_) => "object",
            Value::Bool(_) => "boolean",
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Function(_) => "function",
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&Object> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn as_object_mut(&mut self) -> Option<&mut Object> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// `Number.isInteger(value)`.
    pub fn is_integer(&self) -> bool {
        matches!(self, Value::Number(n) if n.is_finite() && n.fract() == 0.0)
    }

    /// `value[key]` for objects (absent and non-objects read as `undefined`).
    pub fn get(&self, key: &str) -> &Value {
        match self {
            Value::Object(o) => o.get_or_undefined(key),
            Value::Array(a) => key
                .parse::<usize>()
                .ok()
                .filter(|_| is_array_index(key))
                .and_then(|i| a.get(i))
                .unwrap_or(&UNDEFINED),
            _ => &UNDEFINED,
        }
    }

    /// Mutable member access for objects.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        match self {
            Value::Object(o) => o.get_mut(key),
            _ => None,
        }
    }

    /// `value[index]` for arrays.
    pub fn at(&self, index: usize) -> &Value {
        match self {
            Value::Array(a) => a.get(index).unwrap_or(&UNDEFINED),
            _ => &UNDEFINED,
        }
    }

    /// `value[key] = v` for objects; panics on non-objects, like a TypeError would.
    pub fn set(&mut self, key: impl Into<String>, v: Value) {
        match self {
            Value::Object(o) => {
                o.insert(key, v);
            }
            other => panic!("cannot set a property on {}", other.type_of()),
        }
    }

    /// `JSON.stringify(value)` (compact). Returns `None` for top-level `undefined`
    /// or function values, as `JSON.stringify` returns `undefined` for them.
    pub fn to_json(&self) -> Option<String> {
        if matches!(self, Value::Undefined | Value::Function(_)) {
            return None;
        }
        Some(serde_json::to_string(self).expect("Value serialization cannot fail"))
    }

    /// `JSON.stringify(value, null, 2)`.
    pub fn to_json_pretty(&self) -> Option<String> {
        if matches!(self, Value::Undefined | Value::Function(_)) {
            return None;
        }
        Some(serde_json::to_string_pretty(self).expect("Value serialization cannot fail"))
    }

    /// `JSON.parse(text)`. Numbers are correctly rounded, as `JSON.parse` rounds
    /// them (serde_json's default float parsing can be one unit in the last place
    /// off for 17-digit decimals such as `0.9372549019607843`).
    pub fn from_json(text: &str) -> Result<Value, serde_json::Error> {
        json_parse::parse(text).map_err(<serde_json::Error as de::Error>::custom)
    }
}

// --- JSON.parse ----------------------------------------------------------------

/// A JSON text parser with `JSON.parse` semantics: correctly rounded numbers
/// (`-0` stays negative zero), later duplicate keys overwrite earlier ones in
/// place, and lone surrogate escapes (which a Rust string cannot hold) become
/// U+FFFD.
mod json_parse {
    use super::{Object, Value};

    /// Nesting limit (guards the recursive descent against stack exhaustion).
    const MAX_DEPTH: usize = 1024;

    struct Parser<'a> {
        text: &'a str,
        bytes: &'a [u8],
        pos: usize,
        depth: usize,
    }

    pub(super) fn parse(text: &str) -> Result<Value, String> {
        let mut p = Parser {
            text,
            bytes: text.as_bytes(),
            pos: 0,
            depth: 0,
        };
        p.skip_ws();
        let value = p.value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(p.error("trailing characters"));
        }
        Ok(value)
    }

    impl Parser<'_> {
        fn error(&self, msg: &str) -> String {
            let consumed = &self.text[..self.pos];
            let line = consumed.matches('\n').count() + 1;
            let column = consumed.len() - consumed.rfind('\n').map_or(0, |i| i + 1) + 1;
            format!("{msg} at line {line} column {column}")
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn skip_ws(&mut self) {
            while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.peek() {
                self.pos += 1;
            }
        }

        fn expect(&mut self, byte: u8, msg: &str) -> Result<(), String> {
            if self.peek() == Some(byte) {
                self.pos += 1;
                Ok(())
            } else {
                Err(self.error(msg))
            }
        }

        fn value(&mut self) -> Result<Value, String> {
            match self.peek() {
                None => Err(self.error("EOF while parsing a value")),
                Some(b'{') => self.nested(Self::object),
                Some(b'[') => self.nested(Self::array),
                Some(b'"') => Ok(Value::String(self.string()?)),
                Some(b't') => self.literal("true", Value::Bool(true)),
                Some(b'f') => self.literal("false", Value::Bool(false)),
                Some(b'n') => self.literal("null", Value::Null),
                Some(b'-' | b'0'..=b'9') => self.number(),
                Some(_) => Err(self.error("expected value")),
            }
        }

        fn nested(&mut self, f: fn(&mut Self) -> Result<Value, String>) -> Result<Value, String> {
            if self.depth >= MAX_DEPTH {
                return Err(self.error("recursion limit exceeded"));
            }
            self.depth += 1;
            let result = f(self);
            self.depth -= 1;
            result
        }

        fn literal(&mut self, word: &str, value: Value) -> Result<Value, String> {
            if self.text[self.pos..].starts_with(word) {
                self.pos += word.len();
                Ok(value)
            } else {
                Err(self.error("expected value"))
            }
        }

        fn digits(&mut self) -> usize {
            let start = self.pos;
            while let Some(b'0'..=b'9') = self.peek() {
                self.pos += 1;
            }
            self.pos - start
        }

        fn number(&mut self) -> Result<Value, String> {
            let start = self.pos;
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            match self.peek() {
                Some(b'0') => self.pos += 1,
                Some(b'1'..=b'9') => {
                    self.digits();
                }
                _ => return Err(self.error("invalid number")),
            }
            if self.peek() == Some(b'.') {
                self.pos += 1;
                if self.digits() == 0 {
                    return Err(self.error("invalid number"));
                }
            }
            if let Some(b'e' | b'E') = self.peek() {
                self.pos += 1;
                if let Some(b'+' | b'-') = self.peek() {
                    self.pos += 1;
                }
                if self.digits() == 0 {
                    return Err(self.error("invalid number"));
                }
            }
            // Rust's decimal-to-double conversion is correctly rounded.
            self.text[start..self.pos]
                .parse::<f64>()
                .map(Value::Number)
                .map_err(|_| self.error("invalid number"))
        }

        fn hex4(&mut self) -> Result<u32, String> {
            let hex = self
                .text
                .get(self.pos..self.pos + 4)
                .filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()))
                .ok_or_else(|| self.error("invalid escape"))?;
            self.pos += 4;
            Ok(u32::from_str_radix(hex, 16).expect("four hex digits"))
        }

        fn string(&mut self) -> Result<String, String> {
            self.pos += 1; // the opening quote
            let mut out = String::new();
            loop {
                let run = self.pos;
                while let Some(b) = self.peek() {
                    if b == b'"' || b == b'\\' || b < 0x20 {
                        break;
                    }
                    self.pos += 1;
                }
                out.push_str(&self.text[run..self.pos]);
                match self.peek() {
                    None => return Err(self.error("EOF while parsing a string")),
                    Some(b'"') => {
                        self.pos += 1;
                        return Ok(out);
                    }
                    Some(b'\\') => {
                        self.pos += 1;
                        let escape = self.peek();
                        self.pos += 1;
                        match escape {
                            Some(b'"') => out.push('"'),
                            Some(b'\\') => out.push('\\'),
                            Some(b'/') => out.push('/'),
                            Some(b'b') => out.push('\u{8}'),
                            Some(b'f') => out.push('\u{c}'),
                            Some(b'n') => out.push('\n'),
                            Some(b'r') => out.push('\r'),
                            Some(b't') => out.push('\t'),
                            Some(b'u') => {
                                let unit = self.hex4()?;
                                if (0xD800..0xDC00).contains(&unit)
                                    && self.text[self.pos..].starts_with("\\u")
                                {
                                    let save = self.pos;
                                    self.pos += 2;
                                    let low = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&low) {
                                        let c = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                                        out.push(char::from_u32(c).expect("valid surrogate pair"));
                                    } else {
                                        out.push('\u{FFFD}');
                                        self.pos = save;
                                    }
                                } else {
                                    out.push(char::from_u32(unit).unwrap_or('\u{FFFD}'));
                                }
                            }
                            _ => {
                                self.pos -= 1;
                                return Err(self.error("invalid escape"));
                            }
                        }
                    }
                    Some(_) => return Err(self.error("control character found while parsing a string")),
                }
            }
        }

        fn array(&mut self) -> Result<Value, String> {
            self.pos += 1; // [
            let mut out = Vec::new();
            self.skip_ws();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(Value::Array(out));
            }
            loop {
                self.skip_ws();
                out.push(self.value()?);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(Value::Array(out));
                    }
                    _ => return Err(self.error("expected `,` or `]`")),
                }
            }
        }

        fn object(&mut self) -> Result<Value, String> {
            self.pos += 1; // {
            let mut out = Object::new();
            self.skip_ws();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(Value::Object(out));
            }
            loop {
                self.skip_ws();
                if self.peek() != Some(b'"') {
                    return Err(self.error("key must be a string"));
                }
                let key = self.string()?;
                self.skip_ws();
                self.expect(b':', "expected `:`")?;
                self.skip_ws();
                let value = self.value()?;
                out.insert(key, value);
                self.skip_ws();
                match self.peek() {
                    Some(b',') => self.pos += 1,
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(Value::Object(out));
                    }
                    _ => return Err(self.error("expected `,` or `}`")),
                }
            }
        }
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}
impl From<f64> for Value {
    fn from(n: f64) -> Self {
        Value::Number(n)
    }
}
impl From<i32> for Value {
    fn from(n: i32) -> Self {
        Value::Number(n as f64)
    }
}
impl From<i64> for Value {
    fn from(n: i64) -> Self {
        Value::Number(n as f64)
    }
}
impl From<u32> for Value {
    fn from(n: u32) -> Self {
        Value::Number(n as f64)
    }
}
impl From<usize> for Value {
    fn from(n: usize) -> Self {
        Value::Number(n as f64)
    }
}
impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::String(s.to_owned())
    }
}
impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::String(s)
    }
}
impl From<Vec<Value>> for Value {
    fn from(a: Vec<Value>) -> Self {
        Value::Array(a)
    }
}
impl From<Object> for Value {
    fn from(o: Object) -> Self {
        Value::Object(o)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Undefined => write!(f, "undefined"),
            Value::Function(src) => write!(f, "[Function {src}]"),
            other => write!(f, "{}", other.to_json().unwrap_or_default()),
        }
    }
}

impl fmt::Debug for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.map.iter()).finish()
    }
}

// --- serde -------------------------------------------------------------------

/// Serialize a number the way JSON.stringify spells it: integral values without a
/// fraction, non-finite values as `null`.
fn serialize_number<S: Serializer>(n: f64, s: S) -> Result<S::Ok, S::Error> {
    if !n.is_finite() {
        s.serialize_unit()
    } else if n.fract() == 0.0 && n.abs() < 9_007_199_254_740_992.0 {
        if n == 0.0 {
            // -0 stringifies as 0.
            s.serialize_i64(0)
        } else {
            s.serialize_i64(n as i64)
        }
    } else {
        s.serialize_f64(n)
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Value::Undefined | Value::Null | Value::Function(_) => s.serialize_unit(),
            Value::Bool(b) => s.serialize_bool(*b),
            Value::Number(n) => serialize_number(*n, s),
            Value::String(st) => s.serialize_str(st),
            Value::Array(a) => {
                let mut seq = s.serialize_seq(Some(a.len()))?;
                for v in a {
                    seq.serialize_element(v)?;
                }
                seq.end()
            }
            Value::Object(o) => o.serialize(s),
        }
    }
}

impl Serialize for Object {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let members: Vec<_> = self
            .map
            .iter()
            .filter(|(_, v)| !matches!(v, Value::Undefined | Value::Function(_)))
            .collect();
        let mut map = s.serialize_map(Some(members.len()))?;
        for (k, v) in members {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Value;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_bool<E: de::Error>(self, b: bool) -> Result<Value, E> {
                Ok(Value::Bool(b))
            }
            fn visit_i64<E: de::Error>(self, n: i64) -> Result<Value, E> {
                Ok(Value::Number(n as f64))
            }
            fn visit_u64<E: de::Error>(self, n: u64) -> Result<Value, E> {
                Ok(Value::Number(n as f64))
            }
            fn visit_f64<E: de::Error>(self, n: f64) -> Result<Value, E> {
                Ok(Value::Number(n))
            }
            fn visit_str<E: de::Error>(self, s: &str) -> Result<Value, E> {
                Ok(Value::String(s.to_owned()))
            }
            fn visit_string<E: de::Error>(self, s: String) -> Result<Value, E> {
                Ok(Value::String(s))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
                Ok(Value::Null)
            }
            fn visit_none<E: de::Error>(self) -> Result<Value, E> {
                Ok(Value::Null)
            }
            fn visit_some<D2: Deserializer<'de>>(self, d: D2) -> Result<Value, D2::Error> {
                Value::deserialize(d)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
                let mut out = Vec::new();
                while let Some(v) = seq.next_element()? {
                    out.push(v);
                }
                Ok(Value::Array(out))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
                let mut out = Object::new();
                while let Some((k, v)) = map.next_entry::<String, Value>()? {
                    out.insert(k, v);
                }
                Ok(Value::Object(out))
            }
        }
        d.deserialize_any(V)
    }
}

/// Build a [`Value`] with JSON-like syntax: `js!({"a": 1, "b": [true, null]})`.
#[macro_export]
macro_rules! js {
    (null) => { $crate::value::Value::Null };
    (undefined) => { $crate::value::Value::Undefined };
    ([ $($elem:tt),* $(,)? ]) => {
        $crate::value::Value::Array(vec![ $( $crate::js!($elem) ),* ])
    };
    ({ $($key:literal : $val:tt),* $(,)? }) => {{
        #[allow(unused_mut)]
        let mut obj = $crate::value::Object::new();
        $( obj.insert($key, $crate::js!($val)); )*
        $crate::value::Value::Object(obj)
    }};
    ($other:expr) => { $crate::value::Value::from($other) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hidden_members() {
        let mut o = Object::new();
        o.insert("a", Value::from(1.0));
        o.define_hidden("meta", Value::from("x"));
        o.insert("b", Value::from(2.0));
        assert!(o.contains_key("meta"));
        assert_eq!(o.get("meta").and_then(Value::as_str), Some("x"));
        assert_eq!(o.keys().cloned().collect::<Vec<_>>(), ["a", "b"]);
        o.insert("meta", Value::from("y"));
        assert!(o.is_hidden("meta"));
        assert_eq!(Value::Object(o.clone()).to_json().unwrap(), r#"{"a":1,"b":2}"#);
        let mut copy = Object::new();
        copy.assign(&o);
        assert!(!copy.contains_key("meta"));
    }

    #[test]
    fn index_keys_order_first() {
        let mut o = Object::new();
        o.insert("b", Value::Null);
        o.insert("10", Value::Null);
        o.insert("a", Value::Null);
        o.insert("2", Value::Null);
        o.insert("01", Value::Null);
        let keys: Vec<_> = o.keys().cloned().collect();
        assert_eq!(keys, ["2", "10", "b", "a", "01"]);
    }

    #[test]
    fn stringify_semantics() {
        let mut o = Object::new();
        o.insert("u", Value::Undefined);
        o.insert("n", Value::Number(1.0));
        o.insert("f", Value::Number(0.5));
        o.insert("inf", Value::Number(f64::INFINITY));
        o.insert(
            "arr",
            Value::Array(vec![Value::Undefined, Value::Number(-0.0)]),
        );
        o.insert("fn", Value::Function("x".into()));
        assert_eq!(
            Value::Object(o).to_json().unwrap(),
            r#"{"n":1,"f":0.5,"inf":null,"arr":[null,0]}"#
        );
    }

    #[test]
    fn json_parse_semantics() {
        // Correctly rounded 17-digit decimals (255ths of hex colors).
        for text in ["0.9372549019607843", "0.42745098039215684", "0.047058823529411764"] {
            let v = Value::from_json(text).unwrap();
            assert_eq!(v.as_f64().unwrap().to_bits(), text.parse::<f64>().unwrap().to_bits());
            assert_eq!(v.to_json().unwrap(), text);
        }
        assert!(Value::from_json("-0").unwrap().as_f64().unwrap().is_sign_negative());
        assert_eq!(Value::from_json("1e400").unwrap().as_f64(), Some(f64::INFINITY));
        let v = Value::from_json(r#" {"b":1,"a":[true,false,null,"x\u00e9\ud83d\ude00\n"],"b":2} "#).unwrap();
        assert_eq!(v.to_json().unwrap(), r#"{"b":2,"a":[true,false,null,"xé😀\n"]}"#);
        assert_eq!(Value::from_json(r#""\ud800""#).unwrap(), Value::from("\u{FFFD}"));
        for bad in ["", "01", "1.", "-", "[1,]", "{\"a\" 1}", "tru", "\"\u{1}\"", "[1] 2", "{,}"] {
            assert!(Value::from_json(bad).is_err(), "{bad:?} must not parse");
        }
    }

    #[test]
    fn roundtrip_preserves_order() {
        let v = Value::from_json(r#"{"z":1,"a":{"y":2,"b":3}}"#).unwrap();
        assert_eq!(v.to_json().unwrap(), r#"{"z":1,"a":{"y":2,"b":3}}"#);
    }

    #[test]
    fn truthiness() {
        assert!(!Value::Number(f64::NAN).is_truthy());
        assert!(!Value::String(String::new()).is_truthy());
        assert!(Value::object().is_truthy());
    }
}
