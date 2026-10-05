//! Scenario runners of `nm-input-dump`, one module per gate.
//!
//! Shared JSON conventions with `tools/reference-input*.mjs`: a number JSON
//! cannot carry exactly (NaN, ±Infinity, -0) is `{"$num": "NaN" | "Infinity" |
//! "-Infinity" | "-0"}`, and inputs write every number that is not a safe
//! integer as `{"$num": "<decimal>"}`; `Float32Array` contents are arrays of
//! their IEEE bit patterns (`u32`).

pub mod analyser;
pub mod audio;
pub mod automation;
pub mod math;
pub mod midi;

use serde_json::{Value, json};

/// A JavaScript number as JSON (see the module conventions).
pub fn num(x: f64) -> Value {
    if x.is_nan() {
        json!({"$num": "NaN"})
    } else if x == f64::INFINITY {
        json!({"$num": "Infinity"})
    } else if x == f64::NEG_INFINITY {
        json!({"$num": "-Infinity"})
    } else if x == 0.0 && x.is_sign_negative() {
        json!({"$num": "-0"})
    } else if x == x.trunc() && x.abs() < 9.007_199_254_740_992e15 {
        json!(x as i64)
    } else {
        json!(x)
    }
}

/// Reads a JSON number, accepting the `$num` encoding (special values or an
/// exact decimal, parsed with a correctly rounded parser; plain JSON numbers
/// in the inputs are safe integers).
pub fn read_num(value: &Value) -> Option<f64> {
    if let Some(x) = value.as_f64() {
        return Some(x);
    }
    match value.get("$num")?.as_str()? {
        "NaN" => Some(f64::NAN),
        "Infinity" => Some(f64::INFINITY),
        "-Infinity" => Some(f64::NEG_INFINITY),
        "-0" => Some(-0.0),
        decimal => decimal.parse::<f64>().ok(),
    }
}

/// `Float32Array` contents as bit patterns.
pub fn f32_bits(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|v| json!(v.to_bits())).collect())
}

/// A required field.
pub fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("missing field {key:?} in {}", short(value)))
}

/// A required string field.
pub fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| format!("field {key:?} is not a string in {}", short(value)))
}

/// A required numeric field.
pub fn num_field(value: &Value, key: &str) -> Result<f64, String> {
    read_num(field(value, key)?)
        .ok_or_else(|| format!("field {key:?} is not a number in {}", short(value)))
}

/// A required array field.
pub fn array_field<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    field(value, key)?
        .as_array()
        .ok_or_else(|| format!("field {key:?} is not an array in {}", short(value)))
}

/// A short rendering of a JSON value for error messages.
pub fn short(value: &Value) -> String {
    let text = value.to_string();
    if text.len() > 160 {
        format!("{}...", &text[..160])
    } else {
        text
    }
}

/// An array of byte values.
pub fn bytes(value: &Value) -> Result<Vec<u8>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("expected a byte array, got {}", short(value)))?
        .iter()
        .map(|v| {
            v.as_u64()
                .filter(|&b| b <= 255)
                .map(|b| b as u8)
                .ok_or_else(|| format!("not a byte: {v}"))
        })
        .collect()
}
