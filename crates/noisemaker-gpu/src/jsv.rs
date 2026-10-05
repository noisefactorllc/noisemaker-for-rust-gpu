//! JavaScript value coercions the reference runtime performs implicitly.
//!
//! The reference writes uniform data through `DataView`/typed-array setters and
//! WebIDL-converted WebGPU arguments, all of which coerce with `ToNumber`,
//! `ToInt32`, `Math.fround` and `[EnforceRange]`. These helpers apply the same
//! conversions to [`Value`]s.

use noisemaker_dsl::js::{number_to_string, string_to_number, to_int32, to_uint32};
use noisemaker_dsl::{Object, Value};

/// ECMAScript `ToNumber`.
pub fn to_number(v: &Value) -> f64 {
    match v {
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
        Value::Array(_) => string_to_number(&to_js_string(v)),
        Value::Object(_) | Value::Function(_) => f64::NAN,
    }
}

/// `String(value)` (ToString via ToPrimitive for arrays and objects).
pub fn to_js_string(v: &Value) -> String {
    noisemaker_dsl::js::value_to_property_key(v)
}

/// `Math.fround(ToNumber(v))`: the value a `Float32Array` element or
/// `DataView.setFloat32` stores.
pub fn to_f32(v: &Value) -> f32 {
    to_number(v) as f32
}

/// `ToInt32(ToNumber(v))`: an `Int32Array` element / `DataView.setInt32`.
pub fn to_i32(v: &Value) -> i32 {
    to_int32(to_number(v))
}

/// `ToUint32(ToNumber(v))`: `DataView.setUint32`.
pub fn to_u32(v: &Value) -> u32 {
    to_uint32(to_number(v))
}

/// `Math.round(ToNumber(v))`.
pub fn js_round(v: &Value) -> f64 {
    noisemaker_dsl::js::math_round(to_number(v))
}

/// WebIDL `[EnforceRange] unsigned long` conversion (`GPUSize32`,
/// `GPUIntegerCoordinate`): non-finite values throw, fractions truncate, values
/// outside `0..=u32::MAX` throw.
pub fn enforce_range_u32(v: &Value) -> Result<u32, String> {
    let n = to_number(v);
    if !n.is_finite() {
        return Err("Value is not of type 'unsigned long'.".into());
    }
    let t = n.trunc();
    if t < 0.0 || t > u32::MAX as f64 {
        return Err("Value is outside the 'unsigned long' value range.".into());
    }
    Ok(t as u32)
}

/// `a === b` for the primitive comparisons the runtime makes. Distinct objects
/// and arrays are never strictly equal (graph data never shares references).
pub fn strict_equals(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        _ => false,
    }
}

/// `Number.isFinite(v)`.
pub fn is_finite_number(v: &Value) -> bool {
    matches!(v, Value::Number(n) if n.is_finite())
}

/// `a ?? b` for member reads.
pub fn nullish_or<'a>(a: &'a Value, b: &'a Value) -> &'a Value {
    if a.is_nullish() { b } else { a }
}

/// Template-literal interpolation `${value}` (ToString).
pub fn interpolate(v: &Value) -> String {
    match v {
        Value::Number(n) => number_to_string(*n),
        other => to_js_string(other),
    }
}

/// `typeof v === 'object' && v !== null` and not an array.
pub fn is_plain_object(v: &Value) -> bool {
    matches!(v, Value::Object(_))
}

/// Shorthand for an object member read that tolerates non-objects (`obj?.[key]`).
pub fn member<'a>(obj: Option<&'a Object>, key: &str) -> &'a Value {
    static UNDEFINED: Value = Value::Undefined;
    match obj {
        Some(o) => o.get_or_undefined(key),
        None => &UNDEFINED,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coercions_follow_ecmascript() {
        assert!(to_number(&Value::Undefined).is_nan());
        assert_eq!(to_number(&Value::Null), 0.0);
        assert_eq!(to_number(&Value::from("0x10")), 16.0);
        assert_eq!(to_number(&Value::Array(vec![Value::Number(5.0)])), 5.0);
        assert!(to_number(&Value::Array(vec![1.0.into(), 2.0.into()])).is_nan());
        assert_eq!(to_i32(&Value::Number(4294967297.0)), 1);
        assert_eq!(to_u32(&Value::Number(-1.0)), u32::MAX);
        assert_eq!(js_round(&Value::Number(-2.5)), -2.0);
        assert_eq!(enforce_range_u32(&Value::Number(255.9)), Ok(255));
        assert!(enforce_range_u32(&Value::Number(f64::NAN)).is_err());
        assert!(enforce_range_u32(&Value::Number(-1.0)).is_err());
    }
}
