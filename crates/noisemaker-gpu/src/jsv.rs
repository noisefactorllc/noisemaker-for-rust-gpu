//! JavaScript value coercions the reference runtime performs implicitly.
//!
//! The reference writes uniform data through `DataView`/typed-array setters and
//! WebIDL-converted WebGPU arguments, all of which coerce with `ToNumber`,
//! `ToInt32`, `Math.fround` and `[EnforceRange]`. The JavaScript semantics are
//! [`noisemaker_dsl::js`](mod@noisemaker_dsl::js)'s, re-exported here; this module adds the
//! typed-array and WebIDL conversions over [`Value`]s.

use noisemaker_dsl::Value;
pub use noisemaker_dsl::js::{is_finite_number, strict_equals, to_number};
use noisemaker_dsl::js::{to_int32, to_uint32};

/// `String(value)` (template literals, property keys).
pub use noisemaker_dsl::js::value_to_property_key as to_js_string;

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
        assert_eq!(to_f32(&Value::Number(0.1)), 0.1f32);
        assert_eq!(enforce_range_u32(&Value::Number(255.9)), Ok(255));
        assert!(enforce_range_u32(&Value::Number(f64::NAN)).is_err());
        assert!(enforce_range_u32(&Value::Number(-1.0)).is_err());
    }
}
