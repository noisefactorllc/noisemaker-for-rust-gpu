//! The tagged JSON encoding of JavaScript values that the parity tools
//! exchange with the reference (`tools/reference-dsl-tools.mjs`): plain JSON,
//! except that values JSON cannot carry are tagged objects —
//! `{"$js": "undefined" | "NaN" | "Infinity" | "-Infinity" | "-0"}`,
//! `{"$js": "function", "source"}`, typed arrays, Maps, errors, Effect
//! instances and subclasses, and `{"$js": "object", "members"}` for an object
//! with an own `$js` member.

use crate::error::JsError;
use crate::value::{Object, Value};

/// The tag member of an encoded value.
pub const TAG: &str = "$js";

fn tag(name: &str) -> Value {
    let mut o = Object::new();
    o.insert(TAG, Value::from(name));
    Value::Object(o)
}

/// A JavaScript value as tagged JSON.
pub fn encode(v: &Value) -> Value {
    match v {
        Value::Undefined => tag("undefined"),
        Value::Number(n) if n.is_nan() => tag("NaN"),
        Value::Number(n) if n.is_infinite() => tag(if *n > 0.0 { "Infinity" } else { "-Infinity" }),
        Value::Number(n) if *n == 0.0 && n.is_sign_negative() => tag("-0"),
        Value::Function(source) => {
            let mut o = Object::new();
            o.insert(TAG, Value::from("function"));
            o.insert("source", Value::from(source.as_str()));
            Value::Object(o)
        }
        Value::Array(items) => Value::Array(items.iter().map(encode).collect()),
        Value::Object(members) => {
            let encoded: Object = members
                .iter()
                .map(|(k, v)| (k.clone(), encode(v)))
                .collect();
            if encoded.contains_key(TAG) {
                let mut o = Object::new();
                o.insert(TAG, Value::from("object"));
                o.insert("members", Value::Object(encoded));
                Value::Object(o)
            } else {
                Value::Object(encoded)
            }
        }
        other => other.clone(),
    }
}

pub fn decode_members(v: &Value) -> Value {
    match v {
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), decode(v))).collect()),
        _ => Value::Object(Object::new()),
    }
}

/// Tagged JSON back to a JavaScript value.
pub fn decode(v: &Value) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(decode).collect()),
        Value::Object(o) => match o.get(TAG).and_then(Value::as_str) {
            None => decode_members(v),
            Some(t) => match t {
                "undefined" => Value::Undefined,
                "NaN" => Value::Number(f64::NAN),
                "Infinity" => Value::Number(f64::INFINITY),
                "-Infinity" => Value::Number(f64::NEG_INFINITY),
                "-0" => Value::Number(-0.0),
                "function" => {
                    Value::Function(v.get("source").as_str().unwrap_or_default().to_owned())
                }
                // Typed arrays read like arrays of their (already rounded) elements.
                "Float32Array" | "Float64Array" => match v.get("values") {
                    Value::Array(values) => Value::Array(values.iter().map(decode).collect()),
                    _ => Value::Array(Vec::new()),
                },
                // A Map has no enumerable string-keyed members.
                "Map" => Value::Object(Object::new()),
                "object" => decode_members(v.get("members")),
                "error" => {
                    let mut e = Object::new();
                    e.insert("name", v.get("name").clone());
                    e.insert("message", v.get("message").clone());
                    Value::Object(e)
                }
                "effectInstance" => decode_members(v.get("props")),
                "effectSubclass" => Value::Function("class extends Effect {}".into()),
                other => panic!("unknown value tag {other}"),
            },
        },
        other => other.clone(),
    }
}

/// A thrown value from its tagged encoding.
pub fn decode_thrown(v: &Value) -> JsError {
    if v.get(TAG).as_str() == Some("error") {
        JsError::Error {
            name: v.get("name").as_str().unwrap_or("Error").to_owned(),
            message: v.get("message").as_str().unwrap_or_default().to_owned(),
        }
    } else {
        JsError::Thrown(decode(v))
    }
}

/// `{name, message}` for Error instances, `{thrown}` for other thrown values.
pub fn error_record(e: &JsError) -> Value {
    match e {
        JsError::Error { name, message } => {
            let mut o = Object::new();
            o.insert("name", Value::from(name.as_str()));
            o.insert("message", Value::from(message.as_str()));
            Value::Object(o)
        }
        JsError::Thrown(v) => {
            let mut o = Object::new();
            o.insert("thrown", encode(v));
            Value::Object(o)
        }
    }
}
