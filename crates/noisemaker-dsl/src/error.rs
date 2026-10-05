//! Errors as the reference throws them.
//!
//! The reference frontend throws `Error` instances (`SyntaxError`, and the
//! `TypeError`s its own bugs raise on some inputs) and plain objects such as
//! `{code: 'ERR_COMPILATION_FAILED', diagnostics}`. Parity covers what is thrown, so
//! [`JsError`] keeps both forms.

use crate::value::{Object, Value};

/// A thrown value.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum JsError {
    /// An `Error` instance: `name` is the constructor name (`SyntaxError`, `TypeError`, `Error`).
    #[error("{name}: {message}")]
    Error { name: String, message: String },
    /// A thrown non-Error value (the reference throws plain objects).
    #[error("{}", .0.to_json().unwrap_or_else(|| "undefined".into()))]
    Thrown(Value),
}

impl JsError {
    pub fn syntax(message: impl Into<String>) -> Self {
        JsError::Error {
            name: "SyntaxError".into(),
            message: message.into(),
        }
    }

    pub fn type_error(message: impl Into<String>) -> Self {
        JsError::Error {
            name: "TypeError".into(),
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        JsError::Error {
            name: "Error".into(),
            message: message.into(),
        }
    }

    /// The thrown value as the reference oracle serializes it:
    /// `{name, message}` for Error instances, the value itself otherwise.
    pub fn to_value(&self) -> Value {
        match self {
            JsError::Error { name, message } => {
                let mut o = Object::new();
                o.insert("name", Value::from(name.as_str()));
                o.insert("message", Value::from(message.as_str()));
                Value::Object(o)
            }
            JsError::Thrown(v) => v.clone(),
        }
    }
}
