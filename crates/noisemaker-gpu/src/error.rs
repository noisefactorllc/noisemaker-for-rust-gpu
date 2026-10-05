//! Errors as the reference runtime throws them.
//!
//! The reference throws three kinds of values from the render path: plain objects
//! (`{code: 'ERR_PROGRAM_NOT_FOUND', pass, program}`), `ShaderDiagnostic`s from
//! compilation (`backends/diagnostics.js`), and the `TypeError`/`RangeError`s that
//! WebIDL conversions and `DataView` accesses raise. [`RenderError`] keeps all three.

use noisemaker_dsl::{Object, Value};

use crate::reflect::CompilationMessage;

/// A `ShaderDiagnostic` (`backends/diagnostics.js`): a shader compilation failure.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderDiagnostic {
    /// `'ERR_SHADER_COMPILE'`, `'ERR_NO_WGSL_SOURCE'`, ...
    pub code: String,
    pub backend: String,
    /// `'compile'`, `'missing-source'`, ...
    pub stage: String,
    pub program: Option<String>,
    /// The compiler messages as one string (`Line N: message` lines).
    pub detail: String,
    pub messages: Vec<CompilationMessage>,
}

/// A thrown error of the runtime.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RenderError {
    /// A plain thrown object (`{code, ...}`).
    #[error("{}", describe_thrown(.0))]
    Thrown(Value),
    /// A `ShaderDiagnostic`.
    #[error("{}: {}", .0.code, .0.detail)]
    Shader(Box<ShaderDiagnostic>),
    /// A JavaScript `TypeError` / `RangeError` (message includes the error name).
    #[error("{0}")]
    Js(String),
}

fn describe_thrown(v: &Value) -> String {
    // `err.detail || err.message || JSON.stringify(err)` (pipeline.js logging).
    let detail = v.get("detail");
    if detail.is_truthy() {
        return crate::jsv::to_js_string(detail);
    }
    let message = v.get("message");
    if message.is_truthy() {
        return crate::jsv::to_js_string(message);
    }
    v.to_json().unwrap_or_else(|| "undefined".into())
}

impl RenderError {
    /// A thrown `{code, ...fields}` object.
    pub fn thrown(code: &str, fields: &[(&str, Value)]) -> RenderError {
        let mut o = Object::new();
        o.insert("code", Value::from(code));
        for (k, v) in fields {
            o.insert(*k, v.clone());
        }
        RenderError::Thrown(Value::Object(o))
    }

    /// A `TypeError`.
    pub fn type_error(message: impl AsRef<str>) -> RenderError {
        RenderError::Js(format!("TypeError: {}", message.as_ref()))
    }

    /// The machine code of the error (`err.code`), when it has one.
    pub fn code(&self) -> Option<String> {
        match self {
            RenderError::Thrown(v) => v.get("code").as_str().map(str::to_owned),
            RenderError::Shader(d) => Some(d.code.clone()),
            RenderError::Js(_) => None,
        }
    }
}

impl From<String> for RenderError {
    fn from(message: String) -> Self {
        RenderError::Js(message)
    }
}
