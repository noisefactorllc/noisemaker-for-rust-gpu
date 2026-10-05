//! Port of `lang/validator.js` (in progress).

use crate::error::JsError;
use crate::registry::Registry;
use crate::value::Value;

/// `validate(ast)`: `{plans, diagnostics, render, vars, searchNamespaces}`.
pub fn validate(_ast: &Value, _registry: &Registry) -> Result<Value, JsError> {
    Err(JsError::error("validator not yet ported"))
}
