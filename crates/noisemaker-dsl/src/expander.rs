//! Port of `runtime/expander.js` (in progress).

use crate::error::JsError;
use crate::registry::Registry;
use crate::value::Value;

/// `expand(compilationResult)` serialized as the oracle dumps it.
pub fn expand_to_value(_compilation: &Value, _registry: &Registry) -> Result<Value, JsError> {
    Err(JsError::error("expander not yet ported"))
}
