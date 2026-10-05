//! Port of `runtime/compiler.js` (in progress).

use crate::error::JsError;
use crate::registry::Registry;
use crate::value::Value;

/// `compileGraph(src)` normalized as the oracle dumps it (Maps as objects, program
/// specs without shader source texts, no `compiledAt`).
pub fn dump_graph(_src: &str, _registry: &Registry) -> Result<Value, JsError> {
    Err(JsError::error("compileGraph not yet ported"))
}
