//! Port of `runtime/compiler.js` (in progress).

use crate::error::JsError;
use crate::registry::Registry;
use crate::value::Value;

/// `compileGraph(src)` normalized as the oracle dumps it (Maps as objects, program
/// specs without shader source texts, no `compiledAt`).
pub fn dump_graph(_src: &str, _registry: &Registry) -> Result<Value, JsError> {
    Err(JsError::error("compileGraph not yet ported"))
}

/// The graph dump of `compileGraph(src)` built from an already validated program
/// (`validated` is the reference's `validate(parse(lex(src)))` output).
pub fn dump_graph_from_validated(
    _src: &str,
    _validated: &Value,
    _registry: &Registry,
) -> Result<Value, JsError> {
    Err(JsError::error("compileGraph not yet ported"))
}
