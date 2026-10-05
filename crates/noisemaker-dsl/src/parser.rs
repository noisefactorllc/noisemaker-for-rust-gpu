//! Port of `lang/parser.js` (in progress).

use crate::error::JsError;
use crate::lexer::Token;
use crate::value::Value;

/// `parse(tokens)`: the Program AST.
pub fn parse(_tokens: &[Token]) -> Result<Value, JsError> {
    Err(JsError::error("parser not yet ported"))
}
