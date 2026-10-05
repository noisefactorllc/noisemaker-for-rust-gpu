//! Port of `lang/lexer.js` (in progress).

use crate::error::JsError;
use crate::value::Value;

/// A token of `lex(src)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    /// The token as the reference represents it (`{type, lexeme, line, col, ...}`).
    pub value: Value,
}

/// `lex(src)`.
pub fn lex(_src: &str) -> Result<Vec<Token>, JsError> {
    Err(JsError::error("lexer not yet ported"))
}

/// The token array as `JSON.stringify(lex(src))` serializes it.
pub fn tokens_to_value(tokens: &[Token]) -> Value {
    Value::Array(tokens.iter().map(|t| t.value.clone()).collect())
}
