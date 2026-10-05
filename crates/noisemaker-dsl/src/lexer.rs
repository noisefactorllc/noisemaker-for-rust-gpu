//! Port of `lang/lexer.js`: the lexer of the Polymorphic DSL.
//!
//! The reference scans a JavaScript string, so every index, length and column
//! here counts UTF-16 code units, as the reference does (`src[i]`, `src.length`,
//! `slice`). The source is converted to UTF-16 once and scanned by code unit.
//!
//! Tokens keep two coordinate systems, as the reference does:
//!
//! * `line`/`col`: the scanner's legacy bookkeeping, which drifts after some
//!   multiline tokens (a function token spanning lines inside parentheses, a
//!   string with an escaped newline). These are the enumerable token fields and
//!   appear in error messages and AST `loc` objects.
//! * `position`: the non-enumerable, source-derived `{line, column, start, end}`
//!   used for structured diagnostics.
//!
//! Errors carry the structured `diagnostic` the reference attaches to its
//! `SyntaxError`s as a non-enumerable property ([`DiagnosticError`]); [`lex`]
//! reports just the thrown error ([`JsError`]), as the parity oracle sees it.

use std::fmt;

use crate::diagnostics;
use crate::error::JsError;
use crate::js;
use crate::value::{Object, Value};

// --- token types --------------------------------------------------------------

/// A member of `Object.prototype` that `keywords[lexeme]` finds by inheritance.
///
/// The reference looks keywords up with `keywords[lexeme]` on a plain object
/// (`RESERVED_KEYWORDS`), so an identifier named like an `Object.prototype`
/// member reads that inherited, truthy member and the token's `type` becomes it:
/// a function (`constructor` reads `Object` itself) or, for `__proto__`,
/// `Object.prototype`. Such tokens match no token type in the parser, serialize
/// without `type` (functions) or with `type: {}` (`__proto__`), and stringify as
/// `function <name>() { [native code] }` or `[object Object]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InheritedMember {
    Constructor,
    DefineGetter,
    DefineSetter,
    HasOwnProperty,
    LookupGetter,
    LookupSetter,
    IsPrototypeOf,
    PropertyIsEnumerable,
    ToString,
    ValueOf,
    Proto,
    ToLocaleString,
}

impl InheritedMember {
    /// `Object.getOwnPropertyNames(Object.prototype)`, with the member each names.
    pub const ALL: [(&'static str, InheritedMember); 12] = [
        ("constructor", InheritedMember::Constructor),
        ("__defineGetter__", InheritedMember::DefineGetter),
        ("__defineSetter__", InheritedMember::DefineSetter),
        ("hasOwnProperty", InheritedMember::HasOwnProperty),
        ("__lookupGetter__", InheritedMember::LookupGetter),
        ("__lookupSetter__", InheritedMember::LookupSetter),
        ("isPrototypeOf", InheritedMember::IsPrototypeOf),
        (
            "propertyIsEnumerable",
            InheritedMember::PropertyIsEnumerable,
        ),
        ("toString", InheritedMember::ToString),
        ("valueOf", InheritedMember::ValueOf),
        ("__proto__", InheritedMember::Proto),
        ("toLocaleString", InheritedMember::ToLocaleString),
    ];

    /// The inherited member `keywords[name]` reads, if `name` names one.
    pub fn lookup(name: &str) -> Option<InheritedMember> {
        Self::ALL
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, member)| *member)
    }

    /// The name of the function the member holds (`None` for `__proto__`, which
    /// reads `Object.prototype`, an object).
    pub fn function_name(self) -> Option<&'static str> {
        Some(match self {
            InheritedMember::Constructor => "Object",
            InheritedMember::DefineGetter => "__defineGetter__",
            InheritedMember::DefineSetter => "__defineSetter__",
            InheritedMember::HasOwnProperty => "hasOwnProperty",
            InheritedMember::LookupGetter => "__lookupGetter__",
            InheritedMember::LookupSetter => "__lookupSetter__",
            InheritedMember::IsPrototypeOf => "isPrototypeOf",
            InheritedMember::PropertyIsEnumerable => "propertyIsEnumerable",
            InheritedMember::ToString => "toString",
            InheritedMember::ValueOf => "valueOf",
            InheritedMember::Proto => return None,
            InheritedMember::ToLocaleString => "toLocaleString",
        })
    }

    /// `String(member)`.
    pub fn to_js_string(self) -> String {
        match self.function_name() {
            Some(name) => format!("function {name}() {{ [native code] }}"),
            None => "[object Object]".to_owned(),
        }
    }

    /// The member as a value: a function, or `Object.prototype` (which has no own
    /// enumerable members, so it serializes as `{}`).
    pub fn to_value(self) -> Value {
        match self.function_name() {
            Some(_) => Value::Function(self.to_js_string()),
            None => Value::object(),
        }
    }
}

/// A token's `type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenType {
    Comment,
    OutputRef,
    SourceRef,
    VolRef,
    GeoRef,
    XyzRef,
    VelRef,
    RgbaRef,
    MeshRef,
    Hex,
    Func,
    Number,
    Dot,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Equal,
    Semicolon,
    Plus,
    Minus,
    Star,
    Slash,
    String,
    Ident,
    Let,
    Render,
    Write,
    Write3d,
    True,
    False,
    If,
    Elif,
    Else,
    Break,
    Continue,
    Return,
    Search,
    Subchain,
    Eof,
    /// A `type` read from `Object.prototype` (see [`InheritedMember`]).
    Inherited(InheritedMember),
}

impl TokenType {
    /// The reference's type string (`None` for an inherited member, which is not
    /// a string).
    pub fn name(self) -> Option<&'static str> {
        Some(match self {
            TokenType::Comment => "COMMENT",
            TokenType::OutputRef => "OUTPUT_REF",
            TokenType::SourceRef => "SOURCE_REF",
            TokenType::VolRef => "VOL_REF",
            TokenType::GeoRef => "GEO_REF",
            TokenType::XyzRef => "XYZ_REF",
            TokenType::VelRef => "VEL_REF",
            TokenType::RgbaRef => "RGBA_REF",
            TokenType::MeshRef => "MESH_REF",
            TokenType::Hex => "HEX",
            TokenType::Func => "FUNC",
            TokenType::Number => "NUMBER",
            TokenType::Dot => "DOT",
            TokenType::LParen => "LPAREN",
            TokenType::RParen => "RPAREN",
            TokenType::LBrace => "LBRACE",
            TokenType::RBrace => "RBRACE",
            TokenType::LBracket => "LBRACKET",
            TokenType::RBracket => "RBRACKET",
            TokenType::Comma => "COMMA",
            TokenType::Colon => "COLON",
            TokenType::Equal => "EQUAL",
            TokenType::Semicolon => "SEMICOLON",
            TokenType::Plus => "PLUS",
            TokenType::Minus => "MINUS",
            TokenType::Star => "STAR",
            TokenType::Slash => "SLASH",
            TokenType::String => "STRING",
            TokenType::Ident => "IDENT",
            TokenType::Let => "LET",
            TokenType::Render => "RENDER",
            TokenType::Write => "WRITE",
            TokenType::Write3d => "WRITE3D",
            TokenType::True => "TRUE",
            TokenType::False => "FALSE",
            TokenType::If => "IF",
            TokenType::Elif => "ELIF",
            TokenType::Else => "ELSE",
            TokenType::Break => "BREAK",
            TokenType::Continue => "CONTINUE",
            TokenType::Return => "RETURN",
            TokenType::Search => "SEARCH",
            TokenType::Subchain => "SUBCHAIN",
            TokenType::Eof => "EOF",
            TokenType::Inherited(_) => return None,
        })
    }

    /// `String(token.type)`, as error messages interpolate it.
    pub fn to_js_string(self) -> String {
        match self {
            TokenType::Inherited(member) => member.to_js_string(),
            other => other.name().unwrap_or_default().to_owned(),
        }
    }

    /// `token.type` as a value.
    pub fn to_value(self) -> Value {
        match self {
            TokenType::Inherited(member) => member.to_value(),
            other => Value::from(other.name().unwrap_or_default()),
        }
    }
}

impl fmt::Display for TokenType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_js_string())
    }
}

/// `RESERVED_KEYWORDS`: reserved DSL keyword → token type, in declaration order.
/// Shared with namespace registration (`runtime/tags.js` → `registerNamespace`).
pub const RESERVED_KEYWORDS: &[(&str, TokenType)] = &[
    ("let", TokenType::Let),
    ("render", TokenType::Render),
    ("write", TokenType::Write),
    ("write3d", TokenType::Write3d),
    ("true", TokenType::True),
    ("false", TokenType::False),
    ("if", TokenType::If),
    ("elif", TokenType::Elif),
    ("else", TokenType::Else),
    ("break", TokenType::Break),
    ("continue", TokenType::Continue),
    ("return", TokenType::Return),
    ("search", TokenType::Search),
    ("subchain", TokenType::Subchain),
];

/// `Object.prototype.hasOwnProperty.call(RESERVED_KEYWORDS, name)`.
pub fn is_reserved_keyword(name: &str) -> bool {
    RESERVED_KEYWORDS.iter().any(|(k, _)| *k == name)
}

/// `keywords[lexeme]` when truthy: an own keyword, or an inherited
/// `Object.prototype` member (see [`InheritedMember`]).
fn keyword_lookup(lexeme: &str) -> Option<TokenType> {
    RESERVED_KEYWORDS
        .iter()
        .find(|(k, _)| *k == lexeme)
        .map(|(_, t)| *t)
        .or_else(|| InheritedMember::lookup(lexeme).map(TokenType::Inherited))
}

// --- tokens -------------------------------------------------------------------

/// A token's non-enumerable `position`: source-derived one-based `line` and
/// `column` (UTF-16 code units; LF-only line breaks) and the UTF-16 offsets
/// `[start, end)` of the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: usize,
    pub column: usize,
    pub start: usize,
    pub end: usize,
}

impl Position {
    /// The parser's validity test for a position (`hasPosition`): positive
    /// one-based coordinates and `end >= start`.
    pub fn is_valid(&self) -> bool {
        self.line > 0 && self.column > 0 && self.end >= self.start
    }

    /// The position as the reference object `{line, column, start, end}`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("line", Value::from(self.line));
        o.insert("column", Value::from(self.column));
        o.insert("start", Value::from(self.start));
        o.insert("end", Value::from(self.end));
        Value::Object(o)
    }
}

/// A token of `lex(src)`: `{type, lexeme, line, col}` plus the non-enumerable
/// `position`.
///
/// `line` and `col` are the legacy scanner coordinates. The lexer always sets
/// them to positive integers; tokens built by callers may carry any number, or
/// none (`undefined`), and the parser reports such coordinates as the reference
/// does. Likewise `position` is `None` for caller-built tokens without one.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenType,
    pub lexeme: String,
    pub line: Option<f64>,
    pub col: Option<f64>,
    pub position: Option<Position>,
}

/// A legacy coordinate as a value (`undefined` when absent).
pub(crate) fn coordinate_value(c: Option<f64>) -> Value {
    c.map_or(Value::Undefined, Value::Number)
}

/// A legacy coordinate as a template literal interpolates it.
pub(crate) fn coordinate_string(c: Option<f64>) -> String {
    c.map_or_else(|| "undefined".to_owned(), js::number_to_string)
}

impl Token {
    /// The token as the reference object (enumerable members only):
    /// `{type, lexeme, line, col}`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("type", self.kind.to_value());
        o.insert("lexeme", Value::from(self.lexeme.as_str()));
        o.insert("line", coordinate_value(self.line));
        o.insert("col", coordinate_value(self.col));
        Value::Object(o)
    }
}

/// The token array as `JSON.stringify(lex(src))` serializes it.
pub fn tokens_to_value(tokens: &[Token]) -> Value {
    Value::Array(tokens.iter().map(Token::to_value).collect())
}

// --- diagnostics ----------------------------------------------------------------

/// One-based source coordinates of a diagnostic (`{line, column}`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Location {
    pub line: f64,
    pub column: f64,
}

impl Location {
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("line", Value::Number(self.line));
        o.insert("column", Value::Number(self.column));
        Value::Object(o)
    }
}

/// UTF-16 source offsets `[start, end)` of a diagnostic (`{start, end}`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub start: f64,
    pub end: f64,
}

impl Span {
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("start", Value::Number(self.start));
        o.insert("end", Value::Number(self.end));
        Value::Object(o)
    }
}

/// The structured `diagnostic` the reference attaches, non-enumerable, to the
/// `SyntaxError`s its lexer and parser throw:
/// `{code, stage, severity, message, location, span}`, where `location` and
/// `span` are `null` when unavailable.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub code: &'static str,
    pub stage: &'static str,
    pub severity: &'static str,
    pub message: String,
    pub location: Option<Location>,
    pub span: Option<Span>,
}

impl Diagnostic {
    /// A diagnostic of `code` with the table's stage and severity.
    pub fn new(
        code: &'static str,
        message: impl Into<String>,
        location: Option<Location>,
        span: Option<Span>,
    ) -> Diagnostic {
        let info = diagnostics::lookup(code)
            .unwrap_or_else(|| panic!("diagnostic code {code} is not in the table"));
        Diagnostic {
            code,
            stage: info.stage,
            severity: info.severity,
            message: message.into(),
            location,
            span,
        }
    }

    /// The diagnostic as the reference object.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("code", Value::from(self.code));
        o.insert("stage", Value::from(self.stage));
        o.insert("severity", Value::from(self.severity));
        o.insert("message", Value::from(self.message.as_str()));
        o.insert(
            "location",
            self.location
                .as_ref()
                .map_or(Value::Null, Location::to_value),
        );
        o.insert(
            "span",
            self.span.as_ref().map_or(Value::Null, Span::to_value),
        );
        Value::Object(o)
    }
}

/// A thrown error together with its non-enumerable `diagnostic` property (the
/// reference's lexer and parser `SyntaxError`s carry one; the `TypeError` a
/// truncated caller-built token array raises does not). The diagnostic is boxed
/// to keep the error small on the success path.
#[derive(Debug, Clone, PartialEq)]
pub struct DiagnosticError {
    pub error: JsError,
    pub diagnostic: Option<Box<Diagnostic>>,
}

impl DiagnosticError {
    /// `new SyntaxError(diagnostic.message)` with `diagnostic` attached.
    pub fn syntax(diagnostic: Diagnostic) -> DiagnosticError {
        DiagnosticError {
            error: JsError::syntax(diagnostic.message.clone()),
            diagnostic: Some(Box::new(diagnostic)),
        }
    }

    /// A thrown error without a diagnostic.
    pub fn plain(error: JsError) -> DiagnosticError {
        DiagnosticError {
            error,
            diagnostic: None,
        }
    }
}

impl fmt::Display for DiagnosticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.error.fmt(f)
    }
}

impl std::error::Error for DiagnosticError {}

impl From<DiagnosticError> for JsError {
    fn from(e: DiagnosticError) -> JsError {
        e.error
    }
}

// --- lexer ----------------------------------------------------------------------

const NL: u16 = b'\n' as u16;

fn unit(c: u8) -> u16 {
    c as u16
}

/// `isDigit(c)` (`c` may be past the end: `undefined` is not a digit).
fn is_digit(c: Option<u16>) -> bool {
    matches!(c, Some(c) if (unit(b'0')..=unit(b'9')).contains(&c))
}

/// `isLetter(c)`.
fn is_letter(c: Option<u16>) -> bool {
    matches!(c, Some(c) if (unit(b'a')..=unit(b'z')).contains(&c) || (unit(b'A')..=unit(b'Z')).contains(&c))
}

/// `/[0-9a-fA-F]/.test(c)`.
fn is_hex_digit(c: u16) -> bool {
    (unit(b'0')..=unit(b'9')).contains(&c)
        || (unit(b'a')..=unit(b'f')).contains(&c)
        || (unit(b'A')..=unit(b'F')).contains(&c)
}

/// `String.prototype.slice` result for a run of UTF-16 code units. Lexemes never
/// split a surrogate pair; a lone surrogate (only ever a single code unit quoted
/// in an L001 message) becomes U+FFFD, the one thing a Rust string cannot hold.
fn utf16_string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// The scanner state of `lex(src)` (the reference's closure variables).
struct Lexer<'a> {
    src: &'a [u16],
    tokens: Vec<Token>,
    i: usize,
    line: usize,
    col: usize,
    // Correct source coordinates for structured diagnostics. The scanner's
    // legacy line/col bookkeeping drifts after some multiline tokens, so
    // positions are recomputed from source offsets anchored at the previous
    // token's end. Successful token fields are untouched.
    src_line: usize,
    src_col: usize,
    anchor: usize,
}

impl<'a> Lexer<'a> {
    /// `src[k]` (`None` past the end, like `undefined`).
    fn at(&self, k: usize) -> Option<u16> {
        self.src.get(k).copied()
    }

    /// `src[k] === c`.
    fn is(&self, k: usize, c: u8) -> bool {
        self.at(k) == Some(unit(c))
    }

    /// `add(type, lexeme, line, col, end)`.
    fn add(&mut self, kind: TokenType, lexeme: String, line: usize, col: usize, end: usize) {
        for offset in self.anchor..self.i {
            if self.src[offset] == NL {
                self.src_line += 1;
                self.src_col = 1;
            } else {
                self.src_col += 1;
            }
        }
        let start_line = self.src_line;
        let start_column = self.src_col;
        for offset in self.i..end {
            if self.src[offset] == NL {
                self.src_line += 1;
                self.src_col = 1;
            } else {
                self.src_col += 1;
            }
        }
        self.anchor = end;
        self.tokens.push(Token {
            kind,
            lexeme,
            line: Some(line as f64),
            col: Some(col as f64),
            position: Some(Position {
                line: start_line,
                column: start_column,
                start: self.i,
                end,
            }),
        });
    }

    /// `fail(code, message, start, end)`: the error to throw. Only scans source
    /// coordinates on failure; legacy error messages retain their existing
    /// position bookkeeping.
    fn fail(
        &self,
        code: &'static str,
        message: String,
        start: usize,
        end: usize,
    ) -> DiagnosticError {
        let mut error_line = 1usize;
        let mut column = 1usize;
        for offset in 0..start {
            if self.src.get(offset) == Some(&NL) {
                error_line += 1;
                column = 1;
            } else {
                column += 1;
            }
        }
        DiagnosticError::syntax(Diagnostic::new(
            code,
            message,
            Some(Location {
                line: error_line as f64,
                column: column as f64,
            }),
            Some(Span {
                start: start as f64,
                end: end as f64,
            }),
        ))
    }

    /// A reference token (`vol0`, `geo0`, ...): `prefix_len` letters already
    /// matched, then the digits.
    fn scan_reference(
        &mut self,
        kind: TokenType,
        prefix_len: usize,
        start_line: usize,
        start_col: usize,
    ) {
        let i = self.i;
        let mut j = i + prefix_len;
        while j < self.src.len() && is_digit(self.at(j)) {
            j += 1;
        }
        let lexeme = utf16_string(&self.src[i..j]);
        self.add(kind, lexeme, start_line, start_col, j);
        self.col += j - i;
        self.i = j;
    }

    /// One-character punctuation token.
    fn punct(&mut self, kind: TokenType, lexeme: &str, start_line: usize, start_col: usize) {
        let end = self.i + 1;
        self.add(kind, lexeme.to_owned(), start_line, start_col, end);
        self.i += 1;
        self.col += 1;
    }

    /// The scanning loop of `lex(src)`.
    fn run(mut self) -> Result<Vec<Token>, DiagnosticError> {
        let src = self.src;
        let len = src.len();
        while self.i < len {
            let i = self.i;
            let ch = src[i];

            if ch == unit(b' ') || ch == unit(b'\t') || ch == unit(b'\r') {
                self.i += 1;
                self.col += 1;
                continue;
            }
            if ch == NL {
                self.i += 1;
                self.line += 1;
                self.col = 1;
                continue;
            }

            let start_line = self.line;
            let start_col = self.col;

            // line comments - emit as COMMENT token
            if ch == unit(b'/') && self.is(i + 1, b'/') {
                let mut j = i + 2;
                while j < len && src[j] != NL {
                    j += 1;
                }
                let text = utf16_string(&src[i..j]);
                self.add(TokenType::Comment, text, start_line, start_col, j);
                self.col += j - i;
                self.i = j;
                continue;
            }

            // block comments - emit as COMMENT token
            if ch == unit(b'/') && self.is(i + 1, b'*') {
                let mut j = i + 2;
                let mut end_line = self.line;
                let mut end_col = self.col + 2;
                while j < len && !(src[j] == unit(b'*') && self.is(j + 1, b'/')) {
                    if src[j] == NL {
                        end_line += 1;
                        end_col = 1;
                    } else {
                        end_col += 1;
                    }
                    j += 1;
                }
                if j >= len {
                    return Err(self.fail(
                        "L003",
                        format!("Unterminated comment at line {start_line} col {start_col}"),
                        i,
                        len,
                    ));
                }
                j += 2;
                let text = utf16_string(&src[i..j]);
                self.add(TokenType::Comment, text, start_line, start_col, j);
                self.line = end_line;
                self.col = end_col + 2;
                self.i = j;
                continue;
            }

            // output or source reference
            if (ch == unit(b'o') || ch == unit(b's')) && is_digit(self.at(i + 1)) {
                let mut j = i + 1;
                while j < len && is_digit(self.at(j)) {
                    j += 1;
                }
                let lexeme = utf16_string(&src[i..j]);
                let token_type = if ch == unit(b'o') {
                    TokenType::OutputRef
                } else {
                    TokenType::SourceRef
                };
                let is_member_segment = self.tokens.last().map(|t| t.kind) == Some(TokenType::Dot);
                // /^o[0-7]$/
                let in_range = j - i == 2 && (unit(b'0')..=unit(b'7')).contains(&src[i + 1]);
                if token_type == TokenType::OutputRef && !is_member_segment && !in_range {
                    return Err(self.fail(
                        "L004",
                        format!(
                            "Output surface reference '{lexeme}' is out of range; expected o0-o7 at line {start_line} col {start_col}"
                        ),
                        i,
                        j,
                    ));
                }
                self.add(token_type, lexeme, start_line, start_col, j);
                self.col += j - i;
                self.i = j;
                continue;
            }

            // volume reference (vol0-vol7)
            if ch == unit(b'v')
                && self.is(i + 1, b'o')
                && self.is(i + 2, b'l')
                && is_digit(self.at(i + 3))
            {
                self.scan_reference(TokenType::VolRef, 3, start_line, start_col);
                continue;
            }

            // geometry reference (geo0-geo7)
            if ch == unit(b'g')
                && self.is(i + 1, b'e')
                && self.is(i + 2, b'o')
                && is_digit(self.at(i + 3))
            {
                self.scan_reference(TokenType::GeoRef, 3, start_line, start_col);
                continue;
            }

            // xyz reference (xyz0-xyz7) - agent position surfaces
            if ch == unit(b'x')
                && self.is(i + 1, b'y')
                && self.is(i + 2, b'z')
                && is_digit(self.at(i + 3))
            {
                self.scan_reference(TokenType::XyzRef, 3, start_line, start_col);
                continue;
            }

            // vel reference (vel0-vel7) - agent velocity surfaces
            if ch == unit(b'v')
                && self.is(i + 1, b'e')
                && self.is(i + 2, b'l')
                && is_digit(self.at(i + 3))
            {
                self.scan_reference(TokenType::VelRef, 3, start_line, start_col);
                continue;
            }

            // rgba reference (rgba0-rgba7) - agent color surfaces
            if ch == unit(b'r')
                && self.is(i + 1, b'g')
                && self.is(i + 2, b'b')
                && self.is(i + 3, b'a')
                && is_digit(self.at(i + 4))
            {
                self.scan_reference(TokenType::RgbaRef, 4, start_line, start_col);
                continue;
            }

            // mesh reference (mesh0-mesh7) - mesh geometry surfaces
            if ch == unit(b'm')
                && self.is(i + 1, b'e')
                && self.is(i + 2, b's')
                && self.is(i + 3, b'h')
                && is_digit(self.at(i + 4))
            {
                self.scan_reference(TokenType::MeshRef, 4, start_line, start_col);
                continue;
            }

            // html hex color literal
            if ch == unit(b'#') {
                let mut j = i + 1;
                while j < len && is_hex_digit(src[j]) {
                    j += 1;
                }
                let hex_len = j - i;
                if hex_len == 4 || hex_len == 7 || hex_len == 9 {
                    let lexeme = utf16_string(&src[i..j]);
                    self.add(TokenType::Hex, lexeme, start_line, start_col, j);
                    self.col += hex_len;
                    self.i = j;
                    continue;
                }
            }

            // arrow function expression (() => expr)
            if ch == unit(b'(') && self.is(i + 1, b')') {
                let mut j = i + 2;
                while j < len && (src[j] == unit(b' ') || src[j] == unit(b'\t')) {
                    j += 1;
                }
                if self.is(j, b'=') && self.is(j + 1, b'>') {
                    j += 2;
                    while j < len && (src[j] == unit(b' ') || src[j] == unit(b'\t')) {
                        j += 1;
                    }
                    let mut depth = 0usize;
                    let expr_start = j;
                    while j < len {
                        let c = src[j];
                        if c == unit(b'(') {
                            depth += 1;
                        } else if c == unit(b')') {
                            if depth == 0 {
                                break;
                            }
                            depth -= 1;
                        } else if depth == 0
                            && (c == unit(b',') || c == unit(b';') || c == NL || c == unit(b'}'))
                        {
                            break;
                        }
                        j += 1;
                    }
                    let expr = js::trim(&utf16_string(&src[expr_start..j])).to_owned();
                    self.add(TokenType::Func, expr, start_line, start_col, j);
                    self.col += j - i;
                    self.i = j;
                    continue;
                }
            }

            if ch == unit(b'.') && is_digit(self.at(i + 1)) {
                let mut j = i + 1;
                while j < len && is_digit(self.at(j)) {
                    j += 1;
                }
                let lexeme = utf16_string(&src[i..j]);
                self.add(TokenType::Number, lexeme, start_line, start_col, j);
                self.col += j - i;
                self.i = j;
                continue;
            }
            let punctuation = match ch {
                0x2e => Some((TokenType::Dot, ".")),
                0x28 => Some((TokenType::LParen, "(")),
                0x29 => Some((TokenType::RParen, ")")),
                0x7b => Some((TokenType::LBrace, "{")),
                0x7d => Some((TokenType::RBrace, "}")),
                0x5b => Some((TokenType::LBracket, "[")),
                0x5d => Some((TokenType::RBracket, "]")),
                0x2c => Some((TokenType::Comma, ",")),
                0x3a => Some((TokenType::Colon, ":")),
                0x3d => Some((TokenType::Equal, "=")),
                0x3b => Some((TokenType::Semicolon, ";")),
                0x2b => Some((TokenType::Plus, "+")),
                0x2d => Some((TokenType::Minus, "-")),
                0x2a => Some((TokenType::Star, "*")),
                0x2f => Some((TokenType::Slash, "/")),
                _ => None,
            };
            if let Some((kind, lexeme)) = punctuation {
                self.punct(kind, lexeme, start_line, start_col);
                continue;
            }

            // Triple-quoted strings (multi-line) - must check before single quotes
            if ch == unit(b'"') && self.is(i + 1, b'"') && self.is(i + 2, b'"') {
                let mut j = i + 3;
                // Find closing """ (`while (j < src.length - 2)`)
                while j + 2 < len {
                    if src[j] == unit(b'"') && src[j + 1] == unit(b'"') && src[j + 2] == unit(b'"')
                    {
                        break;
                    }
                    if src[j] == NL {
                        self.line += 1;
                        self.col = 0; // Will be set correctly after loop
                    }
                    j += 1;
                }
                if j + 2 >= len
                    || !(src[j] == unit(b'"')
                        && src[j + 1] == unit(b'"')
                        && src[j + 2] == unit(b'"'))
                {
                    return Err(self.fail(
                        "L002",
                        format!(
                            "Unterminated triple-quoted string at line {start_line} col {start_col}"
                        ),
                        i,
                        len,
                    ));
                }
                // Extract string content without the triple quotes
                let content = &src[i + 3..j];
                self.add(
                    TokenType::String,
                    utf16_string(content),
                    start_line,
                    start_col,
                    j + 3,
                );
                // Update position past closing """
                let lines: Vec<&[u16]> = content.split(|&c| c == NL).collect();
                if lines.len() > 1 {
                    // +3 for closing """ +1 for next char
                    self.col = lines[lines.len() - 1].len() + 4;
                } else {
                    self.col += j - i + 3;
                }
                self.i = j + 3;
                continue;
            }

            if ch == unit(b'"') || ch == unit(b'\'') {
                let quote = ch;
                let mut j = i + 1;
                while j < len && src[j] != quote && src[j] != NL {
                    // Handle escape sequences
                    if src[j] == unit(b'\\') && j + 1 < len {
                        j += 2;
                    } else {
                        j += 1;
                    }
                }
                if j >= len || src[j] == NL {
                    return Err(self.fail(
                        "L002",
                        format!(
                            "Unterminated string literal at line {} col {}",
                            self.line, self.col
                        ),
                        i,
                        j,
                    ));
                }
                // Extract string content without quotes
                let content = utf16_string(&src[i + 1..j]);
                self.add(TokenType::String, content, start_line, start_col, j + 1);
                self.col += j - i + 1;
                self.i = j + 1;
                continue;
            }

            if is_digit(Some(ch)) {
                let mut j = i;
                while j < len && is_digit(self.at(j)) {
                    j += 1;
                }
                if self.is(j, b'.') && is_digit(self.at(j + 1)) {
                    j += 1;
                    while j < len && is_digit(self.at(j)) {
                        j += 1;
                    }
                }
                let lexeme = utf16_string(&src[i..j]);
                self.add(TokenType::Number, lexeme, start_line, start_col, j);
                self.col += j - i;
                self.i = j;
                continue;
            }

            if is_letter(Some(ch)) || ch == unit(b'_') {
                let mut j = i;
                while j < len
                    && (is_letter(self.at(j)) || is_digit(self.at(j)) || src[j] == unit(b'_'))
                {
                    j += 1;
                }
                let lexeme = utf16_string(&src[i..j]);
                let kind = keyword_lookup(&lexeme).unwrap_or(TokenType::Ident);
                self.add(kind, lexeme, start_line, start_col, j);
                self.col += j - i;
                self.i = j;
                continue;
            }

            return Err(self.fail(
                "L001",
                format!(
                    "Unexpected character '{}' at line {} col {}",
                    utf16_string(&[ch]),
                    self.line,
                    self.col
                ),
                i,
                i + 1,
            ));
        }

        let (line, col) = (self.line, self.col);
        self.add(TokenType::Eof, String::new(), line, col, len);
        Ok(self.tokens)
    }
}

/// `lex(src)` with the structured diagnostic of a failure.
pub fn lex_with_diagnostics(src: &str) -> Result<Vec<Token>, DiagnosticError> {
    let units: Vec<u16> = src.encode_utf16().collect();
    Lexer {
        src: &units,
        tokens: Vec::new(),
        i: 0,
        line: 1,
        col: 1,
        src_line: 1,
        src_col: 1,
        anchor: 0,
    }
    .run()
}

/// `lex(src)`: the token array, `{type, lexeme, line, col}` per token, ending
/// with `EOF`.
pub fn lex(src: &str) -> Result<Vec<Token>, JsError> {
    lex_with_diagnostics(src).map_err(JsError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_error(src: &str) -> DiagnosticError {
        lex_with_diagnostics(src).expect_err("lexing should fail")
    }

    fn token_json(src: &str) -> String {
        tokens_to_value(&lex(src).unwrap()).to_json().unwrap()
    }

    #[test]
    fn inherited_keyword_lookups_follow_the_reference() {
        assert_eq!(
            token_json("constructor __proto__ toString"),
            r#"[{"lexeme":"constructor","line":1,"col":1},{"type":{},"lexeme":"__proto__","line":1,"col":13},{"lexeme":"toString","line":1,"col":23},{"type":"EOF","lexeme":"","line":1,"col":31}]"#
        );
        let tokens = lex("constructor __proto__ hasOwnProperty").unwrap();
        assert_eq!(
            tokens[0].kind.to_js_string(),
            "function Object() { [native code] }"
        );
        assert_eq!(tokens[1].kind.to_js_string(), "[object Object]");
        assert_eq!(
            tokens[2].kind.to_js_string(),
            "function hasOwnProperty() { [native code] }"
        );
    }

    /// `(source, code, message, (line, column), (start, end))`.
    type LexerFailure = (
        &'static str,
        &'static str,
        &'static str,
        (f64, f64),
        (f64, f64),
    );

    // test_diagnostic_locations.js: lexerFailures.
    #[test]
    fn lexer_failures_carry_source_diagnostics() {
        let cases: &[LexerFailure] = &[
            (
                "// 😀\r\n\t@",
                "L001",
                "Unexpected character '@' at line 2 col 2",
                (2.0, 2.0),
                (8.0, 9.0),
            ),
            (
                "\"abc",
                "L002",
                "Unterminated string literal at line 1 col 1",
                (1.0, 1.0),
                (0.0, 4.0),
            ),
            (
                " 'abc\nnext",
                "L002",
                "Unterminated string literal at line 1 col 2",
                (1.0, 2.0),
                (1.0, 5.0),
            ),
            (
                "\n  \"\"\"a\nb",
                "L002",
                "Unterminated triple-quoted string at line 2 col 3",
                (2.0, 3.0),
                (3.0, 9.0),
            ),
            (
                "\n /* a\nb",
                "L003",
                "Unterminated comment at line 2 col 2",
                (2.0, 2.0),
                (2.0, 8.0),
            ),
            (
                "search synth\nrender(o99)",
                "L004",
                "Output surface reference 'o99' is out of range; expected o0-o7 at line 2 col 8",
                (2.0, 8.0),
                (20.0, 23.0),
            ),
            (
                "\"😀\" @",
                "L001",
                "Unexpected character '@' at line 1 col 6",
                (1.0, 6.0),
                (5.0, 6.0),
            ),
            (
                "() => (1\n + 2), @",
                "L001",
                "Unexpected character '@' at line 1 col 17",
                (2.0, 8.0),
                (16.0, 17.0),
            ),
            (
                "\"a\\\nb\" @",
                "L001",
                "Unexpected character '@' at line 1 col 8",
                (2.0, 4.0),
                (7.0, 8.0),
            ),
        ];
        for &(src, code, message, (line, column), (start, end)) in cases {
            let err = lex_error(src);
            assert_eq!(err.error, JsError::syntax(message), "{src:?}");
            assert_eq!(
                err.diagnostic.map(|d| *d),
                Some(Diagnostic {
                    code,
                    stage: "lexer",
                    severity: "error",
                    message: message.to_owned(),
                    location: Some(Location { line, column }),
                    span: Some(Span { start, end }),
                }),
                "{src:?}"
            );
        }
    }

    #[test]
    fn successful_tokens_keep_their_public_shape_with_positions() {
        let tokens = lex("/*x*/\nfoo.o99 \"😀\"").unwrap();
        assert_eq!(
            tokens_to_value(&tokens).to_json().unwrap(),
            r#"[{"type":"COMMENT","lexeme":"/*x*/","line":1,"col":1},{"type":"IDENT","lexeme":"foo","line":2,"col":1},{"type":"DOT","lexeme":".","line":2,"col":4},{"type":"OUTPUT_REF","lexeme":"o99","line":2,"col":5},{"type":"STRING","lexeme":"😀","line":2,"col":9},{"type":"EOF","lexeme":"","line":2,"col":13}]"#
        );
        assert_eq!(
            tokens[1].position,
            Some(Position {
                line: 2,
                column: 1,
                start: 6,
                end: 9
            })
        );
    }

    // test_diagnostic_locations.js: token positions match an independent
    // source-walk oracle across scanner constructs.
    #[test]
    fn token_positions_match_a_source_walk() {
        let sources = [
            "search synth\nrender o0",
            "// 😀\r\nsearch synth\r\n\tlet x = \"a\\\nb\"; render o0",
            "search synth\nlet x = () => (1\n + 2); render o0",
            "/* a\nb */ search synth\n\"\"\"\nmulti\nline\n\"\"\" render(o0)",
            "search synth\nlet x = [1 2]; let y = \"😀\"; let z = o0",
            "search synth\nread(o0).subchain(name: \"s\") { .diagFilter() }.write(o1)",
        ];
        for src in sources {
            let units: Vec<u16> = src.encode_utf16().collect();
            for token in lex(src).unwrap() {
                let p = token.position.unwrap();
                assert!(p.end >= p.start && p.end <= units.len());
                assert!(p.end > p.start || token.kind == TokenType::Eof);
                let (mut line, mut column) = (1, 1);
                for &u in &units[..p.start] {
                    if u == NL {
                        line += 1;
                        column = 1;
                    } else {
                        column += 1;
                    }
                }
                assert_eq!(
                    (p.line, p.column),
                    (line, column),
                    "{:?} in {src:?}",
                    token.kind
                );
            }
        }
    }

    #[test]
    fn reference_families_and_member_segments() {
        let tokens = lex("s99 vol99 geo99 xyz99 vel99 rgba99 mesh99 foo.o99").unwrap();
        let kinds: Vec<_> = tokens.iter().map(|t| (t.kind, t.lexeme.as_str())).collect();
        assert_eq!(
            kinds,
            [
                (TokenType::SourceRef, "s99"),
                (TokenType::VolRef, "vol99"),
                (TokenType::GeoRef, "geo99"),
                (TokenType::XyzRef, "xyz99"),
                (TokenType::VelRef, "vel99"),
                (TokenType::RgbaRef, "rgba99"),
                (TokenType::MeshRef, "mesh99"),
                (TokenType::Ident, "foo"),
                (TokenType::Dot, "."),
                (TokenType::OutputRef, "o99"),
                (TokenType::Eof, ""),
            ]
        );
        assert_eq!(
            lex("read(o8)").unwrap_err(),
            JsError::syntax(
                "Output surface reference 'o8' is out of range; expected o0-o7 at line 1 col 6"
            )
        );
    }

    #[test]
    fn legacy_coordinates_drift_like_the_reference() {
        // A function token spanning lines inside parentheses does not advance the
        // legacy line; the triple-quoted string resets the column to its last line.
        let tokens = lex("() => (1\n + 2), x\n\"\"\"a\nbc\"\"\" y").unwrap();
        let coords: Vec<_> = tokens
            .iter()
            .map(|t| (t.lexeme.as_str(), t.line.unwrap(), t.col.unwrap()))
            .collect();
        assert_eq!(
            coords,
            [
                ("(1\n + 2)", 1.0, 1.0),
                (",", 1.0, 15.0),
                ("x", 1.0, 17.0),
                ("a\nbc", 2.0, 1.0),
                ("y", 3.0, 7.0),
                ("", 3.0, 8.0),
            ]
        );
    }
}
