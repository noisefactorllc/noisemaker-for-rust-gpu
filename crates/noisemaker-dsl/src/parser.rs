//! Port of `lang/parser.js`: the recursive-descent parser of the Polymorphic DSL.
//!
//! Grammar (EBNF, from the reference):
//!
//! ```text
//! Program        ::= SearchDirective? Statement* RenderDirective?
//! SearchDirective::= 'search' Ident (',' Ident)*
//! Statement      ::= VarAssign | ChainStmt | IfStmt | Break | Continue | Return
//! RenderDirective::= 'render' '(' OutputRef ')'
//! Block          ::= '{' Statement* '}'
//! IfStmt         ::= 'if' '(' Expr ')' Block ('elif' '(' Expr ')' Block)* ('else' Block)?
//! VarAssign      ::= 'let' Ident '=' Expr
//! Chain          ::= ChainElement ('.' ChainElement)*
//! ChainElement   ::= Call | WriteNode | Write3DNode | Subchain
//! Call           ::= Ident '(' ArgList? ')'
//! NumberExpr     ::= Number | 'Math.PI' | '(' NumberExpr ')' | NumberExpr ( '+' | '-' | '*' | '/' ) NumberExpr
//! Member         ::= Ident ('.' Ident)+
//! Func           ::= '(' ')' '=>' Expr
//! ```
//!
//! The AST is a [`Value`] shaped exactly like the reference AST: the same `type`
//! strings, the same members in the same insertion order, and `undefined` members
//! where the reference leaves them `undefined` (they serialize away, as with
//! `JSON.stringify`).
//!
//! Non-enumerable metadata. The reference attaches three non-enumerable
//! properties:
//!
//! * `error.diagnostic` on every thrown `SyntaxError`: returned by
//!   [`parse_with_options`] in [`DiagnosticError::diagnostic`];
//! * `position` on `ArrayLiteral` nodes: read only by the parser's own number
//!   coercion diagnostics, so it travels with the parser's expression results
//!   and is not stored in the AST;
//! * `subchainArgumentDiagnostics` on `Subchain` nodes (GAP-027 reports that
//!   `validate()` surfaces): a hidden (non-enumerable) member of the node
//!   ([`Object::define_hidden`]), which never serializes and does not survive a
//!   JSON round trip. Read it with [`subchain_argument_diagnostics`].

use crate::diagnostics;
use crate::error::JsError;
use crate::js;
use crate::lexer::{
    Diagnostic, DiagnosticError, Location, Position, Span, Token, TokenType, coordinate_string,
    coordinate_value,
};
use crate::registry::Registry;
use crate::value::{Object, Value};

/// The options of `parse(tokens, options)`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParseOptions {
    /// `options.subchainArguments`. `'strict'` (GAP-027) rejects unknown,
    /// duplicate and unseparated subchain arguments with `SyntaxError`s; any
    /// other value keeps the default acceptance and attaches the reports to the
    /// Subchain node instead. The host (`compile(src)`) passes `{}`.
    pub subchain_arguments: Option<String>,
}

impl ParseOptions {
    /// `{ subchainArguments: 'strict' }`.
    pub fn strict_subchain_arguments() -> ParseOptions {
        ParseOptions {
            subchain_arguments: Some("strict".to_owned()),
        }
    }

    /// The options read from a JavaScript options object.
    pub fn from_value(options: &Value) -> ParseOptions {
        ParseOptions {
            subchain_arguments: options.get("subchainArguments").as_str().map(str::to_owned),
        }
    }
}

/// The Subchain node property that holds the parser's subchain-argument reports.
pub const SUBCHAIN_ARGUMENT_DIAGNOSTICS: &str = "subchainArgumentDiagnostics";

/// `node.subchainArgumentDiagnostics`: the GAP-027 reports the parser attached
/// to a Subchain node (an array of `{code, message, severity, location?,
/// span?}`), or `undefined` when it has none. An enumerable array stored under
/// the same key (a hand-built AST) reads back as well, as it would in the
/// reference.
pub fn subchain_argument_diagnostics(node: &Value) -> Value {
    node.get(SUBCHAIN_ARGUMENT_DIAGNOSTICS).clone()
}

/// `Object.defineProperty(node, key, {value, enumerable: false})`.
fn define_non_enumerable(node: &mut Object, key: &str, value: Value) {
    node.define_hidden(key, value);
}

/// `parse(tokens)` as the host calls it (`compile(src)` passes `options = {}`)
/// with the reference's default namespace table (`runtime/tags.js` before any
/// `registerNamespace`: [`crate::registry::BUILTIN_NAMESPACES`]): the Program
/// AST, or the error the reference throws. [`parse_with_registry`] validates
/// `search` against a registry's live namespaces instead.
pub fn parse(tokens: &[Token]) -> Result<Value, JsError> {
    parse_with_registry(tokens, &Registry::new())
}

/// `parse(tokens)` with `search` validated against `registry.namespaces` (the
/// live `VALID_NAMESPACES`).
pub fn parse_with_registry(tokens: &[Token], registry: &Registry) -> Result<Value, JsError> {
    parse_with_options(tokens, registry, &ParseOptions::default()).map_err(JsError::from)
}

/// `parse(tokens, options)`, with the structured diagnostic of a failure.
///
/// `registry` supplies the namespaces of `runtime/tags.js` (`isValidNamespace`,
/// `VALID_NAMESPACES`) that the `search` directive is validated against.
pub fn parse_with_options(
    tokens: &[Token],
    registry: &Registry,
    options: &ParseOptions,
) -> Result<Value, DiagnosticError> {
    let mut parser = Parser {
        tokens,
        current: 0,
        // GAP-027: opt-in strict subchain-argument validation. Default parsing
        // acceptance is unchanged; strict mode rejects unknown keys, duplicate
        // keys, and missing separators with stable diagnostic codes.
        strict_subchain_arguments: options.subchain_arguments.as_deref() == Some("strict"),
        // Track the search order for the program (set by search directive - REQUIRED)
        program_search_order: None,
        // Program namespace starts empty - must be set by search directive
        program_namespace_imports: Vec::new(),
        program_namespace_default: Value::Null,
        registry,
    };
    parser.parse_program()
}

type PResult<T> = Result<T, DiagnosticError>;

/// What `parserError` reads from the token it is given: the non-enumerable
/// `position` and the legacy `line`/`col`. Besides tokens, the reference passes
/// pseudo-tokens built from AST nodes (`toNumber`) and bare `{line, col}`.
#[derive(Debug, Clone, Copy)]
struct ErrorSite {
    position: Option<Position>,
    line: Option<f64>,
    col: Option<f64>,
}

impl ErrorSite {
    fn of(token: &Token) -> ErrorSite {
        ErrorSite {
            position: token.position,
            line: token.line,
            col: token.col,
        }
    }
}

/// `Number.isInteger(x) && x > 0`.
fn is_positive_integer(x: Option<f64>) -> bool {
    matches!(x, Some(n) if n.is_finite() && n.fract() == 0.0 && n > 0.0)
}

/// An expression result with the node's non-enumerable `position` (only
/// `ArrayLiteral` nodes carry one).
struct Expr {
    node: Value,
    position: Option<Position>,
}

impl Expr {
    fn plain(node: Value) -> Expr {
        Expr {
            node,
            position: None,
        }
    }
}

/// `parseChain(context)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChainContext {
    Statement,
    Expression,
}

/// `exprStartTokens.has(type)`.
fn is_expr_start(kind: TokenType) -> bool {
    use TokenType::*;
    matches!(
        kind,
        Plus | Minus
            | Number
            | Hex
            | Func
            | String
            | Ident
            | OutputRef
            | SourceRef
            | VolRef
            | GeoRef
            | MeshRef
            | XyzRef
            | VelRef
            | RgbaRef
            | LParen
            | LBracket
            | True
            | False
    )
}

/// `memberTokenTypes.has(type)`.
fn is_member_token(kind: TokenType) -> bool {
    use TokenType::*;
    matches!(
        kind,
        Ident
            | SourceRef
            | OutputRef
            | VolRef
            | GeoRef
            | MeshRef
            | XyzRef
            | VelRef
            | RgbaRef
            | Let
            | Render
            | True
            | False
            | If
            | Elif
            | Else
            | Break
            | Continue
            | Return
            | Write
            | Write3d
            | Subchain
    )
}

/// `namespaceTokenTypes.has(type)`: token types that can be used as namespace
/// identifiers (keywords like 'render' are valid namespace names in search
/// context).
fn is_namespace_token(kind: TokenType) -> bool {
    use TokenType::*;
    matches!(
        kind,
        Ident
            | Render
            | Write
            | Write3d
            | True
            | False
            | If
            | Elif
            | Else
            | Break
            | Continue
            | Return
    )
}

// GAP-027: subchain arguments accept exactly these keyword keys. A single
// leading positional string literal is shorthand for `name`. Anything
// else is historically accepted but discarded, and is now reported.
const SUBCHAIN_KEYS: [&str; 2] = ["name", "id"];

// --- AST node builders ------------------------------------------------------------

fn object(entries: Vec<(&str, Value)>) -> Object {
    let mut o = Object::new();
    for (k, v) in entries {
        o.insert(k, v);
    }
    o
}

fn node(entries: Vec<(&str, Value)>) -> Value {
    Value::Object(object(entries))
}

/// `{type: 'Number', value}`.
fn number_node(value: f64) -> Value {
    node(vec![
        ("type", "Number".into()),
        ("value", Value::Number(value)),
    ])
}

/// `{type: 'String', value}`.
fn string_node(value: &str) -> Value {
    node(vec![("type", "String".into()), ("value", value.into())])
}

/// `{type, name}` (surface and source references, `Ident`).
fn named_node(kind: &str, name: &str) -> Value {
    node(vec![("type", kind.into()), ("name", name.into())])
}

/// `{type: 'Member', path}`.
fn member_node(path: &[&str]) -> Value {
    node(vec![
        ("type", "Member".into()),
        (
            "path",
            Value::Array(path.iter().map(|s| Value::from(*s)).collect()),
        ),
    ])
}

/// `{ line, col }` of a token.
fn loc_value(line: Option<f64>, col: Option<f64>) -> Value {
    node(vec![
        ("line", coordinate_value(line)),
        ("col", coordinate_value(col)),
    ])
}

fn comments_value(comments: Vec<String>) -> Value {
    Value::Array(comments.into_iter().map(Value::from).collect())
}

/// `value === undefined` for an object member read (absent or `undefined`).
fn member_is_undefined(o: &Object, key: &str) -> bool {
    o.get_or_undefined(key).is_undefined()
}

/// `a || b`.
fn or(a: &Value, b: &Value) -> Value {
    if a.is_truthy() { a.clone() } else { b.clone() }
}

/// `Array.isArray(call.args) ? call.args : []`.
fn call_args(call: &Object) -> Vec<Value> {
    call.get_or_undefined("args")
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// `call.kwargs || {}`.
fn call_kwargs(call: &Object) -> Object {
    match call.get_or_undefined("kwargs") {
        Value::Object(o) => o.clone(),
        _ => Object::new(),
    }
}

/// `Array.prototype.join` over a path of segments.
fn join_path(path: &Value, separator: &str) -> String {
    match path {
        Value::Array(segments) => segments
            .iter()
            .map(|s| {
                if s.is_nullish() {
                    String::new()
                } else {
                    js::value_to_property_key(s)
                }
            })
            .collect::<Vec<_>>()
            .join(separator),
        other => js::value_to_property_key(other),
    }
}

/// The recursive-descent parser (the reference's `parse` closure state).
struct Parser<'a> {
    tokens: &'a [Token],
    current: usize,
    strict_subchain_arguments: bool,
    program_search_order: Option<Vec<String>>,
    program_namespace_imports: Vec<Value>,
    program_namespace_default: Value,
    registry: &'a Registry,
}

impl<'a> Parser<'a> {
    // --- token access ------------------------------------------------------------

    /// `peek()`; reading `.type` of a missing token throws the reference's
    /// `TypeError` (only possible for caller-built token arrays without `EOF`).
    fn peek(&self) -> PResult<&'a Token> {
        self.tokens.get(self.current).ok_or_else(|| {
            DiagnosticError::plain(JsError::type_error(
                "Cannot read properties of undefined (reading 'type')",
            ))
        })
    }

    /// `advance()`. Every call site has checked the token at `current` first.
    fn advance(&mut self) -> &'a Token {
        let token = &self.tokens[self.current];
        self.current += 1;
        token
    }

    /// `tokens[index]?.type`.
    fn type_at(&self, index: usize) -> Option<TokenType> {
        self.tokens.get(index).map(|t| t.kind)
    }

    /// `parserError(code, message, token, severityOverride)`.
    fn parser_error(
        &self,
        code: &'static str,
        message: String,
        token: Option<ErrorSite>,
        severity_override: Option<&'static str>,
    ) -> DiagnosticError {
        // Prefer the lexer's non-enumerable source-derived position when the
        // token came from lex(source). Caller-supplied tokens fall back to
        // their own positive-integer line/col with an explicitly null span.
        let position = token.and_then(|t| t.position).filter(Position::is_valid);
        let has_location =
            token.is_some_and(|t| is_positive_integer(t.line) && is_positive_integer(t.col));
        let location = match (position, token) {
            (Some(p), _) => Some(Location {
                line: p.line as f64,
                column: p.column as f64,
            }),
            (None, Some(t)) if has_location => Some(Location {
                line: t.line.unwrap_or_default(),
                column: t.col.unwrap_or_default(),
            }),
            _ => None,
        };
        let span = position.map(|p| Span {
            start: p.start as f64,
            end: p.end as f64,
        });
        let mut diagnostic = Diagnostic::new(code, message, location, span);
        if let Some(severity) = severity_override {
            diagnostic.severity = severity;
        }
        DiagnosticError::syntax(diagnostic)
    }

    /// `parserError(code, `${text} at line ${t.line} col ${t.col}`, t)`.
    fn error_at(&self, code: &'static str, text: &str, token: &Token) -> DiagnosticError {
        self.parser_error(
            code,
            format!(
                "{text} at line {} col {}",
                coordinate_string(token.line),
                coordinate_string(token.col)
            ),
            Some(ErrorSite::of(token)),
            None,
        )
    }

    /// `expect(type, msg)`.
    fn expect(&mut self, kind: TokenType, msg: &str) -> PResult<&'a Token> {
        let token = self.peek()?;
        if token.kind == kind {
            return Ok(self.advance());
        }
        let code = if kind == TokenType::RParen {
            "P002"
        } else {
            "P001"
        };
        Err(self.error_at(code, msg, token))
    }

    /// `collectComments()`: collect and consume any pending COMMENT tokens.
    fn collect_comments(&mut self) -> Vec<String> {
        let mut comments = Vec::new();
        while self.type_at(self.current) == Some(TokenType::Comment) {
            comments.push(self.advance().lexeme.clone());
        }
        comments
    }

    /// `cloneNamespaceMeta(meta)`: a deep copy (`structuredClone`), `null` for
    /// a non-object.
    fn clone_namespace_meta(meta: &Value) -> Value {
        match meta {
            Value::Object(_) => meta.clone(),
            _ => Value::Null,
        }
    }

    // --- automation and namespace call transforms ----------------------------------

    /// `transformOscInvocation(call, nameToken)`: an `osc()` call as an
    /// Oscillator node.
    ///
    /// `osc(type, min?, max?, speed?, offset?, seed?)`; all params except 'type'
    /// are optional and support kwargs.
    fn transform_osc_invocation(&self, call: &Object, name_token: &Token) -> PResult<Value> {
        let args = call_args(call);
        let kwargs = call_kwargs(call);

        // Parameter order: type, min, max, speed, offset, seed
        let param_order = ["type", "min", "max", "speed", "offset", "seed"];
        let defaults = object(vec![
            ("type", member_node(&["oscKind", "sine"])),
            ("min", number_node(0.0)),
            ("max", number_node(1.0)),
            ("speed", number_node(1.0)),
            ("offset", number_node(0.0)),
            ("seed", number_node(1.0)),
        ]);

        // Validate kwargs - reject unknown parameters
        for key in kwargs.keys() {
            if !param_order.contains(&key.as_str()) {
                return Err(self.parser_error(
                    "P003",
                    format!(
                        "osc() unknown parameter '{key}' at line {} col {}. Valid: {}",
                        coordinate_string(name_token.line),
                        coordinate_string(name_token.col),
                        param_order.join(", ")
                    ),
                    Some(ErrorSite::of(name_token)),
                    None,
                ));
            }
        }

        let mut resolved = Object::new();
        // Resolve each parameter from positional args or kwargs
        for (i, param_name) in param_order.iter().enumerate() {
            if !member_is_undefined(&kwargs, param_name) {
                resolved.insert(*param_name, kwargs.get_or_undefined(param_name).clone());
            } else if i < args.len() {
                resolved.insert(*param_name, args[i].clone());
            } else if !member_is_undefined(&defaults, param_name) {
                resolved.insert(*param_name, defaults.get_or_undefined(param_name).clone());
            }
        }

        let type_node = resolved.get_or_undefined("type").clone();

        Ok(node(vec![
            ("type", "Oscillator".into()),
            ("oscType", type_node),
            ("min", resolved.get_or_undefined("min").clone()),
            ("max", resolved.get_or_undefined("max").clone()),
            ("speed", resolved.get_or_undefined("speed").clone()),
            ("offset", resolved.get_or_undefined("offset").clone()),
            ("seed", resolved.get_or_undefined("seed").clone()),
            ("loc", loc_value(name_token.line, name_token.col)),
        ]))
    }

    /// The P003 error of an automation transform: `${text} at line .. col ..`.
    fn automation_error(&self, text: &str, name_token: &Token) -> DiagnosticError {
        self.error_at("P003", text, name_token)
    }

    /// Resolve the positional/keyword parameters of `midi()`/`audio()`.
    /// Positionals are dense, so keyword arguments do not consume their slots.
    /// Returns the resolved parameters and the positional cursor.
    fn resolve_dense(
        param_order: &[&str],
        args: &[Value],
        kwargs: &Object,
        defaults: &Object,
    ) -> (Object, usize) {
        let mut resolved = Object::new();
        let mut pos_cursor = 0;
        for param_name in param_order {
            if !member_is_undefined(kwargs, param_name) {
                resolved.insert(*param_name, kwargs.get_or_undefined(param_name).clone());
            } else if pos_cursor < args.len() {
                resolved.insert(*param_name, args[pos_cursor].clone());
                pos_cursor += 1;
            } else if !member_is_undefined(defaults, param_name) {
                resolved.insert(*param_name, defaults.get_or_undefined(param_name).clone());
            }
        }
        (resolved, pos_cursor)
    }

    /// The `name`/`id` device-identity checks of `midi()`/`audio()`.
    fn check_identity_strings(
        &self,
        func: &str,
        kwargs: &Object,
        name_token: &Token,
    ) -> PResult<()> {
        for param_name in ["name", "id"] {
            let value = kwargs.get_or_undefined(param_name);
            if value.is_undefined() {
                continue;
            }
            if value.get("type").as_str() != Some("String") {
                return Err(self.automation_error(
                    &format!("{func}() '{param_name}' requires a quoted string"),
                    name_token,
                ));
            }
            let empty = match value.get("value") {
                Value::String(s) => s.is_empty(),
                Value::Array(a) => a.is_empty(),
                _ => false,
            };
            if empty {
                return Err(self.automation_error(
                    &format!("{func}() '{param_name}' must not be empty"),
                    name_token,
                ));
            }
        }
        Ok(())
    }

    /// `transformMidiInvocation(call, nameToken)`: a `midi()` call as a Midi node.
    ///
    /// `midi(channel, mode?, min?, max?, sensitivity?, name:?, id:?, cc:?,
    /// nrpn:?, zone:?, members:?)`.
    fn transform_midi_invocation(&self, call: &Object, name_token: &Token) -> PResult<Value> {
        let args = call_args(call);
        let kwargs = call_kwargs(call);

        // Parameter order: channel, mode, min, max, sensitivity
        let param_order = ["channel", "mode", "min", "max", "sensitivity"];
        let keyword_only_params = ["name", "id", "cc", "nrpn", "zone", "members"];
        let valid_params: Vec<&str> = param_order
            .iter()
            .chain(keyword_only_params.iter())
            .copied()
            .collect();
        if args.len() > param_order.len() {
            return Err(self.automation_error(
                "midi() name, id, cc, nrpn, zone and members are keyword-only",
                name_token,
            ));
        }
        for key in kwargs.keys() {
            if !valid_params.contains(&key.as_str()) {
                return Err(self.parser_error(
                    "P003",
                    format!(
                        "midi() unknown parameter '{key}' at line {} col {}. Valid: {}",
                        coordinate_string(name_token.line),
                        coordinate_string(name_token.col),
                        valid_params.join(", ")
                    ),
                    Some(ErrorSite::of(name_token)),
                    None,
                ));
            }
        }
        let defaults = object(vec![
            ("mode", member_node(&["midiMode", "velocity"])),
            ("min", number_node(0.0)),
            ("max", number_node(1.0)),
            ("sensitivity", number_node(1.0)),
        ]);

        let (resolved, pos_cursor) = Self::resolve_dense(&param_order, &args, &kwargs, &defaults);

        if pos_cursor < args.len() {
            return Err(
                self.automation_error("midi() has an excess positional argument", name_token)
            );
        }

        let channel = resolved.get_or_undefined("channel");
        if !channel.is_truthy() && member_is_undefined(&kwargs, "zone") {
            return Err(
                self.automation_error("midi() requires 'channel' or 'zone' argument", name_token)
            );
        }
        if channel.is_truthy() && !member_is_undefined(&kwargs, "zone") {
            return Err(self.automation_error(
                "midi() 'channel' and 'zone' are mutually exclusive",
                name_token,
            ));
        }
        if !member_is_undefined(&kwargs, "members") && member_is_undefined(&kwargs, "zone") {
            return Err(self.automation_error("midi() 'members' requires 'zone'", name_token));
        }
        if !member_is_undefined(&kwargs, "id") && member_is_undefined(&kwargs, "name") {
            return Err(self.automation_error("midi() 'id' requires readable 'name'", name_token));
        }
        self.check_identity_strings("midi", &kwargs, name_token)?;

        let kw = |key: &str| kwargs.get_or_undefined(key).clone();
        Ok(node(vec![
            ("type", "Midi".into()),
            ("channel", channel.clone()),
            ("mode", resolved.get_or_undefined("mode").clone()),
            ("min", resolved.get_or_undefined("min").clone()),
            ("max", resolved.get_or_undefined("max").clone()),
            (
                "sensitivity",
                resolved.get_or_undefined("sensitivity").clone(),
            ),
            ("cc", kw("cc")),
            ("nrpn", kw("nrpn")),
            ("zone", kw("zone")),
            ("members", kw("members")),
            ("name", kw("name")),
            ("id", kw("id")),
            ("loc", loc_value(name_token.line, name_token.col)),
        ]))
    }

    /// `transformAudioInvocation(call, nameToken)`: an `audio()` call as an
    /// Audio node.
    ///
    /// `audio(band, min?, max?, channel:?, name:?, id:?)`.
    fn transform_audio_invocation(&self, call: &Object, name_token: &Token) -> PResult<Value> {
        let args = call_args(call);
        let kwargs = call_kwargs(call);

        // Parameter order: band, min, max
        let param_order = ["band", "min", "max"];
        let keyword_only_params = ["channel", "name", "id"];
        let valid_params: Vec<&str> = param_order
            .iter()
            .chain(keyword_only_params.iter())
            .copied()
            .collect();
        if args.len() > param_order.len() {
            return Err(
                self.automation_error("audio() channel, name and id are keyword-only", name_token)
            );
        }
        for key in kwargs.keys() {
            if !valid_params.contains(&key.as_str()) {
                return Err(self.parser_error(
                    "P003",
                    format!(
                        "audio() unknown parameter '{key}' at line {} col {}. Valid: {}",
                        coordinate_string(name_token.line),
                        coordinate_string(name_token.col),
                        valid_params.join(", ")
                    ),
                    Some(ErrorSite::of(name_token)),
                    None,
                ));
            }
        }
        let defaults = object(vec![("min", number_node(0.0)), ("max", number_node(1.0))]);

        let (resolved, pos_cursor) = Self::resolve_dense(&param_order, &args, &kwargs, &defaults);

        if pos_cursor < args.len() {
            return Err(
                self.automation_error("audio() has an excess positional argument", name_token)
            );
        }

        if !resolved.get_or_undefined("band").is_truthy() {
            return Err(self.automation_error("audio() requires 'band' argument", name_token));
        }
        if !member_is_undefined(&kwargs, "id") && member_is_undefined(&kwargs, "name") {
            return Err(self.automation_error("audio() 'id' requires readable 'name'", name_token));
        }
        if !member_is_undefined(&kwargs, "name") && member_is_undefined(&kwargs, "channel") {
            return Err(self.automation_error(
                "audio() selected device requires both 'name' and 'channel'",
                name_token,
            ));
        }
        self.check_identity_strings("audio", &kwargs, name_token)?;

        let kw = |key: &str| kwargs.get_or_undefined(key).clone();
        Ok(node(vec![
            ("type", "Audio".into()),
            ("band", resolved.get_or_undefined("band").clone()),
            ("min", resolved.get_or_undefined("min").clone()),
            ("max", resolved.get_or_undefined("max").clone()),
            ("channel", kw("channel")),
            ("name", kw("name")),
            ("id", kw("id")),
            ("loc", loc_value(name_token.line, name_token.col)),
        ]))
    }

    /// `transformFromInvocation(call, nameToken)`: `from(namespace, call())` as
    /// the call with a namespace override that puts the namespace first.
    fn transform_from_invocation(&self, call: &Object, name_token: &Token) -> PResult<Value> {
        let fail = |message: &str| -> DiagnosticError {
            // `typeof nameToken.line === 'number' && typeof nameToken.col === 'number'`
            if name_token.line.is_some() && name_token.col.is_some() {
                return self.error_at("P007", message, name_token);
            }
            self.parser_error(
                "P007",
                message.to_owned(),
                Some(ErrorSite::of(name_token)),
                None,
            )
        };
        if let Value::Object(kwargs) = call.get_or_undefined("kwargs")
            && !kwargs.is_empty()
        {
            return Err(fail("'from' does not support named arguments"));
        }
        let args = call_args(call);
        if args.len() != 2 {
            return Err(fail(
                "'from' requires exactly two arguments (namespace, call)",
            ));
        }
        let namespace_arg = &args[0];
        let target_arg = &args[1];
        let namespace_type = namespace_arg.get("type").as_str();
        if !namespace_arg.is_truthy()
            || (namespace_type != Some("Ident") && namespace_type != Some("Member"))
        {
            return Err(fail("'from' namespace argument must be an identifier"));
        }
        let namespace_name = if namespace_type == Some("Member") {
            join_path(namespace_arg.get("path"), ".")
        } else {
            match namespace_arg.get("name") {
                Value::String(s) => s.clone(),
                other if other.is_truthy() => js::value_to_property_key(other),
                _ => String::new(),
            }
        };
        if namespace_name.is_empty() {
            return Err(fail("'from' namespace argument must be non-empty"));
        }
        let mut target_call: Option<&Object> = None;
        if target_arg.get("type").as_str() == Some("Call") {
            target_call = target_arg.as_object();
        } else if target_arg.get("type").as_str() == Some("Chain")
            && let Value::Array(chain) = target_arg.get("chain")
            && chain.len() == 1
        {
            let head = &chain[0];
            if head.get("type").as_str() == Some("Call") {
                target_call = head.as_object();
            }
        }
        let Some(target_call) = target_call else {
            return Err(fail("'from' second argument must be a call expression"));
        };
        // `{...targetCall, args: [...targetCall.args]}`
        let mut replacement = target_call.clone();
        replacement.insert("args", Value::Array(call_args(target_call)));
        if let kwargs @ Value::Object(_) = target_call.get_or_undefined("kwargs") {
            replacement.insert("kwargs", kwargs.clone());
        }
        // from() creates a namespace override that puts the specified namespace first
        let override_namespace = node(vec![
            ("name", namespace_name.as_str().into()),
            ("path", Value::Array(vec![namespace_name.as_str().into()])),
            ("explicit", true.into()),
            ("source", "from".into()),
            ("resolved", namespace_name.as_str().into()),
            (
                "searchOrder",
                Value::Array(vec![namespace_name.as_str().into()]),
            ),
            ("fromOverride", true.into()),
        ]);
        replacement.insert("namespace", override_namespace);
        Ok(Value::Object(replacement))
    }

    /// `hasCallAfterDot(index)`: whether the member path after `tokens[index]`
    /// ends in a call.
    fn has_call_after_dot(&self, index: usize) -> bool {
        let mut i = index + 1;
        if self.type_at(i) != Some(TokenType::Dot) {
            return false;
        }
        while self.type_at(i) == Some(TokenType::Dot) {
            match self.tokens.get(i + 1) {
                Some(seg) if is_member_token(seg.kind) => {}
                _ => return false,
            }
            i += 2;
        }
        self.type_at(i) == Some(TokenType::LParen)
    }

    // --- program structure ---------------------------------------------------------

    /// `parseRenderDirective()`.
    fn parse_render_directive(&mut self) -> PResult<Value> {
        self.advance();
        self.expect(TokenType::LParen, "Expect '('")?;
        if self.peek()?.kind != TokenType::OutputRef {
            let t = self.peek()?;
            return Err(self.parser_error(
                "P005",
                "Expected output reference in render()".to_owned(),
                Some(ErrorSite::of(t)),
                None,
            ));
        }
        let out = named_node("OutputRef", &self.advance().lexeme);
        self.expect(TokenType::RParen, "Expect ')'")?;
        Ok(out)
    }

    /// `consumeRender()` of `parseProgram`.
    fn consume_render(&mut self, render: &mut Value) -> PResult<()> {
        if render.is_truthy() {
            let t = self.peek()?;
            return Err(self.error_at("P005", "Duplicate render() directive", t));
        }
        *render = self.parse_render_directive()?;
        while self.peek()?.kind == TokenType::Semicolon {
            self.advance();
        }
        Ok(())
    }

    /// `validateNamespace(token)` of `parseSearchDirective`.
    fn validate_namespace(&self, token: &Token) -> PResult<()> {
        let ns = &token.lexeme;
        if !self.registry.is_valid_namespace(ns) {
            let valid: Vec<&str> = self
                .registry
                .namespaces
                .keys()
                .map(String::as_str)
                .collect();
            return Err(self.parser_error(
                "P004",
                format!(
                    "Invalid namespace '{ns}' at line {} col {}. Valid namespaces: {}",
                    coordinate_string(token.line),
                    coordinate_string(token.col),
                    valid.join(", ")
                ),
                Some(ErrorSite::of(token)),
                None,
            ));
        }
        Ok(())
    }

    /// `parseSearchDirective()`: `search ns1, ns2, ns3`.
    fn parse_search_directive(&mut self) -> PResult<()> {
        if self.program_search_order.is_some() {
            let t = self.peek()?;
            return Err(self.error_at(
                "P004",
                "Only one search directive is allowed per program",
                t,
            ));
        }
        self.advance(); // consume 'search'
        let mut namespaces: Vec<String> = Vec::new();

        // Expect at least one namespace identifier (allow keywords as namespace names)
        let first_token = self.peek()?;
        if !is_namespace_token(first_token.kind) {
            return Err(self.error_at(
                "P004",
                "Expected namespace identifier after search",
                first_token,
            ));
        }
        self.advance();
        self.validate_namespace(first_token)?;
        namespaces.push(first_token.lexeme.clone());
        // Parse additional comma-separated namespaces
        while self.peek()?.kind == TokenType::Comma {
            self.advance(); // consume ','
            let ns_token = self.peek()?;
            if !is_namespace_token(ns_token.kind) {
                return Err(self.error_at(
                    "P004",
                    "Expected namespace identifier after comma",
                    ns_token,
                ));
            }
            self.advance();
            self.validate_namespace(ns_token)?;
            namespaces.push(ns_token.lexeme.clone());
        }
        // Update the programNamespace to reflect the explicit search order
        self.program_namespace_imports = namespaces
            .iter()
            .map(|name| {
                node(vec![
                    ("name", name.as_str().into()),
                    ("source", "search".into()),
                    ("explicit", true.into()),
                ])
            })
            .collect();
        self.program_namespace_default = node(vec![
            ("name", namespaces[0].as_str().into()),
            ("source", "search".into()),
            ("explicit", true.into()),
        ]);
        self.program_search_order = Some(namespaces);
        while self.peek()?.kind == TokenType::Semicolon {
            self.advance();
        }
        Ok(())
    }

    /// `parseProgram()`.
    fn parse_program(&mut self) -> PResult<Value> {
        let mut plans: Vec<Value> = Vec::new();
        let mut vars: Vec<Value> = Vec::new();
        let mut render = Value::Null;
        let mut trailing_comments: Vec<String> = Vec::new();

        while self.peek()?.kind != TokenType::Eof {
            if self.peek()?.kind == TokenType::Semicolon {
                self.advance();
                continue;
            }
            // Collect any leading comments before this statement
            let leading_comments = self.collect_comments();
            if self.peek()?.kind == TokenType::Eof {
                // Trailing comments at end of program
                trailing_comments.extend(leading_comments);
                break;
            }
            if self.peek()?.kind == TokenType::Semicolon {
                continue;
            }
            if self.peek()?.kind == TokenType::Search {
                if !plans.is_empty() || !vars.is_empty() || render.is_truthy() {
                    let t = self.peek()?;
                    return Err(self.error_at(
                        "P004",
                        "'search' directive must appear before other statements",
                        t,
                    ));
                }
                self.parse_search_directive()?;
                continue;
            }
            if self.peek()?.kind == TokenType::Render {
                self.consume_render(&mut render)?;
                // Attach leading comments to render if present
                if !leading_comments.is_empty() && render.is_truthy() {
                    render.set("leadingComments", comments_value(leading_comments));
                }
                // Collect any trailing comments after render
                let trailing = self.collect_comments();
                trailing_comments.extend(trailing);
                break;
            }
            let mut stmt = self.parse_statement()?;
            // Attach leading comments to the statement
            if !leading_comments.is_empty() {
                stmt.set("leadingComments", comments_value(leading_comments));
            }
            // appendStatement(stmt)
            if stmt.get("type").as_str() == Some("VarAssign") {
                vars.push(stmt);
            } else {
                plans.push(stmt);
            }
            while self.peek()?.kind == TokenType::Semicolon {
                self.advance();
            }
        }
        let eof = self.expect(TokenType::Eof, "Expected end of input")?;
        let Some(program_search_order) = self
            .program_search_order
            .clone()
            .filter(|order| !order.is_empty())
        else {
            return Err(self.parser_error(
                "P004",
                "Missing required 'search' directive. Every program must start with 'search <namespace>, ...' to specify namespace search order.".to_owned(),
                Some(ErrorSite::of(eof)),
                None,
            ));
        };

        let mut program = object(vec![
            ("type", "Program".into()),
            ("plans", Value::Array(plans)),
            ("render", render),
        ]);
        if !vars.is_empty() {
            program.insert("vars", Value::Array(vars));
        }
        if !trailing_comments.is_empty() {
            program.insert("trailingComments", comments_value(trailing_comments));
        }

        let search_order: Vec<Value> = program_search_order
            .iter()
            .map(|s| Value::from(s.as_str()))
            .collect();
        let mut namespace_meta = Self::clone_namespace_meta(&node(vec![
            (
                "imports",
                Value::Array(self.program_namespace_imports.clone()),
            ),
            ("default", self.program_namespace_default.clone()),
            ("searchOrder", Value::Array(search_order.clone())),
        ]));
        if !namespace_meta.is_truthy() {
            let imports_clone = self.program_namespace_imports.clone();
            let default_clone = if self.program_namespace_default.is_truthy() {
                self.program_namespace_default.clone()
            } else {
                Value::Null
            };
            namespace_meta = node(vec![
                ("imports", Value::Array(imports_clone)),
                ("default", default_clone),
                ("searchOrder", Value::Array(search_order)),
            ]);
        }
        program.insert("namespace", namespace_meta);
        Ok(Value::Object(program))
    }

    /// `parseBlock()`.
    fn parse_block(&mut self) -> PResult<Vec<Value>> {
        self.expect(TokenType::LBrace, "Expect '{'")?;
        let mut body = Vec::new();
        while self.peek()?.kind != TokenType::RBrace {
            let stmt = self.parse_statement()?;
            body.push(stmt);
            while self.peek()?.kind == TokenType::Semicolon {
                self.advance();
            }
        }
        self.expect(TokenType::RBrace, "Expect '}'")?;
        Ok(body)
    }

    /// `parseStatement()`.
    fn parse_statement(&mut self) -> PResult<Value> {
        if self.peek()?.kind == TokenType::Search {
            let t = self.peek()?;
            return Err(self.error_at(
                "P004",
                "'search' directive is only allowed at the start of the program",
                t,
            ));
        }
        if self.peek()?.kind == TokenType::Let {
            self.advance();
            let name = self
                .expect(TokenType::Ident, "Expected identifier")?
                .lexeme
                .clone();
            self.expect(TokenType::Equal, "Expect '='")?;
            if !is_expr_start(self.peek()?.kind) {
                let t = self.peek()?;
                return Err(self.error_at("P001", "Expected expression after '='", t));
            }
            let expr = self.parse_additive()?.node;
            return Ok(node(vec![
                ("type", "VarAssign".into()),
                ("name", name.into()),
                ("expr", expr),
            ]));
        }

        match self.peek()?.kind {
            TokenType::If => {
                self.advance();
                self.expect(TokenType::LParen, "Expect '('")?;
                let condition = self.parse_additive()?.node;
                self.expect(TokenType::RParen, "Expect ')'")?;
                let then = self.parse_block()?;
                let mut elif = Vec::new();
                while self.peek()?.kind == TokenType::Elif {
                    self.advance();
                    self.expect(TokenType::LParen, "Expect '('")?;
                    let ec = self.parse_additive()?.node;
                    self.expect(TokenType::RParen, "Expect ')'")?;
                    let body = self.parse_block()?;
                    elif.push(node(vec![("condition", ec), ("then", Value::Array(body))]));
                }
                let mut else_branch = Value::Null;
                if self.peek()?.kind == TokenType::Else {
                    self.advance();
                    else_branch = Value::Array(self.parse_block()?);
                }
                return Ok(node(vec![
                    ("type", "IfStmt".into()),
                    ("condition", condition),
                    ("then", Value::Array(then)),
                    ("elif", Value::Array(elif)),
                    ("else", else_branch),
                ]));
            }
            TokenType::Break => {
                self.advance();
                return Ok(node(vec![("type", "Break".into())]));
            }
            TokenType::Continue => {
                self.advance();
                return Ok(node(vec![("type", "Continue".into())]));
            }
            TokenType::Return => {
                self.advance();
                if is_expr_start(self.peek()?.kind) {
                    let value = self.parse_additive()?.node;
                    return Ok(node(vec![("type", "Return".into()), ("value", value)]));
                }
                return Ok(node(vec![("type", "Return".into())]));
            }
            _ => {}
        }

        let chain = self.parse_chain(ChainContext::Statement)?;
        // Extract write/write3d only if the chain TERMINATES with a Write/Write3D node
        // Chains must end with write() - mid-chain writes don't count as terminal
        let mut write = Value::Null;
        let mut write3d = Value::Null;
        if let Some(last_node) = chain.last() {
            match last_node.get("type").as_str() {
                Some("Write") => write = last_node.get("surface").clone(),
                Some("Write3D") => {
                    write3d = node(vec![
                        ("tex3d", last_node.get("tex3d").clone()),
                        ("geo", last_node.get("geo").clone()),
                    ]);
                }
                _ => {}
            }
        }
        // If chain doesn't end with write(), write remains null.
        // The validator will produce S006 for starter chains missing terminal write().

        Ok(node(vec![
            ("chain", Value::Array(chain)),
            ("write", write),
            ("write3d", write3d),
        ]))
    }

    /// `parseChain(context)`.
    fn parse_chain(&mut self, context: ChainContext) -> PResult<Vec<Value>> {
        let first_call = self.parse_call()?;
        let mut calls = vec![first_call];
        // Comments can appear before the DOT in a chain
        // e.g., noise() \n // comment \n .bloom()
        loop {
            // Save position before collecting comments
            let saved_pos = self.current;
            // Collect any comments that might precede the DOT
            let leading_comments = self.collect_comments();
            if self.peek()?.kind != TokenType::Dot {
                // No more chaining - restore position so comments belong to next statement
                self.current = saved_pos;
                break;
            }
            self.advance(); // consume '.'
            // Now collect any additional comments after the DOT
            let post_dot_comments = self.collect_comments();
            let all_comments: Vec<String> = leading_comments
                .into_iter()
                .chain(post_dot_comments)
                .collect();

            let next_type = self.peek()?.kind;
            if next_type == TokenType::Write || next_type == TokenType::Write3d {
                if context == ChainContext::Expression {
                    let t = self.peek()?;
                    return Err(self.error_at(
                        "P005",
                        "'.write()' is only allowed in statement context",
                        t,
                    ));
                }
                // Parse write/write3d as a node in the chain (chainable)
                let mut write_node = self.parse_write_call()?;
                if !all_comments.is_empty() {
                    write_node.set("leadingComments", comments_value(all_comments));
                }
                calls.push(write_node);
                // Continue parsing - write is now chainable
                continue;
            }
            if next_type == TokenType::Subchain {
                // Parse subchain as a node in the chain (chainable)
                let mut subchain_node = self.parse_subchain_call()?;
                if !all_comments.is_empty() {
                    subchain_node.set("leadingComments", comments_value(all_comments));
                }
                calls.push(subchain_node);
                // Continue parsing - subchain is chainable
                continue;
            }
            let mut call = self.parse_call()?;
            if !all_comments.is_empty() {
                call.set("leadingComments", comments_value(all_comments));
            }
            calls.push(call);
        }
        Ok(calls)
    }

    /// `parseWriteCall()`: `write(surface)` or `write3d(tex3d, geo)`.
    fn parse_write_call(&mut self) -> PResult<Value> {
        let token = self.peek()?;
        let token_type = token.kind;
        let token_line = token.line;
        let token_col = token.col;

        if token_type == TokenType::Write {
            self.advance(); // consume 'write'
            self.expect(TokenType::LParen, "Expect '('")?;
            let next = self.peek()?;
            let surface_type = match next.kind {
                TokenType::OutputRef => Some("OutputRef"),
                TokenType::XyzRef => Some("XyzRef"),
                TokenType::VelRef => Some("VelRef"),
                TokenType::RgbaRef => Some("RgbaRef"),
                TokenType::MeshRef => Some("MeshRef"),
                // "none" is a valid target meaning "don't write to any surface"
                TokenType::Ident if next.lexeme == "none" => Some("OutputRef"),
                _ => None,
            };
            let Some(surface_type) = surface_type else {
                return Err(self.error_at(
                    "P005",
                    "write() requires an explicit surface reference (e.g., o0, o1, xyz0, vel0, rgba0, mesh0, none)",
                    next,
                ));
            };
            let surface = named_node(surface_type, &self.advance().lexeme);
            self.expect(TokenType::RParen, "Expect ')'")?;
            return Ok(node(vec![
                ("type", "Write".into()),
                ("surface", surface),
                ("loc", loc_value(token_line, token_col)),
            ]));
        } else if token_type == TokenType::Write3d {
            self.advance(); // consume 'write3d'
            self.expect(TokenType::LParen, "Expect '('")?;
            // Parse tex3d reference
            let next = self.peek()?;
            let tex3d = match next.kind {
                TokenType::OutputRef => named_node("OutputRef", &self.advance().lexeme),
                TokenType::VolRef => named_node("VolRef", &self.advance().lexeme),
                TokenType::Ident => named_node("Ident", &self.advance().lexeme),
                _ => {
                    return Err(self.error_at(
                        "P005",
                        "Expected tex3d reference in write3d()",
                        next,
                    ));
                }
            };
            self.expect(
                TokenType::Comma,
                "Expect ',' between tex3d and geo in write3d()",
            )?;
            // Parse geo reference
            let next = self.peek()?;
            let geo = match next.kind {
                TokenType::OutputRef => named_node("OutputRef", &self.advance().lexeme),
                TokenType::GeoRef => named_node("GeoRef", &self.advance().lexeme),
                TokenType::Ident => named_node("Ident", &self.advance().lexeme),
                _ => {
                    return Err(self.error_at("P005", "Expected geo reference in write3d()", next));
                }
            };
            self.expect(TokenType::RParen, "Expect ')'")?;
            return Ok(node(vec![
                ("type", "Write3D".into()),
                ("tex3d", tex3d),
                ("geo", geo),
                ("loc", loc_value(token_line, token_col)),
            ]));
        }
        Err(self.parser_error(
            "P005",
            format!(
                "Expected write or write3d at line {} col {}",
                coordinate_string(token_line),
                coordinate_string(token_col)
            ),
            Some(ErrorSite {
                position: None,
                line: token_line,
                col: token_col,
            }),
            None,
        ))
    }

    /// `reportArgIssue(code, message, token)` of `parseSubchainCall`: throw in
    /// strict mode, otherwise collect a machine-readable report.
    fn report_arg_issue(
        &self,
        arg_diagnostics: &mut Vec<Value>,
        code: &'static str,
        message: String,
        token: &Token,
    ) -> PResult<()> {
        if self.strict_subchain_arguments {
            return Err(self.parser_error(
                code,
                message,
                Some(ErrorSite::of(token)),
                Some("error"),
            ));
        }
        let position = token.position.filter(Position::is_valid);
        let has_location = is_positive_integer(token.line) && is_positive_integer(token.col);
        let severity = diagnostics::lookup(code).map_or("", |d| d.severity);
        let mut report = object(vec![
            ("code", code.into()),
            ("message", message.into()),
            ("severity", severity.into()),
        ]);
        if let Some(p) = position {
            report.insert(
                "location",
                Location {
                    line: p.line as f64,
                    column: p.column as f64,
                }
                .to_value(),
            );
            report.insert(
                "span",
                Span {
                    start: p.start as f64,
                    end: p.end as f64,
                }
                .to_value(),
            );
        } else if has_location {
            report.insert(
                "location",
                Location {
                    line: token.line.unwrap_or_default(),
                    column: token.col.unwrap_or_default(),
                }
                .to_value(),
            );
        }
        arg_diagnostics.push(Value::Object(report));
        Ok(())
    }

    /// `parseSubchainCall()`: `subchain(name: "...", id: "...") { .effect1() .effect2() }`.
    ///
    /// Subchains are atomic encapsulations of contiguous effects within a chain.
    /// The inner chain elements start with dots; the subchain as a whole is
    /// chainable.
    fn parse_subchain_call(&mut self) -> PResult<Value> {
        let name_token = self.peek()?;
        let token_line = name_token.line;
        let token_col = name_token.col;

        self.advance(); // consume 'subchain'
        self.expect(TokenType::LParen, "Expect '(' after subchain")?;

        // Machine-readable subchain-argument reports. Order follows the
        // offending token in the source. Default mode collects them onto the
        // Subchain node as non-enumerable metadata (surfaced by validate());
        // strict mode throws with the same codes.
        let mut arg_diagnostics: Vec<Value> = Vec::new();

        // Parse subchain arguments (name and optional id)
        let mut kwargs = Object::new();
        let is_keyword_start = |p: &Parser<'a>| -> PResult<bool> {
            Ok(p.peek()?.kind == TokenType::Ident
                && p.type_at(p.current + 1) == Some(TokenType::Colon))
        };
        if self.peek()?.kind != TokenType::RParen {
            // Check for keyword or positional first argument
            if self.peek()?.kind == TokenType::String {
                // Positional name: subchain("name")
                let value = self.advance().lexeme.clone();
                kwargs.insert("name", string_node(&value));
            } else if is_keyword_start(self)? {
                // Keyword arguments: subchain(name: "...", id: "...")
                while is_keyword_start(self)? {
                    let key_token = self.advance();
                    let key = key_token.lexeme.clone();
                    self.advance(); // consume ':'
                    if self.peek()?.kind != TokenType::String {
                        let t = self.peek()?;
                        return Err(self.error_at(
                            "P006",
                            &format!("Expected string value for subchain {key}"),
                            t,
                        ));
                    }
                    let value = self.advance().lexeme.clone();
                    let at = format!(
                        "at line {} col {}",
                        coordinate_string(key_token.line),
                        coordinate_string(key_token.col)
                    );
                    if !SUBCHAIN_KEYS.contains(&key.as_str()) {
                        self.report_arg_issue(
                            &mut arg_diagnostics,
                            "P008",
                            format!(
                                "Unknown subchain argument '{key}' {at}. Valid keys: name, id. The value is discarded."
                            ),
                            key_token,
                        )?;
                    } else if kwargs.contains_key(&key) {
                        self.report_arg_issue(
                            &mut arg_diagnostics,
                            "P009",
                            format!(
                                "Duplicate subchain argument '{key}' {at}. The last value wins."
                            ),
                            key_token,
                        )?;
                    }
                    kwargs.insert(key, string_node(&value));
                    if self.peek()?.kind == TokenType::Comma {
                        self.advance(); // consume ','
                    } else if is_keyword_start(self)? {
                        let t = self.peek()?;
                        self.report_arg_issue(
                            &mut arg_diagnostics,
                            "P010",
                            format!(
                                "Missing ',' between subchain arguments at line {} col {}",
                                coordinate_string(t.line),
                                coordinate_string(t.col)
                            ),
                            t,
                        )?;
                    }
                }
            }
        }
        self.expect(TokenType::RParen, "Expect ')' after subchain arguments")?;

        // Parse the subchain body block
        self.expect(TokenType::LBrace, "Expect '{' to start subchain body")?;

        // Parse chain elements inside the block
        // Each element starts with a dot: { .effect1() .effect2() }
        let mut body = Vec::new();
        while self.peek()?.kind != TokenType::RBrace {
            // Collect any leading comments
            let leading_comments = self.collect_comments();
            if self.peek()?.kind == TokenType::RBrace {
                break;
            }

            // Each chain element must start with a dot
            if self.peek()?.kind != TokenType::Dot {
                let t = self.peek()?;
                return Err(self.error_at(
                    "P006",
                    "Expected '.' before chain element in subchain body",
                    t,
                ));
            }
            self.advance(); // consume '.'

            // Collect post-dot comments
            let post_dot_comments = self.collect_comments();
            let all_comments: Vec<String> = leading_comments
                .into_iter()
                .chain(post_dot_comments)
                .collect();

            // Parse the call
            let mut call = self.parse_call()?;
            if !all_comments.is_empty() {
                call.set("leadingComments", comments_value(all_comments));
            }
            body.push(call);
        }

        self.expect(TokenType::RBrace, "Expect '}' to end subchain body")?;

        if body.is_empty() {
            return Err(self.parser_error(
                "P006",
                format!(
                    "Subchain body cannot be empty at line {} col {}",
                    coordinate_string(token_line),
                    coordinate_string(token_col)
                ),
                Some(ErrorSite::of(name_token)),
                None,
            ));
        }

        // `kwargs.name?.value || null`
        let projected = |key: &str| -> Value {
            let value = kwargs.get_or_undefined(key).get("value");
            if value.is_truthy() {
                value.clone()
            } else {
                Value::Null
            }
        };
        let mut node = object(vec![
            ("type", "Subchain".into()),
            ("name", projected("name")),
            ("id", projected("id")),
            ("body", Value::Array(body)),
            ("loc", loc_value(token_line, token_col)),
        ]);
        if !arg_diagnostics.is_empty() {
            // Non-enumerable: keeps the public AST/serialized shape unchanged
            // while validate() can surface the machine-readable reports.
            define_non_enumerable(
                &mut node,
                SUBCHAIN_ARGUMENT_DIAGNOSTICS,
                Value::Array(arg_diagnostics),
            );
        }
        Ok(Value::Object(node))
    }

    /// `parseCall()`.
    fn parse_call(&mut self) -> PResult<Value> {
        let name_token = self.expect(TokenType::Ident, "Expected identifier")?;
        // Inline namespace syntax (e.g., nd.noise()) is forbidden
        // If we see a DOT followed by an IDENT followed by LPAREN, that's an error
        if self.peek()?.kind == TokenType::Dot
            && let Some(next) = self.tokens.get(self.current + 1)
            && next.kind == TokenType::Ident
            && self.type_at(self.current + 2) == Some(TokenType::LParen)
        {
            return Err(self.parser_error(
                "P007",
                format!(
                    "Inline namespace syntax '{}.{}()' is not allowed. Use 'search {}' at the start of the program instead, at line {} col {}",
                    name_token.lexeme,
                    next.lexeme,
                    name_token.lexeme,
                    coordinate_string(name_token.line),
                    coordinate_string(name_token.col)
                ),
                Some(ErrorSite::of(name_token)),
                None,
            ));
        }
        self.expect(TokenType::LParen, "Expect '('")?;
        let mut args: Vec<Value> = Vec::new();
        let mut kwargs = Object::new();
        let mut keyword = false;
        let mut positional = false;
        let allow_mixed = name_token.lexeme == "midi" || name_token.lexeme == "audio";
        if self.peek()?.kind != TokenType::RParen {
            loop {
                if self.peek()?.kind == TokenType::Ident
                    && self.type_at(self.current + 1) == Some(TokenType::Colon)
                {
                    if positional && !allow_mixed {
                        let t = self.peek()?;
                        return Err(self.error_at(
                            "P007",
                            "Cannot mix positional and keyword arguments",
                            t,
                        ));
                    }
                    keyword = true;
                    self.parse_kwarg(&mut kwargs)?;
                } else {
                    if keyword && !allow_mixed {
                        let t = self.peek()?;
                        return Err(self.error_at(
                            "P007",
                            "Cannot mix positional and keyword arguments",
                            t,
                        ));
                    }
                    positional = true;
                    args.push(self.parse_arg()?.node);
                }
                if self.peek()?.kind != TokenType::Comma {
                    break;
                }
                self.advance();
                if self.peek()?.kind == TokenType::RParen {
                    break;
                }
            }
        }
        self.expect(TokenType::RParen, "Expect ')'")?;
        let mut call = object(vec![
            ("type", "Call".into()),
            ("name", name_token.lexeme.as_str().into()),
            ("args", Value::Array(args.clone())),
        ]);
        if keyword {
            call.insert("kwargs", Value::Object(kwargs.clone()));
        }
        let name = name_token.lexeme.as_str();
        if name == "from" {
            return self.transform_from_invocation(&call, name_token);
        }
        // osc() as a value oscillator (not the synth.osc generator effect)
        // Oscillator kwargs: type, min, max, speed, offset, seed
        if name == "osc" {
            let osc_kwargs = ["type", "min", "max", "speed", "offset", "seed"];
            let has_type_kwarg = kwargs.contains_key("type");
            let first_arg_is_osc_kind = args.first().is_some_and(|first| {
                first.is_truthy()
                    && first.get("type").as_str() == Some("Member")
                    && first.get("path").is_truthy()
                    && first.get("path").at(0).as_str() == Some("oscKind")
            });
            let is_bare_osc = args.is_empty() && kwargs.is_empty();
            // Check if all kwargs are valid oscillator parameters
            let has_only_osc_kwargs =
                !kwargs.is_empty() && kwargs.keys().all(|k| osc_kwargs.contains(&k.as_str()));
            if has_type_kwarg || first_arg_is_osc_kind || is_bare_osc || has_only_osc_kwargs {
                return self.transform_osc_invocation(&call, name_token);
            }
            // Fall through to return as regular Call node for synth effect
        }
        if name == "midi" {
            return self.transform_midi_invocation(&call, name_token);
        }
        if name == "audio" {
            return self.transform_audio_invocation(&call, name_token);
        }
        // `kwargs._skip?.type === 'Boolean' && kwargs._skip.value === true`
        let skip = {
            let skip = kwargs.get_or_undefined("_skip");
            skip.get("type").as_str() == Some("Boolean") && skip.get("value") == &Value::Bool(true)
        };
        let first_arg = args.first().cloned().unwrap_or_default();
        // read() is a pipeline built-in for reading 2D surfaces (semantic inverse of write)
        if name == "read" {
            // Extract surface reference from args or kwargs
            let surface = or(
                &or(&first_arg, kwargs.get_or_undefined("tex")),
                kwargs.get_or_undefined("surface"),
            );
            let mut node = object(vec![
                ("type", "Read".into()),
                ("surface", surface),
                ("loc", loc_value(name_token.line, name_token.col)),
            ]);
            // Preserve _skip flag if present (kwargs._skip is a Boolean AST node)
            if skip {
                node.insert("_skip", true.into());
            }
            return Ok(Value::Object(node));
        }
        // read3d() reads from tex3d (and optionally geo) surfaces
        // 1 arg: read3d(vol0) - returns volume reference for use in params
        // 2 args: read3d(vol0, geo0) - starter node that samples 3D texture
        if name == "read3d" {
            let tex3d = or(&first_arg, kwargs.get_or_undefined("tex3d"));
            let second_arg = args.get(1).cloned().unwrap_or_default();
            let geo = or(&second_arg, kwargs.get_or_undefined("geo"));
            let mut node = object(vec![
                ("type", "Read3D".into()),
                ("tex3d", tex3d),
                // null for single-arg form
                ("geo", or(&geo, &Value::Null)),
                ("loc", loc_value(name_token.line, name_token.col)),
            ]);
            // Preserve _skip flag if present (kwargs._skip is a Boolean AST node)
            if skip {
                node.insert("_skip", true.into());
            }
            return Ok(Value::Object(node));
        }
        Ok(Value::Object(call))
    }

    // --- expressions ----------------------------------------------------------------

    /// `parseArg()`.
    fn parse_arg(&mut self) -> PResult<Expr> {
        self.parse_additive()
    }

    /// `parseAdditive()`: `+`/`-` fold constant numbers.
    fn parse_additive(&mut self) -> PResult<Expr> {
        let mut node = self.parse_multiplicative()?;
        while matches!(self.peek()?.kind, TokenType::Plus | TokenType::Minus) {
            let op = self.advance().kind;
            let right = self.parse_multiplicative()?;
            let l = self.to_number(&node)?;
            let r = self.to_number(&right)?;
            node = Expr::plain(number_node(if op == TokenType::Plus {
                l + r
            } else {
                l - r
            }));
        }
        Ok(node)
    }

    /// `parseMultiplicative()`: `*`/`/` fold constant numbers.
    fn parse_multiplicative(&mut self) -> PResult<Expr> {
        let mut node = self.parse_unary()?;
        while matches!(self.peek()?.kind, TokenType::Star | TokenType::Slash) {
            let op = self.advance().kind;
            let right = self.parse_unary()?;
            let l = self.to_number(&node)?;
            let r = self.to_number(&right)?;
            node = Expr::plain(number_node(if op == TokenType::Star {
                l * r
            } else {
                l / r
            }));
        }
        Ok(node)
    }

    /// `parseUnary()`.
    fn parse_unary(&mut self) -> PResult<Expr> {
        if self.peek()?.kind == TokenType::Plus {
            self.advance();
            return self.parse_unary();
        }
        if self.peek()?.kind == TokenType::Minus {
            self.advance();
            let val = self.parse_unary()?;
            return Ok(Expr::plain(number_node(-self.to_number(&val)?)));
        }
        self.parse_primary()
    }

    /// `parsePrimary()`.
    fn parse_primary(&mut self) -> PResult<Expr> {
        let token = self.peek()?;
        match token.kind {
            TokenType::Number => {
                self.advance();
                Ok(Expr::plain(number_node(js::parse_float(&token.lexeme))))
            }
            TokenType::String => {
                self.advance();
                Ok(Expr::plain(string_node(&token.lexeme)))
            }
            TokenType::Hex => {
                self.advance();
                Ok(Expr::plain(color_node(&token.lexeme)))
            }
            TokenType::LBracket => {
                // Array literal — comma-separated arg expressions, used as
                // an alternate input form for vec2/vec3/vec4 parameters.
                let start_line = token.line;
                let start_col = token.col;
                let bracket_position = token.position;
                self.advance();
                let mut elements = Vec::new();
                if self.peek()?.kind != TokenType::RBracket {
                    elements.push(self.parse_arg()?.node);
                    while self.peek()?.kind == TokenType::Comma {
                        self.advance();
                        elements.push(self.parse_arg()?.node);
                    }
                }
                if self.peek()?.kind != TokenType::RBracket {
                    let t = self.peek()?;
                    return Err(self.error_at("P001", "Expected ']'", t));
                }
                self.advance();
                let array_node = node(vec![
                    ("type", "ArrayLiteral".into()),
                    ("elements", Value::Array(elements)),
                    ("loc", loc_value(start_line, start_col)),
                ]);
                // Private source provenance for structured diagnostics; the public
                // AST shape (loc line/col and enumeration) is unchanged.
                Ok(Expr {
                    node: array_node,
                    position: bracket_position,
                })
            }
            TokenType::Func => {
                self.advance();
                Ok(Expr::plain(node(vec![
                    ("type", "Func".into()),
                    ("src", token.lexeme.as_str().into()),
                ])))
            }
            TokenType::True => {
                self.advance();
                Ok(Expr::plain(node(vec![
                    ("type", "Boolean".into()),
                    ("value", true.into()),
                ])))
            }
            TokenType::False => {
                self.advance();
                Ok(Expr::plain(node(vec![
                    ("type", "Boolean".into()),
                    ("value", false.into()),
                ])))
            }
            TokenType::Ident => {
                if token.lexeme == "Math"
                    && self.type_at(self.current + 1) == Some(TokenType::Dot)
                    && self.type_at(self.current + 2) == Some(TokenType::Ident)
                    && self.tokens[self.current + 2].lexeme == "PI"
                {
                    self.advance();
                    self.advance();
                    self.advance();
                    return Ok(Expr::plain(number_node(std::f64::consts::PI)));
                }
                if self.type_at(self.current + 1) == Some(TokenType::LParen)
                    || self.has_call_after_dot(self.current)
                {
                    let mut chain = self.parse_chain(ChainContext::Expression)?;
                    return Ok(Expr::plain(if chain.len() == 1 {
                        chain.pop().unwrap_or_default()
                    } else {
                        node(vec![
                            ("type", "Chain".into()),
                            ("chain", Value::Array(chain)),
                        ])
                    }));
                }
                // handle dotted enum paths like foo.bar.baz. Enum segments may
                // include tokens that would otherwise be treated as keywords or
                // source/output references (e.g. `sparky.loop.tri`,
                // `disp.source.o1`). Allow a broader set of token types in
                // member chains and only terminate when the segment is followed
                // by a call expression.
                self.advance();
                let mut path = vec![Value::from(token.lexeme.as_str())];
                while self.peek()?.kind == TokenType::Dot {
                    let Some(next) = self.tokens.get(self.current + 1) else {
                        break;
                    };
                    if self.type_at(self.current + 2) == Some(TokenType::LParen) {
                        break;
                    }
                    if !is_member_token(next.kind) {
                        return Err(self.error_at("P001", "Expected identifier after '.'", next));
                    }
                    self.advance(); // consume '.'
                    self.advance(); // consume segment token stored in next
                    path.push(Value::from(next.lexeme.as_str()));
                }
                if path.len() > 1 {
                    return Ok(Expr::plain(node(vec![
                        ("type", "Member".into()),
                        ("path", Value::Array(path)),
                    ])));
                }
                Ok(Expr::plain(node(vec![
                    ("type", "Ident".into()),
                    ("name", path.swap_remove(0)),
                ])))
            }
            TokenType::OutputRef => {
                self.advance();
                Ok(Expr::plain(named_node("OutputRef", &token.lexeme)))
            }
            TokenType::SourceRef => {
                self.advance();
                Ok(Expr::plain(named_node("SourceRef", &token.lexeme)))
            }
            TokenType::VolRef => {
                self.advance();
                Ok(Expr::plain(named_node("VolRef", &token.lexeme)))
            }
            TokenType::GeoRef => {
                self.advance();
                Ok(Expr::plain(named_node("GeoRef", &token.lexeme)))
            }
            TokenType::XyzRef => {
                self.advance();
                Ok(Expr::plain(named_node("XyzRef", &token.lexeme)))
            }
            TokenType::VelRef => {
                self.advance();
                Ok(Expr::plain(named_node("VelRef", &token.lexeme)))
            }
            TokenType::RgbaRef => {
                self.advance();
                Ok(Expr::plain(named_node("RgbaRef", &token.lexeme)))
            }
            TokenType::MeshRef => {
                self.advance();
                Ok(Expr::plain(named_node("MeshRef", &token.lexeme)))
            }
            TokenType::LParen => {
                self.advance();
                let expr = self.parse_additive()?;
                self.expect(TokenType::RParen, "Expect ')'")?;
                Ok(expr)
            }
            _ => Err(self.error_at(
                "P001",
                &format!("Unexpected token {}", token.kind.to_js_string()),
                token,
            )),
        }
    }

    /// `toNumber(node)`: the value of a Number node; any other node is a P001.
    fn to_number(&self, expr: &Expr) -> PResult<f64> {
        let node = &expr.node;
        if node.get("type").as_str() != Some("Number") {
            // Number coercion failures locate the offending AST node's private
            // source position when present, its parser-authored loc otherwise,
            // and are explicitly null when neither carries valid coordinates.
            let loc = node.get("loc");
            return Err(self.parser_error(
                "P001",
                "Expected number".to_owned(),
                Some(ErrorSite {
                    position: expr.position,
                    line: loc.get("line").as_f64(),
                    col: loc.get("col").as_f64(),
                }),
                None,
            ));
        }
        Ok(node.get("value").as_f64().unwrap_or(f64::NAN))
    }

    /// `parseKwarg(obj)`.
    fn parse_kwarg(&mut self, obj: &mut Object) -> PResult<()> {
        let key = self
            .expect(TokenType::Ident, "Expected identifier")?
            .lexeme
            .clone();
        self.expect(TokenType::Colon, "Expect ':'")?;
        if !is_expr_start(self.peek()?.kind) {
            let t = self.peek()?;
            return Err(self.error_at("P001", "Expected expression after '='", t));
        }
        let value = self.parse_arg()?.node;
        obj.insert(key, value);
        Ok(())
    }
}

/// The `HEX` case of `parsePrimary`: `#rgb`, `#rrggbb` or `#rrggbbaa` as
/// `{type: 'Color', value: [r, g, b, a]}` in 0..1.
fn color_node(lexeme: &str) -> Value {
    let units: Vec<u16> = lexeme.encode_utf16().collect();
    let hex: &[u16] = units.get(1..).unwrap_or(&[]);
    // `parseInt(<two code units>, 16)`
    let byte = |a: u16, b: u16| js::parse_int(&String::from_utf16_lossy(&[a, b]), 16);
    // `let r, g, b` start undefined: undefined / 255 is NaN.
    let (mut r, mut g, mut b, mut a) = (f64::NAN, f64::NAN, f64::NAN, 1.0);
    if hex.len() == 3 {
        r = byte(hex[0], hex[0]);
        g = byte(hex[1], hex[1]);
        b = byte(hex[2], hex[2]);
    } else if hex.len() == 6 {
        r = byte(hex[0], hex[1]);
        g = byte(hex[2], hex[3]);
        b = byte(hex[4], hex[5]);
    } else if hex.len() == 8 {
        r = byte(hex[0], hex[1]);
        g = byte(hex[2], hex[3]);
        b = byte(hex[4], hex[5]);
        a = byte(hex[6], hex[7]) / 255.0;
    }
    node(vec![
        ("type", "Color".into()),
        (
            "value",
            Value::Array(vec![
                Value::Number(r / 255.0),
                Value::Number(g / 255.0),
                Value::Number(b / 255.0),
                Value::Number(a),
            ]),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{lex, lex_with_diagnostics};

    fn registry() -> Registry {
        Registry::new()
    }

    fn parse_src(src: &str) -> Result<Value, DiagnosticError> {
        let tokens = lex_with_diagnostics(src)?;
        parse_with_options(&tokens, &registry(), &ParseOptions::default())
    }

    fn parse_strict(src: &str) -> Result<Value, DiagnosticError> {
        let tokens = lex_with_diagnostics(src)?;
        parse_with_options(
            &tokens,
            &registry(),
            &ParseOptions::strict_subchain_arguments(),
        )
    }

    /// `sourcePosition(source, line, column)` of test_diagnostic_locations.js:
    /// the span of the one token at those source coordinates.
    fn source_position(src: &str, line: usize, column: usize) -> Span {
        let matches: Vec<Position> = lex(src)
            .unwrap()
            .into_iter()
            .filter_map(|t| t.position)
            .filter(|p| p.line == line && p.column == column)
            .collect();
        assert_eq!(matches.len(), 1, "{src:?} {line}:{column}");
        Span {
            start: matches[0].start as f64,
            end: matches[0].end as f64,
        }
    }

    fn assert_parser_failure(src: &str, code: &str, message: &str, line: usize, column: usize) {
        let err = parse_src(src).expect_err("parse should fail");
        assert_eq!(err.error, JsError::syntax(message), "{src:?}");
        let diagnostic = *err.diagnostic.expect("SyntaxError carries a diagnostic");
        assert_eq!(diagnostic.code, code, "{src:?}");
        assert_eq!(diagnostic.stage, "parser");
        assert_eq!(diagnostic.message, message);
        assert_eq!(
            diagnostic.location,
            Some(Location {
                line: line as f64,
                column: column as f64
            }),
            "{src:?}"
        );
        assert_eq!(
            diagnostic.span,
            Some(source_position(src, line, column)),
            "{src:?}"
        );
    }

    #[test]
    fn expectation_diagnostics() {
        for (src, code, message, line, column) in [
            (
                "search synth\nrender o0",
                "P001",
                "Expect '(' at line 2 col 8",
                2,
                8,
            ),
            (
                "search synth\nrender(o0",
                "P002",
                "Expect ')' at line 2 col 10",
                2,
                10,
            ),
            (
                "search synth\nlet = 1",
                "P001",
                "Expected identifier at line 2 col 5",
                2,
                5,
            ),
            (
                "search synth\nlet x 1",
                "P001",
                "Expect '=' at line 2 col 7",
                2,
                7,
            ),
            (
                "search synth\nif(true) return 1",
                "P001",
                "Expect '{' at line 2 col 10",
                2,
                10,
            ),
            (
                "search synth\nrender(o0) xyz",
                "P001",
                "Expected end of input at line 2 col 12",
                2,
                12,
            ),
            (
                "search synth\nfoo(1",
                "P002",
                "Expect ')' at line 2 col 6",
                2,
                6,
            ),
            (
                "search synth\nfoo().write3d(tex3d0 geo0)",
                "P001",
                "Expect ',' between tex3d and geo in write3d() at line 2 col 22",
                2,
                22,
            ),
            (
                "// 😀\r\nsearch synth\r\n\trender(o0",
                "P002",
                "Expect ')' at line 3 col 11",
                3,
                11,
            ),
            (
                "search synth\nlet x = \"😀\"; render o0",
                "P001",
                "Expect '(' at line 2 col 22",
                2,
                22,
            ),
        ] {
            assert_parser_failure(src, code, message, line, column);
        }
    }

    #[test]
    fn search_diagnostics() {
        let missing = "Missing required 'search' directive. Every program must start with 'search <namespace>, ...' to specify namespace search order.";
        for (src, message, line, column) in [
            ("", missing, 1, 1),
            ("let x = 1", missing, 1, 10),
            (
                "search synth search filter",
                "Only one search directive is allowed per program at line 1 col 14",
                1,
                14,
            ),
            (
                "search bogus",
                "Invalid namespace 'bogus' at line 1 col 8. Valid namespaces: io, classicNoisedeck, synth, mixer, filter, render, points, synth3d, filter3d, user",
                1,
                8,
            ),
            (
                "search",
                "Expected namespace identifier after search at line 1 col 7",
                1,
                7,
            ),
            (
                "search synth,",
                "Expected namespace identifier after comma at line 1 col 14",
                1,
                14,
            ),
            (
                "let x = 1; search synth",
                "'search' directive must appear before other statements at line 1 col 12",
                1,
                12,
            ),
            (
                "search synth\nif(true) { search filter }",
                "'search' directive is only allowed at the start of the program at line 2 col 12",
                2,
                12,
            ),
            (
                "// 😀\r\n\tsearch 1",
                "Expected namespace identifier after search at line 2 col 9",
                2,
                9,
            ),
            (
                "search synth\nlet x = \"😀\"; search filter",
                "'search' directive must appear before other statements at line 2 col 15",
                2,
                15,
            ),
        ] {
            assert_parser_failure(src, "P004", message, line, column);
        }
        let ast =
            parse_src("/* leading */ search render, synth, synth; diagProbe().write(o0)").unwrap();
        assert_eq!(
            ast.get("namespace").get("searchOrder").to_json().unwrap(),
            r#"["render","synth","synth"]"#
        );
    }

    #[test]
    fn automation_diagnostics() {
        for (invocation, prefix, suffix) in [
            (
                "osc(type: oscKind.sine, bogus: 1)",
                "osc() unknown parameter 'bogus'",
                ". Valid: type, min, max, speed, offset, seed",
            ),
            (
                "midi(1, 2, 3, 4, 5, 6)",
                "midi() name, id, cc, nrpn, zone and members are keyword-only",
                "",
            ),
            (
                "midi(bogus: 1)",
                "midi() unknown parameter 'bogus'",
                ". Valid: channel, mode, min, max, sensitivity, name, id, cc, nrpn, zone, members",
            ),
            (
                "midi(1, 2, 3, 4, 5, channel: 1)",
                "midi() has an excess positional argument",
                "",
            ),
            ("midi()", "midi() requires 'channel' or 'zone' argument", ""),
            (
                "midi(1, zone: 1)",
                "midi() 'channel' and 'zone' are mutually exclusive",
                "",
            ),
            (
                "midi(1, members: 2)",
                "midi() 'members' requires 'zone'",
                "",
            ),
            (
                "midi(1, id: \"port\")",
                "midi() 'id' requires readable 'name'",
                "",
            ),
            (
                "midi(1, name: 1)",
                "midi() 'name' requires a quoted string",
                "",
            ),
            ("midi(1, name: \"\")", "midi() 'name' must not be empty", ""),
            (
                "midi(1, name: \"port\", id: 1)",
                "midi() 'id' requires a quoted string",
                "",
            ),
            (
                "midi(1, name: \"port\", id: \"\")",
                "midi() 'id' must not be empty",
                "",
            ),
            (
                "audio(1, 2, 3, 4)",
                "audio() channel, name and id are keyword-only",
                "",
            ),
            (
                "audio(bogus: 1)",
                "audio() unknown parameter 'bogus'",
                ". Valid: band, min, max, channel, name, id",
            ),
            (
                "audio(1, 2, 3, band: 1)",
                "audio() has an excess positional argument",
                "",
            ),
            ("audio()", "audio() requires 'band' argument", ""),
            (
                "audio(1, id: \"device\")",
                "audio() 'id' requires readable 'name'",
                "",
            ),
            (
                "audio(1, name: \"device\")",
                "audio() selected device requires both 'name' and 'channel'",
                "",
            ),
            (
                "audio(1, channel: 1, name: 1)",
                "audio() 'name' requires a quoted string",
                "",
            ),
            (
                "audio(1, channel: 1, name: \"\")",
                "audio() 'name' must not be empty",
                "",
            ),
            (
                "audio(1, channel: 1, name: \"device\", id: 1)",
                "audio() 'id' requires a quoted string",
                "",
            ),
            (
                "audio(1, channel: 1, name: \"device\", id: \"\")",
                "audio() 'id' must not be empty",
                "",
            ),
        ] {
            let src = format!("search synth\nlet x = {invocation}");
            let message = format!("{prefix} at line 2 col 9{suffix}");
            assert_parser_failure(&src, "P003", &message, 2, 9);
        }
    }

    #[test]
    fn valid_automation_invocations_retain_defaults_and_keys() {
        let ast =
            parse_src("search synth\nlet a = osc(); let b = midi(1); let c = audio(audioBand.low)")
                .unwrap();
        let exprs: Vec<String> = ast
            .get("vars")
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                format!(
                    "{:?}",
                    v.get("expr")
                        .as_object()
                        .unwrap()
                        .keys()
                        .collect::<Vec<_>>()
                )
            })
            .collect();
        assert_eq!(
            exprs,
            [
                r#"["type", "oscType", "min", "max", "speed", "offset", "seed", "loc"]"#,
                r#"["type", "channel", "mode", "min", "max", "sensitivity", "cc", "nrpn", "zone", "members", "name", "id", "loc"]"#,
                r#"["type", "band", "min", "max", "channel", "name", "id", "loc"]"#,
            ]
        );
        assert_eq!(
            ast.get("vars").at(1).get("expr").to_json().unwrap(),
            r#"{"type":"Midi","channel":{"type":"Number","value":1},"mode":{"type":"Member","path":["midiMode","velocity"]},"min":{"type":"Number","value":0},"max":{"type":"Number","value":1},"sensitivity":{"type":"Number","value":1},"loc":{"line":2,"col":24}}"#
        );
    }

    #[test]
    fn output_diagnostics() {
        for (src, message, line, column) in [
            (
                "search synth\nrender(1)",
                "Expected output reference in render()",
                2,
                8,
            ),
            (
                "search synth\nrender(",
                "Expected output reference in render()",
                2,
                8,
            ),
            (
                "search synth\nlet x = diagProbe().write(o0)",
                "'.write()' is only allowed in statement context at line 2 col 21",
                2,
                21,
            ),
            (
                "search synth\nlet x = diagProbe().write3d(vol0, geo0)",
                "'.write()' is only allowed in statement context at line 2 col 21",
                2,
                21,
            ),
            (
                "search synth\ndiagProbe().write()",
                "write() requires an explicit surface reference (e.g., o0, o1, xyz0, vel0, rgba0, mesh0, none) at line 2 col 19",
                2,
                19,
            ),
            (
                "search synth\ndiagProbe().write3d(1, geo0)",
                "Expected tex3d reference in write3d() at line 2 col 21",
                2,
                21,
            ),
            (
                "search synth\ndiagProbe().write3d(vol0, 1)",
                "Expected geo reference in write3d() at line 2 col 27",
                2,
                27,
            ),
            (
                "search synth\ndiagProbe().write3d(vol0,",
                "Expected geo reference in write3d() at line 2 col 26",
                2,
                26,
            ),
            (
                "// 😀\r\nsearch synth\r\n\trender(\"😀\")",
                "Expected output reference in render()",
                3,
                9,
            ),
            (
                "search synth\nlet x = \"😀\"; render(none)",
                "Expected output reference in render()",
                2,
                22,
            ),
        ] {
            assert_parser_failure(src, "P005", message, line, column);
        }
    }

    #[test]
    fn subchain_and_call_form_diagnostics() {
        for (src, message, line, column) in [
            (
                "search synth\nread(o0).subchain(name: 1) { .diagProbe() }",
                "Expected string value for subchain name at line 2 col 25",
                2,
                25,
            ),
            (
                "search synth\nread(o0).subchain(name:",
                "Expected string value for subchain name at line 2 col 24",
                2,
                24,
            ),
            (
                "search synth\nread(o0).subchain() { diagProbe() }",
                "Expected '.' before chain element in subchain body at line 2 col 23",
                2,
                23,
            ),
            (
                "search synth\nread(o0).subchain() {}",
                "Subchain body cannot be empty at line 2 col 10",
                2,
                10,
            ),
            (
                "search synth\nread(o0).subchain() { /* empty */ }",
                "Subchain body cannot be empty at line 2 col 10",
                2,
                10,
            ),
            (
                "// 😀\r\nsearch synth\r\n\tread(o0).subchain(name: \"😀\", id: 1) { .diagProbe() }",
                "Expected string value for subchain id at line 3 col 36",
                3,
                36,
            ),
            (
                "search synth\nread(o0).subchain() { .diagProbe()",
                "Expected '.' before chain element in subchain body at line 2 col 35",
                2,
                35,
            ),
        ] {
            assert_parser_failure(src, "P006", message, line, column);
        }
        for (src, message, line, column) in [
            (
                "search synth\nlet x = from(a: 1, b: 2)",
                "'from' does not support named arguments at line 2 col 9",
                2,
                9,
            ),
            (
                "search synth\nlet x = from(synth)",
                "'from' requires exactly two arguments (namespace, call) at line 2 col 9",
                2,
                9,
            ),
            (
                "search synth\nlet x = from(1, probe())",
                "'from' namespace argument must be an identifier at line 2 col 9",
                2,
                9,
            ),
            (
                "search synth\nlet x = from(synth, 1)",
                "'from' second argument must be a call expression at line 2 col 9",
                2,
                9,
            ),
            (
                "search synth\nnd.noise()",
                "Inline namespace syntax 'nd.noise()' is not allowed. Use 'search nd' at the start of the program instead, at line 2 col 1",
                2,
                1,
            ),
            (
                "search synth\ndiagProbe(1, x: 2)",
                "Cannot mix positional and keyword arguments at line 2 col 14",
                2,
                14,
            ),
            (
                "search synth\ndiagProbe(x: 1, 2)",
                "Cannot mix positional and keyword arguments at line 2 col 17",
                2,
                17,
            ),
        ] {
            assert_parser_failure(src, "P007", message, line, column);
        }
        for (src, message, line, column) in [
            (
                "search synth\nlet x = ;",
                "Expected expression after '=' at line 2 col 9",
                2,
                9,
            ),
            (
                "search synth\ndiagProbe(a: )",
                "Expected expression after '=' at line 2 col 14",
                2,
                14,
            ),
            (
                "search synth\nlet x = [1 2]",
                "Expected ']' at line 2 col 12",
                2,
                12,
            ),
            (
                "search synth\nlet x = foo.+",
                "Expected identifier after '.' at line 2 col 13",
                2,
                13,
            ),
            (
                "search synth\ndiagProbe(; 1)",
                "Unexpected token SEMICOLON at line 2 col 11",
                2,
                11,
            ),
        ] {
            assert_parser_failure(src, "P001", message, line, column);
        }
    }

    #[test]
    fn number_coercion_diagnostics() {
        for src in [
            "search synth\nlet x = 1 + o0",
            "search synth\nlet x = diagProbe() + 1",
        ] {
            let err = parse_src(src).unwrap_err();
            assert_eq!(err.error, JsError::syntax("Expected number"));
            let d = err.diagnostic.unwrap();
            assert_eq!((d.code, d.location, d.span), ("P001", None, None));
        }
        for (src, line, column) in [
            ("search synth\nlet y = [1] + 1", 2, 9),
            ("search synth\nlet y = 1 * [1]", 2, 13),
            ("search synth\nlet y = -[1]", 2, 10),
            (
                "search synth\nlet f = () => (1\n + 2); let y = [1] + 1",
                3,
                16,
            ),
            (
                "search synth\nlet f = () => (1\n + 2); let y = 1 * [1]",
                3,
                20,
            ),
            ("search synth\nlet f = () => (1\n + 2); let y = -[1]", 3, 17),
        ] {
            assert_parser_failure(src, "P001", "Expected number", line, column);
        }
        // An Oscillator has a parser-authored loc but no position.
        let err = parse_src("search synth\nlet y = -osc()").unwrap_err();
        let d = err.diagnostic.unwrap();
        assert_eq!(
            d.location,
            Some(Location {
                line: 2.0,
                column: 10.0
            })
        );
        assert_eq!(d.span, None);
    }

    #[test]
    fn drifted_legacy_coordinates_keep_source_diagnostics() {
        for (src, message, location, span) in [
            (
                "search synth\nlet x = () => (1\n + 2); render o0",
                "Expect '(' at line 2 col 32",
                Location {
                    line: 3.0,
                    column: 15.0,
                },
                Span {
                    start: 44.0,
                    end: 46.0,
                },
            ),
            (
                "search synth\nlet x = \"a\\\nb\"; render o0",
                "Expect '(' at line 2 col 24",
                Location {
                    line: 3.0,
                    column: 12.0,
                },
                Span {
                    start: 36.0,
                    end: 38.0,
                },
            ),
        ] {
            let err = parse_src(src).unwrap_err();
            assert_eq!(err.error, JsError::syntax(message));
            let d = err.diagnostic.unwrap();
            assert_eq!((d.location, d.span), (Some(location), Some(span)));
        }
    }

    /// Caller-built tokens: `lex(source).map(({type, lexeme}) => ({type, lexeme, ...coordinates}))`.
    fn caller_tokens(src: &str, line: Option<f64>, col: Option<f64>) -> Vec<Token> {
        lex(src)
            .unwrap()
            .into_iter()
            .map(|t| Token {
                kind: t.kind,
                lexeme: t.lexeme,
                line,
                col,
                position: None,
            })
            .collect()
    }

    #[test]
    fn caller_tokens_without_coordinates() {
        for (line, col, text) in [
            (None, None, "at line undefined col undefined"),
            (Some(1.0), None, "at line 1 col undefined"),
            (Some(0.0), Some(1.0), "at line 0 col 1"),
            (Some(1.0), Some(f64::NAN), "at line 1 col NaN"),
        ] {
            let tokens = caller_tokens("search synth\nrender o0", line, col);
            let err =
                parse_with_options(&tokens, &registry(), &ParseOptions::default()).unwrap_err();
            assert_eq!(err.error, JsError::syntax(format!("Expect '(' {text}")));
            let d = err.diagnostic.unwrap();
            assert_eq!((d.code, d.location, d.span), ("P001", None, None));

            let tokens = caller_tokens("search synth\nlet x = midi()", line, col);
            let err =
                parse_with_options(&tokens, &registry(), &ParseOptions::default()).unwrap_err();
            assert_eq!(
                err.error,
                JsError::syntax(format!(
                    "midi() requires 'channel' or 'zone' argument {text}"
                ))
            );
        }
        // Positive integer coordinates without a position locate the token.
        let tokens = caller_tokens("search synth\nrender o0", Some(2.0), Some(8.0));
        let err = parse_with_options(&tokens, &registry(), &ParseOptions::default()).unwrap_err();
        let d = err.diagnostic.unwrap();
        assert_eq!(
            d.location,
            Some(Location {
                line: 2.0,
                column: 8.0
            })
        );
        assert_eq!(d.span, None);
        // `from` without numeric coordinates reports the bare message.
        let tokens = caller_tokens("search synth\nlet x = from(synth)", None, None);
        let err = parse_with_options(&tokens, &registry(), &ParseOptions::default()).unwrap_err();
        assert_eq!(
            err.error,
            JsError::syntax("'from' requires exactly two arguments (namespace, call)")
        );
        // A token array without EOF reads past its end.
        let tokens: Vec<Token> = lex("search synth").unwrap().into_iter().take(2).collect();
        let err = parse_with_options(&tokens, &registry(), &ParseOptions::default()).unwrap_err();
        assert_eq!(
            err.error,
            JsError::type_error("Cannot read properties of undefined (reading 'type')")
        );
        assert_eq!(err.diagnostic, None);
    }

    fn subchain_source(args: &str) -> String {
        format!(
            "search synth\nsubchainProbe()\n  .subchain({args}) {{\n    .subchainFilter()\n  }}\n  .write(o0)"
        )
    }

    fn reports(ast: &Value) -> Vec<(String, String, String)> {
        let subchain = ast.get("plans").at(0).get("chain").at(1);
        match subchain_argument_diagnostics(subchain) {
            Value::Array(reports) => reports
                .iter()
                .map(|r| {
                    (
                        r.get("code").as_str().unwrap().to_owned(),
                        r.get("severity").as_str().unwrap().to_owned(),
                        r.get("location").to_json().unwrap_or_default(),
                    )
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    #[test]
    fn permissive_subchain_arguments_report_without_changing_the_ast() {
        let src = "search synth\nread(o0).subchain(foo: \"x\" name: \"a\" name: \"b\" id: \"s\") { .diagFilter() }.write(o1)";
        let ast = parse_src(src).unwrap();
        let subchain = ast.get("plans").at(0).get("chain").at(1);
        assert_eq!(
            subchain.to_json().unwrap(),
            r#"{"type":"Subchain","name":"b","id":"s","body":[{"type":"Call","name":"diagFilter","args":[]}],"loc":{"line":2,"col":10}}"#
        );
        let w = |l: u32, c: u32| {
            (
                "warning".to_owned(),
                format!(r#"{{"line":{l},"column":{c}}}"#),
            )
        };
        let got: Vec<_> = reports(&ast)
            .into_iter()
            .map(|(code, sev, loc)| (code, (sev, loc)))
            .collect();
        assert_eq!(
            got,
            [
                ("P008".to_owned(), w(2, 19)),
                ("P010".to_owned(), w(2, 28)),
                ("P010".to_owned(), w(2, 38)),
                ("P009".to_owned(), w(2, 38)),
                ("P010".to_owned(), w(2, 48)),
            ]
        );
        // A JSON round trip drops the reports, like a non-enumerable property.
        let roundtrip = Value::from_json(&ast.to_json().unwrap()).unwrap();
        assert!(
            subchain_argument_diagnostics(roundtrip.get("plans").at(0).get("chain").at(1))
                .is_undefined()
        );

        let codes = |src: &str| -> Vec<String> {
            reports(&parse_src(src).unwrap())
                .into_iter()
                .map(|r| r.0)
                .collect()
        };
        assert_eq!(
            codes(&subchain_source("nme: \"x\", name: \"a\" name: \"b\"")),
            ["P008", "P010", "P009"]
        );
        assert_eq!(
            codes(&subchain_source("nme: \"x\", nme: \"y\", name: \"ok\"")),
            ["P008", "P008"]
        );
        for args in [
            "name: \"a\"",
            "name: \"a\", id: \"b\"",
            "id: \"b\"",
            "\"positional\"",
            "",
        ] {
            assert!(codes(&subchain_source(args)).is_empty(), "{args}");
        }
        let messages: Vec<String> = match subchain_argument_diagnostics(
            parse_src(&subchain_source("nme: \"typo\", name: \"ok\""))
                .unwrap()
                .get("plans")
                .at(0)
                .get("chain")
                .at(1),
        ) {
            Value::Array(r) => r
                .iter()
                .map(|r| r.get("message").as_str().unwrap().to_owned())
                .collect(),
            _ => Vec::new(),
        };
        assert_eq!(
            messages,
            [
                "Unknown subchain argument 'nme' at line 3 col 13. Valid keys: name, id. The value is discarded."
            ]
        );
    }

    #[test]
    fn strict_subchain_arguments_throw() {
        let src = subchain_source("nme: \"typo\", name: \"ok\"");
        let err = parse_strict(&src).unwrap_err();
        let d = err.diagnostic.unwrap();
        assert_eq!((d.code, d.stage, d.severity), ("P008", "parser", "error"));
        assert_eq!(
            d.location,
            Some(Location {
                line: 3.0,
                column: 13.0
            })
        );
        assert_eq!(d.span, Some(source_position(&src, 3, 13)));
        assert_eq!(
            parse_strict(&subchain_source("name: \"a\", name: \"b\""))
                .unwrap_err()
                .diagnostic
                .unwrap()
                .code,
            "P009"
        );
        assert_eq!(
            parse_strict(&subchain_source("name: \"a\" id: \"b\""))
                .unwrap_err()
                .diagnostic
                .unwrap()
                .code,
            "P010"
        );
        for args in [
            "name: \"a\"",
            "name: \"a\", id: \"b\"",
            "\"positional\"",
            "",
        ] {
            assert_eq!(
                parse_strict(&subchain_source(args)),
                parse_src(&subchain_source(args))
            );
        }
    }

    #[test]
    fn subchain_reports_fall_back_to_token_coordinates() {
        let src = subchain_source("nme: \"typo\", name: \"ok\"");
        let strip = |keep_coordinates: bool| -> Vec<Token> {
            lex(&src)
                .unwrap()
                .into_iter()
                .map(|t| Token {
                    position: None,
                    line: if keep_coordinates { t.line } else { None },
                    col: if keep_coordinates { t.col } else { None },
                    ..t
                })
                .collect()
        };
        let ast = parse_with_options(&strip(true), &registry(), &ParseOptions::default()).unwrap();
        assert_eq!(
            reports(&ast),
            [(
                "P008".to_owned(),
                "warning".to_owned(),
                r#"{"line":3,"column":13}"#.to_owned()
            )]
        );
        let ast = parse_with_options(&strip(false), &registry(), &ParseOptions::default()).unwrap();
        let subchain = ast.get("plans").at(0).get("chain").at(1);
        let report = subchain_argument_diagnostics(subchain).at(0).clone();
        assert!(
            report
                .as_object()
                .is_some_and(|r| !r.contains_key("location"))
        );
        // The node itself carries legacy coordinates as given: absent here.
        assert_eq!(subchain.get("loc").to_json().unwrap(), "{}");
    }

    #[test]
    fn valid_forms_keep_their_shapes() {
        let ast = parse_src("search synth\nlet x = from(synth, probe())").unwrap();
        assert_eq!(
            ast.get("vars").at(0).get("expr").to_json().unwrap(),
            r#"{"type":"Call","name":"probe","args":[],"namespace":{"name":"synth","path":["synth"],"explicit":true,"source":"from","resolved":"synth","searchOrder":["synth"],"fromOverride":true}}"#
        );
        let ast = parse_src("search synth\nlet a = midi(1, channel: 2)").unwrap();
        assert_eq!(
            ast.get("vars")
                .at(0)
                .get("expr")
                .get("channel")
                .get("value"),
            &Value::Number(2.0)
        );
        for (name, kind) in [
            ("o0", "OutputRef"),
            ("xyz0", "XyzRef"),
            ("vel0", "VelRef"),
            ("rgba0", "RgbaRef"),
            ("mesh0", "MeshRef"),
            ("none", "OutputRef"),
        ] {
            let ast = parse_src(&format!("search synth\ndiagProbe().write({name})")).unwrap();
            let plan = ast.get("plans").at(0);
            assert_eq!(
                plan.get("chain").at(1).to_json().unwrap(),
                format!(
                    r#"{{"type":"Write","surface":{{"type":"{kind}","name":"{name}"}},"loc":{{"line":2,"col":13}}}}"#
                )
            );
            assert_eq!(
                plan.get("write").to_json().unwrap(),
                format!(r#"{{"type":"{kind}","name":"{name}"}}"#)
            );
        }
        let ast = parse_src("search synth\nlet y = [1, 2]").unwrap();
        assert_eq!(
            ast.get("vars").at(0).get("expr").to_json().unwrap(),
            r#"{"type":"ArrayLiteral","elements":[{"type":"Number","value":1},{"type":"Number","value":2}],"loc":{"line":2,"col":9}}"#
        );
    }

    #[test]
    fn colors_of_every_length() {
        let color = |hex: &str| color_node(hex).get("value").to_json().unwrap();
        assert_eq!(color("#fff"), "[1,1,1,1]");
        assert_eq!(color("#ff0000"), "[1,0,0,1]");
        assert_eq!(color("#00000080"), format!("[0,0,0,{}]", 128.0 / 255.0));
        // Caller-built HEX tokens of other lengths leave r, g, b undefined.
        assert_eq!(color("#ff"), "[null,null,null,1]");
    }
}
