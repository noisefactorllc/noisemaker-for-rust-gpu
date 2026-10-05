//! Whether V8 accepts a `Function`-constructor body.
//!
//! The reference validator compiles every DSL arrow-function value (a `Func`
//! node, `() => expr`) with `new Function('state', \`with(state){ return ${src}; }\`)`
//! and reports S001 when that throws. Nothing ever calls the function, so the only
//! observable result is whether V8 accepts the body. V8 builds the source
//! `(function anonymous(state\n) {\n` + body + `\n})` and requires it to parse as a
//! single function literal, which holds exactly when the body is a valid sloppy-mode
//! FunctionBody of a function with the one simple parameter `state`; this module
//! decides that question for the V8 the parity oracle runs (node 26.10 / V8 14.6).
//!
//! # Why a port of an acorn-based checker, not a parser crate
//!
//! `oxc_parser` 0.153 with `oxc_semantic` (MIT) was evaluated on the exact source
//! V8 builds, with V8's single-function-literal rule applied on top: it agrees
//! with V8 on 263 of the corpus' 271 function bodies and disagrees in both
//! directions. It implements the specification and its proposals, not V8: it
//! rejects call-expression assignment targets (`f() = 1`, `f()++`), which V8
//! accepts and defers to run time, rejects `let` followed by a line break and a
//! keyword, accepts `import.defer(x)`, decorators, `accessor` fields and Unicode 18
//! identifiers (V8 14.6 uses Unicode 17), treats U+0085 as white space, and has no
//! recursion guard (deep input aborts the process). Matching V8 would mean patching
//! its verdicts case by case. This module instead ports the checker the Qt port
//! built for the same question (`js_syntax.cpp`, `js_regexp.cpp`, `js_unicode.cpp`):
//! acorn 8.16's tokenizer, statement, expression, lval and scope modules, extended
//! with the places where V8 accepts or rejects what acorn does not. Where that port
//! simplified acorn in a way V8 observes (the resumption point after an invalid
//! escape in a tagged template), this port follows acorn and V8, and it adds the
//! V8 behaviors listed below, found by differential testing against the oracle's
//! node: the corpus' bodies, about 3200 targeted probes and several million
//! generated and mutated bodies (expressions, statements, destructuring patterns,
//! classes, lazily compiled functions) agree with V8.
//!
//! Acorn's code carries this notice:
//!
//! > MIT License
//! >
//! > Copyright (C) 2012-2022 by various contributors (see AUTHORS)
//! >
//! > Permission is hereby granted, free of charge, to any person obtaining a copy
//! > of this software and associated documentation files (the "Software"), to deal
//! > in the Software without restriction, including without limitation the rights
//! > to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//! > copies of the Software, and to permit persons to whom the Software is
//! > furnished to do so, subject to the following conditions:
//! >
//! > The above copyright notice and this permission notice shall be included in
//! > all copies or substantial portions of the Software.
//! >
//! > THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//! > IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//! > FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//! > AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//! > LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//! > OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//! > SOFTWARE.
//!
//! V8 behaviour that differs from acorn, and that this checker follows:
//!   - A call expression is a valid target for `=`, compound assignment, `++`/`--`
//!     and a for-in/of head (V8 defers the error to run time), but not for logical
//!     assignment or inside a destructuring pattern.
//!   - `let` starts a lexical declaration only when the next token is `[`, `{`, an
//!     identifier or a contextual keyword, or (in sloppy code) a strict-mode
//!     reserved word. Before any other keyword, `let` is an identifier.
//!   - `using` is never a declaration in a single-statement position, and never
//!     heads a for-in loop.
//!   - `const` and `using` may omit the initializer only in a for-in/of head (acorn
//!     also allows it before an `of` or `in` on the next line).
//!   - A labelled function declaration binds its name in the enclosing scope.
//!   - `arguments` is rejected in an arrow function inside a class static block.
//!   - `import.source(x)` (source phase import) is accepted; `import.defer` is not.
//!   - A for-of head whose last token is an unescaped `async` is rejected
//!     (`for (x.async of y)`), not only a bare `async`.
//!   - `using`, `await using` and `for await` are recognised even when the words
//!     are written with escapes; in a for head, `using of` is a declaration only
//!     when `=` follows.
//!   - An arrow function is never the test of a conditional (`() => {} ? a : b`) nor
//!     an operand of `**`, except that V8 parses a conditional chain iteratively
//!     and takes the expression in an alternate position that precedes `?` as the
//!     next test (`c ? a : () => {} ? b : d` is valid).
//!   - An async arrow function's rest parameter must be last, and only a call
//!     directly on `async` heads one.
//!   - Expression errors (shorthand initializers, duplicate `__proto__` keys) are
//!     decided as V8's expression scopes decide them: a call's argument errors
//!     belong to the enclosing expression, which discards them when it is the
//!     call target of a for-in/of head (`for (f({a = 1}.b) of x)` is valid) and
//!     reports them otherwise (an async arrow head's arguments are checked at
//!     once); a member expression as an array or object literal element, as an
//!     assignment target or as a for-in/of target is validated as an expression
//!     (`[{x = 1}.y] = z` is invalid); a destructuring pattern keeps the errors of
//!     its earlier elements when a later element is an assignment
//!     (`[(a = b), c = d] = e` is invalid).
//!   - V8 pre-parses nested functions lazily: every non-arrow function except
//!     one right after `(` or `!` (its "likely called" hint), and everything
//!     inside a pre-parsed function. The pre-parser accepts an optional chain
//!     ending in a private member (`a?.#x = 1`) as an assignment target.
//!   - A class's static initializers (static fields, static blocks) get the
//!     function kind of whichever initializer the class body creates first: after
//!     an instance field, a static block allows `await` as an identifier and
//!     `return`; otherwise `await` is reserved in static field initializers too.
//!     A static block's top-level functions are declared like vars.
//!   - The Katakana_Or_Hiragana script value is not a valid `\p{...}` value.
//!
//! # Nesting depth and size limits
//!
//! V8 parses and compiles the `anonymous` function recursively and throws a
//! RangeError, which the reference's `catch` treats like a SyntaxError, once its
//! stack runs out. That depth depends on the constructs nested and on V8's frame
//! sizes, so it cannot be reproduced exactly. This checker counts nesting units
//! (one per recursive parse step, one per link of a member/call chain and one
//! per link of a binary-operator chain that V8 does not fold into an n-ary
//! node, since V8 compiles those nested trees recursively) and decides every
//! body up to [`MAX_DEPTH`] units exactly; deeper bodies are
//! [`FuncSyntax::Undecidable`]. Measured on the oracle's node over about 150
//! nested constructs, the costliest per unit is `for (using x of y)`, whose
//! nesting overflows V8 at 475 levels (1 unit each); `for (let x of y)` overflows
//! at 654, `try {} catch (e) {...}` at 957, comparison chains at 5649 links,
//! member chains at 8877. The limit keeps every one of these below about half of
//! V8's depth. The deepest decided bodies need at most about 0.3 MiB of stack
//! (with a deepest regular expression inside).
//!
//! The limits V8 applies deterministically are checked exactly: at most 65525
//! arguments per call and parameters per function, and (`regexp.rs`) at most
//! 32767 capturing groups per regular expression, whose nesting has its own
//! undecided depth.

mod regexp;
pub(crate) mod unicode;

use std::collections::{HashMap, HashSet};

use regexp::{RegexCheck, validate_regexp_literal};
use unicode::{is_id_part, is_id_start, is_white_space};

/// The verdict on one function body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncSyntax {
    /// V8 accepts the body.
    Valid,
    /// V8 rejects the body (a SyntaxError); the text is a reason for logs, not a
    /// V8 message.
    Invalid(String),
    /// The body is nested deeper than the checker decides (see the module docs).
    Undecidable(String),
}

/// Whether V8's `new Function(param, body)` accepts `body`: `param` (one simple
/// identifier) is the only formal parameter, and `body` is parsed as a
/// sloppy-mode FunctionBody.
pub fn check_function_body(param: &str, body: &str) -> FuncSyntax {
    let src: Vec<u16> = body.encode_utf16().collect();
    let mut parser = Parser::new(src);
    match parser.run(param) {
        Ok(()) => FuncSyntax::Valid,
        Err(Fail::Syntax { why, pos }) => FuncSyntax::Invalid(format!("{why} (offset {pos})")),
        Err(Fail::Undecidable(what)) => FuncSyntax::Undecidable(what),
    }
}

/// Nesting beyond this many units is not decided (see the module docs).
pub const MAX_DEPTH: i32 = 250;

/// The most arguments a call and the most parameters a function may have
/// (V8 14.6 reports more as a SyntaxError: "Too many arguments in function
/// call", "Too many parameters in function definition").
const MAX_ARGUMENTS: usize = 65525;

/// The binary operators V8 folds into one n-ary node when repeated
/// (`Parser::CollapseNaryExpression`: every binary operator but `**`; the
/// comparisons are separate nodes).
fn is_nary_operator(op: &str) -> bool {
    matches!(
        op,
        "||" | "&&" | "??" | "|" | "^" | "&" | "<<" | ">>" | ">>>" | "+" | "-" | "*" | "/" | "%"
    )
}

type Pos = isize;

#[derive(Debug)]
enum Fail {
    Syntax { why: String, pos: Pos },
    Undecidable(String),
}

type R<T> = Result<T, Fail>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Tk {
    #[default]
    Eof,
    Name,
    PrivateId,
    Num,
    String,
    Regexp,
    Template,
    BracketL,
    BracketR,
    BraceL,
    BraceR,
    ParenL,
    ParenR,
    Comma,
    Semi,
    Colon,
    Dot,
    Question,
    QuestionDot,
    Arrow,
    Ellipsis,
    Eq,
    Assign,
    IncDec,
    Prefix,
    LogicalOr,
    LogicalAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseAnd,
    Equality,
    Relational,
    BitShift,
    PlusMin,
    Modulo,
    Star,
    Slash,
    StarStar,
    Coalesce,
}

#[derive(Debug, Clone, Default)]
struct Token {
    ty: Tk,
    start: Pos,
    end: Pos,
    /// Name/PrivateId: cooked name; String: cooked value; operators: source text.
    value: String,
    /// Name/PrivateId written with a `\u` escape.
    escaped: bool,
    /// Num: legacy octal or leading-zero decimal; String: octal or `\8` `\9` escape.
    octal: bool,
    /// Template: the chunk ends with the closing backtick.
    tail: bool,
    /// Template: NotEscapeSequence (valid in tagged templates only).
    bad_escape: bool,
}

const KEYWORDS: &[&str] = &[
    "break",
    "case",
    "catch",
    "continue",
    "debugger",
    "default",
    "do",
    "else",
    "finally",
    "for",
    "function",
    "if",
    "return",
    "switch",
    "throw",
    "try",
    "var",
    "while",
    "with",
    "null",
    "true",
    "false",
    "instanceof",
    "typeof",
    "void",
    "delete",
    "new",
    "in",
    "this",
    "const",
    "class",
    "extends",
    "export",
    "import",
    "super",
];

const STRICT_RESERVED: &[&str] = &[
    "implements",
    "interface",
    "let",
    "package",
    "private",
    "protected",
    "public",
    "static",
    "yield",
];

/// Keywords whose token type starts an expression (acorn `startsExpr`).
const KEYWORD_STARTS_EXPR: &[&str] = &[
    "function", "class", "new", "this", "super", "import", "null", "true", "false", "typeof",
    "void", "delete",
];

fn is_keyword(w: &str) -> bool {
    KEYWORDS.contains(&w)
}

fn is_strict_bind_reserved(name: &str) -> bool {
    STRICT_RESERVED.contains(&name) || name == "eval" || name == "arguments" || name == "enum"
}

// Scope flags (acorn scopeflags.js).
const SCOPE_TOP: i32 = 1;
const SCOPE_FUNCTION: i32 = 2;
const SCOPE_ASYNC: i32 = 4;
const SCOPE_GENERATOR: i32 = 8;
const SCOPE_ARROW: i32 = 16;
const SCOPE_SIMPLE_CATCH: i32 = 32;
const SCOPE_SUPER: i32 = 64;
const SCOPE_DIRECT_SUPER: i32 = 128;
const SCOPE_CLASS_STATIC_BLOCK: i32 = 256;
const SCOPE_CLASS_FIELD_INIT: i32 = 512;
const SCOPE_SWITCH: i32 = 1024;
/// V8: `await` is not an identifier in this initializer (a static field's or a
/// static block's), whose function kind is the class static initializer.
const SCOPE_AWAIT_RESERVED: i32 = 2048;
const SCOPE_VAR: i32 = SCOPE_TOP | SCOPE_FUNCTION | SCOPE_CLASS_STATIC_BLOCK;

fn function_flags(is_async: bool, generator: bool) -> i32 {
    SCOPE_FUNCTION
        | if is_async { SCOPE_ASYNC } else { 0 }
        | if generator { SCOPE_GENERATOR } else { 0 }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Bind {
    None,
    Var,
    Lexical,
    Function,
    SimpleCatch,
    Outside,
}

#[derive(Debug, Default)]
struct Scope {
    flags: i32,
    var: HashSet<String>,
    lexical: HashSet<String>,
    functions: HashSet<String>,
    first_lexical: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelKind {
    None,
    Loop,
    Switch,
}

#[derive(Debug, Clone)]
struct Label {
    name: Option<String>,
    kind: LabelKind,
    statement_start: Pos,
}

#[derive(Debug, Default)]
struct PrivateScope {
    declared: HashSet<String>,
    used: Vec<(String, Pos)>,
}

/// Destructuring bookkeeping (acorn DestructuringErrors); -1 = unset.
#[derive(Debug, Clone, Copy)]
struct DErr {
    shorthand_assign: Pos,
    trailing_comma: Pos,
    parenthesized_assign: Pos,
    parenthesized_bind: Pos,
    double_proto: Pos,
}

impl Default for DErr {
    fn default() -> Self {
        DErr {
            shorthand_assign: -1,
            trailing_comma: -1,
            parenthesized_assign: -1,
            parenthesized_bind: -1,
            double_proto: -1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum NK {
    Ident,
    PrivateName,
    #[default]
    Literal,
    StringLit,
    Regex,
    Template,
    TaggedTemplate,
    This,
    Super,
    Array,
    Object,
    Property,
    Spread,
    Assign,
    Member,
    Call,
    Chain,
    New,
    Unary,
    Update,
    Binary,
    Logical,
    Conditional,
    Sequence,
    Arrow,
    Function,
    Class,
    Yield,
    Await,
    Meta,
    ImportCall,
    ObjectPattern,
    ArrayPattern,
    AssignPattern,
    Rest,
}

/// A node index; [`NONE`] marks an absent node (an array hole, a missing key).
type NodeId = i32;
const NONE: NodeId = -1;

#[derive(Debug, Clone, Default)]
struct Node {
    kind: NK,
    start: Pos,
    end: Pos,
    /// Ident/PrivateName: name; StringLit: cooked value; Property: kind;
    /// Assign/Unary/Update/Binary/Logical: operator.
    name: String,
    /// object / left / argument / key / callee / expression
    a: NodeId,
    /// property / right / value
    b: NodeId,
    /// elements (NONE = hole) / properties / arguments / params / expressions
    list: Vec<NodeId>,
    computed: bool,
    shorthand: bool,
    method: bool,
    optional: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct VarResult {
    count: i32,
    first_has_init: bool,
    first_is_ident: bool,
}

const FUNC_STATEMENT: i32 = 1;
const FUNC_HANGING_STATEMENT: i32 = 2;
const FUNC_NULLABLE_ID: i32 = 4;
const FUNC_NO_DECLARE: i32 = 8;

fn is_new_line_code(c: u16) -> bool {
    c == 0x0A || c == 0x0D || c == 0x2028 || c == 0x2029
}

/// A statement context (acorn's `context` argument): the statement position a
/// statement is parsed in. `None` is a declaration position.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Ctx {
    None,
    If,
    Label,
    Do,
    For,
    While,
    With,
    /// A labelled statement in another single-statement position (`ifl`, `forl`…).
    LabelIn(Box<Ctx>),
}

impl Ctx {
    fn is_none(&self) -> bool {
        matches!(self, Ctx::None)
    }
    fn contains_label(&self) -> bool {
        matches!(self, Ctx::Label | Ctx::LabelIn(_))
    }
}

struct SavedState {
    pos: Pos,
    tok: Token,
    last_tok_start: Pos,
    last_tok_end: Pos,
}

struct Parser {
    src: Vec<u16>,
    n: Pos,

    pos: Pos,
    tok: Token,
    last_tok_start: Pos,
    last_tok_end: Pos,
    strict: bool,
    potential_arrow_at: Pos,
    potential_arrow_in_for_await: bool,
    /// V8 lazily pre-parses the function being parsed (see [`Parser::parse_function`]).
    preparse: bool,
    /// V8's `next_function_is_likely_called` hint: the next function literal
    /// follows `(` or `!` and is compiled eagerly.
    next_function_likely_called: bool,
    yield_pos: Pos,
    await_pos: Pos,
    await_ident_pos: Pos,
    labels: Vec<Label>,
    scopes: Vec<Scope>,
    private_names: Vec<PrivateScope>,
    nodes: Vec<Node>,
    depth: i32,
}

/// `Unexpected token` at the current token.
macro_rules! unexpected {
    ($p:expr) => {
        return $p.unexpected_at($p.tok.start)
    };
}

impl Parser {
    fn new(src: Vec<u16>) -> Self {
        let n = src.len() as Pos;
        Parser {
            src,
            n,
            pos: 0,
            tok: Token::default(),
            last_tok_start: 0,
            last_tok_end: 0,
            strict: false,
            potential_arrow_at: -1,
            potential_arrow_in_for_await: false,
            preparse: false,
            next_function_likely_called: false,
            yield_pos: -1,
            await_pos: -1,
            await_ident_pos: -1,
            labels: Vec::new(),
            scopes: Vec::new(),
            private_names: Vec::new(),
            nodes: Vec::new(),
            depth: 0,
        }
    }

    // ------------------------------------------------------------ errors, depth

    fn raise<T>(&self, pos: Pos, why: impl Into<String>) -> R<T> {
        Err(Fail::Syntax {
            why: why.into(),
            pos,
        })
    }

    fn unexpected_at<T>(&self, pos: Pos) -> R<T> {
        self.raise(pos, "Unexpected token")
    }

    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(Fail::Undecidable(format!(
                "nesting deeper than {MAX_DEPTH}"
            )));
        }
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    // ------------------------------------------------------------ source access

    fn ch(&self, i: Pos) -> u16 {
        if i >= 0 && i < self.n {
            self.src[i as usize]
        } else {
            0
        }
    }

    fn has(&self, i: Pos) -> bool {
        i >= 0 && i < self.n
    }

    fn full_char_code_at(&self, i: Pos) -> u32 {
        if i >= self.n {
            return 0xFFFF_FFFF;
        }
        let c = self.ch(i);
        if !(0xD800..=0xDBFF).contains(&c) || i + 1 >= self.n {
            return u32::from(c);
        }
        let next = self.ch(i + 1);
        if !(0xDC00..=0xDFFF).contains(&next) {
            return u32::from(c);
        }
        0x10000 + ((u32::from(c) - 0xD800) << 10) + (u32::from(next) - 0xDC00)
    }

    fn slice(&self, from: Pos, len: Pos) -> String {
        let start = from.clamp(0, self.n) as usize;
        let end = (from + len).clamp(0, self.n) as usize;
        String::from_utf16_lossy(&self.src[start..end.max(start)])
    }

    fn index_of(&self, needle: &[u16], from: Pos) -> Pos {
        let from = from.max(0) as usize;
        if needle.is_empty() || from >= self.src.len() {
            return -1;
        }
        self.src[from..]
            .windows(needle.len())
            .position(|w| w == needle)
            .map_or(-1, |p| (from + p) as Pos)
    }

    // ------------------------------------------------------------ nodes

    fn node_of(&mut self, kind: NK, start: Pos) -> NodeId {
        self.nodes.push(Node {
            kind,
            start,
            a: NONE,
            b: NONE,
            ..Node::default()
        });
        (self.nodes.len() - 1) as NodeId
    }

    fn nd(&self, i: NodeId) -> &Node {
        &self.nodes[i as usize]
    }

    fn nd_mut(&mut self, i: NodeId) -> &mut Node {
        &mut self.nodes[i as usize]
    }

    fn finish(&mut self, i: NodeId) -> NodeId {
        let end = self.last_tok_end;
        self.nd_mut(i).end = end;
        i
    }

    // ------------------------------------------------------------ token tests

    fn is_keyword_token(&self) -> bool {
        self.tok.ty == Tk::Name && is_keyword(&self.tok.value)
    }

    fn is_kw(&self, w: &str) -> bool {
        self.tok.ty == Tk::Name && self.tok.value == w
    }

    fn is_contextual(&self, w: &str) -> bool {
        self.tok.ty == Tk::Name && !self.tok.escaped && self.tok.value == w
    }

    fn is_name_tok(&self) -> bool {
        self.tok.ty == Tk::Name && !is_keyword(&self.tok.value)
    }

    fn next(&mut self) -> R<()> {
        self.next_ext(false)
    }

    fn next_ext(&mut self, ignore_escape_in_keyword: bool) -> R<()> {
        if !ignore_escape_in_keyword && self.is_keyword_token() && self.tok.escaped {
            return self.raise(
                self.tok.start,
                format!("Escape sequence in keyword {}", self.tok.value),
            );
        }
        self.last_tok_end = self.tok.end;
        self.last_tok_start = self.tok.start;
        self.next_token()
    }

    fn eat(&mut self, t: Tk) -> R<bool> {
        if self.tok.ty == t {
            self.next()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn eat_kw(&mut self, w: &str) -> R<bool> {
        if self.is_kw(w) {
            self.next()?;
            return Ok(true);
        }
        Ok(false)
    }

    fn eat_contextual(&mut self, w: &str) -> R<bool> {
        if !self.is_contextual(w) {
            return Ok(false);
        }
        self.next()?;
        Ok(true)
    }

    fn expect(&mut self, t: Tk) -> R<()> {
        if !self.eat(t)? {
            unexpected!(self);
        }
        Ok(())
    }

    fn has_line_break(&self, from: Pos, to: Pos) -> bool {
        let mut i = from.max(0);
        while i < to && i < self.n {
            if is_new_line_code(self.ch(i)) {
                return true;
            }
            i += 1;
        }
        false
    }

    fn can_insert_semicolon(&self) -> bool {
        self.tok.ty == Tk::Eof
            || self.tok.ty == Tk::BraceR
            || self.has_line_break(self.last_tok_end, self.tok.start)
    }

    fn semicolon(&mut self) -> R<()> {
        if !self.eat(Tk::Semi)? && !self.can_insert_semicolon() {
            unexpected!(self);
        }
        Ok(())
    }

    fn after_trailing_comma(&mut self, t: Tk, not_next: bool) -> R<bool> {
        if self.tok.ty == t {
            if !not_next {
                self.next()?;
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn save(&self) -> SavedState {
        SavedState {
            pos: self.pos,
            tok: self.tok.clone(),
            last_tok_start: self.last_tok_start,
            last_tok_end: self.last_tok_end,
        }
    }

    fn restore(&mut self, s: SavedState) {
        self.pos = s.pos;
        self.tok = s.tok;
        self.last_tok_start = s.last_tok_start;
        self.last_tok_end = s.last_tok_end;
    }

    /// The token after the current one (scanned then discarded). A scan error
    /// there reports an Eof token.
    fn peek_token(&mut self) -> R<Token> {
        let s = self.save();
        self.last_tok_end = self.tok.end;
        self.last_tok_start = self.tok.start;
        let t = match self.next_token() {
            Ok(()) => self.tok.clone(),
            Err(Fail::Syntax { .. }) => Token {
                ty: Tk::Eof,
                start: self.n,
                end: self.n,
                ..Token::default()
            },
            Err(e) => return Err(e),
        };
        self.restore(s);
        Ok(t)
    }

    // ------------------------------------------------------------ scanner

    fn skip_line_comment(&mut self, start_skip: Pos) {
        self.pos += start_skip;
        while self.pos < self.n && !is_new_line_code(self.ch(self.pos)) {
            self.pos += 1;
        }
    }

    fn skip_block_comment(&mut self) -> R<()> {
        let start = self.pos;
        let end = self.index_of(&[u16::from(b'*'), u16::from(b'/')], self.pos + 2);
        if end == -1 {
            return self.raise(start, "Unterminated comment");
        }
        self.pos = end + 2;
        Ok(())
    }

    fn skip_space(&mut self) -> R<()> {
        while self.pos < self.n {
            let c = self.ch(self.pos);
            if c == 32 || c == 160 {
                self.pos += 1;
            } else if c == 13 {
                if self.ch(self.pos + 1) == 10 {
                    self.pos += 1;
                }
                self.pos += 1;
            } else if c == 10 || c == 0x2028 || c == 0x2029 {
                self.pos += 1;
            } else if c == u16::from(b'/') {
                let n2 = self.ch(self.pos + 1);
                if n2 == u16::from(b'*') && self.has(self.pos + 1) {
                    self.skip_block_comment()?;
                } else if n2 == u16::from(b'/') && self.has(self.pos + 1) {
                    self.skip_line_comment(2);
                } else {
                    break;
                }
            } else if (c > 8 && c < 14) || (c >= 5760 && is_white_space(u32::from(c))) {
                self.pos += 1;
            } else {
                break;
            }
        }
        Ok(())
    }

    fn finish_token(&mut self, ty: Tk, value: String) {
        self.tok.ty = ty;
        self.tok.end = self.pos;
        self.tok.value = value;
    }

    fn finish_op(&mut self, ty: Tk, size: Pos) {
        let s = self.slice(self.pos, size);
        self.pos += size;
        self.finish_token(ty, s);
    }

    fn next_token(&mut self) -> R<()> {
        self.skip_space()?;
        self.tok = Token {
            start: self.pos,
            ..Token::default()
        };
        if self.pos >= self.n {
            self.finish_token(Tk::Eof, String::new());
            return Ok(());
        }
        let code = self.full_char_code_at(self.pos);
        self.read_token(code)
    }

    fn read_token(&mut self, code: u32) -> R<()> {
        if is_id_start(code) || code == u32::from(b'\\') {
            return self.read_word();
        }
        self.get_token_from_code(code)
    }

    fn read_word(&mut self) -> R<()> {
        let (word, escaped) = self.read_word1()?;
        self.finish_token(Tk::Name, word);
        self.tok.escaped = escaped;
        Ok(())
    }

    /// An identifier name from the current position: (cooked word, contains an
    /// escape).
    fn read_word1(&mut self) -> R<(String, bool)> {
        let mut contains_esc = false;
        let mut word: Vec<u16> = Vec::new();
        let mut first = true;
        let mut chunk_start = self.pos;
        while self.pos < self.n {
            let c = self.full_char_code_at(self.pos);
            if is_id_part(c) {
                self.pos += if c <= 0xFFFF { 1 } else { 2 };
            } else if c == u32::from(b'\\') {
                contains_esc = true;
                word.extend_from_slice(&self.src[chunk_start as usize..self.pos as usize]);
                let esc_start = self.pos;
                self.pos += 1;
                if self.ch(self.pos) != u16::from(b'u') || !self.has(self.pos) {
                    return self.raise(self.pos, "Expecting Unicode escape sequence \\uXXXX");
                }
                self.pos += 1;
                let esc = self.read_code_point();
                let ok = esc >= 0 && {
                    let cp = esc as u32;
                    if first {
                        is_id_start(cp)
                    } else {
                        is_id_part(cp)
                    }
                };
                if !ok {
                    return self.raise(esc_start, "Invalid Unicode escape");
                }
                let mut buf = [0u16; 2];
                if let Some(c) = char::from_u32(esc as u32) {
                    word.extend_from_slice(c.encode_utf16(&mut buf));
                }
                chunk_start = self.pos;
            } else {
                break;
            }
            first = false;
        }
        word.extend_from_slice(&self.src[chunk_start as usize..self.pos as usize]);
        Ok((String::from_utf16_lossy(&word), contains_esc))
    }

    /// Reads the digits of `\u{...}` or `\uXXXX` (pos after the `u`). Returns -1
    /// for an invalid escape, leaving the position where acorn's
    /// `readCodePoint` leaves it.
    fn read_code_point(&mut self) -> i64 {
        if self.ch(self.pos) == u16::from(b'{') && self.has(self.pos) {
            self.pos += 1;
            // acorn: readHexChar(input.indexOf("}", pos) - pos); a missing brace
            // gives a negative length, which reads nothing.
            let close = self.index_of(&[u16::from(b'}')], self.pos);
            let code = self.read_hex_char(close - self.pos);
            if code < 0 {
                // readHexChar fails before the closing brace is skipped.
                return -1;
            }
            self.pos += 1;
            if code > 0x10FFFF {
                return -1;
            }
            return code;
        }
        self.read_hex_char(4)
    }

    /// acorn `readInt(16, len)` for an escape: exactly `len` hex digits (no
    /// separators). Returns -1 when they are not there, with the position after
    /// the digits read.
    fn read_hex_char(&mut self, len: Pos) -> i64 {
        let start = self.pos;
        let mut total: i64 = 0;
        let mut i = 0;
        while i < len {
            if !self.has(self.pos) {
                break;
            }
            let c = self.ch(self.pos);
            let val = match c {
                0x30..=0x39 => i64::from(c - 0x30),
                0x61..=0x66 => i64::from(c - 0x61 + 10),
                0x41..=0x46 => i64::from(c - 0x41 + 10),
                _ => break,
            };
            if total <= 0x10FFFF {
                total = total * 16 + val;
            }
            self.pos += 1;
            i += 1;
        }
        if self.pos == start || self.pos - start != len {
            return -1;
        }
        total
    }

    fn get_token_from_code(&mut self, code: u32) -> R<()> {
        let c = |b: u8| u32::from(b);
        let u = |b: u8| u16::from(b);
        match code {
            x if x == c(b'.') => {
                let next = self.ch(self.pos + 1);
                if self.has(self.pos + 1) && (u(b'0')..=u(b'9')).contains(&next) {
                    return self.read_number(true);
                }
                if next == u(b'.') && self.ch(self.pos + 2) == u(b'.') && self.has(self.pos + 2) {
                    self.pos += 3;
                    self.finish_token(Tk::Ellipsis, String::new());
                    return Ok(());
                }
                self.pos += 1;
                self.finish_token(Tk::Dot, String::new());
                Ok(())
            }
            x if x == c(b'(') => self.punct(Tk::ParenL),
            x if x == c(b')') => self.punct(Tk::ParenR),
            x if x == c(b';') => self.punct(Tk::Semi),
            x if x == c(b',') => self.punct(Tk::Comma),
            x if x == c(b'[') => self.punct(Tk::BracketL),
            x if x == c(b']') => self.punct(Tk::BracketR),
            x if x == c(b'{') => self.punct(Tk::BraceL),
            x if x == c(b'}') => self.punct(Tk::BraceR),
            x if x == c(b':') => self.punct(Tk::Colon),
            x if x == c(b'`') => {
                let from = self.pos + 1;
                self.read_template_chunk(from)
            }
            x if x == c(b'0') => {
                let next = self.ch(self.pos + 1);
                if next == u(b'x') || next == u(b'X') {
                    return self.read_radix_number(16);
                }
                if next == u(b'o') || next == u(b'O') {
                    return self.read_radix_number(8);
                }
                if next == u(b'b') || next == u(b'B') {
                    return self.read_radix_number(2);
                }
                self.read_number(false)
            }
            x if (c(b'1')..=c(b'9')).contains(&x) => self.read_number(false),
            x if x == c(b'"') || x == c(b'\'') => self.read_string(x as u16),
            x if x == c(b'/') => {
                if self.ch(self.pos + 1) == u(b'=') {
                    self.finish_op(Tk::Assign, 2);
                } else {
                    self.finish_op(Tk::Slash, 1);
                }
                Ok(())
            }
            x if x == c(b'%') || x == c(b'*') => {
                let mut size = 1;
                let mut ty = if x == c(b'*') { Tk::Star } else { Tk::Modulo };
                let mut next = self.ch(self.pos + 1);
                if x == c(b'*') && next == u(b'*') {
                    size += 1;
                    ty = Tk::StarStar;
                    next = self.ch(self.pos + 2);
                }
                if next == u(b'=') && self.has(self.pos + size) {
                    self.finish_op(Tk::Assign, size + 1);
                } else {
                    self.finish_op(ty, size);
                }
                Ok(())
            }
            x if x == c(b'|') || x == c(b'&') => {
                let next = self.ch(self.pos + 1);
                if u32::from(next) == x {
                    if self.ch(self.pos + 2) == u(b'=') {
                        self.finish_op(Tk::Assign, 3);
                        return Ok(());
                    }
                    self.finish_op(
                        if x == c(b'|') {
                            Tk::LogicalOr
                        } else {
                            Tk::LogicalAnd
                        },
                        2,
                    );
                    return Ok(());
                }
                if next == u(b'=') {
                    self.finish_op(Tk::Assign, 2);
                    return Ok(());
                }
                self.finish_op(
                    if x == c(b'|') {
                        Tk::BitwiseOr
                    } else {
                        Tk::BitwiseAnd
                    },
                    1,
                );
                Ok(())
            }
            x if x == c(b'^') => {
                if self.ch(self.pos + 1) == u(b'=') {
                    self.finish_op(Tk::Assign, 2);
                } else {
                    self.finish_op(Tk::BitwiseXor, 1);
                }
                Ok(())
            }
            x if x == c(b'+') || x == c(b'-') => {
                let next = self.ch(self.pos + 1);
                if u32::from(next) == x {
                    if next == u(b'-')
                        && self.ch(self.pos + 2) == u(b'>')
                        && (self.last_tok_end == 0
                            || self.has_line_break(self.last_tok_end, self.pos))
                    {
                        // A `-->` line comment (Annex B, script goal).
                        self.skip_line_comment(3);
                        return self.next_token();
                    }
                    self.finish_op(Tk::IncDec, 2);
                    return Ok(());
                }
                if next == u(b'=') {
                    self.finish_op(Tk::Assign, 2);
                    return Ok(());
                }
                self.finish_op(Tk::PlusMin, 1);
                Ok(())
            }
            x if x == c(b'<') || x == c(b'>') => {
                let next = self.ch(self.pos + 1);
                if u32::from(next) == x && self.has(self.pos + 1) {
                    let size = if x == c(b'>') && self.ch(self.pos + 2) == u(b'>') {
                        3
                    } else {
                        2
                    };
                    if self.ch(self.pos + size) == u(b'=') && self.has(self.pos + size) {
                        self.finish_op(Tk::Assign, size + 1);
                        return Ok(());
                    }
                    self.finish_op(Tk::BitShift, size);
                    return Ok(());
                }
                if next == u(b'!')
                    && x == c(b'<')
                    && self.ch(self.pos + 2) == u(b'-')
                    && self.ch(self.pos + 3) == u(b'-')
                {
                    // `<!--`, an HTML-like line comment (Annex B, script goal).
                    self.skip_line_comment(4);
                    return self.next_token();
                }
                let size = if next == u(b'=') { 2 } else { 1 };
                self.finish_op(Tk::Relational, size);
                Ok(())
            }
            x if x == c(b'=') || x == c(b'!') => {
                let next = self.ch(self.pos + 1);
                if next == u(b'=') {
                    let size = if self.ch(self.pos + 2) == u(b'=') {
                        3
                    } else {
                        2
                    };
                    self.finish_op(Tk::Equality, size);
                    return Ok(());
                }
                if x == c(b'=') && next == u(b'>') {
                    self.pos += 2;
                    self.finish_token(Tk::Arrow, String::new());
                    return Ok(());
                }
                self.finish_op(if x == c(b'=') { Tk::Eq } else { Tk::Prefix }, 1);
                Ok(())
            }
            x if x == c(b'?') => {
                let next = self.ch(self.pos + 1);
                if next == u(b'.') {
                    let next2 = self.ch(self.pos + 2);
                    if !(self.has(self.pos + 2) && (u(b'0')..=u(b'9')).contains(&next2)) {
                        self.finish_op(Tk::QuestionDot, 2);
                        return Ok(());
                    }
                }
                if next == u(b'?') {
                    if self.ch(self.pos + 2) == u(b'=') {
                        self.finish_op(Tk::Assign, 3);
                        return Ok(());
                    }
                    self.finish_op(Tk::Coalesce, 2);
                    return Ok(());
                }
                self.finish_op(Tk::Question, 1);
                Ok(())
            }
            x if x == c(b'~') => {
                self.finish_op(Tk::Prefix, 1);
                Ok(())
            }
            x if x == c(b'#') => {
                self.pos += 1;
                let c2 = self.full_char_code_at(self.pos);
                if self.has(self.pos) && (is_id_start(c2) || c2 == c(b'\\')) {
                    let (word, escaped) = self.read_word1()?;
                    self.finish_token(Tk::PrivateId, word);
                    self.tok.escaped = escaped;
                    return Ok(());
                }
                self.raise(self.pos, "Unexpected character '#'")
            }
            _ => self.raise(self.pos, "Unexpected character"),
        }
    }

    fn punct(&mut self, ty: Tk) -> R<()> {
        self.pos += 1;
        self.finish_token(ty, String::new());
        Ok(())
    }

    /// Digits in `radix`; separators allowed when `len < 0`. Returns the digit
    /// count, or -1 when none were read (or `len` was not met).
    fn read_int(&mut self, radix: i32, len: Pos, maybe_legacy_octal: bool) -> R<Pos> {
        let allow_separators = len < 0;
        let is_legacy_octal = maybe_legacy_octal && self.ch(self.pos) == u16::from(b'0');
        let start = self.pos;
        let mut last_code: u16 = 0;
        let mut i: Pos = 0;
        while len < 0 || i < len {
            if !self.has(self.pos) {
                break;
            }
            let code = self.ch(self.pos);
            if allow_separators && code == u16::from(b'_') {
                if is_legacy_octal {
                    return self.raise(
                        self.pos,
                        "Numeric separator is not allowed in legacy octal numeric literals",
                    );
                }
                if last_code == u16::from(b'_') {
                    return self
                        .raise(self.pos, "Numeric separator must be exactly one underscore");
                }
                if i == 0 {
                    return self.raise(
                        self.pos,
                        "Numeric separator is not allowed at the first of digits",
                    );
                }
                last_code = code;
                i += 1;
                self.pos += 1;
                continue;
            }
            let code_i = i32::from(code);
            let val = if code >= u16::from(b'a') {
                code_i - i32::from(b'a') + 10
            } else if code >= u16::from(b'A') {
                code_i - i32::from(b'A') + 10
            } else if (u16::from(b'0')..=u16::from(b'9')).contains(&code) {
                code_i - i32::from(b'0')
            } else {
                99
            };
            if val >= radix {
                break;
            }
            last_code = code;
            i += 1;
            self.pos += 1;
        }
        if allow_separators && last_code == u16::from(b'_') {
            return self.raise(
                self.pos - 1,
                "Numeric separator is not allowed at the last of digits",
            );
        }
        if self.pos == start || (len >= 0 && self.pos - start != len) {
            return Ok(-1);
        }
        Ok(self.pos - start)
    }

    fn read_radix_number(&mut self, radix: i32) -> R<()> {
        self.pos += 2;
        if self.read_int(radix, -1, false)? < 0 {
            return self.raise(
                self.tok.start + 2,
                format!("Expected number in radix {radix}"),
            );
        }
        if self.ch(self.pos) == u16::from(b'n') && self.has(self.pos) {
            self.pos += 1;
        }
        if self.has(self.pos) && is_id_start(self.full_char_code_at(self.pos)) {
            return self.raise(self.pos, "Identifier directly after number");
        }
        self.finish_token(Tk::Num, String::new());
        Ok(())
    }

    fn read_number(&mut self, starts_with_dot: bool) -> R<()> {
        let start = self.pos;
        if !starts_with_dot && self.read_int(10, -1, true)? < 0 {
            return self.raise(start, "Invalid number");
        }
        let mut octal = self.pos - start >= 2 && self.ch(start) == u16::from(b'0');
        let leading_zero = octal;
        let mut next = if self.has(self.pos) {
            self.ch(self.pos)
        } else {
            0
        };
        if !octal && !starts_with_dot && next == u16::from(b'n') {
            self.pos += 1;
            if self.has(self.pos) && is_id_start(self.full_char_code_at(self.pos)) {
                return self.raise(self.pos, "Identifier directly after number");
            }
            self.finish_token(Tk::Num, String::new());
            return Ok(());
        }
        if octal {
            for i in start..self.pos {
                if self.ch(i) == u16::from(b'8') || self.ch(i) == u16::from(b'9') {
                    octal = false;
                    break;
                }
            }
        }
        if next == u16::from(b'.') && !octal {
            self.pos += 1;
            self.read_int(10, -1, false)?;
            next = if self.has(self.pos) {
                self.ch(self.pos)
            } else {
                0
            };
        }
        if (next == u16::from(b'E') || next == u16::from(b'e')) && !octal {
            self.pos += 1;
            next = self.ch(self.pos);
            if (next == u16::from(b'+') || next == u16::from(b'-')) && self.has(self.pos) {
                self.pos += 1;
            }
            if self.read_int(10, -1, false)? < 0 {
                return self.raise(start, "Invalid number");
            }
        }
        if self.has(self.pos) && is_id_start(self.full_char_code_at(self.pos)) {
            return self.raise(self.pos, "Identifier directly after number");
        }
        self.finish_token(Tk::Num, String::new());
        self.tok.octal = leading_zero;
        Ok(())
    }

    fn read_string(&mut self, quote: u16) -> R<()> {
        let mut out: Vec<u16> = Vec::new();
        self.pos += 1;
        let mut chunk_start = self.pos;
        let mut octal = false;
        loop {
            if self.pos >= self.n {
                return self.raise(self.tok.start, "Unterminated string constant");
            }
            let c = self.ch(self.pos);
            if c == quote {
                break;
            }
            if c == u16::from(b'\\') {
                out.extend_from_slice(&self.src[chunk_start as usize..self.pos as usize]);
                match self.read_escaped_char(false, &mut out) {
                    Escape::Ok { octal: o } => octal |= o,
                    Escape::Bad => return self.raise(self.pos, "Bad character escape sequence"),
                }
                chunk_start = self.pos;
            } else if c == 0x2028 || c == 0x2029 {
                self.pos += 1;
            } else {
                if is_new_line_code(c) {
                    return self.raise(self.tok.start, "Unterminated string constant");
                }
                self.pos += 1;
            }
        }
        out.extend_from_slice(&self.src[chunk_start as usize..self.pos as usize]);
        self.pos += 1;
        self.finish_token(Tk::String, String::from_utf16_lossy(&out));
        self.tok.octal = octal;
        Ok(())
    }

    /// Reads one escape (pos at the backslash), appending its cooked value to
    /// `out`. Reports legacy octal escapes and `\8` `\9` (which a template rejects
    /// like any invalid escape); for an invalid escape the position is where
    /// acorn's `readEscapedChar` leaves it when it throws.
    fn read_escaped_char(&mut self, in_template: bool, out: &mut Vec<u16>) -> Escape {
        if !self.has(self.pos + 1) {
            self.pos += 1;
            return Escape::Bad;
        }
        self.pos += 1;
        let c = self.ch(self.pos);
        self.pos += 1;
        let push = |out: &mut Vec<u16>, s: &[u16]| out.extend_from_slice(s);
        match c {
            0x6E => push(out, &[0x0A]),
            0x72 => push(out, &[0x0D]),
            0x74 => push(out, &[0x09]),
            0x62 => push(out, &[0x08]),
            0x76 => push(out, &[0x0B]),
            0x66 => push(out, &[0x0C]),
            0x78 => {
                let v = self.read_hex_char(2);
                if v < 0 {
                    return Escape::Bad;
                }
                push(out, &[v as u16]);
            }
            0x75 => {
                let v = self.read_code_point();
                if v < 0 {
                    return Escape::Bad;
                }
                if let Some(ch) = char::from_u32(v as u32) {
                    let mut buf = [0u16; 2];
                    push(out, ch.encode_utf16(&mut buf));
                } else {
                    push(out, &[v as u16]);
                }
            }
            13 => {
                if self.ch(self.pos) == 10 && self.has(self.pos) {
                    self.pos += 1;
                }
            }
            10 => {}
            0x38 | 0x39 => {
                if in_template {
                    return Escape::Bad;
                }
                push(out, &[c]);
                return Escape::Ok { octal: true };
            }
            0x30..=0x37 => {
                // Longest octal run of up to three digits whose value fits 0-255.
                let mut len: Pos = 1;
                while len < 3
                    && self.has(self.pos - 1 + len)
                    && (0x30..=0x37).contains(&self.ch(self.pos - 1 + len))
                {
                    len += 1;
                }
                let digits = |p: &Self, l: Pos| {
                    (0..l).fold(0u32, |acc, k| {
                        acc * 8 + u32::from(p.ch(p.pos - 1 + k) - 0x30)
                    })
                };
                if digits(self, len) > 255 {
                    len -= 1;
                }
                let value = digits(self, len);
                let is_zero = len == 1 && self.ch(self.pos - 1) == 0x30;
                self.pos += len - 1;
                let after = if self.has(self.pos) {
                    self.ch(self.pos)
                } else {
                    0
                };
                let octal = !is_zero || after == 0x38 || after == 0x39;
                if octal && in_template {
                    return Escape::Bad;
                }
                push(out, &[value as u16]);
                return Escape::Ok { octal };
            }
            _ => {
                if !is_new_line_code(c) {
                    push(out, &[c]);
                }
            }
        }
        Escape::Ok { octal: false }
    }

    /// A template chunk from `from` (just after the backtick or `}`) to `${` or
    /// the closing backtick (acorn `readTmplToken`). At the first invalid escape
    /// the chunk is finished without validating escapes, from where the escape
    /// scan stopped (acorn `readInvalidTemplateToken`).
    fn read_template_chunk(&mut self, from: Pos) -> R<()> {
        self.pos = from;
        loop {
            if self.pos >= self.n {
                return self.raise(self.tok.start, "Unterminated template");
            }
            let c = self.ch(self.pos);
            if c == u16::from(b'`') {
                self.pos += 1;
                self.finish_token(Tk::Template, String::new());
                self.tok.tail = true;
                return Ok(());
            }
            if c == u16::from(b'$')
                && self.ch(self.pos + 1) == u16::from(b'{')
                && self.has(self.pos + 1)
            {
                self.pos += 2;
                self.finish_token(Tk::Template, String::new());
                self.tok.tail = false;
                return Ok(());
            }
            if c == u16::from(b'\\') {
                let mut sink = Vec::new();
                if let Escape::Bad = self.read_escaped_char(true, &mut sink) {
                    return self.read_invalid_template_chunk();
                }
            } else {
                self.pos += 1;
            }
        }
    }

    /// acorn `readInvalidTemplateToken`: the rest of a chunk that holds an
    /// invalid escape (valid only in a tagged template).
    fn read_invalid_template_chunk(&mut self) -> R<()> {
        while self.pos < self.n {
            let c = self.ch(self.pos);
            if c == u16::from(b'\\') {
                self.pos += 1;
            } else if c == u16::from(b'`') {
                self.pos += 1;
                self.finish_token(Tk::Template, String::new());
                self.tok.tail = true;
                self.tok.bad_escape = true;
                return Ok(());
            } else if c == u16::from(b'$') && self.ch(self.pos + 1) == u16::from(b'{') {
                self.pos += 2;
                self.finish_token(Tk::Template, String::new());
                self.tok.tail = false;
                self.tok.bad_escape = true;
                return Ok(());
            }
            self.pos += 1;
        }
        self.raise(self.tok.start, "Unterminated template")
    }

    /// Re-reads the current `/` or `/=` token as a regular expression.
    fn read_regexp(&mut self) -> R<()> {
        let start = self.tok.start + 1;
        self.pos = start;
        let mut escaped = false;
        let mut in_class = false;
        loop {
            if self.pos >= self.n {
                return self.raise(start, "Unterminated regular expression");
            }
            let c = self.ch(self.pos);
            if is_new_line_code(c) {
                return self.raise(start, "Unterminated regular expression");
            }
            if !escaped {
                if c == u16::from(b'[') {
                    in_class = true;
                } else if c == u16::from(b']') && in_class {
                    in_class = false;
                } else if c == u16::from(b'/') && !in_class {
                    break;
                }
                escaped = c == u16::from(b'\\');
            } else {
                escaped = false;
            }
            self.pos += 1;
        }
        let pattern: Vec<u16> = self.src[start as usize..self.pos as usize].to_vec();
        self.pos += 1;
        let flags_start = self.pos;
        let (flags, contains_esc) = self.read_word1()?;
        if contains_esc {
            return self.unexpected_at(flags_start);
        }
        let flags: Vec<u16> = flags.encode_utf16().collect();
        match validate_regexp_literal(&pattern, &flags) {
            RegexCheck::Valid => {}
            RegexCheck::Invalid(error) => return self.raise(start, error),
            RegexCheck::Undecidable(error) => return Err(Fail::Undecidable(error)),
        }
        self.finish_token(Tk::Regexp, String::new());
        Ok(())
    }

    // ------------------------------------------------------------ scopes

    fn current_scope(&mut self) -> &mut Scope {
        self.scopes.last_mut().expect("a scope is open")
    }

    fn var_scope_index(&self) -> usize {
        for i in (0..self.scopes.len()).rev() {
            if self.scopes[i].flags
                & (SCOPE_VAR | SCOPE_CLASS_FIELD_INIT | SCOPE_CLASS_STATIC_BLOCK)
                != 0
            {
                return i;
            }
        }
        0
    }

    fn this_scope_flags(&self) -> i32 {
        for s in self.scopes.iter().rev() {
            let f = s.flags;
            if f & (SCOPE_VAR | SCOPE_CLASS_FIELD_INIT | SCOPE_CLASS_STATIC_BLOCK) != 0
                && f & SCOPE_ARROW == 0
            {
                return f;
            }
        }
        self.scopes[0].flags
    }

    fn var_scope_flags(&self) -> i32 {
        self.scopes[self.var_scope_index()].flags
    }

    fn in_function(&self) -> bool {
        self.var_scope_flags() & SCOPE_FUNCTION != 0
    }
    fn in_generator(&self) -> bool {
        self.var_scope_flags() & SCOPE_GENERATOR != 0
    }
    fn in_async(&self) -> bool {
        self.var_scope_flags() & SCOPE_ASYNC != 0
    }
    fn can_await(&self) -> bool {
        for s in self.scopes.iter().rev() {
            let f = s.flags;
            if f & (SCOPE_CLASS_STATIC_BLOCK | SCOPE_CLASS_FIELD_INIT) != 0 {
                return false;
            }
            if f & SCOPE_FUNCTION != 0 {
                return f & SCOPE_ASYNC != 0;
            }
        }
        false
    }
    fn allow_super(&self) -> bool {
        self.this_scope_flags() & SCOPE_SUPER != 0
    }
    fn allow_direct_super(&self) -> bool {
        self.this_scope_flags() & SCOPE_DIRECT_SUPER != 0
    }
    fn treat_functions_as_var_in_scope(s: &Scope) -> bool {
        // V8 (unlike acorn) also declares a static block's top-level functions
        // like vars (`static { var f; function f() {} }` is valid).
        s.flags & (SCOPE_FUNCTION | SCOPE_TOP | SCOPE_CLASS_STATIC_BLOCK) != 0
    }
    fn treat_functions_as_var(&self) -> bool {
        Self::treat_functions_as_var_in_scope(self.scopes.last().expect("a scope is open"))
    }
    fn allow_new_dot_target(&self) -> bool {
        for s in self.scopes.iter().rev() {
            let f = s.flags;
            if f & (SCOPE_CLASS_STATIC_BLOCK | SCOPE_CLASS_FIELD_INIT) != 0
                || (f & SCOPE_FUNCTION != 0 && f & SCOPE_ARROW == 0)
            {
                return true;
            }
        }
        false
    }
    fn allow_using(&self) -> bool {
        let f = self.scopes.last().expect("a scope is open").flags;
        if f & SCOPE_SWITCH != 0 {
            return false;
        }
        if f & SCOPE_TOP != 0 {
            return false;
        }
        true
    }
    fn in_class_static_block(&self) -> bool {
        self.var_scope_flags() & SCOPE_CLASS_STATIC_BLOCK != 0
    }

    fn enter_scope(&mut self, flags: i32) {
        self.scopes.push(Scope {
            flags,
            ..Scope::default()
        });
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn add_lexical(scope: &mut Scope, name: &str) {
        if scope.first_lexical.is_none() {
            scope.first_lexical = Some(name.to_owned());
        }
        scope.lexical.insert(name.to_owned());
    }

    fn declare_name(&mut self, name: &str, binding_type: Bind, pos: Pos) -> R<()> {
        let mut redeclared = false;
        match binding_type {
            Bind::Lexical => {
                let scope = self.current_scope();
                redeclared = scope.lexical.contains(name)
                    || scope.functions.contains(name)
                    || scope.var.contains(name);
                Self::add_lexical(scope, name);
            }
            Bind::SimpleCatch => {
                Self::add_lexical(self.current_scope(), name);
            }
            Bind::Function => {
                let as_var = self.treat_functions_as_var();
                let scope = self.current_scope();
                redeclared = if as_var {
                    scope.lexical.contains(name)
                } else {
                    scope.lexical.contains(name) || scope.var.contains(name)
                };
                scope.functions.insert(name.to_owned());
            }
            _ => {
                for i in (0..self.scopes.len()).rev() {
                    let scope = &mut self.scopes[i];
                    let simple_catch_param = scope.flags & SCOPE_SIMPLE_CATCH != 0
                        && scope.first_lexical.as_deref() == Some(name);
                    if (scope.lexical.contains(name) && !simple_catch_param)
                        || (!Self::treat_functions_as_var_in_scope(scope)
                            && scope.functions.contains(name))
                    {
                        redeclared = true;
                        break;
                    }
                    scope.var.insert(name.to_owned());
                    if scope.flags & SCOPE_VAR != 0 {
                        break;
                    }
                }
            }
        }
        if redeclared {
            return self.raise(
                pos,
                format!("Identifier '{name}' has already been declared"),
            );
        }
        Ok(())
    }

    // ------------------------------------------------------------ identifiers

    fn check_unreserved(&self, name: &str, start: Pos) -> R<()> {
        if self.in_generator() && name == "yield" {
            return self.raise(start, "Cannot use 'yield' as identifier inside a generator");
        }
        if self.in_async() && name == "await" {
            return self.raise(
                start,
                "Cannot use 'await' as identifier inside an async function",
            );
        }
        // V8 (unlike acorn) also rejects `arguments` in an arrow function inside a
        // class static block.
        if self.this_scope_flags() & (SCOPE_CLASS_FIELD_INIT | SCOPE_CLASS_STATIC_BLOCK) != 0
            && name == "arguments"
        {
            return self.raise(
                start,
                "Cannot use 'arguments' in class field initializer or static block",
            );
        }
        if self.in_class_static_block() && (name == "arguments" || name == "await") {
            return self.raise(
                start,
                format!("Cannot use {name} in class static initialization block"),
            );
        }
        if name == "await" && self.var_scope_flags() & SCOPE_AWAIT_RESERVED != 0 {
            return self.raise(start, "Cannot use 'await' in a class static initializer");
        }
        if is_keyword(name) {
            return self.raise(start, format!("Unexpected keyword '{name}'"));
        }
        if name == "enum" || (self.strict && STRICT_RESERVED.contains(&name)) {
            return self.raise(start, format!("The keyword '{name}' is reserved"));
        }
        Ok(())
    }

    fn parse_ident(&mut self, liberal: bool) -> R<NodeId> {
        if self.tok.ty != Tk::Name {
            unexpected!(self);
        }
        let node = self.node_of(NK::Ident, self.tok.start);
        let name = self.tok.value.clone();
        self.nd_mut(node).name = name;
        self.next_ext(liberal)?;
        self.finish(node);
        if !liberal {
            let (name, start) = (self.nd(node).name.clone(), self.nd(node).start);
            self.check_unreserved(&name, start)?;
            if name == "await" && self.await_ident_pos < 0 {
                self.await_ident_pos = start;
            }
        }
        Ok(node)
    }

    fn parse_private_ident(&mut self) -> R<NodeId> {
        if self.tok.ty != Tk::PrivateId {
            unexpected!(self);
        }
        let node = self.node_of(NK::PrivateName, self.tok.start);
        let name = self.tok.value.clone();
        self.nd_mut(node).name = name.clone();
        self.next()?;
        self.finish(node);
        let start = self.nd(node).start;
        let Some(scope) = self.private_names.last_mut() else {
            return self.raise(
                start,
                format!("Private field '#{name}' must be declared in an enclosing class"),
            );
        };
        scope.used.push((name, start));
        Ok(node)
    }

    // ============================================================ statements

    fn is_let(&mut self, context: &Ctx) -> R<bool> {
        if !self.is_contextual("let") {
            return Ok(false);
        }
        let next = self.peek_token()?;
        // `let [` is excluded from ExpressionStatement; in any other
        // single-statement position `let` is an identifier.
        if next.ty == Tk::BracketL {
            return Ok(true);
        }
        if !context.is_none() {
            return Ok(false);
        }
        if next.ty == Tk::BraceL {
            return Ok(true);
        }
        // V8: an identifier, a contextual keyword or a strict-mode reserved word
        // makes a declaration; a keyword (or `enum`) leaves `let` an identifier.
        if next.ty == Tk::Name {
            return Ok(!is_keyword(&next.value) && next.value != "enum");
        }
        Ok(false)
    }

    fn is_async_function(&mut self) -> R<bool> {
        if !self.is_contextual("async") {
            return Ok(false);
        }
        let next = self.peek_token()?;
        Ok(next.ty == Tk::Name
            && next.value == "function"
            && !next.escaped
            && !self.has_line_break(self.tok.end, next.start))
    }

    /// Whether the current token starts a `using` (or `await using`) declaration.
    /// V8 recognises both words even when written with escapes.
    fn is_using_keyword(&mut self, is_await_using: bool, is_for: bool) -> R<bool> {
        if !self.is_kw(if is_await_using { "await" } else { "using" }) {
            return Ok(false);
        }
        let s = self.save();
        let result = self.is_using_keyword_scan(is_await_using, is_for);
        self.restore(s);
        match result {
            Ok(r) => Ok(r),
            Err(Fail::Syntax { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }

    fn is_using_keyword_scan(&mut self, is_await_using: bool, is_for: bool) -> R<bool> {
        let first_end = self.tok.end;
        self.last_tok_end = self.tok.end;
        self.last_tok_start = self.tok.start;
        self.next_token()?;
        if self.has_line_break(first_end, self.tok.start) {
            return Ok(false);
        }
        let mut ok = true;
        if is_await_using {
            ok = self.tok.ty == Tk::Name && self.tok.value == "using";
            if ok {
                let using_end = self.tok.end;
                self.last_tok_end = self.tok.end;
                self.last_tok_start = self.tok.start;
                self.next_token()?;
                ok = !self.has_line_break(using_end, self.tok.start);
            }
        }
        if ok && self.tok.ty == Tk::Name {
            if self.tok.escaped {
                return Ok(true);
            }
            let id = self.tok.value.clone();
            if id == "in" || id == "instanceof" {
                return Ok(false);
            }
            if is_for && !is_await_using && id == "of" {
                // V8: in a for head, `using of` declares `of` only when an
                // initializer follows; otherwise it is the head of a for-of loop
                // over `using`.
                self.last_tok_end = self.tok.end;
                self.last_tok_start = self.tok.start;
                self.next_token()?;
                return Ok(self.tok.ty == Tk::Eq);
            }
            return Ok(true);
        }
        Ok(false)
    }

    fn parse_statement(&mut self, context: &Ctx) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_statement_inner(context);
        self.leave();
        r
    }

    fn parse_statement_inner(&mut self, context: &Ctx) -> R<NodeId> {
        let start = self.tok.start;

        if self.is_let(context)? {
            if !context.is_none() {
                unexpected!(self);
            }
            return self.parse_var_statement("let");
        }

        if self.tok.ty == Tk::Name {
            let v = self.tok.value.clone();
            if is_keyword(&v) {
                match v.as_str() {
                    "break" | "continue" => {
                        return self.parse_break_continue_statement(v == "break");
                    }
                    "debugger" => {
                        self.next()?;
                        self.semicolon()?;
                        return Ok(NONE);
                    }
                    "do" => return self.parse_do_statement(),
                    "for" => return self.parse_for_statement(),
                    "function" => {
                        // A function is the sole body of an if statement or a
                        // labelled statement only in sloppy code (Annex B).
                        if !context.is_none()
                            && (self.strict
                                || !(matches!(context, Ctx::If) || matches!(context, Ctx::Label)))
                        {
                            unexpected!(self);
                        }
                        let declaration_position = context.is_none();
                        return self.parse_function_statement(
                            false,
                            declaration_position,
                            !matches!(context, Ctx::If),
                        );
                    }
                    "class" => {
                        if !context.is_none() {
                            unexpected!(self);
                        }
                        return self.parse_class(start, true);
                    }
                    "if" => return self.parse_if_statement(),
                    "return" => return self.parse_return_statement(),
                    "switch" => return self.parse_switch_statement(),
                    "throw" => return self.parse_throw_statement(),
                    "try" => return self.parse_try_statement(),
                    "const" | "var" => {
                        if !context.is_none() && v != "var" {
                            unexpected!(self);
                        }
                        return self.parse_var_statement(&v);
                    }
                    "while" => return self.parse_while_statement(),
                    "with" => return self.parse_with_statement(),
                    "export" => {
                        return self.raise(self.tok.start, "'export' may appear only in a module");
                    }
                    "import" => {
                        let nt = self.peek_token()?;
                        if nt.ty != Tk::ParenL && nt.ty != Tk::Dot {
                            return self
                                .raise(self.tok.start, "'import' may appear only in a module");
                        }
                        self.parse_expression(0, None)?;
                        self.semicolon()?;
                        return Ok(NONE);
                    }
                    _ => {}
                }
            }
        } else if self.tok.ty == Tk::BraceL {
            self.parse_block(true)?;
            return Ok(NONE);
        } else if self.tok.ty == Tk::Semi {
            self.next()?;
            return Ok(NONE);
        }

        if self.is_async_function()? {
            if !context.is_none() {
                unexpected!(self);
            }
            self.next()?;
            return self.parse_function_statement(true, context.is_none(), true);
        }

        // V8 does not recognise `using` declarations in single-statement
        // positions.
        let mut using_kind: Option<&str> = None;
        if context.is_none() {
            if self.is_using_keyword(true, false)? {
                using_kind = Some("await using");
            } else if self.is_using_keyword(false, false)? {
                using_kind = Some("using");
            }
        }
        if let Some(kind) = using_kind {
            if !self.allow_using() {
                return self.raise(self.tok.start, "Using declaration cannot appear here");
            }
            if kind == "await using" {
                if !self.can_await() {
                    return self.raise(
                        self.tok.start,
                        "Await using cannot appear outside of async function",
                    );
                }
                self.next()?;
            }
            self.next()?;
            self.parse_var(false, kind, false)?;
            self.semicolon()?;
            return Ok(NONE);
        }

        let starts_with_name = self.is_name_tok();
        let maybe_name = self.tok.value.clone();
        let expr = self.parse_expression(0, None)?;
        if starts_with_name && self.nd(expr).kind == NK::Ident && self.eat(Tk::Colon)? {
            return self.parse_labeled_statement(&maybe_name, expr, context, start);
        }
        self.semicolon()?;
        Ok(expr)
    }

    fn parse_break_continue_statement(&mut self, is_break: bool) -> R<NodeId> {
        let start = self.tok.start;
        self.next()?;
        let mut label: Option<String> = None;
        if self.eat(Tk::Semi)? || self.can_insert_semicolon() {
        } else if !self.is_name_tok() {
            unexpected!(self);
        } else {
            let id = self.parse_ident(false)?;
            label = Some(self.nd(id).name.clone());
            self.semicolon()?;
        }
        let mut found = false;
        for lab in &self.labels {
            if label.is_none() || lab.name == label {
                if lab.kind != LabelKind::None && (is_break || lab.kind == LabelKind::Loop) {
                    found = true;
                    break;
                }
                if label.is_some() && is_break {
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return self.raise(
                start,
                if is_break {
                    "Unsyntactic break"
                } else {
                    "Unsyntactic continue"
                },
            );
        }
        Ok(NONE)
    }

    fn parse_do_statement(&mut self) -> R<NodeId> {
        self.next()?;
        self.labels.push(Label {
            name: None,
            kind: LabelKind::Loop,
            statement_start: -1,
        });
        self.parse_statement(&Ctx::Do)?;
        self.labels.pop();
        if !self.is_kw("while") {
            unexpected!(self);
        }
        self.next()?;
        self.expect(Tk::ParenL)?;
        self.parse_expression(0, None)?;
        self.expect(Tk::ParenR)?;
        self.eat(Tk::Semi)?;
        Ok(NONE)
    }

    fn parse_for_statement(&mut self) -> R<NodeId> {
        self.next()?;
        let mut await_at: Pos = -1;
        // V8 takes `for await` even when `await` is written with escapes.
        if self.can_await() && self.is_kw("await") {
            await_at = self.tok.start;
            self.next()?;
        }
        self.labels.push(Label {
            name: None,
            kind: LabelKind::Loop,
            statement_start: -1,
        });
        self.enter_scope(0);
        self.expect(Tk::ParenL)?;
        if self.tok.ty == Tk::Semi {
            if await_at > -1 {
                return self.unexpected_at(await_at);
            }
            return self.parse_for();
        }
        let let_decl = self.is_let(&Ctx::None)?;
        if self.is_kw("var") || self.is_kw("const") || let_decl {
            let init_start = self.tok.start;
            let kind = if let_decl {
                "let".to_owned()
            } else {
                self.tok.value.clone()
            };
            self.next()?;
            let init = self.parse_var(true, &kind, false)?;
            return self.parse_for_after_init(init, &kind, init_start, await_at);
        }
        let starts_with_let = self.is_contextual("let");
        let mut using_kind: Option<&str> = None;
        if self.is_using_keyword(false, true)? {
            using_kind = Some("using");
        } else if self.is_using_keyword(true, true)? {
            using_kind = Some("await using");
        }
        if let Some(kind) = using_kind {
            let init_start = self.tok.start;
            self.next()?;
            if kind == "await using" {
                if !self.can_await() {
                    return self.raise(
                        self.tok.start,
                        "Await using cannot appear outside of async function",
                    );
                }
                self.next()?;
            }
            let init = self.parse_var(true, kind, false)?;
            return self.parse_for_after_init(init, kind, init_start, await_at);
        }
        let contains_esc = self.tok.escaped;
        let mut rf = DErr::default();
        let init_pos = self.tok.start;
        let init = if await_at > -1 {
            self.parse_expr_subscripts(Some(&mut rf), 2)?
        } else {
            self.parse_expression(1, Some(&mut rf))?
        };
        // V8 (unlike acorn) rejects any for-of head whose last token is an
        // unescaped `async` (`for (x.async of y)`), not only a bare `async`.
        let last_is_async =
            self.slice(self.last_tok_start, self.last_tok_end - self.last_tok_start) == "async";
        let is_in = self.is_kw("in");
        let is_for_of = !is_in && self.is_contextual("of");
        if is_in || is_for_of {
            if await_at > -1 {
                if is_in {
                    return self.unexpected_at(await_at);
                }
            } else if is_for_of {
                let n = self.nd(init);
                let bare_async = n.start == init_pos
                    && !contains_esc
                    && n.kind == NK::Ident
                    && n.name == "async";
                if bare_async || last_is_async {
                    unexpected!(self);
                }
            }
            if starts_with_let && is_for_of {
                return self.raise(
                    self.nd(init).start,
                    "The left-hand side of a for-of loop may not start with 'let'.",
                );
            }
            // V8 accepts a call expression here (and throws at run time), and
            // discards the expression errors recorded inside it; a member
            // expression is validated as an expression.
            if self.nd(init).kind == NK::Member {
                self.check_expression_errors(Some(&rf), true)?;
            }
            if self.nd(init).kind != NK::Call {
                self.to_assignable(init, false, Some(&mut rf))?;
                self.check_lval_pattern(init, Bind::None, None)?;
            }
            return self.parse_for_in(None, "", init_pos);
        }
        self.check_expression_errors(Some(&rf), true)?;
        if await_at > -1 {
            return self.unexpected_at(await_at);
        }
        self.parse_for()
    }

    fn parse_for_after_init(
        &mut self,
        init: VarResult,
        kind: &str,
        init_start: Pos,
        await_at: Pos,
    ) -> R<NodeId> {
        if (self.is_kw("in") || self.is_contextual("of")) && init.count == 1 {
            if self.is_kw("in") && await_at > -1 {
                return self.unexpected_at(await_at);
            }
            return self.parse_for_in(Some(init), kind, init_start);
        }
        if await_at > -1 {
            return self.unexpected_at(await_at);
        }
        self.parse_for()
    }

    fn parse_for(&mut self) -> R<NodeId> {
        self.expect(Tk::Semi)?;
        if self.tok.ty != Tk::Semi {
            self.parse_expression(0, None)?;
        }
        self.expect(Tk::Semi)?;
        if self.tok.ty != Tk::ParenR {
            self.parse_expression(0, None)?;
        }
        self.expect(Tk::ParenR)?;
        self.parse_statement(&Ctx::For)?;
        self.exit_scope();
        self.labels.pop();
        Ok(NONE)
    }

    fn parse_for_in(&mut self, decl: Option<VarResult>, kind: &str, init_start: Pos) -> R<NodeId> {
        let is_for_in = self.is_kw("in");
        // V8: a using declaration cannot head a for-in loop.
        if is_for_in && (kind == "using" || kind == "await using") {
            return self.raise(init_start, "Invalid 'using' in for-in loop");
        }
        self.next()?;
        if let Some(decl) = decl
            && decl.first_has_init
            && (!is_for_in || self.strict || kind != "var" || !decl.first_is_ident)
        {
            return self.raise(
                init_start,
                if is_for_in {
                    "for-in loop variable declaration may not have an initializer"
                } else {
                    "for-of loop variable declaration may not have an initializer"
                },
            );
        }
        if is_for_in {
            self.parse_expression(0, None)?;
        } else {
            self.parse_maybe_assign(0, None)?;
        }
        self.expect(Tk::ParenR)?;
        self.parse_statement(&Ctx::For)?;
        self.exit_scope();
        self.labels.pop();
        Ok(NONE)
    }

    fn parse_function_statement(
        &mut self,
        is_async: bool,
        declaration_position: bool,
        declare: bool,
    ) -> R<NodeId> {
        let start = self.tok.start;
        self.next()?;
        let mut flags = FUNC_STATEMENT
            | if declaration_position {
                0
            } else {
                FUNC_HANGING_STATEMENT
            };
        if !declare {
            flags |= FUNC_NO_DECLARE;
        }
        self.parse_function(start, flags, is_async, 0, false)
    }

    fn parse_if_statement(&mut self) -> R<NodeId> {
        self.next()?;
        self.expect(Tk::ParenL)?;
        self.parse_expression(0, None)?;
        self.expect(Tk::ParenR)?;
        self.parse_statement(&Ctx::If)?;
        if self.eat_kw("else")? {
            self.parse_statement(&Ctx::If)?;
        }
        Ok(NONE)
    }

    fn parse_return_statement(&mut self) -> R<NodeId> {
        if !self.in_function() {
            return self.raise(self.tok.start, "'return' outside of function");
        }
        self.next()?;
        if self.eat(Tk::Semi)? || self.can_insert_semicolon() {
            return Ok(NONE);
        }
        self.parse_expression(0, None)?;
        self.semicolon()?;
        Ok(NONE)
    }

    fn parse_switch_statement(&mut self) -> R<NodeId> {
        self.next()?;
        self.expect(Tk::ParenL)?;
        self.parse_expression(0, None)?;
        self.expect(Tk::ParenR)?;
        self.expect(Tk::BraceL)?;
        self.labels.push(Label {
            name: None,
            kind: LabelKind::Switch,
            statement_start: -1,
        });
        self.enter_scope(SCOPE_SWITCH);
        let mut have_case = false;
        let mut saw_default = false;
        while self.tok.ty != Tk::BraceR {
            if self.is_kw("case") || self.is_kw("default") {
                let is_case = self.is_kw("case");
                have_case = true;
                self.next()?;
                if is_case {
                    self.parse_expression(0, None)?;
                } else {
                    if saw_default {
                        return self.raise(self.last_tok_start, "Multiple default clauses");
                    }
                    saw_default = true;
                }
                self.expect(Tk::Colon)?;
            } else {
                if !have_case {
                    unexpected!(self);
                }
                self.parse_statement(&Ctx::None)?;
            }
        }
        self.exit_scope();
        self.next()?;
        self.labels.pop();
        Ok(NONE)
    }

    fn parse_throw_statement(&mut self) -> R<NodeId> {
        self.next()?;
        if self.has_line_break(self.last_tok_end, self.tok.start) {
            return self.raise(self.last_tok_end, "Illegal newline after throw");
        }
        self.parse_expression(0, None)?;
        self.semicolon()?;
        Ok(NONE)
    }

    fn parse_try_statement(&mut self) -> R<NodeId> {
        self.next()?;
        self.parse_block(true)?;
        let mut handler = false;
        if self.is_kw("catch") {
            self.next()?;
            if self.eat(Tk::ParenL)? {
                let param = self.parse_binding_atom()?;
                let simple = self.nd(param).kind == NK::Ident;
                self.enter_scope(if simple { SCOPE_SIMPLE_CATCH } else { 0 });
                self.check_lval_pattern(
                    param,
                    if simple {
                        Bind::SimpleCatch
                    } else {
                        Bind::Lexical
                    },
                    None,
                )?;
                self.expect(Tk::ParenR)?;
            } else {
                self.enter_scope(0);
            }
            self.parse_block(false)?;
            self.exit_scope();
            handler = true;
        }
        let finalizer = self.eat_kw("finally")?;
        if finalizer {
            self.parse_block(true)?;
        }
        if !handler && !finalizer {
            return self.raise(self.tok.start, "Missing catch or finally clause");
        }
        Ok(NONE)
    }

    fn parse_var_statement(&mut self, kind: &str) -> R<NodeId> {
        self.next()?;
        self.parse_var(false, kind, false)?;
        self.semicolon()?;
        Ok(NONE)
    }

    fn parse_while_statement(&mut self) -> R<NodeId> {
        self.next()?;
        self.expect(Tk::ParenL)?;
        self.parse_expression(0, None)?;
        self.expect(Tk::ParenR)?;
        self.labels.push(Label {
            name: None,
            kind: LabelKind::Loop,
            statement_start: -1,
        });
        self.parse_statement(&Ctx::While)?;
        self.labels.pop();
        Ok(NONE)
    }

    fn parse_with_statement(&mut self) -> R<NodeId> {
        if self.strict {
            return self.raise(self.tok.start, "'with' in strict mode");
        }
        self.next()?;
        self.expect(Tk::ParenL)?;
        self.parse_expression(0, None)?;
        self.expect(Tk::ParenR)?;
        self.parse_statement(&Ctx::With)?;
        Ok(NONE)
    }

    fn parse_labeled_statement(
        &mut self,
        maybe_name: &str,
        expr: NodeId,
        context: &Ctx,
        node_start: Pos,
    ) -> R<NodeId> {
        for label in &self.labels {
            if label.name.as_deref() == Some(maybe_name) {
                return self.raise(
                    self.nd(expr).start,
                    format!("Label '{maybe_name}' is already declared"),
                );
            }
        }
        let kind = if self.is_kw("do") || self.is_kw("for") || self.is_kw("while") {
            LabelKind::Loop
        } else if self.is_kw("switch") {
            LabelKind::Switch
        } else {
            LabelKind::None
        };
        let tok_start = self.tok.start;
        for label in self.labels.iter_mut().rev() {
            if label.statement_start == node_start {
                label.statement_start = tok_start;
                label.kind = kind;
            } else {
                break;
            }
        }
        self.labels.push(Label {
            name: Some(maybe_name.to_owned()),
            kind,
            statement_start: tok_start,
        });
        let inner = match context {
            Ctx::None => Ctx::Label,
            c if c.contains_label() => c.clone(),
            c => Ctx::LabelIn(Box::new(c.clone())),
        };
        self.parse_statement(&inner)?;
        self.labels.pop();
        Ok(NONE)
    }

    fn parse_block(&mut self, create_new_lexical_scope: bool) -> R<()> {
        self.expect(Tk::BraceL)?;
        if create_new_lexical_scope {
            self.enter_scope(0);
        }
        while self.tok.ty != Tk::BraceR {
            self.parse_statement(&Ctx::None)?;
        }
        self.next()?;
        if create_new_lexical_scope {
            self.exit_scope();
        }
        Ok(())
    }

    fn parse_var(
        &mut self,
        is_for: bool,
        kind: &str,
        allow_missing_initializer: bool,
    ) -> R<VarResult> {
        let mut result = VarResult::default();
        let is_using = kind == "using" || kind == "await using";
        loop {
            let id = if is_using {
                self.parse_ident(false)?
            } else {
                self.parse_binding_atom()?
            };
            self.check_lval_pattern(
                id,
                if kind == "var" {
                    Bind::Var
                } else {
                    Bind::Lexical
                },
                None,
            )?;
            let mut has_init = false;
            let in_or_of_for_head = |p: &Self| is_for && (p.is_kw("in") || p.is_contextual("of"));
            if self.eat(Tk::Eq)? {
                self.parse_maybe_assign(if is_for { 1 } else { 0 }, None)?;
                has_init = true;
            } else if !allow_missing_initializer && kind == "const" && !in_or_of_for_head(self) {
                // V8 (unlike acorn) exempts `in`/`of` only in a for head.
                unexpected!(self);
            } else if !allow_missing_initializer && is_using && !in_or_of_for_head(self) {
                return self.raise(
                    self.last_tok_end,
                    format!("Missing initializer in {kind} declaration"),
                );
            } else if !allow_missing_initializer
                && self.nd(id).kind != NK::Ident
                && !in_or_of_for_head(self)
            {
                return self.raise(
                    self.last_tok_end,
                    "Complex binding patterns require an initialization value",
                );
            }
            if result.count == 0 {
                result.first_has_init = has_init;
                result.first_is_ident = self.nd(id).kind == NK::Ident;
            }
            result.count += 1;
            if !self.eat(Tk::Comma)? {
                break;
            }
        }
        Ok(result)
    }

    /// A function declaration or expression (acorn `parseFunction`).
    ///
    /// V8 compiles a nested function lazily, pre-parsing it with its PreParser,
    /// unless the function literal follows `(` or `!` (`likely_called`, V8's
    /// "probably an IIFE" hint); everything inside a pre-parsed function is
    /// pre-parsed. The pre-parser accepts a few targets the full parser rejects
    /// (see [`Parser::preparser_target`]).
    fn parse_function(
        &mut self,
        start: Pos,
        statement: i32,
        is_async: bool,
        for_init: i32,
        likely_called: bool,
    ) -> R<NodeId> {
        let fn_node = self.node_of(NK::Function, start);
        if self.tok.ty == Tk::Star && statement & FUNC_HANGING_STATEMENT != 0 {
            unexpected!(self);
        }
        let generator = self.eat(Tk::Star)?;
        let mut id = NONE;
        if statement & FUNC_STATEMENT != 0 {
            id = if statement & FUNC_NULLABLE_ID != 0 && !self.is_name_tok() {
                NONE
            } else {
                self.parse_ident(false)?
            };
            if id >= 0 && statement & FUNC_NO_DECLARE == 0 {
                let bind = if self.strict || generator || is_async {
                    if self.treat_functions_as_var() {
                        Bind::Var
                    } else {
                        Bind::Lexical
                    }
                } else {
                    Bind::Function
                };
                self.check_lval_simple(id, bind, None, false)?;
            }
        }
        let (old_yield_pos, old_await_pos, old_await_ident_pos) =
            (self.yield_pos, self.await_pos, self.await_ident_pos);
        self.yield_pos = -1;
        self.await_pos = -1;
        self.await_ident_pos = -1;
        self.enter_scope(function_flags(is_async, generator));
        if statement & FUNC_STATEMENT == 0 {
            id = if self.is_name_tok() {
                self.parse_ident(false)?
            } else {
                NONE
            };
        }
        let old_preparse = self.preparse;
        self.preparse = old_preparse || !likely_called;
        let params = self.parse_function_params()?;
        self.parse_function_body(&params, false, false, for_init, id)?;
        self.preparse = old_preparse;
        self.yield_pos = old_yield_pos;
        self.await_pos = old_await_pos;
        self.await_ident_pos = old_await_ident_pos;
        self.nd_mut(fn_node).list = params;
        Ok(self.finish(fn_node))
    }

    fn parse_function_params(&mut self) -> R<Vec<NodeId>> {
        self.expect(Tk::ParenL)?;
        let params = self.parse_binding_list(Tk::ParenR, false, true)?;
        self.check_yield_await_in_default_params()?;
        Ok(params)
    }

    fn check_key_name(&self, element: NodeId, name: &str) -> bool {
        let e = self.nd(element);
        if e.computed {
            return false;
        }
        let key = self.nd(e.a);
        (key.kind == NK::Ident || key.kind == NK::StringLit) && key.name == name
    }

    fn is_class_element_name_start(&self) -> bool {
        matches!(
            self.tok.ty,
            Tk::Name | Tk::PrivateId | Tk::Num | Tk::String | Tk::BracketL
        )
    }

    fn parse_class(&mut self, start: Pos, is_statement: bool) -> R<NodeId> {
        let cls = self.node_of(NK::Class, start);
        self.next()?;
        let old_strict = self.strict;
        self.strict = true;
        if self.is_name_tok() {
            let id = self.parse_ident(false)?;
            if is_statement {
                self.check_lval_simple(id, Bind::Lexical, None, false)?;
            } else {
                // V8 (unlike acorn) also rejects a class expression named `eval`
                // or `arguments` (class code is strict).
                self.check_lval_simple(id, Bind::Outside, None, false)?;
            }
        } else if is_statement {
            unexpected!(self);
        }
        let mut has_super = false;
        if self.eat_kw("extends")? {
            self.parse_expr_subscripts(None, 0)?;
            has_super = true;
        }
        self.private_names.push(PrivateScope::default());
        let mut private_name_map: HashMap<String, String> = HashMap::new();
        let mut had_constructor = false;
        // V8 parses the class's static initializers (static fields and static
        // blocks) with the function kind of whichever initializer scope the class
        // body creates first: after an instance field, a static block allows
        // `await` as an identifier and `return`.
        let mut first_initializer_static: Option<bool> = None;
        self.expect(Tk::BraceL)?;
        while self.tok.ty != Tk::BraceR {
            self.parse_class_element(
                has_super,
                &mut private_name_map,
                &mut had_constructor,
                &mut first_initializer_static,
            )?;
        }
        self.strict = old_strict;
        self.next()?;
        self.exit_class_body()?;
        Ok(self.finish(cls))
    }

    fn parse_class_element(
        &mut self,
        constructor_allows_super: bool,
        private_name_map: &mut HashMap<String, String>,
        had_constructor: &mut bool,
        first_initializer_static: &mut Option<bool>,
    ) -> R<()> {
        self.enter()?;
        let r = self.parse_class_element_inner(
            constructor_allows_super,
            private_name_map,
            had_constructor,
            first_initializer_static,
        );
        self.leave();
        r
    }

    fn parse_class_element_inner(
        &mut self,
        constructor_allows_super: bool,
        private_name_map: &mut HashMap<String, String>,
        had_constructor: &mut bool,
        first_initializer_static: &mut Option<bool>,
    ) -> R<()> {
        if self.eat(Tk::Semi)? {
            return Ok(());
        }
        let element = self.node_of(NK::Property, self.tok.start);
        let mut key_name = String::new();
        let mut is_generator = false;
        let mut is_async = false;
        let mut kind = "method".to_owned();
        let mut is_static = false;

        if self.eat_contextual("static")? {
            if self.eat(Tk::BraceL)? {
                let first = *first_initializer_static.get_or_insert(true);
                return self.parse_class_static_block(!first);
            }
            if self.is_class_element_name_start() || self.tok.ty == Tk::Star {
                is_static = true;
            } else {
                key_name = "static".to_owned();
            }
        }
        if key_name.is_empty() && self.eat_contextual("async")? {
            if (self.is_class_element_name_start() || self.tok.ty == Tk::Star)
                && !self.can_insert_semicolon()
            {
                is_async = true;
            } else {
                key_name = "async".to_owned();
            }
        }
        if key_name.is_empty() && self.eat(Tk::Star)? {
            is_generator = true;
        }
        if key_name.is_empty() && !is_async && !is_generator {
            let last_value = self.tok.value.clone();
            if self.eat_contextual("get")? || self.eat_contextual("set")? {
                if self.is_class_element_name_start() {
                    kind = last_value;
                } else {
                    key_name = last_value;
                }
            }
        }

        if !key_name.is_empty() {
            self.nd_mut(element).computed = false;
            let key = self.node_of(NK::Ident, self.last_tok_start);
            let end = self.last_tok_end;
            self.nd_mut(key).name = key_name;
            self.nd_mut(key).end = end;
            self.nd_mut(element).a = key;
        } else if self.tok.ty == Tk::PrivateId {
            if self.tok.value == "constructor" {
                return self.raise(
                    self.tok.start,
                    "Classes can't have an element named '#constructor'",
                );
            }
            self.nd_mut(element).computed = false;
            let key = self.parse_private_ident()?;
            self.nd_mut(element).a = key;
        } else {
            self.parse_property_name(element)?;
        }

        let key = self.nd(element).a;
        let mut is_method = false;
        let mut method_kind = String::new();
        if self.tok.ty == Tk::ParenL || kind != "method" || is_generator || is_async {
            let is_constructor = !is_static && self.check_key_name(element, "constructor");
            let allows_direct_super = is_constructor && constructor_allows_super;
            if is_constructor && kind != "method" {
                return self.raise(
                    self.nd(key).start,
                    "Constructor can't have get/set modifier",
                );
            }
            method_kind = if is_constructor {
                "constructor".to_owned()
            } else {
                kind.clone()
            };
            if method_kind == "constructor" {
                if is_generator {
                    return self.raise(self.nd(key).start, "Constructor can't be a generator");
                }
                if is_async {
                    return self.raise(self.nd(key).start, "Constructor can't be an async method");
                }
            } else if is_static && self.check_key_name(element, "prototype") {
                return self.raise(
                    self.nd(key).start,
                    "Classes may not have a static property named prototype",
                );
            }
            let value = self.parse_method(is_generator, is_async, allows_direct_super)?;
            let params = self.nd(value).list.clone();
            if method_kind == "get" && !params.is_empty() {
                return self.raise(self.nd(value).start, "getter should have no params");
            }
            if method_kind == "set" && params.len() != 1 {
                return self.raise(self.nd(value).start, "setter should have exactly one param");
            }
            if method_kind == "set" && self.nd(params[0]).kind == NK::Rest {
                return self.raise(self.nd(params[0]).start, "Setter cannot use rest params");
            }
            is_method = true;
        } else {
            if self.check_key_name(element, "constructor") {
                return self.raise(
                    self.nd(key).start,
                    "Classes can't have a field named 'constructor'",
                );
            }
            if is_static && self.check_key_name(element, "prototype") {
                return self.raise(
                    self.nd(key).start,
                    "Classes can't have a static field named 'prototype'",
                );
            }
            let first = *first_initializer_static.get_or_insert(is_static);
            if self.eat(Tk::Eq)? {
                let await_reserved = if is_static && first {
                    SCOPE_AWAIT_RESERVED
                } else {
                    0
                };
                self.enter_scope(SCOPE_CLASS_FIELD_INIT | SCOPE_SUPER | await_reserved);
                self.parse_maybe_assign(0, None)?;
                self.exit_scope();
            }
            self.semicolon()?;
        }

        if is_method && method_kind == "constructor" {
            if *had_constructor {
                return self.raise(
                    self.nd(element).start,
                    "Duplicate constructor in the same class",
                );
            }
            *had_constructor = true;
        } else if self.nd(key).kind == NK::PrivateName {
            let name = self.nd(key).name.clone();
            let curr = private_name_map.get(&name).cloned().unwrap_or_default();
            let mut next_kind = "true".to_owned();
            if is_method && (method_kind == "get" || method_kind == "set") {
                next_kind = format!("{}{}", if is_static { "s" } else { "i" }, method_kind);
            }
            let pair = (curr.as_str(), next_kind.as_str());
            if matches!(
                pair,
                ("iget", "iset") | ("iset", "iget") | ("sget", "sset") | ("sset", "sget")
            ) {
                private_name_map.insert(name.clone(), "true".to_owned());
            } else if curr.is_empty() {
                private_name_map.insert(name.clone(), next_kind);
            } else {
                return self.raise(
                    self.nd(key).start,
                    format!("Identifier '#{name}' has already been declared"),
                );
            }
            self.private_names
                .last_mut()
                .expect("inside a class body")
                .declared
                .insert(name);
        }
        Ok(())
    }

    /// A static block; `instance_kind` when the class body created its instance
    /// initializer first (V8 then parses the block with that function kind:
    /// `await` is an identifier and `return` is allowed).
    fn parse_class_static_block(&mut self, instance_kind: bool) -> R<()> {
        let old_labels = std::mem::take(&mut self.labels);
        self.enter_scope(if instance_kind {
            SCOPE_CLASS_FIELD_INIT | SCOPE_SUPER | SCOPE_FUNCTION
        } else {
            SCOPE_CLASS_STATIC_BLOCK | SCOPE_SUPER
        });
        while self.tok.ty != Tk::BraceR {
            self.parse_statement(&Ctx::None)?;
        }
        self.next()?;
        self.exit_scope();
        self.labels = old_labels;
        Ok(())
    }

    fn exit_class_body(&mut self) -> R<()> {
        let scope = self.private_names.pop().expect("inside a class body");
        for used in scope.used {
            if !scope.declared.contains(&used.0) {
                if let Some(parent) = self.private_names.last_mut() {
                    parent.used.push(used);
                } else {
                    return self.raise(
                        used.1,
                        format!(
                            "Private field '#{}' must be declared in an enclosing class",
                            used.0
                        ),
                    );
                }
            }
        }
        Ok(())
    }

    // ============================================================ expressions

    fn parse_expression(&mut self, for_init: i32, mut rf: Option<&mut DErr>) -> R<NodeId> {
        let start = self.tok.start;
        let expr = self.parse_maybe_assign(for_init, rf.as_deref_mut())?;
        if self.tok.ty == Tk::Comma {
            let seq = self.node_of(NK::Sequence, start);
            self.nd_mut(seq).list.push(expr);
            while self.eat(Tk::Comma)? {
                let e = self.parse_maybe_assign(for_init, rf.as_deref_mut())?;
                self.nd_mut(seq).list.push(e);
            }
            return Ok(self.finish(seq));
        }
        Ok(expr)
    }

    fn parse_maybe_assign(&mut self, for_init: i32, rf: Option<&mut DErr>) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_maybe_assign_inner(for_init, rf);
        self.leave();
        r
    }

    fn parse_maybe_assign_inner(&mut self, for_init: i32, rf: Option<&mut DErr>) -> R<NodeId> {
        if self.is_contextual("yield") && self.in_generator() {
            return self.parse_yield(for_init);
        }

        let mut own = DErr::default();
        let own_destructuring_errors = rf.is_none();
        let (mut old_paren_assign, mut old_trailing_comma, mut old_double_proto) = (-1, -1, -1);
        let rf: &mut DErr = match rf {
            Some(r) => {
                old_paren_assign = r.parenthesized_assign;
                old_trailing_comma = r.trailing_comma;
                old_double_proto = r.double_proto;
                r.parenthesized_assign = -1;
                r.trailing_comma = -1;
                r
            }
            None => &mut own,
        };

        let start_pos = self.tok.start;
        if self.tok.ty == Tk::ParenL || self.is_name_tok() {
            self.potential_arrow_at = self.tok.start;
            self.potential_arrow_in_for_await = for_init == 2;
        }
        let mut left = self.parse_maybe_conditional(for_init, Some(rf))?;
        if self.tok.ty == Tk::Eq || self.tok.ty == Tk::Assign {
            let is_eq = self.tok.ty == Tk::Eq;
            let op = self.tok.value.clone();
            let node = self.node_of(NK::Assign, start_pos);
            self.nd_mut(node).name = op.clone();
            // V8 accepts a call expression as the target of `=` and of compound
            // assignment (not logical assignment) and throws at run time.
            let call_target =
                self.nd(left).kind == NK::Call && op != "&&=" && op != "||=" && op != "??=";
            if is_eq && !call_target {
                left = self.to_assignable(left, false, Some(rf))?;
            }
            // V8 (unlike acorn): a target that is not a destructuring pattern is
            // validated as an expression, so the expression errors recorded inside
            // it are final.
            if !matches!(self.nd(left).kind, NK::ObjectPattern | NK::ArrayPattern) {
                let left_start = self.nd(left).start;
                if rf.shorthand_assign >= left_start {
                    return self.raise(
                        rf.shorthand_assign,
                        "Shorthand property assignments are valid only in destructuring patterns",
                    );
                }
                if rf.double_proto >= left_start {
                    return self.raise(rf.double_proto, "Redefinition of __proto__ property");
                }
            }
            if !own_destructuring_errors {
                rf.parenthesized_assign = -1;
                rf.trailing_comma = -1;
                rf.double_proto = -1;
            }
            if rf.shorthand_assign >= self.nd(left).start {
                rf.shorthand_assign = -1;
            }
            if !call_target {
                if is_eq {
                    self.check_lval_pattern(left, Bind::None, None)?;
                } else {
                    self.check_lval_simple(left, Bind::None, None, false)?;
                }
            }
            self.nd_mut(node).a = left;
            self.next()?;
            let right = self.parse_maybe_assign(for_init, None)?;
            self.nd_mut(node).b = right;
            if old_double_proto > -1 {
                rf.double_proto = old_double_proto;
            }
            // The records of the elements before this one (`[(a = b), c = d] = e`,
            // `[...a, b = c] = d`) stand; acorn drops them here.
            if old_paren_assign > -1 {
                rf.parenthesized_assign = old_paren_assign;
            }
            if old_trailing_comma > -1 {
                rf.trailing_comma = old_trailing_comma;
            }
            return Ok(self.finish(node));
        }
        if own_destructuring_errors {
            self.check_expression_errors(Some(rf), true)?;
        }
        if old_paren_assign > -1 {
            rf.parenthesized_assign = old_paren_assign;
        }
        if old_trailing_comma > -1 {
            rf.trailing_comma = old_trailing_comma;
        }
        Ok(left)
    }

    fn parse_maybe_conditional(&mut self, for_init: i32, mut rf: Option<&mut DErr>) -> R<NodeId> {
        let start_pos = self.tok.start;
        let expr = self.parse_expr_ops(for_init, rf.as_deref_mut())?;
        if self.check_expression_errors(rf.as_deref(), false)? {
            return Ok(expr);
        }
        // V8 (unlike acorn) parses an arrow function as a whole
        // AssignmentExpression: it never heads a conditional expression...
        if self.nd(expr).start == start_pos && self.nd(expr).kind == NK::Arrow {
            return Ok(expr);
        }
        if self.eat(Tk::Question)? {
            let node = self.node_of(NK::Conditional, start_pos);
            self.nd_mut(node).a = expr;
            loop {
                self.parse_maybe_assign(0, None)?;
                self.expect(Tk::Colon)?;
                let alternate = self.parse_maybe_assign(for_init, None)?;
                self.nd_mut(node).b = alternate;
                // ...except in the alternate of a conditional: V8 parses a chain
                // `a ? b : c ? d : e` iteratively, taking whatever expression
                // precedes a `?` there as the next test, so an arrow function with
                // a block body may head the next link. A `yield` without operand
                // there takes `?` as the start of its operand, unless a line
                // break precedes it.
                if self.tok.ty != Tk::Question {
                    break;
                }
                let alt = self.nd(alternate);
                if alt.kind == NK::Yield
                    && alt.a < 0
                    && !self.has_line_break(self.last_tok_end, self.tok.start)
                {
                    unexpected!(self);
                }
                self.next()?;
            }
            return Ok(self.finish(node));
        }
        Ok(expr)
    }

    fn parse_expr_ops(&mut self, for_init: i32, mut rf: Option<&mut DErr>) -> R<NodeId> {
        let start_pos = self.tok.start;
        let expr = self.parse_maybe_unary(rf.as_deref_mut(), false, false, for_init)?;
        if self.check_expression_errors(rf.as_deref(), false)? {
            return Ok(expr);
        }
        if self.nd(expr).start == start_pos && self.nd(expr).kind == NK::Arrow {
            return Ok(expr);
        }
        self.parse_expr_op(expr, start_pos, -1, for_init)
    }

    fn binop_prec(&self) -> i32 {
        match self.tok.ty {
            Tk::LogicalOr | Tk::Coalesce => 1,
            Tk::LogicalAnd => 2,
            Tk::BitwiseOr => 3,
            Tk::BitwiseXor => 4,
            Tk::BitwiseAnd => 5,
            Tk::Equality => 6,
            Tk::Relational => 7,
            Tk::BitShift => 8,
            Tk::PlusMin => 9,
            Tk::Modulo | Tk::Star | Tk::Slash => 10,
            Tk::Name if self.tok.value == "in" || self.tok.value == "instanceof" => 7,
            _ => -1,
        }
    }

    /// Operator-precedence parsing (acorn parseExprOp), with the tail recursion
    /// over a left-associative chain written as a loop so that a long chain
    /// (`a + b + c + ...`) does not deepen the stack.
    fn parse_expr_op(
        &mut self,
        left: NodeId,
        left_start: Pos,
        min_prec: i32,
        for_init: i32,
    ) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_expr_op_inner(left, left_start, min_prec, for_init);
        self.leave();
        r
    }

    fn parse_expr_op_inner(
        &mut self,
        mut left: NodeId,
        left_start: Pos,
        min_prec: i32,
        for_init: i32,
    ) -> R<NodeId> {
        // V8 nests the nodes of a left-associative chain (`a < b < c`,
        // `a + b - c`) and compiles them recursively, except that a run of one
        // n-ary operator (`a + b + c`, `a || b || c`) becomes a single node: every
        // other link stays counted until the chain ends.
        let mut links = 0;
        let mut last_op = String::new();
        loop {
            let mut prec = self.binop_prec();
            if prec < 0 || (for_init != 0 && self.is_kw("in")) || prec <= min_prec {
                self.depth -= links;
                return Ok(left);
            }
            let logical = self.tok.ty == Tk::LogicalOr || self.tok.ty == Tk::LogicalAnd;
            let coalesce = self.tok.ty == Tk::Coalesce;
            if coalesce {
                // the precedence range of the logical operators
                prec = 2;
            }
            let op = self.tok.value.clone();
            if op != last_op || !is_nary_operator(&op) {
                self.enter()?;
                links += 1;
            }
            last_op.clone_from(&op);
            self.next()?;
            let start_pos = self.tok.start;
            let operand = self.parse_maybe_unary(None, false, false, for_init)?;
            let right = self.parse_expr_op(operand, start_pos, prec, for_init)?;
            left = self.build_binary(left_start, left, right, &op, logical || coalesce)?;
            if (logical && self.tok.ty == Tk::Coalesce)
                || (coalesce && (self.tok.ty == Tk::LogicalOr || self.tok.ty == Tk::LogicalAnd))
            {
                return self.raise(
                    self.tok.start,
                    "Logical expressions and coalesce expressions cannot be mixed. Wrap either by parentheses",
                );
            }
        }
    }

    fn build_binary(
        &mut self,
        start: Pos,
        left: NodeId,
        right: NodeId,
        op: &str,
        logical: bool,
    ) -> R<NodeId> {
        if self.nd(right).kind == NK::PrivateName {
            return self.raise(
                self.nd(right).start,
                "Private identifier can only be left side of binary expression",
            );
        }
        let node = self.node_of(if logical { NK::Logical } else { NK::Binary }, start);
        let n = self.nd_mut(node);
        n.a = left;
        n.b = right;
        n.name = op.to_owned();
        Ok(self.finish(node))
    }

    fn parse_maybe_unary(
        &mut self,
        rf: Option<&mut DErr>,
        saw_unary: bool,
        inc_dec: bool,
        for_init: i32,
    ) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_maybe_unary_inner(rf, saw_unary, inc_dec, for_init);
        self.leave();
        r
    }

    fn parse_maybe_unary_inner(
        &mut self,
        mut rf: Option<&mut DErr>,
        mut saw_unary: bool,
        inc_dec: bool,
        for_init: i32,
    ) -> R<NodeId> {
        let start_pos = self.tok.start;
        let expr;
        let is_prefix_keyword = self.tok.ty == Tk::Name
            && matches!(self.tok.value.as_str(), "typeof" | "void" | "delete");
        if self.is_contextual("await") && self.can_await() {
            expr = self.parse_await(for_init)?;
            saw_unary = true;
        } else if matches!(self.tok.ty, Tk::IncDec | Tk::Prefix | Tk::PlusMin) || is_prefix_keyword
        {
            let update = self.tok.ty == Tk::IncDec;
            let node = self.node_of(if update { NK::Update } else { NK::Unary }, start_pos);
            let op = self.tok.value.clone();
            self.nd_mut(node).name = op.clone();
            self.next()?;
            // V8: `!function` is probably an immediately called function.
            if op == "!" && self.is_kw("function") {
                self.next_function_likely_called = true;
            }
            let arg = self.parse_maybe_unary(None, true, update, for_init)?;
            self.nd_mut(node).a = arg;
            self.check_expression_errors(rf.as_deref(), true)?;
            if update {
                self.check_lval_simple(arg, Bind::None, None, true)?;
            } else if self.strict && op == "delete" && self.nd(arg).kind == NK::Ident {
                return self.raise(
                    self.nd(node).start,
                    "Deleting local variable in strict mode",
                );
            } else if op == "delete" && self.is_private_field_access(arg) {
                return self.raise(self.nd(node).start, "Private fields can not be deleted");
            } else {
                saw_unary = true;
            }
            expr = self.finish(node);
        } else if !saw_unary && self.tok.ty == Tk::PrivateId {
            if for_init != 0 || self.private_names.is_empty() {
                unexpected!(self);
            }
            expr = self.parse_private_ident()?;
            if !self.is_kw("in") {
                unexpected!(self);
            }
        } else {
            let mut e = self.parse_expr_subscripts(rf.as_deref_mut(), for_init)?;
            if self.check_expression_errors(rf.as_deref(), false)? {
                return Ok(e);
            }
            while self.tok.ty == Tk::IncDec && !self.can_insert_semicolon() {
                let node = self.node_of(NK::Update, start_pos);
                let op = self.tok.value.clone();
                self.nd_mut(node).name = op;
                self.nd_mut(node).a = e;
                self.check_lval_simple(e, Bind::None, None, true)?;
                self.next()?;
                e = self.finish(node);
            }
            expr = e;
        }

        // V8: an arrow function is never an operand of `**`.
        if self.nd(expr).start == start_pos && self.nd(expr).kind == NK::Arrow {
            return Ok(expr);
        }
        if !inc_dec && self.eat(Tk::StarStar)? {
            if saw_unary {
                return self.unexpected_at(self.last_tok_start);
            }
            let right = self.parse_maybe_unary(None, false, false, for_init)?;
            return self.build_binary(start_pos, expr, right, "**", false);
        }
        Ok(expr)
    }

    fn parse_expr_subscripts(&mut self, rf: Option<&mut DErr>, for_init: i32) -> R<NodeId> {
        let start_pos = self.tok.start;
        let mut rf = rf;
        let expr = self.parse_expr_atom(rf.as_deref_mut(), for_init, false)?;
        if self.nd(expr).kind == NK::Arrow
            && self.slice(self.last_tok_start, self.last_tok_end - self.last_tok_start) != ")"
        {
            return Ok(expr);
        }
        let result = self.parse_subscripts(expr, start_pos, false, for_init, rf.as_deref_mut())?;
        if let Some(r) = rf
            && self.nd(result).kind == NK::Member
        {
            let start = self.nd(result).start;
            if r.parenthesized_assign >= start {
                r.parenthesized_assign = -1;
            }
            if r.parenthesized_bind >= start {
                r.parenthesized_bind = -1;
            }
            if r.trailing_comma >= start {
                r.trailing_comma = -1;
            }
        }
        Ok(result)
    }

    fn parse_subscripts(
        &mut self,
        mut base: NodeId,
        start: Pos,
        no_calls: bool,
        for_init: i32,
        mut rf: Option<&mut DErr>,
    ) -> R<NodeId> {
        let b = self.nd(base);
        let maybe_async_arrow = b.kind == NK::Ident
            && b.name == "async"
            && self.last_tok_end == b.end
            && !self.can_insert_semicolon()
            && b.end - b.start == 5
            && self.potential_arrow_at == b.start;
        let mut optional_chained = false;
        let atom = base;
        // Every link of the chain stays counted until the chain ends: V8
        // compiles the nested member/call AST recursively.
        let mut links = 0;
        let result = loop {
            if let Err(e) = self.enter() {
                break Err(e);
            }
            links += 1;
            // V8 (unlike acorn): only a call directly on `async` can head an async
            // arrow function (`async (x) () => {}` is malformed).
            let element = match self.parse_subscript(
                base,
                start,
                no_calls,
                maybe_async_arrow && base == atom,
                optional_chained,
                for_init,
                rf.as_deref_mut(),
            ) {
                Ok(element) => element,
                Err(e) => break Err(e),
            };
            if self.nd(element).optional {
                optional_chained = true;
            }
            if element == base || self.nd(element).kind == NK::Arrow {
                if optional_chained {
                    let chain = self.node_of(NK::Chain, start);
                    self.nd_mut(chain).a = element;
                    break Ok(self.finish(chain));
                }
                break Ok(element);
            }
            base = element;
        };
        self.depth -= links;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_subscript(
        &mut self,
        base: NodeId,
        start: Pos,
        no_calls: bool,
        maybe_async_arrow: bool,
        optional_chained: bool,
        for_init: i32,
        outer: Option<&mut DErr>,
    ) -> R<NodeId> {
        let optional = self.eat(Tk::QuestionDot)?;
        if no_calls && optional {
            return self.raise(
                self.last_tok_start,
                "Optional chaining cannot appear in the callee of new expressions",
            );
        }
        let computed = self.eat(Tk::BracketL)?;
        if computed
            || (optional && self.tok.ty != Tk::ParenL && self.tok.ty != Tk::Template)
            || self.eat(Tk::Dot)?
        {
            let node = self.node_of(NK::Member, start);
            self.nd_mut(node).a = base;
            let prop = if computed {
                let p = self.parse_expression(0, None)?;
                self.expect(Tk::BracketR)?;
                p
            } else if self.tok.ty == Tk::PrivateId && self.nd(base).kind != NK::Super {
                self.parse_private_ident()?
            } else {
                self.parse_ident(true)?
            };
            let n = self.nd_mut(node);
            n.b = prop;
            n.computed = computed;
            n.optional = optional;
            return Ok(self.finish(node));
        }
        if !no_calls && self.eat(Tk::ParenL)? {
            let mut rf = DErr::default();
            let (old_yield_pos, old_await_pos, old_await_ident_pos) =
                (self.yield_pos, self.await_pos, self.await_ident_pos);
            self.yield_pos = -1;
            self.await_pos = -1;
            self.await_ident_pos = -1;
            let expr_list = self.parse_expr_list(Tk::ParenR, true, false, Some(&mut rf))?;
            if expr_list.len() > MAX_ARGUMENTS {
                return self.raise(
                    start,
                    "Too many arguments in function call (only 65525 allowed)",
                );
            }
            if maybe_async_arrow
                && !optional
                && !self.can_insert_semicolon()
                && self.eat(Tk::Arrow)?
            {
                self.check_pattern_errors(Some(&rf), false)?;
                // A rest element must be the last parameter (acorn loses the
                // trailing-comma record when an assignment follows the rest).
                if let Some(&rest) = expr_list
                    .iter()
                    .rev()
                    .skip(1)
                    .find(|&&e| e >= 0 && self.nd(e).kind == NK::Spread)
                {
                    return self.raise(
                        self.nd(rest).start,
                        "Rest parameter must be last formal parameter",
                    );
                }
                self.check_yield_await_in_default_params()?;
                if self.await_ident_pos > -1 {
                    return self.raise(
                        self.await_ident_pos,
                        "Cannot use 'await' as identifier inside an async function",
                    );
                }
                self.yield_pos = old_yield_pos;
                self.await_pos = old_await_pos;
                self.await_ident_pos = old_await_ident_pos;
                return self.parse_arrow_expression(start, expr_list, true, for_init);
            }
            // V8 (unlike acorn): the arguments' expression errors (shorthand
            // initializers, duplicate `__proto__`) belong to the enclosing
            // expression, which reports them unless it turns out to be the call
            // target of a for-in/of head; only the arguments of a possible async
            // arrow head (`async (...)`) are validated at once.
            match outer {
                Some(_) if maybe_async_arrow && !optional => {
                    self.check_expression_errors(Some(&rf), true)?;
                }
                Some(o) => {
                    if o.shorthand_assign < 0 {
                        o.shorthand_assign = rf.shorthand_assign;
                    }
                    if o.double_proto < 0 {
                        o.double_proto = rf.double_proto;
                    }
                }
                None => {
                    self.check_expression_errors(Some(&rf), true)?;
                }
            }
            if old_yield_pos > -1 {
                self.yield_pos = old_yield_pos;
            }
            if old_await_pos > -1 {
                self.await_pos = old_await_pos;
            }
            if old_await_ident_pos > -1 {
                self.await_ident_pos = old_await_ident_pos;
            }
            let node = self.node_of(NK::Call, start);
            let n = self.nd_mut(node);
            n.a = base;
            n.list = expr_list;
            n.optional = optional;
            return Ok(self.finish(node));
        }
        if self.tok.ty == Tk::Template {
            if optional || optional_chained {
                return self.raise(
                    self.tok.start,
                    "Optional chaining cannot appear in the tag of tagged template expressions",
                );
            }
            let node = self.node_of(NK::TaggedTemplate, start);
            self.nd_mut(node).a = base;
            self.parse_template(true)?;
            return Ok(self.finish(node));
        }
        Ok(base)
    }

    fn parse_expr_atom(
        &mut self,
        rf: Option<&mut DErr>,
        for_init: i32,
        for_new: bool,
    ) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_expr_atom_inner(rf, for_init, for_new);
        self.leave();
        r
    }

    fn parse_expr_atom_inner(
        &mut self,
        rf: Option<&mut DErr>,
        for_init: i32,
        for_new: bool,
    ) -> R<NodeId> {
        if self.tok.ty == Tk::Slash || (self.tok.ty == Tk::Assign && self.tok.value == "/=") {
            self.read_regexp()?;
        }
        let can_be_arrow = self.potential_arrow_at == self.tok.start;
        let start = self.tok.start;
        match self.tok.ty {
            Tk::Name => {
                let v = self.tok.value.clone();
                match v.as_str() {
                    "super" => {
                        if !self.allow_super() {
                            return self.raise(self.tok.start, "'super' keyword outside a method");
                        }
                        let node = self.node_of(NK::Super, start);
                        self.next()?;
                        if self.tok.ty == Tk::ParenL && !self.allow_direct_super() {
                            return self
                                .raise(start, "super() call outside constructor of a subclass");
                        }
                        if self.tok.ty != Tk::Dot
                            && self.tok.ty != Tk::BracketL
                            && self.tok.ty != Tk::ParenL
                        {
                            unexpected!(self);
                        }
                        return Ok(self.finish(node));
                    }
                    "this" => {
                        let node = self.node_of(NK::This, start);
                        self.next()?;
                        return Ok(self.finish(node));
                    }
                    "null" | "true" | "false" => {
                        let node = self.node_of(NK::Literal, start);
                        self.next()?;
                        return Ok(self.finish(node));
                    }
                    "function" => {
                        let likely_called = std::mem::take(&mut self.next_function_likely_called);
                        self.next()?;
                        return self.parse_function(start, 0, false, 0, likely_called);
                    }
                    "class" => return self.parse_class(start, false),
                    "new" => return self.parse_new(),
                    "import" => return self.parse_expr_import(for_new),
                    _ => {}
                }
                if is_keyword(&v) {
                    unexpected!(self);
                }
                let contains_esc = self.tok.escaped;
                let mut id = self.parse_ident(false)?;
                if !contains_esc
                    && self.nd(id).name == "async"
                    && !self.can_insert_semicolon()
                    && self.is_kw("function")
                {
                    let likely_called = std::mem::take(&mut self.next_function_likely_called);
                    self.next()?;
                    return self.parse_function(start, 0, true, for_init, likely_called);
                }
                if can_be_arrow && !self.can_insert_semicolon() {
                    if self.eat(Tk::Arrow)? {
                        return self.parse_arrow_expression(start, vec![id], false, for_init);
                    }
                    if self.nd(id).name == "async"
                        && self.is_name_tok()
                        && !contains_esc
                        && (!self.potential_arrow_in_for_await
                            || self.tok.value != "of"
                            || self.tok.escaped)
                    {
                        id = self.parse_ident(false)?;
                        if self.can_insert_semicolon() || !self.eat(Tk::Arrow)? {
                            unexpected!(self);
                        }
                        return self.parse_arrow_expression(start, vec![id], true, for_init);
                    }
                }
                Ok(id)
            }
            Tk::Regexp => {
                let node = self.node_of(NK::Regex, start);
                self.next()?;
                Ok(self.finish(node))
            }
            Tk::Num | Tk::String => self.parse_literal_token(),
            Tk::ParenL => {
                let expr = self.parse_paren_and_distinguish_expression(can_be_arrow, for_init)?;
                if let Some(r) = rf {
                    if r.parenthesized_assign < 0 && !self.is_simple_assign_target(expr) {
                        r.parenthesized_assign = start;
                    }
                    if r.parenthesized_bind < 0 {
                        r.parenthesized_bind = start;
                    }
                }
                Ok(expr)
            }
            Tk::BracketL => {
                let node = self.node_of(NK::Array, start);
                self.next()?;
                let list = self.parse_expr_list(Tk::BracketR, true, true, rf)?;
                self.nd_mut(node).list = list;
                Ok(self.finish(node))
            }
            Tk::BraceL => self.parse_obj(false, rf),
            Tk::Template => self.parse_template(false),
            _ => self.unexpected_at(self.tok.start),
        }
    }

    fn parse_expr_import(&mut self, for_new: bool) -> R<NodeId> {
        let start = self.tok.start;
        if self.tok.escaped {
            return self.raise(self.tok.start, "Escape sequence in keyword import");
        }
        self.next()?;
        if self.tok.ty == Tk::ParenL && !for_new {
            let node = self.node_of(NK::ImportCall, start);
            self.next()?;
            self.parse_maybe_assign(0, None)?;
            if !self.eat(Tk::ParenR)? {
                self.expect(Tk::Comma)?;
                if !self.after_trailing_comma(Tk::ParenR, false)? {
                    self.parse_maybe_assign(0, None)?;
                    if !self.eat(Tk::ParenR)? {
                        self.expect(Tk::Comma)?;
                        if !self.after_trailing_comma(Tk::ParenR, false)? {
                            unexpected!(self);
                        }
                    }
                }
            }
            return Ok(self.finish(node));
        }
        if self.tok.ty == Tk::Dot {
            self.next()?;
            let contains_esc = self.tok.escaped;
            let prop = self.parse_ident(true)?;
            if self.nd(prop).name == "source" && !contains_esc {
                // Source phase import: `import.source(specifier)`, one argument.
                if for_new {
                    return self.raise(start, "Cannot use new with import");
                }
                let node = self.node_of(NK::ImportCall, start);
                self.expect(Tk::ParenL)?;
                self.parse_maybe_assign(0, None)?;
                self.expect(Tk::ParenR)?;
                return Ok(self.finish(node));
            }
            if self.nd(prop).name != "meta" {
                return self.raise(
                    self.nd(prop).start,
                    "The only valid meta property for import is 'import.meta'",
                );
            }
            return self.raise(start, "Cannot use 'import.meta' outside a module");
        }
        self.unexpected_at(self.tok.start)
    }

    fn parse_literal_token(&mut self) -> R<NodeId> {
        let is_string = self.tok.ty == Tk::String;
        let node = self.node_of(
            if is_string {
                NK::StringLit
            } else {
                NK::Literal
            },
            self.tok.start,
        );
        if self.strict && self.tok.octal {
            return self.raise(
                self.tok.start,
                if is_string {
                    "Octal escape sequences are not allowed in strict mode"
                } else {
                    "Octal literals are not allowed in strict mode"
                },
            );
        }
        if is_string {
            let value = self.tok.value.clone();
            self.nd_mut(node).name = value;
        }
        self.next()?;
        Ok(self.finish(node))
    }

    fn parse_paren_and_distinguish_expression(
        &mut self,
        can_be_arrow: bool,
        for_init: i32,
    ) -> R<NodeId> {
        let start_pos = self.tok.start;
        self.next()?;
        // V8: a function literal right after `(` is probably called at once.
        if self.is_kw("function")
            || (self.is_contextual("async") && self.peek_token()?.value == "function")
        {
            self.next_function_likely_called = true;
        }
        let inner_start_pos = self.tok.start;
        let mut expr_list: Vec<NodeId> = Vec::new();
        let mut first = true;
        let mut last_is_comma = false;
        let mut rf = DErr::default();
        let (old_yield_pos, old_await_pos) = (self.yield_pos, self.await_pos);
        let mut spread_start: Pos = -1;
        self.yield_pos = -1;
        self.await_pos = -1;
        while self.tok.ty != Tk::ParenR {
            if first {
                first = false;
            } else {
                self.expect(Tk::Comma)?;
            }
            if self.after_trailing_comma(Tk::ParenR, true)? {
                last_is_comma = true;
                break;
            } else if self.tok.ty == Tk::Ellipsis {
                spread_start = self.tok.start;
                let rest = self.parse_rest_binding()?;
                expr_list.push(rest);
                if self.tok.ty == Tk::Comma {
                    return self.raise(
                        self.tok.start,
                        "Comma is not permitted after the rest element",
                    );
                }
                break;
            } else {
                let e = self.parse_maybe_assign(0, Some(&mut rf))?;
                expr_list.push(e);
            }
        }
        let inner_end_pos = self.last_tok_end;
        self.expect(Tk::ParenR)?;

        if can_be_arrow && !self.can_insert_semicolon() && self.eat(Tk::Arrow)? {
            self.check_pattern_errors(Some(&rf), false)?;
            self.check_yield_await_in_default_params()?;
            self.yield_pos = old_yield_pos;
            self.await_pos = old_await_pos;
            return self.parse_arrow_expression(start_pos, expr_list, false, for_init);
        }
        if expr_list.is_empty() || last_is_comma {
            return self.unexpected_at(self.last_tok_start);
        }
        if spread_start > -1 {
            return self.unexpected_at(spread_start);
        }
        self.check_expression_errors(Some(&rf), true)?;
        if old_yield_pos > -1 {
            self.yield_pos = old_yield_pos;
        }
        if old_await_pos > -1 {
            self.await_pos = old_await_pos;
        }
        if expr_list.len() > 1 {
            let seq = self.node_of(NK::Sequence, inner_start_pos);
            let n = self.nd_mut(seq);
            n.list = expr_list;
            n.end = inner_end_pos;
            return Ok(seq);
        }
        Ok(expr_list[0])
    }

    fn parse_new(&mut self) -> R<NodeId> {
        let start = self.tok.start;
        if self.tok.escaped {
            return self.raise(self.tok.start, "Escape sequence in keyword new");
        }
        self.next()?;
        if self.tok.ty == Tk::Dot {
            let node = self.node_of(NK::Meta, start);
            self.next()?;
            let contains_esc = self.tok.escaped;
            let prop = self.parse_ident(true)?;
            if self.nd(prop).name != "target" {
                return self.raise(
                    self.nd(prop).start,
                    "The only valid meta property for new is 'new.target'",
                );
            }
            if contains_esc {
                return self.raise(start, "'new.target' must not contain escaped characters");
            }
            if !self.allow_new_dot_target() {
                return self.raise(
                    start,
                    "'new.target' can only be used in functions and class static block",
                );
            }
            return Ok(self.finish(node));
        }
        let node = self.node_of(NK::New, start);
        let callee_start = self.tok.start;
        let atom = self.parse_expr_atom(None, 0, true)?;
        let callee = self.parse_subscripts(atom, callee_start, true, 0, None)?;
        self.nd_mut(node).a = callee;
        if self.eat(Tk::ParenL)? {
            let list = self.parse_expr_list(Tk::ParenR, true, false, None)?;
            if list.len() > MAX_ARGUMENTS {
                return self.raise(
                    start,
                    "Too many arguments in function call (only 65525 allowed)",
                );
            }
            self.nd_mut(node).list = list;
        }
        Ok(self.finish(node))
    }

    /// The current token is the first chunk (from the backtick).
    fn parse_template(&mut self, is_tagged: bool) -> R<NodeId> {
        let node = self.node_of(NK::Template, self.tok.start);
        loop {
            if self.tok.ty != Tk::Template {
                unexpected!(self);
            }
            if self.tok.bad_escape && !is_tagged {
                return self.raise(
                    self.tok.start,
                    "Bad escape sequence in untagged template literal",
                );
            }
            if self.tok.tail {
                self.next()?;
                return Ok(self.finish(node));
            }
            self.next()?;
            self.parse_expression(0, None)?;
            if self.tok.ty != Tk::BraceR {
                unexpected!(self);
            }
            let from = self.tok.start + 1;
            self.read_template_chunk(from)?;
        }
    }

    fn parse_obj(&mut self, is_pattern: bool, rf: Option<&mut DErr>) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_obj_inner(is_pattern, rf);
        self.leave();
        r
    }

    fn parse_obj_inner(&mut self, is_pattern: bool, mut rf: Option<&mut DErr>) -> R<NodeId> {
        let node = self.node_of(
            if is_pattern {
                NK::ObjectPattern
            } else {
                NK::Object
            },
            self.tok.start,
        );
        let mut first = true;
        let mut has_proto = false;
        self.next()?;
        while !self.eat(Tk::BraceR)? {
            if !first {
                self.expect(Tk::Comma)?;
                if self.after_trailing_comma(Tk::BraceR, false)? {
                    break;
                }
            } else {
                first = false;
            }
            let prop = self.parse_property(is_pattern, rf.as_deref_mut())?;
            if !is_pattern {
                self.check_prop_clash(prop, &mut has_proto, rf.as_deref_mut())?;
            }
            self.nd_mut(node).list.push(prop);
        }
        Ok(self.finish(node))
    }

    fn parse_property(&mut self, is_pattern: bool, mut rf: Option<&mut DErr>) -> R<NodeId> {
        let start = self.tok.start;
        if self.eat(Tk::Ellipsis)? {
            if is_pattern {
                let rest = self.node_of(NK::Rest, start);
                let arg = self.parse_ident(false)?;
                self.nd_mut(rest).a = arg;
                if self.tok.ty == Tk::Comma {
                    return self.raise(
                        self.tok.start,
                        "Comma is not permitted after the rest element",
                    );
                }
                return Ok(self.finish(rest));
            }
            let spread = self.node_of(NK::Spread, start);
            let arg = self.parse_maybe_assign(0, rf.as_deref_mut())?;
            self.nd_mut(spread).a = arg;
            self.validate_sub_pattern(arg, rf.as_deref())?;
            if self.tok.ty == Tk::Comma
                && let Some(r) = rf
                && r.trailing_comma < 0
            {
                r.trailing_comma = self.tok.start;
            }
            return Ok(self.finish(spread));
        }
        let prop = self.node_of(NK::Property, start);
        let mut is_generator = false;
        let mut is_async = false;
        let start_pos = self.tok.start;
        if !is_pattern {
            is_generator = self.eat(Tk::Star)?;
        }
        let contains_esc = self.tok.escaped;
        self.parse_property_name(prop)?;
        let key = self.nd(self.nd(prop).a).clone();
        if !is_pattern
            && !contains_esc
            && !is_generator
            && !self.nd(prop).computed
            && key.kind == NK::Ident
            && key.name == "async"
            && matches!(
                self.tok.ty,
                Tk::Name | Tk::Num | Tk::String | Tk::BracketL | Tk::Star
            )
            && !self.has_line_break(self.last_tok_end, self.tok.start)
        {
            is_async = true;
            is_generator = self.eat(Tk::Star)?;
            self.parse_property_name(prop)?;
        }
        self.parse_property_value(
            prop,
            is_pattern,
            is_generator,
            is_async,
            start_pos,
            rf,
            contains_esc,
        )?;
        Ok(self.finish(prop))
    }

    fn parse_property_name(&mut self, prop: NodeId) -> R<()> {
        if self.eat(Tk::BracketL)? {
            self.nd_mut(prop).computed = true;
            let key = self.parse_maybe_assign(0, None)?;
            self.nd_mut(prop).a = key;
            self.expect(Tk::BracketR)?;
            return Ok(());
        }
        self.nd_mut(prop).computed = false;
        let key = if self.tok.ty == Tk::Num || self.tok.ty == Tk::String {
            self.parse_literal_token()?
        } else {
            self.parse_ident(true)?
        };
        self.nd_mut(prop).a = key;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn parse_property_value(
        &mut self,
        prop: NodeId,
        is_pattern: bool,
        is_generator: bool,
        is_async: bool,
        start_pos: Pos,
        rf: Option<&mut DErr>,
        contains_esc: bool,
    ) -> R<()> {
        if (is_generator || is_async) && self.tok.ty == Tk::Colon {
            unexpected!(self);
        }
        let key = self.nd(prop).a;
        if self.eat(Tk::Colon)? {
            let value = if is_pattern {
                self.parse_maybe_default(self.tok.start, NONE)?
            } else {
                let mut rf = rf;
                let value = self.parse_maybe_assign(0, rf.as_deref_mut())?;
                self.validate_sub_pattern(value, rf.as_deref())?;
                value
            };
            let p = self.nd_mut(prop);
            p.b = value;
            p.name = "init".to_owned();
        } else if self.tok.ty == Tk::ParenL {
            if is_pattern {
                unexpected!(self);
            }
            self.nd_mut(prop).method = true;
            let value = self.parse_method(is_generator, is_async, false)?;
            let p = self.nd_mut(prop);
            p.b = value;
            p.name = "init".to_owned();
        } else if !is_pattern
            && !contains_esc
            && !self.nd(prop).computed
            && self.nd(key).kind == NK::Ident
            && (self.nd(key).name == "get" || self.nd(key).name == "set")
            && self.tok.ty != Tk::Comma
            && self.tok.ty != Tk::BraceR
            && self.tok.ty != Tk::Eq
        {
            if is_generator || is_async {
                unexpected!(self);
            }
            let kind = self.nd(key).name.clone();
            self.parse_property_name(prop)?;
            let value = self.parse_method(false, false, false)?;
            {
                let p = self.nd_mut(prop);
                p.b = value;
                p.name = kind.clone();
            }
            let params = self.nd(value).list.clone();
            let param_count = if kind == "get" { 0 } else { 1 };
            if params.len() != param_count {
                return self.raise(
                    self.nd(value).start,
                    if kind == "get" {
                        "getter should have no params"
                    } else {
                        "setter should have exactly one param"
                    },
                );
            }
            if kind == "set" && self.nd(params[0]).kind == NK::Rest {
                return self.raise(self.nd(params[0]).start, "Setter cannot use rest params");
            }
        } else if !self.nd(prop).computed && self.nd(key).kind == NK::Ident {
            if is_generator || is_async {
                unexpected!(self);
            }
            let (key_name, key_start, key_end) = {
                let k = self.nd(key);
                (k.name.clone(), k.start, k.end)
            };
            self.check_unreserved(&key_name, key_start)?;
            if key_name == "await" && self.await_ident_pos < 0 {
                self.await_ident_pos = start_pos;
            }
            let copy = self.node_of(NK::Ident, key_start);
            {
                let c = self.nd_mut(copy);
                c.name = key_name;
                c.end = key_end;
            }
            let value = if is_pattern {
                self.parse_maybe_default(start_pos, copy)?
            } else if self.tok.ty == Tk::Eq
                && let Some(r) = rf
            {
                if r.shorthand_assign < 0 {
                    r.shorthand_assign = self.tok.start;
                }
                self.parse_maybe_default(start_pos, copy)?
            } else {
                copy
            };
            let p = self.nd_mut(prop);
            p.b = value;
            p.name = "init".to_owned();
            p.shorthand = true;
        } else {
            unexpected!(self);
        }
        Ok(())
    }

    fn parse_method(
        &mut self,
        is_generator: bool,
        is_async: bool,
        allow_direct_super: bool,
    ) -> R<NodeId> {
        let fn_node = self.node_of(NK::Function, self.tok.start);
        let (old_yield_pos, old_await_pos, old_await_ident_pos) =
            (self.yield_pos, self.await_pos, self.await_ident_pos);
        self.yield_pos = -1;
        self.await_pos = -1;
        self.await_ident_pos = -1;
        self.enter_scope(
            function_flags(is_async, is_generator)
                | SCOPE_SUPER
                | if allow_direct_super {
                    SCOPE_DIRECT_SUPER
                } else {
                    0
                },
        );
        // V8 always compiles a method (and a getter, setter or constructor)
        // lazily.
        let old_preparse = std::mem::replace(&mut self.preparse, true);
        self.expect(Tk::ParenL)?;
        let params = self.parse_binding_list(Tk::ParenR, false, true)?;
        self.check_yield_await_in_default_params()?;
        self.parse_function_body(&params, false, true, 0, NONE)?;
        self.preparse = old_preparse;
        self.yield_pos = old_yield_pos;
        self.await_pos = old_await_pos;
        self.await_ident_pos = old_await_ident_pos;
        self.nd_mut(fn_node).list = params;
        Ok(self.finish(fn_node))
    }

    fn parse_arrow_expression(
        &mut self,
        start: Pos,
        params: Vec<NodeId>,
        is_async: bool,
        for_init: i32,
    ) -> R<NodeId> {
        let fn_node = self.node_of(NK::Arrow, start);
        let (old_yield_pos, old_await_pos, old_await_ident_pos) =
            (self.yield_pos, self.await_pos, self.await_ident_pos);
        self.enter_scope(function_flags(is_async, false) | SCOPE_ARROW);
        self.yield_pos = -1;
        self.await_pos = -1;
        self.await_ident_pos = -1;
        self.to_assignable_list(&params, true)?;
        self.parse_function_body(&params, true, false, for_init, NONE)?;
        self.yield_pos = old_yield_pos;
        self.await_pos = old_await_pos;
        self.await_ident_pos = old_await_ident_pos;
        self.nd_mut(fn_node).list = params;
        Ok(self.finish(fn_node))
    }

    fn is_simple_param_list(&self, params: &[NodeId]) -> bool {
        params.iter().all(|&p| self.nd(p).kind == NK::Ident)
    }

    fn check_params(&mut self, params: &[NodeId], allow_duplicates: bool) -> R<()> {
        let mut name_hash: HashSet<String> = HashSet::new();
        for &p in params {
            self.check_lval_inner_pattern(
                p,
                Bind::Var,
                if allow_duplicates {
                    None
                } else {
                    Some(&mut name_hash)
                },
            )?;
        }
        Ok(())
    }

    /// The function scope has been entered and the params are parsed. Parses the
    /// body (with its directive prologue), applies the checks that depend on the
    /// final strictness, and exits the scope.
    fn parse_function_body(
        &mut self,
        params: &[NodeId],
        is_arrow_function: bool,
        is_method: bool,
        for_init: i32,
        id_node: NodeId,
    ) -> R<()> {
        if params.len() > MAX_ARGUMENTS {
            return self.raise(
                self.last_tok_start,
                "Too many parameters in function definition (only 65534 allowed)",
            );
        }
        let is_expression = is_arrow_function && self.tok.ty != Tk::BraceL;
        let old_strict = self.strict;
        if is_expression {
            self.parse_maybe_assign(for_init, None)?;
            self.check_params(params, false)?;
            self.exit_scope();
            return Ok(());
        }
        let simple = self.is_simple_param_list(params);
        let old_labels = std::mem::take(&mut self.labels);
        // Declare the parameters before the body so that a body `let` clashing
        // with one is reported. Duplicates and strict-mode names are judged once
        // the directive prologue has settled the function's strictness.
        self.check_params(params, true)?;
        if self.strict && id_node >= 0 {
            self.check_lval_simple(id_node, Bind::Outside, None, false)?;
        }
        let body_start = self.tok.start;
        self.expect(Tk::BraceL)?;
        let use_strict = self.parse_statements_with_directives(Tk::BraceR)?;
        if use_strict && !simple {
            return self.raise(
                body_start,
                "Illegal 'use strict' directive in function with non-simple parameter list",
            );
        }
        let allow_duplicates =
            !old_strict && !use_strict && !is_arrow_function && !is_method && simple;
        let final_strict = old_strict || use_strict;
        let mut names: Vec<(String, Pos)> = Vec::new();
        for &p in params {
            self.collect_bound_names(p, &mut names)?;
        }
        let mut seen: HashSet<String> = HashSet::new();
        for (name, pos) in &names {
            if final_strict && is_strict_bind_reserved(name) {
                return self.raise(*pos, format!("Binding {name} in strict mode"));
            }
            if !allow_duplicates {
                if seen.contains(name) {
                    return self.raise(*pos, "Argument name clash");
                }
                seen.insert(name.clone());
            }
        }
        if final_strict && id_node >= 0 && is_strict_bind_reserved(&self.nd(id_node).name) {
            return self.raise(
                self.nd(id_node).start,
                format!("Binding {} in strict mode", self.nd(id_node).name),
            );
        }
        self.next()?; // the closing brace
        self.strict = old_strict;
        self.labels = old_labels;
        self.exit_scope();
        Ok(())
    }

    fn parse_expr_list(
        &mut self,
        close: Tk,
        allow_trailing_comma: bool,
        allow_empty: bool,
        mut rf: Option<&mut DErr>,
    ) -> R<Vec<NodeId>> {
        let mut elts = Vec::new();
        let mut first = true;
        while !self.eat(close)? {
            if !first {
                self.expect(Tk::Comma)?;
                if allow_trailing_comma && self.after_trailing_comma(close, false)? {
                    break;
                }
            } else {
                first = false;
            }
            let elt = if allow_empty && self.tok.ty == Tk::Comma {
                NONE
            } else if self.tok.ty == Tk::Ellipsis {
                let e = self.parse_spread(rf.as_deref_mut())?;
                if self.tok.ty == Tk::Comma
                    && let Some(r) = rf.as_deref_mut()
                    && r.trailing_comma < 0
                {
                    r.trailing_comma = self.tok.start;
                }
                e
            } else {
                self.parse_maybe_assign(0, rf.as_deref_mut())?
            };
            // Array literal elements (the only lists that allow holes) are
            // possible sub-patterns; call arguments are not (V8 lets their
            // errors accumulate in the enclosing expression).
            if allow_empty {
                self.validate_sub_pattern(elt, rf.as_deref())?;
            }
            elts.push(elt);
        }
        Ok(elts)
    }

    /// V8's `ParsePossibleDestructuringSubPattern`: an element of an array or
    /// object literal or a call argument that is a property reference (a member
    /// expression) is validated as an expression at once, so the expression
    /// errors recorded inside it are final.
    fn validate_sub_pattern(&self, element: NodeId, rf: Option<&DErr>) -> R<()> {
        let Some(r) = rf else { return Ok(()) };
        if element < 0 {
            return Ok(());
        }
        let target = if self.nd(element).kind == NK::Spread {
            self.nd(element).a
        } else {
            element
        };
        if target < 0 || self.nd(target).kind != NK::Member {
            return Ok(());
        }
        let start = self.nd(element).start;
        if r.shorthand_assign >= start {
            return self.raise(
                r.shorthand_assign,
                "Shorthand property assignments are valid only in destructuring patterns",
            );
        }
        if r.double_proto >= start {
            return self.raise(r.double_proto, "Redefinition of __proto__ property");
        }
        Ok(())
    }

    fn tok_starts_expr(&self) -> bool {
        match self.tok.ty {
            Tk::Name => {
                !is_keyword(&self.tok.value)
                    || KEYWORD_STARTS_EXPR.contains(&self.tok.value.as_str())
            }
            Tk::PrivateId
            | Tk::Num
            | Tk::String
            | Tk::Regexp
            | Tk::Template
            | Tk::BracketL
            | Tk::BraceL
            | Tk::ParenL
            | Tk::IncDec
            | Tk::Prefix
            | Tk::PlusMin
            | Tk::Slash => true,
            Tk::Assign => self.tok.value == "/=",
            _ => false,
        }
    }

    fn parse_yield(&mut self, for_init: i32) -> R<NodeId> {
        if self.yield_pos < 0 {
            self.yield_pos = self.tok.start;
        }
        let node = self.node_of(NK::Yield, self.tok.start);
        self.next()?;
        if self.tok.ty == Tk::Semi
            || self.can_insert_semicolon()
            || (self.tok.ty != Tk::Star && !self.tok_starts_expr())
        {
            return Ok(self.finish(node));
        }
        self.eat(Tk::Star)?;
        let arg = self.parse_maybe_assign(for_init, None)?;
        self.nd_mut(node).a = arg;
        Ok(self.finish(node))
    }

    fn parse_await(&mut self, for_init: i32) -> R<NodeId> {
        if self.await_pos < 0 {
            self.await_pos = self.tok.start;
        }
        let node = self.node_of(NK::Await, self.tok.start);
        self.next()?;
        let arg = self.parse_maybe_unary(None, true, false, for_init)?;
        self.nd_mut(node).a = arg;
        Ok(self.finish(node))
    }

    /// V8's pre-parser takes an optional chain that ends in a private member
    /// access (`a?.#x`) for a valid simple assignment target; the full parser
    /// rejects it.
    fn preparser_target(&self, node: NodeId) -> bool {
        self.preparse && self.nd(node).kind == NK::Chain && self.is_private_field_access(node)
    }

    fn is_private_field_access(&self, node: NodeId) -> bool {
        let nd = self.nd(node);
        match nd.kind {
            NK::Member => nd.b >= 0 && self.nd(nd.b).kind == NK::PrivateName,
            NK::Chain => self.is_private_field_access(nd.a),
            _ => false,
        }
    }

    fn is_simple_assign_target(&self, expr: NodeId) -> bool {
        matches!(self.nd(expr).kind, NK::Ident | NK::Member)
    }

    fn check_prop_clash(
        &mut self,
        prop: NodeId,
        has_proto: &mut bool,
        rf: Option<&mut DErr>,
    ) -> R<()> {
        let p = self.nd(prop);
        if p.kind == NK::Spread || p.computed || p.method || p.shorthand {
            return Ok(());
        }
        let key = self.nd(p.a);
        if !(key.kind == NK::Ident || key.kind == NK::StringLit) {
            return Ok(());
        }
        if key.name == "__proto__" && p.name == "init" {
            let key_start = key.start;
            if *has_proto {
                if let Some(r) = rf {
                    if r.double_proto < 0 {
                        r.double_proto = key_start;
                    }
                } else {
                    return self.raise(key_start, "Redefinition of __proto__ property");
                }
            }
            *has_proto = true;
        }
        Ok(())
    }

    fn check_pattern_errors(&self, rf: Option<&DErr>, is_assign: bool) -> R<()> {
        let Some(r) = rf else { return Ok(()) };
        if r.trailing_comma > -1 {
            return self.raise(
                r.trailing_comma,
                "Comma is not permitted after the rest element",
            );
        }
        let parens = if is_assign {
            r.parenthesized_assign
        } else {
            r.parenthesized_bind
        };
        if parens > -1 {
            return self.raise(
                parens,
                if is_assign {
                    "Assigning to rvalue"
                } else {
                    "Parenthesized pattern"
                },
            );
        }
        Ok(())
    }

    fn check_expression_errors(&self, rf: Option<&DErr>, and_throw: bool) -> R<bool> {
        let Some(r) = rf else { return Ok(false) };
        if !and_throw {
            return Ok(r.shorthand_assign >= 0 || r.double_proto >= 0);
        }
        if r.shorthand_assign >= 0 {
            return self.raise(
                r.shorthand_assign,
                "Shorthand property assignments are valid only in destructuring patterns",
            );
        }
        if r.double_proto >= 0 {
            return self.raise(r.double_proto, "Redefinition of __proto__ property");
        }
        Ok(false)
    }

    fn check_yield_await_in_default_params(&self) -> R<()> {
        if self.yield_pos > -1 && (self.await_pos < 0 || self.yield_pos < self.await_pos) {
            return self.raise(self.yield_pos, "Yield expression cannot be a default value");
        }
        if self.await_pos > -1 {
            return self.raise(self.await_pos, "Await expression cannot be a default value");
        }
        Ok(())
    }

    // ============================================================ lval

    // acorn's name, kept for cross-checking.
    #[allow(clippy::wrong_self_convention)]
    fn to_assignable(
        &mut self,
        node: NodeId,
        is_binding: bool,
        rf: Option<&mut DErr>,
    ) -> R<NodeId> {
        self.enter()?;
        let r = self.to_assignable_inner(node, is_binding, rf);
        self.leave();
        r
    }

    // acorn's name, kept for cross-checking.
    #[allow(clippy::wrong_self_convention)]
    fn to_assignable_inner(
        &mut self,
        node: NodeId,
        is_binding: bool,
        rf: Option<&mut DErr>,
    ) -> R<NodeId> {
        if node < 0 {
            if let Some(r) = rf {
                self.check_pattern_errors(Some(r), true)?;
            }
            return Ok(node);
        }
        let kind = self.nd(node).kind;
        match kind {
            NK::Ident => {
                if self.in_async() && self.nd(node).name == "await" {
                    return self.raise(
                        self.nd(node).start,
                        "Cannot use 'await' as identifier inside an async function",
                    );
                }
            }
            NK::ObjectPattern | NK::ArrayPattern | NK::AssignPattern | NK::Rest => {}
            NK::Object => {
                self.nd_mut(node).kind = NK::ObjectPattern;
                if let Some(r) = rf {
                    self.check_pattern_errors(Some(r), true)?;
                }
                let props = self.nd(node).list.clone();
                for prop in props {
                    self.to_assignable(prop, is_binding, None)?;
                    let p = self.nd(prop);
                    if p.kind == NK::Rest
                        && matches!(self.nd(p.a).kind, NK::ArrayPattern | NK::ObjectPattern)
                    {
                        return self.raise(self.nd(p.a).start, "Unexpected token");
                    }
                }
            }
            NK::Property => {
                let (name, method, a, b) = {
                    let p = self.nd(node);
                    (p.name.clone(), p.method, p.a, p.b)
                };
                if name != "init" {
                    return self.raise(
                        self.nd(a).start,
                        "Object pattern can't contain getter or setter",
                    );
                }
                if method {
                    return self.raise(self.nd(a).start, "Object pattern can't contain methods");
                }
                self.to_assignable(b, is_binding, None)?;
            }
            NK::Array => {
                self.nd_mut(node).kind = NK::ArrayPattern;
                if let Some(r) = rf {
                    self.check_pattern_errors(Some(r), true)?;
                }
                let elems = self.nd(node).list.clone();
                self.to_assignable_list(&elems, is_binding)?;
            }
            NK::Spread => {
                self.nd_mut(node).kind = NK::Rest;
                let arg = self.nd(node).a;
                self.to_assignable(arg, is_binding, None)?;
                if self.nd(arg).kind == NK::AssignPattern {
                    return self.raise(
                        self.nd(arg).start,
                        "Rest elements cannot have a default value",
                    );
                }
            }
            NK::Assign => {
                if self.nd(node).name != "=" {
                    let a = self.nd(node).a;
                    return self.raise(
                        self.nd(a).end,
                        "Only '=' operator can be used for specifying default value.",
                    );
                }
                self.nd_mut(node).kind = NK::AssignPattern;
                let a = self.nd(node).a;
                self.to_assignable(a, is_binding, None)?;
            }
            NK::Chain if self.preparser_target(node) => {
                if is_binding {
                    return self.raise(self.nd(node).start, "Assigning to rvalue");
                }
            }
            NK::Chain => {
                return self.raise(
                    self.nd(node).start,
                    "Optional chaining cannot appear in left-hand side",
                );
            }
            NK::Member => {
                if is_binding {
                    return self.raise(self.nd(node).start, "Assigning to rvalue");
                }
            }
            _ => return self.raise(self.nd(node).start, "Assigning to rvalue"),
        }
        Ok(node)
    }

    // acorn's name, kept for cross-checking.
    #[allow(clippy::wrong_self_convention)]
    fn to_assignable_list(&mut self, list: &[NodeId], is_binding: bool) -> R<()> {
        for &elt in list {
            if elt >= 0 {
                self.to_assignable(elt, is_binding, None)?;
            }
        }
        Ok(())
    }

    fn parse_spread(&mut self, rf: Option<&mut DErr>) -> R<NodeId> {
        let node = self.node_of(NK::Spread, self.tok.start);
        self.next()?;
        let arg = self.parse_maybe_assign(0, rf)?;
        self.nd_mut(node).a = arg;
        Ok(self.finish(node))
    }

    fn parse_rest_binding(&mut self) -> R<NodeId> {
        let node = self.node_of(NK::Rest, self.tok.start);
        self.next()?;
        let arg = self.parse_binding_atom()?;
        self.nd_mut(node).a = arg;
        Ok(self.finish(node))
    }

    fn parse_binding_atom(&mut self) -> R<NodeId> {
        self.enter()?;
        let r = self.parse_binding_atom_inner();
        self.leave();
        r
    }

    fn parse_binding_atom_inner(&mut self) -> R<NodeId> {
        if self.tok.ty == Tk::BracketL {
            let node = self.node_of(NK::ArrayPattern, self.tok.start);
            self.next()?;
            let list = self.parse_binding_list(Tk::BracketR, true, true)?;
            self.nd_mut(node).list = list;
            return Ok(self.finish(node));
        }
        if self.tok.ty == Tk::BraceL {
            return self.parse_obj(true, None);
        }
        self.parse_ident(false)
    }

    fn parse_binding_list(
        &mut self,
        close: Tk,
        allow_empty: bool,
        allow_trailing_comma: bool,
    ) -> R<Vec<NodeId>> {
        let mut elts = Vec::new();
        let mut first = true;
        while !self.eat(close)? {
            if first {
                first = false;
            } else {
                self.expect(Tk::Comma)?;
            }
            if allow_empty && self.tok.ty == Tk::Comma {
                elts.push(NONE);
            } else if allow_trailing_comma && self.after_trailing_comma(close, false)? {
                break;
            } else if self.tok.ty == Tk::Ellipsis {
                let rest = self.parse_rest_binding()?;
                elts.push(rest);
                if self.tok.ty == Tk::Comma {
                    return self.raise(
                        self.tok.start,
                        "Comma is not permitted after the rest element",
                    );
                }
                self.expect(close)?;
                break;
            } else {
                let start = self.tok.start;
                let e = self.parse_maybe_default(start, NONE)?;
                elts.push(e);
            }
        }
        Ok(elts)
    }

    fn parse_maybe_default(&mut self, start_pos: Pos, left: NodeId) -> R<NodeId> {
        let left = if left < 0 {
            self.parse_binding_atom()?
        } else {
            left
        };
        if !self.eat(Tk::Eq)? {
            return Ok(left);
        }
        let node = self.node_of(NK::AssignPattern, start_pos);
        self.nd_mut(node).a = left;
        let right = self.parse_maybe_assign(0, None)?;
        self.nd_mut(node).b = right;
        Ok(self.finish(node))
    }

    fn check_lval_simple(
        &mut self,
        expr: NodeId,
        binding_type: Bind,
        check_clashes: Option<&mut HashSet<String>>,
        allow_call: bool,
    ) -> R<()> {
        let is_bind = binding_type != Bind::None;
        let (kind, name, start) = {
            let nd = self.nd(expr);
            (nd.kind, nd.name.clone(), nd.start)
        };
        match kind {
            NK::Ident => {
                if self.strict && is_strict_bind_reserved(&name) {
                    return self.raise(
                        start,
                        format!(
                            "{} {name} in strict mode",
                            if is_bind { "Binding" } else { "Assigning to" }
                        ),
                    );
                }
                if is_bind {
                    if binding_type == Bind::Lexical && name == "let" {
                        return self.raise(start, "let is disallowed as a lexically bound name");
                    }
                    if let Some(clashes) = check_clashes {
                        if clashes.contains(&name) {
                            return self.raise(start, "Argument name clash");
                        }
                        clashes.insert(name.clone());
                    }
                    if binding_type != Bind::Outside {
                        self.declare_name(&name, binding_type, start)?;
                    }
                }
                Ok(())
            }
            NK::Chain if self.preparser_target(expr) => {
                if is_bind {
                    return self.raise(start, "Binding member expression");
                }
                Ok(())
            }
            NK::Chain => self.raise(start, "Optional chaining cannot appear in left-hand side"),
            NK::Member => {
                if is_bind {
                    return self.raise(start, "Binding member expression");
                }
                Ok(())
            }
            NK::Call => {
                // V8 accepts a call expression as a simple assignment target and
                // throws at run time.
                if allow_call && !is_bind {
                    return Ok(());
                }
                self.raise(start, "Assigning to rvalue")
            }
            _ => self.raise(
                start,
                if is_bind {
                    "Binding rvalue"
                } else {
                    "Assigning to rvalue"
                },
            ),
        }
    }

    fn check_lval_pattern(
        &mut self,
        expr: NodeId,
        binding_type: Bind,
        check_clashes: Option<&mut HashSet<String>>,
    ) -> R<()> {
        self.enter()?;
        let r = self.check_lval_pattern_inner(expr, binding_type, check_clashes);
        self.leave();
        r
    }

    fn check_lval_pattern_inner(
        &mut self,
        expr: NodeId,
        binding_type: Bind,
        mut check_clashes: Option<&mut HashSet<String>>,
    ) -> R<()> {
        let kind = self.nd(expr).kind;
        if kind == NK::ObjectPattern {
            let props = self.nd(expr).list.clone();
            for prop in props {
                self.check_lval_inner_pattern(prop, binding_type, check_clashes.as_deref_mut())?;
            }
            return Ok(());
        }
        if kind == NK::ArrayPattern {
            let elems = self.nd(expr).list.clone();
            for elem in elems {
                if elem >= 0 {
                    self.check_lval_inner_pattern(
                        elem,
                        binding_type,
                        check_clashes.as_deref_mut(),
                    )?;
                }
            }
            return Ok(());
        }
        self.check_lval_simple(expr, binding_type, check_clashes, false)
    }

    fn check_lval_inner_pattern(
        &mut self,
        expr: NodeId,
        binding_type: Bind,
        check_clashes: Option<&mut HashSet<String>>,
    ) -> R<()> {
        let (kind, a, b) = {
            let nd = self.nd(expr);
            (nd.kind, nd.a, nd.b)
        };
        match kind {
            NK::Property => self.check_lval_inner_pattern(b, binding_type, check_clashes),
            NK::AssignPattern | NK::Rest => self.check_lval_pattern(a, binding_type, check_clashes),
            _ => self.check_lval_pattern(expr, binding_type, check_clashes),
        }
    }

    fn collect_bound_names(&mut self, pattern: NodeId, out: &mut Vec<(String, Pos)>) -> R<()> {
        self.enter()?;
        let r = self.collect_bound_names_inner(pattern, out);
        self.leave();
        r
    }

    fn collect_bound_names_inner(
        &mut self,
        pattern: NodeId,
        out: &mut Vec<(String, Pos)>,
    ) -> R<()> {
        let (kind, a, b) = {
            let nd = self.nd(pattern);
            (nd.kind, nd.a, nd.b)
        };
        match kind {
            NK::Ident => {
                let nd = self.nd(pattern);
                out.push((nd.name.clone(), nd.start));
            }
            NK::ObjectPattern | NK::ArrayPattern => {
                let items = self.nd(pattern).list.clone();
                for item in items {
                    if item >= 0 {
                        self.collect_bound_names(item, out)?;
                    }
                }
            }
            NK::Property => self.collect_bound_names(b, out)?,
            NK::AssignPattern | NK::Rest => self.collect_bound_names(a, out)?,
            _ => {}
        }
        Ok(())
    }

    /// Parses statements until `terminator` (not consumed). A leading run of
    /// string-literal expression statements is the directive prologue; a
    /// "use strict" directive makes the rest strict. Returns whether it did.
    fn parse_statements_with_directives(&mut self, terminator: Tk) -> R<bool> {
        let mut use_strict = false;
        let mut in_prologue = true;
        let mut octal_in_prologue = false;
        while self.tok.ty != terminator {
            if self.tok.ty == Tk::Eof {
                unexpected!(self);
            }
            if in_prologue && self.tok.ty == Tk::String {
                let str_tok = self.tok.clone();
                let stmt = self.parse_statement(&Ctx::None)?;
                if stmt >= 0
                    && self.nd(stmt).kind == NK::StringLit
                    && self.nd(stmt).start == str_tok.start
                    && self.nd(stmt).end == str_tok.end
                {
                    if str_tok.octal {
                        octal_in_prologue = true;
                    }
                    if self.slice(str_tok.start + 1, str_tok.end - str_tok.start - 2)
                        == "use strict"
                    {
                        if octal_in_prologue {
                            return self
                                .raise(str_tok.start, "Octal escape sequence before 'use strict'");
                        }
                        use_strict = true;
                        self.strict = true;
                    }
                    continue;
                }
                in_prologue = false;
                continue;
            }
            in_prologue = false;
            self.parse_statement(&Ctx::None)?;
        }
        Ok(use_strict)
    }

    fn run(&mut self, param: &str) -> R<()> {
        self.enter_scope(SCOPE_FUNCTION);
        self.next_token()?;
        let id = self.node_of(NK::Ident, 0);
        self.nd_mut(id).name = param.to_owned();
        self.declare_name(param, Bind::Var, 0)?;
        let use_strict = self.parse_statements_with_directives(Tk::Eof)?;
        if use_strict && is_strict_bind_reserved(param) {
            return self.raise(0, format!("Binding {param} in strict mode"));
        }
        self.exit_scope();
        Ok(())
    }
}

/// The result of reading one escape sequence.
enum Escape {
    Ok { octal: bool },
    Bad,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid(src: &str) -> bool {
        matches!(
            check_function_body("state", &format!("with(state){{ return {src}; }}")),
            FuncSyntax::Valid
        )
    }

    #[test]
    fn expressions() {
        assert!(valid("time * 4"));
        assert!(valid("Math.sin(time) * 0.5 + 0.5"));
        assert!(!valid("time +"));
        assert!(!valid("-time ** 2"));
        assert!(valid("(-time) ** 2"));
        assert!(valid("f() = 1"));
        assert!(!valid("f() &&= 1"));
        assert!(!valid("import.defer(time)"));
        assert!(valid("import.source(time)"));
        assert!(!valid("\"(\" ; } let state = 1 ; { \")\""));
        assert!(valid("\"(\" ; } var state = 1 ; { \")\""));
        assert!(!valid("f`\\u{`x}`"));
        assert!(valid("f`\\u{`"));
    }

    fn verdict(src: &str) -> FuncSyntax {
        check_function_body("state", &format!("with(state){{ return {src}; }}"))
    }

    #[test]
    fn argument_and_parameter_limits() {
        let args = |n: usize| vec!["a"; n].join(",");
        let params = |n: usize| {
            (0..n)
                .map(|i| format!("a{i}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        assert!(valid(&format!("f({})", args(65525))));
        assert!(!valid(&format!("f({})", args(65526))));
        assert!(!valid(&format!("new f({})", args(65526))));
        assert!(!valid(&format!("f(...b,{})", args(65525))));
        assert!(valid(&format!("(function({}){{}})", params(65525))));
        assert!(!valid(&format!("(function({}){{}})", params(65526))));
        assert!(!valid(&format!("(({})=>1)", params(65526))));
        // A parenthesized sequence has no such limit.
        assert!(valid(&format!("({})", args(70000))));
    }

    #[test]
    fn nesting_beyond_the_limit_is_undecided() {
        let nested =
            |open: &str, n: usize, close: &str| format!("{}a{}", open.repeat(n), close.repeat(n));
        assert!(matches!(verdict(&nested("(", 50, ")")), FuncSyntax::Valid));
        assert!(matches!(
            verdict(&nested("(", 5000, ")")),
            FuncSyntax::Undecidable(_)
        ));
        // Chains that V8 nests count every link; a run of one n-ary operator
        // does not.
        assert!(matches!(
            verdict(&format!("a{}", ".b".repeat(5000))),
            FuncSyntax::Undecidable(_)
        ));
        assert!(matches!(
            verdict(&"a<".repeat(5000)),
            FuncSyntax::Undecidable(_)
        ));
        assert!(matches!(
            verdict(&"a+a-".repeat(2000)),
            FuncSyntax::Undecidable(_)
        ));
        assert!(matches!(
            verdict(&format!("{}a", "a+".repeat(100_000))),
            FuncSyntax::Valid
        ));
        assert!(matches!(
            verdict(&format!("{}a", "a||".repeat(100_000))),
            FuncSyntax::Valid
        ));
    }
}
