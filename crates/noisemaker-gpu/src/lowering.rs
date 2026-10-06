//! Shader lowering that mirrors the reference's shader compiler on Metal.
//!
//! The reference renders through Dawn, whose Tint compiler translates WGSL to
//! MSL; wgpu translates the same WGSL with naga. Where the two translations
//! differ in a way that changes rendered pixels, this module rewrites the WGSL
//! handed to wgpu so naga emits the code shape Tint emits. The reference's own
//! analyses (bindings, uniform layouts, entry points) keep reading the
//! unmodified source; every rewrite preserves the WGSL semantics exactly.
//! Evidence below was measured on an Apple M4 against goldens minted by the
//! reference on Chromium's WebGPU (Dawn + Tint) on the same machine.
//!
//! On Metal the backend compiles with Tint itself by default
//! ([`crate::backend::ShaderCompiler::Tint`]: Tint at Chromium's Dawn
//! revision generates the MSL, compiled with Dawn's options), which removes
//! every difference described here, including the ones below that no WGSL
//! rewrite can reach. This module applies only to the naga path
//! ([`crate::backend::ShaderCompiler::Naga`]: `NM_SHADER_COMPILER=naga`, the
//! opt-out kept for A/B comparisons).
//!
//! # Rules
//!
//! In the order [`lower_for_tint_msl`] applies them:
//!
//! * **Loop rotation** ([`rewrite_loops`]). Tint writes a loop as
//!   `while (true) { if (cond) {} else { break; } body { continuing } }`,
//!   the exit test inline. Metal's optimizer rotates that loop and fully
//!   unrolls it when the trip count is a constant and the unrolled body is
//!   small (an fbm-style body unrolls up to 15 iterations, not 16), then
//!   reassociates arithmetic across the unrolled iterations. naga writes a
//!   continuing block first, behind a `loop_init` gate, and bakes the test's
//!   load of the induction variable into a temporary declared before the
//!   `break` (`int _e11 = i; if (_e11 < 5) {} else { break; }`); with that
//!   declaration in the loop header Metal does not unroll, so every iteration
//!   rounds on its own. Hoisting the declaration out of the loop, or testing
//!   at the bottom, restores the unrolled arithmetic; the rule emits the
//!   guarded bottom-tested form (`init; if (cond) { loop { body update; if
//!   (cond) {} else { break; } } }`), which naga writes without a gate.
//!   Measured in a fragment-shader harness over 65536 random inputs, the
//!   fbm loop's results differed from Tint's in 78674 to 148776 of 262144
//!   components for trip counts 2 to 15 and in none after the rule; runtime
//!   trip counts and trip counts of 16 and 32 matched before and after.
//!   Corpus: 33 fixtures flip to exact (curl ×5, oil paint ×6, strokes ×4,
//!   hatch conte ×3, lighting ×3, distortion ×3 including two failing
//!   mixer_distortion cases, cel shading ×2, sacred geometry ×2, the oklab
//!   moodscape and noise cases, effects bloom, scatter aniso, stamp torn),
//!   the third failing mixer_distortion case drops from 4 levels to 1, and
//!   no fixture regresses.
//! * **Tint's loop counter** (part of [`rewrite_loops`]). Tint's
//!   `PreventInfiniteLoops` bounds a loop with a two-word counter (`if
//!   (all(idx == vec2(0))) { break; }` first in the body, the decrement first
//!   in the continuing block) unless its loop analysis proves the loop finite:
//!   an integer `var` declared by the initializer, stepped by exactly 1 in the
//!   continuing block (its only store), compared with a constant in the
//!   direction of the step. wgpu bounds every loop with naga's own counter at
//!   the top of the body. The rule writes Tint's counter for exactly the loops
//!   Tint bounds (the counters match Tint's MSL in all 149 dumped modules of
//!   the corpus) and the backend then compiles the module without naga's
//!   (`ShaderRuntimeChecks::force_loop_bounding`). Under Dawn's compile
//!   options naga's counter in craquelure's 3x3 Voronoi loops alone changes
//!   4764 pixels; under wgpu's options the rule flips no fixture (see below).
//! * **`pow(x, y)`** ([`lower_pow_to_powr`]). Tint's MSL writer emits
//!   `powr(x, y)`, naga's emits `pow(x, y)`. Under Metal's math mode the two
//!   compile to different instruction sequences, except for integral constant
//!   exponents, which the Metal compiler strength-reduces identically for
//!   both. `powr` compiles to `exp2(y * log2(x))`, so every other call site is
//!   rewritten to a helper computing exactly that (verified bit-identical to
//!   `powr` on this hardware).
//!
//! # Differences that are not WGSL rules
//!
//! With every stage's MSL replaced by Tint's and compiled the way Dawn
//! compiles it, each of the corpus's non-exact codegen fixtures renders
//! byte-identical (130 of 131; Chromium did not dump the 131st, the
//! classicNoisedeck fractal), so the remaining differences are naga's MSL or
//! how wgpu compiles it. Three mechanisms remain, none expressible as a WGSL
//! rewrite:
//!
//! * **Compile options.** Dawn compiles with `#pragma METAL fp
//!   math_mode(relaxed)` and sets `preserveInvariance` only for shaders with
//!   an `@invariant` output; wgpu-hal compiles with the default (fast) math
//!   mode and always sets `preserveInvariance`. Without `preserveInvariance`
//!   Metal's reassociation ranks the results of math-library calls (`sin`,
//!   `cos`, `tan`, `atan`, `log2`, `sqrt`, `inverseSqrt`, `abs`, `floor`,
//!   `ceil`, `round`, `trunc`, `min`, `max`, `clamp`, `saturate`, `mix`,
//!   `fma`, `length`, ...) as ordered instructions, the later call ranking
//!   higher; with it they rank by operand depth and ties keep source order.
//!   The product fused into an FMA and the association of sums follow these
//!   ranks: for `x * s + y * c` after `c = cos(t); s = sin(t)` Dawn's
//!   compile fuses `y * c` and wgpu's fuses `x * s`. WGSL has no opaque,
//!   ordered identity to restore Dawn's ranks under wgpu's options
//!   (`bitcast`, `insertBits`, `extractBits`, `reverseBits` twice,
//!   `min(x, x)`, `fma(x, 1, -0)` and `ldexp(x, 0)` all fold or rank by
//!   depth). 67 non-exact fixtures are exact with naga's own MSL once their
//!   culprit stage is compiled without `preserveInvariance` and/or in
//!   relaxed mode, among them the failing ones: craquelure ×3 (the Voronoi
//!   distances differ at 65527 of 65536 pixels), heightmap3d and
//!   renderLandscape3d ×4 (the isometric ray origin's z: wgpu computes
//!   `fma(fma(up.z, ty, right.z * tx), span, a)`, Dawn
//!   `fma(fma(right.z, tx, up.z * ty), span, a)`), filter_wormhole ×2 (the
//!   point-deposit vertex stage: exact once that stage alone drops
//!   `preserveInvariance`), classicNoisedeck_fractal's newton and
//!   synth_newton's spiralJunction3. The other eight failing newton cases
//!   also need Tint's MSL text: under wgpu's options Tint's own MSL still
//!   differs in 170 pixels of `newton`, naga's under Dawn's options in 176.
//! * **Function calls.** Tint passes module-scope variables to a helper as
//!   one `tint_module_vars` struct, naga as one parameter per variable, and
//!   Metal's inliner decides differently: in shapes3d naga's `getDist` stays
//!   a call where Tint's is inlined (forcing `always_inline` on naga's makes
//!   the fixture exact, forcing `noinline` on Tint's breaks it). Grouping
//!   the private globals into one struct does not change the decision.
//!   20 shapes3d fixtures, 1 level.
//! * **Load placement.** naga writes every load into a temporary declared
//!   before its statement; Tint loads inline, interleaved with the calls of
//!   the expression. Under wgpu's options only loads are ordered, so this is
//!   harmless; under Dawn's options it reorders the ranked calls, which is
//!   why compiling naga's MSL with Dawn's options instead scores far below
//!   wgpu's options (1403 against 2035 exact codegen fixtures on the graph
//!   path): matching the compile options needs Tint's MSL text as well.

use std::collections::BTreeSet;

use crate::reflect::ShaderReflection;

/// A source rewritten by [`lower_for_tint_msl`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lowered {
    /// The rewritten WGSL.
    pub source: String,
    /// The WGSL itself bounds exactly the loops Tint bounds, so the device
    /// must compile it without wgpu's own loop bounding
    /// (`ShaderRuntimeChecks::force_loop_bounding`).
    pub self_bounded: bool,
}

/// Apply every rule of this module to `source`. Returns `None` when no rule
/// changes the source or the rewritten source does not parse (the caller then
/// compiles the original with wgpu's default runtime checks).
pub fn lower_for_tint_msl(source: &str) -> Option<Lowered> {
    let (mut text, self_bounded, mut changed) = match rewrite_loops(source) {
        Some((rewritten, bounded)) => (rewritten, bounded, true),
        None => (source.to_owned(), false, false),
    };
    let reflection = ShaderReflection::parse(&text).ok()?;
    if let Some(rewritten) = lower_pow_to_powr(&text, &reflection) {
        text = rewritten;
        changed = true;
    }
    changed.then_some(Lowered {
        source: text,
        self_bounded,
    })
}

// ---------------------------------------------------------------------------
// WGSL tokens
// ---------------------------------------------------------------------------

/// A WGSL token: an identifier or keyword, a number, or one punctuation byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Ident,
    Number,
    Punct(u8),
}

#[derive(Debug, Clone, Copy)]
struct Token {
    kind: TokenKind,
    start: usize,
    end: usize,
}

/// Tokenize WGSL, skipping whitespace and (nested block or line) comments.
fn tokenize(source: &str) -> Vec<Token> {
    let bytes = source.as_bytes();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c.is_ascii_whitespace() {
            i += 1;
        } else if c == b'/' && bytes.get(i + 1) == Some(&b'/') {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
        } else if c == b'/' && bytes.get(i + 1) == Some(&b'*') {
            let mut depth = 0usize;
            while i < bytes.len() {
                if bytes[i] == b'/' && bytes.get(i + 1) == Some(&b'*') {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes.get(i + 1) == Some(&b'/') {
                    depth -= 1;
                    i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    i += 1;
                }
            }
        } else if c.is_ascii_alphabetic() || c == b'_' || c >= 0x80 {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] >= 0x80)
            {
                i += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Ident,
                start,
                end: i,
            });
        } else if c.is_ascii_digit()
            || (c == b'.' && bytes.get(i + 1).is_some_and(u8::is_ascii_digit))
        {
            let start = i;
            while i < bytes.len()
                && (bytes[i].is_ascii_alphanumeric()
                    || bytes[i] == b'.'
                    || ((bytes[i] == b'+' || bytes[i] == b'-')
                        && matches!(bytes[i - 1], b'e' | b'E' | b'p' | b'P')))
            {
                i += 1;
            }
            tokens.push(Token {
                kind: TokenKind::Number,
                start,
                end: i,
            });
        } else {
            tokens.push(Token {
                kind: TokenKind::Punct(c),
                start: i,
                end: i + 1,
            });
            i += 1;
        }
    }
    tokens
}

fn is_ident(source: &str, token: &Token, word: &str) -> bool {
    token.kind == TokenKind::Ident && &source[token.start..token.end] == word
}

fn is_punct(token: Option<&Token>, c: u8) -> bool {
    token.is_some_and(|t| t.kind == TokenKind::Punct(c))
}

/// The index of the token closing the bracket opened at `open` (`(`, `[` or `{`).
fn matching(tokens: &[Token], open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (i, token) in tokens.iter().enumerate().skip(open) {
        match token.kind {
            TokenKind::Punct(b'(' | b'[' | b'{') => depth += 1,
            TokenKind::Punct(b')' | b']' | b'}') => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Loops
// ---------------------------------------------------------------------------

/// What a `continue` statement inside a loop body becomes.
enum LoopContext {
    /// Outside any loop, or inside a loop whose `continue` needs no rewrite.
    Keep,
    /// `continue` first runs these statements (what the loop runs between
    /// iterations, which a rotated or self-bounded loop moves into its body).
    Tail(String),
}

/// The WGSL statements decrementing a loop counter the way Tint's
/// `PreventInfiniteLoops` transform does in the continuing block.
fn loop_counter_decrement(name: &str) -> String {
    format!(
        "{{ let {name}_low = {name}.x - 1u; {name}.x = {name}_low; \
         {name}.y = {name}.y - u32({name}_low == 4294967295u); }} "
    )
}

/// Rewrite every loop of `source` into the shape Tint gives it (see the
/// module documentation): `for` and `while` loops are rotated into guarded
/// bottom-tested loops,
///
/// ```text
/// for (init; cond; update) { body }
///   =>  { init; if (cond) { loop { body update; if (cond) {} else { break; } } } }
/// while (cond) { body }
///   =>  { if (cond) { loop { body if (cond) {} else { break; } } } }
/// ```
///
/// with each `continue` of the loop running the update and the test first,
/// and every loop that Tint's loop analysis cannot prove finite gets Tint's
/// loop counter (`if (all(idx == vec2(0))) { break; }` first in the body, the
/// two-word decrement before the update). The rewrite preserves the
/// evaluation order and count of every statement and condition.
///
/// Returns the rewritten source and whether every loop now carries the bound
/// Tint gives it (so the device must not add its own), or `None` when the
/// source has no loop.
pub fn rewrite_loops(source: &str) -> Option<(String, bool)> {
    let tokens = tokenize(source);
    let consts = tokens
        .windows(2)
        .filter(|w| is_ident(source, &w[0], "const") && w[1].kind == TokenKind::Ident)
        .map(|w| &source[w[1].start..w[1].end])
        .collect();
    let mut rewriter = LoopRewriter {
        source,
        tokens,
        consts,
        out: String::with_capacity(source.len() + 2048),
        pos: 0,
        changed: false,
        bounded: true,
        counters: 0,
    };
    let count = rewriter.tokens.len();
    rewriter.block(0, count, &LoopContext::Keep)?;
    if !rewriter.changed {
        return None;
    }
    rewriter.copy_to(source.len());
    Some((rewriter.out, rewriter.bounded))
}

struct LoopRewriter<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    /// Names declared with `const` (constant-expression operands).
    consts: BTreeSet<&'a str>,
    out: String,
    /// The source byte up to which `out` holds the rewritten text.
    pos: usize,
    changed: bool,
    /// `false` once a loop could not be given Tint's bound.
    bounded: bool,
    counters: usize,
}

/// The parts of a loop statement.
struct LoopParts<'a> {
    init: &'a str,
    cond: Option<&'a str>,
    update: &'a str,
    /// Token range of the header (between the keyword and the body).
    header: (usize, usize),
    body_open: usize,
    body_close: usize,
    /// Tint's loop analysis proves the loop finite.
    finite: bool,
}

impl<'a> LoopRewriter<'a> {
    fn copy_to(&mut self, byte: usize) {
        self.out.push_str(&self.source[self.pos..byte]);
        self.pos = byte;
    }

    /// The trimmed source text of the tokens `from..to`.
    fn text(&self, from: usize, to: usize) -> &'a str {
        if from >= to {
            return "";
        }
        self.source[self.tokens[from].start..self.tokens[to - 1].end].trim()
    }

    fn word(&self, i: usize) -> Option<&'a str> {
        let token = self.tokens.get(i)?;
        (token.kind == TokenKind::Ident).then(|| &self.source[token.start..token.end])
    }

    /// Rewrite the tokens `lo..hi`, retargeting `continue` per `context`.
    fn block(&mut self, lo: usize, hi: usize, context: &LoopContext) -> Option<()> {
        let mut i = lo;
        while i < hi {
            match self.word(i) {
                Some("for") if is_punct(self.tokens.get(i + 1), b'(') => {
                    let parts = self.for_parts(i)?;
                    i = self.loop_statement(i, parts)?;
                }
                Some("while") => {
                    let parts = self.while_parts(i)?;
                    i = self.loop_statement(i, parts)?;
                }
                Some("loop") if is_punct(self.tokens.get(i + 1), b'{') => {
                    let body_close = matching(&self.tokens, i + 1)?;
                    let parts = LoopParts {
                        init: "",
                        cond: None,
                        update: "",
                        header: (i + 1, i + 1),
                        body_open: i + 1,
                        body_close,
                        finite: false,
                    };
                    i = self.loop_statement(i, parts)?;
                }
                Some("continuing") => {
                    // A continuing block would need its own placement of the
                    // counter; keep naga's bound for such modules.
                    self.bounded = false;
                    i += 1;
                }
                Some("continue") => {
                    if let LoopContext::Tail(tail) = context {
                        if !is_punct(self.tokens.get(i + 1), b';') {
                            return None;
                        }
                        self.copy_to(self.tokens[i].start);
                        self.out.push_str(&format!("{{ {tail}continue; }}"));
                        self.pos = self.tokens[i + 1].end;
                        self.changed = true;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                _ => i += 1,
            }
        }
        Some(())
    }

    /// Rewrite the loop statement starting at token `at`, returning the index
    /// of the token after it.
    fn loop_statement(&mut self, at: usize, parts: LoopParts<'a>) -> Option<usize> {
        let keyword = self.word(at)?;
        let rotate = keyword != "loop" && !self.body_shadows(&parts);
        let counter = (!parts.finite).then(|| {
            self.counters += 1;
            format!("nm_loop_idx_{}", self.counters)
        });
        let decrement = counter
            .as_deref()
            .map(loop_counter_decrement)
            .unwrap_or_default();
        if !rotate && counter.is_none() {
            // Nothing moves: rewrite only inside the body.
            self.block(parts.body_open + 1, parts.body_close, &LoopContext::Keep)?;
            return Some(parts.body_close + 1);
        }
        self.copy_to(self.tokens[at].start);
        self.out.push_str("{ ");
        if let Some(counter) = &counter {
            self.out.push_str(&format!(
                "var {counter}: vec2<u32> = vec2<u32>(4294967295u); "
            ));
        }
        let check = counter
            .as_deref()
            .map(|c| format!(" if (all({c} == vec2<u32>(0u))) {{ break; }}"))
            .unwrap_or_default();
        let tail = if rotate {
            let mut tail = decrement.clone();
            if !parts.update.is_empty() {
                tail.push_str(parts.update);
                tail.push_str("; ");
            }
            if let Some(cond) = parts.cond {
                tail.push_str(&format!("if ({cond}) {{}} else {{ break; }} "));
            }
            if !parts.init.is_empty() {
                self.out.push_str(parts.init);
                self.out.push_str("; ");
            }
            if let Some(cond) = parts.cond {
                self.out.push_str(&format!("if ({cond}) {{ "));
            }
            self.out.push_str("loop {");
            tail
        } else {
            // Keep the header; the counter runs at the end of the body.
            let header_end = self.tokens[parts.body_open].start;
            self.out
                .push_str(&self.source[self.tokens[at].start..header_end]);
            self.out.push('{');
            decrement.clone()
        };
        self.out.push_str(&check);
        self.pos = self.tokens[parts.body_open].end;
        let context = LoopContext::Tail(tail.clone());
        self.block(parts.body_open + 1, parts.body_close, &context)?;
        self.copy_to(self.tokens[parts.body_close].start);
        self.out.push_str(&tail);
        self.out.push('}');
        if rotate && parts.cond.is_some() {
            self.out.push_str(" }");
        }
        self.out.push_str(" }");
        self.pos = self.tokens[parts.body_close].end;
        self.changed = true;
        Some(parts.body_close + 1)
    }

    fn for_parts(&self, at: usize) -> Option<LoopParts<'a>> {
        let open = at + 1;
        let close = matching(&self.tokens, open)?;
        let mut semis = Vec::new();
        let mut depth = 0i32;
        for (i, token) in self.tokens.iter().enumerate().take(close).skip(open + 1) {
            match token.kind {
                TokenKind::Punct(b'(' | b'[' | b'{') => depth += 1,
                TokenKind::Punct(b')' | b']' | b'}') => depth -= 1,
                TokenKind::Punct(b';') if depth == 0 => semis.push(i),
                _ => {}
            }
        }
        let [first, second] = semis[..] else {
            return None;
        };
        let body_open = close + 1;
        if !is_punct(self.tokens.get(body_open), b'{') {
            return None;
        }
        let body_close = matching(&self.tokens, body_open)?;
        let cond = self.text(first + 1, second);
        let mut parts = LoopParts {
            init: self.text(open + 1, first),
            cond: (!cond.is_empty()).then_some(cond),
            update: self.text(second + 1, close),
            header: (open + 1, close),
            body_open,
            body_close,
            finite: false,
        };
        parts.finite = self.finite_for(
            &parts,
            (open + 1, first),
            (first + 1, second),
            (second + 1, close),
        );
        Some(parts)
    }

    fn while_parts(&self, at: usize) -> Option<LoopParts<'a>> {
        // `while cond {`: the condition runs to the first top-level `{`.
        let mut body_open = at + 1;
        while !is_punct(self.tokens.get(body_open), b'{') {
            if matches!(
                self.tokens.get(body_open)?.kind,
                TokenKind::Punct(b'(' | b'[')
            ) {
                body_open = matching(&self.tokens, body_open)?;
            }
            body_open += 1;
        }
        let cond = self.text(at + 1, body_open);
        if cond.is_empty() {
            return None;
        }
        Some(LoopParts {
            init: "",
            cond: Some(cond),
            update: "",
            header: (at + 1, body_open),
            body_open,
            body_close: matching(&self.tokens, body_open)?,
            finite: false,
        })
    }

    /// Tint's `LoopAnalysis::IsFinite` for a `for` loop: an integer `var`
    /// declared by the initializer, stepped by exactly `+ 1` or `- 1` in the
    /// update (its only store; the body only loads it), and compared with a
    /// constant by the loop condition in the direction of the step. An
    /// inclusive bound must not be the extreme value the index never passes.
    fn finite_for(
        &self,
        parts: &LoopParts<'a>,
        init: (usize, usize),
        cond: (usize, usize),
        update: (usize, usize),
    ) -> bool {
        let t = &self.tokens;
        // init: `var NAME: i32 = ...`, `var NAME: u32 = ...` or `var NAME = INT`.
        if self.word(init.0) != Some("var") {
            return false;
        }
        let Some(name) = self.word(init.0 + 1) else {
            return false;
        };
        let mut k = init.0 + 2;
        let declared = match (is_punct(t.get(k), b':'), self.word(k + 1)) {
            (true, Some(ty @ ("i32" | "u32"))) => {
                k += 2;
                Some(ty)
            }
            (true, _) => return false,
            (false, _) => None,
        };
        if !is_punct(t.get(k), b'=') || k + 1 >= init.1 {
            return false;
        }
        let unsigned = match declared {
            Some(ty) => ty == "u32",
            None => {
                if self.int_constant(k + 1, init.1).is_none() {
                    return false;
                }
                self.text(k + 1, init.1).ends_with('u')
            }
        };
        // update: NAME++ / NAME-- / NAME += 1 / NAME -= 1 / NAME = NAME +- 1.
        let u: Vec<&str> = (update.0..update.1)
            .map(|i| &self.source[t[i].start..t[i].end])
            .collect();
        let step = match u[..] {
            [n, "+", "+"] if n == name => 1,
            [n, "-", "-"] if n == name => -1,
            [n, "+", "=", one] if n == name && is_one(one) => 1,
            [n, "-", "=", one] if n == name && is_one(one) => -1,
            [n, "=", m, "+", one] if n == name && m == name && is_one(one) => 1,
            [n, "=", one, "+", m] if n == name && m == name && is_one(one) => 1,
            [n, "=", m, "-", one] if n == name && m == name && is_one(one) => -1,
            _ => return false,
        };
        // cond: NAME op CONST or CONST op NAME, in the step's direction.
        let Some(op) =
            (cond.0..cond.1).find(|&i| matches!(t[i].kind, TokenKind::Punct(b'<' | b'>')))
        else {
            return false;
        };
        let inclusive = is_punct(t.get(op + 1), b'=') && t[op].end == t[op + 1].start;
        let rhs = op + if inclusive { 2 } else { 1 };
        let less = t[op].kind == TokenKind::Punct(b'<');
        let (index_left, constant) = if cond.0 + 1 == op && self.word(cond.0) == Some(name) {
            (true, (rhs, cond.1))
        } else if rhs + 1 == cond.1 && self.word(rhs) == Some(name) {
            (false, (cond.0, op))
        } else {
            return false;
        };
        if !self.constant_expression(constant.0, constant.1) {
            return false;
        }
        let upward = less == index_left;
        if (step == 1) != upward {
            return false;
        }
        if inclusive {
            // `i <= MAX` and `i >= MIN` never fail.
            let Some(bound) = self.int_constant(constant.0, constant.1) else {
                return false;
            };
            let extreme = if upward {
                bound >= i64::from(i32::MAX)
            } else {
                bound <= if unsigned { 0 } else { i64::from(i32::MIN) }
            };
            if extreme {
                return false;
            }
        }
        // The body only loads the index.
        !self.stores_to(name, parts.body_open, parts.body_close)
    }

    /// The value of an integer constant: an integer literal or a `const`
    /// name declared with one, optionally negated.
    fn int_constant(&self, lo: usize, hi: usize) -> Option<i64> {
        let t = &self.tokens;
        match hi.checked_sub(lo)? {
            2 if is_punct(t.get(lo), b'-') => self.int_constant(lo + 1, hi).map(|v| -v),
            1 => match t[lo].kind {
                TokenKind::Number => parse_int_literal(&self.source[t[lo].start..t[lo].end]),
                TokenKind::Ident => {
                    let name = &self.source[t[lo].start..t[lo].end];
                    // `const NAME (: i32 | : u32)? = VALUE;`
                    let at = (0..t.len().saturating_sub(1)).find(|&i| {
                        self.word(i) == Some("const") && self.word(i + 1) == Some(name)
                    })?;
                    let mut k = at + 2;
                    if is_punct(t.get(k), b':') {
                        if !matches!(self.word(k + 1), Some("i32" | "u32")) {
                            return None;
                        }
                        k += 2;
                    }
                    if !is_punct(t.get(k), b'=') {
                        return None;
                    }
                    let end = (k + 1..t.len()).find(|&i| is_punct(t.get(i), b';'))?;
                    if end > k + 3 {
                        return None;
                    }
                    self.int_constant(k + 1, end)
                }
                TokenKind::Punct(_) => None,
            },
            _ => None,
        }
    }

    /// `true` when the tokens `lo..hi` assign, increment, declare or take the
    /// address of `name`.
    fn stores_to(&self, name: &str, lo: usize, hi: usize) -> bool {
        let t = &self.tokens;
        let punct = |i: usize| match t.get(i).map(|t| t.kind) {
            Some(TokenKind::Punct(c)) if i < hi => Some(c),
            _ => None,
        };
        let adjacent = |i: usize| t[i].end == t[i + 1].start;
        (lo..hi).any(|i| {
            if self.word(i) != Some(name) {
                return false;
            }
            let declared = i > lo && matches!(self.word(i - 1), Some("var" | "let" | "const"));
            let address =
                i > lo && punct(i - 1) == Some(b'&') && !(i > lo + 1 && punct(i - 2) == Some(b'&'));
            let assigned = punct(i + 1) == Some(b'=') && punct(i + 2) != Some(b'=');
            let compound = matches!(
                punct(i + 1),
                Some(b'+' | b'-' | b'*' | b'/' | b'%' | b'&' | b'|' | b'^')
            ) && punct(i + 2) == Some(b'=')
                && adjacent(i + 1);
            let shift = matches!(punct(i + 1), Some(b'<' | b'>'))
                && punct(i + 2) == punct(i + 1)
                && punct(i + 3) == Some(b'=');
            let step = matches!(punct(i + 1), Some(b'+' | b'-'))
                && punct(i + 2) == punct(i + 1)
                && adjacent(i + 1);
            declared || address || assigned || compound || shift || step
        })
    }

    /// `true` when the tokens `lo..hi` form a constant expression (literals,
    /// `const` names, arithmetic, parentheses and scalar conversions).
    fn constant_expression(&self, lo: usize, hi: usize) -> bool {
        lo < hi
            && self.tokens[lo..hi].iter().all(|t| match t.kind {
                TokenKind::Number => true,
                TokenKind::Ident => {
                    let w = &self.source[t.start..t.end];
                    self.consts.contains(w) || matches!(w, "i32" | "u32")
                }
                TokenKind::Punct(c) => b"+-*/%()&|^~".contains(&c),
            })
    }

    /// `true` when the loop body declares a name the condition or update reads.
    fn body_shadows(&self, parts: &LoopParts<'a>) -> bool {
        let names: BTreeSet<&str> = (parts.header.0..parts.header.1)
            .filter(|&i| self.tokens[i].kind == TokenKind::Ident)
            .map(|i| &self.source[self.tokens[i].start..self.tokens[i].end])
            .collect();
        let body = &self.tokens[parts.body_open..parts.body_close];
        body.windows(2).any(|pair| {
            ["var", "let", "const"]
                .iter()
                .any(|w| is_ident(self.source, &pair[0], w))
                && pair[1].kind == TokenKind::Ident
                && names.contains(&self.source[pair[1].start..pair[1].end])
        })
    }
}

fn is_one(literal: &str) -> bool {
    matches!(literal, "1" | "1i" | "1u")
}

/// The value of a WGSL integer literal (decimal or hexadecimal, optional
/// `i`/`u` suffix).
fn parse_int_literal(literal: &str) -> Option<i64> {
    let digits = literal.strip_suffix(['i', 'u']).unwrap_or(literal);
    match digits
        .strip_prefix("0x")
        .or_else(|| digits.strip_prefix("0X"))
    {
        Some(hex) => i64::from_str_radix(hex, 16).ok(),
        None => digits.parse().ok(),
    }
}

// ---------------------------------------------------------------------------
// pow
// ---------------------------------------------------------------------------

/// Rewrite the non-integral-exponent `pow` calls of `source` (already parsed
/// into `reflection`) into `powr`-equivalent helpers. Returns `None` when the
/// source has no such call.
pub fn lower_pow_to_powr(source: &str, reflection: &ShaderReflection) -> Option<String> {
    let sites = reflection.pow_sites(source);
    if sites.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(source.len() + 512);
    let mut last = 0usize;
    let mut used = BTreeSet::new();
    for (start, suffix) in sites {
        if start < last {
            continue;
        }
        out.push_str(&source[last..start]);
        out.push_str("nm_rt_powr_");
        out.push_str(suffix);
        last = start + "pow".len();
        used.insert(suffix);
    }
    out.push_str(&source[last..]);
    for suffix in used {
        let ty = match suffix {
            "f32" => "f32",
            "vec2f" => "vec2<f32>",
            "vec3f" => "vec3<f32>",
            _ => "vec4<f32>",
        };
        out.push_str(&format!(
            "\nfn nm_rt_powr_{suffix}(x: {ty}, y: {ty}) -> {ty} {{ return exp2(y * log2(x)); }}\n"
        ));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewrites_non_integral_exponents_only() {
        let src = "@fragment fn main(@location(0) v: vec2<f32>) -> @location(0) vec4<f32> {\n\
                   let a = pow(v.x, 2.4);\n\
                   let b = pow(v, vec2<f32>(3.0));\n\
                   let c = pow (v, v);\n\
                   return vec4<f32>(a, b.x, c.y, pow(v.y, 5.0));\n}";
        let reflection = ShaderReflection::parse(src).unwrap();
        let lowered = lower_pow_to_powr(src, &reflection).unwrap();
        assert!(lowered.contains("let a = nm_rt_powr_f32(v.x, 2.4);"));
        assert!(lowered.contains("let b = pow(v, vec2<f32>(3.0));"));
        assert!(lowered.contains("let c = nm_rt_powr_vec2f (v, v);"));
        assert!(lowered.contains("pow(v.y, 5.0)"));
        assert!(lowered.contains("fn nm_rt_powr_vec2f(x: vec2<f32>, y: vec2<f32>) -> vec2<f32>"));
        ShaderReflection::parse(&lowered).unwrap();
    }

    #[test]
    fn no_rewrite_without_dynamic_pow() {
        let src =
            "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(pow(2.0, 3.0)); }";
        let reflection = ShaderReflection::parse(src).unwrap();
        assert!(lower_pow_to_powr(src, &reflection).is_none());
    }

    fn flat(s: &str) -> String {
        s.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn finite_loops_rotate_into_guarded_bottom_tested_loops() {
        let src = "const N: i32 = 3;\n\
                   fn f(n: i32) -> i32 {\n\
                   var s = 0;\n\
                   for (var i: i32 = 0; i < 9; i = i + 1) {\n\
                     if (i == 2) { continue; }\n\
                     for (var j = -1; j <= N; j++) { if (j == 1) { continue; } s += j; }\n\
                     s += i;\n\
                   }\n\
                   return s;\n}";
        let (out, bounded) = rewrite_loops(src).unwrap();
        assert!(bounded);
        let out = flat(&out);
        assert!(!out.contains("nm_loop_idx"), "{out}");
        assert!(
            out.contains("{ var i: i32 = 0; if (i < 9) { loop {"),
            "{out}"
        );
        assert!(
            out.contains("if (i == 2) { { i = i + 1; if (i < 9) {} else { break; } continue; } }"),
            "{out}"
        );
        assert!(out.contains("{ var j = -1; if (j <= N) { loop { if (j == 1) { { j++; if (j <= N) {} else { break; } continue; } } s += j; j++; if (j <= N) {} else { break; } } } }"), "{out}");
        assert!(
            out.contains("s += i; i = i + 1; if (i < 9) {} else { break; } } } }"),
            "{out}"
        );
        ShaderReflection::parse(&out).unwrap();
    }

    #[test]
    fn unprovable_loops_get_tints_counter() {
        let src = "fn f(n: i32) -> i32 {\n\
                   var s = 0;\n\
                   let m = 4;\n\
                   for (var i = 0; i < m; i++) { s += i; }\n\
                   for (var i = 0; i < 4; i++) { i += 1; }\n\
                   for (var i = 9; i < 4; i--) { s -= 1; }\n\
                   for (var t = 0.0; t < 4.0; t += 1.0) { s += 1; }\n\
                   for (var i: i32 = 0; i <= 2147483647; i++) { s += 1; if (s > 8) { break; } }\n\
                   for (var i = 3u; i >= 0u; i--) { s += 1; if (s > 9) { break; } }\n\
                   while (s > 100) { s -= 7; if (s == 50) { continue; } }\n\
                   loop { s += 1; if (s > 9) { break; } continue; }\n\
                   return s;\n}";
        let (out, bounded) = rewrite_loops(src).unwrap();
        assert!(bounded);
        let out = flat(&out);
        for k in 1..=8 {
            assert!(
                out.contains(&format!(
                    "var nm_loop_idx_{k}: vec2<u32> = vec2<u32>(4294967295u);"
                )),
                "{k}: {out}"
            );
        }
        assert!(out.contains("{ var nm_loop_idx_1: vec2<u32> = vec2<u32>(4294967295u); var i = 0; if (i < m) { loop { if (all(nm_loop_idx_1 == vec2<u32>(0u))) { break; } s += i; { let nm_loop_idx_1_low = nm_loop_idx_1.x - 1u; nm_loop_idx_1.x = nm_loop_idx_1_low; nm_loop_idx_1.y = nm_loop_idx_1.y - u32(nm_loop_idx_1_low == 4294967295u); } i++; if (i < m) {} else { break; } } } }"), "{out}");
        assert!(
            out.contains("if (s == 50) { { { let nm_loop_idx_7_low"),
            "{out}"
        );
        assert!(
            out.contains("if ((s > 100)) {} else { break; } continue; }"),
            "{out}"
        );
        assert!(out.contains("loop { if (all(nm_loop_idx_8 == vec2<u32>(0u))) { break; } s += 1; if (s > 9) { break; } { { let nm_loop_idx_8_low"), "{out}");
        ShaderReflection::parse(&out).unwrap();
    }

    #[test]
    fn shadowed_loops_keep_their_header() {
        let src = "fn f() -> i32 { var s = 0; for (var i = 0; i < 4; i++) { let i = 7; s += i; } return s; }";
        let (out, _) = rewrite_loops(src).unwrap();
        let out = flat(&out);
        // `let i` makes the index not provably finite and keeps the header.
        assert!(out.contains("{ var nm_loop_idx_1: vec2<u32> = vec2<u32>(4294967295u); for (var i = 0; i < 4; i++) { if (all(nm_loop_idx_1 == vec2<u32>(0u))) { break; } let i = 7; s += i; { let nm_loop_idx_1_low"), "{out}");
        ShaderReflection::parse(&out).unwrap();
        assert!(rewrite_loops("fn f() -> i32 { return 1; }").is_none());
    }
}
