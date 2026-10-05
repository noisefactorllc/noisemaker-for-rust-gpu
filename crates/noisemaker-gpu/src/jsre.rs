//! JavaScript regular expressions on the `regex` crate.
//!
//! The reference backend decides uniform packing, bindings and entry points with
//! JavaScript regexes. JavaScript's `\w`, `\d` and `\b` are ASCII-only, `\s` is the
//! ECMAScript whitespace set and `.` stops at all four line terminators, while the
//! `regex` crate defaults to Unicode classes (and the reference WGSL sources contain
//! letters such as `π` in comments). [`JsRegex::new`] translates a JavaScript
//! pattern into an equivalent Rust pattern so every match, capture and count is the
//! one the reference sees. The `i` flag is expanded to explicit ASCII case pairs,
//! which is exact for JavaScript's non-Unicode case-insensitive mode (it never folds
//! a non-ASCII character onto an ASCII one).
//!
//! Matching uses leftmost-first semantics with backtracking-equivalent captures, the
//! same results a JavaScript `exec` loop produces for patterns that cannot match the
//! empty string (all patterns of the reference backend).

use std::cell::RefCell;
use std::collections::HashMap;

use regex::{Captures, Regex};

/// The ECMAScript `WhiteSpace` and `LineTerminator` code points (`\s`).
const JS_WS: &str = r"\t\n\x0B\x0C\r \x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";
const JS_WORD: &str = "0-9A-Za-z_";

/// A compiled JavaScript regular expression.
#[derive(Clone, Debug)]
pub struct JsRegex {
    re: Regex,
}

fn push_literal(out: &mut String, c: char) {
    let mut buf = [0u8; 4];
    out.push_str(&regex::escape(c.encode_utf8(&mut buf)));
}

fn push_case_pair(out: &mut String, c: char, in_class: bool) {
    let lower = c.to_ascii_lowercase();
    let upper = c.to_ascii_uppercase();
    if in_class {
        out.push(lower);
        out.push(upper);
    } else {
        out.push('[');
        out.push(lower);
        out.push(upper);
        out.push(']');
    }
}

/// `true` when `chars[i..]` starts a JavaScript quantifier `{n}`, `{n,}` or `{n,m}`.
fn is_quantifier(chars: &[char], i: usize) -> bool {
    let mut j = i + 1;
    let start = j;
    while j < chars.len() && chars[j].is_ascii_digit() {
        j += 1;
    }
    if j == start {
        return false;
    }
    if j < chars.len() && chars[j] == '}' {
        return true;
    }
    if j < chars.len() && chars[j] == ',' {
        j += 1;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        return j < chars.len() && chars[j] == '}';
    }
    false
}

/// Translate a JavaScript (non-Unicode-mode) pattern into an equivalent `regex`
/// crate pattern.
pub fn translate(pattern: &str, ignore_case: bool) -> String {
    let chars: Vec<char> = pattern.chars().collect();
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut in_class = false;
    let mut class_start = false;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if in_class {
            match c {
                ']' if !class_start => {
                    out.push(']');
                    in_class = false;
                }
                '\\' => {
                    let e = *chars.get(i + 1).expect("dangling escape in pattern");
                    i += 1;
                    match e {
                        'w' => out.push_str(JS_WORD),
                        'd' => out.push_str("0-9"),
                        's' => out.push_str(JS_WS),
                        'n' => out.push_str(r"\n"),
                        't' => out.push_str(r"\t"),
                        'r' => out.push_str(r"\r"),
                        'f' => out.push_str(r"\x0C"),
                        'v' => out.push_str(r"\x0B"),
                        'b' => out.push_str(r"\x08"),
                        'W' | 'D' | 'S' => {
                            panic!("negated class escape \\{e} inside a class is not supported")
                        }
                        other => push_literal(&mut out, other),
                    }
                }
                '[' | '&' | '~' => {
                    out.push('\\');
                    out.push(c);
                }
                c if ignore_case && c.is_ascii_alphabetic() => {
                    // A letter range `a-z` keeps its range and gains the other case.
                    if chars.get(i + 1) == Some(&'-')
                        && chars.get(i + 2).is_some_and(|e| e.is_ascii_alphabetic())
                    {
                        let end = chars[i + 2];
                        out.push(c);
                        out.push('-');
                        out.push(end);
                        let swap = |x: char| {
                            if x.is_ascii_lowercase() {
                                x.to_ascii_uppercase()
                            } else {
                                x.to_ascii_lowercase()
                            }
                        };
                        if c.is_ascii_lowercase() == end.is_ascii_lowercase() {
                            out.push(swap(c));
                            out.push('-');
                            out.push(swap(end));
                        }
                        i += 2;
                    } else {
                        push_case_pair(&mut out, c, true);
                    }
                }
                other => out.push(other),
            }
            class_start = false;
            i += 1;
            continue;
        }
        match c {
            '\\' => {
                let e = *chars.get(i + 1).expect("dangling escape in pattern");
                i += 1;
                match e {
                    'w' => out.push_str(&format!("[{JS_WORD}]")),
                    'W' => out.push_str(&format!("[^{JS_WORD}]")),
                    'd' => out.push_str("[0-9]"),
                    'D' => out.push_str("[^0-9]"),
                    's' => out.push_str(&format!("[{JS_WS}]")),
                    'S' => out.push_str(&format!("[^{JS_WS}]")),
                    'b' => out.push_str(r"(?-u:\b)"),
                    'B' => out.push_str(r"(?-u:\B)"),
                    'n' => out.push_str(r"\n"),
                    't' => out.push_str(r"\t"),
                    'r' => out.push_str(r"\r"),
                    'f' => out.push_str(r"\x0C"),
                    'v' => out.push_str(r"\x0B"),
                    '1'..='9' => panic!("backreferences are not supported"),
                    other => push_literal(&mut out, other),
                }
            }
            '[' => {
                in_class = true;
                class_start = true;
                out.push('[');
                if chars.get(i + 1) == Some(&'^') {
                    out.push('^');
                    i += 1;
                }
                if chars.get(i + 1) == Some(&']') {
                    panic!("empty JavaScript classes are not supported");
                }
            }
            '.' => out.push_str(r"[^\n\r\x{2028}\x{2029}]"),
            '{' => {
                if is_quantifier(&chars, i) {
                    // Copy the whole `{n}` / `{n,}` / `{n,m}` quantifier.
                    while chars[i] != '}' {
                        out.push(chars[i]);
                        i += 1;
                    }
                    out.push('}');
                } else {
                    out.push_str(r"\{");
                }
            }
            // A brace that does not close a quantifier is a literal in JavaScript.
            '}' => out.push_str(r"\}"),
            '(' if chars.get(i + 1) == Some(&'?')
                && matches!(chars.get(i + 2), Some('=') | Some('!') | Some('<')) =>
            {
                panic!("lookaround is not supported")
            }
            c if ignore_case && c.is_ascii_alphabetic() => push_case_pair(&mut out, c, false),
            other => out.push(other),
        }
        i += 1;
    }
    assert!(!in_class, "unterminated class in pattern {pattern}");
    out
}

thread_local! {
    static CACHE: RefCell<HashMap<(String, bool), Regex>> = RefCell::new(HashMap::new());
}

impl JsRegex {
    /// Compile a JavaScript pattern. `flags` may contain `g` (ignored: iteration is
    /// explicit) and `i`.
    pub fn new(pattern: &str, flags: &str) -> JsRegex {
        let ignore_case = flags.contains('i');
        let key = (pattern.to_owned(), ignore_case);
        let re = CACHE.with(|cache| {
            if let Some(re) = cache.borrow().get(&key) {
                return re.clone();
            }
            let translated = translate(pattern, ignore_case);
            let re = Regex::new(&translated).unwrap_or_else(|e| {
                panic!("translated JavaScript pattern /{pattern}/ does not compile: {e}")
            });
            cache.borrow_mut().insert(key, re.clone());
            re
        });
        JsRegex { re }
    }

    /// `re.test(text)`.
    pub fn test(&self, text: &str) -> bool {
        self.re.is_match(text)
    }

    /// `re.exec(text)` from position 0.
    pub fn exec<'t>(&self, text: &'t str) -> Option<Captures<'t>> {
        self.re.captures(text)
    }

    /// Every match of a global `exec` loop.
    pub fn exec_all<'r, 't>(&'r self, text: &'t str) -> regex::CaptureMatches<'r, 't> {
        self.re.captures_iter(text)
    }

    /// `(text.match(globalRe) || []).length`.
    pub fn count(&self, text: &str) -> usize {
        self.re.find_iter(text).count()
    }

    /// `text.replace(globalRe, replacement)` with a literal replacement.
    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        self.re
            .replace_all(text, regex::NoExpand(replacement))
            .into_owned()
    }

    /// `text.replace(re, replacement)` (first match) with a literal replacement.
    pub fn replace_first(&self, text: &str, replacement: &str) -> String {
        self.re
            .replace(text, regex::NoExpand(replacement))
            .into_owned()
    }
}

/// The text of capture group `i`, or `None` when the group did not participate.
pub fn group<'t>(caps: &Captures<'t>, i: usize) -> Option<&'t str> {
    caps.get(i).map(|m| m.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_classes_are_ascii() {
        let re = JsRegex::new(r"\bfoo\b", "g");
        // `π` is not a JavaScript word character, so it bounds the word.
        assert_eq!(re.count("πfoo foo_ xfoo foo"), 2);
        let w = JsRegex::new(r"^\w+$", "");
        assert!(!w.test("aπ"));
        assert!(w.test("a_9Z"));
    }

    #[test]
    fn dot_stops_at_js_line_terminators() {
        let re = JsRegex::new(r"a.b", "");
        assert!(re.test("axb"));
        assert!(!re.test("a\rb"));
        assert!(!re.test("a\u{2028}b"));
    }

    #[test]
    fn ignore_case_is_ascii_exact() {
        let re = JsRegex::new(r"struct\s+(\w*(?:Params|Uniforms))\s*\{([^}]+)\}", "gi");
        let caps = re.exec("STRUCT myparams { a: f32 }").unwrap();
        assert_eq!(group(&caps, 1), Some("myparams"));
        // Kelvin sign and long s never fold onto ASCII in JavaScript.
        assert!(!JsRegex::new("k", "i").test("\u{212A}"));
        assert!(!JsRegex::new("s", "i").test("\u{17F}"));
        let class = JsRegex::new("^[xyzw]+$", "i");
        assert!(class.test("XyZw"));
    }

    #[test]
    fn braces_and_quantifiers() {
        assert_eq!(translate(r"a{2}", false), "a{2}");
        assert_eq!(translate(r"\{([^}]+)\}", false), r"\{([^}]+)\}");
        let re = JsRegex::new(r"x{1,2}", "");
        assert_eq!(re.exec("xxx").unwrap().get(0).unwrap().as_str(), "xx");
    }

    #[test]
    fn lazy_captures_match_backtracking() {
        let re = JsRegex::new(r"^array\s*<\s*(.+?)\s*,\s*(\d+)\s*>$", "");
        let caps = re.exec("array<array<f32,2>,3>").unwrap();
        assert_eq!(group(&caps, 1), Some("array<f32,2>"));
        assert_eq!(group(&caps, 2), Some("3"));
    }
}
