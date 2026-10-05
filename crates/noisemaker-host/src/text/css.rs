//! The canvas `font` value: the CSS `font` shorthand as Blink parses it for
//! `CanvasRenderingContext2D.font`, restricted to what a value can hold
//! when it starts with the size, as the demo's `${fontSize}px ${font}`
//! always does: `<font-size> [/ <line-height>]? <font-family>#`.
//!
//! An invalid value leaves the context's font unchanged (`10px
//! sans-serif` on a new context), as the canvas ignores the assignment.

/// A CSS generic font family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GenericFamily {
    /// `serif`
    Serif,
    /// `sans-serif`
    SansSerif,
    /// `monospace`
    Monospace,
    /// `cursive`
    Cursive,
    /// `fantasy`
    Fantasy,
    /// `system-ui`
    SystemUi,
    /// `math`
    Math,
}

/// One entry of a `font-family` list.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FontFamily {
    /// A family name (quoted, or a sequence of identifiers joined by single
    /// spaces).
    Named(String),
    /// A generic family keyword.
    Generic(GenericFamily),
}

/// A parsed canvas font.
#[derive(Clone, Debug, PartialEq)]
pub struct CanvasFont {
    /// The computed font size in CSS pixels.
    pub size: f64,
    /// The family list, in order.
    pub families: Vec<FontFamily>,
}

impl Default for CanvasFont {
    /// `10px sans-serif`, a new context's font.
    fn default() -> CanvasFont {
        CanvasFont {
            size: 10.0,
            families: vec![FontFamily::Generic(GenericFamily::SansSerif)],
        }
    }
}

/// Blink clamps computed font sizes to this many pixels
/// (`kMaximumAllowedFontSize`).
pub const MAX_FONT_SIZE: f64 = 10000.0;

#[derive(Clone, Debug, PartialEq)]
enum Token {
    Whitespace,
    Ident(String),
    String(String),
    /// A number with a unit (`26px`), or `%`.
    Dimension(f64, String),
    Number(f64),
    Comma,
    Delim(char),
    BadString,
}

fn is_name_start(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || !c.is_ascii()
}

fn is_name(c: char) -> bool {
    is_name_start(c) || c.is_ascii_digit() || c == '-'
}

fn is_css_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\u{c}')
}

/// Consumes an escape after a backslash: a hexadecimal code point (one to
/// six digits, an optional whitespace) or the next character.
fn consume_escape(chars: &[char], i: &mut usize) -> char {
    let start = *i;
    while *i < chars.len() && *i - start < 6 && chars[*i].is_ascii_hexdigit() {
        *i += 1;
    }
    if *i > start {
        let hex: String = chars[start..*i].iter().collect();
        if *i < chars.len() && is_css_space(chars[*i]) {
            *i += 1;
        }
        let cp = u32::from_str_radix(&hex, 16).unwrap_or(0xFFFD);
        return match char::from_u32(cp) {
            Some(c) if cp != 0 => c,
            _ => '\u{FFFD}',
        };
    }
    if *i < chars.len() {
        *i += 1;
        chars[*i - 1]
    } else {
        '\u{FFFD}'
    }
}

fn valid_escape(chars: &[char], i: usize) -> bool {
    chars.get(i) == Some(&'\\') && chars.get(i + 1).is_some_and(|c| *c != '\n')
}

fn starts_ident(chars: &[char], i: usize) -> bool {
    match chars.get(i) {
        Some('-') => {
            chars
                .get(i + 1)
                .is_some_and(|c| is_name_start(*c) || *c == '-')
                || valid_escape(chars, i + 1)
        }
        Some('\\') => valid_escape(chars, i),
        Some(c) => is_name_start(*c),
        None => false,
    }
}

fn starts_number(chars: &[char], i: usize) -> bool {
    match chars.get(i) {
        Some('+' | '-') => {
            chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())
                || (chars.get(i + 1) == Some(&'.')
                    && chars.get(i + 2).is_some_and(|c| c.is_ascii_digit()))
        }
        Some('.') => chars.get(i + 1).is_some_and(|c| c.is_ascii_digit()),
        Some(c) => c.is_ascii_digit(),
        None => false,
    }
}

fn consume_name(chars: &[char], i: &mut usize) -> String {
    let mut out = String::new();
    while *i < chars.len() {
        let c = chars[*i];
        if is_name(c) {
            out.push(c);
            *i += 1;
        } else if valid_escape(chars, *i) {
            *i += 1;
            out.push(consume_escape(chars, i));
        } else {
            break;
        }
    }
    out
}

fn consume_number(chars: &[char], i: &mut usize) -> f64 {
    let start = *i;
    if matches!(chars[*i], '+' | '-') {
        *i += 1;
    }
    while *i < chars.len() && chars[*i].is_ascii_digit() {
        *i += 1;
    }
    if *i + 1 < chars.len() && chars[*i] == '.' && chars[*i + 1].is_ascii_digit() {
        *i += 1;
        while *i < chars.len() && chars[*i].is_ascii_digit() {
            *i += 1;
        }
    }
    if *i < chars.len() && matches!(chars[*i], 'e' | 'E') {
        let mut k = *i + 1;
        if k < chars.len() && matches!(chars[k], '+' | '-') {
            k += 1;
        }
        if k < chars.len() && chars[k].is_ascii_digit() {
            while k < chars.len() && chars[k].is_ascii_digit() {
                k += 1;
            }
            *i = k;
        }
    }
    let text: String = chars[start..*i].iter().collect();
    text.parse().unwrap_or(f64::NAN)
}

/// CSS Syntax tokenization of the value (the subset a font value uses).
fn tokenize(value: &str) -> Vec<Token> {
    let chars: Vec<char> = value.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if is_css_space(c) {
            while i < chars.len() && is_css_space(chars[i]) {
                i += 1;
            }
            tokens.push(Token::Whitespace);
        } else if c == '"' || c == '\'' {
            i += 1;
            let mut s = String::new();
            let mut bad = false;
            while i < chars.len() {
                let d = chars[i];
                if d == c {
                    i += 1;
                    break;
                } else if d == '\n' {
                    bad = true;
                    break;
                } else if d == '\\' {
                    if i + 1 >= chars.len() {
                        i += 1;
                    } else if chars[i + 1] == '\n' {
                        i += 2;
                    } else {
                        i += 1;
                        s.push(consume_escape(&chars, &mut i));
                    }
                } else {
                    s.push(d);
                    i += 1;
                }
            }
            tokens.push(if bad {
                Token::BadString
            } else {
                Token::String(s)
            });
        } else if starts_number(&chars, i) {
            let n = consume_number(&chars, &mut i);
            if starts_ident(&chars, i) {
                tokens.push(Token::Dimension(n, consume_name(&chars, &mut i)));
            } else if chars.get(i) == Some(&'%') {
                i += 1;
                tokens.push(Token::Dimension(n, "%".into()));
            } else {
                tokens.push(Token::Number(n));
            }
        } else if starts_ident(&chars, i) {
            tokens.push(Token::Ident(consume_name(&chars, &mut i)));
        } else if c == ',' {
            i += 1;
            tokens.push(Token::Comma);
        } else {
            i += 1;
            tokens.push(Token::Delim(c));
        }
    }
    tokens
}

fn generic(ident: &str) -> Option<GenericFamily> {
    Some(match ident.to_ascii_lowercase().as_str() {
        "serif" => GenericFamily::Serif,
        "sans-serif" => GenericFamily::SansSerif,
        "monospace" => GenericFamily::Monospace,
        "cursive" => GenericFamily::Cursive,
        "fantasy" => GenericFamily::Fantasy,
        "system-ui" => GenericFamily::SystemUi,
        "math" => GenericFamily::Math,
        _ => return None,
    })
}

/// CSS-wide keywords and `default`, which cannot start an unquoted family
/// name.
fn reserved(ident: &str) -> bool {
    matches!(
        ident.to_ascii_lowercase().as_str(),
        "initial" | "inherit" | "unset" | "revert" | "revert-layer" | "default"
    )
}

/// `<font-family>#`: generic keywords, quoted names and identifier
/// sequences, separated by commas.
fn parse_family_list(tokens: &[Token]) -> Option<Vec<FontFamily>> {
    let mut families = Vec::new();
    let mut idents: Vec<String> = Vec::new();
    let mut quoted: Option<String> = None;
    let mut expect_family = true;
    let finish = |idents: &mut Vec<String>,
                  quoted: &mut Option<String>,
                  families: &mut Vec<FontFamily>|
     -> bool {
        if let Some(q) = quoted.take() {
            if !idents.is_empty() {
                return false;
            }
            families.push(FontFamily::Named(q));
            return true;
        }
        if idents.is_empty() {
            return false;
        }
        if let [ident] = idents.as_slice()
            && let Some(g) = generic(ident)
        {
            families.push(FontFamily::Generic(g));
            idents.clear();
            return true;
        }
        if reserved(&idents[0]) {
            return false;
        }
        families.push(FontFamily::Named(idents.join(" ")));
        idents.clear();
        true
    };
    for token in tokens {
        match token {
            Token::Whitespace => {}
            Token::Ident(name) => {
                if quoted.is_some() {
                    return None;
                }
                idents.push(name.clone());
                expect_family = false;
            }
            Token::String(s) => {
                if quoted.is_some() || !idents.is_empty() {
                    return None;
                }
                quoted = Some(s.clone());
                expect_family = false;
            }
            Token::Comma => {
                if expect_family || !finish(&mut idents, &mut quoted, &mut families) {
                    return None;
                }
                expect_family = true;
            }
            _ => return None,
        }
    }
    if expect_family || !finish(&mut idents, &mut quoted, &mut families) {
        return None;
    }
    Some(families)
}

/// A `<length>` dimension in CSS pixels (absolute units only; relative
/// units cannot occur in the demo's value).
fn length_px(value: f64, unit: &str) -> Option<f64> {
    let factor = match unit.to_ascii_lowercase().as_str() {
        "px" => 1.0,
        "pt" => 4.0 / 3.0,
        "pc" => 16.0,
        "in" => 96.0,
        "cm" => 96.0 / 2.54,
        "mm" => 96.0 / 25.4,
        "q" => 96.0 / 101.6,
        _ => return None,
    };
    Some(value * factor)
}

/// Parses a canvas `font` value of the form `<size> [/ <line-height>]?
/// <family-list>`. Returns `None` for an invalid value.
pub fn parse_canvas_font(value: &str) -> Option<CanvasFont> {
    let tokens = tokenize(value);
    let mut i = 0;
    while tokens.get(i) == Some(&Token::Whitespace) {
        i += 1;
    }
    let size = match tokens.get(i)? {
        Token::Dimension(v, unit) if *v >= 0.0 => length_px(*v, unit)?,
        _ => return None,
    };
    i += 1;
    // optional `/ <line-height>`
    let mut j = i;
    while tokens.get(j) == Some(&Token::Whitespace) {
        j += 1;
    }
    if tokens.get(j) == Some(&Token::Delim('/')) {
        j += 1;
        while tokens.get(j) == Some(&Token::Whitespace) {
            j += 1;
        }
        match tokens.get(j)? {
            Token::Number(v) | Token::Dimension(v, _) if *v >= 0.0 => {}
            Token::Ident(n) if n.eq_ignore_ascii_case("normal") => {}
            _ => return None,
        }
        i = j + 1;
    }
    // the family list must be separated from the size by whitespace
    if tokens.get(i) != Some(&Token::Whitespace) {
        return None;
    }
    let families = parse_family_list(&tokens[i..])?;
    Some(CanvasFont {
        size: size.min(MAX_FONT_SIZE),
        families,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(s: &str) -> FontFamily {
        FontFamily::Named(s.into())
    }

    #[test]
    fn parses_the_demo_values() {
        let f = parse_canvas_font("26px Nunito").unwrap();
        assert_eq!(f.size, 26.0);
        assert_eq!(f.families, [named("Nunito")]);
        let f = parse_canvas_font("26px Times New Roman").unwrap();
        assert_eq!(f.families, [named("Times New Roman")]);
        let f = parse_canvas_font("26px \"No Such Font\", Georgia, serif").unwrap();
        assert_eq!(
            f.families,
            [
                named("No Such Font"),
                named("Georgia"),
                FontFamily::Generic(GenericFamily::Serif)
            ]
        );
        assert_eq!(
            parse_canvas_font("0px SANS-SERIF").unwrap().families,
            [FontFamily::Generic(GenericFamily::SansSerif)]
        );
        assert_eq!(parse_canvas_font("'serif' 1").map(|f| f.size), None);
        assert_eq!(
            parse_canvas_font("12px 'serif'").unwrap().families,
            [named("serif")]
        );
        assert_eq!(
            parse_canvas_font("1e+21px Nunito").unwrap().size,
            MAX_FONT_SIZE
        );
    }

    #[test]
    fn rejects_invalid_values() {
        for value in [
            "26px ",
            "NaNpx Nunito",
            "Infinitypx Nunito",
            "-3px Nunito",
            "26px 3D Font",
            "26px Nunito,",
            "26px ,Nunito",
            "26px inherit",
            "26px default",
            "26px 'a' b",
            "26pxNunito",
            "26 Nunito",
        ] {
            assert_eq!(parse_canvas_font(value), None, "{value}");
        }
    }
}
