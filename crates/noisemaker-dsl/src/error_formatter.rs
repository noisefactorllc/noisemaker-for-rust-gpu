//! Port of `lang/error-formatter.js`: DSL syntax errors rendered with source
//! context ([`format_dsl_error`]) and the syntax-error test
//! ([`is_dsl_syntax_error`]).
//!
//! A thrown value is a [`JsError`]: `JsError::Error` stands for an `Error`
//! instance (its `name` is the constructor name, so `instanceof SyntaxError`
//! holds for `name == "SyntaxError"`), `JsError::Thrown` for any other thrown
//! value. Options and the source are read as plain JavaScript values.

use crate::error::JsError;
use crate::js::{is_js_whitespace, math_max, math_min, number_to_string, trim};
use crate::unparser::jsv::{self, get, get_opt, pad_start, repeat, to_number, to_string};
use crate::value::Value;

/// `parseLocation(message)`: the first `at line N col M` (or `column M`).
fn parse_location(message: &str) -> Option<(f64, f64)> {
    let bytes = message.as_bytes();
    let digits = |from: usize| -> usize {
        bytes[from..]
            .iter()
            .take_while(|b| b.is_ascii_digit())
            .count()
    };
    let mut search = 0;
    while let Some(found) = message[search..].find("at line ") {
        let start = search + found;
        search = start + 1;
        let mut i = start + "at line ".len();
        let line_len = digits(i);
        if line_len == 0 {
            continue;
        }
        let line = &message[i..i + line_len];
        i += line_len;
        if !message[i..].starts_with(" col") {
            continue;
        }
        i += " col".len();
        // `col(?:umn)? ` — the longer alternative first.
        if message[i..].starts_with("umn ") {
            i += "umn ".len();
        } else if message[i..].starts_with(' ') {
            i += 1;
        } else {
            continue;
        }
        let col_len = digits(i);
        if col_len == 0 {
            continue;
        }
        let col = &message[i..i + col_len];
        // parseInt(digits, 10) is the correctly rounded decimal value.
        return Some((
            line.parse::<f64>().unwrap_or(f64::NAN),
            col.parse::<f64>().unwrap_or(f64::NAN),
        ));
    }
    None
}

/// Whether `s` is exactly `at line \d+ col(?:umn)? \d+` (the suffix the
/// location regex anchors at the end of the message).
fn location_suffix_matches(s: &str) -> bool {
    let Some(rest) = s.strip_prefix("at line ") else {
        return false;
    };
    let n = rest.bytes().take_while(u8::is_ascii_digit).count();
    if n == 0 {
        return false;
    }
    let Some(rest) = rest[n..].strip_prefix(" col") else {
        return false;
    };
    let tail = |r: &str| {
        let n = r.bytes().take_while(u8::is_ascii_digit).count();
        n > 0 && n == r.len()
    };
    rest.strip_prefix("umn ").is_some_and(tail) || rest.strip_prefix(' ').is_some_and(tail)
}

/// `extractMessage(message)`: the message without its trailing location suffix
/// (`message.replace(/\s+at line \d+ col(?:umn)? \d+$/, '').trim()`).
fn extract_message(message: &str) -> String {
    let chars: Vec<(usize, char)> = message.char_indices().collect();
    // The leftmost match: the start of a whitespace run followed by the suffix.
    let mut i = 0;
    let mut stripped: Option<&str> = None;
    while i < chars.len() {
        if !is_js_whitespace(chars[i].1) {
            i += 1;
            continue;
        }
        let run_start = chars[i].0;
        let mut j = i;
        while j < chars.len() && is_js_whitespace(chars[j].1) {
            j += 1;
        }
        let after = chars.get(j).map_or(message.len(), |c| c.0);
        if j > i && location_suffix_matches(&message[after..]) {
            stripped = Some(&message[..run_start]);
            break;
        }
        i = j;
    }
    trim(stripped.unwrap_or(message)).to_owned()
}

/// `a + b` where `a` is a number: string concatenation when `b` is (or converts
/// to) a string, numeric addition otherwise.
fn add(a: f64, b: &Value) -> Result<Value, JsError> {
    Ok(match b {
        Value::String(s) => Value::String(format!("{}{s}", number_to_string(a))),
        Value::Array(_) | Value::Object(_) | Value::Function(_) => {
            Value::String(format!("{}{}", number_to_string(a), to_string(b)?))
        }
        other => Value::Number(a + to_number(other)?),
    })
}

/// `lines[index]` for a numeric index (`undefined` outside the array or for a
/// non-integral index).
fn line_at(lines: &[&str], index: f64) -> Value {
    if index >= 0.0
        && index.fract() == 0.0
        && (index as usize) < lines.len()
        && index < 4_294_967_295.0
    {
        Value::from(lines[index as usize])
    } else {
        Value::Undefined
    }
}

/// `String(n).padStart(width, ' ')`.
fn line_number(n: f64, width: usize) -> String {
    pad_start(&number_to_string(n), width, ' ')
}

/// `formatDslError(source, error, options)`: a syntax error with the offending
/// line, `options.contextLines` (default 2) lines of context on each side, and a
/// caret under the error column. Errors without a location (or without a
/// source) format as `SyntaxError: <message>`.
pub fn format_dsl_error(
    source: &Value,
    error: &JsError,
    options: &Value,
) -> Result<String, JsError> {
    let context_lines = match options {
        Value::Undefined => Value::Number(2.0),
        _ => {
            let v = get(options, "contextLines")?;
            if v.is_undefined() {
                Value::Number(2.0)
            } else {
                v
            }
        }
    };

    let message = match error {
        JsError::Error { message, .. } => message.clone(),
        JsError::Thrown(value) => {
            if !value.is_truthy() {
                return Ok("Unknown error".into());
            }
            match get_opt(value, "message") {
                Value::String(m) => m,
                _ => return to_string(value),
            }
        }
    };

    let loc = parse_location(&message);
    let core_message = extract_message(&message);
    let (Some((error_line, error_col)), true) = (loc, source.is_truthy()) else {
        // No location info, just return the message without stack
        return Ok(format!("SyntaxError: {core_message}"));
    };

    let source = match source {
        Value::String(s) => s.as_str(),
        _ => return Err(jsv::not_a_function("source.split")),
    };
    let lines: Vec<&str> = source.split('\n').collect();
    let line_count = lines.len() as f64;

    // Calculate line number width for padding
    let last_line_num = math_min(to_number(&add(error_line, &context_lines)?)?, line_count);
    let line_num_width = number_to_string(last_line_num).len();

    let mut parts = vec![
        format!("SyntaxError: {core_message}"),
        format!(
            "  --> line {}, column {}",
            number_to_string(error_line),
            number_to_string(error_col)
        ),
        String::new(),
    ];

    // Context lines before
    let start_line = math_max(1.0, error_line - to_number(&context_lines)?);
    let mut i = start_line;
    while i < error_line {
        parts.push(format!(
            "  {} | {}",
            line_number(i, line_num_width),
            to_string(&line_at(&lines, i - 1.0))?
        ));
        i += 1.0;
    }

    // Error line
    let error_line_content = match line_at(&lines, error_line - 1.0) {
        Value::String(s) if !s.is_empty() => s,
        _ => String::new(),
    };
    parts.push(format!(
        "  {} | {error_line_content}",
        line_number(error_line, line_num_width)
    ));

    // Pointer line
    let pointer_padding = repeat(" ", (line_num_width + 3) as f64)?; // "  N | " prefix
    let col_padding = repeat(" ", math_max(0.0, error_col - 1.0))?;
    parts.push(format!("{pointer_padding}{col_padding}^-- error here"));

    // Context lines after
    let end_line = math_min(line_count, to_number(&add(error_line, &context_lines)?)?);
    let mut i = error_line + 1.0;
    while i <= end_line {
        parts.push(format!(
            "  {} | {}",
            line_number(i, line_num_width),
            to_string(&line_at(&lines, i - 1.0))?
        ));
        i += 1.0;
    }

    Ok(parts.join("\n"))
}

/// A failed compile as terminal text, the way the reference demo reports one
/// (`UIController.formatCompilationError(err, source)`): a DSL syntax error
/// with a location is formatted with its source context by
/// [`format_dsl_error`] (what the demo logs to the console); any other error
/// is the demo's status text ([`crate::compiler::format_error`]: validator
/// diagnostics with their locations, expansion errors, messages).
pub fn format_compile_error(source: &str, error: &JsError) -> String {
    if is_dsl_syntax_error(error)
        && let Ok(text) = format_dsl_error(&Value::from(source), error, &Value::Undefined)
    {
        return text;
    }
    crate::compiler::format_error(error)
}

/// `isDslSyntaxError(error)`: a `SyntaxError` whose message carries a location.
pub fn is_dsl_syntax_error(error: &JsError) -> bool {
    match error {
        JsError::Error { name, message } => {
            name == "SyntaxError" && parse_location(message).is_some()
        }
        JsError::Thrown(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locations_and_messages() {
        assert_eq!(
            parse_location("Unexpected token at line 3 col 7"),
            Some((3.0, 7.0))
        );
        assert_eq!(
            parse_location("x at line 12 column 5. Valid: a"),
            Some((12.0, 5.0))
        );
        assert_eq!(
            parse_location("at line x col 1 then at line 2 col 3"),
            Some((2.0, 3.0))
        );
        assert_eq!(parse_location("at line 1 colum 2"), None);
        assert_eq!(
            extract_message("Unexpected token at line 3 col 7"),
            "Unexpected token"
        );
        assert_eq!(
            extract_message("bad at line 1 col 2 and at line 3 col 4"),
            "bad at line 1 col 2 and"
        );
        assert_eq!(
            extract_message("  osc() unknown at line 1 col 2. Valid: x  "),
            "osc() unknown at line 1 col 2. Valid: x"
        );
    }

    #[test]
    fn formats_context() {
        let src = Value::from("search synth\nnoise(\n  .write(o0)\nrender(o0)");
        let err = JsError::syntax("Unexpected token DOT at line 3 col 3");
        let out = format_dsl_error(&src, &err, &Value::Undefined).unwrap();
        assert_eq!(
            out,
            "SyntaxError: Unexpected token DOT\n  --> line 3, column 3\n\n  1 | search synth\n  2 | noise(\n  3 |   .write(o0)\n      ^-- error here\n  4 | render(o0)"
        );
        assert!(is_dsl_syntax_error(&err));
        assert_eq!(
            format_compile_error("search synth\nnoise(\n  .write(o0)\nrender(o0)", &err),
            out
        );
        assert!(!is_dsl_syntax_error(&JsError::type_error(
            "at line 1 col 1"
        )));
        assert_eq!(
            format_dsl_error(&src, &JsError::Thrown(Value::Null), &Value::Undefined).unwrap(),
            "Unknown error"
        );
        // A line past the end of the source prints `undefined` context lines.
        let out = format_dsl_error(
            &Value::from("a"),
            &JsError::syntax("x at line 3 col 1"),
            &Value::Undefined,
        )
        .unwrap();
        assert_eq!(
            out,
            "SyntaxError: x\n  --> line 3, column 1\n\n  1 | a\n  2 | undefined\n  3 | \n    ^-- error here"
        );
    }
}
