//! Port of `lang/stringLiterals.js`.

/// `decodeJsonStringLiteralContent(raw)`: the raw content of a DSL string (the
/// lexer keeps escapes) decoded as a JSON string when it is one, else with the
/// small escape set the lexer accepts in single-quoted strings (unknown escapes
/// kept as written).
pub fn decode_json_string_literal_content(raw: &str) -> String {
    if let Some(decoded) = json_parse_string_content(raw) {
        return decoded;
    }
    let mut decoded = String::new();
    let mut chars = raw.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' || chars.peek().is_none() {
            decoded.push(c);
            continue;
        }
        let next = chars.next().expect("peeked");
        match next {
            '\'' => decoded.push('\''),
            '"' => decoded.push('"'),
            '\\' => decoded.push('\\'),
            'n' => decoded.push('\n'),
            'r' => decoded.push('\r'),
            't' => decoded.push('\t'),
            'b' => decoded.push('\u{8}'),
            'f' => decoded.push('\u{c}'),
            'v' => decoded.push('\u{b}'),
            '0' => decoded.push('\0'),
            other => {
                decoded.push('\\');
                decoded.push(other);
            }
        }
    }
    decoded
}

/// `JSON.parse('"' + raw + '"')`, or `None` when that throws.
fn json_parse_string_content(raw: &str) -> Option<String> {
    let mut units: Vec<u16> = Vec::new();
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return None,
            '\u{0}'..='\u{1f}' => return None,
            '\\' => match chars.next()? {
                '"' => units.push(0x22),
                '\\' => units.push(0x5C),
                '/' => units.push(0x2F),
                'b' => units.push(0x08),
                'f' => units.push(0x0C),
                'n' => units.push(0x0A),
                'r' => units.push(0x0D),
                't' => units.push(0x09),
                'u' => {
                    let hex: String = (0..4).map(|_| chars.next()).collect::<Option<String>>()?;
                    if !hex.chars().all(|h| h.is_ascii_hexdigit()) {
                        return None;
                    }
                    units.push(u16::from_str_radix(&hex, 16).ok()?);
                }
                _ => return None,
            },
            other => {
                let mut buf = [0u16; 2];
                units.extend_from_slice(other.encode_utf16(&mut buf));
            }
        }
    }
    // A lone surrogate escape decodes to a lone surrogate in JavaScript, which a
    // Rust string cannot hold; the replacement character stands for it.
    Some(String::from_utf16_lossy(&units))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoding() {
        assert_eq!(decode_json_string_literal_content(r#"a\"b"#), "a\"b");
        assert_eq!(decode_json_string_literal_content(r"A"), "A");
        assert_eq!(decode_json_string_literal_content(r"it\'s"), "it's");
        assert_eq!(decode_json_string_literal_content(r"\q"), r"\q");
        assert_eq!(decode_json_string_literal_content("tab\there"), "tab\there");
        assert_eq!(decode_json_string_literal_content(r"x\"), r"x\");
    }
}
