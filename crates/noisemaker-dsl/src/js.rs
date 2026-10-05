//! JavaScript number and string semantics the reference engine relies on.

/// `Number.prototype.toString()` (radix 10) — ECMAScript Number::toString.
pub fn number_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        };
    }
    if x < 0.0 {
        return format!("-{}", number_to_string(-x));
    }
    // Shortest round-trip digits s (k digits) and exponent n with s × 10^(n-k) = x.
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("{:e} always has an exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp.parse::<i32>().expect("exponent is an integer") + 1;
    if k <= n && n <= 21 {
        let mut out = digits;
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
        out
    } else if 0 < n && n <= 21 {
        format!("{}.{}", &digits[..n as usize], &digits[n as usize..])
    } else if -6 < n && n <= 0 {
        format!("0.{}{}", "0".repeat((-n) as usize), digits)
    } else {
        let e = n - 1;
        let sign = if e < 0 { '-' } else { '+' };
        if k == 1 {
            format!("{digits}e{sign}{}", e.abs())
        } else {
            format!("{}.{}e{sign}{}", &digits[..1], &digits[1..], e.abs())
        }
    }
}

/// `String(value)` for primitive values (property-key conversion).
pub fn value_to_property_key(v: &crate::value::Value) -> String {
    use crate::value::Value;
    match v {
        Value::Undefined => "undefined".into(),
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => number_to_string(*n),
        Value::String(s) => s.clone(),
        Value::Array(a) => a
            .iter()
            .map(|e| {
                if e.is_nullish() {
                    String::new()
                } else {
                    value_to_property_key(e)
                }
            })
            .collect::<Vec<_>>()
            .join(","),
        Value::Object(_) => "[object Object]".into(),
        Value::Function(src) => src.clone(),
    }
}

/// `Number.prototype.toString(radix)` for integral values (used for hash ids).
pub fn integer_to_string_radix(mut n: i64, radix: u32) -> String {
    assert!((2..=36).contains(&radix));
    if n == 0 {
        return "0".into();
    }
    let negative = n < 0;
    let mut digits = Vec::new();
    while n != 0 {
        let d = (n % radix as i64).unsigned_abs() as u32;
        digits.push(std::char::from_digit(d, radix).unwrap());
        n /= radix as i64;
    }
    if negative {
        digits.push('-');
    }
    digits.iter().rev().collect()
}

/// `Math.round(x)`: round half toward +∞.
pub fn math_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x;
    }
    let floor = x.floor();
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// ECMAScript `ToInt32`.
pub fn to_int32(x: f64) -> i32 {
    if !x.is_finite() {
        return 0;
    }
    let t = x.trunc();
    let m = t.rem_euclid(4_294_967_296.0);
    (m as u64 as u32) as i32
}

/// ECMAScript `ToUint32`.
pub fn to_uint32(x: f64) -> u32 {
    to_int32(x) as u32
}

const JS_WHITESPACE: &[char] = &[
    '\u{9}', '\u{a}', '\u{b}', '\u{c}', '\u{d}', ' ', '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}',
    '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}',
    '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}', '\u{feff}',
];

/// `String.prototype.trim()` whitespace set.
pub fn trim(s: &str) -> &str {
    s.trim_matches(JS_WHITESPACE)
}

/// Length of the longest prefix of `s` that is a StrDecimalLiteral (no sign).
fn decimal_prefix_len(s: &str) -> usize {
    let b = s.as_bytes();
    let mut i = 0;
    let mut saw_digit = false;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
        saw_digit = true;
    }
    if i < b.len() && b[i] == b'.' {
        let mut j = i + 1;
        let mut frac = false;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
            frac = true;
        }
        if saw_digit || frac {
            i = j;
            saw_digit = true;
        }
    }
    if !saw_digit {
        return 0;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'+' || b[j] == b'-') {
            j += 1;
        }
        let start = j;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > start {
            i = j;
        }
    }
    i
}

/// `parseFloat(s)`.
pub fn parse_float(s: &str) -> f64 {
    let s = s.trim_start_matches(JS_WHITESPACE);
    let (sign, rest) = match s.as_bytes().first() {
        Some(b'-') => (-1.0, &s[1..]),
        Some(b'+') => (1.0, &s[1..]),
        _ => (1.0, s),
    };
    if rest.starts_with("Infinity") {
        return sign * f64::INFINITY;
    }
    let len = decimal_prefix_len(rest);
    if len == 0 {
        return f64::NAN;
    }
    sign * rest[..len].parse::<f64>().unwrap_or(f64::NAN)
}

/// `Number(s)` for strings (StringToNumber).
pub fn string_to_number(s: &str) -> f64 {
    let t = trim(s);
    if t.is_empty() {
        return 0.0;
    }
    let lower = t.to_ascii_lowercase();
    for (prefix, radix) in [("0x", 16), ("0o", 8), ("0b", 2)] {
        if let Some(digits) = lower.strip_prefix(prefix) {
            if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
                return f64::NAN;
            }
            return digits.chars().fold(0.0, |acc, c| {
                acc * radix as f64 + c.to_digit(radix).unwrap() as f64
            });
        }
    }
    let (sign, rest) = match t.as_bytes()[0] {
        b'-' => (-1.0, &t[1..]),
        b'+' => (1.0, &t[1..]),
        _ => (1.0, t),
    };
    if rest == "Infinity" {
        return sign * f64::INFINITY;
    }
    let len = decimal_prefix_len(rest);
    if len == 0 || len != rest.len() {
        return f64::NAN;
    }
    sign * rest.parse::<f64>().unwrap_or(f64::NAN)
}

/// `parseInt(s, radix)` for radix 10 or 16 (0 = auto-detect).
pub fn parse_int(s: &str, radix: u32) -> f64 {
    let s = s.trim_start_matches(JS_WHITESPACE);
    let (sign, mut rest) = match s.as_bytes().first() {
        Some(b'-') => (-1.0, &s[1..]),
        Some(b'+') => (1.0, &s[1..]),
        _ => (1.0, s),
    };
    let mut radix = radix;
    if (radix == 0 || radix == 16) && (rest.starts_with("0x") || rest.starts_with("0X")) {
        rest = &rest[2..];
        radix = 16;
    }
    if radix == 0 {
        radix = 10;
    }
    let digits: String = rest.chars().take_while(|c| c.is_digit(radix)).collect();
    if digits.is_empty() {
        return f64::NAN;
    }
    sign * digits.chars().fold(0.0, |acc, c| {
        acc * radix as f64 + c.to_digit(radix).unwrap() as f64
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formatting_matches_js() {
        for (x, s) in [
            (1.0, "1"),
            (0.1, "0.1"),
            (0.30000000000000004, "0.30000000000000004"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (0.000001, "0.000001"),
            (123.456, "123.456"),
            (-2.5, "-2.5"),
            (300.0, "300"),
        ] {
            assert_eq!(number_to_string(x), s, "{x}");
        }
    }

    #[test]
    fn rounding_and_parsing() {
        assert_eq!(math_round(-2.5), -2.0);
        assert_eq!(math_round(2.5), 3.0);
        assert_eq!(parse_float("  3.5abc"), 3.5);
        assert!(parse_float("abc").is_nan());
        assert_eq!(string_to_number(" 0x1f "), 31.0);
        assert!(string_to_number("1px").is_nan());
        assert_eq!(parse_int("42px", 10), 42.0);
        assert_eq!(to_int32(4294967297.0), 1);
        assert_eq!(integer_to_string_radix(-35, 36), "-z");
    }
}
