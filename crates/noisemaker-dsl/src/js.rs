//! JavaScript number and string semantics the reference engine relies on —
//! the one implementation every crate of the port shares.
//!
//! The reference runs in JavaScript: every number is an IEEE double, strings
//! become numbers through `Number(s)`, `parseFloat` and `parseInt`, numbers
//! become strings through Number::toString, `>>> 0` and `| 0` truncate
//! through ToUint32 and ToInt32, and `Math.round`, `Math.min` and `Math.max`
//! keep JavaScript's signed zeros and NaN propagation. The basic coercions and
//! comparisons of [`Value`]s (`ToNumber`, `String(v)`, `===`, SameValueZero)
//! are here too. Other modules present these under their own facades
//! ([`crate::unparser::jsv`] adds the DSL tooling's property semantics and
//! V8's error messages; `noisemaker_gpu::jsv` the WebIDL conversions;
//! `noisemaker_host::js` re-exports this module) without reimplementing them.
//!
//! Transcendental functions are deliberately not here, because JavaScript
//! hosts differ on them: [`crate::jsmath`] ports V8's fdlibm
//! `Math.sin`/`Math.cos` (what Node computes, where the reference's
//! automation and palettes are checked), while `noisemaker_host::js` evaluates
//! `Math.sin`/`Math.cos`/`Math.log` correctly rounded, as Chromium returns
//! them to the reference's host code in the browser.

use crate::value::Value;

// ----------------------------------------------------------------- strings

/// ECMAScript WhiteSpace and LineTerminator code points: the set shared by
/// `String.prototype.trim`, the regular-expression class `\s`, `parseFloat`,
/// `parseInt` and StringToNumber.
pub fn is_js_whitespace(c: char) -> bool {
    matches!(
        c,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

/// `String.prototype.trim()`.
pub fn trim(s: &str) -> &str {
    s.trim_matches(is_js_whitespace)
}

/// `s.split(/\s+/)` for a string without leading or trailing whitespace (a
/// trimmed line): the maximal runs of non-whitespace.
pub fn split_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_js_whitespace).filter(|t| !t.is_empty())
}

// ----------------------------------------------------- numbers to strings

/// `Number.prototype.toString()` (radix 10), ECMAScript Number::toString: the
/// shortest round-trip digits, in fixed notation for decimal exponents
/// -6 < n <= 21 and in exponential notation otherwise.
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
    // Rust's LowerExp without a precision prints the shortest digits that
    // round-trip (closest to the value), the digits ECMAScript requires:
    // s (k digits) and n with s × 10^(n-k) = x.
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

// ----------------------------------------------------- strings to numbers

/// Length of the longest prefix of `s` that is a StrUnsignedDecimalLiteral
/// (digits with an optional fraction, or a fraction alone, then an optional
/// exponent with digits).
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

/// The value of `digits` (all valid in `radix`), rounded to the nearest
/// double as V8 rounds radix literals and `parseInt` results: exactly for
/// power-of-two radixes (round half to even over the exact binary
/// expansion), correctly rounded for radix 10, and by accumulation for the
/// other radixes (as V8 does for them).
fn digits_to_f64(digits: &str, radix: u32) -> f64 {
    if radix == 10 {
        // Rust's conversion is correctly rounded, as V8's Strtod is.
        return digits.parse::<f64>().unwrap_or(f64::INFINITY);
    }
    if !radix.is_power_of_two() {
        return digits.chars().fold(0.0, |acc, c| {
            acc * radix as f64 + c.to_digit(radix).unwrap() as f64
        });
    }
    let bits_per_digit = radix.trailing_zeros();
    let mut bits: Vec<u8> = Vec::with_capacity(digits.len() * bits_per_digit as usize);
    for c in digits.chars() {
        let d = c.to_digit(radix).expect("a digit of the radix");
        for shift in (0..bits_per_digit).rev() {
            bits.push(((d >> shift) & 1) as u8);
        }
    }
    let Some(first) = bits.iter().position(|&b| b == 1) else {
        return 0.0;
    };
    let bits = &bits[first..];
    if bits.len() <= 53 {
        return bits.iter().fold(0.0, |acc, &b| acc * 2.0 + f64::from(b));
    }
    let mut mantissa: u64 = bits[..53]
        .iter()
        .fold(0u64, |acc, &b| (acc << 1) | u64::from(b));
    let round = bits[53] == 1;
    let sticky = bits[54..].contains(&1);
    if round && (sticky || mantissa & 1 == 1) {
        mantissa += 1;
    }
    (mantissa as f64) * 2f64.powi((bits.len() - 53) as i32)
}

/// `parseFloat(s)`: the value of the longest prefix of `s` (after leading
/// whitespace) that is a StrDecimalLiteral, or NaN when no prefix is one.
pub fn parse_float(s: &str) -> f64 {
    let s = s.trim_start_matches(is_js_whitespace);
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
    // Correctly rounded, as V8's conversion is; overflow gives infinity and
    // underflow (signed) zero, as in JavaScript.
    sign * rest[..len].parse::<f64>().unwrap_or(f64::NAN)
}

/// `Number(s)` for strings (StringToNumber): whitespace-trimmed decimal
/// literals, `Infinity`, and `0x` / `0o` / `0b` integers; 0 when empty; NaN
/// otherwise.
pub fn string_to_number(s: &str) -> f64 {
    let t = trim(s);
    if t.is_empty() {
        return 0.0;
    }
    let b = t.as_bytes();
    if b.len() >= 2 && b[0] == b'0' {
        let radix = match b[1] {
            b'x' | b'X' => Some(16),
            b'o' | b'O' => Some(8),
            b'b' | b'B' => Some(2),
            _ => None,
        };
        if let Some(radix) = radix {
            let digits = &t[2..];
            if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
                return f64::NAN;
            }
            return digits_to_f64(digits, radix);
        }
    }
    let (sign, rest) = match b[0] {
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

/// `parseInt(s, radix)` (radix 0: auto-detect `0x`, else 10): an optional
/// sign, then the longest run of digits of the radix after leading
/// whitespace; NaN when there are none.
pub fn parse_int(s: &str, radix: u32) -> f64 {
    let s = s.trim_start_matches(is_js_whitespace);
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
    let end = rest
        .char_indices()
        .find(|(_, c)| !c.is_digit(radix))
        .map_or(rest.len(), |(i, _)| i);
    if end == 0 {
        return f64::NAN;
    }
    sign * digits_to_f64(&rest[..end], radix)
}

// ------------------------------------------------------------------- Math

/// `Math.round(x)`: the nearest integer, ties toward +Infinity; -0 for
/// arguments in [-0.5, -0].
pub fn math_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if (-0.5..0.0).contains(&x) {
        return -0.0;
    }
    let floor = x.floor();
    // x - floor(x) is exact for every finite double.
    if x - floor >= 0.5 { floor + 1.0 } else { floor }
}

/// `Math.max(a, b)`: NaN when either is NaN; +0 is above -0.
pub fn math_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_positive() { a } else { b };
    }
    if a > b { a } else { b }
}

/// `Math.min(a, b)`: NaN when either is NaN; -0 is below +0.
pub fn math_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a < b { a } else { b }
}

/// `Math.max(lo, Math.min(hi, x))`.
pub fn math_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    math_max(lo, math_min(hi, x))
}

/// `Number.isInteger(x)` for a number.
pub fn is_integer(x: f64) -> bool {
    x.is_finite() && x.trunc() == x
}

/// `value || 0` for a number: NaN, +0 and -0 all become +0.
pub fn or_zero(value: f64) -> f64 {
    if value.is_nan() || value == 0.0 {
        0.0
    } else {
        value
    }
}

/// ECMAScript `ToInt32`.
pub fn to_int32(x: f64) -> i32 {
    to_uint32(x) as i32
}

/// ECMAScript `ToUint32`.
pub fn to_uint32(x: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    x.trunc().rem_euclid(4_294_967_296.0) as u32
}

// ----------------------------------------------------------------- values

/// `String(value)`, the conversion of template literals and property keys,
/// for the reference's data: objects have the default `toString`
/// (`[object Object]`), arrays join their elements with commas (`null` and
/// `undefined` as empty), functions print their source.
/// [`crate::unparser::jsv::to_string`] is the exact `ToString` for the DSL
/// tooling, where a plain object's own `toString`/`valueOf` members can make
/// the conversion throw.
pub fn value_to_property_key(v: &Value) -> String {
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

/// `ToNumber(value)` for the reference's data (objects with the default
/// `valueOf`/`toString`): arrays convert through their string form (`[3]` is
/// 3), objects and functions are NaN.
pub fn to_number(value: &Value) -> f64 {
    match value {
        Value::Undefined => f64::NAN,
        Value::Null => 0.0,
        Value::Bool(b) => f64::from(u8::from(*b)),
        Value::Number(n) => *n,
        Value::String(s) => string_to_number(s),
        Value::Array(_) => string_to_number(&value_to_property_key(value)),
        Value::Object(_) | Value::Function(_) => f64::NAN,
    }
}

/// `a === b`. Objects, arrays and functions compare by identity in
/// JavaScript; a [`Value`] owns its members, so two of them are never the
/// same reference and never strictly equal.
pub fn strict_equals(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        _ => false,
    }
}

/// `SameValueZero(a, b)`: the key equality of `Map` and `Set` and of
/// `Array.prototype.includes` (`===`, except that NaN equals NaN).
pub fn same_value_zero(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x == y || (x.is_nan() && y.is_nan()),
        _ => strict_equals(a, b),
    }
}

/// `Number.isFinite(value)`.
pub fn is_finite_number(value: &Value) -> bool {
    matches!(value, Value::Number(n) if n.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formatting_matches_js() {
        for (x, s) in [
            (1.0, "1"),
            (0.5, "0.5"),
            (0.1, "0.1"),
            (-0.0, "0"),
            (0.30000000000000004, "0.30000000000000004"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (123456789012345680000.0, "123456789012345680000"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (-2.5e-10, "-2.5e-10"),
            (0.000001, "0.000001"),
            (123.456, "123.456"),
            (-2.5, "-2.5"),
            (300.0, "300"),
            (f64::INFINITY, "Infinity"),
            (f64::NEG_INFINITY, "-Infinity"),
            (f64::NAN, "NaN"),
        ] {
            assert_eq!(number_to_string(x), s, "{x:e}");
        }
        assert_eq!(integer_to_string_radix(-35, 36), "-z");
    }

    #[test]
    fn parse_float_matches_js() {
        assert_eq!(parse_float("  3.5abc"), 3.5);
        assert!(parse_float("abc").is_nan());
        assert!(parse_float(".").is_nan());
        assert!(parse_float("").is_nan());
        assert!(parse_float("-").is_nan());
        assert!(parse_float("+-1").is_nan());
        assert_eq!(parse_float("1."), 1.0);
        assert_eq!(parse_float(".5"), 0.5);
        assert_eq!(parse_float("-.5e-3"), -0.0005);
        assert_eq!(parse_float("1e"), 1.0);
        assert_eq!(parse_float("1e+"), 1.0);
        assert_eq!(parse_float("1.2.3"), 1.2);
        assert_eq!(parse_float("0x10"), 0.0);
        assert_eq!(parse_float("Infinityx"), f64::INFINITY);
        assert_eq!(parse_float("-Infinity"), f64::NEG_INFINITY);
        assert!(parse_float("infinity").is_nan());
        assert_eq!(parse_float("1e400"), f64::INFINITY);
        assert_eq!(parse_float("-1e-400").to_bits(), (-0.0f64).to_bits());
        assert_eq!(parse_float("-0").to_bits(), (-0.0f64).to_bits());
        assert_eq!(parse_float("\u{a0} 7"), 7.0);
    }

    #[test]
    fn parse_int_matches_js() {
        assert_eq!(parse_int("42px", 10), 42.0);
        assert_eq!(parse_int("-1", 10), -1.0);
        assert_eq!(parse_int("+5", 10), 5.0);
        assert_eq!(parse_int("1.5", 10), 1.0);
        assert_eq!(parse_int("1e3", 10), 1.0);
        assert!(parse_int("", 10).is_nan());
        assert!(parse_int("x1", 10).is_nan());
        assert_eq!(parse_int("-0", 10).to_bits(), (-0.0f64).to_bits());
        assert_eq!(parse_int("ff", 16), 255.0);
        assert_eq!(parse_int("0x1F", 0), 31.0);
        assert_eq!(parse_int("0x1F", 16), 31.0);
        assert_eq!(parse_int("0x1F", 10), 0.0);
        assert_eq!(parse_int("08", 0), 8.0);
        // Correctly rounded beyond 2^53, as V8's Strtod and its power-of-two
        // radix conversion round.
        assert_eq!(parse_int("9007199254740993", 10), 9007199254740992.0);
        assert_eq!(parse_int("9007199254740995", 10), 9007199254740996.0);
        assert_eq!(parse_int("20000000000003", 16), 9007199254740996.0);
    }

    #[test]
    fn string_to_number_follows_the_spec() {
        assert_eq!(string_to_number(""), 0.0);
        assert_eq!(string_to_number("  12 \n"), 12.0);
        assert_eq!(string_to_number("-1.5e3"), -1500.0);
        assert_eq!(string_to_number(".5"), 0.5);
        assert_eq!(string_to_number("5."), 5.0);
        assert_eq!(string_to_number(" 0x1f "), 31.0);
        assert_eq!(string_to_number("0X1F"), 31.0);
        assert_eq!(string_to_number("0b101"), 5.0);
        assert_eq!(string_to_number("0o17"), 15.0);
        assert_eq!(string_to_number("-Infinity"), f64::NEG_INFINITY);
        for nan in [
            "1e", "abc", "1_000", "-0x10", "inf", "NaN", "0x", ".", "+-1", "1 2", "1px",
        ] {
            assert!(string_to_number(nan).is_nan(), "{nan}");
        }
        assert_eq!(string_to_number("0x20000000000001"), 9007199254740992.0);
        assert_eq!(string_to_number("0x20000000000003"), 9007199254740996.0);
        assert_eq!(string_to_number("0x20000000000005"), 9007199254740996.0);
    }

    #[test]
    fn math_follows_js() {
        assert_eq!(math_round(-2.5), -2.0);
        assert_eq!(math_round(2.5), 3.0);
        assert_eq!(math_round(-0.6), -1.0);
        assert_eq!(math_round(-0.3).to_bits(), (-0.0f64).to_bits());
        assert_eq!(math_round(-0.5).to_bits(), (-0.0f64).to_bits());
        assert_eq!(math_round(0.49999999999999994), 0.0);
        assert!(math_min(1.0, f64::NAN).is_nan());
        assert!(math_max(f64::NAN, 0.0).is_nan());
        assert!(math_min(0.0, -0.0).is_sign_negative());
        assert!(math_max(-0.0, 0.0).is_sign_positive());
        assert_eq!(math_clamp(1.5, 0.0, 1.0), 1.0);
        assert!(math_clamp(-0.0, 0.0, 1.0).is_sign_positive());
        assert!(is_integer(-0.0) && !is_integer(0.5) && !is_integer(f64::INFINITY));
        assert_eq!(or_zero(f64::NAN).to_bits(), 0);
        assert_eq!(or_zero(-0.0).to_bits(), 0);
        assert_eq!(to_int32(4294967297.0), 1);
        assert_eq!(to_int32(4294967295.0), -1);
        assert_eq!(to_uint32(-1.0), 4294967295);
        assert_eq!(to_uint32(4294967296.0 + 5.0), 5);
        assert_eq!(to_uint32(f64::NAN), 0);
        assert_eq!(to_uint32(-0.5), 0);
        assert_eq!(to_uint32(1e300), 0);
    }

    #[test]
    fn value_coercions() {
        assert!(to_number(&Value::Undefined).is_nan());
        assert_eq!(to_number(&Value::Null), 0.0);
        assert_eq!(to_number(&Value::from("0x10")), 16.0);
        assert_eq!(to_number(&Value::Array(vec![Value::Number(5.0)])), 5.0);
        assert!(to_number(&Value::Array(vec![1.0.into(), 2.0.into()])).is_nan());
        assert!(to_number(&Value::object()).is_nan());
        assert_eq!(
            value_to_property_key(&Value::Array(vec![Value::Null, 1.5.into(), "x".into()])),
            ",1.5,x"
        );
        assert!(strict_equals(&Value::Number(0.0), &Value::Number(-0.0)));
        assert!(!strict_equals(
            &Value::Number(f64::NAN),
            &Value::Number(f64::NAN)
        ));
        assert!(same_value_zero(
            &Value::Number(f64::NAN),
            &Value::Number(f64::NAN)
        ));
        assert!(!strict_equals(&Value::object(), &Value::object()));
        assert!(is_finite_number(&Value::Number(1.0)));
        assert!(!is_finite_number(&Value::from("1")));
    }

    #[test]
    fn whitespace() {
        assert_eq!(trim("\u{feff}\u{2028} a b \u{3000}"), "a b");
        assert_eq!(
            split_whitespace("v 1\t2  3").collect::<Vec<_>>(),
            ["v", "1", "2", "3"]
        );
        assert!(!is_js_whitespace('\u{200b}'));
    }
}
