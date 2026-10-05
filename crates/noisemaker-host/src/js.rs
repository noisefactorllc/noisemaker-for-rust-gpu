//! ECMAScript number semantics the reference's host code relies on.
//!
//! The reference runs in JavaScript: every value is an IEEE double, strings
//! are parsed with `parseFloat` / `parseInt`, `>>> 0` truncates through
//! ToUint32, and numbers become strings through Number::toString. The ports
//! in this crate evaluate the same operations in the same order with these
//! helpers so their results are bit-identical to the reference's.
//!
//! `Math.sin`, `Math.cos` and `Math.log` are evaluated in double-double
//! arithmetic (about 104 significant bits) and rounded once, i.e. correctly
//! rounded. Chromium 153's V8 returns correctly rounded results for these
//! functions (no difference on 3 million arguments of the tracer's ranges);
//! the platform libm need not (macOS's differs in 4.0 % of sin, 4.2 % of
//! cos and 0.07 % of log results, by 1 ulp), and neither does Node 26's V8
//! (3.3 %, 3.2 % and 6.9 %). `tools/reference-host.mjs math-check` re-measures
//! against the browser, the worm differential test checks every call of the
//! tracer, and the unit tests below check arguments recorded from Chromium.

use std::fmt::Write as _;

/// ECMAScript WhiteSpace and LineTerminator code points: the set shared by
/// `String.prototype.trim`, the regular-expression class `\s`, `parseFloat`
/// and `parseInt`.
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

/// `String.prototype.trim`.
pub fn trim(s: &str) -> &str {
    s.trim_matches(is_js_whitespace)
}

/// `s.split(/\s+/)` for a string without leading or trailing whitespace
/// (a trimmed line): the maximal runs of non-whitespace.
pub fn split_whitespace(s: &str) -> impl Iterator<Item = &str> {
    s.split(is_js_whitespace).filter(|t| !t.is_empty())
}

/// JavaScript `value || 0` for a Number: NaN, +0 and -0 all become +0.
pub fn or_zero(value: f64) -> f64 {
    if value.is_nan() || value == 0.0 {
        0.0
    } else {
        value
    }
}

/// `parseFloat(s)`: the value of the longest prefix of `s` (after leading
/// whitespace) that is a StrDecimalLiteral, or NaN when no prefix is one.
pub fn parse_float(s: &str) -> f64 {
    let t = s.trim_start_matches(is_js_whitespace).as_bytes();
    let n = t.len();
    let mut i = 0;
    let mut negative = false;
    if i < n && (t[i] == b'+' || t[i] == b'-') {
        negative = t[i] == b'-';
        i += 1;
    }
    if t[i..].starts_with(b"Infinity") {
        return if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
    }
    let int_start = i;
    while i < n && t[i].is_ascii_digit() {
        i += 1;
    }
    let int_end = i;
    let mut frac = (i, i);
    if i < n && t[i] == b'.' {
        let mut j = i + 1;
        while j < n && t[j].is_ascii_digit() {
            j += 1;
        }
        // "1." and ".5" are literals, "." is not.
        if int_end > int_start || j > i + 1 {
            frac = (i + 1, j);
            i = j;
        }
    }
    if int_end == int_start && frac.0 == frac.1 {
        return f64::NAN;
    }
    let mut exponent: Option<(bool, usize, usize)> = None;
    if i < n && (t[i] == b'e' || t[i] == b'E') {
        let mut k = i + 1;
        let mut exp_negative = false;
        if k < n && (t[k] == b'+' || t[k] == b'-') {
            exp_negative = t[k] == b'-';
            k += 1;
        }
        let digits_start = k;
        while k < n && t[k].is_ascii_digit() {
            k += 1;
        }
        if k > digits_start {
            exponent = Some((exp_negative, digits_start, k));
        }
    }
    // A canonical literal for Rust's correctly rounded conversion (V8's is
    // correctly rounded too). Overflow gives infinity and underflow zero, as
    // in JavaScript.
    let mut text = String::with_capacity(n + 4);
    if negative {
        text.push('-');
    }
    if int_end > int_start {
        text.push_str(std::str::from_utf8(&t[int_start..int_end]).unwrap());
    } else {
        text.push('0');
    }
    if frac.1 > frac.0 {
        text.push('.');
        text.push_str(std::str::from_utf8(&t[frac.0..frac.1]).unwrap());
    }
    if let Some((exp_negative, a, b)) = exponent {
        text.push('e');
        if exp_negative {
            text.push('-');
        }
        text.push_str(std::str::from_utf8(&t[a..b]).unwrap());
    }
    text.parse::<f64>().unwrap_or(f64::NAN)
}

/// `parseInt(s, 10)`: optional sign, then the longest run of ASCII digits
/// (after leading whitespace); NaN when there are none.
pub fn parse_int10(s: &str) -> f64 {
    let t = s.trim_start_matches(is_js_whitespace).as_bytes();
    let mut i = 0;
    let mut negative = false;
    if i < t.len() && (t[i] == b'+' || t[i] == b'-') {
        negative = t[i] == b'-';
        i += 1;
    }
    let start = i;
    while i < t.len() && t[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return f64::NAN;
    }
    let value: f64 = std::str::from_utf8(&t[start..i])
        .unwrap()
        .parse()
        .unwrap_or(f64::INFINITY);
    if negative { -value } else { value }
}

/// `Math.round`: the nearest integer, ties toward +Infinity; -0 for
/// arguments in [-0.5, -0).
pub fn math_round(x: f64) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    if (-0.5..0.0).contains(&x) {
        return -0.0;
    }
    let floored = x.floor();
    // x - floor(x) is exact for every finite double.
    if x - floored >= 0.5 {
        floored + 1.0
    } else {
        floored
    }
}

/// `Math.max(a, b)` (NaN propagates; +0 is larger than -0).
pub fn math_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_positive() { a } else { b };
    }
    if a > b { a } else { b }
}

/// `Math.min(a, b)` (NaN propagates; -0 is smaller than +0).
pub fn math_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a < b { a } else { b }
}

/// ECMAScript ToUint32.
pub fn to_uint32(x: f64) -> u32 {
    if !x.is_finite() {
        return 0;
    }
    let m = x.trunc() % 4294967296.0;
    let m = if m < 0.0 { m + 4294967296.0 } else { m };
    m as u32
}

/// ECMAScript ToInt32.
pub fn to_int32(x: f64) -> i32 {
    to_uint32(x) as i32
}

/// ECMAScript Number::toString(x) (radix 10): the shortest round-trip
/// digits, in fixed notation for decimal exponents -6 < n <= 21 and in
/// exponential notation otherwise.
pub fn number_to_string(x: f64) -> String {
    if x.is_nan() {
        return "NaN".into();
    }
    if x == 0.0 {
        return "0".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let mut out = String::new();
    let x = if x < 0.0 {
        out.push('-');
        -x
    } else {
        x
    };
    // Rust's LowerExp without a precision prints the shortest digits that
    // round-trip (closest to the value), the digits ECMAScript requires.
    let sci = format!("{x:e}");
    let (mantissa, exp) = sci.split_once('e').expect("LowerExp has an exponent");
    let exp: i32 = exp.parse().expect("LowerExp exponent");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let k = digits.len() as i32;
    let n = exp + 1;
    if k <= n && n <= 21 {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', (n - k) as usize));
    } else if 0 < n && n <= 21 {
        out.push_str(&digits[..n as usize]);
        out.push('.');
        out.push_str(&digits[n as usize..]);
    } else if -6 < n && n <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-n) as usize));
        out.push_str(&digits);
    } else {
        let e = n - 1;
        out.push_str(&digits[..1]);
        if k > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(out, "e{}{}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    out
}

/// A JavaScript value as the reference host code reads it from a
/// parameter object (`params.seed`, `textState.textContent`, ...).
#[derive(Clone, Debug, Default, PartialEq)]
pub enum JsValue {
    /// `undefined` (the key is absent).
    #[default]
    Undefined,
    /// `null`.
    Null,
    /// A boolean.
    Bool(bool),
    /// A Number.
    Number(f64),
    /// A string.
    String(String),
    /// An object or array (an automation descriptor, a vector value):
    /// truthy, and NaN as a Number.
    Object,
}

impl JsValue {
    /// ECMAScript ToBoolean.
    pub fn truthy(&self) -> bool {
        match self {
            JsValue::Undefined | JsValue::Null => false,
            JsValue::Bool(b) => *b,
            JsValue::Number(n) => *n != 0.0 && !n.is_nan(),
            JsValue::String(s) => !s.is_empty(),
            JsValue::Object => true,
        }
    }

    /// ECMAScript ToString, as a template literal interpolates the value
    /// (`${font}`): `undefined`, `null`, `true`, number strings.
    pub fn to_js_string(&self) -> String {
        match self {
            JsValue::Undefined => "undefined".into(),
            JsValue::Null => "null".into(),
            JsValue::Bool(b) => b.to_string(),
            JsValue::Number(n) => number_to_string(*n),
            JsValue::String(s) => s.clone(),
            JsValue::Object => "[object Object]".into(),
        }
    }

    /// `String(value || '')`: the string the demo draws for a text value.
    /// Falsy values give "", numbers their JavaScript string, objects
    /// `[object Object]`.
    pub fn to_text_or_empty(&self) -> String {
        if !self.truthy() {
            return String::new();
        }
        match self {
            JsValue::Bool(b) => b.to_string(),
            JsValue::Number(n) => number_to_string(*n),
            JsValue::String(s) => s.clone(),
            JsValue::Object => "[object Object]".into(),
            JsValue::Undefined | JsValue::Null => String::new(),
        }
    }

    /// ECMAScript ToNumber.
    pub fn to_number(&self) -> f64 {
        match self {
            JsValue::Undefined | JsValue::Object => f64::NAN,
            JsValue::Null => 0.0,
            JsValue::Bool(b) => f64::from(u8::from(*b)),
            JsValue::Number(n) => *n,
            JsValue::String(s) => string_to_number(s),
        }
    }
}

impl From<f64> for JsValue {
    fn from(n: f64) -> JsValue {
        JsValue::Number(n)
    }
}

/// ECMAScript StringToNumber: whitespace-trimmed decimal literals,
/// `Infinity`, and 0x / 0o / 0b integers; NaN otherwise; 0 when empty.
fn string_to_number(s: &str) -> f64 {
    let t = trim(s);
    if t.is_empty() {
        return 0.0;
    }
    for (prefix, radix) in [
        ("0x", 16),
        ("0X", 16),
        ("0o", 8),
        ("0O", 8),
        ("0b", 2),
        ("0B", 2),
    ] {
        if let Some(digits) = t.strip_prefix(prefix) {
            if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
                return f64::NAN;
            }
            return digits.chars().fold(0.0, |acc, c| {
                acc * radix as f64 + c.to_digit(radix).unwrap() as f64
            });
        }
    }
    // The whole string must be a StrDecimalLiteral.
    let value = parse_float(t);
    let unsigned = t.trim_start_matches(['+', '-']);
    let valid = if unsigned == "Infinity" {
        true
    } else {
        let b = unsigned.as_bytes();
        let mut i = 0;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let int_digits = i;
        let mut frac_digits = 0;
        if i < b.len() && b[i] == b'.' {
            i += 1;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
                frac_digits += 1;
            }
        }
        let mut ok = int_digits + frac_digits > 0;
        if ok && i < b.len() && (b[i] == b'e' || b[i] == b'E') {
            i += 1;
            if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
                i += 1;
            }
            let start = i;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
            ok = i > start;
        }
        ok && i == b.len() && t.len() - unsigned.len() <= 1
    };
    if valid { value } else { f64::NAN }
}

// ------------------------------------------------------------ double-double

#[derive(Clone, Copy, Debug)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const fn new(hi: f64) -> Dd {
        Dd { hi, lo: 0.0 }
    }
}

fn two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    let bb = s - a;
    Dd {
        hi: s,
        lo: (a - (s - bb)) + (b - bb),
    }
}

fn quick_two_sum(a: f64, b: f64) -> Dd {
    let s = a + b;
    Dd {
        hi: s,
        lo: b - (s - a),
    }
}

fn two_prod(a: f64, b: f64) -> Dd {
    let p = a * b;
    Dd {
        hi: p,
        lo: a.mul_add(b, -p),
    }
}

fn dd_add(a: Dd, b: Dd) -> Dd {
    let mut s = two_sum(a.hi, b.hi);
    let t = two_sum(a.lo, b.lo);
    s.lo += t.hi;
    s = quick_two_sum(s.hi, s.lo);
    s.lo += t.lo;
    quick_two_sum(s.hi, s.lo)
}

fn dd_neg(a: Dd) -> Dd {
    Dd {
        hi: -a.hi,
        lo: -a.lo,
    }
}

fn dd_mul(a: Dd, b: Dd) -> Dd {
    let mut p = two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    quick_two_sum(p.hi, p.lo)
}

fn dd_mul_d(a: Dd, b: f64) -> Dd {
    let mut p = two_prod(a.hi, b);
    p.lo += a.lo * b;
    quick_two_sum(p.hi, p.lo)
}

fn dd_div_d(a: Dd, b: f64) -> Dd {
    let q1 = a.hi / b;
    let p = two_prod(q1, b);
    let r = ((a.hi - p.hi) - p.lo + a.lo) / b;
    quick_two_sum(q1, r)
}

fn dd_div(a: Dd, b: Dd) -> Dd {
    let q1 = a.hi / b.hi;
    let mut r = dd_add(a, dd_neg(dd_mul_d(b, q1)));
    let q2 = r.hi / b.hi;
    r = dd_add(r, dd_neg(dd_mul_d(b, q2)));
    let q3 = r.hi / b.hi;
    dd_add(quick_two_sum(q1, q2), Dd::new(q3))
}

/// Taylor series of sin for |r| <= pi/4.
fn sin_series(r: Dd) -> Dd {
    let r2 = dd_mul(r, r);
    let mut term = r;
    let mut sum = r;
    let mut n = 3;
    while n <= 33 {
        term = dd_div_d(dd_mul(term, r2), -(((n - 1) * n) as f64));
        sum = dd_add(sum, term);
        n += 2;
    }
    sum
}

/// Taylor series of cos for |r| <= pi/4.
fn cos_series(r: Dd) -> Dd {
    let r2 = dd_mul(r, r);
    let mut term = Dd::new(1.0);
    let mut sum = Dd::new(1.0);
    let mut n = 2;
    while n <= 32 {
        term = dd_div_d(dd_mul(term, r2), -(((n - 1) * n) as f64));
        sum = dd_add(sum, term);
        n += 2;
    }
    sum
}

/// x = k * pi / 2 + r for |x| < 2^20, with pi / 2 in four parts (fdlibm's
/// pio2_1, pio2_2, pio2_3 and pio2_3t; the first three have 33 significant
/// bits, so k times each is exact). Returns r and k mod 4.
fn reduce_quadrant(x: f64) -> Option<(Dd, u32)> {
    // x is finite here (the callers handle NaN and infinities)
    if x.abs() >= 1048576.0 {
        return None;
    }
    // round-half-even, as C nearbyint in the default rounding mode;
    // FRAC_2_PI is fdlibm's invpio2 (6.36619772367581382433e-01)
    let k = (x * std::f64::consts::FRAC_2_PI).round_ties_even();
    let mut acc = Dd::new(x);
    acc = dd_add(acc, Dd::new(-k * 1.570_796_326_734_125_6e0));
    acc = dd_add(acc, Dd::new(-k * 6.077_100_506_303_966e-11));
    acc = dd_add(acc, Dd::new(-k * 2.022_266_248_711_166_5e-21));
    acc = dd_add(acc, dd_neg(two_prod(k, 8.478_427_660_368_9e-32)));
    let q = (k as i64).rem_euclid(4) as u32;
    Some((acc, q))
}

/// `Math.sin`, correctly rounded for |x| < 2^20 (the platform function
/// beyond, where the reference never evaluates it).
pub fn sin(x: f64) -> f64 {
    if x == 0.0 {
        return x;
    }
    if !x.is_finite() {
        return f64::NAN;
    }
    let Some((r, quadrant)) = reduce_quadrant(x) else {
        return x.sin();
    };
    match quadrant {
        0 => sin_series(r).hi,
        1 => cos_series(r).hi,
        2 => -sin_series(r).hi,
        _ => -cos_series(r).hi,
    }
}

/// `Math.cos`, correctly rounded for |x| < 2^20 (the platform function
/// beyond).
pub fn cos(x: f64) -> f64 {
    if !x.is_finite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let Some((r, quadrant)) = reduce_quadrant(x) else {
        return x.cos();
    };
    match quadrant {
        0 => cos_series(r).hi,
        1 => -sin_series(r).hi,
        2 => -cos_series(r).hi,
        _ => sin_series(r).hi,
    }
}

/// `Math.log`, correctly rounded: log(x) = e ln 2 + 2 atanh((m - 1) / (m + 1))
/// with x = m 2^e.
pub fn log(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    if x == 1.0 {
        return 0.0;
    }
    let (mut m, mut e) = frexp(x);
    if m < std::f64::consts::FRAC_1_SQRT_2 {
        m *= 2.0;
        e -= 1;
    }
    let s = dd_div(Dd::new(m - 1.0), two_sum(m, 1.0));
    let s2 = dd_mul(s, s);
    let mut power = s;
    let mut sum = s;
    let mut n = 3;
    while n <= 61 {
        power = dd_mul(power, s2);
        sum = dd_add(sum, dd_div_d(power, n as f64));
        n += 2;
    }
    let mut result = dd_mul_d(sum, 2.0);
    if e != 0 {
        let ln2 = Dd {
            hi: std::f64::consts::LN_2,
            lo: 2.319_046_813_846_299_6e-17,
        };
        result = dd_add(dd_mul_d(ln2, e as f64), result);
    }
    result.hi
}

/// C frexp for a finite non-zero double: x = m 2^e with 0.5 <= |m| < 1.
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let exp = ((bits >> 52) & 0x7ff) as i32;
    if exp == 0 {
        // subnormal: scale into the normal range first
        let (m, e) = frexp(x * f64::from_bits(0x4350_0000_0000_0000)); // 2^54
        return (m, e - 54);
    }
    let m = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (m, exp - 1022)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_float_matches_javascript() {
        assert_eq!(parse_float("1.5abc"), 1.5);
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
        assert!(parse_float("nan").is_nan());
        assert_eq!(parse_float("1e400"), f64::INFINITY);
        assert_eq!(parse_float("-1e-400").to_bits(), (-0.0f64).to_bits());
        assert_eq!(parse_float("-0").to_bits(), (-0.0f64).to_bits());
        assert_eq!(parse_float("\u{a0} 7"), 7.0);
    }

    #[test]
    fn parse_int_matches_javascript() {
        assert_eq!(parse_int10("12abc"), 12.0);
        assert_eq!(parse_int10("-1"), -1.0);
        assert_eq!(parse_int10("+5"), 5.0);
        assert_eq!(parse_int10("1.5"), 1.0);
        assert_eq!(parse_int10("1e3"), 1.0);
        assert!(parse_int10("").is_nan());
        assert!(parse_int10("x1").is_nan());
        assert_eq!(parse_int10("-0").to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn round_and_conversions() {
        assert_eq!(math_round(-2.5), -2.0);
        assert_eq!(math_round(2.5), 3.0);
        assert_eq!(math_round(-0.3).to_bits(), (-0.0f64).to_bits());
        assert_eq!(math_round(0.49999999999999994), 0.0);
        assert_eq!(to_uint32(-1.0), 4294967295);
        assert_eq!(to_uint32(4294967296.0 + 5.0), 5);
        assert_eq!(to_uint32(f64::NAN), 0);
        assert_eq!(to_int32(4294967295.0), -1);
    }

    #[test]
    fn number_to_string_matches_javascript() {
        for (value, text) in [
            (0.5, "0.5"),
            (1.0, "1"),
            (-0.0, "0"),
            (123.456, "123.456"),
            (1e21, "1e+21"),
            (1e20, "100000000000000000000"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1.5e-7, "1.5e-7"),
            (0.1 + 0.2, "0.30000000000000004"),
            (-2.5e-10, "-2.5e-10"),
            (123456789012345680000.0, "123456789012345680000"),
            (f64::INFINITY, "Infinity"),
            (f64::NAN, "NaN"),
        ] {
            assert_eq!(number_to_string(value), text, "{value:e}");
        }
    }

    /// Chromium 153's Math.sin / Math.cos / Math.log for arguments where
    /// the platform libm (macOS) returns a different double: (function,
    /// argument bits, Chromium result bits).
    #[test]
    fn transcendentals_match_chromium() {
        enum Op {
            Sin,
            Cos,
            Log,
        }
        const CASES: &[(Op, u64, u64)] = &[
            (Op::Sin, 0x403394c4bb555be7, 0x3fe560944807c14c), // sin(1.9581126888599012e1)
            (Op::Sin, 0x4014bde89b7bcb7b, 0xbfec7c4ee546c9c6), // sin(5.185457639151418e0)
            (Op::Sin, 0x4014ff5f37f4a5ec, 0xbfeb7ef685c6d898), // sin(5.249386667544496e0)
            (Op::Sin, 0x4074c79c0a140168, 0xbfe0441b7dee1897), // sin(3.3247559554876943e2)
            (Op::Sin, 0xc08f3dee67d3fc63, 0xbfe4fa7a7d330c9b), // sin(-9.997414089738271e2)
            (Op::Sin, 0x4076c4717714096b, 0xbfc2c111cdb61859), // sin(3.642777014525115e2)
            (Op::Cos, 0x4033991b35776c4a, 0x3fe772472c1d7c76), // cos(1.9598071424156196e1)
            (Op::Cos, 0x4014ff5f37f4a5ec, 0x3fe05eb0133dea20), // cos(5.249386667544496e0)
            (Op::Cos, 0x400f748eb1d8cdbb, 0xbfe684099e03485f), // cos(3.9319127935684413e0)
            (Op::Cos, 0x406feaf64fc80e82, 0xbfe48c147edfd1cd), // cos(2.553425673396451e2)
            (Op::Cos, 0xc08367d65d73fadb, 0x3fdf876bd62d8c33), // cos(-6.209796704350405e2)
            (Op::Cos, 0xc08df78d619bfb62, 0xbfe73b96003d0ad2), // cos(-9.589440338312236e2)
            (Op::Log, 0x3fe18a6fcf5f5f5f, 0xbfe33d18c3c140cf), // log(5.48149018311033e-1)
            (Op::Log, 0x3fdd73ea579fe900, 0xbfe8d5c3afb94ea7), // log(4.601999145230735e-1)
            (Op::Log, 0x3fe4b8b283daed2e, 0xbfdbcfeb3260bd17), // log(6.475460601135745e-1)
            (Op::Log, 0x3fdf514ad1988182, 0xbfe6dedbfcaafa87), // log(4.893366858323348e-1)
            (Op::Log, 0x3fe3798dd14caab2, 0xbfdfc8842b27027b), // log(6.085881317024671e-1)
            (Op::Log, 0x3fda803837bb7035, 0xbfec36ef75e49a59), // log(4.140759033450679e-1)
        ];
        for (op, argument, expected) in CASES {
            let x = f64::from_bits(*argument);
            let got = match op {
                Op::Sin => sin(x),
                Op::Cos => cos(x),
                Op::Log => log(x),
            };
            assert_eq!(got.to_bits(), *expected, "{x:e}");
        }
    }

    #[test]
    fn transcendental_special_values() {
        assert_eq!(sin(0.0), 0.0);
        assert_eq!(sin(-0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(cos(0.0), 1.0);
        assert!(sin(f64::INFINITY).is_nan());
        assert_eq!(log(1.0), 0.0);
        assert_eq!(log(0.0), f64::NEG_INFINITY);
        assert!(log(-1.0).is_nan());
        assert_eq!(log(std::f64::consts::E), 1.0);
    }
}
