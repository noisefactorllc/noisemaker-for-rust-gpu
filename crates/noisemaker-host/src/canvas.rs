//! The traced overlays' canvas operations: [`CallRecorder`] logs the
//! reference's canvas operations (the differential test compares the log
//! with the one the reference writes), and [`parse_rgba_style`] reads the
//! stroke styles the tracer assigns as Blink stores them. The canvas that
//! rasterizes the strokes is [`crate::raster::RasterCanvas`].

use crate::worm::Canvas2d;
use std::fmt::Write as _;

// ------------------------------------------------------------ call recorder

/// A [`Canvas2d`] that records every operation as one line of text, in the
/// format `tools/reference-host.mjs worm` writes for the reference:
///
/// ```text
/// clearRect <x> <y> <w> <h>
/// lineCap <value>
/// lineJoin <value>
/// lineWidth <n>
/// strokeStyle <value>
/// beginPath
/// moveTo <x> <y>
/// lineTo <x> <y>
/// stroke
/// update <textureName>
/// ```
///
/// Numbers are the 16 hexadecimal digits of their IEEE double bits, so the
/// logs compare bit for bit. `update` lines are written by the overlay
/// driver where the reference calls `updateTexture(name, canvas)`.
#[derive(Clone, Debug)]
pub struct CallRecorder {
    width: u32,
    height: u32,
    lines: Vec<String>,
}

/// The log encoding of a number: its IEEE double bits in hexadecimal.
pub fn number_bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

impl CallRecorder {
    /// A recorder standing in for a `width` x `height` canvas.
    pub fn new(width: u32, height: u32) -> CallRecorder {
        CallRecorder {
            width,
            height,
            lines: Vec::new(),
        }
    }

    /// The recorded lines.
    pub fn lines(&self) -> &[String] {
        &self.lines
    }

    /// Appends a free-form line (the overlay driver's `update` events).
    pub fn note(&mut self, line: String) {
        self.lines.push(line);
    }
}

impl Canvas2d for CallRecorder {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn clear_rect(&mut self, x: f64, y: f64, w: f64, h: f64) {
        let mut line = String::from("clearRect");
        for v in [x, y, w, h] {
            let _ = write!(line, " {}", number_bits(v));
        }
        self.lines.push(line);
    }
    fn set_line_cap(&mut self, cap: &str) {
        self.lines.push(format!("lineCap {cap}"));
    }
    fn set_line_join(&mut self, join: &str) {
        self.lines.push(format!("lineJoin {join}"));
    }
    fn set_line_width(&mut self, width: f64) {
        self.lines.push(format!("lineWidth {}", number_bits(width)));
    }
    fn set_stroke_style(&mut self, style: &str) {
        self.lines.push(format!("strokeStyle {style}"));
    }
    fn begin_path(&mut self) {
        self.lines.push("beginPath".into());
    }
    fn move_to(&mut self, x: f64, y: f64) {
        self.lines
            .push(format!("moveTo {} {}", number_bits(x), number_bits(y)));
    }
    fn line_to(&mut self, x: f64, y: f64) {
        self.lines
            .push(format!("lineTo {} {}", number_bits(x), number_bits(y)));
    }
    fn stroke(&mut self) {
        self.lines.push("stroke".into());
    }
}

// ------------------------------------------------------------ CSS colours

/// A CSS `<number>` token at the start of `s`: the value and the rest.
fn css_number(s: &str) -> Option<(f64, &str)> {
    let b = s.as_bytes();
    let mut i = 0;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    let mut digits = i > int_start;
    if i < b.len() && b[i] == b'.' && i + 1 < b.len() && b[i + 1].is_ascii_digit() {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        digits = true;
    }
    if !digits {
        return None;
    }
    if i < b.len() && (b[i] == b'e' || b[i] == b'E') {
        let mut k = i + 1;
        if k < b.len() && (b[k] == b'+' || b[k] == b'-') {
            k += 1;
        }
        let exp_start = k;
        while k < b.len() && b[k].is_ascii_digit() {
            k += 1;
        }
        if k > exp_start {
            i = k;
        }
    }
    let value: f64 = s[..i].parse().ok()?;
    Some((value, &s[i..]))
}

fn skip_css_space(s: &str) -> &str {
    s.trim_start_matches([' ', '\t', '\n', '\r', '\u{c}'])
}

/// An 8-bit straight-alpha colour as Blink stores a canvas style given as
/// legacy `rgb()`/`rgba()` with comma-separated numbers (the only form the
/// reference's tracer assigns): channels clamp to 0..255 and round, alpha
/// clamps to 0..1 and becomes floor(a * 255 + 0.5). Returns `None` for any
/// other syntax (an invalid assignment, which the canvas ignores).
pub fn parse_rgba_style(style: &str) -> Option<[u8; 4]> {
    let s = style.trim_matches([' ', '\t', '\n', '\r', '\u{c}']);
    let lower = s.get(..5).map(|p| p.to_ascii_lowercase());
    let (body, has_alpha_name) = if lower.as_deref() == Some("rgba(") {
        (&s[5..], true)
    } else if s.get(..4).map(|p| p.to_ascii_lowercase()).as_deref() == Some("rgb(") {
        (&s[4..], false)
    } else {
        return None;
    };
    let _ = has_alpha_name; // rgb() and rgba() are aliases
    let mut rest = body;
    let mut channels = [0u8; 3];
    for (i, channel) in channels.iter_mut().enumerate() {
        rest = skip_css_space(rest);
        let (value, r) = css_number(rest)?;
        rest = skip_css_space(r);
        if rest.starts_with('%') {
            return None; // percentages are never written by the tracer
        }
        *channel = (value.clamp(0.0, 255.0) + 0.5).floor() as u8;
        if i < 2 {
            rest = rest.strip_prefix(',')?;
        }
    }
    let mut alpha = 255u8;
    if let Some(r) = rest.strip_prefix(',') {
        let (value, r) = css_number(skip_css_space(r))?;
        rest = skip_css_space(r);
        if rest.starts_with('%') {
            return None;
        }
        alpha = (value.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
    }
    if skip_css_space(rest) != ")" {
        return None;
    }
    Some([channels[0], channels[1], channels[2], alpha])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tracer_styles() {
        assert_eq!(
            parse_rgba_style("rgba(255, 255, 255, 1)"),
            Some([255, 255, 255, 255])
        );
        assert_eq!(
            parse_rgba_style("rgba(12, 34, 56, 0)"),
            Some([12, 34, 56, 0])
        );
        assert_eq!(parse_rgba_style("rgba(1, 2, 3, 0.5)"), Some([1, 2, 3, 128]));
        assert_eq!(parse_rgba_style("rgba(1, 2, 3, 1e-7)"), Some([1, 2, 3, 0]));
        assert_eq!(parse_rgba_style("rgba(1, 2, 3, NaN)"), None);
        assert_eq!(parse_rgba_style("rgb(1, 2, 3)"), Some([1, 2, 3, 255]));
    }

    #[test]
    fn records_number_bits() {
        let mut r = CallRecorder::new(4, 4);
        r.move_to(1.5, -0.0);
        assert_eq!(r.lines(), ["moveTo 3ff8000000000000 8000000000000000"]);
    }
}
