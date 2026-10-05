//! The 2D-canvas text of filter/text: port of the reference demo host's
//! text canvas (demo/shaders/lib/demo-ui.js `_renderTextToCanvas` and
//! `_updateTextTexture`), which draws the effect's text globals on a canvas
//! and uploads it as `textTex_step_<N>` with
//! `updateTextureFromSource(texId, canvas, { flipY: true })`.
//!
//! # The demo host
//!
//! - The canvas is `resolution x resolution`, resolution = the renderer's
//!   width ([`demo_canvas_size`]), cleared transparent.
//! - The text is split on `"\n"`; the font size is `Math.round(size *
//!   canvas.height)` px and the line height 1.2 x the font size.
//! - `ctx.font = `${fontSize}px ${font}``, `textAlign = justify`,
//!   `textBaseline = 'middle'`, `fillStyle = rgba(round(r * 255), round(g *
//!   255), round(b * 255), 1)` from the colour (`#rrggbb`, or an `[r, g, b]`
//!   array in 0..1; anything else is white).
//! - It translates to (posX * width, posY * height), rotates by `rotation`
//!   degrees and fills line i at y = -(lines - 1) * lineHeight / 2 + i *
//!   lineHeight.
//!
//! # The canvas (Blink and Skia in Chromium)
//!
//! - Invalid `font`, `textAlign` and `fillStyle` assignments keep the
//!   previous value (`10px sans-serif`, `start`, black); non-finite
//!   translate, rotate and fillText arguments are ignored ([`css`]).
//! - The canvas inherits `font-variation-settings`, `letter-spacing` and
//!   `-webkit-font-smoothing` from its page ([`HostStyle`]).
//! - Families resolve as Blink resolves them, with per-character fallback
//!   ([`fonts`]); `font-optical-sizing: auto` sets an `opsz` axis to the
//!   font size.
//! - `textBaseline` middle puts the baseline (ascent - descent) / 2 below y,
//!   from the OS/2 typographic ascent and descent normalized to the font
//!   size and rounded to 1/64 px (fonts without them: the rounded font
//!   metrics, with WebKit's 15 % ascent for Times, Helvetica and Courier on
//!   macOS).
//! - Text is shaped with HarfBuzz (rustybuzz) word by word as Blink's
//!   CachingWordShaper splits it (no kerning across spaces), per
//!   bidirectional run and font; advances are the platform's unhinted
//!   advances in 16.16 plus HarfBuzz-scaled GPOS adjustments, kerning
//!   variation deltas unrounded ([`kern`]); the system UI font gets its
//!   `trak` tracking, as CoreText reports it; letter spacing follows each
//!   character.
//! - `center` / `end` / `right` shift by the advance width.
//! - The drawing matrix is Skia's float `SkMatrix` (`setRotate` snaps
//!   near-zero sines and cosines, so quarter turns are axis-aligned).
//!   Glyphs are placed with Skia's subpixel positioning: quarter pixels
//!   along the baseline and whole pixels across it when the matrix is
//!   axis-aligned, quarter pixels on both axes otherwise.
//! - Glyphs are drawn as Skia's GPU text draws them, by the run's device
//!   text size (`size * getMaxScale()`): below 162 px as glyph masks
//!   rasterized as CoreGraphics does with font smoothing off ([`raster`]);
//!   from 162 px as distance fields of CoreGraphics' smoothed masks
//!   ([`sdf`]); from 256 px, and any glyph too large for the 256 px atlas,
//!   as paths tessellated on the GPU and filled with 4x multisampling
//!   ([`msaa`]). The fill colour is composited source-over.
//!
//! # Measured against Chromium 153 on macOS 26
//!
//! `tests/text_raster.rs` compares 53 captured cases
//! (`tools/reference-host.mjs text`) and the golden minter's
//! `textTex_step_N` captures.
//!
//! - Glyph masks: the default text (Nunito, 26 px) differs in 37 of 65536
//!   pixels, each by 1 in alpha (alpha SSIM 1.00000); Nunito, Times,
//!   Helvetica, Arial, Georgia, Courier, Apple Chancery and Papyrus from 19
//!   to 161 px, in every alignment and rotation, differ by at most 1 in
//!   alpha in 34 of 38 such cases; the others differ by at most 3 (20
//!   pixels of punctuation), 5 (88 pixels of composite accented glyphs), 2
//!   (one pixel of rotated text) and 35 (4 pixels of one curve whose
//!   flattening falls on the other side of the subdivision threshold).
//! - Distance fields (162 to 255 px, 5 cases): 0.7 to 2.1 % of the drawn
//!   pixels differ by more than 1 in alpha, by at most 14 to 48 (alpha SSIM
//!   0.99998 to 1.00000); edges agree within 0.02 px on average. The
//!   residual is the smoothing model at a few glyph features (a straight
//!   edge running into a curve, sharp joins), where it is up to 65 levels
//!   off CoreGraphics' own smoothed mask, and the GPU's half-precision
//!   arithmetic.
//! - Paths (from 256 px): 9 and 24 pixels differ, each by one of four
//!   samples (64 in alpha), out of 27744 and 64078 drawn: samples within
//!   rounding of an edge. Probes from 256 to 330 px: 6 to 14 pixels each;
//!   the resolve's rounding (half up) was read off a colour whose channels
//!   tell it from round-half-even.
//!
//! Residual cases: text below 19 px, which CoreGraphics grid-fits
//! vertically (alpha SSIM 0.94 to 0.96); characters Chromium draws with
//! PingFang SC, whose `hvgl` outlines no available parser reads (symbols
//! alpha SSIM 0.95, CJK 0.79, drawn with Lucida Grande and Hiragino Sans
//! instead); `system-ui` (SF, outlined by skrifa with rounded variation
//! deltas; alpha SSIM 0.99999); Zapfino (AAT shaping, overlapping swash
//! contours) at 128 px: ink equal, glyph centroids up to 0.2 px apart, 12 %
//! of the drawn pixels off by more than 1, cause not identified.

pub mod css;
pub mod fonts;
pub mod kern;
pub mod msaa;
pub mod raster;
pub mod sdf;

pub use css::{CanvasFont, FontFamily, GenericFamily, parse_canvas_font};
pub use fonts::{ResolvedFace, TextFonts};

use crate::Rgba8Image;
use crate::canvas::parse_rgba_style;
use crate::js;
use rustybuzz::ttf_parser;

/// The text colour as the demo's `_hexToRgb` receives it.
#[derive(Clone, Debug, PartialEq)]
pub enum TextColor {
    /// A `#rrggbb` (or `rrggbb`) string; anything else reads as white.
    Hex(String),
    /// An array of channels in 0..1 (its first three are used).
    Array(Vec<f64>),
}

impl Default for TextColor {
    fn default() -> TextColor {
        TextColor::Hex("#ffffff".into())
    }
}

/// The filter/text globals the demo host reads (`textState`). Numbers are
/// JavaScript Numbers: an absent value is NaN.
#[derive(Clone, Debug, PartialEq)]
pub struct TextParams {
    /// `text` (the demo draws `String(text || '')`).
    pub text: String,
    /// `font`: the family list written after the size.
    pub font: String,
    /// `size`: font size as a fraction of the canvas height.
    pub size: f64,
    /// `posX`: x of the text origin as a fraction of the canvas width.
    pub pos_x: f64,
    /// `posY`: y of the text origin as a fraction of the canvas height.
    pub pos_y: f64,
    /// `rotation` in degrees (clockwise on screen).
    pub rotation: f64,
    /// `color`.
    pub color: TextColor,
    /// `justify`: the canvas `textAlign` value.
    pub justify: String,
}

impl Default for TextParams {
    /// The filter/text definition defaults.
    fn default() -> TextParams {
        TextParams {
            text: "Hello World".into(),
            font: "Nunito".into(),
            size: 0.1,
            pos_x: 0.5,
            pos_y: 0.5,
            rotation: 0.0,
            color: TextColor::default(),
            justify: "center".into(),
        }
    }
}

/// Text properties the canvas element inherits from its page. Blink
/// resolves a canvas `font` against the canvas element's computed style,
/// and the `font` shorthand does not reset `font-variation-settings`,
/// `letter-spacing` or `-webkit-font-smoothing`, so canvas text takes them
/// from the page. The demo appends its text canvas to `<body>`, whose style
/// (demo/common-layout.css, handfish index.css) sets them:
///
/// - `font-variation-settings: "wght" 580`: variable faces with a `wght`
///   axis (the bundled Nunito, SF) are drawn at wght 580 while the font
///   matching weight stays 400;
/// - `letter-spacing: 0.01em` of the body's 16 px, i.e. 0.16 px after each
///   character (with optional ligatures off, as Blink does for non-zero
///   letter spacing);
/// - `-webkit-font-smoothing: antialiased`: grayscale coverage without the
///   platform's stem darkening, which is what the outline rasterizer
///   computes.
///
/// A plain page with exactly these three properties draws the demo's text
/// canvas byte for byte (measured in Chromium 153).
#[derive(Clone, Debug, PartialEq)]
pub struct HostStyle {
    /// `font-variation-settings` axis values (OpenType tag, value).
    pub variation_settings: Vec<([u8; 4], f32)>,
    /// Computed `letter-spacing` in CSS pixels.
    pub letter_spacing: f32,
}

impl HostStyle {
    /// The reference demo page's inherited style.
    pub fn demo() -> HostStyle {
        HostStyle {
            variation_settings: vec![(*b"wght", 580.0)],
            letter_spacing: 0.01 * 16.0,
        }
    }

    /// A canvas whose page sets none of these properties.
    pub fn plain() -> HostStyle {
        HostStyle {
            variation_settings: Vec::new(),
            letter_spacing: 0.0,
        }
    }
}

impl Default for HostStyle {
    /// [`HostStyle::demo`].
    fn default() -> HostStyle {
        HostStyle::demo()
    }
}

/// The demo host's text canvas size for a renderer of `render_width` x
/// `render_height`: square, `render_width` on both sides.
pub fn demo_canvas_size(render_width: u32, _render_height: u32) -> (u32, u32) {
    (render_width, render_width)
}

/// The demo's `_hexToRgb`: channels in 0..1.
fn hex_to_rgb(color: &TextColor) -> Vec<f64> {
    match color {
        TextColor::Array(values) => values.iter().take(3).copied().collect(),
        TextColor::Hex(s) => {
            let s = s.strip_prefix('#').unwrap_or(s);
            let ok = s.len() == 6 && s.chars().all(|c| c.is_ascii_hexdigit());
            if !ok {
                return vec![1.0, 1.0, 1.0];
            }
            (0..3)
                .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap() as f64 / 255.0)
                .collect()
        }
    }
}

/// The fill style string the demo assigns.
fn fill_style(color: &TextColor) -> String {
    let rgb = hex_to_rgb(color);
    // `${Math.round(textColor[i] * 255)}`; a missing channel is undefined
    let channel = |i: usize| {
        js::number_to_string(js::math_round(
            rgb.get(i).copied().unwrap_or(f64::NAN) * 255.0,
        ))
    };
    format!("rgba({}, {}, {}, 1)", channel(0), channel(1), channel(2))
}

/// `textAlign` after `ctx.textAlign = justify` (invalid values are ignored).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Align {
    Start,
    End,
    Left,
    Right,
    Center,
}

fn text_align(value: &str) -> Align {
    match value {
        "start" => Align::Start,
        "end" => Align::End,
        "left" => Align::Left,
        "right" => Align::Right,
        "center" => Align::Center,
        _ => Align::Start,
    }
}

/// The canvas drawing matrix as Skia holds it: an `SkMatrix` in floats,
/// `[sx kx tx; ky sy ty]`. Blink forwards `translate(x, y)` and
/// `rotate(angle)` to `SkCanvas::translate(float, float)` and
/// `SkCanvas::rotate(float degrees)`; `SkMatrix::setRotate` snaps a sine or
/// cosine within 1/4096 of zero to zero, so quarter turns are exactly
/// axis-aligned.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SkMatrix {
    sx: f32,
    kx: f32,
    tx: f32,
    ky: f32,
    sy: f32,
    ty: f32,
}

impl SkMatrix {
    const IDENTITY: SkMatrix = SkMatrix {
        sx: 1.0,
        kx: 0.0,
        tx: 0.0,
        ky: 0.0,
        sy: 1.0,
        ty: 0.0,
    };

    /// `preTranslate(dx, dy)`.
    fn translate(self, dx: f32, dy: f32) -> SkMatrix {
        SkMatrix {
            tx: self.tx + (self.sx * dx + self.kx * dy),
            ty: self.ty + (self.ky * dx + self.sy * dy),
            ..self
        }
    }

    /// `preRotate(degrees)`: `this * R` with R from `setRotate(degrees)`.
    fn rotate(self, degrees: f32) -> SkMatrix {
        let radians = degrees * (std::f32::consts::PI / 180.0);
        let snap = |v: f32| if v.abs() <= 1.0 / 4096.0 { 0.0 } else { v };
        let (sin, cos) = (snap(radians.sin()), snap(radians.cos()));
        // setConcat: products summed in doubles, rounded once
        let mul =
            |a: f32, b: f32, c: f32, d: f32| (a as f64 * b as f64 + c as f64 * d as f64) as f32;
        SkMatrix {
            sx: mul(self.sx, cos, self.kx, sin),
            kx: mul(self.sx, -sin, self.kx, cos),
            ky: mul(self.ky, cos, self.sy, sin),
            sy: mul(self.ky, -sin, self.sy, cos),
            ..self
        }
    }

    /// `getMaxScale()`: the larger singular value of the 2x2 part, in
    /// floats (an orthogonal matrix takes the larger column length).
    fn max_scale(&self) -> f32 {
        if self.kx == 0.0 && self.ky == 0.0 {
            return self.sx.abs().max(self.sy.abs());
        }
        let a = self.sx * self.sx + self.ky * self.ky;
        let b = self.sx * self.kx + self.sy * self.ky;
        let c = self.kx * self.kx + self.sy * self.sy;
        let b2 = b * b;
        let nearly_zero = 1.0f32 / 4096.0;
        let squared = if b2 <= nearly_zero * nearly_zero {
            a.max(c)
        } else {
            (a + c) * 0.5 + ((a - c) * (a - c) + 4.0 * b2).sqrt() * 0.5
        };
        squared.max(0.0).sqrt()
    }

    /// `mapXY`: (x * sx + y * kx) + tx, in floats.
    fn map(&self, x: f32, y: f32) -> (f32, f32) {
        (
            x * self.sx + y * self.kx + self.tx,
            x * self.ky + y * self.sy + self.ty,
        )
    }
}

/// One shaped glyph of a line: the face it comes from, its id and its
/// position relative to the line origin in CSS pixels (y down).
struct PlacedGlyph {
    face: usize,
    glyph: u16,
    x: f32,
    y: f32,
}

/// HarfBuzz positions are 16.16 fixed point in Blink.
fn to_16_16(v: f64) -> i64 {
    (v * 65536.0) as i64 // ClampTo<int>: truncation
}

/// A shaped line: glyphs in visual order and the advance width.
struct ShapedLine {
    glyphs: Vec<PlacedGlyph>,
    width: f32,
}

/// A resolved face as the canvas instances it: optical size from the font
/// size (`font-optical-sizing: auto`), then the inherited
/// `font-variation-settings`.
fn prepare_face(face: ResolvedFace, size: f64, style: &HostStyle) -> ResolvedFace {
    face.with_optical_size(size as f32)
        .with_variation_settings(&style.variation_settings)
}

/// Blink `Character::IsCJKIdeographOrSymbol` (the ranges that make a
/// character its own shaping word).
fn is_cjk_ideograph_or_symbol(c: char) -> bool {
    let u = c as u32;
    if u < 0x2C7 {
        return false;
    }
    is_cjk_ideograph(c)
        || matches!(u,
            0x2C7 | 0x2CA | 0x2CB | 0x2D9 | 0x2EA | 0x2EB
            | 0x2015 | 0x2016 | 0x2025 | 0x2026 | 0x2030 | 0x203B | 0x203C | 0x2042
            | 0x2047..=0x2049 | 0x2051 | 0x20DD | 0x20DE | 0x2100 | 0x2103 | 0x2105
            | 0x2109 | 0x210A | 0x2113 | 0x2116 | 0x2121 | 0x212B | 0x213B
            | 0x2150..=0x2152 | 0x2160..=0x216B | 0x2170..=0x217B | 0x217F | 0x2189
            | 0x2307 | 0x2312 | 0x23BE..=0x23CC | 0x23CE | 0x2423 | 0x2460..=0x2492
            | 0x249C..=0x24FF | 0x25A0..=0x25A2 | 0x25AA | 0x25AB | 0x25B1..=0x25B3
            | 0x25B6 | 0x25B7 | 0x25BC | 0x25BD | 0x25C0 | 0x25C1 | 0x25C6 | 0x25C7
            | 0x25C9 | 0x25CB | 0x25CC | 0x25CE..=0x25D3 | 0x25E2..=0x25E5 | 0x25EF
            | 0x2600..=0x2603 | 0x2605 | 0x2606 | 0x260E | 0x2616 | 0x2617 | 0x261D
            | 0x2640 | 0x2642 | 0x2660..=0x266F | 0x26A0 | 0x26BD | 0x26BE
            | 0x26C4 | 0x26C5 | 0x26CE | 0x2702 | 0x2713 | 0x271A | 0x2756 | 0x2763
            | 0x2764 | 0x2768..=0x2775 | 0x2B1A | 0x2B50 | 0x2B55
            | 0x2E80..=0x2FDF | 0x2FF0..=0x303F | 0x3040..=0x31FF | 0x3200..=0x33FF
            | 0xA960..=0xA97F | 0xAC00..=0xD7AF | 0xF900..=0xFAFF | 0xFE10..=0xFE1F
            | 0xFE30..=0xFE6F | 0xFF00..=0xFFEF | 0x1B000..=0x1B16F | 0x1F000..=0x1FAFF)
}

fn is_cjk_ideograph(c: char) -> bool {
    matches!(c as u32,
        0x2E80..=0x2FDF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF
        | 0x20000..=0x2FA1F | 0x30000..=0x323AF)
}

/// Characters a CJK word does not end before: marks, modifier letters,
/// ZWJ, emoji modifiers and variation selectors.
fn extends_cjk_word(c: char) -> bool {
    let u = c as u32;
    matches!(u,
        0x0300..=0x036F | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF
        | 0x3099..=0x309C | 0x30FC | 0xFE20..=0xFE2F | 0x200D | 0xFE00..=0xFE0F
        | 0x1F3FB..=0x1F3FF | 0xE0100..=0xE01EF)
}

/// Blink `CachingWordShapeIterator::NextWordEndIndex`: a run of text split
/// into the pieces Blink shapes separately: each U+0020 alone, each CJK
/// ideograph or symbol (with the marks and joiners after it) alone, and the
/// text between them. Byte ranges of `text`.
fn word_ranges(text: &str) -> Vec<std::ops::Range<usize>> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut words = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = i;
        let c = chars[i].1;
        if i + 1 == chars.len() || c == ' ' {
            i += 1;
        } else if is_cjk_ideograph_or_symbol(c) {
            i += 1;
            while i < chars.len() && extends_cjk_word(chars[i].1) {
                i += 1;
            }
        } else {
            i += 1;
            while i < chars.len() && chars[i].1 != ' ' && !is_cjk_ideograph_or_symbol(chars[i].1) {
                i += 1;
            }
        }
        let end = chars.get(i).map_or(text.len(), |(b, _)| *b);
        words.push(chars[start].0..end);
    }
    words
}

/// Blink `Character::TreatAsZeroWidthSpace`: characters that take no
/// letter spacing.
fn treat_as_zero_width_space(c: char) -> bool {
    let u = c as u32;
    u < 0x20
        || (0x7F..0xA0).contains(&u)
        || u == 0xAD
        || (0x200B..=0x200F).contains(&u)
        || (0x202A..=0x202E).contains(&u)
        || (0x2060..=0x2064).contains(&u)
        || u == 0xFEFF
        || u == 0xFFFC
}

/// Characters canvas text draws as U+0020: ASCII whitespace (tab, line
/// feed, vertical tab, form feed, carriage return), as the HTML text
/// preparation step requires, and the line and paragraph separators
/// (measured: Chromium measures each as a space).
const CANVAS_SPACES: [char; 7] = ['\t', '\n', '\u{b}', '\u{c}', '\r', '\u{2028}', '\u{2029}'];

/// Blink's canvas text space normalization ([`CANVAS_SPACES`]).
fn normalize_spaces(text: &str) -> std::borrow::Cow<'_, str> {
    if text.contains(CANVAS_SPACES) {
        text.replace(CANVAS_SPACES, " ").into()
    } else {
        text.into()
    }
}

/// Default-ignorable format characters (zero-width spaces and joiners,
/// directional marks and isolates, the BOM): they draw nothing and take
/// the font of the text around them.
fn is_default_ignorable(c: char) -> bool {
    matches!(c,
        '\u{00AD}' | '\u{034F}' | '\u{061C}' | '\u{180B}'..='\u{180F}' | '\u{200B}'..='\u{200F}'
        | '\u{202A}'..='\u{202E}' | '\u{2060}'..='\u{206F}' | '\u{FE00}'..='\u{FE0F}'
        | '\u{FEFF}' | '\u{E0000}'..='\u{E0FFF}')
}

/// Font of each byte of `text` (an index into `faces`): Blink's font
/// fallback per character. A character takes the first face of the list
/// that maps it, else the platform fallback face (appended to `faces`).
/// Combining marks and default-ignorable characters stay in the preceding
/// character's font when it maps them, so a cluster is not split; a
/// character no font maps takes its neighbour's font (the primary font's
/// .notdef at the start of a line).
fn font_runs(
    text: &str,
    size: f64,
    faces: &mut Vec<ResolvedFace>,
    fonts: &TextFonts,
    style: &HostStyle,
) -> Vec<usize> {
    use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};
    let mut choice = Vec::with_capacity(text.len());
    let mut fallback: Vec<(char, Option<usize>)> = Vec::new();
    let mut previous: Option<usize> = None;
    for c in text.chars() {
        let attaches =
            is_default_ignorable(c) || c.general_category_group() == GeneralCategoryGroup::Mark;
        let index = match previous {
            Some(p) if attaches && (is_default_ignorable(c) || faces[p].has_char(c)) => Some(p),
            _ => faces.iter().position(|f| f.has_char(c)).or_else(|| {
                if is_default_ignorable(c) {
                    return None;
                }
                if let Some((_, i)) = fallback.iter().find(|(fc, _)| *fc == c) {
                    return *i;
                }
                let found = fonts.fallback_for(c).map(|face| {
                    let face = prepare_face(face, size, style);
                    match faces
                        .iter()
                        .position(|f| f.label == face.label && f.index == face.index)
                    {
                        Some(i) => i,
                        None => {
                            faces.push(face);
                            faces.len() - 1
                        }
                    }
                });
                fallback.push((c, found));
                found
            }),
        };
        let index = index.or(previous).unwrap_or(0);
        previous = Some(index);
        for _ in 0..c.len_utf8() {
            choice.push(index);
        }
    }
    choice
}

/// Shapes one line: bidirectional runs (base direction LTR, the canvas
/// direction "inherit" of an LTR document), split by font, shaped with
/// HarfBuzz; advances as Blink accumulates them.
fn shape_line(
    line: &str,
    size: f64,
    faces: &mut Vec<ResolvedFace>,
    fonts: &TextFonts,
    style: &HostStyle,
) -> ShapedLine {
    let mut out = ShapedLine {
        glyphs: Vec::new(),
        width: 0.0,
    };
    if line.is_empty() || faces.is_empty() {
        return out;
    }
    let font_of = font_runs(line, size, faces, fonts, style);
    // Blink turns optional ligatures off when letter spacing is not zero.
    let features: Vec<rustybuzz::Feature> = if style.letter_spacing != 0.0 {
        [b"liga", b"clig"]
            .iter()
            .map(|tag| rustybuzz::Feature::new(ttf_parser::Tag::from_bytes(tag), 0, ..))
            .collect()
    } else {
        Vec::new()
    };
    let bidi = unicode_bidi::BidiInfo::new(line, Some(unicode_bidi::Level::ltr()));
    let mut pen = 0f32;
    for paragraph in &bidi.paragraphs {
        let (levels, runs) = bidi.visual_runs(paragraph, paragraph.range.clone());
        for run in runs {
            let rtl = levels[run.start].is_rtl();
            // Blink shapes word by word (CachingWordShaper), then each
            // word's characters by font.
            let mut pieces: Vec<(usize, std::ops::Range<usize>)> = Vec::new();
            for word in word_ranges(&line[run.clone()]) {
                let word = run.start + word.start..run.start + word.end;
                let mut start = word.start;
                let mut word_pieces = Vec::new();
                for i in word.clone() {
                    if !line.is_char_boundary(i) {
                        continue;
                    }
                    if font_of[i] != font_of[start] {
                        word_pieces.push((font_of[start], start..i));
                        start = i;
                    }
                }
                word_pieces.push((font_of[start], start..word.end));
                pieces.extend(word_pieces);
            }
            if rtl {
                pieces.reverse();
            }
            for (face_index, range) in pieces {
                let face_ref = &faces[face_index];
                let Some(face) = face_ref.shaping_face() else {
                    continue;
                };
                // CoreText folds the system UI font's tracking into its
                // advances (Blink takes advances from it); other fonts'
                // `trak` tables are not applied.
                let tracking = if face_ref.system_ui {
                    trak_tracking(&face, size as f32)
                } else {
                    0.0
                };
                let upem = face.units_per_em() as f64;
                let mut buffer = rustybuzz::UnicodeBuffer::new();
                buffer.push_str(&line[range.clone()]);
                buffer.guess_segment_properties();
                buffer.set_direction(if rtl {
                    rustybuzz::Direction::RightToLeft
                } else {
                    rustybuzz::Direction::LeftToRight
                });
                let shaped = rustybuzz::shape(&face, &features, buffer);
                let run_text = &line[range.clone()];
                // hb_font scale: the font size in 16.16 (truncated), and
                // HarfBuzz's em scaling of font-unit values.
                let x_scale = to_16_16(size);
                let x_mult = (x_scale << 16) / upem as i64;
                let em_scale = |v: i32| -> i64 { (v as i64 * x_mult + 32768) >> 16 };
                let infos = shaped.glyph_infos();
                // the fractional kerning variation deltas rustybuzz rounded
                // (computed for left-to-right runs, the order kern_residuals
                // pairs glyphs in)
                let residuals = if rtl {
                    vec![0.0; infos.len()]
                } else {
                    let ids: Vec<u16> = infos.iter().map(|i| i.glyph_id as u16).collect();
                    kern::kern_residuals(&face, &ids)
                };
                let em_scalef = |v: f64| -> i64 { (v * x_scale as f64 / upem).round() as i64 };
                for (g, (info, pos)) in infos.iter().zip(shaped.glyph_positions()).enumerate() {
                    let gid = ttf_parser::GlyphId(info.glyph_id as u16);
                    // The advance Blink's HarfBuzz font functions return:
                    // the platform's unhinted advance (hmtx + HVAR) at the
                    // size, as a float, in 16.16; GPOS adjustments are
                    // HarfBuzz-scaled font units.
                    let nominal = face.glyph_hor_advance(gid).unwrap_or(0) as i32;
                    let exact =
                        exact_advance(&face, gid).unwrap_or(nominal as f64) + tracking as f64;
                    let advance_px = (exact * size / upem) as f32;
                    let adjust = pos.x_advance - nominal;
                    let mut advance = (to_16_16(advance_px as f64)
                        + em_scale(adjust)
                        + em_scalef(residuals[g])) as f32
                        / 65536.0;
                    // letter spacing after each character, on the last glyph
                    // of its cluster
                    let cluster_end = infos
                        .get(g + 1)
                        .is_none_or(|next| next.cluster != info.cluster);
                    if style.letter_spacing != 0.0 && cluster_end {
                        let start = info.cluster as usize;
                        let end = infos
                            .iter()
                            .map(|i| i.cluster as usize)
                            .filter(|c| *c > start)
                            .min()
                            .unwrap_or(run_text.len());
                        let characters = run_text[start..end.max(start)]
                            .chars()
                            .filter(|c| !treat_as_zero_width_space(*c))
                            .count();
                        advance += style.letter_spacing * characters as f32;
                    }
                    let x_offset = em_scale(pos.x_offset) as f32 / 65536.0;
                    let y_offset = em_scale(pos.y_offset) as f32 / 65536.0;
                    out.glyphs.push(PlacedGlyph {
                        face: face_index,
                        glyph: info.glyph_id as u16,
                        x: pen + x_offset,
                        y: -y_offset,
                    });
                    pen += advance;
                }
            }
        }
    }
    out.width = pen;
    out
}

/// The normal track (track value 0) of the face's `trak` table at `size`
/// (in CSS pixels, which CoreText takes as points here), in font units:
/// linear between the table's sizes, clamped at its ends. Measured: SF at
/// 26 px adds the +17 units this gives to each advance in Chromium.
fn trak_tracking(face: &rustybuzz::Face<'_>, size: f32) -> f32 {
    let Some(trak) = face.tables().trak else {
        return 0.0;
    };
    let data = trak.horizontal;
    let Some(track) = data.tracks.into_iter().find(|t| t.value == 0.0) else {
        return 0.0;
    };
    let sizes: Vec<f32> = data.sizes.into_iter().map(|f| f.0).collect();
    let values: Vec<f32> = track.values.into_iter().map(|v| v as f32).collect();
    if sizes.is_empty() || values.len() != sizes.len() {
        return 0.0;
    }
    if size <= sizes[0] {
        return values[0];
    }
    for i in 1..sizes.len() {
        if size <= sizes[i] {
            let t = (size - sizes[i - 1]) / (sizes[i] - sizes[i - 1]);
            return values[i - 1] + (values[i] - values[i - 1]) * t;
        }
    }
    values[values.len() - 1]
}

/// The advance of `glyph` in font units with the face's variation applied,
/// unrounded (hmtx plus the HVAR delta).
fn exact_advance(face: &rustybuzz::Face<'_>, glyph: ttf_parser::GlyphId) -> Option<f64> {
    let base = face.tables().hmtx?.advance(glyph)? as f64;
    if face.is_variable()
        && let Some(hvar) = face.tables().hvar
        && let Some(delta) = hvar.advance_offset(glyph, face.variation_coordinates())
    {
        return Some(base + delta as f64);
    }
    Some(base)
}

/// Blink `TextMetrics::GetFontBaseline(kMiddleTextBaseline)` from
/// `SimpleFontData::NormalizedTypoAscentAndDescent`: the OS/2 typographic
/// ascent and descent scaled to sum to the font size, each rounded to a
/// LayoutUnit (1/64 px), and the middle is half their difference. A font
/// without usable typographic metrics uses its font metrics instead
/// ([`font_metrics_ascent_descent`]), normalized the same way.
fn middle_baseline(face: &rustybuzz::Face<'_>, family: &str, size: f32) -> f32 {
    let normalized = |ascent: f32, descent: f32| -> Option<f32> {
        let height = ascent + descent;
        if height <= 0.0 || ascent < 0.0 || ascent > height {
            return None;
        }
        let layout_unit = |v: f32| (v * 64.0).round() / 64.0;
        let a = layout_unit(size * ascent / height);
        let d = layout_unit(size * descent / height);
        Some((a - d) / 2.0)
    };
    let typo = face
        .typographic_ascender()
        .zip(face.typographic_descender())
        .and_then(|(a, d)| normalized(a as f32, -(d as f32)));
    typo.or_else(|| {
        let (a, d) = font_metrics_ascent_descent(face, family, size);
        normalized(a, d)
    })
    .unwrap_or(0.0)
}

/// Blink `FontMetrics::AscentDescentWithHacks`: the platform's ascent and
/// descent at `size` (the hhea values CoreText reports for a TrueType
/// font), rounded to whole pixels (unrounded for tiny fonts); on macOS,
/// Times, Helvetica and Courier get 15 % of their height added to the
/// ascent, as WebKit did to match the metrics of their Windows
/// counterparts.
fn font_metrics_ascent_descent(face: &rustybuzz::Face<'_>, family: &str, size: f32) -> (f32, f32) {
    let scale = size / face.units_per_em() as f32;
    let float_ascent = face.ascender() as f32 * scale;
    let float_descent = -(face.descender() as f32) * scale;
    let (mut ascent, descent) = if float_ascent < 3.0 || float_ascent + float_descent < 2.0 {
        (float_ascent, float_descent)
    } else {
        (float_ascent.round(), float_descent.round())
    };
    if cfg!(target_os = "macos") && matches!(family, "Times" | "Helvetica" | "Courier") {
        ascent += ((ascent + descent) * 0.15 + 0.5).floor();
    }
    (ascent, descent)
}

/// A glyph outline transformed to device pixels; `bounds` collects the
/// control box of the outline in font units (min x, min y, max x, max y).
struct DevicePath<'a> {
    path: &'a mut raster::Path,
    m: [f64; 4],
    origin: (f64, f64),
    scale: f64,
    bounds: Option<[f32; 4]>,
}

impl DevicePath<'_> {
    fn map(&mut self, x: f32, y: f32) -> (f64, f64) {
        self.bounds = Some(match self.bounds {
            None => [x, y, x, y],
            Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
        });
        // font units, y up -> user space, y down -> device
        let ux = x as f64 * self.scale;
        let uy = -(y as f64) * self.scale;
        (
            self.origin.0 + self.m[0] * ux + self.m[2] * uy,
            self.origin.1 + self.m[1] * ux + self.m[3] * uy,
        )
    }
}

impl ttf_parser::OutlineBuilder for DevicePath<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.path.move_to(p);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.path.line_to(p);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let c = self.map(x1, y1);
        let p = self.map(x, y);
        self.path.quad_to(c, p);
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let c1 = self.map(x1, y1);
        let c2 = self.map(x2, y2);
        let p = self.map(x, y);
        self.path.cubic_to(c1, c2, p);
    }
    fn close(&mut self) {
        self.path.close();
    }
}

/// Outlines `glyph` with skrifa when ttf-parser cannot (it rejects the
/// `gvar` data of some variable fonts, such as macOS's SF). skrifa rounds
/// variation deltas to whole font units as FreeType does, so outlines differ
/// from CoreText's by up to a unit. Returns whether an outline was drawn.
fn skrifa_outline(
    face: &ResolvedFace,
    glyph: u16,
    builder: &mut dyn ttf_parser::OutlineBuilder,
) -> bool {
    use skrifa::MetadataProvider;
    use skrifa::instance::{LocationRef, Size};
    use skrifa::outline::{DrawSettings, OutlinePen};
    struct Pen<'a>(&'a mut dyn ttf_parser::OutlineBuilder, bool);
    impl OutlinePen for Pen<'_> {
        fn move_to(&mut self, x: f32, y: f32) {
            self.1 = true;
            self.0.move_to(x, y);
        }
        fn line_to(&mut self, x: f32, y: f32) {
            self.0.line_to(x, y);
        }
        fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
            self.0.quad_to(cx, cy, x, y);
        }
        fn curve_to(&mut self, c0x: f32, c0y: f32, c1x: f32, c1y: f32, x: f32, y: f32) {
            self.0.curve_to(c0x, c0y, c1x, c1y, x, y);
        }
        fn close(&mut self) {
            self.0.close();
        }
    }
    let Ok(font) = skrifa::FontRef::from_index(face.bytes.as_slice(), face.index) else {
        return false;
    };
    let settings: Vec<(String, f32)> = face
        .variations
        .iter()
        .map(|(tag, value)| (String::from_utf8_lossy(tag).into_owned(), *value))
        .collect();
    let location = font
        .axes()
        .location(settings.iter().map(|(t, v)| (t.as_str(), *v)));
    let Some(outline) = font
        .outline_glyphs()
        .get(skrifa::GlyphId::new(glyph as u32))
    else {
        return false;
    };
    let mut pen = Pen(builder, false);
    outline
        .draw(
            DrawSettings::unhinted(Size::unscaled(), LocationRef::from(&location)),
            &mut pen,
        )
        .is_ok()
        && pen.1
}

/// The transparent canvas the demo draws on: premultiplied RGBA8, row 0 at
/// the top.
#[derive(Clone, Debug)]
pub struct TextCanvas {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Premultiplied RGBA8 backing store, row 0 at the top.
    pub pixels: Vec<u8>,
    /// The four samples of pixels a multisampled path covers partly
    /// ([`msaa`]), by pixel index; every other pixel's samples equal it.
    /// Resolved into `pixels` by [`TextCanvas::resolve`].
    samples: std::collections::BTreeMap<usize, [[u8; 4]; 4]>,
}

/// Source-over of premultiplied `color` at `coverage` onto a stored
/// premultiplied RGBA8 value: dst = color * c + dst * (1 - alpha * c).
fn blend(dst: &mut [u8], color: [f32; 4], coverage: f32) {
    let inverse = 1.0 - color[3] * coverage;
    for (stored, channel) in dst.iter_mut().zip(color) {
        let d = *stored as f32 / 255.0;
        let v = channel * coverage + d * inverse;
        *stored = ((v as f64).clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8;
    }
}

impl TextCanvas {
    /// The texture the demo uploads from this canvas (flipY, straight
    /// alpha; see [`crate::upload_canvas_rgba8`]).
    pub fn upload_image(&self) -> Rgba8Image {
        crate::upload_canvas_rgba8(&self.pixels, self.width, self.height)
    }

    /// Composites an 8-bit glyph mask in `color` (premultiplied f32)
    /// source-over: dst = color * m / 255 + dst * (1 - alpha * m / 255), on
    /// every sample of a multisampled pixel.
    fn composite(&mut self, mask: &raster::Mask, color: [f32; 4]) {
        let (w, h) = (self.width as i64, self.height as i64);
        for my in 0..mask.height {
            let y = mask.top + my as i64;
            if y < 0 || y >= h {
                continue;
            }
            for mx in 0..mask.width {
                let x = mask.left + mx as i64;
                if x < 0 || x >= w {
                    continue;
                }
                let m = mask.value(mx, my);
                if m == 0 {
                    continue;
                }
                self.blend_pixel((y * w + x) as usize, color, m as f32 / 255.0);
            }
        }
    }

    /// Composites `color` at `coverage` on a pixel (on each of its samples
    /// when it is multisampled).
    fn blend_pixel(&mut self, index: usize, color: [f32; 4], coverage: f32) {
        match self.samples.get_mut(&index) {
            Some(samples) => samples.iter_mut().for_each(|s| blend(s, color, coverage)),
            None => blend(&mut self.pixels[index * 4..index * 4 + 4], color, coverage),
        }
    }

    /// Composites `color` source-over on the samples a multisampled path
    /// covers.
    fn composite_samples(&mut self, mask: &msaa::SampleMask, color: [f32; 4]) {
        let w = self.width as usize;
        for my in 0..mask.height {
            for mx in 0..mask.width {
                let bits = mask.bits[my * mask.width + mx];
                if bits == 0 {
                    continue;
                }
                let index = (mask.top as usize + my) * w + mask.left as usize + mx;
                let pixel = &mut self.pixels[index * 4..index * 4 + 4];
                if bits == 0b1111 && !self.samples.contains_key(&index) {
                    blend(pixel, color, 1.0);
                    continue;
                }
                let value: [u8; 4] = (&*pixel).try_into().expect("an RGBA8 pixel");
                let samples = self.samples.entry(index).or_insert([value; 4]);
                for (k, sample) in samples.iter_mut().enumerate() {
                    if bits & (1 << k) != 0 {
                        blend(sample, color, 1.0);
                    }
                }
            }
        }
    }

    /// The multisample resolve: each multisampled pixel becomes
    /// (s0 + s1 + s2 + s3 + 2) >> 2 per channel.
    fn resolve(&mut self) {
        for (index, samples) in std::mem::take(&mut self.samples) {
            for c in 0..4 {
                let sum: u32 = samples.iter().map(|s| s[c] as u32).sum();
                self.pixels[index * 4 + c] = ((sum + 2) >> 2) as u8;
            }
        }
    }
}

/// `ctx.measureText(text).width` for a canvas `font` value (an invalid
/// value measures with `10px sans-serif`): the advance width of `text` as
/// [`draw_text_canvas`] lays it out, letter spacing included.
pub fn measure_text(fonts: &TextFonts, style: &HostStyle, font: &str, text: &str) -> f32 {
    let font = parse_canvas_font(font).unwrap_or_default();
    let mut faces: Vec<ResolvedFace> = fonts
        .resolve(&font.families)
        .into_iter()
        .map(|f| prepare_face(f, font.size, style))
        .collect();
    shape_line(&normalize_spaces(text), font.size, &mut faces, fonts, style).width
}

/// Draws `params` as the demo host does on a `width` x `height` canvas
/// (the demo's canvas is [`demo_canvas_size`]).
pub fn draw_text_canvas(
    fonts: &TextFonts,
    style: &HostStyle,
    params: &TextParams,
    width: u32,
    height: u32,
) -> TextCanvas {
    let mut canvas = TextCanvas {
        width,
        height,
        pixels: vec![0; width as usize * height as usize * 4],
        samples: Default::default(),
    };
    let text = params.text.as_str();
    let lines: Vec<&str> = text.split('\n').collect();
    let font_size = js::math_round(params.size * height as f64);
    let line_height = font_size * 1.2;
    let rotation = params.rotation * std::f64::consts::PI / 180.0;

    // ctx.font = `${fontSize}px ${font}`
    let font = parse_canvas_font(&format!(
        "{}px {}",
        js::number_to_string(font_size),
        params.font
    ))
    .unwrap_or_default();
    let align = text_align(&params.justify);
    // fillStyle: an invalid colour keeps the default black
    let [r, g, b, a] = parse_rgba_style(&fill_style(&params.color)).unwrap_or([0, 0, 0, 255]);
    let inv255 = 1.0f32 / 255.0;
    let alpha = a as f32 * inv255;
    let color = [
        r as f32 * inv255 * alpha,
        g as f32 * inv255 * alpha,
        b as f32 * inv255 * alpha,
        alpha,
    ];

    let x = params.pos_x * width as f64;
    let y = params.pos_y * height as f64;
    // ctx.translate(x, y); ctx.rotate(rotation): Blink ignores non-finite
    // arguments and no-op calls, and passes floats to SkCanvas.
    let mut ctm = SkMatrix::IDENTITY;
    if x.is_finite() && y.is_finite() && (x != 0.0 || y != 0.0) {
        ctm = ctm.translate(x as f32, y as f32);
    }
    if rotation.is_finite() && rotation != 0.0 {
        ctm = ctm.rotate((rotation * 180.0 / std::f64::consts::PI) as f32);
    }

    let total_height = (lines.len() as f64 - 1.0) * line_height;
    let start_y = -total_height / 2.0;

    let mut faces: Vec<ResolvedFace> = fonts
        .resolve(&font.families)
        .into_iter()
        .map(|f| prepare_face(f, font.size, style))
        .collect();
    if faces.is_empty() || font.size <= 0.0 {
        return canvas;
    }
    let size = font.size;
    let baseline_shift = faces[0]
        .shaping_face()
        .map(|f| middle_baseline(&f, &faces[0].family, size as f32))
        .unwrap_or(0.0);

    for (i, line) in lines.iter().enumerate() {
        let line_y = start_y + i as f64 * line_height;
        // fillText(line, 0, lineY): non-finite coordinates draw nothing
        if !line_y.is_finite() || line.is_empty() {
            continue;
        }
        let shaped = shape_line(&normalize_spaces(line), size, &mut faces, fonts, style);
        let offset = match align {
            Align::Center => -shaped.width / 2.0,
            Align::Right | Align::End => -shaped.width,
            Align::Left | Align::Start => 0.0,
        };
        let origin_x = offset;
        let origin_y = (line_y as f32) + baseline_shift;
        draw_glyphs(
            &mut canvas,
            &shaped,
            &faces,
            size,
            (origin_x, origin_y),
            ctm,
            color,
        );
    }
    canvas.resolve();
    canvas
}

/// Places and rasterizes the glyphs of a line whose origin (left end of
/// the baseline) is `origin` in user space. Like Skia's GPU text
/// (`SubRunContainer`), a run of one face with device text size `size *
/// getMaxScale()` is drawn
///
/// - below 162 px as glyph masks;
/// - from 162 px to below 256 px as distance fields ([`sdf`]), the glyphs
///   whose field is too large then as glyph masks;
/// - glyphs too large for a mask, and every glyph from 256 px, as paths
///   ([`msaa`]).
fn draw_glyphs(
    canvas: &mut TextCanvas,
    line: &ShapedLine,
    faces: &[ResolvedFace],
    size: f64,
    origin: (f32, f32),
    ctm: SkMatrix,
    color: [f32; 4],
) {
    // The text blob is drawn at `origin`: Skia's position matrix is the
    // canvas matrix translated to it, and glyph positions are mapped
    // through it.
    let position = ctm.translate(origin.0, origin.1);
    let device_size = size as f32 * position.max_scale();
    let as_masks = device_size < msaa::MAX_MASK_SIDE && device_size.abs() > 1.0 / 4096.0;
    let as_fields = as_masks && device_size >= sdf::MIN_SIZE;
    let parsed: Vec<Option<rustybuzz::Face<'_>>> = faces.iter().map(|f| f.shaping_face()).collect();
    let mut start = 0;
    while start < line.glyphs.len() {
        let face_index = line.glyphs[start].face;
        let end = line.glyphs[start..]
            .iter()
            .position(|g| g.face != face_index)
            .map_or(line.glyphs.len(), |n| start + n);
        let Some(face) = parsed[face_index].as_ref() else {
            start = end;
            continue;
        };
        let resolved = &faces[face_index];
        // the run's sub-runs in order: distance fields, masks, paths
        let field = FieldRun {
            size: size as f32,
            device_size,
            origin,
            ctm,
        };
        let mut masks: Vec<&PlacedGlyph> = Vec::new();
        for glyph in &line.glyphs[start..end] {
            if !as_fields || !draw_field_glyph(canvas, glyph, face, resolved, &field, color) {
                masks.push(glyph);
            }
        }
        let mut paths: Vec<&PlacedGlyph> = Vec::new();
        for glyph in masks {
            if !as_masks || !draw_mask_glyph(canvas, glyph, face, resolved, size, position, color) {
                paths.push(glyph);
            }
        }
        for glyph in paths {
            draw_path_glyph(canvas, glyph, face, resolved, size, origin, ctm, color);
        }
        start = end;
    }
}

/// Draws a glyph mask: the CoreGraphics coverage of the glyph at its
/// subpixel position. Returns false without drawing when the mask would be
/// larger than the glyph atlas allows (the glyph is then drawn as a path):
/// `SkScalerContext_Mac::generateMetrics` bounds the glyph by its control
/// box, transformed and offset by its subpixel position, rounded out and
/// outset by 1 pixel.
fn draw_mask_glyph(
    canvas: &mut TextCanvas,
    glyph: &PlacedGlyph,
    face: &rustybuzz::Face<'_>,
    resolved: &ResolvedFace,
    size: f64,
    position: SkMatrix,
    color: [f32; 4],
) -> bool {
    // Skia's subpixel rounding (SkGlyphPositionRoundingSpec): quarter
    // pixels along the axis the baseline maps to and whole pixels across
    // it when the matrix is axis-aligned, quarter pixels on both axes
    // otherwise.
    let axis_x = position.ky == 0.0;
    let axis_y = !axis_x && position.sx == 0.0;
    let quarter = |v: f32| ((v + 0.125) * 4.0).floor() / 4.0;
    let whole = |v: f32| (v + 0.5).floor();
    let (dx, dy) = position.map(glyph.x, glyph.y);
    let (qx, qy) = if axis_x {
        (quarter(dx), whole(dy))
    } else if axis_y {
        (whole(dx), quarter(dy))
    } else {
        (quarter(dx), quarter(dy))
    };
    let mut path = raster::Path::new();
    let scale = size / face.units_per_em() as f64;
    let m = [
        position.sx as f64,
        position.ky as f64,
        position.kx as f64,
        position.sy as f64,
    ];
    let mut builder = DevicePath {
        path: &mut path,
        m,
        origin: (qx as f64, qy as f64),
        scale,
        bounds: None,
    };
    if face
        .outline_glyph(ttf_parser::GlyphId(glyph.glyph), &mut builder)
        .is_none()
        && !skrifa_outline(resolved, glyph.glyph, &mut builder)
    {
        return true;
    }
    if let Some([x0, y0, x1, y1]) = builder.bounds {
        let corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)].map(|(x, y)| {
            let (ux, uy) = (x as f64 * scale, -(y as f64) * scale);
            (m[0] * ux + m[2] * uy, m[1] * ux + m[3] * uy)
        });
        let left = corners.iter().map(|c| c.0).fold(f64::INFINITY, f64::min) as f32;
        let top = corners.iter().map(|c| c.1).fold(f64::INFINITY, f64::min) as f32;
        let right = corners
            .iter()
            .map(|c| c.0)
            .fold(f64::NEG_INFINITY, f64::max) as f32;
        let bottom = corners
            .iter()
            .map(|c| c.1)
            .fold(f64::NEG_INFINITY, f64::max) as f32;
        let right = right + (qx - qx.floor());
        let bottom = bottom + (qy - qy.floor());
        let width = right.ceil() - left.floor() + 2.0;
        let height = bottom.ceil() - top.floor() + 2.0;
        if width.max(height) > msaa::MAX_MASK_SIDE {
            return false;
        }
    }
    if let Some(mask) = path.fill() {
        canvas.composite(&mask, color);
    }
    true
}

/// CoreGraphics font smoothing at the distance-field strike sizes: the
/// dilation half-width, in pixels ([`raster::Path::fill_smoothed`]).
const SMOOTHING: f64 = 0.3;

/// The placement of a distance-field run.
struct FieldRun {
    size: f32,
    device_size: f32,
    origin: (f32, f32),
    ctm: SkMatrix,
}

/// Draws a glyph from its distance field ([`sdf`]): the CoreGraphics mask
/// of the glyph at the strike size (unrotated, at a whole-pixel origin),
/// its field, and the field's quad scaled by `size / strike` at the glyph's
/// exact position. Returns false without drawing when the padded field is
/// larger than the atlas allows.
fn draw_field_glyph(
    canvas: &mut TextCanvas,
    glyph: &PlacedGlyph,
    face: &rustybuzz::Face<'_>,
    resolved: &ResolvedFace,
    run: &FieldRun,
    color: [f32; 4],
) -> bool {
    let strike = sdf::strike_size(run.size, run.device_size);
    let k = strike as f64 / face.units_per_em() as f64;
    let mut path = raster::Path::new();
    let mut builder = DevicePath {
        path: &mut path,
        m: [1.0, 0.0, 0.0, 1.0],
        origin: (0.0, 0.0),
        scale: k,
        bounds: None,
    };
    if face
        .outline_glyph(ttf_parser::GlyphId(glyph.glyph), &mut builder)
        .is_none()
        && !skrifa_outline(resolved, glyph.glyph, &mut builder)
    {
        return true;
    }
    let Some([x0, y0, x1, y1]) = builder.bounds else {
        return true;
    };
    // the glyph image: bounds rounded out and outset by 1 pixel
    let bounds = [
        (x0 as f64 * k) as f32,
        (-(y1 as f64) * k) as f32,
        (x1 as f64 * k) as f32,
        (-(y0 as f64) * k) as f32,
    ];
    if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
        return true;
    }
    let left = bounds[0].floor() as i64 - 1;
    let top = bounds[1].floor() as i64 - 1;
    let width = (bounds[2].ceil() as i64 + 1 - left) as usize;
    let height = (bounds[3].ceil() as i64 + 1 - top) as usize;
    let pad = sdf::PAD as usize;
    let side = msaa::MAX_MASK_SIDE as usize;
    if width + 2 * pad > side || height + 2 * pad > side {
        return false;
    }
    // the strike is hinted, so CoreGraphics smooths the glyph
    // the strike is hinted, so CoreGraphics smooths the glyph
    let Some(mask) = path.fill_smoothed(SMOOTHING, SMOOTHING) else {
        return true;
    };
    let mut image = vec![0u8; width * height];
    for my in 0..mask.height {
        let y = mask.top + my as i64 - top;
        if y < 0 || y >= height as i64 {
            continue;
        }
        for mx in 0..mask.width {
            let x = mask.left + mx as i64 - left;
            if x < 0 || x >= width as i64 {
                continue;
            }
            image[y as usize * width + x as usize] = sdf::linear_coverage(mask.value(mx, my));
        }
    }
    let field = sdf::distance_field(&image, width, height);
    let field_width = width + 2 * pad;
    // texel (u, v) of the field is strike point (left - 4 + u, top - 4 + v);
    // the quad spans texels [2, width + 6) x [2, height + 6)
    let inset = sdf::INSET as f64;
    let (u0, u1) = (inset, (width + 2 * pad) as f64 - inset);
    let (v0, v1) = (inset, (height + 2 * pad) as f64 - inset);
    let ctm = run.ctm;
    let (a, b, c, d) = (ctm.sx as f64, ctm.kx as f64, ctm.ky as f64, ctm.sy as f64);
    let det = a * d - b * c;
    if det == 0.0 || !det.is_finite() {
        return true;
    }
    let s = (run.size / strike) as f64;
    let (px, py) = (
        (run.origin.0 + glyph.x) as f64,
        (run.origin.1 + glyph.y) as f64,
    );
    let g = (
        a * px + b * py + ctm.tx as f64,
        c * px + d * py + ctm.ty as f64,
    );
    let field_origin = ((left - sdf::PAD) as f64, (top - sdf::PAD) as f64);
    let to_device = |u: f64, v: f64| {
        let (sx, sy) = ((field_origin.0 + u) * s, (field_origin.1 + v) * s);
        (g.0 + a * sx + b * sy, g.1 + c * sx + d * sy)
    };
    let corners = [
        to_device(u0, v0),
        to_device(u1, v0),
        to_device(u0, v1),
        to_device(u1, v1),
    ];
    let min_x = corners.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
    let max_x = corners
        .iter()
        .map(|p| p.0)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = corners.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
    let max_y = corners
        .iter()
        .map(|p| p.1)
        .fold(f64::NEG_INFINITY, f64::max);
    let (w, h) = (canvas.width as i64, canvas.height as i64);
    let x_start = ((min_x - 0.5).floor() as i64).max(0);
    let x_end = ((max_x + 0.5).ceil() as i64).min(w);
    let y_start = ((min_y - 0.5).floor() as i64).max(0);
    let y_end = ((max_y + 0.5).ceil() as i64).min(h);
    let aa_width = sdf::aa_width((1.0 / (s * ctm.max_scale() as f64)) as f32);
    for y in y_start..y_end {
        for x in x_start..x_end {
            // the pixel centre in field texels
            let (dx, dy) = (x as f64 + 0.5 - g.0, y as f64 + 0.5 - g.1);
            let sx = (d * dx - b * dy) / det;
            let sy = (a * dy - c * dx) / det;
            let u = sx / s - field_origin.0;
            let v = sy / s - field_origin.1;
            if u < u0 || u >= u1 || v < v0 || v >= v1 {
                continue;
            }
            let coverage = sdf::coverage(&field, field_width, u as f32, v as f32, aa_width);
            if coverage > 0.0 {
                canvas.blend_pixel((y * w + x) as usize, color, coverage);
            }
        }
    }
    true
}

/// Draws a glyph as a path (`PathOpSubmitter`): its outline at 64 px,
/// scaled by `size / 64` and translated to the glyph's exact position,
/// concatenated with the canvas matrix in floats (`SkM44::setConcat`),
/// filled with 4x multisampling.
#[allow(clippy::too_many_arguments)]
fn draw_path_glyph(
    canvas: &mut TextCanvas,
    glyph: &PlacedGlyph,
    face: &rustybuzz::Face<'_>,
    resolved: &ResolvedFace,
    size: f64,
    origin: (f32, f32),
    ctm: SkMatrix,
    color: [f32; 4],
) {
    let mut path = msaa::GlyphPath::new(face.units_per_em());
    if face
        .outline_glyph(ttf_parser::GlyphId(glyph.glyph), &mut path)
        .is_none()
        && !skrifa_outline(resolved, glyph.glyph, &mut path)
    {
        return;
    }
    let s = size as f32 / msaa::PATH_STRIKE_SIZE;
    let tx = origin.0 + glyph.x;
    let ty = origin.1 + glyph.y;
    let matrix = msaa::PathMatrix {
        m00: ctm.sx * s,
        m01: ctm.kx * s,
        m03: ctm.sx * tx + (ctm.kx * ty + ctm.tx),
        m10: ctm.ky * s,
        m11: ctm.sy * s,
        m13: ctm.ky * tx + (ctm.sy * ty + ctm.ty),
    };
    let contours = path.tessellate(&matrix);
    if let Some(mask) = msaa::sample_coverage(&contours, canvas.width, canvas.height) {
        canvas.composite_samples(&mask, color);
    }
}

/// The texture the demo host uploads for `params` on a `width` x `height`
/// canvas (the demo's is [`demo_canvas_size`]), with the bundled and
/// installed fonts and the demo page's inherited style: straight-alpha
/// RGBA8, row 0 first (texture rows, i.e. the canvas bottom row first).
pub fn render_text_canvas(params: &TextParams, width: u32, height: u32) -> Rgba8Image {
    render_text_canvas_with(
        &TextFonts::default(),
        &HostStyle::demo(),
        params,
        width,
        height,
    )
}

/// [`render_text_canvas`] with a given font set and page style.
pub fn render_text_canvas_with(
    fonts: &TextFonts,
    style: &HostStyle,
    params: &TextParams,
    width: u32,
    height: u32,
) -> Rgba8Image {
    draw_text_canvas(fonts, style, params, width, height).upload_image()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_style_matches_the_demo() {
        assert_eq!(
            fill_style(&TextColor::Hex("#ff8000".into())),
            "rgba(255, 128, 0, 1)"
        );
        assert_eq!(
            fill_style(&TextColor::Hex("bogus".into())),
            "rgba(255, 255, 255, 1)"
        );
        assert_eq!(
            fill_style(&TextColor::Array(vec![0.2, 0.6, 1.0, 0.5])),
            "rgba(51, 153, 255, 1)"
        );
        assert_eq!(
            fill_style(&TextColor::Array(vec![0.5])),
            "rgba(128, NaN, NaN, 1)"
        );
    }

    #[test]
    fn bundled_nunito_draws_the_default_text() {
        let canvas = draw_text_canvas(
            &TextFonts::bundled_only(),
            &HostStyle::demo(),
            &TextParams::default(),
            256,
            256,
        );
        let drawn = canvas
            .pixels
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|p| p[3] > 0)
            .count();
        assert!(drawn > 800, "{drawn} pixels drawn");
        // centered on (128, 128): ink on both sides of the center
        let ink_x: Vec<usize> = (0..256 * 256)
            .filter(|i| canvas.pixels[i * 4 + 3] > 0)
            .map(|i| i % 256)
            .collect();
        assert!(*ink_x.iter().min().unwrap() < 100 && *ink_x.iter().max().unwrap() > 156);
    }
}
