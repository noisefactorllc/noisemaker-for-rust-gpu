//! Canvas 2D back ends for the traced overlays: [`CallRecorder`] logs the
//! reference's canvas operations (the differential test compares the log
//! with the one the reference writes), and [`StrokeCanvas`] rasterizes them
//! as the browser canvas does.
//!
//! # The browser canvas
//!
//! The reference draws on an `HTMLCanvasElement` 2D context in Chromium.
//! Until reference 1.0.233 the canvas was GPU accelerated: Skia Graphite on
//! Metal draws each stroked line segment of the tracer (one
//! `moveTo`/`lineTo` with round caps per `stroke()`) as an
//! `AnalyticRRectRenderStep` instance. Since 1.0.233 the overlays request
//! `willReadFrequently`, which pins the canvas to Skia's software
//! rasterizer; [`StrokeCanvas`] does not model that rasterizer yet, so its
//! overlays no longer match the reference's. [`StrokeCanvas`] models the
//! Graphite pipeline exactly:
//!
//! - Blink culls a stroke whose bounding box, outset by lineWidth / 2 and
//!   rounded out, does not intersect the canvas (BaseRenderingContext2D
//!   dirty rect).
//! - `"rgba(r, g, b, a)"` is parsed to an 8-bit colour: channels clamp to
//!   0..255 and round, alpha becomes floor(a * 255 + 0.5). Skia converts it
//!   to float as c * (1 / 255.0f) and premultiplies in float.
//! - Graphite draws the line as a 36-vertex instance (four corners of nine
//!   vertices, a 69-index triangle strip) whose fragment shader computes
//!   analytic coverage from interpolated edge distances (Skia
//!   src/sksl/sksl_graphite_vert.sksl `analytic_rrect_vertex_fn`,
//!   sksl_graphite_frag.sksl `analytic_rrect_coverage_fn`). The model runs
//!   the same vertex and fragment arithmetic in f32, snaps vertex positions
//!   to the rasterizer's 1/256 pixel grid, rasterizes with the top-left fill
//!   rule and shades each pixel at most once per stroke (Graphite's depth
//!   test rejects a second triangle of the same draw).
//! - Source-over blending in f32, dst' = src * coverage + dst * (1 - srcA *
//!   coverage), stored as floor(v * 255 + 0.5).
//!
//! The model was established by the Qt port of the engine
//! (noisemaker-for-qt `stroke_canvas.cpp`, measured on Chromium 153, Skia
//! Graphite on Metal, Apple M4) and is checked here against the overlays
//! the reference uploads (`tests/overlay_raster.rs`). Other GPUs and
//! back ends rasterize strokes differently (Chromium with Skia Ganesh on
//! OpenGL draws different edge pixels); the traced geometry is the same.
//!
//! # Residual
//!
//! Against 16 distinct overlays the reference uploaded on Chromium 153 /
//! Apple M4 (fibers, scratches and strayHair at 256 and 512 px, seeds 1 to
//! 9, densities 0.25 to 1), 8 are byte-exact and 103 of 360893 drawn
//! pixels differ (0.03 %; global SSIM >= 0.999996). The drawn footprint is
//! identical in all. Every differing pixel is one rounding flip in one
//! blend of its stroke sequence, where this model's value lies within
//! 8.2e-5 (in 8-bit units, typically 1e-5) of a half-unit tie: the GPU's
//! coverage arithmetic differs from the model at the 1e-5 relative level
//! near stroke edges. Plane-equation and barycentric interpolation
//! variants, FMA contraction in the vertex, coverage and blend arithmetic,
//! reciprocal forms and half-precision coverage were measured and none
//! removes them (fp16 coverage multiplies them by 30). The flip changes a
//! premultiplied channel by 1, which the straight-alpha upload amplifies
//! at low alpha (max-abs-diff 85 where alpha is 3).

use crate::Rgba8Image;
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

// ------------------------------------------------------------ stroke model

#[derive(Clone, Copy, Default)]
struct TemplateVertex {
    corner: usize,
    pos_x: f32,
    pos_y: f32,
    normal_x: f32,
    normal_y: f32,
    /// +1 device outset, 0 outer anchor, -1 inset.
    normal_scale: f32,
    /// 1 for the center-fill vertex.
    center_weight: f32,
}

const CORNER_VERTEX_COUNT: usize = 9;
const VERTEX_COUNT: usize = 4 * CORNER_VERTEX_COUNT;
const INDEX_COUNT: usize = 69;

/// `get_per_corner_vertex_attrs<kCornerID>()` repeated for TL, TR, BR, BL.
fn vertex_template() -> [TemplateVertex; VERTEX_COUNT] {
    let hr2 = 0.5f32 * std::f32::consts::SQRT_2; // SK_FloatSqrt2 (1.41421356f)
    let mut out = [TemplateVertex::default(); VERTEX_COUNT];
    for c in 0..4 {
        let v = |pos_x, pos_y, normal_x, normal_y, normal_scale, center_weight| TemplateVertex {
            corner: c,
            pos_x,
            pos_y,
            normal_x,
            normal_y,
            normal_scale,
            center_weight,
        };
        let corner = [
            v(1.0, 0.0, 1.0, 0.0, 1.0, 0.0),
            v(1.0, 0.0, hr2, hr2, 1.0, 0.0),
            v(0.0, 1.0, hr2, hr2, 1.0, 0.0),
            v(0.0, 1.0, 0.0, 1.0, 1.0, 0.0),
            v(1.0, 0.0, hr2, hr2, 0.0, 0.0),
            v(0.0, 1.0, hr2, hr2, 0.0, 0.0),
            v(1.0, 0.0, 1.0, 0.0, -1.0, 0.0),
            v(0.0, 1.0, 0.0, 1.0, -1.0, 0.0),
            v(1.0, 0.0, 1.0, 0.0, -1.0, 1.0),
        ];
        out[c * CORNER_VERTEX_COUNT..(c + 1) * CORNER_VERTEX_COUNT].copy_from_slice(&corner);
    }
    out
}

/// `write_index_buffer()`: a triangle strip over the 36 vertices.
const INDICES: [usize; INDEX_COUNT] = [
    0, 4, 1, 5, 2, 3, 5, 9, 13, 10, 14, 11, 12, 14, 18, 22, 19, 23, 20, 21, 23, 27, 31, 28, 32, 29,
    30, 32, 0, 4, 4, 6, 5, 7, 13, 15, 14, 16, 22, 24, 23, 25, 31, 33, 32, 34, 4, 6, 6, 8, 7, 7, 17,
    15, 17, 16, 16, 26, 24, 26, 25, 25, 35, 33, 35, 34, 34, 8, 6,
];

#[derive(Clone, Copy, Default)]
struct Varyings {
    jacobian: [f32; 4],
    edge_distances: [f32; 4],
    stroke_radius: f32,
    join_style: f32,
    per_pixel_x: f32,
    per_pixel_y: f32,
}

#[derive(Clone, Copy, Default)]
struct DeviceVertex {
    x: f32,
    y: f32,
    v: Varyings,
}

/// The per-instance part of `analytic_rrect_vertex_fn`, identical for all
/// 36 vertices of a stroke.
struct LineInstance {
    /// ltrb.LLRR: corners ordered TL, TR, BR, BL.
    xs: [f32; 4],
    /// ltrb.TTBB.
    ys: [f32; 4],
    /// Normalized edge vectors, ordered L, T, R, B.
    dx: [f32; 4],
    dy: [f32; 4],
    edge_aa: [f32; 4],
    stroke_radius: f32,
    center_x: f32,
    center_y: f32,
}

fn line_instance(x0: f32, y0: f32, x1: f32, y1: f32, stroke_radius: f32) -> LineInstance {
    let xs = [x0, x0, x1, x1];
    let ys = [y0, y0, y1, y1];
    let mut dx = [0f32; 4];
    let mut dy = [0f32; 4];
    let mut edge_aa = [1f32; 4];
    let mut edge_squared_len = [0f32; 4];
    let mut edge_mask = [0f32; 4];
    for i in 0..4 {
        let w = (i + 3) % 4; // .wxyz
        let mut ex = xs[i] - xs[w];
        let mut ey = ys[i] - ys[w];
        let inv_mag = 1.0f32 / ex.abs().max(ey.abs().max(1.0));
        ex *= inv_mag;
        ey *= inv_mag;
        dx[i] = ex;
        dy[i] = ey;
        edge_squared_len[i] = ex * ex + ey * ey;
        // sign() of a non-negative value
        edge_mask[i] = if edge_squared_len[i] > 0.0 { 1.0 } else { 0.0 };
    }
    // A line has two empty edges (the caps); each takes the left-hand
    // normal of the adjacent edge. mix(a, b, t) = a + (b - a) * t. A
    // zero-length line never gets here (StrokeCanvas::stroke_segment).
    let mut nx = [0f32; 4];
    let mut ny = [0f32; 4];
    let mut nl = [0f32; 4];
    let mut naa = [0f32; 4];
    for i in 0..4 {
        let j = (i + 1) % 4; // .yzwx
        let edge_x = dy[j];
        let edge_y = -dx[j];
        nx[i] = edge_x + (dx[i] - edge_x) * edge_mask[i];
        ny[i] = edge_y + (dy[i] - edge_y) * edge_mask[i];
        nl[i] = edge_squared_len[j] + (edge_squared_len[i] - edge_squared_len[j]) * edge_mask[i];
        naa[i] = edge_aa[j] + (edge_aa[i] - edge_aa[j]) * edge_mask[i];
    }
    dx = nx;
    dy = ny;
    edge_squared_len = nl;
    edge_aa = naa;
    for i in 0..4 {
        let inverse_edge_len = 1.0f32 / edge_squared_len[i].sqrt();
        dx[i] *= inverse_edge_len;
        dy[i] *= inverse_edge_len;
    }
    LineInstance {
        xs,
        ys,
        dx,
        dy,
        edge_aa,
        stroke_radius,
        // bounds.center() of the line's bounding box
        center_x: (x0.min(x1) + x0.max(x1)) * 0.5,
        center_y: (y0.min(y1) + y0.max(y1)) * 0.5,
    }
}

/// The per-vertex part of `analytic_rrect_vertex_fn`.
fn line_vertex(t: &TemplateVertex, line: &LineInstance) -> DeviceVertex {
    const ROUND_SCALE: f32 = 0.414_213_56;
    let corner = t.corner;
    let next = (corner + 1) % 4;
    let join_scale = ROUND_SCALE; // round cap == round join

    let x_axis_x = -line.dx[next];
    let x_axis_y = -line.dy[next];
    let y_axis_x = line.dx[corner];
    let y_axis_y = line.dy[corner];

    let (local_x, local_y) = if t.normal_scale < 0.0 {
        // Inset vertices snap to the center (center.w < 0).
        (line.center_x, line.center_y)
    } else {
        // (cornerRadii + strokeRadius) * (position + joinScale *
        // position.yx), then from the corner basis to local coordinates.
        let px = line.stroke_radius * (t.pos_x + join_scale * t.pos_y);
        let py = line.stroke_radius * (t.pos_y + join_scale * t.pos_x);
        (
            line.xs[corner] + x_axis_x * px + y_axis_x * py,
            line.ys[corner] + x_axis_y * px + y_axis_y * py,
        )
    };

    let mut out = DeviceVertex::default();
    for i in 0..4 {
        out.v.edge_distances[i] =
            line.dy[i] * (line.xs[i] - local_x) - line.dx[i] * (line.ys[i] - local_y);
    }
    out.x = local_x;
    out.y = local_y;
    if t.normal_scale > 0.0 {
        // Device-space AA outset by one pixel along the corner normal.
        let normal_x = line.edge_aa[corner] * t.normal_x;
        let normal_y = line.edge_aa[next] * t.normal_y;
        // perp(-yAxis) and perp(xAxis), perp(v) = (-v.y, v.x).
        let sum_x = normal_x * y_axis_y + normal_y * -x_axis_y;
        let sum_y = normal_x * -y_axis_x + normal_y * x_axis_x;
        let inverse_length = 1.0f32 / (sum_x * sum_x + sum_y * sum_y).sqrt();
        out.x += sum_x * inverse_length;
        out.y += sum_y * inverse_length;
        out.v.per_pixel_y = -1.0;
    } else {
        out.v.per_pixel_y = 0.0;
    }
    out.v.per_pixel_x = if t.center_weight != 0.0 { 1.0 } else { 0.0 };
    // The fragment shader works in the line's own basis.
    out.v.jacobian = [line.dy[0], -line.dy[1], -line.dx[0], line.dx[1]];
    out.v.stroke_radius = line.stroke_radius;
    out.v.join_style = -1.0;
    out
}

/// `$inverse_grad_len(localGrad, J)` with J's columns (J0, J1), (J2, J3).
fn inverse_grad_length(gx: f32, gy: f32, j: &[f32; 4]) -> f32 {
    let a = gx * j[0] + gy * j[1];
    let b = gx * j[2] + gy * j[3];
    1.0f32 / (a * a + b * b).sqrt()
}

/// `analytic_rrect_coverage_fn` for a stroked line (solid interior, round
/// corners of radius 0 with a stroke radius).
fn line_coverage(v: &Varyings) -> f32 {
    if v.per_pixel_x > 0.0 {
        return 1.0;
    }
    let j = &v.jacobian;
    let inv_grad_x = inverse_grad_length(1.0, 0.0, j);
    let inv_grad_y = inverse_grad_length(0.0, 1.0, j);
    let s = v.stroke_radius;
    let e = &v.edge_distances;
    let outer_x = inv_grad_x * (s + e[0].min(e[2]));
    let outer_y = inv_grad_y * (s + e[1].min(e[3]));
    let mut dist_outer = outer_x.min(outer_y);
    let mut dist_inner = -1.0f32;

    let dim_x = inv_grad_x * (e[0] + e[2] + 2.0 * s);
    let dim_y = inv_grad_y * (e[1] + e[3] + 2.0 * s);
    let scale = dim_x.min(dim_y).min(1.0);
    let bias = 1.0f32 - 0.5 * scale;

    // $corner_distances: TL (L,T), TR (R,T), BR (R,B), BL (L,B).
    const CORNER_EDGES: [[usize; 2]; 4] = [[0, 1], [2, 1], [2, 3], [0, 3]];
    const FLIPS: [[f32; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    for k in 0..4 {
        let u = 0.0f32 - e[CORNER_EDGES[k][0]];
        let w = 0.0f32 - e[CORNER_EDGES[k][1]];
        if !(u > 0.0 && w > 0.0) {
            continue;
        }
        if !(s > 0.0 && v.join_style < 0.0) {
            continue;
        }
        // $elliptical_distance(uv * xyFlip, radii = 0, strokeRadius, J)
        let uu = u * FLIPS[k][0];
        let ww = w * FLIPS[k][1];
        let inv_r2 = 1.0f32 / (0.0 * 0.0 + s * s);
        let nu = inv_r2 * uu;
        let nw = inv_r2 * ww;
        let inv_grad = inverse_grad_length(nu, nw, j);
        let f = 0.5f32 * inv_grad * ((uu * nu + ww * nw) - 1.0);
        let width = 0.0f32 * s * inv_r2 * inv_grad;
        dist_outer = dist_outer.min(width - f);
        // radii.x - strokeRadius <= 0: the inner curve collapsed.
        dist_inner = dist_inner.min(1.0);
    }

    let outset = v.per_pixel_y.min(0.0);
    let coverage = scale * ((dist_outer + outset).min(-dist_inner) + bias);
    coverage.clamp(0.0, 1.0)
}

/// A texel of an RGBA8 render target read as float: c / 255.
fn unorm8_to_float(c: u8) -> f32 {
    c as f32 / 255.0
}

fn to_unorm8(value: f32) -> u8 {
    let clamped = (value as f64).clamp(0.0, 1.0);
    (clamped * 255.0 + 0.5).floor() as u8
}

/// Rasterizer fixed point: 8 bits of subpixel precision.
const SUBPIXEL: f64 = 256.0;

/// Device coordinate -> NDC (Graphite's fused rtAdjust) -> viewport -> the
/// 1/256 pixel grid, rounding half up.
pub(crate) fn snap_x(x: f32, dimension: f32) -> i64 {
    let scale = 2.0f32 / dimension;
    let half = dimension * 0.5;
    let ndc = x.mul_add(scale, -1.0);
    let back = ndc.mul_add(half, half);
    (back as f64 * SUBPIXEL + 0.5).floor() as i64
}

pub(crate) fn snap_y(y: f32, dimension: f32) -> i64 {
    let scale = 2.0f32 / dimension;
    let half = dimension * 0.5;
    let ndc = y.mul_add(-scale, 1.0);
    let back = (-ndc).mul_add(half, half);
    (back as f64 * SUBPIXEL + 0.5).floor() as i64
}

/// A top or left edge of a triangle with its interior on the right (y down).
fn is_top_left(dx: i64, dy: i64) -> bool {
    dy < 0 || (dy == 0 && dx > 0)
}

fn first_pixel(fixed_min: i64) -> i64 {
    ((fixed_min as f64 - 128.0) / SUBPIXEL).ceil() as i64
}

fn last_pixel(fixed_max: i64) -> i64 {
    ((fixed_max as f64 - 128.0) / SUBPIXEL).floor() as i64
}

/// The browser canvas the overlays are traced on, as a [`Canvas2d`]: a
/// transparent premultiplied RGBA8 backing store that strokes the
/// tracer's round-capped segments with the Graphite model described in the
/// module documentation.
///
/// It implements the operations the reference issues: `clearRect`,
/// `lineWidth`, `strokeStyle` as `rgb()`/`rgba()`, `lineCap`/`lineJoin`
/// (recorded; every stroke is drawn with round caps, the only cap the
/// reference sets), and paths of `moveTo`/`lineTo` segments. A path whose
/// subpath has several segments is stroked segment by segment.
#[derive(Clone, Debug)]
pub struct StrokeCanvas {
    width: u32,
    height: u32,
    line_width: f32,
    line_cap: String,
    line_join: String,
    /// Premultiplied stroke colour.
    color: [f32; 4],
    /// Premultiplied RGBA8, row 0 at the top.
    pixels: Vec<u8>,
    /// Per-stroke "pixel already shaded" mask.
    shaded: Vec<u8>,
    /// Subpaths of the current path, in canvas coordinates.
    path: Vec<Vec<(f64, f64)>>,
}

impl StrokeCanvas {
    /// A transparent `width` x `height` canvas (the reference's
    /// `document.createElement('canvas')` sized by the asyncInit).
    pub fn new(width: u32, height: u32) -> StrokeCanvas {
        let count = width as usize * height as usize;
        StrokeCanvas {
            width,
            height,
            line_width: 1.0,
            line_cap: "butt".into(),
            line_join: "miter".into(),
            color: [0.0, 0.0, 0.0, 1.0],
            pixels: vec![0; count * 4],
            shaded: vec![0; count],
            path: Vec::new(),
        }
    }

    /// The backing store: premultiplied RGBA8, row 0 at the top.
    pub fn premultiplied(&self) -> &[u8] {
        &self.pixels
    }

    /// The current `lineCap` and `lineJoin` values.
    pub fn line_style(&self) -> (&str, &str) {
        (&self.line_cap, &self.line_join)
    }

    /// The texture the reference uploads from this canvas:
    /// `copyExternalImageToTexture({source: canvas, flipY: true}, {texture})`
    /// into rgba8unorm (see [`crate::upload_canvas_rgba8`]).
    pub fn upload_image(&self) -> Rgba8Image {
        crate::upload_canvas_rgba8(&self.pixels, self.width, self.height)
    }

    fn stroke_segment(&mut self, x0d: f64, y0d: f64, x1d: f64, y1d: f64) {
        // Nothing to draw with a transparent source-over paint.
        if self.color[3] == 0.0 {
            return;
        }
        let (x0, y0, x1, y1) = (x0d as f32, y0d as f32, x1d as f32, y1d as f32);
        // The canvas draws nothing for a segment whose end points are equal
        // once converted to the path's float coordinates, while a segment
        // one float ulp long draws a round dot. A length whose square
        // underflows would give the GPU NaN vertices, which rasterize
        // nothing.
        let length2 = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);
        if length2.is_nan() || length2 <= 0.0 {
            return;
        }
        let stroke_radius = 0.5f32 * self.line_width;

        // Blink InflateStrokeRect + ComputeDirtyRect: the path bounds
        // outset by lineWidth / 2, rounded out, must intersect the canvas.
        {
            let left = x0.min(x1) - stroke_radius;
            let top = y0.min(y1) - stroke_radius;
            let width = (x0.max(x1) - x0.min(x1)) + 2.0 * stroke_radius;
            let height = (y0.max(y1) - y0.min(y1)) + 2.0 * stroke_radius;
            let l = (left as f64).floor();
            let t = (top as f64).floor();
            let r = ((left + width) as f64).ceil();
            let b = ((top + height) as f64).ceil();
            if r <= 0.0 || b <= 0.0 || l >= self.width as f64 || t >= self.height as f64 {
                return;
            }
        }

        let template = vertex_template();
        let line = line_instance(x0, y0, x1, y1, stroke_radius);
        let mut vertices = [DeviceVertex::default(); VERTEX_COUNT];
        let mut fixed_x = [0i64; VERTEX_COUNT];
        let mut fixed_y = [0i64; VERTEX_COUNT];
        let width_f = self.width as f32;
        let height_f = self.height as f32;
        for i in 0..VERTEX_COUNT {
            vertices[i] = line_vertex(&template[i], &line);
            fixed_x[i] = snap_x(vertices[i].x, width_f);
            fixed_y[i] = snap_y(vertices[i].y, height_f);
        }
        let min_x = *fixed_x.iter().min().unwrap();
        let max_x = *fixed_x.iter().max().unwrap();
        let min_y = *fixed_y.iter().min().unwrap();
        let max_y = *fixed_y.iter().max().unwrap();
        // Pixels whose centers can lie inside the instance, clipped.
        let box_x0 = first_pixel(min_x).max(0);
        let box_y0 = first_pixel(min_y).max(0);
        let box_x1 = last_pixel(max_x).min(self.width as i64 - 1);
        let box_y1 = last_pixel(max_y).min(self.height as i64 - 1);
        if box_x0 > box_x1 || box_y0 > box_y1 {
            return;
        }
        let stride = self.width as usize;

        for t in 0..INDEX_COUNT - 2 {
            let ia = INDICES[t];
            let mut ib = INDICES[t + 1];
            let mut ic = INDICES[t + 2];
            let (ax, ay) = (fixed_x[ia], fixed_y[ia]);
            let (mut bx, mut by) = (fixed_x[ib], fixed_y[ib]);
            let (mut cx, mut cy) = (fixed_x[ic], fixed_y[ic]);
            let mut area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if area == 0 {
                continue;
            }
            if area < 0 {
                std::mem::swap(&mut ib, &mut ic);
                std::mem::swap(&mut bx, &mut cx);
                std::mem::swap(&mut by, &mut cy);
                area = -area;
            }
            let va = vertices[ia].v;
            let vb = vertices[ib].v;
            let vc = vertices[ic].v;
            let top_left_ab = is_top_left(bx - ax, by - ay);
            let top_left_bc = is_top_left(cx - bx, cy - by);
            let top_left_ca = is_top_left(ax - cx, ay - cy);
            let px0 = box_x0.max(first_pixel(ax.min(bx).min(cx)));
            let px1 = box_x1.min(last_pixel(ax.max(bx).max(cx)));
            let py0 = box_y0.max(first_pixel(ay.min(by).min(cy)));
            let py1 = box_y1.min(last_pixel(ay.max(by).max(cy)));
            let area_f = area as f32;
            // Edge functions at the first pixel center of the box, and
            // their steps per pixel (256 fixed-point units).
            let start_x = px0 * 256 + 128;
            let start_y = py0 * 256 + 128;
            let mut row_c = (bx - ax) * (start_y - ay) - (by - ay) * (start_x - ax);
            let mut row_a = (cx - bx) * (start_y - by) - (cy - by) * (start_x - bx);
            let mut row_b = (ax - cx) * (start_y - cy) - (ay - cy) * (start_x - cx);
            let (step_xc, step_yc) = (-(by - ay) * 256, (bx - ax) * 256);
            let (step_xa, step_ya) = (-(cy - by) * 256, (cx - bx) * 256);
            let (step_xb, step_yb) = (-(ay - cy) * 256, (ax - cx) * 256);
            let mut y = py0;
            while y <= py1 {
                let (mut wc, mut wa, mut wb) = (row_c, row_a, row_b);
                let mut x = px0;
                while x <= px1 {
                    let inside = !(wc < 0 || (wc == 0 && !top_left_ab))
                        && !(wa < 0 || (wa == 0 && !top_left_bc))
                        && !(wb < 0 || (wb == 0 && !top_left_ca));
                    if inside {
                        let index = y as usize * stride + x as usize;
                        if self.shaded[index] == 0 {
                            self.shaded[index] = 1;
                            let b0 = wa as f32 / area_f;
                            let b1 = wb as f32 / area_f;
                            let b2 = wc as f32 / area_f;
                            let lerp = |a: f32, b: f32, c: f32| b0 * a + b1 * b + b2 * c;
                            let mut v = Varyings::default();
                            for i in 0..4 {
                                v.jacobian[i] =
                                    lerp(va.jacobian[i], vb.jacobian[i], vc.jacobian[i]);
                                v.edge_distances[i] = lerp(
                                    va.edge_distances[i],
                                    vb.edge_distances[i],
                                    vc.edge_distances[i],
                                );
                            }
                            v.stroke_radius =
                                lerp(va.stroke_radius, vb.stroke_radius, vc.stroke_radius);
                            v.join_style = lerp(va.join_style, vb.join_style, vc.join_style);
                            v.per_pixel_x = lerp(va.per_pixel_x, vb.per_pixel_x, vc.per_pixel_x);
                            v.per_pixel_y = lerp(va.per_pixel_y, vb.per_pixel_y, vc.per_pixel_y);
                            let coverage = line_coverage(&v);

                            let pixel = &mut self.pixels[index * 4..index * 4 + 4];
                            let source = self.color.map(|c| c * coverage);
                            let inverse_alpha = 1.0f32 - source[3];
                            for i in 0..4 {
                                pixel[i] = to_unorm8(
                                    source[i] + unorm8_to_float(pixel[i]) * inverse_alpha,
                                );
                            }
                        }
                    }
                    wc += step_xc;
                    wa += step_xa;
                    wb += step_xb;
                    x += 1;
                }
                row_c += step_yc;
                row_a += step_ya;
                row_b += step_yb;
                y += 1;
            }
        }
        for y in box_y0..=box_y1 {
            let row = y as usize * stride;
            self.shaded[row + box_x0 as usize..=row + box_x1 as usize].fill(0);
        }
    }
}

impl Canvas2d for StrokeCanvas {
    fn width(&self) -> u32 {
        self.width
    }
    fn height(&self) -> u32 {
        self.height
    }
    fn clear_rect(&mut self, x: f64, y: f64, w: f64, h: f64) {
        if ![x, y, w, h].iter().all(|v| v.is_finite()) {
            return;
        }
        // Pixel-aligned rectangles clear exactly; the reference only clears
        // the whole canvas.
        let (x0, x1) = (x.min(x + w), x.max(x + w));
        let (y0, y1) = (y.min(y + h), y.max(y + h));
        let cx0 = x0.round().clamp(0.0, self.width as f64) as usize;
        let cx1 = x1.round().clamp(0.0, self.width as f64) as usize;
        let cy0 = y0.round().clamp(0.0, self.height as f64) as usize;
        let cy1 = y1.round().clamp(0.0, self.height as f64) as usize;
        let stride = self.width as usize * 4;
        for row in cy0..cy1 {
            self.pixels[row * stride + cx0 * 4..row * stride + cx1 * 4].fill(0);
        }
    }
    fn set_line_cap(&mut self, cap: &str) {
        if matches!(cap, "butt" | "round" | "square") {
            self.line_cap = cap.into();
        }
    }
    fn set_line_join(&mut self, join: &str) {
        if matches!(join, "round" | "bevel" | "miter") {
            self.line_join = join.into();
        }
    }
    fn set_line_width(&mut self, width: f64) {
        // Non-finite and non-positive values are ignored, as the canvas does.
        if width.is_finite() && width > 0.0 {
            self.line_width = width as f32;
        }
    }
    fn set_stroke_style(&mut self, style: &str) {
        // An invalid colour string leaves the previous style.
        if let Some([r, g, b, a8]) = parse_rgba_style(style) {
            // SkColor4f::FromColor: c * (1 / 255.0f); SkColor4f::premul().
            let inv255 = 1.0f32 / 255.0;
            let a = a8 as f32 * inv255;
            self.color = [
                r as f32 * inv255 * a,
                g as f32 * inv255 * a,
                b as f32 * inv255 * a,
                a,
            ];
        }
    }
    fn begin_path(&mut self) {
        self.path.clear();
    }
    fn move_to(&mut self, x: f64, y: f64) {
        if x.is_finite() && y.is_finite() {
            self.path.push(vec![(x, y)]);
        }
    }
    fn line_to(&mut self, x: f64, y: f64) {
        if !(x.is_finite() && y.is_finite()) {
            return;
        }
        match self.path.last_mut() {
            Some(subpath) => subpath.push((x, y)),
            // lineTo on an empty path behaves as moveTo
            None => self.path.push(vec![(x, y)]),
        }
    }
    fn stroke(&mut self) {
        let path = std::mem::take(&mut self.path);
        for subpath in &path {
            for pair in subpath.windows(2) {
                self.stroke_segment(pair[0].0, pair[0].1, pair[1].0, pair[1].1);
            }
        }
        self.path = path;
    }
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
