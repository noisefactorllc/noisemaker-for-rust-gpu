//! Glyphs the canvas draws as paths instead of glyph masks.
//!
//! Skia's GPU text (sktext::gpu::SubRunContainer) draws a glyph run with
//! glyph masks only while `size * matrix.getMaxScale()` is below 256
//! (`SkGlyphDigest::kSkSideTooBigForAtlas`); a larger run, and any glyph
//! whose mask would be larger than 256 pixels on a side, is drawn as a path:
//! the glyph's outline at the canonical 64 px (`kCanonicalTextSizeForPaths`)
//! scaled to the font size (`PathOpSubmitter`), at its exact position (no
//! subpixel rounding). Graphite fills such a path by GPU tessellation
//! into a 4x multisampled target:
//!
//! - curves are evaluated in the vertex shader at 2^L equal parameter steps,
//!   L = ceil(log2 n) with n from Wang's formula at 1/4 pixel precision
//!   (`tessellate_filled_curve`; at most 32 steps per patch, longer curves
//!   are chopped into patches first);
//! - vertices are snapped to the rasterizer's 1/256 pixel grid
//!   ([`crate::canvas`]), and each of a pixel's four samples at the
//!   standard positions (3/8, 1/8), (7/8, 3/8), (1/8, 5/8), (5/8, 7/8) is
//!   covered when the path's winding number there is nonzero; a sample
//!   exactly on an edge counts as on the edge's right side (below it for a
//!   horizontal edge), which is what the top-left rule gives for every
//!   triangulation of the path;
//! - samples hold 8-bit premultiplied colour, and the resolve stores
//!   (s0 + s1 + s2 + s3 + 2) >> 2 per channel.
//!
//! Measured against Chromium 153 on macOS: runs switch from masks to paths
//! between 255 and 256 px (a 256 px run rotated by 10 degrees stays masks,
//! its f32 max scale being below 1); a 323 x 305 px Zapfino ligature at
//! 128 px is a path among mask glyphs; edge pixels hold exactly the
//! coverage levels 0, 64, 128, 191 and 255 of white text, and coloured
//! edges round as the resolve above.

use rustybuzz::ttf_parser;

/// Rasterizer fixed point: 8 bits of subpixel precision.
const SUBPIXEL: f64 = 256.0;

/// Device coordinate -> NDC (Graphite's fused rtAdjust) -> viewport -> the
/// 1/256 pixel grid, rounding half up.
fn snap_x(x: f32, dimension: f32) -> i64 {
    let scale = 2.0f32 / dimension;
    let half = dimension * 0.5;
    let ndc = x.mul_add(scale, -1.0);
    let back = ndc.mul_add(half, half);
    (back as f64 * SUBPIXEL + 0.5).floor() as i64
}

fn snap_y(y: f32, dimension: f32) -> i64 {
    let scale = 2.0f32 / dimension;
    let half = dimension * 0.5;
    let ndc = y.mul_add(-scale, 1.0);
    let back = (-ndc).mul_add(half, half);
    (back as f64 * SUBPIXEL + 0.5).floor() as i64
}

/// `SkFont::kCanonicalTextSizeForPaths`: glyph paths are taken at this size.
pub const PATH_STRIKE_SIZE: f32 = 64.0;

/// `SkGlyphDigest::kSkSideTooBigForAtlas`: the largest glyph mask side, and
/// the device text size from which a run is drawn as paths.
pub const MAX_MASK_SIDE: f32 = 256.0;

/// Tessellation precision, segments per pixel (`skgpu::tess::kPrecision`).
const PRECISION: f32 = 4.0;

/// `skgpu::tess::kMaxResolveLevel`: a patch is tessellated into at most
/// 2^5 segments.
const MAX_RESOLVE_LEVEL: i32 = 5;

/// `kMaxSegmentsPerCurve`: the most segments a curve is chopped into.
const MAX_SEGMENTS_PER_CURVE: f32 = 1024.0;

/// The 4x sample positions within a pixel, in 1/256 pixel from its top-left
/// corner (x, y).
pub const SAMPLE_POSITIONS: [(i64, i64); 4] = [(96, 32), (224, 96), (32, 160), (160, 224)];

type Point = (f32, f32);

#[derive(Clone, Copy, Debug)]
enum Verb {
    Move(Point),
    Line(Point),
    Quad(Point, Point),
    Cubic(Point, Point, Point),
}

/// A glyph outline as Skia's path strike holds it: CoreText's outline at
/// [`PATH_STRIKE_SIZE`], in floats, y down. Built from font units with
/// [`ttf_parser::OutlineBuilder`].
#[derive(Clone, Debug)]
pub struct GlyphPath {
    verbs: Vec<Verb>,
    scale: f64,
}

impl GlyphPath {
    /// An empty path for a face with `units_per_em`.
    pub fn new(units_per_em: i32) -> GlyphPath {
        GlyphPath {
            verbs: Vec::new(),
            scale: PATH_STRIKE_SIZE as f64 / units_per_em.max(1) as f64,
        }
    }

    /// Font units (y up) to the strike's float coordinates (y down).
    fn point(&self, x: f32, y: f32) -> Point {
        (
            (x as f64 * self.scale) as f32,
            (-(y as f64) * self.scale) as f32,
        )
    }

    /// The path's contours tessellated as the GPU tessellates them and
    /// mapped to device pixels by `matrix`.
    pub fn tessellate(&self, matrix: &PathMatrix) -> Vec<Vec<Point>> {
        let mut contours: Vec<Vec<Point>> = Vec::new();
        let mut current: Point = (0.0, 0.0);
        let mut contour: Vec<Point> = Vec::new();
        for verb in &self.verbs {
            match *verb {
                Verb::Move(p) => {
                    if contour.len() > 2 {
                        contours.push(std::mem::take(&mut contour));
                    }
                    contour.clear();
                    contour.push(matrix.map(p));
                    current = p;
                }
                Verb::Line(p) => {
                    contour.push(matrix.map(p));
                    current = p;
                }
                Verb::Quad(c, p) => {
                    let n = quad_segments_p2(matrix, current, c, p);
                    for (q0, q1, q2) in chop_quad(current, c, p, patches(n)) {
                        let level = quad_level(matrix, q0, q1, q2);
                        tessellate_patch(&mut contour, matrix, level, |t| {
                            mix(mix(q0, q1, t), mix(q1, q2, t), t)
                        });
                        contour.push(matrix.map(q2));
                    }
                    current = p;
                }
                Verb::Cubic(c1, c2, p) => {
                    let n = cubic_segments_p2(matrix, current, c1, c2, p);
                    for (q0, q1, q2, q3) in chop_cubic(current, c1, c2, p, patches(n)) {
                        let level = cubic_level(matrix, q0, q1, q2, q3);
                        tessellate_patch(&mut contour, matrix, level, |t| {
                            let ab = mix(q0, q1, t);
                            let bc = mix(q1, q2, t);
                            let cd = mix(q2, q3, t);
                            mix(mix(ab, bc, t), mix(bc, cd, t), t)
                        });
                        contour.push(matrix.map(q3));
                    }
                    current = p;
                }
            }
        }
        if contour.len() > 2 {
            contours.push(contour);
        }
        contours
    }
}

impl ttf_parser::OutlineBuilder for GlyphPath {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.verbs.push(Verb::Move(p));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.point(x, y);
        self.verbs.push(Verb::Line(p));
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let c = self.point(x1, y1);
        let p = self.point(x, y);
        self.verbs.push(Verb::Quad(c, p));
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let c1 = self.point(x1, y1);
        let c2 = self.point(x2, y2);
        let p = self.point(x, y);
        self.verbs.push(Verb::Cubic(c1, c2, p));
    }
    fn close(&mut self) {}
}

/// The local-to-device matrix of a path glyph (`[m00 m01 m03; m10 m11
/// m13]`), applied as the vertex shader applies it: `localToDevice *
/// float4(local, 0, 1)` with fused multiply-adds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathMatrix {
    /// x scale.
    pub m00: f32,
    /// x skew.
    pub m01: f32,
    /// x translation.
    pub m03: f32,
    /// y skew.
    pub m10: f32,
    /// y scale.
    pub m11: f32,
    /// y translation.
    pub m13: f32,
}

impl PathMatrix {
    fn map(&self, p: Point) -> Point {
        (
            self.m01.mul_add(p.1, self.m00 * p.0) + self.m03,
            self.m11.mul_add(p.1, self.m10 * p.0) + self.m13,
        )
    }

    /// The 2x2 part applied to a vector (`float2x2(localToDevice) * v`).
    fn vector(&self, v: Point) -> Point {
        (
            self.m01.mul_add(v.1, self.m00 * v.0),
            self.m11.mul_add(v.1, self.m10 * v.0),
        )
    }
}

/// SkSL `mix(a, b, t)` = a + (b - a) * t, fused.
fn mix(a: Point, b: Point, t: f32) -> Point {
    ((b.0 - a.0).mul_add(t, a.0), (b.1 - a.1).mul_add(t, a.1))
}

/// Appends the interior vertices of a patch tessellated at `level`
/// (`fixedVertexID` steps of 1/32), evaluated by `eval`.
fn tessellate_patch(
    contour: &mut Vec<Point>,
    matrix: &PathMatrix,
    level: i32,
    eval: impl Fn(f32) -> Point,
) {
    let level = level.clamp(0, MAX_RESOLVE_LEVEL);
    let step = 1 << (MAX_RESOLVE_LEVEL - level);
    for i in 1..(1 << level) {
        let t = (i * step) as f32 * (1.0 / (1 << MAX_RESOLVE_LEVEL) as f32);
        contour.push(matrix.map(eval(t)));
    }
}

/// `wangs_formula_conic_log2` for a quadratic (a conic of weight 1): the
/// resolve level of the curve in device space.
fn quad_level(matrix: &PathMatrix, p0: Point, p1: Point, p2: Point) -> i32 {
    let (a, b, c) = (matrix.vector(p0), matrix.vector(p1), matrix.vector(p2));
    // translate the bounding box center to the origin
    let cx = (a.0.min(b.0).min(c.0) + a.0.max(b.0).max(c.0)) * 0.5;
    let cy = (a.1.min(b.1).min(c.1) + a.1.max(b.1).max(c.1)) * 0.5;
    let (a, b, c) = (
        (a.0 - cx, a.1 - cy),
        (b.0 - cx, b.1 - cy),
        (c.0 - cx, c.1 - cy),
    );
    let dp = (
        (-2.0f32).mul_add(b.0, a.0) + c.0,
        (-2.0f32).mul_add(b.1, a.1) + c.1,
    );
    let length = dp.0.mul_add(dp.0, dp.1 * dp.1).sqrt();
    // numer = length * precision (+ 0 * dw), denom = 4 * min(w, 1)
    let n2 = length * PRECISION / 4.0;
    (n2.max(1.0).log2() * 0.5).ceil() as i32
}

/// `wangs_formula_cubic_log2`: the resolve level of a cubic in device space.
fn cubic_level(matrix: &PathMatrix, p0: Point, p1: Point, p2: Point, p3: Point) -> i32 {
    let d0 = matrix.vector((
        (-2.0f32).mul_add(p1.0, p2.0) + p0.0,
        (-2.0f32).mul_add(p1.1, p2.1) + p0.1,
    ));
    let d1 = matrix.vector((
        (-2.0f32).mul_add(p2.0, p3.0) + p1.0,
        (-2.0f32).mul_add(p2.1, p3.1) + p1.1,
    ));
    let m =
        d0.0.mul_add(d0.0, d0.1 * d0.1)
            .max(d1.0.mul_add(d1.0, d1.1 * d1.1));
    // n^4 = (3 * 2 / 8 * precision)^2 * m
    let length_term_p2 = (0.75 * PRECISION) * (0.75 * PRECISION);
    ((length_term_p2 * m).max(1.0).log2() * 0.25).ceil() as i32
}

/// The CPU side's segment count estimate of a quadratic (`PatchWriter`,
/// `wangs_formula::quadratic`), for chopping.
fn quad_segments_p2(matrix: &PathMatrix, p0: Point, p1: Point, p2: Point) -> f32 {
    let v = matrix.vector((p0.0 - 2.0 * p1.0 + p2.0, p0.1 - 2.0 * p1.1 + p2.1));
    // n^2 = (2 * 1 / 8 * precision) * |v|
    0.25 * PRECISION * (v.0 * v.0 + v.1 * v.1).sqrt()
}

/// The CPU side's segment count estimate of a cubic, squared.
fn cubic_segments_p2(matrix: &PathMatrix, p0: Point, p1: Point, p2: Point, p3: Point) -> f32 {
    let d0 = matrix.vector((p0.0 - 2.0 * p1.0 + p2.0, p0.1 - 2.0 * p1.1 + p2.1));
    let d1 = matrix.vector((p1.0 - 2.0 * p2.0 + p3.0, p1.1 - 2.0 * p2.1 + p3.1));
    let m = (d0.0 * d0.0 + d0.1 * d0.1).max(d1.0 * d1.0 + d1.1 * d1.1);
    0.75 * PRECISION * m.sqrt()
}

/// Patches a curve needing n segments (`n2` = n^2) is chopped into: one
/// while n <= 32, else ceil(n / 32) with n at most 1024.
fn patches(n2: f32) -> usize {
    let max = (1 << MAX_RESOLVE_LEVEL) as f32;
    // NaN (a degenerate curve) also takes one patch
    if n2.is_nan() || n2 <= max * max {
        return 1;
    }
    let n = n2.sqrt().min(MAX_SEGMENTS_PER_CURVE);
    (n / max).ceil() as usize
}

/// `count` equal parameter pieces of a quadratic.
fn chop_quad(p0: Point, p1: Point, p2: Point, count: usize) -> Vec<(Point, Point, Point)> {
    let mut out = Vec::with_capacity(count);
    let (mut a, mut b) = (p0, p1);
    for remaining in (1..=count).rev() {
        if remaining == 1 {
            out.push((a, b, p2));
            break;
        }
        let t = 1.0 / remaining as f32;
        let ab = mix(a, b, t);
        let bc = mix(b, p2, t);
        let abc = mix(ab, bc, t);
        out.push((a, ab, abc));
        a = abc;
        b = bc;
    }
    out
}

/// `count` equal parameter pieces of a cubic.
fn chop_cubic(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    count: usize,
) -> Vec<(Point, Point, Point, Point)> {
    let mut out = Vec::with_capacity(count);
    let (mut a, mut b, mut c) = (p0, p1, p2);
    for remaining in (1..=count).rev() {
        if remaining == 1 {
            out.push((a, b, c, p3));
            break;
        }
        let t = 1.0 / remaining as f32;
        let ab = mix(a, b, t);
        let bc = mix(b, c, t);
        let cd = mix(c, p3, t);
        let abc = mix(ab, bc, t);
        let bcd = mix(bc, cd, t);
        let abcd = mix(abc, bcd, t);
        out.push((a, ab, abc, abcd));
        a = abcd;
        b = bcd;
        c = cd;
    }
    out
}

/// Which samples of each pixel a filled path covers: bit k of a pixel's
/// byte is sample [`SAMPLE_POSITIONS`]`[k]`. Rows and columns are clipped to
/// the canvas.
#[derive(Clone, Debug)]
pub struct SampleMask {
    /// Canvas x of column 0.
    pub left: i64,
    /// Canvas y of row 0.
    pub top: i64,
    /// Columns.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Covered samples per pixel, row-major.
    pub bits: Vec<u8>,
}

/// An edge in rasterizer fixed point with `y0 < y1`, and its winding
/// direction.
struct Edge {
    x0: i64,
    y0: i64,
    x1: i64,
    y1: i64,
    dir: i32,
}

/// `ceil(n / d)` for `d > 0`.
fn ceil_div(n: i128, d: i128) -> i128 {
    let q = n.div_euclid(d);
    if q * d == n { q } else { q + 1 }
}

/// Snaps device contours to the rasterizer grid of a `width` x `height`
/// target and computes the samples their nonzero fill covers.
pub fn sample_coverage(contours: &[Vec<Point>], width: u32, height: u32) -> Option<SampleMask> {
    let (w, h) = (width as f32, height as f32);
    let mut edges = Vec::new();
    let (mut x_min, mut y_min, mut x_max, mut y_max) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for contour in contours {
        let fixed: Vec<(i64, i64)> = contour
            .iter()
            .map(|&(x, y)| (snap_x(x, w), snap_y(y, h)))
            .collect();
        for (i, &a) in fixed.iter().enumerate() {
            let b = fixed[(i + 1) % fixed.len()];
            if a.1 == b.1 {
                continue;
            }
            let (lo, hi, dir) = if a.1 < b.1 { (a, b, 1) } else { (b, a, -1) };
            x_min = x_min.min(lo.0.min(hi.0));
            x_max = x_max.max(lo.0.max(hi.0));
            y_min = y_min.min(lo.1);
            y_max = y_max.max(hi.1);
            edges.push(Edge {
                x0: lo.0,
                y0: lo.1,
                x1: hi.0,
                y1: hi.1,
                dir,
            });
        }
    }
    if edges.is_empty() {
        return None;
    }
    let top = y_min.div_euclid(256).max(0);
    let bottom = (y_max.div_euclid(256) + 1).min(height as i64);
    let left = x_min.div_euclid(256).max(0);
    let right = (x_max.div_euclid(256) + 1).min(width as i64);
    if top >= bottom || left >= right {
        return None;
    }
    let (mw, mh) = ((right - left) as usize, (bottom - top) as usize);
    let mut bits = vec![0u8; mw * mh];
    let mut diff = vec![0i32; mw + 1];
    for py in top..bottom {
        for (k, &(sx, sy)) in SAMPLE_POSITIONS.iter().enumerate() {
            let ys = py * 256 + sy;
            diff.fill(0);
            let mut crossed = false;
            for e in &edges {
                // the sample row crosses [y0, y1): a sample on a vertex row
                // counts as just below it
                if ys < e.y0 || ys >= e.y1 {
                    continue;
                }
                // first column whose sample x (256 px + sx) is at or right
                // of the crossing: px >= ((ys - y0)(x1 - x0) + (x0 - sx)(y1 - y0)) / 256 (y1 - y0)
                let dy = (e.y1 - e.y0) as i128;
                let n = (ys - e.y0) as i128 * (e.x1 - e.x0) as i128 + (e.x0 - sx) as i128 * dy;
                let first = ceil_div(n, 256 * dy);
                let column = (first - left as i128).clamp(0, mw as i128) as usize;
                diff[column] += e.dir;
                crossed = true;
            }
            if !crossed {
                continue;
            }
            let row = (py - top) as usize * mw;
            let mut winding = 0;
            for (x, d) in diff[..mw].iter().enumerate() {
                winding += d;
                if winding != 0 {
                    bits[row + x] |= 1 << k;
                }
            }
        }
    }
    Some(SampleMask {
        left,
        top,
        width: mw,
        height: mh,
        bits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<Point> {
        vec![(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    }

    #[test]
    fn rectangle_covers_samples_inside_it() {
        // x from 1.5 to 3.0, y from 1.25 to 2.0 on an 8 x 8 target
        let mask = sample_coverage(&[rect(1.5, 1.25, 3.0, 2.0)], 8, 8).unwrap();
        let at = |x: i64, y: i64| {
            mask.bits[((y - mask.top) as usize) * mask.width + (x - mask.left) as usize]
        };
        // pixel (1, 1): samples with x >= 0.5 and y >= 0.25: (7/8, 3/8), (5/8, 7/8)
        assert_eq!(at(1, 1), 0b1010);
        // pixel (2, 1): samples with y >= 0.25: all but (3/8, 1/8)
        assert_eq!(at(2, 1), 0b1110);
        // no pixel of row 2 is covered (the rectangle ends at y = 2)
        assert!(
            mask.top + mask.height as i64 <= 2
                || (0..mask.width)
                    .all(|x| mask.bits[(2 - mask.top) as usize * mask.width + x] == 0)
        );
    }

    #[test]
    fn samples_on_edges_count_right_and_below() {
        // left edge through the sample (5/8, 7/8), top edge through the
        // sample (7/8, 3/8): both count as inside
        let mask = sample_coverage(&[rect(0.625, 0.375, 1.0, 1.0)], 4, 4).unwrap();
        assert_eq!(mask.bits[0], 0b1010);
        // right edge through (5/8, 7/8): outside; (3/8, 1/8) and (1/8, 5/8)
        // inside
        let mask = sample_coverage(&[rect(0.0, 0.0, 0.625, 1.0)], 4, 4).unwrap();
        assert_eq!(mask.bits[0], 0b0101);
        // bottom edge through (1/8, 5/8): outside; (3/8, 1/8) and (7/8, 3/8)
        // inside
        let mask = sample_coverage(&[rect(0.0, 0.0, 1.0, 0.625)], 4, 4).unwrap();
        assert_eq!(mask.bits[0], 0b0011);
    }

    #[test]
    fn opposite_windings_cancel_and_nonzero_fills_overlaps() {
        let outer = rect(0.0, 0.0, 4.0, 4.0);
        let mut hole = rect(1.0, 1.0, 3.0, 3.0);
        hole.reverse();
        let mask = sample_coverage(&[outer.clone(), hole], 4, 4).unwrap();
        assert_eq!(mask.bits[0], 0b1111);
        assert_eq!(mask.bits[mask.width + 1], 0);
        let same = rect(1.0, 1.0, 3.0, 3.0);
        let mask = sample_coverage(&[outer, same], 4, 4).unwrap();
        assert_eq!(mask.bits[mask.width + 1], 0b1111);
    }

    #[test]
    fn quadratic_levels_follow_wangs_formula() {
        let identity = PathMatrix {
            m00: 1.0,
            m01: 0.0,
            m03: 0.0,
            m10: 0.0,
            m11: 1.0,
            m13: 0.0,
        };
        // |p0 - 2 p1 + p2| = 2 * 50 = 100: n = 10 -> 16 segments (level 4)
        assert_eq!(
            quad_level(&identity, (0.0, 0.0), (50.0, 50.0), (100.0, 0.0)),
            4
        );
        // straight: one segment
        assert_eq!(quad_level(&identity, (0.0, 0.0), (1.0, 0.0), (2.0, 0.0)), 0);
        // cubic with second differences of length 100: n^4 = 9 * 100^2, n = 17.3 -> 32
        assert_eq!(
            cubic_level(
                &identity,
                (0.0, 0.0),
                (0.0, 100.0),
                (100.0, 100.0),
                (100.0, 0.0)
            ),
            5
        );
    }
}
