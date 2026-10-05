//! Antialiased glyph coverage as the platform rasterizer computes it on
//! the reference platform (CoreGraphics, which Chromium's Skia asks for
//! glyph masks on macOS with `-webkit-font-smoothing: antialiased`).
//!
//! - Coverage is the exact area of each pixel inside the outline, computed
//!   by accumulating the signed area each edge sweeps (the accumulation
//!   rasterizer of font-rs), in f64; coverage is the absolute accumulated
//!   area clamped to 1, which equals the nonzero rule wherever contours do
//!   not partially overlap inside one pixel. Where they do (variable fonts
//!   keep overlapping contours), CoreGraphics counts the overlap twice as
//!   this does: the exact nonzero area misses its full pixels (Nunito at
//!   26 px: 6 pixels off by up to 9 with it, none by more than 1 without).
//! - Curves are flattened by halving: a quadratic is split into the
//!   smallest power of two of equal-parameter pieces whose control point
//!   lies within [`QUAD_FLATNESS`] (max norm) of its chord's midpoint; a
//!   cubic is halved recursively until the max-norm second differences of
//!   its control points sum to at most [`CUBIC_FLATNESS`].
//! - An 8-bit mask value is min(255, floor(256 * coverage)).
//!
//! Measured against the masks CoreGraphics draws on macOS 26 with Skia's
//! settings (`SkScalerContext_Mac`: subpixel quantization off, subpixel
//! positioning on) for 8 letters of Nunito (wght 580), Times New Roman,
//! Georgia and Arial at 31, 64, 120, 161 and 233 px, 1.33 million pixels:
//! 0.9 % differ by 1 (8-bit), 0.09 % by 2, 0.03 % by 3 to 5, none by more;
//! tolerance-based flattening (0.01 px) differs by up to 131. Chromium's
//! canvas text, which draws glyph masks up to 161 px, matches within 1 as
//! well (`tests/text_raster.rs`). Composite glyphs of the variable Nunito
//! (accented letters) differ by up to 65 (64 to 256 px): CoreText places
//! their varied components differently. For CFF (cubic) glyphs
//! (STIXGeneral, 48 glyphs) the rule leaves 2220 of their pixels
//! different, 627 by more than 1, at most 66: CoreGraphics' cubic
//! criterion is not identified. Below 19 px CoreGraphics also grid-fits
//! outlines vertically (cap heights and overshoots snap to whole pixels),
//! which this rasterizer does not reproduce.

/// Quadratic flatness: the largest max-norm distance between a piece's
/// control point and its chord's midpoint, in pixels.
pub const QUAD_FLATNESS: f64 = 0.4;

/// Cubic flatness: the largest sum of the max norms of a piece's two
/// control-point second differences, in pixels.
pub const CUBIC_FLATNESS: f64 = 0.52;

/// Halvings at most (a curve is split into at most 2^16 pieces).
const MAX_DEPTH: u32 = 16;

/// A coverage mask: `width * height` values in [0, 1], row-major, with its
/// top-left pixel at (`left`, `top`) on the canvas.
#[derive(Clone, Debug)]
pub struct Mask {
    /// Canvas x of column 0.
    pub left: i64,
    /// Canvas y of row 0.
    pub top: i64,
    /// Columns.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Coverage per pixel.
    pub coverage: Vec<f64>,
}

impl Mask {
    /// The 8-bit mask value of a pixel: min(255, floor(256 * coverage)).
    pub fn value(&self, x: usize, y: usize) -> u8 {
        (self.coverage[y * self.width + x] * 256.0)
            .floor()
            .min(255.0) as u8
    }
}

/// A path in device pixels (y down), flattened to line segments.
#[derive(Clone, Debug, Default)]
pub struct Path {
    segments: Vec<[(f64, f64); 2]>,
    /// Index of the first segment of each contour.
    contours: Vec<usize>,
    start: (f64, f64),
    current: (f64, f64),
    open: bool,
}

impl Path {
    /// An empty path.
    pub fn new() -> Path {
        Path::default()
    }

    /// Starts a contour (closing the previous one).
    pub fn move_to(&mut self, p: (f64, f64)) {
        self.close();
        self.contours.push(self.segments.len());
        self.start = p;
        self.current = p;
        self.open = true;
    }

    /// A straight edge.
    pub fn line_to(&mut self, p: (f64, f64)) {
        if self.current != p {
            self.segments.push([self.current, p]);
        }
        self.current = p;
    }

    /// A quadratic Bezier edge.
    pub fn quad_to(&mut self, c: (f64, f64), p: (f64, f64)) {
        let p0 = self.current;
        let mid = (0.5 * (p0.0 + p.0), 0.5 * (p0.1 + p.1));
        // each halving divides the control point's offset from the chord
        // midpoint by 4
        let mut deviation = (c.0 - mid.0).abs().max((c.1 - mid.1).abs());
        let mut n = 1usize;
        let mut depth = 0;
        while deviation > QUAD_FLATNESS && depth < MAX_DEPTH {
            deviation /= 4.0;
            n *= 2;
            depth += 1;
        }
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let mt = 1.0 - t;
            let q = (
                mt * mt * p0.0 + 2.0 * mt * t * c.0 + t * t * p.0,
                mt * mt * p0.1 + 2.0 * mt * t * c.1 + t * t * p.1,
            );
            self.line_to(if i == n { p } else { q });
        }
    }

    /// A cubic Bezier edge.
    pub fn cubic_to(&mut self, c1: (f64, f64), c2: (f64, f64), p: (f64, f64)) {
        let p0 = self.current;
        self.cubic_piece(p0, c1, c2, p, 0);
    }

    fn cubic_piece(
        &mut self,
        p0: (f64, f64),
        c1: (f64, f64),
        c2: (f64, f64),
        p3: (f64, f64),
        depth: u32,
    ) {
        let linf = |x: f64, y: f64| x.abs().max(y.abs());
        let flatness = linf(p0.0 - 2.0 * c1.0 + c2.0, p0.1 - 2.0 * c1.1 + c2.1)
            + linf(c1.0 - 2.0 * c2.0 + p3.0, c1.1 - 2.0 * c2.1 + p3.1);
        if flatness <= CUBIC_FLATNESS || depth >= MAX_DEPTH {
            self.line_to(p3);
            return;
        }
        let half = |a: (f64, f64), b: (f64, f64)| (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1));
        let ab = half(p0, c1);
        let bc = half(c1, c2);
        let cd = half(c2, p3);
        let abc = half(ab, bc);
        let bcd = half(bc, cd);
        let m = half(abc, bcd);
        self.cubic_piece(p0, ab, abc, m, depth + 1);
        self.cubic_piece(m, bcd, cd, p3, depth + 1);
    }

    /// Closes the current contour.
    pub fn close(&mut self) {
        if self.open {
            let start = self.start;
            self.line_to(start);
            self.open = false;
        }
    }

    /// Whether the path has no edges.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// The coverage of the (closed) path.
    pub fn fill(mut self) -> Option<Mask> {
        self.close();
        accumulated_mask(&self.segments)
    }
}

/// The coverage of `edges` as CoreGraphics accumulates it: the signed area
/// each edge sweeps, summed along each row, absolute value clamped to 1.
fn accumulated_mask(edges: &[[(f64, f64); 2]]) -> Option<Mask> {
    let mut it = edges.iter().flatten();
    let first = it.next()?;
    let (mut x0, mut y0, mut x1, mut y1) = (first.0, first.1, first.0, first.1);
    for p in it {
        x0 = x0.min(p.0);
        y0 = y0.min(p.1);
        x1 = x1.max(p.0);
        y1 = y1.max(p.1);
    }
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return None;
    }
    let (left, top) = (x0.floor() as i64, y0.floor() as i64);
    let width = (x1.ceil() as i64 - left) as usize + 2;
    let height = ((y1.ceil() as i64) - top).max(1) as usize;
    let mut acc = vec![0f64; width * height + 2];
    for [p0, p1] in edges {
        accumulate_line(
            &mut acc,
            width,
            height,
            (p0.0 - left as f64, p0.1 - top as f64),
            (p1.0 - left as f64, p1.1 - top as f64),
        );
    }
    let mut coverage = vec![0f64; width * height];
    for y in 0..height {
        let mut sum = 0.0;
        for x in 0..width {
            sum += acc[y * width + x];
            coverage[y * width + x] = sum.abs().min(1.0);
        }
    }
    Some(Mask {
        left,
        top,
        width,
        height,
        coverage,
    })
}

impl Path {
    /// The coverage of the path drawn with CoreGraphics font smoothing:
    /// every flattened edge moved outward by the point of the box
    /// `[-dx, dx] x [-2 dy, 0]` (y down) that is extreme in the edge's
    /// outward normal direction, the middle of the box's side when the edge
    /// is exactly horizontal or vertical, and consecutive moved edges joined
    /// by straight segments; filled as [`Path::fill`] fills.
    ///
    /// Measured on CoreGraphics (macOS 26) from about 100 px up, where the
    /// dilation stops growing: dx = dy = 0.3 px. A stem widens by 0.3 px on
    /// each side and a flat top rises by 0.6 px while the base stays; a
    /// slanted edge facing up and left moves 0.3 + 0.6 s horizontally (s its
    /// slope), its opposite edge 0.3; a sharp corner of a rectangle loses
    /// a 0.3 x 0.3 px triangle. Against CoreGraphics' smoothed masks of 8
    /// letters of Nunito, Times New Roman, Georgia and Arial at 162 and
    /// 256 px, 99.53 to 99.89 % of the pixels match within 1; a few glyph
    /// features (a straight edge running into a curve, sharp apexes and
    /// joins) leave up to 65 levels.
    pub fn fill_smoothed(mut self, dx: f64, dy: f64) -> Option<Mask> {
        self.close();
        // the fill side: the contour of largest area is an outer contour
        let area = |c: &[[(f64, f64); 2]]| -> f64 {
            c.iter().map(|[a, b]| a.0 * b.1 - b.0 * a.1).sum::<f64>() * 0.5
        };
        let mut ranges = Vec::new();
        for (k, &start) in self.contours.iter().enumerate() {
            let end = self
                .contours
                .get(k + 1)
                .copied()
                .unwrap_or(self.segments.len());
            if end > start {
                ranges.push(start..end);
            }
        }
        let orientation =
            ranges
                .iter()
                .map(|r| area(&self.segments[r.clone()]))
                .fold(
                    0.0f64,
                    |best, a| if a.abs() > best.abs() { a } else { best },
                );
        let sign = |v: f64| {
            if v > 0.0 {
                1.0
            } else if v < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
        let offset = |[a, b]: [(f64, f64); 2]| -> (f64, f64) {
            let (ex, ey) = (b.0 - a.0, b.1 - a.1);
            // outward normal: away from the fill side
            let (nx, ny) = if orientation > 0.0 {
                (ey, -ex)
            } else {
                (-ey, ex)
            };
            (dx * sign(nx), -dy + dy * sign(ny))
        };
        let mut edges = Vec::new();
        for range in ranges {
            let contour = &self.segments[range.clone()];
            let moves: Vec<(f64, f64)> = contour.iter().map(|&e| offset(e)).collect();
            for (i, &[a, b]) in contour.iter().enumerate() {
                let t = moves[i];
                let next = moves[(i + 1) % moves.len()];
                let (a1, b1) = ((a.0 + t.0, a.1 + t.1), (b.0 + t.0, b.1 + t.1));
                edges.push([a1, b1]);
                let b2 = (b.0 + next.0, b.1 + next.1);
                if b1 != b2 {
                    edges.push([b1, b2]);
                }
            }
        }
        accumulated_mask(&edges)
    }
}

/// Adds the signed area a line edge sweeps to the accumulation buffer:
/// each pixel receives the part of the edge's coverage that starts in it,
/// so the running sum along a row is the area left of the edges.
fn accumulate_line(acc: &mut [f64], width: usize, height: usize, p0: (f64, f64), p1: (f64, f64)) {
    if p0.1 == p1.1 {
        return;
    }
    let (dir, p0, p1) = if p0.1 < p1.1 {
        (1.0, p0, p1)
    } else {
        (-1.0, p1, p0)
    };
    let dxdy = (p1.0 - p0.0) / (p1.1 - p0.1);
    let y_start = p0.1.max(0.0);
    let y_end = p1.1.min(height as f64);
    if y_start >= y_end {
        return;
    }
    let mut x = p0.0 + (y_start - p0.1) * dxdy;
    let mut y = y_start.floor() as usize;
    let mut y_top = y_start;
    while (y as f64) < y_end {
        let row_bottom = ((y + 1) as f64).min(y_end);
        let dy = row_bottom - y_top;
        let x_next = x + dxdy * dy;
        let d = dy * dir;
        let (x0, x1) = if x < x_next { (x, x_next) } else { (x_next, x) };
        let line = y * width;
        let x0_floor = x0.floor();
        let x0i = x0_floor as i64;
        let x1_ceil = x1.ceil();
        let x1i = x1_ceil as i64;
        let at = |i: i64| line + i.clamp(0, width as i64 - 1) as usize;
        if x1i <= x0i + 1 {
            // within one pixel column: the area right of the edge's midpoint
            let xmf = 0.5 * (x + x_next) - x0_floor;
            acc[at(x0i)] += d - d * xmf;
            acc[at(x0i + 1)] += d * xmf;
        } else {
            let s = 1.0 / (x1 - x0);
            let x0f = x0 - x0_floor;
            let a0 = 0.5 * s * (1.0 - x0f) * (1.0 - x0f);
            let x1f = x1 - x1_ceil + 1.0;
            let am = 0.5 * s * x1f * x1f;
            acc[at(x0i)] += d * a0;
            if x1i == x0i + 2 {
                acc[at(x0i + 1)] += d * (1.0 - a0 - am);
            } else {
                let a1 = s * (1.5 - x0f);
                acc[at(x0i + 1)] += d * (a1 - a0);
                for xi in x0i + 2..x1i - 1 {
                    acc[at(xi)] += d * s;
                }
                let a2 = a1 + (x1i - x0i - 3) as f64 * s;
                acc[at(x1i - 1)] += d * (1.0 - a2 - am);
            }
            acc[at(x1i)] += d * am;
        }
        x = x_next;
        y_top = row_bottom;
        y += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_coverage_is_exact() {
        let mut p = Path::new();
        p.move_to((1.25, 1.5));
        p.line_to((3.75, 1.5));
        p.line_to((3.75, 3.0));
        p.line_to((1.25, 3.0));
        let m = p.fill().unwrap();
        let at =
            |x: i64, y: i64| m.coverage[((y - m.top) as usize) * m.width + (x - m.left) as usize];
        assert!((at(1, 1) - 0.75 * 0.5).abs() < 1e-12);
        assert!((at(2, 1) - 0.5).abs() < 1e-12);
        assert!((at(2, 2) - 1.0).abs() < 1e-12);
        assert!((at(3, 2) - 0.75).abs() < 1e-12);
        let total: f64 = m.coverage.iter().sum();
        assert!((total - 2.5 * 1.5).abs() < 1e-9);
    }

    #[test]
    fn triangle_area_is_exact() {
        let mut p = Path::new();
        p.move_to((0.3, 0.2));
        p.line_to((5.7, 1.9));
        p.line_to((2.2, 4.6));
        let m = p.fill().unwrap();
        let total: f64 = m.coverage.iter().sum();
        let area = 0.5 * ((5.7f64 - 0.3) * (4.6 - 0.2) - (2.2 - 0.3) * (1.9 - 0.2)).abs();
        assert!((total - area).abs() < 1e-9, "{total} vs {area}");
    }
}
