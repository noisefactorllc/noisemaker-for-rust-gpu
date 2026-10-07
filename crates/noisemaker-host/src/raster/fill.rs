//! A stroke wider than 1 px, as Chromium's software canvas fills it:
//! `SkStroke` turns the segment into a closed outline of lines and
//! round-cap conics, and `SkScan::AntiFillPath` fills the outline with
//! analytic anti-aliasing (`SkScan_AAAPath.cpp`), clipped to the canvas.
//!
//! The outline's convexity (`SkPathPriv::ComputeConvexity`) chooses the
//! edge walker and the additive blitter: a small outline accumulates its
//! coverage in an A8 mask that is blended once (`MaskAdditiveBlitter`), a
//! larger one in run-length rows (`RunBasedAdditiveBlitter`, or
//! `SafeRLEAdditiveBlitter` for an outline that is not convex). The mask
//! takes outlines up to 32 px wide and 1024 bytes; the tracer's strokes,
//! whose width and length scale with the canvas, exceed that only on
//! canvases above about 7500 px. The overlays measured against the
//! reference (up to 1024 px) all took the mask. A stroke whose bounds
//! reach coordinate 8192 takes Skia's non-anti-aliased `SkScan::FillPath`,
//! which this module does not model; it fills such a stroke anti-aliased.

// A port of Skia; see the license notice in `raster.rs`.

use super::{Blitter, DynBlitter, IRect, Mask, Rect, RectClipBlitter};

type Pt = (f32, f32);

fn add(a: Pt, b: Pt) -> Pt {
    (a.0 + b.0, a.1 + b.1)
}

fn sub(a: Pt, b: Pt) -> Pt {
    (a.0 - b.0, a.1 - b.1)
}

fn is_finite(p: Pt) -> bool {
    p.0.is_finite() && p.1.is_finite()
}

// ------------------------------------------------------------------ path

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verb {
    Move,
    Line,
    Conic,
    Close,
}

/// An `SkPathBuilder`'s points, verbs and conic weights.
#[derive(Clone, Debug, Default)]
struct Path {
    pts: Vec<Pt>,
    verbs: Vec<Verb>,
    weights: Vec<f32>,
}

impl Path {
    fn move_to(&mut self, p: Pt) {
        self.pts.push(p);
        self.verbs.push(Verb::Move);
    }
    fn line_to(&mut self, p: Pt) {
        self.pts.push(p);
        self.verbs.push(Verb::Line);
    }
    /// `conicTo` with the stroker's weight (finite, not 0 or 1).
    fn conic_to(&mut self, p1: Pt, p2: Pt, w: f32) {
        self.pts.push(p1);
        self.pts.push(p2);
        self.verbs.push(Verb::Conic);
        self.weights.push(w);
    }
    fn close(&mut self) {
        self.verbs.push(Verb::Close);
    }

    /// `computeFiniteBounds`: the bounds of every point, `None` when one is
    /// not finite.
    fn bounds(&self) -> Option<Rect> {
        let mut r = Rect {
            left: f32::INFINITY,
            top: f32::INFINITY,
            right: f32::NEG_INFINITY,
            bottom: f32::NEG_INFINITY,
        };
        for &p in &self.pts {
            if !is_finite(p) {
                return None;
            }
            r.left = r.left.min(p.0);
            r.top = r.top.min(p.1);
            r.right = r.right.max(p.0);
            r.bottom = r.bottom.max(p.1);
        }
        (!self.pts.is_empty()).then_some(r)
    }

    /// The path's edges as `SkPathEdgeIter` returns them: each segment with
    /// its start point, and the implicit line that closes a contour.
    fn edges(&self) -> Vec<PathEdge> {
        let mut out = Vec::new();
        let (mut pi, mut wi) = (0usize, 0usize);
        let mut move_to = 0usize;
        let mut needs_close = false;
        let close_line = |out: &mut Vec<PathEdge>, last: usize, move_to: usize, pts: &[Pt]| {
            out.push(PathEdge::Line([pts[last], pts[move_to]]));
        };
        for &verb in &self.verbs {
            match verb {
                Verb::Move => {
                    if needs_close {
                        close_line(&mut out, pi - 1, move_to, &self.pts);
                        needs_close = false;
                    }
                    move_to = pi;
                    pi += 1;
                }
                Verb::Close => {
                    if needs_close {
                        close_line(&mut out, pi - 1, move_to, &self.pts);
                        needs_close = false;
                    }
                }
                Verb::Line => {
                    out.push(PathEdge::Line([self.pts[pi - 1], self.pts[pi]]));
                    pi += 1;
                    needs_close = true;
                }
                Verb::Conic => {
                    out.push(PathEdge::Conic(
                        [self.pts[pi - 1], self.pts[pi], self.pts[pi + 1]],
                        self.weights[wi],
                    ));
                    pi += 2;
                    wi += 1;
                    needs_close = true;
                }
            }
        }
        if needs_close {
            close_line(&mut out, pi - 1, move_to, &self.pts);
        }
        out
    }
}

enum PathEdge {
    Line([Pt; 2]),
    Conic([Pt; 3], f32),
}

// --------------------------------------------------------------- stroker

/// `SK_ScalarRoot2Over2`.
const ROOT2_OVER2: f32 = 0.707_106_77;

/// `SkStroke::strokePath` of `moveTo(p0) lineTo(p1)` with round caps (a
/// single segment has no join) at `width` > 0: the outer line, the end
/// cap, the reversed inner line and the start cap, closed.
fn stroke_line(p0: Pt, p1: Pt, width: f32) -> Path {
    let radius = width / 2.0;
    // set_normal_unitnormal: the unit vector rotated CCW, scaled by the
    // radius; round caps draw a zero-length segment upright.
    let normal = match super::normalize(p1.0 - p0.0, p1.1 - p0.1) {
        Some((ux, uy)) => (uy * radius, -ux * radius),
        None => (radius, 0.0),
    };
    let first_outer = add(p0, normal);
    let mut path = Path::default();
    path.move_to(first_outer);
    path.line_to(add(p1, normal));
    round_cap(&mut path, p1, normal, sub(p1, normal));
    path.line_to(sub(p0, normal));
    round_cap(&mut path, p0, (-normal.0, -normal.1), first_outer);
    path.close();
    path
}

/// `RoundCapper`: two quarter-circle conics around `pivot`.
fn round_cap(path: &mut Path, pivot: Pt, normal: Pt, stop: Pt) {
    let parallel = (-normal.1, normal.0); // RotateCW
    let center = add(pivot, parallel);
    path.conic_to(add(center, normal), center, ROOT2_OVER2);
    path.conic_to(sub(center, normal), stop, ROOT2_OVER2);
}

// ------------------------------------------------------------- convexity

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirChange {
    Unknown,
    Left,
    Right,
    Straight,
    Backwards,
    Invalid,
}

/// `Convexicator`.
struct Convexicator {
    first_pt: Pt,
    first_vec: Pt,
    last_pt: Pt,
    last_vec: Pt,
    expected_dir: DirChange,
    first_direction_known: bool,
    reversals: i32,
}

impl Convexicator {
    fn new() -> Convexicator {
        Convexicator {
            first_pt: (0.0, 0.0),
            first_vec: (0.0, 0.0),
            last_pt: (0.0, 0.0),
            last_vec: (0.0, 0.0),
            expected_dir: DirChange::Invalid,
            first_direction_known: false,
            reversals: 0,
        }
    }

    fn set_move_pt(&mut self, p: Pt) {
        self.first_pt = p;
        self.last_pt = p;
        self.expected_dir = DirChange::Invalid;
    }

    fn add_pt(&mut self, p: Pt) -> bool {
        if self.last_pt == p {
            return true;
        }
        if self.first_pt == self.last_pt
            && self.expected_dir == DirChange::Invalid
            && self.last_vec == (0.0, 0.0)
        {
            self.last_vec = sub(p, self.last_pt);
            self.first_vec = self.last_vec;
        } else if !self.add_vec(sub(p, self.last_pt)) {
            return false;
        }
        self.last_pt = p;
        true
    }

    fn close(&mut self) -> bool {
        self.add_pt(self.first_pt) && self.add_vec(self.first_vec)
    }

    fn direction_change(&self, cur: Pt) -> DirChange {
        let cross = self.last_vec.0 * cur.1 - self.last_vec.1 * cur.0;
        if !cross.is_finite() {
            return DirChange::Unknown;
        }
        if cross == 0.0 {
            let dot = self.last_vec.0 * cur.0 + self.last_vec.1 * cur.1;
            return if dot < 0.0 {
                DirChange::Backwards
            } else {
                DirChange::Straight
            };
        }
        if cross > 0.0 {
            DirChange::Right
        } else {
            DirChange::Left
        }
    }

    fn add_vec(&mut self, cur: Pt) -> bool {
        match self.direction_change(cur) {
            dir @ (DirChange::Left | DirChange::Right) => {
                if self.expected_dir == DirChange::Invalid {
                    self.expected_dir = dir;
                    self.first_direction_known = true;
                } else if dir != self.expected_dir {
                    self.first_direction_known = false;
                    return false;
                }
                self.last_vec = cur;
                true
            }
            DirChange::Straight => true,
            DirChange::Backwards => {
                self.last_vec = cur;
                self.reversals += 1;
                self.reversals < 3
            }
            DirChange::Unknown => false,
            DirChange::Invalid => unreachable!("invalid direction change"),
        }
    }

    /// `IsConcaveBySign`: more than three sign changes of dx or dy.
    fn is_concave_by_sign(points: &[Pt]) -> bool {
        if points.len() <= 3 {
            return false;
        }
        let sign = |x: f32| (x < 0.0) as i32;
        let mut curr = points[0];
        let first = curr;
        let (mut dxes, mut dyes) = (0, 0);
        let (mut last_sx, mut last_sy) = (2, 2);
        // The points after the first, then the closing vector to the first.
        for &next in points[1..].iter().chain(std::iter::once(&first)) {
            let vec = sub(next, curr);
            if vec != (0.0, 0.0) {
                if !is_finite(vec) {
                    return true;
                }
                let (sx, sy) = (sign(vec.0), sign(vec.1));
                dxes += (sx != last_sx) as i32;
                dyes += (sy != last_sy) as i32;
                if dxes > 3 || dyes > 3 {
                    return true;
                }
                last_sx = sx;
                last_sy = sy;
            }
            curr = next;
        }
        false
    }
}

/// `SkPathPriv::ComputeConvexity(...)` is convex (any direction or
/// degenerate) for the stroker's single closed contour.
fn is_convex(path: &Path) -> bool {
    if path.verbs.is_empty() {
        return true;
    }
    if Convexicator::is_concave_by_sign(&path.pts) {
        return false;
    }
    let mut state = Convexicator::new();
    let mut contour_count = 0;
    let mut needs_close = false;
    let mut pi = 0usize;
    for &verb in &path.verbs {
        let new_pts: &[Pt] = match verb {
            Verb::Move => &path.pts[pi..pi + 1],
            Verb::Line => &path.pts[pi..pi + 1],
            Verb::Conic => &path.pts[pi..pi + 2],
            Verb::Close => &[],
        };
        if contour_count == 0 {
            if verb == Verb::Move {
                state.set_move_pt(new_pts[0]);
            } else {
                contour_count += 1;
                needs_close = true;
            }
        }
        if contour_count == 1 {
            if verb == Verb::Close || verb == Verb::Move {
                if !state.close() {
                    return false;
                }
                needs_close = false;
                contour_count += 1;
            } else {
                for &p in new_pts {
                    if !state.add_pt(p) {
                        return false;
                    }
                }
            }
        } else if contour_count > 1 && verb != Verb::Move {
            return false;
        }
        pi += new_pts.len();
    }
    if needs_close && !state.close() {
        return false;
    }
    !(!state.first_direction_known && state.reversals >= 3)
}

// --------------------------------------------------------------- geometry

/// `valid_unit_divide`.
fn valid_unit_divide(mut numer: f32, mut denom: f32) -> Option<f32> {
    if numer < 0.0 {
        numer = -numer;
        denom = -denom;
    }
    if denom == 0.0 || numer == 0.0 || numer >= denom {
        return None;
    }
    let r = numer / denom;
    if r.is_nan() || r == 0.0 {
        return None;
    }
    Some(r)
}

/// `is_not_monotonic(a, b, c)`.
fn is_not_monotonic(a: f32, b: f32, c: f32) -> bool {
    let ab = a - b;
    let mut bc = b - c;
    if ab < 0.0 {
        bc = -bc;
    }
    ab == 0.0 || bc < 0.0
}

fn interp(a: Pt, b: Pt, t: f32) -> Pt {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

/// `SkChopQuadAt(src, dst, t)`.
fn chop_quad_at(src: &[Pt; 3], t: f32) -> [Pt; 5] {
    let p01 = interp(src[0], src[1], t);
    let p12 = interp(src[1], src[2], t);
    [src[0], p01, interp(p01, p12, t), p12, src[2]]
}

/// `SkChopQuadAtYExtrema` (`y` true) or `SkChopQuadAtXExtrema`: one or two
/// monotonic quads.
fn chop_quad_at_extrema(src: &[Pt; 3], y: bool) -> Vec<[Pt; 3]> {
    let get = |p: Pt| if y { p.1 } else { p.0 };
    let (a, mut b, c) = (get(src[0]), get(src[1]), get(src[2]));
    if is_not_monotonic(a, b, c) {
        if let Some(t) = valid_unit_divide(a - b, a - b - b + c) {
            let mut dst = chop_quad_at(src, t);
            // flatten_double_quad_extrema
            let mid = get(dst[2]);
            for i in [1, 3] {
                if y {
                    dst[i].1 = mid;
                } else {
                    dst[i].0 = mid;
                }
            }
            return vec![[dst[0], dst[1], dst[2]], [dst[2], dst[3], dst[4]]];
        }
        b = if (a - b).abs() < (b - c).abs() { a } else { c };
    }
    let mut out = *src;
    if y {
        out[1].1 = b;
    } else {
        out[1].0 = b;
    }
    vec![out]
}

/// An `SkConic`.
#[derive(Clone, Copy, Debug)]
struct Conic {
    pts: [Pt; 3],
    w: f32,
}

impl Conic {
    /// `SkConic::chop`, the `SK_SUPPORT_LEGACY_CONIC_CHOP` form Chromium
    /// builds.
    fn chop(&self) -> [Conic; 2] {
        let scale = 1.0f32 / (1.0 + self.w);
        let new_w = (0.5f32 + self.w * 0.5).sqrt();
        let [p0, p1, p2] = self.pts;
        let wp1 = (self.w * p1.0, self.w * p1.1);
        let m = |a: f32, b: f32, c: f32| ((a + (b + b)) + c) * scale * 0.5;
        let mut mid = (m(p0.0, wp1.0, p2.0), m(p0.1, wp1.1, p2.1));
        if !is_finite(mid) {
            let w_d = self.w as f64;
            let w_2 = w_d * 2.0;
            let scale_half = 1.0 / (1.0 + w_d) * 0.5;
            mid = (
                ((p0.0 as f64 + w_2 * p1.0 as f64 + p2.0 as f64) * scale_half) as f32,
                ((p0.1 as f64 + w_2 * p1.1 as f64 + p2.1 as f64) * scale_half) as f32,
            );
        }
        let c0 = ((p0.0 + wp1.0) * scale, (p0.1 + wp1.1) * scale);
        let c1 = ((wp1.0 + p2.0) * scale, (wp1.1 + p2.1) * scale);
        [
            Conic {
                pts: [p0, c0, mid],
                w: new_w,
            },
            Conic {
                pts: [mid, c1, p2],
                w: new_w,
            },
        ]
    }

    /// `computeQuadPOW2(tol)`.
    fn quad_pow2(&self, tol: f32) -> u32 {
        if tol < 0.0
            || !tol.is_finite()
            || !self.pts.iter().all(|&p| is_finite(p))
            || self.w < 0.0
            || !self.w.is_finite()
        {
            return 0;
        }
        let a = self.w - 1.0;
        let k = a / (4.0 * (2.0 + a));
        let [p0, p1, p2] = self.pts;
        let x = k * (p0.0 - 2.0 * p1.0 + p2.0);
        let y = k * (p0.1 - 2.0 * p1.1 + p2.1);
        let mut error = (x * x + y * y).sqrt();
        let mut pow2 = 0;
        while pow2 < 5 {
            if error <= tol {
                break;
            }
            error *= 0.25;
            pow2 += 1;
        }
        pow2
    }
}

fn between(a: f32, b: f32, c: f32) -> bool {
    (a - b) * (c - b) <= 0.0
}

/// `subdivide(src, pts, level)`: appends each quad's control and end point.
fn subdivide(src: &Conic, out: &mut Vec<Pt>, level: u32) {
    if level == 0 {
        out.push(src.pts[1]);
        out.push(src.pts[2]);
        return;
    }
    let mut dst = src.chop();
    let start_y = src.pts[0].1;
    let end_y = src.pts[2].1;
    if between(start_y, src.pts[1].1, end_y) {
        let mid_y = dst[0].pts[2].1;
        if !between(start_y, mid_y, end_y) {
            let closer = if (mid_y - start_y).abs() < (mid_y - end_y).abs() {
                start_y
            } else {
                end_y
            };
            dst[0].pts[2].1 = closer;
            dst[1].pts[0].1 = closer;
        }
        if !between(start_y, dst[0].pts[1].1, dst[0].pts[2].1) {
            dst[0].pts[1].1 = start_y;
        }
        if !between(dst[1].pts[0].1, dst[1].pts[1].1, end_y) {
            dst[1].pts[1].1 = end_y;
        }
    }
    subdivide(&dst[0], out, level - 1);
    subdivide(&dst[1], out, level - 1);
}

/// `SkAutoConicToQuads::computeQuads(pts, w, tol)`: 1 + 2n points of n
/// quads.
fn conic_to_quads(pts: [Pt; 3], w: f32, tol: f32) -> Vec<Pt> {
    if w <= 0.0 {
        let mid = ((pts[0].0 + pts[2].0) * 0.5, (pts[0].1 + pts[2].1) * 0.5);
        return vec![pts[0], mid, pts[2]];
    }
    let conic = Conic { pts, w };
    let mut pow2 = conic.quad_pow2(tol);
    if w < 0.0 || !w.is_finite() {
        pow2 = 0;
    }
    let mut out = vec![pts[0]];
    let mut lines = false;
    if pow2 == 5 {
        let dst = conic.chop();
        let near = |a: Pt, b: Pt| {
            let (dx, dy) = (a.0 - b.0, a.1 - b.1);
            !(dx.is_finite() && dy.is_finite() && (dx != 0.0 || dy != 0.0))
        };
        if near(dst[0].pts[1], dst[0].pts[2]) && near(dst[1].pts[0], dst[1].pts[1]) {
            out.extend([dst[0].pts[1], dst[0].pts[1], dst[0].pts[1], dst[1].pts[2]]);
            lines = true;
        }
    }
    if !lines {
        subdivide(&conic, &mut out, pow2);
    }
    if !out.iter().all(|&p| is_finite(p)) {
        let n = out.len();
        for p in &mut out[1..n - 1] {
            *p = pts[1];
        }
    }
    out
}

/// The quads of a conic edge, `kConicTol` 0.25.
fn conic_quads(pts: [Pt; 3], w: f32) -> Vec<[Pt; 3]> {
    let q = conic_to_quads(pts, w, 0.25);
    (0..(q.len() - 1) / 2)
        .map(|i| [q[2 * i], q[2 * i + 1], q[2 * i + 2]])
        .collect()
}

/// `SkFindUnitQuadRoots`: the first root in (0, 1).
fn first_unit_quad_root(a: f32, b: f32, c: f32) -> Option<f32> {
    if a == 0.0 {
        return valid_unit_divide(-c, b);
    }
    let dr = (b as f64) * (b as f64) - 4.0 * (a as f64) * (c as f64);
    if dr < 0.0 {
        return None;
    }
    let r = dr.sqrt() as f32;
    if !r.is_finite() {
        return None;
    }
    let q = if b < 0.0 {
        -(b - r) / 2.0
    } else {
        -(b + r) / 2.0
    };
    let mut roots = Vec::with_capacity(2);
    roots.extend(valid_unit_divide(q, a));
    roots.extend(valid_unit_divide(c, q));
    if roots.len() == 2 && roots[0] > roots[1] {
        roots.swap(0, 1);
    }
    roots.first().copied()
}

// ------------------------------------------------------------ edge clipper

/// `sect_with_horizontal` / `sect_clamp_with_vertical` from SkLineClipper.
fn clamp_sect_with_vertical(src: &[Pt; 2], x: f32) -> f32 {
    let y = super::sect_with_vertical(src, x) as f64;
    super::pin_unsorted(y, src[0].1 as f64, src[1].1 as f64) as f32
}

/// `SkLineClipper::ClipLine`: 0 to 3 segments wholly inside `clip` in X
/// (vertical ones on its sides), as a polyline.
fn clip_line(pts: &[Pt; 2], clip: &Rect, can_cull_right: bool) -> Vec<Pt> {
    let (i0, i1) = if pts[0].1 < pts[1].1 { (0, 1) } else { (1, 0) };
    if pts[i1].1 <= clip.top || pts[i0].1 >= clip.bottom {
        return Vec::new();
    }
    let mut tmp = *pts;
    if pts[i0].1 < clip.top {
        tmp[i0] = (super::sect_with_horizontal(pts, clip.top), clip.top);
    }
    if tmp[i1].1 > clip.bottom {
        tmp[i1] = (super::sect_with_horizontal(pts, clip.bottom), clip.bottom);
    }
    let (i0, i1, mut reverse) = if pts[0].0 < pts[1].0 {
        (0, 1, false)
    } else {
        (1, 0, true)
    };
    let result: Vec<Pt>;
    if tmp[i1].0 <= clip.left {
        tmp[0].0 = clip.left;
        tmp[1].0 = clip.left;
        result = tmp.to_vec();
        reverse = false;
    } else if tmp[i0].0 >= clip.right {
        if can_cull_right {
            return Vec::new();
        }
        tmp[0].0 = clip.right;
        tmp[1].0 = clip.right;
        result = tmp.to_vec();
        reverse = false;
    } else {
        let mut r = Vec::with_capacity(4);
        if tmp[i0].0 < clip.left {
            r.push((clip.left, tmp[i0].1));
            r.push((clip.left, clamp_sect_with_vertical(&tmp, clip.left)));
        } else {
            r.push(tmp[i0]);
        }
        if tmp[i1].0 > clip.right {
            r.push((clip.right, clamp_sect_with_vertical(&tmp, clip.right)));
            r.push((clip.right, tmp[i1].1));
        } else {
            r.push(tmp[i1]);
        }
        result = r;
    }
    if reverse {
        result.into_iter().rev().collect()
    } else {
        result
    }
}

/// A clipped edge: `SkEdgeClipper`'s line and quad verbs.
#[derive(Clone, Copy, Debug)]
enum ClippedEdge {
    Line([Pt; 2]),
    Quad([Pt; 3]),
}

fn chop_mono_quad_at(c0: f32, c1: f32, c2: f32, target: f32) -> Option<f32> {
    let a = c0 - c1 - c1 + c2;
    let b = 2.0 * (c1 - c0);
    let c = c0 - target;
    first_unit_quad_root(a, b, c)
}

/// `chop_quad_in_Y`.
fn chop_quad_in_y(pts: &mut [Pt; 3], clip: &Rect) {
    if pts[0].1 < clip.top {
        if let Some(t) = chop_mono_quad_at(pts[0].1, pts[1].1, pts[2].1, clip.top) {
            let mut tmp = chop_quad_at(pts, t);
            tmp[2].1 = clip.top;
            if tmp[3].1 < clip.top {
                tmp[3].1 = clip.top;
            }
            pts[0] = tmp[2];
            pts[1] = tmp[3];
        } else {
            for p in pts.iter_mut() {
                if p.1 < clip.top {
                    p.1 = clip.top;
                }
            }
        }
    }
    if pts[2].1 > clip.bottom {
        if let Some(t) = chop_mono_quad_at(pts[0].1, pts[1].1, pts[2].1, clip.bottom) {
            let mut tmp = chop_quad_at(pts, t);
            if tmp[1].1 > clip.bottom {
                tmp[1].1 = clip.bottom;
            }
            tmp[2].1 = clip.bottom;
            pts[1] = tmp[1];
            pts[2] = tmp[2];
        } else {
            for p in pts.iter_mut() {
                if p.1 > clip.bottom {
                    p.1 = clip.bottom;
                }
            }
        }
    }
}

struct EdgeClipper {
    can_cull_right: bool,
    out: Vec<ClippedEdge>,
}

impl EdgeClipper {
    fn append_vline(&mut self, x: f32, mut y0: f32, mut y1: f32, reverse: bool) {
        if reverse {
            std::mem::swap(&mut y0, &mut y1);
        }
        self.out.push(ClippedEdge::Line([(x, y0), (x, y1)]));
    }

    fn append_quad(&mut self, pts: &[Pt; 3], reverse: bool) {
        self.out.push(ClippedEdge::Quad(if reverse {
            [pts[2], pts[1], pts[0]]
        } else {
            *pts
        }));
    }

    fn clip_line(&mut self, p0: Pt, p1: Pt, clip: &Rect) {
        let lines = clip_line(&[p0, p1], clip, self.can_cull_right);
        for pair in lines.windows(2) {
            self.out.push(ClippedEdge::Line([pair[0], pair[1]]));
        }
    }

    /// `clipMonoQuad` of a quad monotonic in X and Y.
    fn clip_mono_quad(&mut self, src: &[Pt; 3], clip: &Rect) {
        let (mut pts, mut reverse) = if src[0].1 > src[2].1 {
            ([src[2], src[1], src[0]], true)
        } else {
            (*src, false)
        };
        if pts[2].1 <= clip.top || pts[0].1 >= clip.bottom {
            return;
        }
        chop_quad_in_y(&mut pts, clip);
        if pts[0].0 > pts[2].0 {
            pts.swap(0, 2);
            reverse = !reverse;
        }
        if pts[2].0 <= clip.left {
            self.append_vline(clip.left, pts[0].1, pts[2].1, reverse);
            return;
        }
        if pts[0].0 >= clip.right {
            if !self.can_cull_right {
                self.append_vline(clip.right, pts[0].1, pts[2].1, reverse);
            }
            return;
        }
        if pts[0].0 < clip.left {
            match chop_mono_quad_at(pts[0].0, pts[1].0, pts[2].0, clip.left) {
                Some(t) => {
                    let mut tmp = chop_quad_at(&pts, t);
                    self.append_vline(clip.left, tmp[0].1, tmp[2].1, reverse);
                    tmp[2].0 = clip.left;
                    if tmp[3].0 < clip.left {
                        tmp[3].0 = clip.left;
                    }
                    pts[0] = tmp[2];
                    pts[1] = tmp[3];
                }
                None => {
                    self.append_vline(clip.left, pts[0].1, pts[2].1, reverse);
                    return;
                }
            }
        }
        if pts[2].0 > clip.right {
            match chop_mono_quad_at(pts[0].0, pts[1].0, pts[2].0, clip.right) {
                Some(t) => {
                    let mut tmp = chop_quad_at(&pts, t);
                    if tmp[1].0 > clip.right {
                        tmp[1].0 = clip.right;
                    }
                    tmp[2].0 = clip.right;
                    self.append_quad(&[tmp[0], tmp[1], tmp[2]], reverse);
                    self.append_vline(clip.right, tmp[2].1, tmp[4].1, reverse);
                }
                None => {
                    pts[1].0 = pts[1].0.min(clip.right);
                    pts[2].0 = pts[2].0.min(clip.right);
                    self.append_quad(&pts, reverse);
                }
            }
        } else {
            self.append_quad(&pts, reverse);
        }
    }

    /// `clipQuad`.
    fn clip_quad(&mut self, src: &[Pt; 3], clip: &Rect) {
        let top = src.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let bottom = src.iter().map(|p| p.1).fold(f32::NEG_INFINITY, f32::max);
        if top >= clip.bottom || bottom <= clip.top {
            return;
        }
        for mono_y in chop_quad_at_extrema(src, true) {
            for mono_x in chop_quad_at_extrema(&mono_y, false) {
                self.clip_mono_quad(&mono_x, clip);
            }
        }
    }
}

// ----------------------------------------------------------- fixed point

type Fixed = i32;
type FDot6 = i32;

const FIXED1: Fixed = 1 << 16;
const FIXED_HALF: Fixed = 1 << 15;
const MAX_S32: i32 = i32::MAX;
const MIN_S32: i32 = i32::MIN;
const DEFAULT_ACCURACY: i32 = 2;

fn fixed_mul(a: Fixed, b: Fixed) -> Fixed {
    ((a as i64 * b as i64) >> 16) as i32
}

fn fixed_floor(x: Fixed) -> i32 {
    x >> 16
}

fn fixed_ceil(x: Fixed) -> i32 {
    x.wrapping_add(FIXED1 - 1) >> 16
}

fn fixed_round_to_int(x: Fixed) -> i32 {
    x.wrapping_add(FIXED_HALF) >> 16
}

fn fixed_round_to_fixed(x: Fixed) -> Fixed {
    (x.wrapping_add(FIXED_HALF) as u32 & 0xFFFF_0000) as i32
}

fn fixed_ceil_to_fixed(x: Fixed) -> Fixed {
    (x.wrapping_add(FIXED1 - 1) as u32 & 0xFFFF_0000) as i32
}

fn fixed_floor_to_fixed(x: Fixed) -> Fixed {
    (x as u32 & 0xFFFF_0000) as i32
}

fn int_to_fixed(n: i32) -> Fixed {
    ((n as u32) << 16) as i32
}

fn fixed_to_fdot6(x: Fixed) -> FDot6 {
    x >> 10
}

fn fdot6_to_fixed(x: FDot6) -> Fixed {
    ((x as u32) << 10) as i32
}

fn sat_add(a: i32, b: i32) -> i32 {
    (a as i64 + b as i64).clamp(MIN_S32 as i64, MAX_S32 as i64) as i32
}

fn sat_sub(a: i32, b: i32) -> i32 {
    (a as i64 - b as i64).clamp(MIN_S32 as i64, MAX_S32 as i64) as i32
}

/// `SkAnalyticEdge::SnapY`.
fn snap_y(y: Fixed) -> Fixed {
    const ACC: u32 = DEFAULT_ACCURACY as u32;
    (((y as u32).wrapping_add((FIXED1 as u32) >> (ACC + 1)) >> (16 - ACC)) << (16 - ACC)) as i32
}

/// `quick_inverse`'s table: -(65536 * 64) / (1024 - i) for i in 0..1024, and 0.
#[rustfmt::skip]
static INVERSE_TABLE: [i32; 1025] = [
    -4096, -4100, -4104, -4108, -4112, -4116, -4120, -4124, -4128, -4132, -4136, -4140,
    -4144, -4148, -4152, -4156, -4161, -4165, -4169, -4173, -4177, -4181, -4185, -4190,
    -4194, -4198, -4202, -4206, -4211, -4215, -4219, -4223, -4228, -4232, -4236, -4240,
    -4245, -4249, -4253, -4258, -4262, -4266, -4271, -4275, -4279, -4284, -4288, -4293,
    -4297, -4301, -4306, -4310, -4315, -4319, -4324, -4328, -4332, -4337, -4341, -4346,
    -4350, -4355, -4359, -4364, -4369, -4373, -4378, -4382, -4387, -4391, -4396, -4401,
    -4405, -4410, -4415, -4419, -4424, -4429, -4433, -4438, -4443, -4447, -4452, -4457,
    -4462, -4466, -4471, -4476, -4481, -4485, -4490, -4495, -4500, -4505, -4510, -4514,
    -4519, -4524, -4529, -4534, -4539, -4544, -4549, -4554, -4559, -4563, -4568, -4573,
    -4578, -4583, -4588, -4593, -4599, -4604, -4609, -4614, -4619, -4624, -4629, -4634,
    -4639, -4644, -4650, -4655, -4660, -4665, -4670, -4675, -4681, -4686, -4691, -4696,
    -4702, -4707, -4712, -4718, -4723, -4728, -4733, -4739, -4744, -4750, -4755, -4760,
    -4766, -4771, -4777, -4782, -4788, -4793, -4798, -4804, -4809, -4815, -4821, -4826,
    -4832, -4837, -4843, -4848, -4854, -4860, -4865, -4871, -4877, -4882, -4888, -4894,
    -4899, -4905, -4911, -4917, -4922, -4928, -4934, -4940, -4946, -4951, -4957, -4963,
    -4969, -4975, -4981, -4987, -4993, -4999, -5005, -5011, -5017, -5023, -5029, -5035,
    -5041, -5047, -5053, -5059, -5065, -5071, -5077, -5084, -5090, -5096, -5102, -5108,
    -5115, -5121, -5127, -5133, -5140, -5146, -5152, -5159, -5165, -5171, -5178, -5184,
    -5190, -5197, -5203, -5210, -5216, -5223, -5229, -5236, -5242, -5249, -5256, -5262,
    -5269, -5275, -5282, -5289, -5295, -5302, -5309, -5315, -5322, -5329, -5336, -5343,
    -5349, -5356, -5363, -5370, -5377, -5384, -5391, -5398, -5405, -5412, -5418, -5426,
    -5433, -5440, -5447, -5454, -5461, -5468, -5475, -5482, -5489, -5497, -5504, -5511,
    -5518, -5526, -5533, -5540, -5548, -5555, -5562, -5570, -5577, -5584, -5592, -5599,
    -5607, -5614, -5622, -5629, -5637, -5645, -5652, -5660, -5667, -5675, -5683, -5691,
    -5698, -5706, -5714, -5722, -5729, -5737, -5745, -5753, -5761, -5769, -5777, -5785,
    -5793, -5801, -5809, -5817, -5825, -5833, -5841, -5849, -5857, -5866, -5874, -5882,
    -5890, -5899, -5907, -5915, -5924, -5932, -5940, -5949, -5957, -5966, -5974, -5983,
    -5991, -6000, -6009, -6017, -6026, -6034, -6043, -6052, -6061, -6069, -6078, -6087,
    -6096, -6105, -6114, -6123, -6132, -6141, -6150, -6159, -6168, -6177, -6186, -6195,
    -6204, -6213, -6223, -6232, -6241, -6250, -6260, -6269, -6278, -6288, -6297, -6307,
    -6316, -6326, -6335, -6345, -6355, -6364, -6374, -6384, -6393, -6403, -6413, -6423,
    -6432, -6442, -6452, -6462, -6472, -6482, -6492, -6502, -6512, -6523, -6533, -6543,
    -6553, -6563, -6574, -6584, -6594, -6605, -6615, -6626, -6636, -6647, -6657, -6668,
    -6678, -6689, -6700, -6710, -6721, -6732, -6743, -6754, -6765, -6775, -6786, -6797,
    -6808, -6820, -6831, -6842, -6853, -6864, -6875, -6887, -6898, -6909, -6921, -6932,
    -6944, -6955, -6967, -6978, -6990, -7002, -7013, -7025, -7037, -7049, -7061, -7073,
    -7084, -7096, -7108, -7121, -7133, -7145, -7157, -7169, -7182, -7194, -7206, -7219,
    -7231, -7244, -7256, -7269, -7281, -7294, -7307, -7319, -7332, -7345, -7358, -7371,
    -7384, -7397, -7410, -7423, -7436, -7449, -7463, -7476, -7489, -7503, -7516, -7530,
    -7543, -7557, -7570, -7584, -7598, -7612, -7626, -7639, -7653, -7667, -7681, -7695,
    -7710, -7724, -7738, -7752, -7767, -7781, -7796, -7810, -7825, -7839, -7854, -7869,
    -7884, -7898, -7913, -7928, -7943, -7958, -7973, -7989, -8004, -8019, -8035, -8050,
    -8065, -8081, -8097, -8112, -8128, -8144, -8160, -8176, -8192, -8208, -8224, -8240,
    -8256, -8272, -8289, -8305, -8322, -8338, -8355, -8371, -8388, -8405, -8422, -8439,
    -8456, -8473, -8490, -8507, -8525, -8542, -8559, -8577, -8594, -8612, -8630, -8648,
    -8665, -8683, -8701, -8719, -8738, -8756, -8774, -8793, -8811, -8830, -8848, -8867,
    -8886, -8905, -8924, -8943, -8962, -8981, -9000, -9020, -9039, -9058, -9078, -9098,
    -9118, -9137, -9157, -9177, -9198, -9218, -9238, -9258, -9279, -9300, -9320, -9341,
    -9362, -9383, -9404, -9425, -9446, -9467, -9489, -9510, -9532, -9554, -9576, -9597,
    -9619, -9642, -9664, -9686, -9709, -9731, -9754, -9776, -9799, -9822, -9845, -9868,
    -9892, -9915, -9939, -9962, -9986, -10010, -10034, -10058, -10082, -10106, -10131, -10155,
    -10180, -10205, -10230, -10255, -10280, -10305, -10330, -10356, -10381, -10407, -10433, -10459,
    -10485, -10512, -10538, -10564, -10591, -10618, -10645, -10672, -10699, -10727, -10754, -10782,
    -10810, -10837, -10866, -10894, -10922, -10951, -10979, -11008, -11037, -11066, -11096, -11125,
    -11155, -11184, -11214, -11244, -11275, -11305, -11335, -11366, -11397, -11428, -11459, -11491,
    -11522, -11554, -11586, -11618, -11650, -11683, -11715, -11748, -11781, -11814, -11848, -11881,
    -11915, -11949, -11983, -12018, -12052, -12087, -12122, -12157, -12192, -12228, -12264, -12300,
    -12336, -12372, -12409, -12446, -12483, -12520, -12557, -12595, -12633, -12671, -12710, -12748,
    -12787, -12826, -12865, -12905, -12945, -12985, -13025, -13066, -13107, -13148, -13189, -13231,
    -13273, -13315, -13357, -13400, -13443, -13486, -13530, -13573, -13617, -13662, -13706, -13751,
    -13797, -13842, -13888, -13934, -13981, -14027, -14074, -14122, -14169, -14217, -14266, -14315,
    -14364, -14413, -14463, -14513, -14563, -14614, -14665, -14716, -14768, -14820, -14873, -14926,
    -14979, -15033, -15087, -15141, -15196, -15252, -15307, -15363, -15420, -15477, -15534, -15592,
    -15650, -15709, -15768, -15827, -15887, -15947, -16008, -16070, -16131, -16194, -16256, -16320,
    -16384, -16448, -16513, -16578, -16644, -16710, -16777, -16844, -16912, -16980, -17050, -17119,
    -17189, -17260, -17331, -17403, -17476, -17549, -17623, -17697, -17772, -17848, -17924, -18001,
    -18078, -18157, -18236, -18315, -18396, -18477, -18558, -18641, -18724, -18808, -18893, -18978,
    -19065, -19152, -19239, -19328, -19418, -19508, -19599, -19691, -19784, -19878, -19972, -20068,
    -20164, -20262, -20360, -20460, -20560, -20661, -20763, -20867, -20971, -21076, -21183, -21290,
    -21399, -21509, -21620, -21732, -21845, -21959, -22075, -22192, -22310, -22429, -22550, -22671,
    -22795, -22919, -23045, -23172, -23301, -23431, -23563, -23696, -23831, -23967, -24105, -24244,
    -24385, -24528, -24672, -24818, -24966, -25115, -25266, -25420, -25575, -25731, -25890, -26051,
    -26214, -26379, -26546, -26715, -26886, -27060, -27235, -27413, -27594, -27776, -27962, -28149,
    -28339, -28532, -28728, -28926, -29127, -29330, -29537, -29746, -29959, -30174, -30393, -30615,
    -30840, -31068, -31300, -31536, -31775, -32017, -32263, -32513, -32768, -33026, -33288, -33554,
    -33825, -34100, -34379, -34663, -34952, -35246, -35544, -35848, -36157, -36472, -36792, -37117,
    -37449, -37786, -38130, -38479, -38836, -39199, -39568, -39945, -40329, -40721, -41120, -41527,
    -41943, -42366, -42799, -43240, -43690, -44150, -44620, -45100, -45590, -46091, -46603, -47127,
    -47662, -48210, -48770, -49344, -49932, -50533, -51150, -51781, -52428, -53092, -53773, -54471,
    -55188, -55924, -56679, -57456, -58254, -59074, -59918, -60787, -61680, -62601, -63550, -64527,
    -65536, -66576, -67650, -68759, -69905, -71089, -72315, -73584, -74898, -76260, -77672, -79137,
    -80659, -82241, -83886, -85598, -87381, -89240, -91180, -93206, -95325, -97541, -99864, -102300,
    -104857, -107546, -110376, -113359, -116508, -119837, -123361, -127100, -131072, -135300, -139810, -144631,
    -149796, -155344, -161319, -167772, -174762, -182361, -190650, -199728, -209715, -220752, -233016, -246723,
    -262144, -279620, -299593, -322638, -349525, -381300, -419430, -466033, -524288, -599186, -699050, -838860,
    -1048576, -1398101, -2097152, -4194304, 0,
];

/// `quick_inverse(x)` for |x| <= 1024.
fn quick_inverse(x: FDot6) -> Fixed {
    if x > 0 {
        -INVERSE_TABLE[(1024 - x) as usize]
    } else {
        INVERSE_TABLE[(1024 + x) as usize]
    }
}

/// `SkFDot6Div`.
fn fdot6_div(a: FDot6, b: FDot6) -> Fixed {
    if a as i16 as i32 == a {
        (((a as u32) << 16) as i32) / b
    } else {
        ((((a as i64) << 16) / b as i64).clamp(MIN_S32 as i64, MAX_S32 as i64)) as i32
    }
}

/// `quick_div`.
fn quick_div(a: FDot6, b: FDot6) -> Fixed {
    const MIN_BITS: i32 = 3;
    const MAX_ABS_A: i32 = 1 << (31 - (22 - MIN_BITS));
    let (abs_a, abs_b) = (a.wrapping_abs(), b.wrapping_abs());
    if ((1 << MIN_BITS)..1024).contains(&abs_b) && abs_a < MAX_ABS_A {
        return a.wrapping_mul(quick_inverse(b)) >> 6;
    }
    fdot6_div(a, b)
}

// ------------------------------------------------------- analytic edges

const NONE: usize = usize::MAX;

/// An `SkAnalyticEdge` (a line, or a quadratic's current line segment),
/// linked by index.
#[derive(Clone, Copy, Debug)]
struct Edge {
    next: usize,
    prev: usize,
    x: Fixed,
    dx: Fixed,
    upper_x: Fixed,
    y: Fixed,
    upper_y: Fixed,
    lower_y: Fixed,
    dy: Fixed,
    is_line: bool,
    curve_count: i32,
    curve_shift: i32,
    winding: i32,
    qx: Fixed,
    qy: Fixed,
    qdx: Fixed,
    qdy: Fixed,
    qddx: Fixed,
    qddy: Fixed,
    qlast_x: Fixed,
    qlast_y: Fixed,
    snapped_x: Fixed,
    snapped_y: Fixed,
}

impl Edge {
    fn sentinel() -> Edge {
        Edge {
            next: NONE,
            prev: NONE,
            x: 0,
            dx: 0,
            upper_x: 0,
            y: 0,
            upper_y: 0,
            lower_y: 0,
            dy: 0,
            is_line: true,
            curve_count: 0,
            curve_shift: 0,
            winding: 1,
            qx: 0,
            qy: 0,
            qdx: 0,
            qdy: 0,
            qddx: 0,
            qddy: 0,
            qlast_x: 0,
            qlast_y: 0,
            snapped_x: 0,
            snapped_y: 0,
        }
    }

    /// `fDY` of a segment from its FDot6 deltas and slope.
    fn inverse_dy(dx: FDot6, dy: FDot6, slope: Fixed, abs_slope: i32) -> Fixed {
        if dx == 0 || slope == 0 {
            MAX_S32
        } else if abs_slope < 1024 {
            quick_inverse(abs_slope)
        } else {
            quick_div(dy, dx).wrapping_abs()
        }
    }

    /// `setLine(p0, p1)`.
    fn line(p0: Pt, p1: Pt) -> Option<Edge> {
        let to = |v: f32| fdot6_to_fixed(super::to_fdot6(v * 4.0)) >> DEFAULT_ACCURACY;
        let (mut x0, mut y0) = (to(p0.0), snap_y(to(p0.1)));
        let (mut x1, mut y1) = (to(p1.0), snap_y(to(p1.1)));
        let mut winding = 1;
        if y0 > y1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
            winding = -1;
        }
        let dy = fixed_to_fdot6(y1.wrapping_sub(y0));
        if dy == 0 {
            return None;
        }
        let dx = fixed_to_fdot6(x1.wrapping_sub(x0));
        let slope = quick_div(dx, dy);
        let mut e = Edge::sentinel();
        e.x = x0;
        e.dx = slope;
        e.upper_x = x0;
        e.y = y0;
        e.upper_y = y0;
        e.lower_y = y1;
        e.dy = Edge::inverse_dy(dx, dy, slope, slope.wrapping_abs());
        e.is_line = true;
        e.winding = winding;
        Some(e)
    }

    /// `updateLine(x0, y0, x1, y1, slope)`.
    fn update_line(
        &mut self,
        mut x0: Fixed,
        mut y0: Fixed,
        mut x1: Fixed,
        mut y1: Fixed,
        slope: Fixed,
    ) -> bool {
        if y0 > y1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
            self.winding = -self.winding;
        }
        let dx = fixed_to_fdot6(x1.wrapping_sub(x0));
        let dy = fixed_to_fdot6(y1.wrapping_sub(y0));
        if dy == 0 {
            return false;
        }
        let abs_slope = fixed_to_fdot6(slope).wrapping_abs();
        self.x = x0;
        self.dx = slope;
        self.upper_x = x0;
        self.y = y0;
        self.upper_y = y0;
        self.lower_y = y1;
        self.dy = Edge::inverse_dy(dx, dy, slope, abs_slope);
        true
    }

    /// `setQuadratic(pts)`.
    fn quad(pts: &[Pt; 3]) -> Option<Edge> {
        let scale = (1 << (DEFAULT_ACCURACY + 6)) as f32;
        let fd = |v: f32| (v * scale) as i32;
        let (mut x0, mut y0) = (fd(pts[0].0), fd(pts[0].1));
        let (x1, y1) = (fd(pts[1].0), fd(pts[1].1));
        let (mut x2, mut y2) = (fd(pts[2].0), fd(pts[2].1));
        let mut winding = 1;
        if y0 > y2 {
            std::mem::swap(&mut x0, &mut x2);
            std::mem::swap(&mut y0, &mut y2);
            winding = -1;
        }
        let top = (y0 + 32) >> 6;
        let bot = (y2 + 32) >> 6;
        if top == bot {
            return None;
        }
        let shift = {
            let dx = ((((x1 as u32) << 1) as i32)
                .wrapping_sub(x0)
                .wrapping_sub(x2))
                >> 2;
            let dy = ((((y1 as u32) << 1) as i32)
                .wrapping_sub(y0)
                .wrapping_sub(y2))
                >> 2;
            // diff_to_shift(dx, dy, shiftAA = 2)
            let (adx, ady) = (dx.wrapping_abs(), dy.wrapping_abs());
            let mut dist = if adx > ady {
                adx + (ady >> 1)
            } else {
                ady + (adx >> 1)
            };
            dist = (dist + (1 << 4)) >> 5;
            let shift = (32 - (dist as u32).leading_zeros() as i32) >> 1;
            shift.clamp(1, 6)
        };
        let mut e = Edge::sentinel();
        e.winding = winding;
        e.is_line = false;
        e.curve_count = 1 << shift;
        e.curve_shift = shift - 1;
        let div2 = |v: i32| ((v as u32) << 9) as i32;
        let a = div2(x0.wrapping_sub(x1).wrapping_sub(x1).wrapping_add(x2));
        let b = fdot6_to_fixed(x1.wrapping_sub(x0));
        e.qx = fdot6_to_fixed(x0);
        e.qdx = b.wrapping_add(a >> shift);
        e.qddx = a >> (shift - 1);
        let a = div2(y0.wrapping_sub(y1).wrapping_sub(y1).wrapping_add(y2));
        let b = fdot6_to_fixed(y1.wrapping_sub(y0));
        e.qy = fdot6_to_fixed(y0);
        e.qdy = b.wrapping_add(a >> shift);
        e.qddy = a >> (shift - 1);
        e.qlast_x = fdot6_to_fixed(x2);
        e.qlast_y = fdot6_to_fixed(y2);
        // setQuadratic: back from the AA scale, y snapped.
        for v in [
            &mut e.qx,
            &mut e.qy,
            &mut e.qdx,
            &mut e.qdy,
            &mut e.qddx,
            &mut e.qddy,
            &mut e.qlast_x,
            &mut e.qlast_y,
        ] {
            *v >>= DEFAULT_ACCURACY;
        }
        e.qy = snap_y(e.qy);
        e.qlast_y = snap_y(e.qlast_y);
        e.snapped_x = e.qx;
        e.snapped_y = e.qy;
        e.update_quadratic().then_some(e)
    }

    /// `updateQuadratic()`.
    fn update_quadratic(&mut self) -> bool {
        let mut success = false;
        let mut count = self.curve_count;
        let (mut oldx, mut oldy) = (self.qx, self.qy);
        let (mut dx, mut dy) = (self.qdx, self.qdy);
        let shift = self.curve_shift;
        let (mut newx, mut newy, mut new_snapped_x, mut new_snapped_y);
        loop {
            let slope;
            count -= 1;
            if count > 0 {
                newx = oldx.wrapping_add(dx >> shift);
                newy = oldy.wrapping_add(dy >> shift);
                if (dy >> shift).wrapping_abs() >= FIXED1 * 2
                    && ((dy.wrapping_abs() as i64) << 6) > dx.wrapping_abs() as i64
                {
                    let diff_y = fixed_to_fdot6(newy.wrapping_sub(self.snapped_y));
                    slope = if diff_y != 0 {
                        quick_div(fixed_to_fdot6(newx.wrapping_sub(self.snapped_x)), diff_y)
                    } else {
                        MAX_S32
                    };
                    new_snapped_y = self.qlast_y.min(fixed_round_to_fixed(newy));
                    new_snapped_x =
                        newx.wrapping_sub(fixed_mul(slope, newy.wrapping_sub(new_snapped_y)));
                } else {
                    new_snapped_y = self.qlast_y.min(snap_y(newy));
                    new_snapped_x = newx;
                    let diff_y = fixed_to_fdot6(new_snapped_y.wrapping_sub(self.snapped_y));
                    slope = if diff_y != 0 {
                        quick_div(fixed_to_fdot6(newx.wrapping_sub(self.snapped_x)), diff_y)
                    } else {
                        MAX_S32
                    };
                }
                dx = dx.wrapping_add(self.qddx);
                dy = dy.wrapping_add(self.qddy);
            } else {
                newx = self.qlast_x;
                newy = self.qlast_y;
                new_snapped_y = newy;
                new_snapped_x = newx;
                let diff_y = fixed_to_fdot6(newy.wrapping_sub(self.snapped_y));
                slope = if diff_y != 0 {
                    quick_div(fixed_to_fdot6(newx.wrapping_sub(self.snapped_x)), diff_y)
                } else {
                    MAX_S32
                };
            }
            if slope < MAX_S32 {
                success = self.update_line(
                    self.snapped_x,
                    self.snapped_y,
                    new_snapped_x,
                    new_snapped_y,
                    slope,
                );
            }
            oldx = newx;
            oldy = newy;
            if !(count > 0 && !success) {
                break;
            }
        }
        self.qx = newx;
        self.qy = newy;
        self.qdx = dx;
        self.qdy = dy;
        self.snapped_x = new_snapped_x;
        self.snapped_y = new_snapped_y;
        self.curve_count = count;
        success
    }

    /// `update(last_y)`: the next segment of a quadratic; false for a line.
    fn update(&mut self) -> bool {
        self.curve_count > 0 && self.update_quadratic()
    }

    /// `goY(y)`.
    fn go_y(&mut self, y: Fixed) {
        if y == self.y.wrapping_add(FIXED1) {
            self.x = self.x.wrapping_add(self.dx);
            self.y = y;
        } else if y != self.y {
            self.x = self
                .upper_x
                .wrapping_add(fixed_mul(self.dx, y.wrapping_sub(self.upper_y)));
            self.y = y;
        }
    }

    /// `goY(y, yShift)`.
    fn go_y_shift(&mut self, y: Fixed, y_shift: i32) {
        self.y = y;
        self.x = self.x.wrapping_add(self.dx >> y_shift);
    }

    /// `keepContinuous()` of a quadratic.
    fn keep_continuous(&mut self) {
        self.snapped_x = self.x;
        self.snapped_y = self.y;
    }
}

/// `SkAnalyticEdgeBuilder`: the edges in path order.
struct EdgeBuilder {
    list: Vec<Edge>,
}

impl EdgeBuilder {
    /// `addLine`, merging a vertical line into the previous one
    /// (`combineVertical`).
    fn add_line(&mut self, p0: Pt, p1: Pt) {
        let Some(edge) = Edge::line(p0, p1) else {
            return;
        };
        if edge.dx == 0 && edge.is_line && !self.list.is_empty() {
            let last = self.list.last_mut().unwrap();
            match combine_vertical(&edge, last) {
                Combine::Total => {
                    self.list.pop();
                }
                Combine::Partial => {}
                Combine::No => self.list.push(edge),
            }
        } else {
            self.list.push(edge);
        }
    }

    fn add_quad(&mut self, pts: &[Pt; 3]) {
        if let Some(edge) = Edge::quad(pts) {
            self.list.push(edge);
        }
    }

    /// `buildEdges(path, clip)`.
    fn build(path: &Path, clip: Option<IRect>, can_cull_right: bool) -> Vec<Edge> {
        let mut b = EdgeBuilder { list: Vec::new() };
        match clip {
            None => {
                for edge in path.edges() {
                    match edge {
                        PathEdge::Line([p0, p1]) => b.add_line(p0, p1),
                        PathEdge::Conic(pts, w) => {
                            for quad in conic_quads(pts, w) {
                                for mono in chop_quad_at_extrema(&quad, true) {
                                    b.add_quad(&mono);
                                }
                            }
                        }
                    }
                }
            }
            Some(c) => {
                let clip = Rect {
                    left: c.left as f32,
                    top: c.top as f32,
                    right: c.right as f32,
                    bottom: c.bottom as f32,
                };
                let mut clipper = EdgeClipper {
                    can_cull_right,
                    out: Vec::new(),
                };
                for edge in path.edges() {
                    match edge {
                        PathEdge::Line([p0, p1]) => clipper.clip_line(p0, p1, &clip),
                        PathEdge::Conic(pts, w) => {
                            for quad in conic_quads(pts, w) {
                                clipper.clip_quad(&quad, &clip);
                            }
                        }
                    }
                }
                for e in clipper.out {
                    match e {
                        ClippedEdge::Line([p0, p1]) => {
                            if !is_finite(p0) || !is_finite(p1) {
                                return Vec::new();
                            }
                            b.add_line(p0, p1);
                        }
                        ClippedEdge::Quad(q) => {
                            if !q.iter().all(|&p| is_finite(p)) {
                                return Vec::new();
                            }
                            b.add_quad(&q);
                        }
                    }
                }
            }
        }
        b.list
    }
}

enum Combine {
    No,
    Partial,
    Total,
}

/// `SkAnalyticEdgeBuilder::combineVertical(edge, last)`.
fn combine_vertical(edge: &Edge, last: &mut Edge) -> Combine {
    let approx = |a: Fixed, b: Fixed| a.wrapping_sub(b).wrapping_abs() < 0x100;
    if !last.is_line || last.dx != 0 || edge.x != last.x {
        return Combine::No;
    }
    if edge.winding == last.winding {
        if edge.lower_y == last.upper_y {
            last.upper_y = edge.upper_y;
            last.y = last.upper_y;
            return Combine::Partial;
        }
        if approx(edge.upper_y, last.lower_y) {
            last.lower_y = edge.lower_y;
            return Combine::Partial;
        }
        return Combine::No;
    }
    if approx(edge.upper_y, last.upper_y) {
        if approx(edge.lower_y, last.lower_y) {
            return Combine::Total;
        }
        if edge.lower_y < last.lower_y {
            last.upper_y = edge.lower_y;
            last.y = last.upper_y;
            return Combine::Partial;
        }
        last.upper_y = last.lower_y;
        last.y = last.upper_y;
        last.lower_y = edge.lower_y;
        last.winding = edge.winding;
        return Combine::Partial;
    }
    if approx(edge.lower_y, last.lower_y) {
        if edge.upper_y > last.upper_y {
            last.lower_y = edge.upper_y;
            return Combine::Partial;
        }
        last.lower_y = last.upper_y;
        last.upper_y = edge.upper_y;
        last.y = last.upper_y;
        last.winding = edge.winding;
        return Combine::Partial;
    }
    Combine::No
}

/// `compare_edges`.
fn edge_less(a: &Edge, b: &Edge) -> bool {
    if a.upper_y != b.upper_y {
        return a.upper_y < b.upper_y;
    }
    if a.x != b.x {
        return a.x < b.x;
    }
    a.dx < b.dx
}

/// `SkTQSort` with `compare_edges`.
fn sort_edges(list: &mut [Edge]) {
    let n = list.len();
    if n <= 1 {
        return;
    }
    let depth = 2 * (usize::BITS - (n - 2).leading_zeros()) as i32;
    intro_sort(depth, list);
}

fn intro_sort(mut depth: i32, mut list: &mut [Edge]) {
    loop {
        let count = list.len();
        if count <= 32 {
            insertion_sort(list);
            return;
        }
        if depth == 0 {
            heap_sort(list);
            return;
        }
        depth -= 1;
        let middle = (count - 1) >> 1;
        let pivot = partition(list, middle);
        let (left, rest) = list.split_at_mut(pivot);
        intro_sort(depth, left);
        list = &mut rest[1..];
    }
}

fn insertion_sort(list: &mut [Edge]) {
    for next in 1..list.len() {
        if !edge_less(&list[next], &list[next - 1]) {
            continue;
        }
        let insert = list[next];
        let mut hole = next;
        loop {
            list[hole] = list[hole - 1];
            hole -= 1;
            if !(hole > 0 && edge_less(&insert, &list[hole - 1])) {
                break;
            }
        }
        list[hole] = insert;
    }
}

fn partition(list: &mut [Edge], pivot: usize) -> usize {
    let right = list.len() - 1;
    let pivot_value = list[pivot];
    list.swap(pivot, right);
    let mut new_pivot = 0;
    for left in 0..right {
        if edge_less(&list[left], &pivot_value) {
            list.swap(left, new_pivot);
            new_pivot += 1;
        }
    }
    list.swap(new_pivot, right);
    new_pivot
}

fn heap_sort(list: &mut [Edge]) {
    let count = list.len();
    let sift_down = |a: &mut [Edge], mut root: usize, bottom: usize| {
        let x = a[root - 1];
        let mut child = root << 1;
        while child <= bottom {
            if child < bottom && edge_less(&a[child - 1], &a[child]) {
                child += 1;
            }
            if edge_less(&x, &a[child - 1]) {
                a[root - 1] = a[child - 1];
                root = child;
                child = root << 1;
            } else {
                break;
            }
        }
        a[root - 1] = x;
    };
    let sift_up = |a: &mut [Edge], mut root: usize, bottom: usize| {
        let x = a[root - 1];
        let start = root;
        let mut j = root << 1;
        while j <= bottom {
            if j < bottom && edge_less(&a[j - 1], &a[j]) {
                j += 1;
            }
            a[root - 1] = a[j - 1];
            root = j;
            j = root << 1;
        }
        j = root >> 1;
        while j >= start {
            if edge_less(&a[j - 1], &x) {
                a[root - 1] = a[j - 1];
                root = j;
                j = root >> 1;
            } else {
                break;
            }
        }
        a[root - 1] = x;
    };
    for i in (1..=count >> 1).rev() {
        sift_down(list, i, count);
    }
    for i in (1..count).rev() {
        list.swap(0, i);
        sift_up(list, 1, i);
    }
}

// --------------------------------------------------------- additive blitters

fn catch_overflow(alpha: u32) -> u8 {
    (alpha - (alpha >> 8)) as u8
}

fn safely_add(a: u8, delta: u8) -> u8 {
    (a as u32 + delta as u32).min(0xFF) as u8
}

/// `MaskAdditiveBlitter`: coverage accumulated in an A8 mask over the
/// path's bounds, blended once at the end.
struct MaskAdditive {
    mask: Mask,
    clip_rect: IRect,
}

const MASK_MAX_WIDTH: i32 = 32;
const MASK_MAX_STORAGE: i64 = 1024;

impl MaskAdditive {
    fn can_handle(ir: &IRect) -> bool {
        let width = ir.right - ir.left;
        if width > MASK_MAX_WIDTH {
            return false;
        }
        let rb = ((width + 3) & !3) as i64;
        rb * (ir.bottom - ir.top) as i64 <= MASK_MAX_STORAGE
    }

    fn new(ir: IRect, clip: IRect) -> MaskAdditive {
        // fStorage: (kMAX_STORAGE >> 2) + 2 words, the image one byte in.
        MaskAdditive {
            mask: Mask {
                image: vec![0; (MASK_MAX_STORAGE as usize >> 2 << 2) + 8],
                offset: 1,
                bounds: ir,
                row_bytes: (ir.right - ir.left) as usize,
            },
            clip_rect: ir.intersect(&clip).unwrap_or(IRect {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
            }),
        }
    }

    fn index(&self, x: i32, y: i32) -> Option<usize> {
        let i = self.mask.offset as isize
            + (y - self.mask.bounds.top) as isize * self.mask.row_bytes as isize
            + (x - self.mask.bounds.left) as isize;
        (0..self.mask.image.len() as isize)
            .contains(&i)
            .then_some(i as usize)
    }

    fn update(&mut self, x: i32, y: i32, f: impl Fn(u8) -> u8) {
        if let Some(i) = self.index(x, y) {
            self.mask.image[i] = f(self.mask.image[i]);
        }
    }
}

/// `RunBasedAdditiveBlitter` (`safe` false) or `SafeRLEAdditiveBlitter`:
/// one row of accumulated coverage, flushed to the real blitter when the
/// row changes.
struct RunAdditive<'a> {
    real: &'a mut dyn Blitter,
    left: i32,
    width: i32,
    top: i32,
    curr_y: i32,
    alpha: Vec<u8>,
    safe: bool,
}

impl<'a> RunAdditive<'a> {
    fn new(real: &'a mut dyn Blitter, ir: IRect, clip: IRect, safe: bool) -> RunAdditive<'a> {
        let sect = ir.intersect(&clip).unwrap_or(IRect {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        });
        let width = sect.right - sect.left;
        RunAdditive {
            real,
            left: sect.left,
            width,
            top: sect.top,
            curr_y: sect.top - 1,
            alpha: vec![0; width.max(0) as usize],
            safe,
        }
    }

    fn add(&self, a: u8, delta: u8) -> u8 {
        if self.safe {
            safely_add(a, delta)
        } else {
            catch_overflow(a as u32 + delta as u32)
        }
    }

    fn flush(&mut self) {
        if self.curr_y >= self.top {
            // snapAlpha: values near 0 and 255 blit as 0 and 255.
            for a in &mut self.alpha {
                *a = if *a > 247 {
                    0xFF
                } else if *a < 8 {
                    0
                } else {
                    *a
                };
            }
            if self.alpha.iter().any(|&a| a != 0) {
                let row = std::mem::take(&mut self.alpha);
                self.real.blit_anti_h(self.left, self.curr_y, &row);
                self.alpha = vec![0; row.len()];
            }
            self.curr_y = self.top - 1;
        }
    }

    fn check_y(&mut self, y: i32) {
        if y != self.curr_y {
            self.flush();
            self.curr_y = y;
        }
    }

    fn check(&self, x: i32, width: i32) -> bool {
        x >= 0 && x + width <= self.width
    }
}

/// The additive blitter AAA accumulates into.
enum Additive<'a> {
    Mask(MaskAdditive),
    Run(RunAdditive<'a>),
}

impl Additive<'_> {
    fn is_mask(&self) -> bool {
        matches!(self, Additive::Mask(_))
    }

    /// `blitAntiH(x, y, antialias, len)`.
    fn blit_anti_h_alphas(&mut self, x: i32, y: i32, aa: &[u8]) {
        match self {
            Additive::Mask(_) => unreachable!("mask coverage is written directly"),
            Additive::Run(r) => {
                r.check_y(y);
                let mut x = x - r.left;
                let mut aa = aa;
                if x < 0 {
                    aa = &aa[(-x) as usize..];
                    x = 0;
                }
                let len = (aa.len() as i32).min(r.width - x);
                for i in 0..len.max(0) {
                    let idx = (x + i) as usize;
                    r.alpha[idx] = r.add(r.alpha[idx], aa[i as usize]);
                }
            }
        }
    }

    /// `blitAntiH(x, y, alpha)`.
    fn blit_anti_h1(&mut self, x: i32, y: i32, alpha: u8) {
        self.blit_anti_hn(x, y, 1, alpha);
    }

    /// `blitAntiH(x, y, width, alpha)`.
    fn blit_anti_hn(&mut self, x: i32, y: i32, width: i32, alpha: u8) {
        match self {
            Additive::Mask(m) => {
                for i in 0..width {
                    m.update(x + i, y, |a| catch_overflow(a as u32 + alpha as u32));
                }
            }
            Additive::Run(r) => {
                r.check_y(y);
                let x = x - r.left;
                if r.check(x, width) {
                    for i in 0..width {
                        let idx = (x + i) as usize;
                        r.alpha[idx] = r.add(r.alpha[idx], alpha);
                    }
                }
            }
        }
    }

    fn flush_if_y_changed(&mut self, y: Fixed, next_y: Fixed) {
        if let Additive::Run(r) = self
            && fixed_floor(y) != fixed_floor(next_y)
        {
            r.flush();
        }
    }

    // getRealBlitter(): the mask itself for the mask blitter (its blits set
    // coverage), the canvas blitter for the run-length ones.

    fn real_blit_v(&mut self, x: i32, y: i32, height: i32, alpha: u8) {
        match self {
            Additive::Mask(m) => {
                if alpha == 0 {
                    return;
                }
                for i in 0..height {
                    m.update(x, y + i, |_| alpha);
                }
            }
            Additive::Run(r) => r.real.blit_v(x, y, height, alpha),
        }
    }

    fn real_blit_rect(&mut self, x: i32, y: i32, width: i32, height: i32) {
        match self {
            Additive::Mask(m) => {
                for row in 0..height {
                    for i in 0..width {
                        m.update(x + i, y + row, |_| 0xFF);
                    }
                }
            }
            Additive::Run(r) => r.real.blit_rect(x, y, width, height),
        }
    }

    fn real_blit_anti_rect(&mut self, x: i32, y: i32, width: i32, height: i32, l: u8, r_alpha: u8) {
        match self {
            Additive::Mask(_) => {
                self.real_blit_v(x, y, height, l);
                self.real_blit_v(x + 1 + width, y, height, r_alpha);
                self.real_blit_rect(x + 1, y, width, height);
            }
            Additive::Run(r) => r.real.blit_anti_rect(x, y, width, height, l, r_alpha),
        }
    }

    fn real_blit_h(&mut self, x: i32, y: i32, width: i32) {
        // AdditiveBlitter::blitH draws nothing; only the RLE blitters reach it.
        if let Additive::Run(r) = self {
            r.real.blit_h(x, y, width);
        }
    }

    fn real_blit_anti_h2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        if let Additive::Run(r) = self {
            r.real.blit_anti_h2(x, y, a0, a1);
        }
    }

    fn real_blit_anti_h(&mut self, x: i32, y: i32, aa: &[u8]) {
        if let Additive::Run(r) = self {
            r.real.blit_anti_h(x, y, aa);
        }
    }

    /// The mask row a walker writes into (`getRow(y)`), or `None` for the
    /// run-length blitters.
    fn mask_row(&self, y: i32) -> Option<i32> {
        self.is_mask().then_some(y)
    }

    fn mask_set(&mut self, x: i32, y: i32, a: u8) {
        if let Additive::Mask(m) = self {
            m.update(x, y, |_| a);
        }
    }

    fn mask_safely_add(&mut self, x: i32, y: i32, a: u8) {
        if let Additive::Mask(m) = self {
            m.update(x, y, |v| safely_add(v, a));
        }
    }
}

// ------------------------------------------------------- coverage helpers

fn trapezoid_to_alpha(l1: Fixed, l2: Fixed) -> u8 {
    (((l1.wrapping_add(l2)) / 2) >> 8) as u8
}

fn partial_triangle_to_alpha(a: Fixed, b: Fixed) -> u8 {
    let area = (a >> 11).wrapping_mul(a >> 11).wrapping_mul(b >> 11);
    ((area >> 8) & 0xFF) as u8
}

/// `get_partial_alpha(SkAlpha, SkFixed)`.
fn partial_alpha_fixed(alpha: u8, partial_height: Fixed) -> u8 {
    fixed_round_to_int((alpha as i32).wrapping_mul(partial_height)) as u8
}

/// `get_partial_alpha(SkAlpha, SkAlpha)`.
fn partial_alpha(alpha: u8, full: u8) -> u8 {
    ((alpha as u32 * full as u32) >> 8) as u8
}

/// `fixed_to_alpha` of SkScan_AAAPath.
fn fixed_to_alpha(f: Fixed) -> u8 {
    partial_alpha_fixed(0xFF, f)
}

fn approximate_intersection(mut l1: Fixed, mut r1: Fixed, mut l2: Fixed, mut r2: Fixed) -> Fixed {
    if l1 > r1 {
        std::mem::swap(&mut l1, &mut r1);
    }
    if l2 > r2 {
        std::mem::swap(&mut l2, &mut r2);
    }
    (l1.max(l2).wrapping_add(r1.min(r2))) / 2
}

fn compute_alpha_above_line(alphas: &mut [u8], l: Fixed, r: Fixed, dy: Fixed, full: u8) {
    let rr = fixed_ceil(r);
    if rr == 0 {
        return;
    }
    if rr == 1 {
        let a = ((((rr as u32) << 17) as i32).wrapping_sub(l).wrapping_sub(r) >> 9) as u8;
        alphas[0] = partial_alpha(a, full);
        return;
    }
    let first = FIXED1 - l;
    let last = r - ((rr - 1) << 16);
    let first_h = fixed_mul(first, dy);
    alphas[0] = (fixed_mul(first, first_h) >> 9) as u8;
    let mut alpha16 = sat_add(first_h, dy >> 1);
    for a in alphas.iter_mut().take((rr - 1) as usize).skip(1) {
        *a = (alpha16 >> 8) as u8;
        alpha16 = sat_add(alpha16, dy);
    }
    alphas[(rr - 1) as usize] = (full as i32 - partial_triangle_to_alpha(last, dy) as i32) as u8;
}

fn compute_alpha_below_line(alphas: &mut [u8], l: Fixed, r: Fixed, dy: Fixed, full: u8) {
    let rr = fixed_ceil(r);
    if rr == 0 {
        return;
    }
    if rr == 1 {
        alphas[0] = partial_alpha(trapezoid_to_alpha(l, r), full);
        return;
    }
    let first = FIXED1 - l;
    let last = r - ((rr - 1) << 16);
    let last_h = fixed_mul(last, dy);
    alphas[(rr - 1) as usize] = (fixed_mul(last, last_h) >> 9) as u8;
    let mut alpha16 = sat_add(last_h, dy >> 1);
    let mut i = rr - 2;
    while i > 0 {
        alphas[i as usize] = ((alpha16 >> 8) & 0xFF) as u8;
        alpha16 = sat_add(alpha16, dy);
        i -= 1;
    }
    alphas[0] = (full as i32 - partial_triangle_to_alpha(first, dy) as i32) as u8;
}

struct Row {
    y: i32,
    mask_row: Option<i32>,
    full: u8,
    no_real: bool,
}

fn blit_single_alpha(b: &mut Additive, row: &Row, x: i32, alpha: u8) {
    match row.mask_row {
        Some(my) => {
            if row.full == 0xFF && !row.no_real {
                b.mask_set(x, my, alpha);
            } else {
                b.mask_safely_add(x, my, partial_alpha(alpha, row.full));
            }
        }
        None => {
            if row.full == 0xFF && !row.no_real {
                b.real_blit_v(x, row.y, 1, alpha);
            } else {
                b.blit_anti_h1(x, row.y, partial_alpha(alpha, row.full));
            }
        }
    }
}

fn blit_two_alphas(b: &mut Additive, row: &Row, x: i32, a1: u8, a2: u8) {
    match row.mask_row {
        Some(my) => {
            b.mask_safely_add(x, my, a1);
            b.mask_safely_add(x + 1, my, a2);
        }
        None => {
            if row.full == 0xFF && !row.no_real {
                b.real_blit_anti_h2(x, row.y, a1, a2);
            } else {
                b.blit_anti_h1(x, row.y, a1);
                b.blit_anti_h1(x + 1, row.y, a2);
            }
        }
    }
}

fn blit_full_alpha(b: &mut Additive, row: &Row, x: i32, len: i32) {
    match row.mask_row {
        Some(my) => {
            for i in 0..len {
                b.mask_safely_add(x + i, my, row.full);
            }
        }
        None => {
            if row.full == 0xFF && !row.no_real {
                b.real_blit_h(x, row.y, len);
            } else {
                b.blit_anti_hn(x, row.y, len, row.full);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn blit_aaa_trapezoid_row(
    b: &mut Additive,
    row: &Row,
    ul: Fixed,
    ur: Fixed,
    ll: Fixed,
    lr: Fixed,
    l_dy: Fixed,
    r_dy: Fixed,
) {
    let full = row.full;
    let l = fixed_floor(ul);
    let r = fixed_ceil(lr);
    let len = r - l;
    if len == 1 {
        let alpha = trapezoid_to_alpha(ur.wrapping_sub(ul), lr.wrapping_sub(ll));
        blit_single_alpha(b, row, l, alpha);
        return;
    }
    let n = len.max(0) as usize;
    let mut alphas = vec![full; n];
    let mut temp = vec![0u8; n + 1];

    let u_l = fixed_floor(ul);
    let l_l = fixed_ceil(ll);
    if u_l + 2 == l_l {
        let first = int_to_fixed(u_l) + FIXED1 - ul;
        let second = ll - ul - first;
        let a1 = (full as i32 - partial_triangle_to_alpha(first, l_dy) as i32) as u8;
        let a2 = partial_triangle_to_alpha(second, l_dy);
        alphas[0] = alphas[0].saturating_sub(a1);
        alphas[1] = alphas[1].saturating_sub(a2);
    } else {
        let off = (u_l - l) as usize;
        compute_alpha_below_line(
            &mut temp[off..],
            ul - int_to_fixed(u_l),
            ll - int_to_fixed(u_l),
            l_dy,
            full,
        );
        for i in u_l..l_l {
            let k = (i - l) as usize;
            alphas[k] = alphas[k].saturating_sub(temp[k]);
        }
    }

    let u_r = fixed_floor(ur);
    let l_r = fixed_ceil(lr);
    if u_r + 2 == l_r {
        let first = int_to_fixed(u_r) + FIXED1 - ur;
        let second = lr - ur - first;
        let a1 = partial_triangle_to_alpha(first, r_dy);
        let a2 = (full as i32 - partial_triangle_to_alpha(second, r_dy) as i32) as u8;
        alphas[n - 2] = alphas[n - 2].saturating_sub(a1);
        alphas[n - 1] = alphas[n - 1].saturating_sub(a2);
    } else {
        let off = (u_r - l) as usize;
        compute_alpha_above_line(
            &mut temp[off..],
            ur - int_to_fixed(u_r),
            lr - int_to_fixed(u_r),
            r_dy,
            full,
        );
        for i in u_r..l_r {
            let k = (i - l) as usize;
            alphas[k] = alphas[k].saturating_sub(temp[k]);
        }
    }

    match row.mask_row {
        Some(my) => {
            for (i, &a) in alphas.iter().enumerate() {
                b.mask_safely_add(l + i as i32, my, a);
            }
        }
        None => {
            if full == 0xFF && !row.no_real {
                b.real_blit_anti_h(l, row.y, &alphas);
            } else {
                b.blit_anti_h_alphas(l, row.y, &alphas);
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn blit_trapezoid_row(
    b: &mut Additive,
    row: &Row,
    mut ul: Fixed,
    mut ur: Fixed,
    mut ll: Fixed,
    mut lr: Fixed,
    l_dy: Fixed,
    r_dy: Fixed,
) {
    if ul > ur {
        return;
    }
    if ll > lr {
        let x = approximate_intersection(ul, ll, ur, lr);
        ll = x;
        lr = x;
    }
    if ul == ur && ll == lr {
        return;
    }
    if ul > ll {
        std::mem::swap(&mut ul, &mut ll);
    }
    if ur > lr {
        std::mem::swap(&mut ur, &mut lr);
    }
    let full = row.full;
    let join_left = fixed_ceil_to_fixed(ll);
    let join_rite = fixed_floor_to_fixed(ur);
    if join_left <= join_rite {
        if ul < join_left {
            let len = fixed_ceil(join_left - ul);
            if len == 1 {
                let alpha = trapezoid_to_alpha(join_left - ul, join_left - ll);
                blit_single_alpha(b, row, ul >> 16, alpha);
            } else if len == 2 {
                let first = join_left - FIXED1 - ul;
                let second = ll - ul - first;
                let a1 = partial_triangle_to_alpha(first, l_dy);
                let a2 = (full as i32 - partial_triangle_to_alpha(second, l_dy) as i32) as u8;
                blit_two_alphas(b, row, ul >> 16, a1, a2);
            } else {
                blit_aaa_trapezoid_row(b, row, ul, join_left, ll, join_left, l_dy, MAX_S32);
            }
        }
        if join_left < join_rite {
            blit_full_alpha(
                b,
                row,
                fixed_floor(join_left),
                fixed_floor(join_rite - join_left),
            );
        }
        if lr > join_rite {
            let len = fixed_ceil(lr - join_rite);
            if len == 1 {
                let alpha = trapezoid_to_alpha(ur - join_rite, lr - join_rite);
                blit_single_alpha(b, row, join_rite >> 16, alpha);
            } else if len == 2 {
                let first = join_rite + FIXED1 - ur;
                let second = lr - ur - first;
                let a1 = (full as i32 - partial_triangle_to_alpha(first, r_dy) as i32) as u8;
                let a2 = partial_triangle_to_alpha(second, r_dy);
                blit_two_alphas(b, row, join_rite >> 16, a1, a2);
            } else {
                blit_aaa_trapezoid_row(b, row, join_rite, ur, join_rite, lr, MAX_S32, r_dy);
            }
        }
    } else {
        blit_aaa_trapezoid_row(b, row, ul, ur, ll, lr, l_dy, r_dy);
    }
}

// ----------------------------------------------------------------- walkers

/// The sorted edge list between its head (index 0) and tail (index 1)
/// sentinels.
struct EdgeList {
    e: Vec<Edge>,
}

const HEAD: usize = 0;
const TAIL: usize = 1;

impl EdgeList {
    fn new(mut edges: Vec<Edge>) -> EdgeList {
        sort_edges(&mut edges);
        let count = edges.len();
        let mut head = Edge::sentinel();
        head.upper_y = MIN_S32;
        head.lower_y = MIN_S32;
        head.x = MIN_S32;
        head.dx = 0;
        head.dy = MAX_S32;
        head.upper_x = MIN_S32;
        let mut tail = Edge::sentinel();
        tail.upper_y = MAX_S32;
        tail.lower_y = MAX_S32;
        tail.x = MAX_S32;
        tail.dx = 0;
        tail.dy = MAX_S32;
        tail.upper_x = MAX_S32;
        let mut e = vec![head, tail];
        e.extend(edges);
        for i in 0..count {
            let idx = i + 2;
            e[idx].prev = if i == 0 { HEAD } else { idx - 1 };
            e[idx].next = if i + 1 == count { TAIL } else { idx + 1 };
        }
        e[HEAD].next = 2;
        e[TAIL].prev = count + 1;
        EdgeList { e }
    }

    fn remove(&mut self, edge: usize) {
        let (p, n) = (self.e[edge].prev, self.e[edge].next);
        self.e[p].next = n;
        self.e[n].prev = p;
    }

    fn insert_after(&mut self, edge: usize, after: usize) {
        let n = self.e[after].next;
        self.e[edge].prev = after;
        self.e[edge].next = n;
        self.e[n].prev = edge;
        self.e[after].next = edge;
    }

    fn backward_insert_based_on_x(&mut self, edge: usize) {
        let x = self.e[edge].x;
        let mut prev = self.e[edge].prev;
        while self.e[prev].prev != NONE && self.e[prev].x > x {
            prev = self.e[prev].prev;
        }
        if self.e[prev].next != edge {
            self.remove(edge);
            self.insert_after(edge, prev);
        }
    }

    fn backward_insert_start(&self, mut prev: usize, x: Fixed) -> usize {
        while self.e[prev].prev != NONE && self.e[prev].x > x {
            prev = self.e[prev].prev;
        }
        prev
    }
}

/// `is_smooth_enough(thisEdge, nextEdge, stop_y)`.
fn smooth_one(this: &Edge, next: &Edge) -> bool {
    if this.curve_count > 0 {
        return (this.qdx.wrapping_abs() >> 1) >= this.qddx.wrapping_abs()
            && (this.qdy.wrapping_abs() >> 1) >= this.qddy.wrapping_abs()
            && (this.qdy.wrapping_sub(this.qddy) >> this.curve_shift) >= FIXED1;
    }
    sat_sub(next.dx, this.dx).wrapping_abs() <= FIXED1
        && next.lower_y.wrapping_sub(next.upper_y) >= FIXED1
}

/// `is_smooth_enough(leftE, riteE, currE, stop_y)`.
fn smooth_enough(l: &EdgeList, left: usize, rite: usize, mut curr: usize, stop_y: i32) -> bool {
    let stop = ((stop_y as u32) << 16) as i32;
    if l.e[curr].upper_y >= stop {
        return false;
    }
    let (le, re) = (&l.e[left], &l.e[rite]);
    if le.lower_y.wrapping_add(FIXED1) < re.lower_y {
        return smooth_one(le, &l.e[curr]);
    } else if le.lower_y > re.lower_y.wrapping_add(FIXED1) {
        return smooth_one(re, &l.e[curr]);
    }
    let mut next_curr = l.e[curr].next;
    if l.e[next_curr].upper_y >= stop {
        return false;
    }
    if l.e[next_curr].upper_x < l.e[curr].upper_x {
        std::mem::swap(&mut curr, &mut next_curr);
    }
    smooth_one(le, &l.e[curr]) && smooth_one(re, &l.e[next_curr])
}

/// `aaa_walk_convex_edges`.
fn walk_convex(
    l: &mut EdgeList,
    b: &mut Additive,
    _start_y: i32,
    stop_y: i32,
    left_bound: Fixed,
    rite_bound: Fixed,
) {
    let mut left_e = l.e[HEAD].next;
    let mut rite_e = l.e[left_e].next;
    let mut curr_e = l.e[rite_e].next;
    let mut y = l.e[left_e].upper_y.max(l.e[rite_e].upper_y);
    let using_mask = b.is_mask();

    'walk: loop {
        while l.e[left_e].lower_y <= y {
            if !l.e[left_e].update() {
                if fixed_floor(l.e[curr_e].upper_y) >= stop_y {
                    break 'walk;
                }
                left_e = curr_e;
                curr_e = l.e[curr_e].next;
            }
        }
        while l.e[rite_e].lower_y <= y {
            if !l.e[rite_e].update() {
                if fixed_floor(l.e[curr_e].upper_y) >= stop_y {
                    break 'walk;
                }
                rite_e = curr_e;
                curr_e = l.e[curr_e].next;
            }
        }
        if fixed_floor(y) >= stop_y {
            break;
        }
        l.e[left_e].go_y(y);
        l.e[rite_e].go_y(y);
        {
            let (le, re) = (&l.e[left_e], &l.e[rite_e]);
            if le.x > re.x || (le.x == re.x && le.dx > re.dx) {
                std::mem::swap(&mut left_e, &mut rite_e);
            }
        }
        let mut local_bot = l.e[left_e].lower_y.min(l.e[rite_e].lower_y);
        if smooth_enough(l, left_e, rite_e, curr_e, stop_y) {
            local_bot = fixed_ceil_to_fixed(local_bot);
        }
        local_bot = local_bot.min(int_to_fixed(stop_y));

        let mut left = left_bound.max(l.e[left_e].x);
        let d_left = l.e[left_e].dx;
        let mut rite = rite_bound.min(l.e[rite_e].x);
        let d_rite = l.e[rite_e].dx;
        if (d_left | d_rite) == 0 {
            let full_left = fixed_ceil(left);
            let full_rite = fixed_floor(rite);
            let partial_left = int_to_fixed(full_left) - left;
            let partial_rite = rite - int_to_fixed(full_rite);
            let full_top = fixed_ceil(y);
            let full_bot = fixed_floor(local_bot);
            let mut partial_top = int_to_fixed(full_top) - y;
            let mut partial_bot = local_bot - int_to_fixed(full_bot);
            if full_top > full_bot {
                partial_top -= FIXED1 - partial_bot;
                partial_bot = 0;
            }
            if full_rite >= full_left {
                if partial_top > 0 {
                    if partial_left > 0 {
                        b.blit_anti_h1(
                            full_left - 1,
                            full_top - 1,
                            fixed_to_alpha(fixed_mul(partial_top, partial_left)),
                        );
                    }
                    b.blit_anti_hn(
                        full_left,
                        full_top - 1,
                        full_rite - full_left,
                        fixed_to_alpha(partial_top),
                    );
                    if partial_rite > 0 {
                        b.blit_anti_h1(
                            full_rite,
                            full_top - 1,
                            fixed_to_alpha(fixed_mul(partial_top, partial_rite)),
                        );
                    }
                    b.flush_if_y_changed(y, y + partial_top);
                }
                if full_bot > full_top
                    && (full_rite > full_left
                        || fixed_to_alpha(partial_left) > 0
                        || fixed_to_alpha(partial_rite) > 0)
                {
                    b.real_blit_anti_rect(
                        full_left - 1,
                        full_top,
                        full_rite - full_left,
                        full_bot - full_top,
                        fixed_to_alpha(partial_left),
                        fixed_to_alpha(partial_rite),
                    );
                }
                if partial_bot > 0 {
                    if partial_left > 0 {
                        b.blit_anti_h1(
                            full_left - 1,
                            full_bot,
                            fixed_to_alpha(fixed_mul(partial_bot, partial_left)),
                        );
                    }
                    b.blit_anti_hn(
                        full_left,
                        full_bot,
                        full_rite - full_left,
                        fixed_to_alpha(partial_bot),
                    );
                    if partial_rite > 0 {
                        b.blit_anti_h1(
                            full_rite,
                            full_bot,
                            fixed_to_alpha(fixed_mul(partial_bot, partial_rite)),
                        );
                    }
                }
            } else {
                let width = rite - left;
                if width > 0 {
                    if partial_top > 0 {
                        b.blit_anti_hn(
                            full_left - 1,
                            full_top - 1,
                            1,
                            fixed_to_alpha(fixed_mul(partial_top, width)),
                        );
                        b.flush_if_y_changed(y, y + partial_top);
                    }
                    if full_bot > full_top {
                        b.real_blit_v(
                            full_left - 1,
                            full_top,
                            full_bot - full_top,
                            fixed_to_alpha(width),
                        );
                    }
                    if partial_bot > 0 {
                        b.blit_anti_hn(
                            full_left - 1,
                            full_bot,
                            1,
                            fixed_to_alpha(fixed_mul(partial_bot, width)),
                        );
                    }
                }
            }
            y = local_bot;
        } else {
            const SNAP_DIGIT: Fixed = FIXED1 >> 4;
            const SNAP_HALF: Fixed = SNAP_DIGIT >> 1;
            const SNAP_MASK: Fixed = -1 ^ (SNAP_DIGIT - 1);
            left += SNAP_HALF;
            rite += SNAP_HALF;
            let mut count = fixed_ceil(local_bot) - fixed_floor(y);
            let (l_dy, r_dy) = (l.e[left_e].dy, l.e[rite_e].dy);
            let row = |y: Fixed, full: u8| Row {
                y: y >> 16,
                mask_row: using_mask.then_some(y >> 16),
                full,
                no_real: false,
            };
            if count > 1 {
                if (y as u32 & 0xFFFF_0000) as i32 != y {
                    count -= 1;
                    let next_y = fixed_ceil_to_fixed(y + 1);
                    let dy = next_y - y;
                    let next_left = left + fixed_mul(d_left, dy);
                    let next_rite = rite + fixed_mul(d_rite, dy);
                    blit_trapezoid_row(
                        b,
                        &row(y, partial_alpha_fixed(0xFF, dy)),
                        left & SNAP_MASK,
                        rite & SNAP_MASK,
                        next_left & SNAP_MASK,
                        next_rite & SNAP_MASK,
                        l_dy,
                        r_dy,
                    );
                    b.flush_if_y_changed(y, next_y);
                    left = next_left;
                    rite = next_rite;
                    y = next_y;
                }
                while count > 1 {
                    count -= 1;
                    let next_y = y + FIXED1;
                    let next_left = left + d_left;
                    let next_rite = rite + d_rite;
                    blit_trapezoid_row(
                        b,
                        &row(y, 0xFF),
                        left & SNAP_MASK,
                        rite & SNAP_MASK,
                        next_left & SNAP_MASK,
                        next_rite & SNAP_MASK,
                        l_dy,
                        r_dy,
                    );
                    b.flush_if_y_changed(y, next_y);
                    left = next_left;
                    rite = next_rite;
                    y = next_y;
                }
            }
            let dy = local_bot - y;
            let next_left = (left + fixed_mul(d_left, dy)).max(left_bound + SNAP_HALF);
            let next_rite = (rite + fixed_mul(d_rite, dy)).min(rite_bound + SNAP_HALF);
            blit_trapezoid_row(
                b,
                &row(y, partial_alpha_fixed(0xFF, dy)),
                left & SNAP_MASK,
                rite & SNAP_MASK,
                next_left & SNAP_MASK,
                next_rite & SNAP_MASK,
                l_dy,
                r_dy,
            );
            b.flush_if_y_changed(y, local_bot);
            left = next_left - SNAP_HALF;
            rite = next_rite - SNAP_HALF;
            y = local_bot;
        }
        l.e[left_e].x = left;
        l.e[rite_e].x = rite;
        l.e[left_e].y = y;
        l.e[rite_e].y = y;
    }
}

fn update_next_next_y(y: Fixed, next_y: Fixed, next_next_y: &mut Fixed) {
    if y > next_y && y < *next_next_y {
        *next_next_y = y;
    }
}

fn check_intersection(l: &EdgeList, edge: usize, next_y: Fixed, next_next_y: &mut Fixed) {
    let e = &l.e[edge];
    let prev = &l.e[e.prev];
    if prev.prev != NONE && prev.x.wrapping_add(prev.dx) > e.x.wrapping_add(e.dx) {
        *next_next_y = next_y + (FIXED1 >> DEFAULT_ACCURACY);
    }
}

fn check_intersection_fwd(l: &EdgeList, edge: usize, next_y: Fixed, next_next_y: &mut Fixed) {
    let e = &l.e[edge];
    let next = &l.e[e.next];
    if next.next != NONE && e.x.wrapping_add(e.dx) > next.x.wrapping_add(next.dx) {
        *next_next_y = next_y + (FIXED1 >> DEFAULT_ACCURACY);
    }
}

fn insert_new_edges(l: &mut EdgeList, mut new_edge: usize, y: Fixed, next_next_y: &mut Fixed) {
    if l.e[new_edge].upper_y > y {
        update_next_next_y(l.e[new_edge].upper_y, y, next_next_y);
        return;
    }
    let prev = l.e[new_edge].prev;
    if l.e[prev].x <= l.e[new_edge].x {
        while l.e[new_edge].upper_y <= y {
            check_intersection(l, new_edge, y, next_next_y);
            update_next_next_y(l.e[new_edge].lower_y, y, next_next_y);
            new_edge = l.e[new_edge].next;
        }
        update_next_next_y(l.e[new_edge].upper_y, y, next_next_y);
        return;
    }
    let mut start = l.backward_insert_start(prev, l.e[new_edge].x);
    loop {
        let next = l.e[new_edge].next;
        let mut moved = true;
        loop {
            if l.e[start].next == new_edge {
                moved = false;
                break;
            }
            let after = l.e[start].next;
            if l.e[after].x >= l.e[new_edge].x {
                break;
            }
            start = after;
        }
        if moved {
            l.remove(new_edge);
            l.insert_after(new_edge, start);
        }
        check_intersection(l, new_edge, y, next_next_y);
        check_intersection_fwd(l, new_edge, y, next_next_y);
        update_next_next_y(l.e[new_edge].lower_y, y, next_next_y);
        start = new_edge;
        new_edge = next;
        if l.e[new_edge].upper_y > y {
            break;
        }
    }
    update_next_next_y(l.e[new_edge].upper_y, y, next_next_y);
}

fn edges_too_close(l: &EdgeList, prev: usize, next: usize, lower_y: Fixed) -> bool {
    if prev == NONE || next == NONE {
        return false;
    }
    let (p, n) = (&l.e[prev], &l.e[next]);
    n.upper_y < lower_y && p.x.wrapping_add(FIXED1) >= n.x.wrapping_sub(n.dx.wrapping_abs())
}

fn edges_too_close_rite(prev_rite: i32, ul: Fixed, ll: Fixed) -> bool {
    prev_rite > fixed_floor(ul) || prev_rite > fixed_floor(ll)
}

/// `aaa_walk_edges` for a winding fill, not inverse.
#[allow(clippy::too_many_arguments)]
fn walk_edges(
    l: &mut EdgeList,
    b: &mut Additive,
    start_y: i32,
    stop_y: i32,
    left_clip: Fixed,
    right_clip: Fixed,
    skip_intersect: bool,
) {
    l.e[HEAD].x = left_clip;
    l.e[HEAD].upper_x = left_clip;
    l.e[TAIL].x = right_clip;
    l.e[TAIL].upper_x = right_clip;
    let mut y = l.e[l.e[HEAD].next].upper_y.max(int_to_fixed(start_y));
    let mut next_next_y = MAX_S32;
    {
        let mut edge = l.e[HEAD].next;
        while l.e[edge].upper_y <= y {
            l.e[edge].go_y(y);
            update_next_next_y(l.e[edge].lower_y, y, &mut next_next_y);
            edge = l.e[edge].next;
        }
        update_next_next_y(l.e[edge].upper_y, y, &mut next_next_y);
    }

    loop {
        let mut w = 0i32;
        let mut in_interval = false;
        let mut prev_x = l.e[HEAD].x;
        let mut next_y = next_next_y.min(fixed_ceil_to_fixed(y + 1));
        let mut curr_e = l.e[HEAD].next;
        let mut left_e = HEAD;
        let mut left = left_clip;
        let mut left_dy = 0;
        let mut prev_rite = fixed_floor(left_clip);
        next_next_y = MAX_S32;

        let mut y_shift = 0;
        if (next_y - y) & (FIXED1 >> 2) != 0 {
            y_shift = 2;
            next_y = y + (FIXED1 >> 2);
        } else if (next_y - y) & (FIXED1 >> 1) != 0 {
            y_shift = 1;
        }
        let full_alpha = fixed_to_alpha(next_y - y);
        let mask_row = b.mask_row(fixed_floor(y));
        let no_real_blitter = false;

        while l.e[curr_e].upper_y <= y {
            w += l.e[curr_e].winding;
            let prev_in_interval = in_interval;
            in_interval = w != 0;
            let is_left = in_interval && !prev_in_interval;
            let is_rite = !in_interval && prev_in_interval;

            if is_rite {
                let mut rite = l.e[curr_e].x;
                l.e[curr_e].go_y_shift(next_y, y_shift);
                let next_left = left_clip.max(l.e[left_e].x);
                rite = right_clip.min(rite);
                let next_rite = right_clip.min(l.e[curr_e].x);
                let too_close = full_alpha == 0xFF
                    && (edges_too_close_rite(prev_rite, left, l.e[left_e].x)
                        || edges_too_close(l, curr_e, l.e[curr_e].next, next_y));
                let row = Row {
                    y: y >> 16,
                    mask_row,
                    full: full_alpha,
                    no_real: no_real_blitter || too_close,
                };
                blit_trapezoid_row(
                    b,
                    &row,
                    left,
                    rite,
                    next_left,
                    next_rite,
                    left_dy,
                    l.e[curr_e].dy,
                );
                prev_rite = fixed_ceil(rite.max(l.e[curr_e].x));
            } else {
                if is_left {
                    left = l.e[curr_e].x.max(left_clip);
                    left_dy = l.e[curr_e].dy;
                    left_e = curr_e;
                }
                l.e[curr_e].go_y_shift(next_y, y_shift);
            }

            let next = l.e[curr_e].next;
            while l.e[curr_e].lower_y <= next_y {
                if l.e[curr_e].curve_count > 0 {
                    l.e[curr_e].keep_continuous();
                    if !l.e[curr_e].update_quadratic() {
                        break;
                    }
                } else {
                    break;
                }
            }

            if l.e[curr_e].lower_y <= next_y {
                l.remove(curr_e);
            } else {
                update_next_next_y(l.e[curr_e].lower_y, next_y, &mut next_next_y);
                let new_x = l.e[curr_e].x;
                if new_x < prev_x {
                    l.backward_insert_based_on_x(curr_e);
                } else {
                    prev_x = new_x;
                }
                if !skip_intersect {
                    check_intersection(l, curr_e, next_y, &mut next_next_y);
                }
            }
            curr_e = next;
        }

        if in_interval {
            let too_close =
                full_alpha == 0xFF && edges_too_close(l, l.e[left_e].prev, left_e, next_y);
            let row = Row {
                y: y >> 16,
                mask_row,
                full: full_alpha,
                no_real: no_real_blitter || too_close,
            };
            blit_trapezoid_row(
                b,
                &row,
                left,
                right_clip,
                left_clip.max(l.e[left_e].x),
                right_clip,
                left_dy,
                0,
            );
        }

        y = next_y;
        if y >= int_to_fixed(stop_y) {
            break;
        }
        insert_new_edges(l, curr_e, y, &mut next_next_y);
    }
}

// ----------------------------------------------------------------- entry

/// Rounded-out bounds (`SkRect::roundOut`, saturating).
fn round_out(r: &Rect) -> IRect {
    IRect {
        left: r.left.floor() as i32,
        top: r.top.floor() as i32,
        right: r.right.ceil() as i32,
        bottom: r.bottom.ceil() as i32,
    }
}

/// `aaa_fill_path` with a non-inverse winding fill.
#[allow(clippy::too_many_arguments)]
fn aaa_fill_path(
    path: &Path,
    convex: bool,
    clip: IRect,
    b: &mut Additive,
    mut start_y: i32,
    mut stop_y: i32,
    contained: bool,
    path_ir: IRect,
) {
    let edges = EdgeBuilder::build(path, (!contained).then_some(clip), !convex);
    let count = edges.len();
    if count == 0 {
        return;
    }
    let mut list = EdgeList::new(edges);
    if !contained {
        start_y = start_y.max(clip.top);
        stop_y = stop_y.min(clip.bottom);
    }
    let mut left_bound = int_to_fixed(clip.left);
    let mut rite_bound = int_to_fixed(clip.right);
    if b.is_mask() {
        left_bound = left_bound.max(int_to_fixed(path_ir.left));
        rite_bound = rite_bound.min(int_to_fixed(path_ir.right));
    }
    if convex && count >= 2 {
        walk_convex(&mut list, b, start_y, stop_y, left_bound, rite_bound);
    } else {
        let span = (stop_y - start_y) * 2;
        let skip_intersect = span < 0 || path.pts.len() > span as usize;
        walk_edges(
            &mut list,
            b,
            start_y,
            stop_y,
            left_bound,
            rite_bound,
            skip_intersect,
        );
    }
}

/// Fill the stroke of `p0`→`p1` (width > 1) with `blitter` on a canvas of
/// `canvas` bounds: `Draw::drawPath` → `SkScan::AntiFillPath` →
/// `SkScan::AAAFillPath`.
pub(super) fn fill_stroke(p0: Pt, p1: Pt, width: f32, canvas: IRect, blitter: &mut dyn Blitter) {
    let path = stroke_line(p0, p1, width);
    let Some(bounds) = path.bounds() else {
        return;
    };
    // Draw::drawDevPath: SkPathPriv::TooBigForMath.
    let max = f32::MAX * 0.25;
    if !(bounds.left >= -max && bounds.top >= -max && bounds.right <= max && bounds.bottom <= max) {
        return;
    }
    let convex = is_convex(&path);

    // AntiFillPath: safeRoundOut (rounded out, limited to what the
    // supersampling shift can hold) and the clip intersection.
    let limit = i32::MAX >> 2;
    let Some(ir) = round_out(&bounds).intersect(&IRect {
        left: -limit,
        top: -limit,
        right: limit,
        bottom: limit,
    }) else {
        return;
    };
    if !ir.intersects(&canvas) {
        return;
    }

    // SkScanClipper: a rectangle clip wraps the blitter only when the path
    // pokes out horizontally.
    let contained = canvas.contains(&ir);
    let mut inner = DynBlitter(blitter);
    let mut rect_clipped;
    let real: &mut dyn Blitter = if !contained && (canvas.left > ir.left || canvas.right < ir.right)
    {
        rect_clipped = RectClipBlitter {
            inner: &mut inner,
            clip: canvas,
        };
        &mut rect_clipped
    } else {
        &mut inner
    };

    // AAAFillPath.
    if MaskAdditive::can_handle(&ir) {
        let mut b = Additive::Mask(MaskAdditive::new(ir, canvas));
        aaa_fill_path(
            &path, convex, canvas, &mut b, ir.top, ir.bottom, contained, ir,
        );
        if let Additive::Mask(m) = b {
            real.blit_mask(&m.mask, m.clip_rect);
        }
    } else {
        let run = RunAdditive::new(real, ir, canvas, !convex);
        let mut b = Additive::Run(run);
        aaa_fill_path(
            &path, convex, canvas, &mut b, ir.top, ir.bottom, contained, ir,
        );
        if let Additive::Run(mut r) = b {
            r.flush();
        }
    }
}
