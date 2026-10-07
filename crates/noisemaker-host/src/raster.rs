//! Chromium's software 2D canvas, as the traced overlays draw on it.
//!
//! The reference creates the overlay canvases with `getContext('2d', {
//! willReadFrequently: true })`, which pins them to Skia's CPU raster
//! back end. In Chromium 153 (Skia 9d07e5ba) each `stroke()` of the
//! tracer's one-segment path takes this route, which [`RasterCanvas`]
//! ports:
//!
//! - Blink `Canvas2DRecorderContext::DrawPathInternal`: a path whose points
//!   coincide is pruned; a stroke whose bounds, outset by lineWidth / 2 and
//!   rounded out, miss the canvas is culled; a moveTo/lineTo path is a line
//!   and is drawn with `drawLine`.
//! - `SkCanvas::drawLine` → `drawPoints(kLines_PointMode)` → (a round cap
//!   declines `PtProcRec`) `skcpu::Draw::drawDevicePoints` →
//!   `drawPath(SkPath::Line(p0, p1))` with the stroke paint (round caps,
//!   anti-aliased).
//! - `Draw::drawPath`: a stroke at most 1 px wide becomes a hairline
//!   (`modifyPaintForHairlines`): width 1 keeps the paint's alpha, a
//!   thinner width w scales it to alpha * (int)(w * 256) >> 8. A hairline
//!   is drawn by `SkScan::AntiHairRoundPath`: each end moves out by π/8
//!   along the segment, and `do_anti_hairline` draws it in 26.6 and 16.16
//!   fixed point. A wider stroke is converted to a fill path (`SkStroke`)
//!   and filled by `SkScan::AntiFillPath`.
//! - Pixels: the legacy N32 blitters (`SkARGB32_Blitter` and its opaque
//!   and black variants), which a solid src-over paint on the sRGB canvas
//!   selects (`SkBlitter::UseLegacyBlitter`), in their integer arithmetic.
//!
//! Chromium compiles Skia with `-ffp-contract=off`, so its float
//! expressions round as the f32 and f64 operations here do.

// This module and its submodule `fill` port parts of Skia
// (https://skia.org) at commit 9d07e5bad9e3e21da2426946e589daa647218271:
// src/core/SkDraw.cpp, SkScan_Hairline.cpp, SkScan_Antihair.cpp,
// SkScan_AntiPath.cpp, SkScan_AAAPath.cpp, SkAnalyticEdge.cpp,
// SkEdgeBuilder.cpp, SkEdgeClipper.cpp, SkLineClipper.cpp, SkStroke.cpp,
// SkStrokerPriv.cpp, SkGeometry.cpp, SkPathPriv.cpp, SkPoint.cpp,
// SkBlitter.cpp, SkBlitter_ARGB32.cpp, SkTSort.h, SkColorData.h,
// SkColorPriv.h and src/opts/SkBlitRow_opts.h and SkBlitMask_opts.h,
// copyright Google Inc., Google LLC and The Android Open Source Project.
// Skia's license:
//
// Copyright (c) 2011 Google Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in
//     the documentation and/or other materials provided with the
//     distribution.
//
//   * Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::Rgba8Image;
use crate::canvas::parse_rgba_style;
use crate::worm::Canvas2d;

mod fill;

// ------------------------------------------------------------ pixel math

/// An `SkPMColor`: premultiplied, alpha in bits 24..31 and red, green,
/// blue in bits 16, 8 and 0. The blend arithmetic treats every channel
/// alike, so the order only matters for alpha.
type PmColor = u32;

const A_SHIFT: u32 = 24;

fn packed_a(c: PmColor) -> u32 {
    c >> A_SHIFT
}

fn pack_argb(a: u32, r: u32, g: u32, b: u32) -> PmColor {
    (a << 24) | (r << 16) | (g << 8) | b
}

/// `SkAlpha255To256`.
fn alpha_255_to_256(alpha: u32) -> u32 {
    alpha + 1
}

/// `SkMulDiv255Round`.
fn mul_div_255_round(a: u32, b: u32) -> u32 {
    let prod = a * b + 128;
    (prod + (prod >> 8)) >> 8
}

/// `SkPreMultiplyColor` of a straight 8-bit colour.
fn premultiply([r, g, b, a]: [u8; 4]) -> PmColor {
    let (r, g, b, a) = (r as u32, g as u32, b as u32, a as u32);
    if a == 255 {
        pack_argb(a, r, g, b)
    } else {
        pack_argb(
            a,
            mul_div_255_round(r, a),
            mul_div_255_round(g, a),
            mul_div_255_round(b, a),
        )
    }
}

/// `SkAlphaMulQ`: every channel times `scale` (0..256), shifted down 8.
fn alpha_mul_q(c: PmColor, scale: u32) -> PmColor {
    const MASK: u32 = 0x00FF_00FF;
    let rb = ((c & MASK).wrapping_mul(scale)) >> 8;
    let ag = ((c >> 8) & MASK).wrapping_mul(scale);
    (rb & MASK) | (ag & !MASK)
}

/// `SkAlphaMulInv256`.
fn alpha_mul_inv_256(value: u32, alpha256: u32) -> u32 {
    let prod = 0xFFFF - value * alpha256;
    (prod + (prod >> 8)) >> 8
}

/// `SkBlendARGB32(src, dst, aa)`.
fn blend_argb32(src: PmColor, dst: PmColor, aa: u32) -> PmColor {
    const MASK: u32 = 0x00FF_00FF;
    let src_scale = alpha_255_to_256(aa);
    let dst_scale = alpha_mul_inv_256(packed_a(src), src_scale);
    let src_rb = (src & MASK) * src_scale;
    let src_ag = ((src >> 8) & MASK) * src_scale;
    let dst_rb = (dst & MASK) * dst_scale;
    let dst_ag = ((dst >> 8) & MASK) * dst_scale;
    (((src_rb + dst_rb) >> 8) & MASK) | ((src_ag + dst_ag) & !MASK)
}

/// `SkFastFourByteInterp(src, dst, srcWeight)` (the 64-bit form).
fn fast_four_byte_interp(src: PmColor, dst: PmColor, weight: u32) -> PmColor {
    let scale = (weight + (weight >> 7)) as u64;
    let splay =
        |c: u32| -> u64 { ((((c >> 8) & 0x00FF_00FF) as u64) << 32) | (c & 0x00FF_00FF) as u64 };
    let agrb = splay(src) * scale + (256 - scale) * splay(dst);
    const MASK: u64 = 0xFF00_FF00;
    (((agrb & MASK) >> 8) | ((agrb >> 32) & MASK)) as u32
}

/// `SkBlitRow::Color32` on one pixel: memset for an opaque colour,
/// nothing for a transparent one, else `blit_row_color32`'s
/// `(d * (256 - a)) >> 8 + c` per channel.
fn color32(dst: &mut PmColor, color: PmColor) {
    match packed_a(color) {
        0 => {}
        255 => *dst = color,
        a => {
            let inv = 256 - a;
            let mut out = 0u32;
            for shift in [0, 8, 16, 24] {
                let d = (*dst >> shift) & 0xFF;
                let c = (color >> shift) & 0xFF;
                out |= ((((d * inv) >> 8) + c) & 0xFF) << shift;
            }
            *dst = out;
        }
    }
}

// --------------------------------------------------------------- blitters

/// An integer rectangle, left/top inclusive, right/bottom exclusive
/// (`SkIRect`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl IRect {
    fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }
    fn intersects(&self, o: &IRect) -> bool {
        !self.is_empty()
            && !o.is_empty()
            && self.left < o.right
            && o.left < self.right
            && self.top < o.bottom
            && o.top < self.bottom
    }
    fn contains(&self, o: &IRect) -> bool {
        !o.is_empty()
            && !self.is_empty()
            && self.left <= o.left
            && self.top <= o.top
            && self.right >= o.right
            && self.bottom >= o.bottom
    }
    fn intersect(&self, o: &IRect) -> Option<IRect> {
        let r = IRect {
            left: self.left.max(o.left),
            top: self.top.max(o.top),
            right: self.right.min(o.right),
            bottom: self.bottom.min(o.bottom),
        };
        (!r.is_empty()).then_some(r)
    }
}

/// The `SkBlitter` calls the scan converters make. `blit_anti_h` takes one
/// coverage per pixel (the runs of `blitAntiH` expanded; every blitter
/// here computes each pixel independently, so the run boundaries do not
/// change the result). The provided methods are `SkBlitter`'s defaults.
trait Blitter {
    fn blit_h(&mut self, x: i32, y: i32, width: i32);
    fn blit_anti_h(&mut self, x: i32, y: i32, aa: &[u8]);

    fn blit_v(&mut self, x: i32, y: i32, height: i32, alpha: u8) {
        if alpha == 255 {
            self.blit_rect(x, y, 1, height);
        } else {
            for row in 0..height {
                self.blit_anti_h(x, y + row, &[alpha]);
            }
        }
    }

    fn blit_rect(&mut self, x: i32, y: i32, width: i32, height: i32) {
        for row in 0..height {
            self.blit_h(x, y + row, width);
        }
    }

    fn blit_anti_h2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        self.blit_anti_h(x, y, &[a0, a1]);
    }

    fn blit_anti_v2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        self.blit_anti_h(x, y, &[a0]);
        self.blit_anti_h(x, y + 1, &[a1]);
    }

    fn blit_anti_rect(&mut self, x: i32, y: i32, width: i32, height: i32, left: u8, right: u8) {
        let mut x = x;
        if left > 0 {
            self.blit_v(x, y, height, left);
        }
        x += 1;
        if width > 0 {
            self.blit_rect(x, y, width, height);
            x += width;
        }
        if right > 0 {
            self.blit_v(x, y, height, right);
        }
    }

    /// `blitMask` of an A8 mask restricted to `clip`.
    fn blit_mask(&mut self, mask: &Mask, clip: IRect);
}

/// An A8 coverage mask (`SkMask`): `image[offset + (y - top) * row_bytes +
/// (x - left)]` is the coverage at (x, y).
struct Mask {
    image: Vec<u8>,
    offset: usize,
    bounds: IRect,
    row_bytes: usize,
}

impl Mask {
    fn get(&self, x: i32, y: i32) -> u8 {
        let i = self.offset as isize
            + (y - self.bounds.top) as isize * self.row_bytes as isize
            + (x - self.bounds.left) as isize;
        self.image[i as usize]
    }
}

/// Which legacy N32 blitter `SkBlitter::Choose` returns for the paint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlitterKind {
    /// `SkARGB32_Blitter`: alpha below 255.
    Translucent,
    /// `SkARGB32_Opaque_Blitter`.
    Opaque,
    /// `SkARGB32_Black_Blitter`: opaque black.
    Black,
}

/// The device blitter: a solid colour over the canvas pixels.
struct ArgbBlitter<'a> {
    pixels: &'a mut [PmColor],
    width: usize,
    kind: BlitterKind,
    /// `fPMColor`.
    pm: PmColor,
    /// `fSrcA`.
    src_a: u32,
}

impl<'a> ArgbBlitter<'a> {
    fn new(pixels: &'a mut [PmColor], width: usize, color: [u8; 4]) -> ArgbBlitter<'a> {
        let kind = if color == [0, 0, 0, 255] {
            BlitterKind::Black
        } else if color[3] == 255 {
            BlitterKind::Opaque
        } else {
            BlitterKind::Translucent
        };
        ArgbBlitter {
            pixels,
            width,
            kind,
            pm: premultiply(color),
            src_a: color[3] as u32,
        }
    }

    fn px(&mut self, x: i32, y: i32) -> &mut PmColor {
        &mut self.pixels[y as usize * self.width + x as usize]
    }
}

impl Blitter for ArgbBlitter<'_> {
    fn blit_h(&mut self, x: i32, y: i32, width: i32) {
        let pm = self.pm;
        for i in 0..width {
            color32(self.px(x + i, y), pm);
        }
    }

    fn blit_anti_h(&mut self, x: i32, y: i32, aa: &[u8]) {
        if self.kind == BlitterKind::Translucent && self.src_a == 0 {
            return;
        }
        for (i, &a) in aa.iter().enumerate() {
            let a = a as u32;
            if a == 0 {
                continue;
            }
            let (kind, pm) = (self.kind, self.pm);
            let px = self.px(x + i as i32, y);
            match kind {
                BlitterKind::Black => {
                    if a == 255 {
                        *px = 0xFF00_0000;
                    } else {
                        *px = (a << A_SHIFT) + alpha_mul_q(*px, alpha_255_to_256(255 - a));
                    }
                }
                BlitterKind::Opaque if a == 255 => *px = pm,
                _ => color32(px, alpha_mul_q(pm, alpha_255_to_256(a))),
            }
        }
    }

    fn blit_v(&mut self, x: i32, y: i32, height: i32, alpha: u8) {
        if alpha == 0 || self.src_a == 0 {
            return;
        }
        let mut color = self.pm;
        if alpha != 255 {
            color = alpha_mul_q(color, alpha_255_to_256(alpha as u32));
        }
        let dst_scale = alpha_255_to_256(255 - packed_a(color));
        for row in 0..height {
            let px = self.px(x, y + row);
            *px = color + alpha_mul_q(*px, dst_scale);
        }
    }

    fn blit_rect(&mut self, x: i32, y: i32, width: i32, height: i32) {
        if self.src_a == 0 {
            return;
        }
        for row in 0..height {
            self.blit_h(x, y + row, width);
        }
    }

    fn blit_anti_h2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        self.blend_pixel(x, y, a0 as u32);
        self.blend_pixel(x + 1, y, a1 as u32);
    }

    fn blit_mask(&mut self, mask: &Mask, clip: IRect) {
        // SkARGB32_Blitter::blitMask returns for a transparent colour; every
        // variant then blits through blit_color → SkOpts::blit_mask_d32_a8.
        if self.kind == BlitterKind::Translucent && self.src_a == 0 {
            return;
        }
        let (kind, pm) = (self.kind, self.pm);
        let color_alpha = packed_a(pm);
        for y in clip.top..clip.bottom {
            for x in clip.left..clip.right {
                let m = mask.get(x, y) as u32;
                let px = self.px(x, y);
                *px = match kind {
                    BlitterKind::Black => alpha_mul_q(*px, 256 - m).wrapping_add(m << A_SHIFT),
                    _ => {
                        let m256 = alpha_255_to_256(m);
                        let scale = match kind {
                            BlitterKind::Translucent => 256 - ((color_alpha * m256) >> 8),
                            _ => 256 - m,
                        };
                        // Per channel in the NEON form: two u8 products,
                        // added with u8 wrap-around.
                        let mut out = 0u32;
                        for shift in [0, 8, 16, 24] {
                            let c = (pm >> shift) & 0xFF;
                            let d = (*px >> shift) & 0xFF;
                            out |= ((((c * m256) >> 8) + ((d * scale) >> 8)) & 0xFF) << shift;
                        }
                        out
                    }
                };
            }
        }
    }

    fn blit_anti_v2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        self.blend_pixel(x, y, a0 as u32);
        self.blend_pixel(x, y + 1, a1 as u32);
    }
}

impl ArgbBlitter<'_> {
    /// One pixel of `blitAntiH2` / `blitAntiV2`.
    fn blend_pixel(&mut self, x: i32, y: i32, a: u32) {
        let (kind, pm) = (self.kind, self.pm);
        let px = self.px(x, y);
        *px = match kind {
            BlitterKind::Translucent => blend_argb32(pm, *px, a),
            BlitterKind::Opaque => fast_four_byte_interp(pm, *px, a),
            BlitterKind::Black => (a << A_SHIFT) + alpha_mul_q(*px, 256 - a),
        };
    }
}

/// `SkRectClipBlitter`: clips each call to a rectangle. It does not
/// override `blitAntiH2`/`blitAntiV2`, so those take `SkBlitter`'s
/// defaults through the clipped `blitAntiH`.
struct RectClipBlitter<'b, B: Blitter> {
    inner: &'b mut B,
    clip: IRect,
}

impl<B: Blitter> Blitter for RectClipBlitter<'_, B> {
    fn blit_h(&mut self, x: i32, y: i32, width: i32) {
        if y < self.clip.top || y >= self.clip.bottom {
            return;
        }
        let left = x.max(self.clip.left);
        let right = (x + width).min(self.clip.right);
        if right > left {
            self.inner.blit_h(left, y, right - left);
        }
    }

    fn blit_anti_h(&mut self, x: i32, y: i32, aa: &[u8]) {
        if y < self.clip.top || y >= self.clip.bottom || x >= self.clip.right {
            return;
        }
        let x1 = x + aa.len() as i32;
        if x1 <= self.clip.left {
            return;
        }
        let x0 = x.max(self.clip.left);
        let end = x1.min(self.clip.right);
        self.inner
            .blit_anti_h(x0, y, &aa[(x0 - x) as usize..(end - x) as usize]);
    }

    fn blit_v(&mut self, x: i32, y: i32, height: i32, alpha: u8) {
        if x < self.clip.left || x >= self.clip.right {
            return;
        }
        let y0 = y.max(self.clip.top);
        let y1 = (y + height).min(self.clip.bottom);
        if y0 < y1 {
            self.inner.blit_v(x, y0, y1 - y0, alpha);
        }
    }

    fn blit_rect(&mut self, x: i32, y: i32, width: i32, height: i32) {
        let r = IRect {
            left: x,
            top: y,
            right: x + width,
            bottom: y + height,
        };
        if let Some(r) = r.intersect(&self.clip) {
            self.inner
                .blit_rect(r.left, r.top, r.right - r.left, r.bottom - r.top);
        }
    }

    fn blit_anti_rect(&mut self, x: i32, y: i32, width: i32, height: i32, left: u8, right: u8) {
        // The true width of the rectangle is width + 2.
        let full = IRect {
            left: x,
            top: y,
            right: x + width + 2,
            bottom: y + height,
        };
        let Some(r) = full.intersect(&self.clip) else {
            return;
        };
        let (mut left, mut right) = (left, right);
        if r.left != x {
            left = 255;
        }
        if r.right != x + width + 2 {
            right = 255;
        }
        let (rw, rh) = (r.right - r.left, r.bottom - r.top);
        if left == 255 && right == 255 {
            self.inner.blit_rect(r.left, r.top, rw, rh);
        } else if rw == 1 {
            let alpha = if r.left == x { left } else { right };
            self.inner.blit_v(r.left, r.top, rh, alpha);
        } else {
            self.inner
                .blit_anti_rect(r.left, r.top, rw - 2, rh, left, right);
        }
    }

    fn blit_mask(&mut self, mask: &Mask, clip: IRect) {
        if let Some(r) = clip.intersect(&self.clip) {
            self.inner.blit_mask(mask, r);
        }
    }
}

// ---------------------------------------------------- anti-aliased hairlines

/// 26.6 fixed point (`SkFDot6`).
type FDot6 = i32;
/// 16.16 fixed point (`SkFixed`).
type Fixed = i32;

const FDOT6_ONE: i32 = 64;
const FDOT6_HALF: i32 = 32;
const FIXED_HALF: i32 = 1 << 15;

/// `SkScalarToFDot6`: `(int)(x * 64)`, truncating.
fn to_fdot6(x: f32) -> FDot6 {
    (x * 64.0) as i32
}

fn fdot6_floor(x: FDot6) -> i32 {
    x >> 6
}

fn fdot6_ceil(x: FDot6) -> i32 {
    (x + 63) >> 6
}

/// `SkFDot6ToFixed`.
fn fdot6_to_fixed(x: FDot6) -> Fixed {
    ((x as u32) << 10) as i32
}

fn fixed_floor(x: Fixed) -> i32 {
    x >> 16
}

fn fixed_ceil(x: Fixed) -> i32 {
    x.wrapping_add(0xFFFF) >> 16
}

/// `fastfixdiv`: `(a << 16) / b`.
fn fast_fix_div(a: FDot6, b: FDot6) -> Fixed {
    (((a as u32) << 16) as i32) / b
}

fn fd6_frac(x: FDot6) -> i32 {
    x & (FDOT6_ONE - 1)
}

/// `partial_pixel_coverage`.
fn partial_pixel_coverage(pos: FDot6) -> i32 {
    fd6_frac(pos - 1) + 1
}

/// `scale_alpha_by_coverage`.
fn scale_alpha(value: u32, coverage: i32) -> u8 {
    ((value * coverage as u32) >> 6) as u8
}

/// `fixed_to_alpha`.
fn fixed_to_alpha(f: Fixed) -> u32 {
    ((f >> 8) & 0xFF) as u32
}

/// `call_hline_blitter`: `count` pixels of one coverage.
fn hline(blitter: &mut dyn Blitter, x: i32, y: i32, count: i32, alpha: u8) {
    let aa = vec![alpha; count as usize];
    blitter.blit_anti_h(x, y, &aa);
}

/// The four `SkAntiHairBlitter` strategies by slope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HairKind {
    HLine,
    Horish,
    VLine,
    Vertish,
}

impl HairKind {
    /// `drawCap(x, fy, slope, coverage)`.
    fn draw_cap(
        self,
        blitter: &mut dyn Blitter,
        x: i32,
        f: Fixed,
        slope: Fixed,
        coverage: i32,
    ) -> Fixed {
        let f = f.wrapping_add(FIXED_HALF);
        let i = fixed_floor(f);
        let a = fixed_to_alpha(f);
        match self {
            HairKind::HLine => {
                let ma = scale_alpha(a, coverage);
                if ma != 0 {
                    hline(blitter, x, i, 1, ma);
                }
                let ma = scale_alpha(255 - a, coverage);
                if ma != 0 {
                    hline(blitter, x, i - 1, 1, ma);
                }
                f - FIXED_HALF
            }
            HairKind::Horish => {
                let a0 = scale_alpha(255 - a, coverage);
                let a1 = scale_alpha(a, coverage);
                blitter.blit_anti_v2(x, i - 1, a0, a1);
                f.wrapping_add(slope) - FIXED_HALF
            }
            HairKind::VLine => {
                let ma = scale_alpha(a, coverage);
                if ma != 0 {
                    blitter.blit_v(i, x, 1, ma);
                }
                let ma = scale_alpha(255 - a, coverage);
                if ma != 0 {
                    blitter.blit_v(i - 1, x, 1, ma);
                }
                f - FIXED_HALF
            }
            HairKind::Vertish => {
                blitter.blit_anti_h2(
                    i - 1,
                    x,
                    scale_alpha(255 - a, coverage),
                    scale_alpha(a, coverage),
                );
                f.wrapping_add(slope) - FIXED_HALF
            }
        }
    }

    /// `drawLine(x, stopx, fy, slope)`.
    fn draw_line(
        self,
        blitter: &mut dyn Blitter,
        start: i32,
        stop: i32,
        f: Fixed,
        slope: Fixed,
    ) -> Fixed {
        let mut f = f.wrapping_add(FIXED_HALF);
        match self {
            HairKind::HLine => {
                let y = fixed_floor(f);
                let a = fixed_to_alpha(f);
                if a != 0 {
                    hline(blitter, start, y, stop - start, a as u8);
                }
                let a = 255 - a;
                if a != 0 {
                    hline(blitter, start, y - 1, stop - start, a as u8);
                }
            }
            HairKind::Horish => {
                for x in start..stop {
                    let lower = fixed_floor(f);
                    let a = fixed_to_alpha(f);
                    blitter.blit_anti_v2(x, lower - 1, (255 - a) as u8, a as u8);
                    f = f.wrapping_add(slope);
                }
            }
            HairKind::VLine => {
                let x = fixed_floor(f);
                let a = fixed_to_alpha(f);
                if a != 0 {
                    blitter.blit_v(x, start, stop - start, a as u8);
                }
                let a = 255 - a;
                if a != 0 {
                    blitter.blit_v(x - 1, start, stop - start, a as u8);
                }
            }
            HairKind::Vertish => {
                for y in start..stop {
                    let x = fixed_floor(f);
                    let a = fixed_to_alpha(f);
                    blitter.blit_anti_h2(x - 1, y, (255 - a) as u8, a as u8);
                    f = f.wrapping_add(slope);
                }
            }
        }
        f - FIXED_HALF
    }
}

/// `do_anti_hairline(x0, y0, x1, y1, clip, blitter)`.
fn do_anti_hairline(
    mut x0: FDot6,
    mut y0: FDot6,
    mut x1: FDot6,
    mut y1: FDot6,
    clip: Option<IRect>,
    blitter: &mut dyn Blitter,
) {
    // Integer NaN (0x80000000): a huge or non-finite coordinate.
    if [x0, y0, x1, y1].contains(&i32::MIN) {
        return;
    }
    if (x1 - x0).abs() > 511 * FDOT6_ONE || (y1 - y0).abs() > 511 * FDOT6_ONE {
        let hx = (x0 >> 1) + (x1 >> 1);
        let hy = (y0 >> 1) + (y1 >> 1);
        do_anti_hairline(x0, y0, hx, hy, clip, blitter);
        do_anti_hairline(hx, hy, x1, y1, clip, blitter);
        return;
    }

    let mut clip = clip;
    let (mut istart, mut istop, mut fstart, slope, kind);
    let (mut start_coverage, mut stop_coverage);

    if (x1 - x0).abs() > (y1 - y0).abs() {
        // Mostly horizontal: left to right.
        if x0 > x1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }
        istart = fdot6_floor(x0);
        istop = fdot6_ceil(x1);
        if y0 == y1 {
            slope = 0;
            kind = HairKind::HLine;
            fstart = fdot6_to_fixed(y0);
        } else {
            slope = fast_fix_div(y1 - y0, x1 - x0);
            let dx_to_center = FDOT6_HALF - fd6_frac(x0);
            fstart = fdot6_to_fixed(y0)
                .wrapping_add((slope.wrapping_mul(dx_to_center) + FDOT6_HALF) >> 6);
            kind = HairKind::Horish;
        }
        if istop - istart == 1 {
            start_coverage = x1 - x0;
            stop_coverage = 0;
        } else {
            start_coverage = FDOT6_ONE - fd6_frac(x0);
            stop_coverage = fd6_frac(x1);
        }
        if let Some(c) = clip {
            if istart >= c.right || istop <= c.left {
                return;
            }
            if istart < c.left {
                fstart = fstart.wrapping_add(slope.wrapping_mul(c.left - istart));
                istart = c.left;
                start_coverage = FDOT6_ONE;
                if istop - istart == 1 {
                    start_coverage = partial_pixel_coverage(x1);
                    stop_coverage = 0;
                }
            }
            if istop > c.right {
                istop = c.right;
                stop_coverage = 0;
            }
            if istart == istop {
                return;
            }
            let span = slope.wrapping_mul(istop - istart - 1);
            let (mut top, mut bottom);
            if slope >= 0 {
                top = fixed_floor(fstart - FIXED_HALF);
                bottom = fixed_ceil(fstart.wrapping_add(span).wrapping_add(FIXED_HALF));
            } else {
                bottom = fixed_ceil(fstart.wrapping_add(FIXED_HALF));
                top = fixed_floor(fstart.wrapping_add(span) - FIXED_HALF);
            }
            top -= 1;
            bottom += 1;
            if top >= c.bottom || bottom <= c.top {
                return;
            }
            if c.top <= top && c.bottom >= bottom {
                clip = None;
            }
        }
    } else {
        // Mostly vertical: top to bottom.
        if y0 > y1 {
            std::mem::swap(&mut x0, &mut x1);
            std::mem::swap(&mut y0, &mut y1);
        }
        istart = fdot6_floor(y0);
        istop = fdot6_ceil(y1);
        if x0 == x1 {
            if y0 == y1 {
                return;
            }
            slope = 0;
            kind = HairKind::VLine;
            fstart = fdot6_to_fixed(x0);
        } else {
            slope = fast_fix_div(x1 - x0, y1 - y0);
            let dy_to_center = FDOT6_HALF - fd6_frac(y0);
            fstart = fdot6_to_fixed(x0)
                .wrapping_add((slope.wrapping_mul(dy_to_center) + FDOT6_HALF) >> 6);
            kind = HairKind::Vertish;
        }
        if istop - istart == 1 {
            start_coverage = y1 - y0;
            stop_coverage = 0;
        } else {
            start_coverage = FDOT6_ONE - fd6_frac(y0);
            stop_coverage = fd6_frac(y1);
        }
        if let Some(c) = clip {
            if istart >= c.bottom || istop <= c.top {
                return;
            }
            if istart < c.top {
                fstart = fstart.wrapping_add(slope.wrapping_mul(c.top - istart));
                istart = c.top;
                start_coverage = FDOT6_ONE;
                if istop - istart == 1 {
                    start_coverage = partial_pixel_coverage(y1);
                    stop_coverage = 0;
                }
            }
            if istop > c.bottom {
                istop = c.bottom;
                stop_coverage = 0;
            }
            if istart == istop {
                return;
            }
            let span = slope.wrapping_mul(istop - istart - 1);
            let (mut left, mut right);
            if slope >= 0 {
                left = fixed_floor(fstart - FIXED_HALF);
                right = fixed_ceil(fstart.wrapping_add(span).wrapping_add(FIXED_HALF));
            } else {
                right = fixed_ceil(fstart.wrapping_add(FIXED_HALF));
                left = fixed_floor(fstart.wrapping_add(span) - FIXED_HALF);
            }
            left -= 1;
            right += 1;
            if left >= c.right || right <= c.left {
                return;
            }
            if c.left <= left && c.right >= right {
                clip = None;
            }
        }
    }

    let coverage = (start_coverage, stop_coverage);
    match clip {
        Some(c) => {
            let mut inner = DynBlitter(blitter);
            let mut clipped = RectClipBlitter {
                inner: &mut inner,
                clip: c,
            };
            draw_hair(kind, &mut clipped, istart, istop, fstart, slope, coverage);
        }
        None => draw_hair(kind, blitter, istart, istop, fstart, slope, coverage),
    }
}

/// The tail of `do_anti_hairline`: the start cap, the full spans and the
/// stop cap.
fn draw_hair(
    kind: HairKind,
    blitter: &mut dyn Blitter,
    mut istart: i32,
    istop: i32,
    fstart: Fixed,
    slope: Fixed,
    (start_coverage, stop_coverage): (i32, i32),
) {
    let mut f = kind.draw_cap(blitter, istart, fstart, slope, start_coverage);
    istart += 1;
    let full_spans = istop - istart - (stop_coverage > 0) as i32;
    if full_spans > 0 {
        f = kind.draw_line(blitter, istart, istart + full_spans, f, slope);
    }
    if stop_coverage > 0 {
        kind.draw_cap(blitter, istop - 1, f, slope, stop_coverage);
    }
}

/// A `&mut dyn Blitter` as a sized [`Blitter`] for [`RectClipBlitter`].
struct DynBlitter<'a>(&'a mut dyn Blitter);

impl Blitter for DynBlitter<'_> {
    fn blit_h(&mut self, x: i32, y: i32, width: i32) {
        self.0.blit_h(x, y, width)
    }
    fn blit_anti_h(&mut self, x: i32, y: i32, aa: &[u8]) {
        self.0.blit_anti_h(x, y, aa)
    }
    fn blit_v(&mut self, x: i32, y: i32, height: i32, alpha: u8) {
        self.0.blit_v(x, y, height, alpha)
    }
    fn blit_rect(&mut self, x: i32, y: i32, width: i32, height: i32) {
        self.0.blit_rect(x, y, width, height)
    }
    fn blit_anti_h2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        self.0.blit_anti_h2(x, y, a0, a1)
    }
    fn blit_anti_v2(&mut self, x: i32, y: i32, a0: u8, a1: u8) {
        self.0.blit_anti_v2(x, y, a0, a1)
    }
    fn blit_anti_rect(&mut self, x: i32, y: i32, width: i32, height: i32, left: u8, right: u8) {
        self.0.blit_anti_rect(x, y, width, height, left, right)
    }
    fn blit_mask(&mut self, mask: &Mask, clip: IRect) {
        self.0.blit_mask(mask, clip)
    }
}

// ------------------------------------------------------------- line clipper

/// An `SkRect`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

const SCALAR_NEARLY_ZERO: f32 = 1.0 / 4096.0;

/// `sk_float_midpoint`.
fn midpoint(a: f32, b: f32) -> f32 {
    ((a as f64 + b as f64) * 0.5) as f32
}

fn pin_unsorted(value: f64, mut lo: f64, mut hi: f64) -> f64 {
    if hi < lo {
        std::mem::swap(&mut lo, &mut hi);
    }
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}

fn sect_with_horizontal(src: &[(f32, f32); 2], y: f32) -> f32 {
    let dy = src[1].1 - src[0].1;
    if dy.abs() <= SCALAR_NEARLY_ZERO {
        midpoint(src[0].0, src[1].0)
    } else {
        let (x0, y0, x1, y1) = (
            src[0].0 as f64,
            src[0].1 as f64,
            src[1].0 as f64,
            src[1].1 as f64,
        );
        let result = x0 + (y as f64 - y0) * (x1 - x0) / (y1 - y0);
        pin_unsorted(result, x0, x1) as f32
    }
}

fn sect_with_vertical(src: &[(f32, f32); 2], x: f32) -> f32 {
    let dx = src[1].0 - src[0].0;
    if dx.abs() <= SCALAR_NEARLY_ZERO {
        midpoint(src[0].1, src[1].1)
    } else {
        let (x0, y0, x1, y1) = (
            src[0].0 as f64,
            src[0].1 as f64,
            src[1].0 as f64,
            src[1].1 as f64,
        );
        (y0 + (x as f64 - x0) * (y1 - y0) / (x1 - x0)) as f32
    }
}

fn nested_lt(a: f32, b: f32, dim: f32) -> bool {
    a <= b && (a < b || dim > 0.0)
}

/// `SkLineClipper::IntersectLine(src, clip, dst)`.
fn intersect_line(src: &[(f32, f32); 2], clip: &Rect) -> Option<[(f32, f32); 2]> {
    let bounds = Rect {
        left: src[0].0.min(src[1].0),
        top: src[0].1.min(src[1].1),
        right: src[0].0.max(src[1].0),
        bottom: src[0].1.max(src[1].1),
    };
    if clip.left <= bounds.left
        && clip.top <= bounds.top
        && clip.right >= bounds.right
        && clip.bottom >= bounds.bottom
    {
        return Some(*src);
    }
    let (bw, bh) = (bounds.right - bounds.left, bounds.bottom - bounds.top);
    if nested_lt(bounds.right, clip.left, bw)
        || nested_lt(clip.right, bounds.left, bw)
        || nested_lt(bounds.bottom, clip.top, bh)
        || nested_lt(clip.bottom, bounds.top, bh)
    {
        return None;
    }
    let (i0, i1) = if src[0].1 < src[1].1 { (0, 1) } else { (1, 0) };
    let mut tmp = *src;
    if tmp[i0].1 < clip.top {
        tmp[i0] = (sect_with_horizontal(src, clip.top), clip.top);
    }
    if tmp[i1].1 > clip.bottom {
        tmp[i1] = (sect_with_horizontal(src, clip.bottom), clip.bottom);
    }
    let (i0, i1) = if tmp[0].0 < tmp[1].0 { (0, 1) } else { (1, 0) };
    if (tmp[i1].0 <= clip.left || tmp[i0].0 >= clip.right)
        && (tmp[0].0 != tmp[1].0 || tmp[0].0 < clip.left || tmp[0].0 > clip.right)
    {
        return None;
    }
    if tmp[i0].0 < clip.left {
        tmp[i0] = (clip.left, sect_with_vertical(&tmp, clip.left));
    }
    if tmp[i1].0 > clip.right {
        tmp[i1] = (clip.right, sect_with_vertical(&tmp, clip.right));
    }
    Some(tmp)
}

/// `SkScan::AntiHairLineRgn` for one segment, `clip` the device clip (the
/// canvas) or `None` when the segment's bounds lie inside it.
fn anti_hair_line(pts: [(f32, f32); 2], clip: Option<IRect>, blitter: &mut dyn Blitter) {
    const MAX: f32 = 32767.0;
    let fixed_bounds = Rect {
        left: -MAX,
        top: -MAX,
        right: MAX,
        bottom: MAX,
    };
    let Some(mut pts) = intersect_line(&pts, &fixed_bounds) else {
        return;
    };
    if let Some(c) = clip {
        let clip_bounds = Rect {
            left: c.left as f32 - 1.0,
            top: c.top as f32 - 1.0,
            right: c.right as f32 + 1.0,
            bottom: c.bottom as f32 + 1.0,
        };
        match intersect_line(&pts, &clip_bounds) {
            Some(p) => pts = p,
            None => return,
        }
    }
    let (x0, y0) = (to_fdot6(pts[0].0), to_fdot6(pts[0].1));
    let (x1, y1) = (to_fdot6(pts[1].0), to_fdot6(pts[1].1));
    if let Some(c) = clip {
        let ir = IRect {
            left: fdot6_floor(x0.min(x1)) - 1,
            top: fdot6_floor(y0.min(y1)) - 1,
            right: fdot6_ceil(x0.max(x1)) + 1,
            bottom: fdot6_ceil(y0.max(y1)) + 1,
        };
        if !c.intersects(&ir) {
            return;
        }
        if !c.contains(&ir) {
            // SkRegion::Cliperator over a rectangular region: the clip
            // intersected with the segment's bounds.
            if let Some(r) = c.intersect(&ir) {
                do_anti_hairline(x0, y0, x1, y1, Some(r), blitter);
            }
            return;
        }
    }
    do_anti_hairline(x0, y0, x1, y1, None, blitter);
}

/// `SkPoint::normalize` (through doubles); `None` for a zero or
/// non-finite vector.
fn normalize(x: f32, y: f32) -> Option<(f32, f32)> {
    let (xx, yy) = (x as f64, y as f64);
    let dmag = (xx * xx + yy * yy).sqrt();
    let dscale = 1.0 / dmag;
    let (nx, ny) = ((x as f64 * dscale) as f32, (y as f64 * dscale) as f32);
    if !nx.is_finite() || !ny.is_finite() || (nx == 0.0 && ny == 0.0) {
        return None;
    }
    Some((nx, ny))
}

/// `SkScan::AntiHairRoundPath` of a moveTo/lineTo path: `hair_path` with
/// `extend_pts<kRound_Cap>` at both ends.
fn anti_hair_round_line(p0: (f32, f32), p1: (f32, f32), canvas: IRect, blitter: &mut dyn Blitter) {
    // The path's bounds, rounded out and outset by 2 for the caps.
    let ibounds = IRect {
        left: p0.0.min(p1.0).floor() as i32 - 2,
        top: p0.1.min(p1.1).floor() as i32 - 2,
        right: p0.0.max(p1.0).ceil() as i32 + 2,
        bottom: p0.1.max(p1.1).ceil() as i32 + 2,
    };
    if !canvas.intersects(&ibounds) {
        return;
    }
    let clip = (!canvas.contains(&ibounds)).then_some(canvas);

    // extend_pts: each end moves out along its tangent by the area of a
    // half disc of radius 1/2 (π/8). The end is extended after the start
    // has moved, from the moved start.
    let cap_outset = std::f32::consts::PI / 8.0;
    let mut pts = [p0, p1];
    let first_tangent = normalize(pts[0].0 - pts[1].0, pts[0].1 - pts[1].1).unwrap_or((1.0, 0.0));
    pts[0].0 += first_tangent.0 * cap_outset;
    pts[0].1 += first_tangent.1 * cap_outset;
    let last_tangent = normalize(pts[1].0 - pts[0].0, pts[1].1 - pts[0].1).unwrap_or((-1.0, 0.0));
    pts[1].0 += last_tangent.0 * cap_outset;
    pts[1].1 += last_tangent.1 * cap_outset;

    anti_hair_line(pts, clip, blitter);
}

// ------------------------------------------------------------------- canvas

/// The overlay canvas: a transparent premultiplied N32 backing store that
/// strokes the tracer's segments as Chromium's software canvas does (see
/// the module documentation).
///
/// It implements the operations the reference issues: `clearRect` of the
/// whole canvas, `lineWidth`, `strokeStyle` as `rgb()`/`rgba()`,
/// `lineCap`/`lineJoin` (recorded; every stroke is drawn with round caps,
/// the only cap the reference sets), and paths of `moveTo`/`lineTo`
/// segments, each stroked as the one-segment line the tracer draws.
#[derive(Clone, Debug)]
pub struct RasterCanvas {
    width: u32,
    height: u32,
    line_width: f32,
    line_cap: String,
    line_join: String,
    /// The stroke colour, straight 8-bit as Blink stores it.
    color: [u8; 4],
    pixels: Vec<PmColor>,
    path: Vec<Vec<(f64, f64)>>,
}

impl RasterCanvas {
    /// A transparent `width` x `height` canvas.
    pub fn new(width: u32, height: u32) -> RasterCanvas {
        RasterCanvas {
            width,
            height,
            line_width: 1.0,
            line_cap: "butt".into(),
            line_join: "miter".into(),
            color: [0, 0, 0, 255],
            pixels: vec![0; width as usize * height as usize],
            path: Vec::new(),
        }
    }

    /// The backing store: premultiplied RGBA8, row 0 at the top.
    pub fn premultiplied(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 4);
        for &p in &self.pixels {
            out.extend_from_slice(&[(p >> 16) as u8, (p >> 8) as u8, p as u8, (p >> 24) as u8]);
        }
        out
    }

    /// The current `lineCap` and `lineJoin` values.
    pub fn line_style(&self) -> (&str, &str) {
        (&self.line_cap, &self.line_join)
    }

    /// The texture the reference uploads from this canvas (see
    /// [`crate::upload_canvas_rgba8`]).
    pub fn upload_image(&self) -> Rgba8Image {
        crate::upload_canvas_rgba8(&self.premultiplied(), self.width, self.height)
    }

    fn canvas_rect(&self) -> IRect {
        IRect {
            left: 0,
            top: 0,
            right: self.width as i32,
            bottom: self.height as i32,
        }
    }

    fn stroke_segment(&mut self, x0d: f64, y0d: f64, x1d: f64, y1d: f64) {
        // Blink keeps path points as floats.
        let (x0, y0, x1, y1) = (x0d as f32, y0d as f32, x1d as f32, y1d as f32);
        // A path whose points coincide is pruned before stroking.
        if x0 == x1 && y0 == y1 {
            return;
        }
        // InflateStrokeRect + ComputeDirtyRect: the bounds outset by
        // lineWidth / 2, rounded out, must intersect the canvas.
        let stroke_radius = 0.5f32 * self.line_width;
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
        if self.color[3] == 0 {
            return;
        }
        // modifyPaintForHairlines: fast_len of (w, 0) is w.
        let w = self.line_width;
        if w <= 1.0 {
            let mut color = self.color;
            if w != 1.0 {
                let scale = (w * 256.0) as u32;
                color[3] = ((color[3] as u32 * scale) >> 8) as u8;
            }
            let canvas = self.canvas_rect();
            let width = self.width as usize;
            let mut blitter = ArgbBlitter::new(&mut self.pixels, width, color);
            if blitter.kind == BlitterKind::Translucent && blitter.src_a == 0 {
                return;
            }
            anti_hair_round_line((x0, y0), (x1, y1), canvas, &mut blitter);
        } else {
            let canvas = self.canvas_rect();
            let width = self.width as usize;
            let mut blitter = ArgbBlitter::new(&mut self.pixels, width, self.color);
            fill::fill_stroke((x0, y0), (x1, y1), w, canvas, &mut blitter);
        }
    }
}

impl Canvas2d for RasterCanvas {
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
        let stride = self.width as usize;
        for row in cy0..cy1 {
            self.pixels[row * stride + cx0..row * stride + cx1].fill(0);
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
        if let Some(color) = parse_rgba_style(style) {
            self.color = color;
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

    fn stroke(canvas: &mut RasterCanvas, style: &str, width: f64, p0: (f64, f64), p1: (f64, f64)) {
        canvas.set_line_cap("round");
        canvas.set_line_join("round");
        canvas.set_line_width(width);
        canvas.set_stroke_style(style);
        canvas.begin_path();
        canvas.move_to(p0.0, p0.1);
        canvas.line_to(p1.0, p1.1);
        canvas.stroke();
    }

    const WIDTHS: [f64; 6] = [0.3, 0.5, 1.0, 1.28, 1.5, 7.0];
    const STYLES: [&str; 3] = [
        "rgba(255, 255, 255, 0.333)",
        "rgba(255, 255, 255, 1)",
        "rgba(0, 0, 0, 1)",
    ];

    #[test]
    fn strokes_crossing_the_edges_stay_inside_the_canvas() {
        // Segments through every edge and corner, for each width class
        // (scaled hairline, hairline, filled) and blitter (translucent,
        // opaque, black): nothing panics and every drawn pixel is on the
        // canvas, which the backing store's size guarantees.
        for width in WIDTHS {
            for style in STYLES {
                let mut canvas = RasterCanvas::new(16, 12);
                for (p0, p1) in [
                    ((-3.0, 5.0), (4.0, 6.2)),
                    ((12.5, -2.0), (14.0, 3.0)),
                    ((15.2, 11.6), (18.0, 14.0)),
                    ((-1.0, -1.0), (0.7, 0.4)),
                    ((7.0, 11.9), (7.3, 13.5)),
                    ((-40.0, 6.0), (60.0, 6.5)),
                    ((8.0, -30.0), (8.5, 40.0)),
                    ((-1e9, 5.0), (1e9, 7.0)),
                    ((9.0, -1e7), (9.5, 1e7)),
                ] {
                    stroke(&mut canvas, style, width, p0, p1);
                }
                assert!(
                    canvas.premultiplied().iter().any(|&b| b != 0),
                    "width {width} {style} drew nothing"
                );
            }
        }
    }

    #[test]
    fn strokes_off_the_canvas_or_degenerate_draw_nothing() {
        for width in WIDTHS {
            for style in STYLES {
                let mut canvas = RasterCanvas::new(16, 12);
                for (p0, p1) in [
                    ((-20.0, 5.0), (-10.0, 6.0)),
                    ((30.0, 5.0), (40.0, 6.0)),
                    ((5.0, -20.0), (6.0, -9.0)),
                    ((5.0, 30.0), (6.0, 40.0)),
                    ((5.5, 5.5), (5.5, 5.5)),
                    ((1e9, -1e9), (2e9, 1e9)),
                    ((f64::NAN, 2.0), (3.0, 4.0)),
                ] {
                    stroke(&mut canvas, style, width, p0, p1);
                }
                assert!(
                    canvas.premultiplied().iter().all(|&b| b == 0),
                    "width {width} {style} drew pixels"
                );
            }
        }
    }

    #[test]
    fn transparent_strokes_draw_nothing() {
        let mut canvas = RasterCanvas::new(8, 8);
        for width in WIDTHS {
            stroke(
                &mut canvas,
                "rgba(255, 0, 0, 0)",
                width,
                (1.0, 1.0),
                (6.0, 5.0),
            );
        }
        assert!(canvas.premultiplied().iter().all(|&b| b == 0));
    }
}
