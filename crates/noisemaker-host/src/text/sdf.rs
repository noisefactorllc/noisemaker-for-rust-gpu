//! Distance-field glyphs: how Skia's GPU text draws a run whose device text
//! size is at least 162 px (`kLargeDFFontLimit`) and below the path size
//! (`sktext::gpu::SDFTControl`).
//!
//! - The glyph's mask is rendered once, unrotated and without subpixel
//!   offset, at the strike size: 162 px for a 162 px run, 256 px above
//!   (`kExtraLargeDFFontLimit` on macOS); the run is scaled by `size /
//!   strike`. The strike is hinted, so CoreGraphics smooths the glyph
//!   ([`super::raster::Path::fill_smoothed`]) and Skia converts the smoothed
//!   values to linear coverage ([`linear_coverage`]).
//! - [`distance_field`] turns the mask into an 8-bit distance field padded
//!   by 4 texels (`SkGenerateDistanceFieldFromA8Image`); a glyph whose
//!   padded field is larger than 256 texels is drawn as a glyph mask (or a
//!   path) instead.
//! - The glyph's quad (the field inset by 2 texels) is drawn with the field
//!   sampled bilinearly: distance = 7.96875 * (texel - 128/255), coverage =
//!   smoothstep(-w, w, distance) with w = 0.65 times the texels per device
//!   pixel ([`coverage`]).
//!
//! Measured against Chromium 153 on macOS 26 (rows of stems at 162, 200 and
//! 240 px): with the smoothed, linearized mask, no gamma distance
//! adjustment (Chromium on macOS does not define `SK_GAMMA_APPLY_TO_A8`,
//! and its direct A8 masks show no gamma either) and the dark-on-light
//! linearization whatever the text colour, stem edges agree within 0.02 px
//! on average; without smoothing they are 0.2 to 0.6 px off.

/// `SK_DistanceFieldPad`: texels of field around the glyph mask.
pub const PAD: i64 = 4;

/// `SK_DistanceFieldInset`: texels of the field outside the drawn quad.
pub const INSET: i64 = 2;

/// `SK_DistanceFieldMagnitude`: the largest distance stored, in texels.
const MAGNITUDE: f32 = 4.0;

/// `kLargeDFFontLimit`: the device text size from which runs are drawn
/// with distance fields.
pub const MIN_SIZE: f32 = 162.0;

/// `kExtraLargeDFFontLimit` (macOS): the strike size above 162 px.
const EXTRA_LARGE_SIZE: f32 = 256.0;

/// `SK_DistanceFieldAAFactor`.
const AA_FACTOR: f32 = 0.65;

/// `SkScalarNearlyZero` tolerance.
const NEARLY_ZERO: f32 = 1.0 / 4096.0;

/// The strike size of a distance-field run (`SDFTControl::getSDFFont`):
/// `device_size` is `size * getMaxScale()`, replaced by `text_size` when
/// nearly equal.
pub fn strike_size(text_size: f32, device_size: f32) -> f32 {
    let scaled = if device_size <= 0.0 || (text_size - device_size).abs() <= NEARLY_ZERO {
        text_size
    } else {
        device_size
    };
    if scaled <= MIN_SIZE {
        MIN_SIZE
    } else {
        EXTRA_LARGE_SIZE
    }
}

#[derive(Clone, Copy)]
struct Cell {
    alpha: f32,
    dist_sq: f32,
    dist: (f32, f32),
}

const LEFT: u8 = 0x01;
const RIGHT: u8 = 0x02;
const TOP_LEFT: u8 = 0x04;
const TOP: u8 = 0x08;
const TOP_RIGHT: u8 = 0x10;
const BOTTOM_LEFT: u8 = 0x20;
const BOTTOM: u8 = 0x40;
const BOTTOM_RIGHT: u8 = 0x80;

/// `found_edge`: a sharp transition across 128, or two partial pixels.
fn found_edge(image: &[u8], index: usize, width: usize, flags: u8) -> bool {
    let offsets: [isize; 8] = [
        -1,
        1,
        -(width as isize) - 1,
        -(width as isize),
        -(width as isize) + 1,
        width as isize - 1,
        width as isize,
        width as isize + 1,
    ];
    let current = image[index];
    let current_check = current >> 7;
    for (i, offset) in offsets.iter().enumerate() {
        let neighbor = if flags & (1 << i) != 0 {
            image[(index as isize + offset) as usize]
        } else {
            0
        };
        let neighbor_check = neighbor >> 7;
        if current_check != neighbor_check
            || (current_check == 0 && neighbor_check == 0 && current != 0 && neighbor != 0)
        {
            return true;
        }
    }
    false
}

/// `edge_distance` (Gustavson 2011): the distance to an edge from a pixel
/// with coverage `alpha`, given the unit edge normal.
fn edge_distance(direction: (f32, f32), alpha: f32) -> f32 {
    let (mut dx, mut dy) = direction;
    if dx.abs() <= NEARLY_ZERO || dy.abs() <= NEARLY_ZERO {
        return 0.5 - alpha;
    }
    dx = dx.abs();
    dy = dy.abs();
    if dx < dy {
        std::mem::swap(&mut dx, &mut dy);
    }
    let a1num = 0.5 * dy;
    if alpha * dx < a1num {
        0.5 * (dx + dy) - (2.0 * dx * dy * alpha).sqrt()
    } else if alpha * dx < (dx - a1num) {
        (0.5 - alpha) * dx
    } else {
        -0.5 * (dx + dy) + (2.0 * dx * dy * (1.0 - alpha)).sqrt()
    }
}

/// `SkPointPriv::SetLengthFast(v, 1)`: normalized through doubles; a zero
/// or non-finite result is (0, 0).
fn normalize(v: (f32, f32)) -> (f32, f32) {
    let (x, y) = (v.0 as f64, v.1 as f64);
    let scale = 1.0 / (x * x + y * y).sqrt();
    let (nx, ny) = ((x * scale) as f32, (y * scale) as f32);
    if !nx.is_finite() || !ny.is_finite() || (nx == 0.0 && ny == 0.0) {
        (0.0, 0.0)
    } else {
        (nx, ny)
    }
}

/// `pack_distance_field_val<4>`: zero at 128, inside above.
fn pack(dist: f32) -> u8 {
    let d = (-dist).clamp(-MAGNITUDE, MAGNITUDE * 127.0 / 128.0) + MAGNITUDE;
    (d / (2.0 * MAGNITUDE) * 256.0 + 0.5).floor() as u8
}

/// `SkGenerateDistanceFieldFromA8Image`: the distance field of a `width` x
/// `height` A8 mask, `(width + 8) x (height + 8)` texels, the mask at
/// texel (4, 4).
pub fn distance_field(mask: &[u8], width: usize, height: usize) -> Vec<u8> {
    // the mask with a one-pixel zero border
    let (cw, ch) = (width + 2, height + 2);
    let mut copy = vec![0u8; cw * ch];
    for y in 0..height {
        copy[(y + 1) * cw + 1..(y + 1) * cw + 1 + width]
            .copy_from_slice(&mask[y * width..(y + 1) * width]);
    }
    // working data with one more cell of "infinitely far" on each side
    let pad = PAD as usize + 1;
    let (dw, dh) = (width + 2 * pad, height + 2 * pad);
    let far = Cell {
        alpha: 0.0,
        dist_sq: 0.0,
        dist: (0.0, 0.0),
    };
    let mut data = vec![far; dw * dh];
    let mut edges = vec![false; dw * dh];
    // init_glyph_data: the bordered copy at (PAD, PAD)
    let base = PAD as usize;
    for j in 0..ch {
        for i in 0..cw {
            let value = copy[j * cw + i];
            let d = (base + j) * dw + base + i;
            data[d].alpha = if value == 255 {
                1.0
            } else {
                #[allow(clippy::excessive_precision)]
                let scale = 0.00392156862f32;
                value as f32 * scale
            };
            let mut flags = 0xffu8;
            if i == 0 {
                flags &= !(LEFT | TOP_LEFT | BOTTOM_LEFT);
            }
            if i == cw - 1 {
                flags &= !(RIGHT | TOP_RIGHT | BOTTOM_RIGHT);
            }
            if j == 0 {
                flags &= !(TOP_LEFT | TOP | TOP_RIGHT);
            }
            if j == ch - 1 {
                flags &= !(BOTTOM_LEFT | BOTTOM | BOTTOM_RIGHT);
            }
            if found_edge(&copy, j * cw + i, cw, flags) {
                edges[d] = true;
            }
        }
    }
    // init_distances
    let sqrt2 = std::f32::consts::SQRT_2;
    for j in 0..dh {
        for i in 0..dw {
            let c = j * dw + i;
            if edges[c] {
                let a = |di: isize, dj: isize| {
                    data[((j as isize + dj) as usize) * dw + (i as isize + di) as usize].alpha
                };
                let gx =
                    a(1, -1) - a(-1, -1) + sqrt2 * a(1, 0) - sqrt2 * a(-1, 0) + a(1, 1) - a(-1, 1);
                let gy =
                    a(-1, 1) - a(-1, -1) + sqrt2 * a(0, 1) - sqrt2 * a(0, -1) + a(1, 1) - a(1, -1);
                let grad = normalize((gx, gy));
                let dist = edge_distance(grad, data[c].alpha);
                data[c].dist = (grad.0 * dist, grad.1 * dist);
                data[c].dist_sq = dist * dist;
            } else {
                data[c].dist_sq = 2_000_000.0;
                data[c].dist = (1000.0, 1000.0);
            }
        }
    }
    // Danielsson's 8SSEDT: each pass offers the cell a neighbour's vector
    // extended by one step, at the neighbour's squared distance updated
    // incrementally (in the reference's operation order).
    fn offer(data: &mut [Cell], c: usize, n: usize, step: (f32, f32)) {
        let Cell {
            dist_sq, dist: v, ..
        } = data[n];
        let candidate = match step {
            (-1.0, -1.0) => dist_sq - 2.0 * (v.0 + v.1 - 1.0),
            (0.0, -1.0) => dist_sq - 2.0 * v.1 + 1.0,
            (1.0, -1.0) => dist_sq + 2.0 * (v.0 - v.1 + 1.0),
            (-1.0, 0.0) => dist_sq - 2.0 * v.0 + 1.0,
            (1.0, 0.0) => dist_sq + 2.0 * v.0 + 1.0,
            (-1.0, 1.0) => dist_sq - 2.0 * (v.0 - v.1 - 1.0),
            (0.0, 1.0) => dist_sq + 2.0 * v.1 + 1.0,
            _ => dist_sq + 2.0 * (v.0 + v.1 + 1.0),
        };
        if candidate < data[c].dist_sq {
            data[c].dist_sq = candidate;
            data[c].dist = (v.0 + step.0, v.1 + step.1);
        }
    }
    for j in 1..dh - 1 {
        for i in 1..dw - 1 {
            let c = j * dw + i;
            if !edges[c] {
                // F1: upper left, up, upper right, left
                offer(&mut data, c, c - dw - 1, (-1.0, -1.0));
                offer(&mut data, c, c - dw, (0.0, -1.0));
                offer(&mut data, c, c - dw + 1, (1.0, -1.0));
                offer(&mut data, c, c - 1, (-1.0, 0.0));
            }
        }
        for i in (1..dw - 1).rev() {
            let c = j * dw + i;
            if !edges[c] {
                // F2: right
                offer(&mut data, c, c + 1, (1.0, 0.0));
            }
        }
    }
    for j in (1..dh - 1).rev() {
        for i in 1..dw - 1 {
            let c = j * dw + i;
            if !edges[c] {
                // B1: left
                offer(&mut data, c, c - 1, (-1.0, 0.0));
            }
        }
        for i in (1..dw - 1).rev() {
            let c = j * dw + i;
            if !edges[c] {
                // B2: right, bottom left, bottom, bottom right
                offer(&mut data, c, c + 1, (1.0, 0.0));
                offer(&mut data, c, c + dw - 1, (-1.0, 1.0));
                offer(&mut data, c, c + dw, (0.0, 1.0));
                offer(&mut data, c, c + dw + 1, (1.0, 1.0));
            }
        }
    }
    let (ow, oh) = (dw - 2, dh - 2);
    let mut field = vec![0u8; ow * oh];
    for j in 1..dh - 1 {
        for i in 1..dw - 1 {
            let cell = data[j * dw + i];
            let dist = if cell.alpha > 0.5 {
                -cell.dist_sq.sqrt()
            } else {
                cell.dist_sq.sqrt()
            };
            field[(j - 1) * ow + i - 1] = pack(dist);
        }
    }
    field
}

/// `SkScalerContext_Mac::generateImage` for a smoothed A8 glyph: CoreGraphics
/// encodes smoothed coverage with gamma 2 on the black-on-white bitmap
/// (`gLinearCoverageFromCGLCDValue`, round(x^2 * 255) of each channel), so
/// coverage `value` becomes 255 - round(((255 - value) / 255)^2 * 255).
pub fn linear_coverage(value: u8) -> u8 {
    let x = (255 - value) as f32 / 255.0;
    255 - (x * x * 255.0 + 0.5).floor() as u8
}

/// Coverage at a texel coordinate (`u`, `v`) of a `width`-texel-wide field:
/// the bilinear sample, unpacked, through the smoothstep of half-width
/// `aa_width`.
pub fn coverage(field: &[u8], width: usize, u: f32, v: f32, aa_width: f32) -> f32 {
    let height = field.len() / width;
    let texel = |x: i64, y: i64| -> f32 {
        if x < 0 || y < 0 || x >= width as i64 || y >= height as i64 {
            0.0
        } else {
            field[y as usize * width + x as usize] as f32 / 255.0
        }
    };
    let (x, y) = (u - 0.5, v - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (ix, iy) = (x0 as i64, y0 as i64);
    let top = texel(ix, iy) + (texel(ix + 1, iy) - texel(ix, iy)) * fx;
    let bottom = texel(ix, iy + 1) + (texel(ix + 1, iy + 1) - texel(ix, iy + 1)) * fx;
    let sample = top + (bottom - top) * fy;
    let dist = 7.96875 * (sample - 0.501_960_8);
    let t = ((dist + aa_width) / (2.0 * aa_width)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The smoothstep half-width for `texels_per_pixel`.
pub fn aa_width(texels_per_pixel: f32) -> f32 {
    AA_FACTOR * texels_per_pixel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoothed_coverage_is_linearized_on_white() {
        assert_eq!(linear_coverage(0), 0);
        assert_eq!(linear_coverage(255), 255);
        // half coverage: 255 - round(127.5^2 / 255)
        assert_eq!(linear_coverage(128), 192);
    }

    #[test]
    fn field_of_a_square_measures_distances() {
        // a 6 x 6 opaque square in an 8 x 8 mask
        let mut mask = vec![0u8; 64];
        for y in 1..7 {
            for x in 1..7 {
                mask[y * 8 + x] = 255;
            }
        }
        let field = distance_field(&mask, 8, 8);
        assert_eq!(field.len(), 16 * 16);
        let at = |x: usize, y: usize| field[(y + 4) * 16 + x + 4];
        // inside is above 128, outside below, far outside clamps to 0
        assert!(at(4, 4) > 160);
        assert!(at(0, 4) < 128);
        assert_eq!(field[0], 0);
        // symmetric
        assert_eq!(at(1, 4), at(6, 4));
        assert_eq!(at(4, 1), at(4, 6));
    }

    #[test]
    fn strike_sizes_follow_the_size_bands() {
        assert_eq!(strike_size(162.0, 162.0), 162.0);
        assert_eq!(strike_size(163.0, 163.0), 256.0);
        assert_eq!(strike_size(255.0, 255.0), 256.0);
    }
}
