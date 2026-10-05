//! Port of the reference CPU worm tracer (shaders/src/cpu/wormTracer.js:
//! `SeededRNG`, `valueNoiseField`, `traceWorms`), which draws the overlay
//! textures of filter/fibers, filter/scratches and filter/strayHair.
//!
//! The reference runs in JavaScript, so every value here is an IEEE double
//! evaluated in the reference's operation order: [`SeededRng`] reproduces
//! the generator bit for bit, including the rounding of its
//! double-precision multiplications; the flow field is stored as f32, as the
//! reference's Float32Array is, and read with typed-array semantics (an
//! index outside the array reads `undefined`, i.e. NaN); `Math.sin`,
//! `Math.cos` and `Math.log` are the correctly rounded functions Chromium's
//! V8 returns ([`crate::js`]). The tracer emits the reference's exact
//! sequence of canvas operations to a [`Canvas2d`]: the stroke colour as the
//! same `rgba(...)` string, coordinates as the same doubles.

use crate::js;

/// `Math.PI * 2`.
pub const TAU: f64 = std::f64::consts::PI * 2.0;

/// The 2D canvas operations the reference's asyncInit overlays issue, in
/// the reference's vocabulary (`CanvasRenderingContext2D`). Numbers are the
/// doubles JavaScript passes; styles are the strings it assigns.
pub trait Canvas2d {
    /// `canvas.width`.
    fn width(&self) -> u32;
    /// `canvas.height`.
    fn height(&self) -> u32;
    /// `ctx.clearRect(x, y, w, h)`.
    fn clear_rect(&mut self, x: f64, y: f64, w: f64, h: f64);
    /// `ctx.lineCap = cap`.
    fn set_line_cap(&mut self, cap: &str);
    /// `ctx.lineJoin = join`.
    fn set_line_join(&mut self, join: &str);
    /// `ctx.lineWidth = width`.
    fn set_line_width(&mut self, width: f64);
    /// `ctx.strokeStyle = style`.
    fn set_stroke_style(&mut self, style: &str);
    /// `ctx.beginPath()`.
    fn begin_path(&mut self);
    /// `ctx.moveTo(x, y)`.
    fn move_to(&mut self, x: f64, y: f64);
    /// `ctx.lineTo(x, y)`.
    fn line_to(&mut self, x: f64, y: f64);
    /// `ctx.stroke()`.
    fn stroke(&mut self);
}

/// The reference `SeededRNG` (a PCG-style generator evaluated in doubles).
#[derive(Clone, Debug)]
pub struct SeededRng {
    state: u32,
}

impl SeededRng {
    /// `new SeededRNG(seed)`: `((seed >>> 0) * 747796405 + 2891336453) >>> 0`.
    pub fn new(seed: f64) -> SeededRng {
        let seed = js::to_uint32(seed) as f64;
        SeededRng {
            state: js::to_uint32(seed * 747796405.0 + 2891336453.0),
        }
    }

    /// `next()`: a 32-bit word.
    pub fn next_u32(&mut self) -> u32 {
        // The products exceed 2^53 and round, as in the reference.
        self.state = js::to_uint32(self.state as f64 * 747796405.0 + 2891336453.0);
        let shift = (self.state >> 28) + 4;
        // `^` works on Int32: the operands are reinterpreted as signed.
        let mixed = ((self.state >> shift) ^ self.state) as i32;
        let word = js::to_uint32(mixed as f64 * 277803737.0);
        (word >> 22) ^ word
    }

    /// `float()`: `next() / 4294967295`, in [0, 1].
    pub fn float(&mut self) -> f64 {
        self.next_u32() as f64 / 4294967295.0
    }

    /// `int(min, max)`: `min + (next() % (max - min + 1))`.
    pub fn int(&mut self, min: f64, max: f64) -> f64 {
        min + (self.next_u32() as f64 % (max - min + 1.0))
    }

    /// `normal(mean, std)`: Box-Muller.
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = js::math_max(self.float(), 1e-10);
        let u2 = self.float();
        mean + std * (-2.0 * js::log(u1)).sqrt() * js::cos(TAU * u2)
    }
}

/// A Float32Array read with a double index: `undefined` (NaN in
/// arithmetic) off the array or at a non-integer index.
#[derive(Clone, Debug)]
pub struct Float32Field {
    values: Vec<f32>,
}

impl Float32Field {
    fn new(len: usize) -> Float32Field {
        Float32Field {
            values: vec![0.0; len],
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// True when the field has no elements.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// `array[index]` as a Number.
    pub fn at(&self, index: f64) -> f64 {
        if index.is_nan()
            || index < 0.0
            || index >= self.values.len() as f64
            || index.floor() != index
        {
            return f64::NAN;
        }
        self.values[index as usize] as f64
    }

    /// The stored f32 values.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Largest Float32Array the port allocates for the flow field and its grid:
/// 2^28 elements, the pixel area of the largest canvas Chromium draws on
/// (16384 x 16384). Beyond it the reference fails (a RangeError from the
/// allocation, or a canvas without a backing store); the three effects stay
/// far below it (their grids are at most 6 x 6).
const MAX_FIELD_ELEMENTS: f64 = 268435456.0;

/// Port of `valueNoiseField(w, h, freq, rng)`: a low-frequency value noise
/// field of `w * h` values in [0, 1], stored as f32. `None` where the
/// reference throws (a typed-array length it cannot allocate).
pub fn value_noise_field(w: u32, h: u32, freq: f64, rng: &mut SeededRng) -> Option<Float32Field> {
    let gw = freq.ceil() + 2.0;
    let gh = freq.ceil() + 2.0;
    let mut length = gw * gh;
    if length.is_nan() {
        length = 0.0; // ToIndex(NaN) = 0
    }
    if !length.is_finite() || length < 0.0 || length > MAX_FIELD_ELEMENTS {
        return None;
    }
    let mut grid = Float32Field::new(length as usize);
    for v in grid.values.iter_mut() {
        *v = rng.float() as f32;
    }
    if (w as f64) * (h as f64) > MAX_FIELD_ELEMENTS {
        return None;
    }
    let mut field = Float32Field::new(w as usize * h as usize);
    let (wd, hd) = (w as f64, h as f64);
    for y in 0..h {
        for x in 0..w {
            let fx = (x as f64 / wd) * freq;
            let fy = (y as f64 / hd) * freq;
            let ix = fx.floor();
            let iy = fy.floor();
            let dx = fx - ix;
            let dy = fy - iy;
            let sx = dx * dx * (3.0 - 2.0 * dx);
            let sy = dy * dy * (3.0 - 2.0 * dy);
            let tl = grid.at(iy * gw + ix);
            let tr = grid.at(iy * gw + ix + 1.0);
            let bl = grid.at((iy + 1.0) * gw + ix);
            let br = grid.at((iy + 1.0) * gw + ix + 1.0);
            field.values[(y * w + x) as usize] = ((tl * (1.0 - sx) + tr * sx) * (1.0 - sy)
                + (bl * (1.0 - sx) + br * sx) * sy)
                as f32;
        }
    }
    Some(field)
}

/// Worm behavior (`opts.behavior`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WormBehavior {
    /// `'obedient'`: every worm shares one base rotation.
    Obedient,
    /// `'unruly'`: one random rotation per worm.
    Unruly,
    /// `'chaotic'`: one random rotation per worm.
    Chaotic,
}

/// A worm's colour as `colorFn` returns it: `{r, g, b, a}` Numbers that the
/// tracer writes into `rgba(${r}, ${g}, ${b}, ${a * exposure})`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WormColor {
    /// Red, as written into the style string.
    pub r: f64,
    /// Green.
    pub g: f64,
    /// Blue.
    pub b: f64,
    /// Alpha before the exposure ramp.
    pub a: f64,
}

/// The reference `traceWorms` options. `width` and `height` are the
/// canvas size.
pub struct WormTraceOptions<'a> {
    /// Canvas width.
    pub width: u32,
    /// Canvas height.
    pub height: u32,
    /// Generator seed.
    pub seed: f64,
    /// Worm count scaling: `max(1, floor(maxDim * density))` worms.
    pub density: f64,
    /// Flow field influence multiplier.
    pub kink: f64,
    /// Base stride in normalized units.
    pub stride: f64,
    /// Stride standard deviation.
    pub stride_deviation: f64,
    /// Iterations: `max(1, floor(sqrt(minDim) * duration))`.
    pub duration: f64,
    /// Rotation behavior.
    pub behavior: WormBehavior,
    /// Frequency of the flow noise field.
    pub flow_freq: f64,
    /// Trail width in pixels.
    pub line_width: f64,
    /// `colorFn(rng, wormIndex)`: called once per worm, in worm order, after
    /// the worm's position, stride and rotation were drawn.
    pub color_fn: &'a mut dyn FnMut(&mut SeededRng, usize) -> WormColor,
}

/// How a trace ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceOutcome {
    /// Every worm was drawn.
    Completed,
    /// `isCancelled()` returned true before a worm.
    Cancelled,
    /// The reference throws before drawing (an allocation it cannot make)
    /// or cannot finish (more worms or segments than a page can hold);
    /// nothing was drawn.
    Failed,
}

/// The reference stops with a RangeError (or exhausts the page) beyond
/// these: more worms than a JavaScript array of worm objects can hold in
/// practice, more segments than any browser finishes.
pub const MAX_TRACE_WORMS: f64 = 16777216.0;
/// See [`MAX_TRACE_WORMS`].
pub const MAX_TRACE_SEGMENTS: f64 = 4294967296.0;

struct Worm {
    x: f64,
    y: f64,
    stride: f64,
    rot: f64,
    color: WormColor,
}

/// Port of `traceWorms(ctx, opts)`. `is_cancelled` is polled before each
/// worm; `on_progress` is called where the reference calls
/// `onProgress(ctx.canvas)`: after worms 0, 3, 6, ... (the reference yields
/// to the event loop there) and once at the end.
pub fn trace_worms<C: Canvas2d + ?Sized>(
    ctx: &mut C,
    opts: WormTraceOptions<'_>,
    is_cancelled: &mut dyn FnMut() -> bool,
    on_progress: &mut dyn FnMut(&mut C),
) -> TraceOutcome {
    let width = opts.width as f64;
    let height = opts.height as f64;
    let mut rng = SeededRng::new(opts.seed);
    let min_dim = js::math_min(width, height);
    let max_dim = js::math_max(width, height);
    let stride_scale = max_dim / 1024.0;

    let mut field_rng = SeededRng::new(opts.seed * 31337.0);
    let Some(flow_field) =
        value_noise_field(opts.width, opts.height, opts.flow_freq, &mut field_rng)
    else {
        return TraceOutcome::Failed;
    };

    let count = js::math_max(1.0, (max_dim * opts.density).floor());
    if count > MAX_TRACE_WORMS {
        return TraceOutcome::Failed;
    }
    let iterations = js::math_max(1.0, (min_dim.sqrt() * opts.duration).floor());
    if count * iterations > MAX_TRACE_SEGMENTS {
        return TraceOutcome::Failed;
    }

    let shared_rot = rng.float() * TAU;
    let obedient = opts.behavior == WormBehavior::Obedient;

    // `for (let i = 0; i < count; i++)`: false at once for a NaN count.
    let mut worms: Vec<Worm> = Vec::new();
    let mut i = 0.0;
    while i < count {
        let x = rng.float() * width;
        let y = rng.float() * height;
        let stride = rng.normal(opts.stride, opts.stride_deviation) * stride_scale;
        let rot = if obedient {
            shared_rot
        } else {
            rng.float() * TAU
        };
        let color = (opts.color_fn)(&mut rng, worms.len());
        worms.push(Worm {
            x,
            y,
            stride,
            rot,
            color,
        });
        i += 1.0;
    }

    ctx.set_line_cap("round");
    ctx.set_line_join("round");
    ctx.set_line_width(opts.line_width);

    for (w, worm) in worms.iter().enumerate() {
        if is_cancelled() {
            return TraceOutcome::Cancelled;
        }
        let WormColor { r, g, b, a } = worm.color;
        let prefix = format!(
            "rgba({}, {}, {}, ",
            js::number_to_string(r),
            js::number_to_string(g),
            js::number_to_string(b)
        );
        let mut wx = worm.x;
        let mut wy = worm.y;
        let mut iter = 0.0;
        while iter < iterations {
            // Exposure ramp: 0 -> 1 -> 0 over the worm's lifetime.
            let t = if iterations > 1.0 {
                iter / (iterations - 1.0)
            } else {
                1.0
            };
            let exposure = 1.0 - (1.0 - t * 2.0).abs();

            // Flow field lookup (wrapped coordinates).
            let fx = (((wx % width) + width) % width).floor();
            let fy = (((wy % height) + height) % height).floor();
            let field_val = flow_field.at(fy * width + fx);
            let mut angle = field_val * TAU * opts.kink;
            angle += if obedient { shared_rot } else { worm.rot };

            let new_x = wx + js::sin(angle) * worm.stride;
            let new_y = wy + js::cos(angle) * worm.stride;

            let mut style = prefix.clone();
            style.push_str(&js::number_to_string(a * exposure));
            style.push(')');
            ctx.set_stroke_style(&style);
            ctx.begin_path();
            ctx.move_to(wx, wy);
            ctx.line_to(new_x, new_y);
            ctx.stroke();

            wx = new_x;
            wy = new_y;
            iter += 1.0;
        }
        if w % 3 == 0 {
            on_progress(ctx);
        }
    }
    on_progress(ctx);
    TraceOutcome::Completed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rng_matches_reference_values() {
        // Recorded from the reference SeededRNG in Node:
        // r = new SeededRNG(seed); [r.next(), r.next(), r.next()], r.float()
        for (seed, words, float) in [
            (
                1000.0,
                [1620979842, 3759771664, 3250681287],
                0.6000252083409636,
            ),
            (0.0, [718263531, 106116617, 3991160471], 0.9253103220661427),
            (1.0, [1892593347, 875373008, 2815485151], 0.8897669277828575),
            (
                4294967295.0,
                [3836633866, 1650125961, 1696029588],
                0.4629659118277407,
            ),
            (
                -7.0,
                [1789160138, 2051777833, 1807167790],
                0.18321884311344913,
            ),
            (1.5, [1892593347, 875373008, 2815485151], 0.8897669277828575),
            (
                123456789.0,
                [983580458, 1426394372, 1195579357],
                0.9344786100402658,
            ),
        ] {
            let mut rng = SeededRng::new(seed);
            let got = [rng.next_u32(), rng.next_u32(), rng.next_u32()];
            assert_eq!(got, words, "seed {seed}");
            assert_eq!(rng.float(), float, "seed {seed}");
        }
    }
}
