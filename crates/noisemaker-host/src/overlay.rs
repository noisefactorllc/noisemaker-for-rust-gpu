//! The asyncInit overlays of filter/fibers, filter/scratches and
//! filter/strayHair: ports of their `asyncInit` methods
//! (shaders/effects/filter/{fibers,scratches,strayHair}/definition.js).
//!
//! Each creates a canvas of the render size, clears it, uploads it as the
//! node's `overlayTex`, then traces layers of worms on it
//! ([`crate::worm::trace_worms`]) and uploads it again after every third
//! worm and at the end of each layer. The reference pipeline maps the name
//! to the graph-scoped texture `${nodeId}_overlayTex` and uploads with
//! `updateTextureFromSource(texId, canvas, { flipY: true })`.
//!
//! Reference lifecycle (shaders/src/runtime/pipeline.js): `initAsyncEffects`
//! runs every asyncInit after texture allocation with `params =
//! {...globalUniforms}`; `checkAsyncRegen` re-runs a node 300 ms after one of
//! its scalar step values (other than `alpha`) changed, with `params` = the
//! step's values. A new run cancels the previous one (`isCancelled`). The
//! overlay a settled pipeline shows is the completed trace of the step's
//! values, which [`render_async_overlay`] returns.

use crate::Rgba8Image;
use crate::canvas::StrokeCanvas;
use crate::js;
pub use crate::js::JsValue;
use crate::worm::{
    Canvas2d, SeededRng, TraceOutcome, WormBehavior, WormColor, WormTraceOptions, trace_worms,
};

/// The texture name every overlay effect uploads (`textures.overlayTex`).
pub const OVERLAY_TEXTURE: &str = "overlayTex";

/// An effect whose definition has an asyncInit overlay.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OverlayEffect {
    /// filter/fibers: four layers of chaotic coloured worms.
    Fibers,
    /// filter/scratches: four layers of near-straight white worms.
    Scratches,
    /// filter/strayHair: one layer of sparse dark worms.
    StrayHair,
}

impl OverlayEffect {
    /// All overlay effects.
    pub const ALL: [OverlayEffect; 3] = [
        OverlayEffect::Fibers,
        OverlayEffect::Scratches,
        OverlayEffect::StrayHair,
    ];

    /// The effect for a name: `"fibers"`, `"filter/fibers"` or
    /// `"filter.fibers"` (and likewise `scratches`, `strayHair`).
    pub fn from_name(name: &str) -> Option<OverlayEffect> {
        let func = name
            .strip_prefix("filter/")
            .or_else(|| name.strip_prefix("filter."))
            .unwrap_or(name);
        match func {
            "fibers" => Some(OverlayEffect::Fibers),
            "scratches" => Some(OverlayEffect::Scratches),
            "strayHair" => Some(OverlayEffect::StrayHair),
            _ => None,
        }
    }

    /// The effect function name (`func` of the definition).
    pub fn func(self) -> &'static str {
        match self {
            OverlayEffect::Fibers => "fibers",
            OverlayEffect::Scratches => "scratches",
            OverlayEffect::StrayHair => "strayHair",
        }
    }

    /// The default of the definition's `density` global, which the
    /// asyncInit also uses when `params.density` is undefined.
    pub fn default_density(self) -> f64 {
        match self {
            OverlayEffect::Fibers => 0.5,
            OverlayEffect::Scratches => 0.3,
            OverlayEffect::StrayHair => 0.5,
        }
    }

    /// The graph-scoped id the reference uploads the overlay under.
    pub fn texture_id(node_id: &str) -> String {
        format!("{node_id}_{OVERLAY_TEXTURE}")
    }
}

/// The `params` object an asyncInit receives; it reads `seed` and
/// `density`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OverlayParams {
    /// `params.seed` (read as `params.seed || 1`).
    pub seed: JsValue,
    /// `params.density` (read as `params.density !== undefined ?
    /// params.density : <default>`).
    pub density: JsValue,
}

impl OverlayParams {
    /// Numeric `seed` and `density`, as a step's values hold them.
    pub fn new(seed: f64, density: f64) -> OverlayParams {
        OverlayParams {
            seed: JsValue::Number(seed),
            density: JsValue::Number(density),
        }
    }

    /// `params.seed || 1`, as a Number.
    fn seed(&self) -> f64 {
        if self.seed.truthy() {
            self.seed.to_number()
        } else {
            1.0
        }
    }

    /// `params.density !== undefined ? params.density : fallback`, as a
    /// Number.
    fn density(&self, fallback: f64) -> f64 {
        match self.density {
            JsValue::Undefined => fallback,
            ref v => v.to_number(),
        }
    }
}

/// Runs the port of `effect`'s asyncInit on `canvas`, which the caller
/// sizes to the render size. `on_update(name, canvas)` is called wherever
/// the reference calls `updateTexture(name, canvas)`; `is_cancelled` is
/// polled where the reference polls `isCancelled()`.
pub fn run_async_init<C: Canvas2d + ?Sized>(
    effect: OverlayEffect,
    canvas: &mut C,
    params: &OverlayParams,
    is_cancelled: &mut dyn FnMut() -> bool,
    on_update: &mut dyn FnMut(&str, &mut C),
) -> TraceOutcome {
    let (width, height) = (canvas.width(), canvas.height());
    let w = width as f64;
    canvas.clear_rect(0.0, 0.0, w, height as f64);
    on_update(OVERLAY_TEXTURE, canvas);
    let seed = params.seed();
    let density = params.density(effect.default_density());
    let mut progress = |c: &mut C| on_update(OVERLAY_TEXTURE, c);

    match effect {
        OverlayEffect::Fibers => {
            // Python reference: 4 layers of chaotic worms; density scales
            // the worm count: 0.5 + density * 2.0.
            let base_density = 0.5 + density * 2.0;
            for layer in 0..4 {
                if is_cancelled() {
                    return TraceOutcome::Cancelled;
                }
                let layer_seed = seed * 1000.0 + layer as f64 * 137.0;
                let mut color_fn = |rng: &mut SeededRng, _: usize| WormColor {
                    r: (rng.float() * 200.0 + 55.0).floor(),
                    g: (rng.float() * 200.0 + 55.0).floor(),
                    b: (rng.float() * 200.0 + 55.0).floor(),
                    a: 0.5,
                };
                let outcome = trace_worms(
                    canvas,
                    WormTraceOptions {
                        width,
                        height,
                        seed: layer_seed,
                        density: base_density,
                        kink: 5.0 + layer_seed % 5.0,
                        stride: 0.75,
                        stride_deviation: 0.125,
                        duration: 1.0,
                        behavior: WormBehavior::Chaotic,
                        flow_freq: 4.0,
                        line_width: js::math_max(1.5, w / 384.0),
                        color_fn: &mut color_fn,
                    },
                    is_cancelled,
                    &mut progress,
                );
                if outcome != TraceOutcome::Completed {
                    return outcome;
                }
            }
            TraceOutcome::Completed
        }
        OverlayEffect::Scratches => {
            // Python reference: 4 layers alternating obedient / unruly,
            // low kink (0.125 - 0.25) for nearly straight scratches.
            for layer in 0..4 {
                if is_cancelled() {
                    return TraceOutcome::Cancelled;
                }
                let layer_seed = seed * 1000.0 + layer as f64 * 251.0;
                let is_obedient = layer_seed % 2.0 == 0.0;
                let mut color_fn = |_: &mut SeededRng, _: usize| WormColor {
                    r: 255.0,
                    g: 255.0,
                    b: 255.0,
                    a: 1.0,
                };
                let outcome = trace_worms(
                    canvas,
                    WormTraceOptions {
                        width,
                        height,
                        seed: layer_seed,
                        density: 0.1 + density * 0.4,
                        kink: 0.125 + (layer_seed % 50.0) / 400.0,
                        stride: 0.75,
                        stride_deviation: 0.5,
                        duration: 2.0 + layer_seed % 3.0,
                        behavior: if is_obedient {
                            WormBehavior::Obedient
                        } else {
                            WormBehavior::Unruly
                        },
                        flow_freq: 2.0 + layer_seed % 3.0,
                        line_width: js::math_max(0.5, w / 1024.0),
                        color_fn: &mut color_fn,
                    },
                    is_cancelled,
                    &mut progress,
                );
                if outcome != TraceOutcome::Completed {
                    return outcome;
                }
            }
            TraceOutcome::Completed
        }
        OverlayEffect::StrayHair => {
            // Python reference: one layer of sparse unruly worms; dark
            // strands, high kink (5 - 50).
            let layer_seed = seed * 1000.0 + 42.0;
            let mut color_fn = |rng: &mut SeededRng, _: usize| WormColor {
                r: (rng.float() * 30.0).floor(),
                g: (rng.float() * 30.0).floor(),
                b: (rng.float() * 30.0).floor(),
                a: 0.666,
            };
            trace_worms(
                canvas,
                WormTraceOptions {
                    width,
                    height,
                    seed: layer_seed,
                    density: 0.001 + density * 0.004,
                    kink: 5.0 + layer_seed % 45.0,
                    stride: 0.5,
                    stride_deviation: 0.25,
                    duration: 8.0 + layer_seed % 8.0,
                    behavior: WormBehavior::Unruly,
                    flow_freq: 4.0,
                    line_width: js::math_max(1.0, w / 400.0),
                    color_fn: &mut color_fn,
                },
                is_cancelled,
                &mut progress,
            )
        }
    }
}

/// The overlay texture `effect` uploads once its trace completes at
/// `width` x `height`: the RGBA8 bytes of the reference's rgba8unorm
/// texture, row 0 first (see [`crate::upload_canvas_rgba8`]).
pub fn render_async_overlay(
    effect: OverlayEffect,
    width: u32,
    height: u32,
    params: &OverlayParams,
) -> Rgba8Image {
    render_async_overlay_with(effect, width, height, params, &mut || false, &mut |_, _| {}).1
}

/// [`render_async_overlay`] with the reference's cancellation and
/// progressive uploads: `is_cancelled` is polled before each layer and
/// worm, and `on_update(name, canvas)` receives the canvas wherever the
/// reference uploads it (convert with [`StrokeCanvas::upload_image`]).
/// Returns how the trace ended and the canvas as the texture receives it
/// when the trace completes. A cancelled trace returns the canvas where it
/// stopped; the reference no longer uploads it (its texture keeps the last
/// `on_update` state until the trace that cancelled it uploads its own
/// cleared canvas). A failed trace returns the cleared canvas.
pub fn render_async_overlay_with(
    effect: OverlayEffect,
    width: u32,
    height: u32,
    params: &OverlayParams,
    is_cancelled: &mut dyn FnMut() -> bool,
    on_update: &mut dyn FnMut(&str, &StrokeCanvas),
) -> (TraceOutcome, Rgba8Image) {
    let mut canvas = StrokeCanvas::new(width, height);
    let outcome = run_async_init(effect, &mut canvas, params, is_cancelled, &mut |name, c| {
        on_update(name, c)
    });
    (outcome, canvas.upload_image())
}
