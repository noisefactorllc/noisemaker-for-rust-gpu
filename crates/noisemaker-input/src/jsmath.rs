//! `Math.sin` and `Math.cos` as V8 computes them in Node: V8's port of fdlibm,
//! which the automation evaluators use so oscillator values match the
//! reference bit for bit. The implementation lives in
//! [`noisemaker_dsl::jsmath`] (the cosine palettes of `noisemaker_dsl` use it
//! too); this module re-exports it under its established path.

pub use noisemaker_dsl::jsmath::{js_cos, js_sin};
