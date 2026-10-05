//! GPU renderer for the Noisemaker effect catalog.
//!
//! A port of the reference engine's pipeline executor (`runtime/pipeline.js`) and
//! WebGPU backend (`runtime/backends/webgpu.js`) onto wgpu, rendering the reference
//! WGSL programs unmodified.

pub use noisemaker_dsl as dsl;
