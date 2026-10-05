//! Structured diagnostics (port of `runtime/backends/diagnostics.js`).
//!
//! Recorded (not thrown) diagnostics: the dimension fallback of
//! `Pipeline.resolveDimension` and the device validation errors the WebGPU backend
//! observes through its uncaptured-error listener.

use noisemaker_dsl::Value;

/// `DIAGNOSTIC_CODES`.
pub mod codes {
    pub const COMPILE: &str = "ERR_SHADER_COMPILE";
    pub const LINK: &str = "ERR_SHADER_LINK";
    pub const MISSING_SOURCE: &str = "ERR_SHADER_MISSING";
    pub const NO_SOURCE: &str = "ERR_NO_WGSL_SOURCE";
    pub const UNIFORM_BLOCK: &str = "ERR_UNIFORM_BLOCK_TOO_LARGE";
    pub const UNKNOWN_FORMAT_FALLBACK: &str = "ERR_UNKNOWN_FORMAT_FALLBACK";
    pub const DIMENSION_FALLBACK: &str = "ERR_DIMENSION_FALLBACK";
    pub const MISSING_RENDER_TARGET: &str = "ERR_MISSING_RENDER_TARGET";
    pub const GL_ERROR: &str = "ERR_GL_ERROR";
    pub const DEVICE_VALIDATION: &str = "ERR_DEVICE_VALIDATION";
}

/// `DiagnosticCollector`: a capped, queryable list of diagnostic records.
#[derive(Debug, Clone)]
pub struct DiagnosticCollector {
    pub cap: usize,
    pub records: Vec<Value>,
}

impl Default for DiagnosticCollector {
    fn default() -> Self {
        DiagnosticCollector::new(64)
    }
}

impl DiagnosticCollector {
    pub fn new(cap: usize) -> Self {
        DiagnosticCollector {
            cap,
            records: Vec::new(),
        }
    }

    /// `add(record)`: append, dropping the oldest record past the cap.
    pub fn add(&mut self, record: Value) {
        self.records.push(record);
        if self.records.len() > self.cap {
            self.records.remove(0);
        }
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.records.clear();
    }
}
