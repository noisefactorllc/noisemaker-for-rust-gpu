//! Port of `palettes.js`: the cosine palette collection
//! (`share/palettes.json`, <https://iquilezles.org/articles/palettes/>) and
//! `samplePalette`.
//!
//! `samplePalette` evaluates `Math.cos` as V8 does in Node
//! ([`crate::jsmath::js_cos`]), where the reference's own palette tests run,
//! so samples match the reference bit for bit (`parity/check_api.mjs` checks
//! every palette over a sweep of positions).

use std::sync::OnceLock;

use crate::error::JsError;
use crate::jsmath::js_cos;
use crate::unparser::jsv::{cannot_read, get_opt, object_member};
use crate::value::{Object, Value};

/// `Math.PI * 2`.
const TAU: f64 = std::f64::consts::PI * 2.0;

/// One cosine palette: `offset + amp * cos(2π (freq t + phase))` per channel,
/// and the color space its output is in (`mode`: `none`, `rgb`, `hsv`,
/// `oklab`).
#[derive(Debug, Clone, PartialEq)]
pub struct CosinePalette {
    pub mode: String,
    pub amp: [f64; 3],
    pub freq: [f64; 3],
    pub offset: [f64; 3],
    pub phase: [f64; 3],
}

/// `PALETTES`: the palette table (`share/palettes.json`), in file order, as
/// the reference's object (`{name: {mode, amp, freq, offset, phase}}`).
pub fn palettes() -> &'static Object {
    static PALETTES: OnceLock<Object> = OnceLock::new();
    PALETTES.get_or_init(|| {
        let text = noisemaker_effects::share_file("share/palettes.json")
            .expect("the catalog ships share/palettes.json");
        let text = std::str::from_utf8(text).expect("palettes.json is UTF-8");
        Value::from_json(text)
            .expect("palettes.json is valid JSON")
            .as_object()
            .cloned()
            .expect("palettes.json is an object")
    })
}

/// The palette names, in table order.
pub fn palette_names() -> Vec<&'static str> {
    palettes().keys().map(String::as_str).collect()
}

/// The palette `name` as a typed record, if the table has it.
pub fn palette(name: &str) -> Option<CosinePalette> {
    let p = palettes().get(name)?;
    let triple = |key: &str| -> [f64; 3] {
        let v = match p.get(key) {
            Value::Array(items) => items.as_slice(),
            _ => &[],
        };
        [0, 1, 2].map(|i| v.get(i).and_then(Value::as_f64).unwrap_or(f64::NAN))
    };
    Some(CosinePalette {
        mode: p.get("mode").as_str().unwrap_or_default().to_owned(),
        amp: triple("amp"),
        freq: triple("freq"),
        offset: triple("offset"),
        phase: triple("phase"),
    })
}

/// One channel of `samplePalette`, in the reference's evaluation order.
fn channel(offset: f64, amp: f64, freq: f64, phase: f64, t: f64) -> f64 {
    offset + amp * js_cos(TAU * (freq * t * 0.875 + 0.0625 + phase))
}

impl CosinePalette {
    /// The palette at position `t` (0..1), as `samplePalette` computes it.
    pub fn sample(&self, t: f64) -> [f64; 3] {
        [0, 1, 2].map(|i| channel(self.offset[i], self.amp[i], self.freq[i], self.phase[i], t))
    }
}

/// `samplePalette(name, t)`: the RGB (or the palette's color space) values of
/// palette `name` at position `t` (0..1). An unknown name throws
/// `Error: Unknown palette <name>`; the names of `Object.prototype` members
/// (`constructor`, `toString`, ...) read an inherited member that has no
/// channel arrays, and throw the reference's `TypeError`.
pub fn sample_palette(name: &str, t: f64) -> Result<[f64; 3], JsError> {
    if let Some(p) = palette(name) {
        return Ok(p.sample(t));
    }
    let p = object_member(palettes(), name);
    if !p.is_truthy() {
        return Err(JsError::error(format!("Unknown palette {name}")));
    }
    // `p.offset[i]` on an inherited member: `p.offset` is undefined.
    Err(cannot_read(&get_opt(&p, "offset"), "0"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_samples() {
        // test/palettes.test.js, as Node prints the samples.
        assert_eq!(
            sample_palette("grayscale", 0.0).unwrap(),
            [0.9619397662556434; 3]
        );
        assert_eq!(
            sample_palette("hypercolor", 0.5).unwrap(),
            [0.0577177227653477, 0.20208660251050242, 0.32675983715483087]
        );
        assert!(
            sample_palette("bogus", 0.0)
                .unwrap_err()
                .to_string()
                .contains("Unknown palette bogus")
        );
        assert!(
            sample_palette("constructor", 0.0)
                .unwrap_err()
                .to_string()
                .contains("reading '0'")
        );
        assert_eq!(palette_names().len(), palettes().len());
    }
}
