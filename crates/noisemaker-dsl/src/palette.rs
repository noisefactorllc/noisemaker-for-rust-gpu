//! Port of `runtime/palette-expansion.js`: the legacy classicNoisedeck palette
//! presets, expanded into the uniforms their shaders read.
//!
//! classicNoisedeck shaders take `paletteOffset`, `paletteAmp`, `paletteFreq`,
//! `palettePhase` (vec3) and `paletteMode` (int) while the UI only sets a 1-based
//! palette index; [`expand_palette`] maps that index to the concrete values. The
//! expander calls it for every `type: "palette"` global so the first frame renders
//! the selected palette, and the runtime calls it again from `setUniform` and from
//! parameter application.
//!
//! [`PALETTES`] is the reference's `PALETTES` table, entry for entry and bit for bit
//! (generated from the reference module; the `reference_parity` test re-checks
//! every entry against the reference's own `expandPalette`). `mode` is already in
//! the classicNoisedeck convention (0 = none, 1 = hsv, 2 = oklab, 3 = rgb).

use crate::error::JsError;
use crate::js::to_number;
use crate::value::{Object, Value};

/// One preset of the reference `PALETTES` table: `{ amp, freq, offset, phase, mode }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PalettePreset {
    pub amp: [f64; 3],
    pub freq: [f64; 3],
    pub offset: [f64; 3],
    pub phase: [f64; 3],
    pub mode: u32,
}

/// The reference `PALETTES` table, in its order (index 1 is the first entry).
pub const PALETTES: [PalettePreset; 55] = [
    // 1: seventiesShirt (rgb)
    PalettePreset {
        amp: [0.76, 0.88, 0.37],
        freq: [1.0, 1.0, 1.0],
        offset: [0.93, 0.97, 0.52],
        phase: [0.21, 0.41, 0.56],
        mode: 3,
    },
    // 2: fiveG (rgb)
    PalettePreset {
        amp: [0.56851584, 0.7740668, 0.23485267],
        freq: [1.0, 1.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.727029, 0.08039695, 0.10427457],
        mode: 3,
    },
    // 3: afterimage (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.3, 0.2, 0.2],
        mode: 3,
    },
    // 4: barstow (rgb)
    PalettePreset {
        amp: [0.45, 0.2, 0.1],
        freq: [1.0, 1.0, 1.0],
        offset: [0.7, 0.2, 0.2],
        phase: [0.5, 0.4, 0.0],
        mode: 3,
    },
    // 5: bloob (rgb)
    PalettePreset {
        amp: [0.09, 0.59, 0.48],
        freq: [1.0, 1.0, 1.0],
        offset: [0.2, 0.31, 0.98],
        phase: [0.88, 0.4, 0.33],
        mode: 3,
    },
    // 6: blueSkies (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.1, 0.4, 0.7],
        phase: [0.1, 0.1, 0.1],
        mode: 3,
    },
    // 7: brushedMetal (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.0, 0.1, 0.2],
        mode: 3,
    },
    // 8: burningSky (rgb)
    PalettePreset {
        amp: [0.7259015, 0.7004237, 0.9494409],
        freq: [1.0, 1.0, 1.0],
        offset: [0.63290054, 0.37883538, 0.29405284],
        phase: [0.0, 0.1, 0.2],
        mode: 3,
    },
    // 9: california (rgb)
    PalettePreset {
        amp: [0.94, 0.33, 0.27],
        freq: [1.0, 1.0, 1.0],
        offset: [0.74, 0.37, 0.73],
        phase: [0.44, 0.17, 0.88],
        mode: 3,
    },
    // 10: columbia (rgb)
    PalettePreset {
        amp: [1.0, 0.7, 1.0],
        freq: [1.0, 1.0, 1.0],
        offset: [1.0, 0.4, 0.9],
        phase: [0.4, 0.5, 0.6],
        mode: 3,
    },
    // 11: cottonCandy (rgb)
    PalettePreset {
        amp: [0.51, 0.39, 0.41],
        freq: [1.0, 1.0, 1.0],
        offset: [0.59, 0.53, 0.94],
        phase: [0.15, 0.41, 0.46],
        mode: 3,
    },
    // 12: darkSatin (hsv)
    PalettePreset {
        amp: [0.0, 0.0, 0.51],
        freq: [1.0, 1.0, 1.0],
        offset: [0.0, 0.0, 0.43],
        phase: [0.0, 0.0, 0.36],
        mode: 1,
    },
    // 13: dealerHat (rgb)
    PalettePreset {
        amp: [0.83, 0.45, 0.19],
        freq: [1.0, 1.0, 1.0],
        offset: [0.79, 0.45, 0.35],
        phase: [0.28, 0.91, 0.61],
        mode: 3,
    },
    // 14: dreamy (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.0, 0.2, 0.25],
        mode: 3,
    },
    // 15: eventHorizon (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.22, 0.48, 0.62],
        phase: [0.1, 0.3, 0.2],
        mode: 3,
    },
    // 16: ghostly (hsv)
    PalettePreset {
        amp: [0.02, 0.92, 0.76],
        freq: [1.0, 1.0, 1.0],
        offset: [0.51, 0.49, 0.51],
        phase: [0.71, 0.23, 0.66],
        mode: 1,
    },
    // 17: grayscale (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [2.0, 2.0, 2.0],
        offset: [0.5, 0.5, 0.5],
        phase: [1.0, 1.0, 1.0],
        mode: 3,
    },
    // 18: hazySunset (rgb)
    PalettePreset {
        amp: [0.79, 0.56, 0.22],
        freq: [1.0, 1.0, 1.0],
        offset: [0.96, 0.5, 0.49],
        phase: [0.15, 0.98, 0.87],
        mode: 3,
    },
    // 19: heatmap (rgb)
    PalettePreset {
        amp: [0.75804377, 0.62868536, 0.2227562],
        freq: [1.0, 1.0, 1.0],
        offset: [0.35536355, 0.12935615, 0.17060602],
        phase: [0.0, 0.25, 0.5],
        mode: 3,
    },
    // 20: hypercolor (rgb)
    PalettePreset {
        amp: [0.79, 0.5, 0.23],
        freq: [1.0, 1.0, 1.0],
        offset: [0.75, 0.47, 0.45],
        phase: [0.08, 0.84, 0.16],
        mode: 3,
    },
    // 21: jester (rgb)
    PalettePreset {
        amp: [0.7, 0.81, 0.73],
        freq: [1.0, 1.0, 1.0],
        offset: [0.1, 0.22, 0.27],
        phase: [0.99, 0.12, 0.94],
        mode: 3,
    },
    // 22: justBlue (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [0.0, 0.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.5, 0.5, 0.5],
        mode: 3,
    },
    // 23: justCyan (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [0.0, 1.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.5, 0.5, 0.5],
        mode: 3,
    },
    // 24: justGreen (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [0.0, 1.0, 0.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.5, 0.5, 0.5],
        mode: 3,
    },
    // 25: justPurple (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 0.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.5, 0.5, 0.5],
        mode: 3,
    },
    // 26: justRed (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 0.0, 0.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.5, 0.5, 0.5],
        mode: 3,
    },
    // 27: justYellow (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 0.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.5, 0.5, 0.5],
        mode: 3,
    },
    // 28: mars (rgb)
    PalettePreset {
        amp: [0.74, 0.33, 0.09],
        freq: [1.0, 1.0, 1.0],
        offset: [0.62, 0.2, 0.2],
        phase: [0.2, 0.1, 0.0],
        mode: 3,
    },
    // 29: modesto (rgb)
    PalettePreset {
        amp: [0.56, 0.68, 0.39],
        freq: [1.0, 1.0, 1.0],
        offset: [0.72, 0.07, 0.62],
        phase: [0.25, 0.4, 0.41],
        mode: 3,
    },
    // 30: moss (rgb)
    PalettePreset {
        amp: [0.78, 0.39, 0.07],
        freq: [1.0, 1.0, 1.0],
        offset: [0.0, 0.53, 0.33],
        phase: [0.94, 0.92, 0.9],
        mode: 3,
    },
    // 31: neptune (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.2, 0.64, 0.62],
        phase: [0.15, 0.2, 0.3],
        mode: 3,
    },
    // 32: netOfGems (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.64, 0.12, 0.84],
        phase: [0.1, 0.25, 0.15],
        mode: 3,
    },
    // 33: organic (rgb)
    PalettePreset {
        amp: [0.42, 0.42, 0.04],
        freq: [1.0, 1.0, 1.0],
        offset: [0.47, 0.27, 0.27],
        phase: [0.41, 0.14, 0.11],
        mode: 3,
    },
    // 34: papaya (rgb)
    PalettePreset {
        amp: [0.65, 0.4, 0.11],
        freq: [1.0, 1.0, 1.0],
        offset: [0.72, 0.45, 0.08],
        phase: [0.71, 0.8, 0.84],
        mode: 3,
    },
    // 35: radioactive (rgb)
    PalettePreset {
        amp: [0.62, 0.79, 0.11],
        freq: [1.0, 1.0, 1.0],
        offset: [0.22, 0.56, 0.17],
        phase: [0.15, 0.1, 0.25],
        mode: 3,
    },
    // 36: royal (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.41, 0.22, 0.67],
        phase: [0.2, 0.25, 0.2],
        mode: 3,
    },
    // 37: santaCruz (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.5, 0.5, 0.5],
        phase: [0.25, 0.5, 0.75],
        mode: 3,
    },
    // 38: sherbet (rgb)
    PalettePreset {
        amp: [0.6059281, 0.17591387, 0.17166573],
        freq: [1.0, 1.0, 1.0],
        offset: [0.5224456, 0.3864609, 0.36020845],
        phase: [0.0, 0.25, 0.5],
        mode: 3,
    },
    // 39: sherbetDouble (rgb)
    PalettePreset {
        amp: [0.6059281, 0.17591387, 0.17166573],
        freq: [2.0, 2.0, 2.0],
        offset: [0.5224456, 0.3864609, 0.36020845],
        phase: [0.0, 0.25, 0.5],
        mode: 3,
    },
    // 40: silvermane (oklab)
    PalettePreset {
        amp: [0.42, 0.0, 0.0],
        freq: [2.0, 2.0, 2.0],
        offset: [0.45, 0.5, 0.42],
        phase: [0.63, 1.0, 1.0],
        mode: 2,
    },
    // 41: skykissed (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.83, 0.6, 0.63],
        phase: [0.3, 0.1, 0.0],
        mode: 3,
    },
    // 42: solaris (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.6, 0.4, 0.1],
        phase: [0.3, 0.2, 0.1],
        mode: 3,
    },
    // 43: spooky (oklab)
    PalettePreset {
        amp: [0.46, 0.73, 0.19],
        freq: [1.0, 1.0, 1.0],
        offset: [0.27, 0.79, 0.78],
        phase: [0.27, 0.16, 0.04],
        mode: 2,
    },
    // 44: springtime (rgb)
    PalettePreset {
        amp: [0.67, 0.25, 0.27],
        freq: [1.0, 1.0, 1.0],
        offset: [0.74, 0.48, 0.46],
        phase: [0.07, 0.79, 0.39],
        mode: 3,
    },
    // 45: sproingtime (rgb)
    PalettePreset {
        amp: [0.9, 0.43, 0.34],
        freq: [1.0, 1.0, 1.0],
        offset: [0.56, 0.69, 0.32],
        phase: [0.03, 0.8, 0.4],
        mode: 3,
    },
    // 46: sulphur (rgb)
    PalettePreset {
        amp: [0.73, 0.36, 0.52],
        freq: [1.0, 1.0, 1.0],
        offset: [0.78, 0.68, 0.15],
        phase: [0.74, 0.93, 0.28],
        mode: 3,
    },
    // 47: summoning (rgb)
    PalettePreset {
        amp: [1.0, 0.0, 0.8],
        freq: [1.0, 1.0, 1.0],
        offset: [0.0, 0.0, 0.0],
        phase: [0.0, 0.5, 0.1],
        mode: 3,
    },
    // 48: superhero (rgb)
    PalettePreset {
        amp: [1.0, 0.25, 0.5],
        freq: [0.5, 0.5, 0.5],
        offset: [0.0, 0.0, 0.25],
        phase: [0.5, 0.0, 0.0],
        mode: 3,
    },
    // 49: toxic (rgb)
    PalettePreset {
        amp: [0.5, 0.5, 0.5],
        freq: [1.0, 1.0, 1.0],
        offset: [0.26, 0.57, 0.03],
        phase: [0.0, 0.1, 0.3],
        mode: 3,
    },
    // 50: tropicalia (oklab)
    PalettePreset {
        amp: [0.28, 0.08, 0.65],
        freq: [1.0, 1.0, 1.0],
        offset: [0.48, 0.6, 0.03],
        phase: [0.1, 0.15, 0.3],
        mode: 2,
    },
    // 51: tungsten (rgb)
    PalettePreset {
        amp: [0.65, 0.93, 0.73],
        freq: [1.0, 1.0, 1.0],
        offset: [0.31, 0.21, 0.27],
        phase: [0.43, 0.45, 0.48],
        mode: 3,
    },
    // 52: vaporwave (rgb)
    PalettePreset {
        amp: [0.9, 0.76, 0.63],
        freq: [1.0, 1.0, 1.0],
        offset: [0.0, 0.19, 0.68],
        phase: [0.43, 0.23, 0.32],
        mode: 3,
    },
    // 53: vibrant (rgb)
    PalettePreset {
        amp: [0.78, 0.63, 0.68],
        freq: [1.0, 1.0, 1.0],
        offset: [0.41, 0.03, 0.16],
        phase: [0.81, 0.61, 0.06],
        mode: 3,
    },
    // 54: vintage (rgb)
    PalettePreset {
        amp: [0.97, 0.74, 0.23],
        freq: [1.0, 1.0, 1.0],
        offset: [0.97, 0.38, 0.35],
        phase: [0.34, 0.41, 0.44],
        mode: 3,
    },
    // 55: vintagePhoto (rgb)
    PalettePreset {
        amp: [0.68, 0.79, 0.57],
        freq: [1.0, 1.0, 1.0],
        offset: [0.56, 0.35, 0.14],
        phase: [0.73, 0.9, 0.99],
        mode: 3,
    },
];

fn vec3(v: [f64; 3]) -> Value {
    Value::Array(v.iter().map(|c| Value::Number(*c)).collect())
}

/// `expandPalette(index)` for a number `index`.
///
/// Returns `Ok(None)` when `index` is out of range (`index <= 0 ||
/// index > PALETTES.length`, which is also false for NaN), otherwise the uniform
/// values `{ paletteOffset, paletteAmp, paletteFreq, palettePhase, paletteMode }`
/// (fresh arrays, in the reference's member order). Like the reference, an index
/// in range that is not an integer (or NaN) reads `PALETTES[index - 1]` as
/// `undefined` and throws the TypeError the reference throws.
pub fn expand_palette(index: f64) -> Result<Option<Object>, JsError> {
    if index <= 0.0 || index > PALETTES.len() as f64 {
        return Ok(None);
    }
    // PALETTES[index - 1]: only an integral key names an element.
    let position = index - 1.0;
    if position.fract() != 0.0 || !(0.0..PALETTES.len() as f64).contains(&position) {
        return Err(JsError::type_error(
            "Cannot read properties of undefined (reading 'offset')",
        ));
    }
    let entry = &PALETTES[position as usize];
    let mut out = Object::new();
    out.insert("paletteOffset", vec3(entry.offset));
    out.insert("paletteAmp", vec3(entry.amp));
    out.insert("paletteFreq", vec3(entry.freq));
    out.insert("palettePhase", vec3(entry.phase));
    out.insert("paletteMode", Value::Number(entry.mode as f64));
    Ok(Some(out))
}

/// `expandPalette(value)` for any argument, with the reference's implicit numeric
/// coercion (the runtime's parameter application passes the converted parameter
/// value through unchanged, whatever its type).
pub fn expand_palette_value(index: &Value) -> Result<Option<Object>, JsError> {
    expand_palette(to_number(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_and_shape() {
        assert_eq!(expand_palette(0.0), Ok(None));
        assert_eq!(expand_palette(-3.0), Ok(None));
        assert_eq!(expand_palette(56.0), Ok(None));
        assert_eq!(expand_palette(f64::INFINITY), Ok(None));
        assert_eq!(expand_palette(f64::NEG_INFINITY), Ok(None));
        let first = expand_palette(1.0).unwrap().unwrap();
        assert_eq!(
            Value::Object(first).to_json().unwrap(),
            r#"{"paletteOffset":[0.93,0.97,0.52],"paletteAmp":[0.76,0.88,0.37],"paletteFreq":[1,1,1],"palettePhase":[0.21,0.41,0.56],"paletteMode":3}"#
        );
        let err = JsError::type_error("Cannot read properties of undefined (reading 'offset')");
        assert_eq!(expand_palette(1.5), Err(err.clone()));
        assert_eq!(expand_palette(f64::NAN), Err(err.clone()));
        assert_eq!(expand_palette_value(&Value::Undefined), Err(err));
        assert_eq!(expand_palette_value(&Value::Null), Ok(None));
        assert_eq!(expand_palette_value(&Value::from("2")), expand_palette(2.0));
        assert_eq!(
            expand_palette_value(&Value::Array(vec![Value::from(3.0)])),
            expand_palette(3.0)
        );
        assert_eq!(
            expand_palette_value(&Value::Bool(true)),
            expand_palette(1.0)
        );
    }

    /// Every entry against the reference's own `expandPalette` (bit-identical
    /// doubles), plus the out-of-range and non-integral behavior. Runs when
    /// `NM_REFERENCE_ROOT` points at the reference checkout and `node` is on PATH.
    #[test]
    fn reference_parity() {
        let Ok(root) = std::env::var("NM_REFERENCE_ROOT") else {
            eprintln!("skipping palette reference parity: NM_REFERENCE_ROOT is not set");
            return;
        };
        let module = std::path::Path::new(&root).join("shaders/src/runtime/palette-expansion.js");
        let script = format!(
            r#"import {{ expandPalette }} from {url};
const probes = [-1, 0, 0.5, 1, 1.5, 2, 54, 55, 55.5, 56, 57, NaN, Infinity, -Infinity];
for (let i = 1; i <= 55; i++) probes.push(i);
const out = probes.map(i => {{
  try {{ const r = expandPalette(i); return {{ ok: r === null ? null : r }} }}
  catch (e) {{ return {{ error: e.name + ': ' + e.message }} }}
}});
process.stdout.write(JSON.stringify({{ probes: probes.map(String), out }}));"#,
            url = serde_json::to_string(&format!("file://{}", module.display())).unwrap()
        );
        let output = match std::process::Command::new("node")
            .args(["--input-type=module", "-e", &script])
            .output()
        {
            Ok(o) if o.status.success() => o,
            Ok(o) => panic!(
                "reference expandPalette failed: {}",
                String::from_utf8_lossy(&o.stderr)
            ),
            Err(e) => {
                eprintln!("skipping palette reference parity: cannot run node: {e}");
                return;
            }
        };
        let parsed: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        let probes = parsed["probes"].as_array().unwrap();
        let outs = parsed["out"].as_array().unwrap();
        assert_eq!(probes.len(), outs.len());
        for (probe, expected) in probes.iter().zip(outs) {
            let index = crate::js::string_to_number(probe.as_str().unwrap());
            let got = expand_palette(index);
            if let Some(message) = expected.get("error") {
                let err = got.expect_err(&format!("index {index} must throw"));
                assert_eq!(format!("{err}"), message.as_str().unwrap(), "index {index}");
                continue;
            }
            let expected = &expected["ok"];
            let got = got.unwrap_or_else(|e| panic!("index {index}: {e}"));
            match got {
                None => assert!(expected.is_null(), "index {index}: expected {expected}"),
                Some(obj) => {
                    let keys: Vec<&String> = obj.keys().collect();
                    let expected_keys: Vec<&String> =
                        expected.as_object().unwrap().keys().collect();
                    assert_eq!(keys, expected_keys, "index {index}: member order");
                    for (key, value) in obj.iter() {
                        let want = &expected[key.as_str()];
                        match value {
                            Value::Array(components) => {
                                let want = want.as_array().unwrap();
                                assert_eq!(components.len(), want.len());
                                for (c, w) in components.iter().zip(want) {
                                    assert_eq!(
                                        c.as_f64().unwrap().to_bits(),
                                        w.as_f64().unwrap().to_bits(),
                                        "index {index} {key}"
                                    );
                                }
                            }
                            Value::Number(n) => assert_eq!(
                                n.to_bits(),
                                want.as_f64().unwrap().to_bits(),
                                "index {index} {key}"
                            ),
                            other => panic!("index {index} {key}: unexpected {other:?}"),
                        }
                    }
                }
            }
        }
    }
}
