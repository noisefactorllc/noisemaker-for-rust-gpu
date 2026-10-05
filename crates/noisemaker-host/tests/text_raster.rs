//! Text raster test: the port's text canvases against the canvases the
//! reference demo host uploads, captured by `tools/reference-host.mjs
//! text` (the demo's own `_renderTextToCanvas` in Chromium, uploaded with
//! `copyExternalImageToTexture({ flipY: true })` and read back).
//!
//! `NM_HOST_TEXT_GOLDENS` is the capture directory (its manifest.json
//! lists the cases); `NM_HOST_GOLDENS` lists golden-minter output
//! directories (`:`-separated), whose `<program>.textTex_step_<N>.png`
//! captures are checked with the parameters of the program's filter/text
//! pass in `<program>.graph.json`. Without them the tests explain why and
//! pass. Reports
//! per case the differing pixels, the pixels whose alpha differs by more
//! than 1, max-abs-diff, mean alpha difference, SSIM (as parity/compare.py
//! computes it, on colour and on alpha), the ink (alpha sum) ratio and the
//! drawn-footprint overlap, and asserts the measured bounds:
//!
//! - text the bundled or installed TrueType fonts draw at 19 px and more:
//!   alpha SSIM >= 0.9999 and footprint >= 0.99 (glyph masks reproduce
//!   CoreGraphics' coverage to within 1 except a few pixels, distance
//!   fields from 162 px and multisampled paths from 256 px leave 0 to 2 %
//!   of the drawn pixels off by more than 1; see `src/text/mod.rs`);
//! - the documented residual cases: text below 19 px (CoreGraphics
//!   grid-fits it vertically), characters only fallback fonts draw (the
//!   reference uses PingFang SC, whose `hvgl` outlines cannot be read), and
//!   `system-ui` (SF, outlined by skrifa with rounded variation deltas):
//!   alpha SSIM >= 0.75 and footprint >= 0.7.

use noisemaker_host::text::{HostStyle, TextColor, TextFonts, TextParams, draw_text_canvas};
use std::path::Path;

fn read_png(path: &Path) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

fn global_ssim(a: &[u8], b: &[u8]) -> f64 {
    let luma = |p: &[u8]| -> Vec<f64> {
        p.as_chunks::<4>()
            .0
            .iter()
            .map(|c| (0.299 * c[0] as f64 + 0.587 * c[1] as f64 + 0.114 * c[2] as f64) / 255.0)
            .collect()
    };
    let (la, lb) = (luma(a), luma(b));
    let n = la.len() as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / n;
    let (ma, mb) = (mean(&la), mean(&lb));
    let var = |v: &[f64], m: f64| v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n;
    let (va, vb) = (var(&la, ma), var(&lb, mb));
    let cov = la
        .iter()
        .zip(&lb)
        .map(|(x, y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / n;
    let (c1, c2) = (0.01f64.powi(2), 0.03f64.powi(2));
    let den = (ma * ma + mb * mb + c1) * (va + vb + c2);
    if den == 0.0 {
        1.0
    } else {
        (2.0 * ma * mb + c1) * (2.0 * cov + c2) / den
    }
}

/// SSIM of the alpha channel (the glyph coverage), which carries the text
/// even when the colour is uniform.
fn alpha_ssim(a: &[u8], b: &[u8]) -> f64 {
    let alpha = |p: &[u8]| -> Vec<u8> {
        p.as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| [c[3], c[3], c[3], 255])
            .collect()
    };
    global_ssim(&alpha(a), &alpha(b))
}

/// Compares the port's canvas for `params` with a captured upload and
/// returns a report line, or a failure when the case misses its bounds.
fn compare_case(
    name: &str,
    golden: &[u8],
    w: u32,
    h: u32,
    params: &TextParams,
    fonts: &TextFonts,
) -> Result<String, String> {
    let canvas = draw_text_canvas(fonts, &HostStyle::demo(), params, w, h);
    let ours = canvas.upload_image();
    if let Ok(out) = std::env::var("NM_HOST_TEXT_OUT") {
        std::fs::create_dir_all(&out).unwrap();
        std::fs::write(Path::new(&out).join(format!("{name}.rgba")), &ours.data).unwrap();
    }
    let mut differing = 0usize;
    let mut alpha_beyond_one = 0usize;
    let mut max_abs = 0u8;
    let mut alpha_abs = 0u64;
    let (mut ink_g, mut ink_o) = (0u64, 0u64);
    let (mut both, mut either, mut drawn) = (0usize, 0usize, 0usize);
    for (g, o) in golden
        .as_chunks::<4>()
        .0
        .iter()
        .zip(ours.data.as_chunks::<4>().0)
    {
        let d = g.iter().zip(o).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        differing += usize::from(d > 0);
        max_abs = max_abs.max(d);
        alpha_abs += g[3].abs_diff(o[3]) as u64;
        alpha_beyond_one += usize::from(g[3].abs_diff(o[3]) > 1);
        ink_g += g[3] as u64;
        ink_o += o[3] as u64;
        both += usize::from(g[3] > 0 && o[3] > 0);
        either += usize::from(g[3] > 0 || o[3] > 0);
        drawn += usize::from(g[3] > 0);
    }
    let ssim = global_ssim(golden, &ours.data);
    let assim = alpha_ssim(golden, &ours.data);
    let ink_ratio = if ink_g > 0 {
        ink_o as f64 / ink_g as f64
    } else if ink_o == 0 {
        1.0
    } else {
        f64::INFINITY
    };
    let overlap = if either > 0 {
        both as f64 / either as f64
    } else {
        1.0
    };
    let mean_alpha = alpha_abs as f64 / (w as f64 * h as f64);
    let line = format!(
        "{name:24} {w}x{h} differ {differing:5} px  |dA|>1 {alpha_beyond_one:4} of {drawn:5} drawn  max {max_abs:3}  mean|dA| {mean_alpha:.4}  SSIM {ssim:.5}  alphaSSIM {assim:.5}  ink {ink_ratio:.3}  footprint {overlap:.3}"
    );
    // the documented residual cases (see the file header)
    let font_px = (params.size * h as f64).round();
    let residual = font_px < 19.0
        || params.font.trim().is_empty()
        || params.font.eq_ignore_ascii_case("system-ui")
        || params.text.chars().any(|c| c as u32 > 0x24F);
    let (min_ssim, min_footprint) = if residual {
        (0.75, 0.7)
    } else {
        (0.9999, 0.99)
    };
    if assim < min_ssim || overlap < min_footprint {
        return Err(format!(
            "{line}\n  -> alpha SSIM {assim:.5} (min {min_ssim}), footprint {overlap:.3} (min {min_footprint})"
        ));
    }
    Ok(line)
}

fn color_of(value: &serde_json::Value) -> TextColor {
    match value {
        serde_json::Value::Array(values) => TextColor::Array(
            values
                .iter()
                .map(|v| v.as_f64().unwrap_or(f64::NAN))
                .collect(),
        ),
        serde_json::Value::String(s) => TextColor::Hex(s.clone()),
        _ => TextColor::Hex(String::new()),
    }
}

/// The demo's text state from a JSON object holding the filter/text globals
/// (a capture manifest case or a compiled pass's uniforms).
fn params_of(values: &serde_json::Value) -> TextParams {
    let num = |k: &str| values[k].as_f64().unwrap_or(f64::NAN);
    TextParams {
        text: values["text"].as_str().unwrap_or("").to_owned(),
        font: values["font"].as_str().unwrap_or("").to_owned(),
        size: num("size"),
        pos_x: num("posX"),
        pos_y: num("posY"),
        rotation: num("rotation"),
        color: color_of(&values["color"]),
        justify: values["justify"].as_str().unwrap_or("").to_owned(),
    }
}

#[test]
fn text_canvases_match_the_reference_uploads() {
    let Some(dir) = std::env::var_os("NM_HOST_TEXT_GOLDENS") else {
        eprintln!("text_raster: NM_HOST_TEXT_GOLDENS is not set; skipping the text comparison");
        return;
    };
    let dir = Path::new(&dir);
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    let mut fonts = TextFonts::default();
    // the capture registers the bundled TTF under this name as well
    fonts.register(
        "NunitoBundledTTF",
        noisemaker_effects::share_file("share/fonts/Nunito/Nunito-VariableFont_wght.ttf")
            .unwrap()
            .to_vec(),
    );
    let only = std::env::var("NM_HOST_TEXT_CASE").ok();
    let (mut cases, mut failures) = (0usize, Vec::new());
    for case in manifest["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        let (w, h, golden) = read_png(&dir.join(format!("{name}.png")));
        cases += 1;
        match compare_case(name, &golden, w, h, &params_of(case), &fonts) {
            Ok(line) => eprintln!("text_raster: {line}"),
            Err(failure) => {
                eprintln!("text_raster: {failure}");
                failures.push(failure);
            }
        }
    }
    eprintln!(
        "text_raster: {cases} cases, {} outside their bounds",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn minted_text_textures_match() {
    let Some(dirs) = std::env::var_os("NM_HOST_GOLDENS") else {
        eprintln!("text_raster: NM_HOST_GOLDENS is not set; skipping the minted text textures");
        return;
    };
    let fonts = TextFonts::default();
    let (mut cases, mut failures) = (0usize, Vec::new());
    for dir in std::env::split_paths(&dirs) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.contains(".textTex_step_") && n.ends_with(".png"))
            .collect();
        names.sort();
        for file in names {
            let (program, texture) = file.trim_end_matches(".png").split_once('.').unwrap();
            let Ok(graph) = std::fs::read_to_string(dir.join(format!("{program}.graph.json")))
            else {
                continue;
            };
            let graph: serde_json::Value = serde_json::from_str(&graph).unwrap();
            let Some(pass) = graph["passes"].as_array().and_then(|passes| {
                passes.iter().find(|p| {
                    p["effectKey"] == "filter.text"
                        && p["inputs"]
                            .as_object()
                            .is_some_and(|i| i.values().any(|v| v == texture))
                })
            }) else {
                continue;
            };
            let (w, h, golden) = read_png(&dir.join(&file));
            if golden.as_chunks::<4>().0.iter().all(|p| p[3] == 0) {
                // a capture of the minter before its external-texture fix
                eprintln!("text_raster: {file}: empty capture, skipped");
                continue;
            }
            cases += 1;
            match compare_case(&file, &golden, w, h, &params_of(&pass["uniforms"]), &fonts) {
                Ok(line) => eprintln!("text_raster: {line}"),
                Err(failure) => {
                    eprintln!("text_raster: {failure}");
                    failures.push(failure);
                }
            }
        }
    }
    eprintln!(
        "text_raster: {cases} minted text textures, {} outside their bounds",
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
