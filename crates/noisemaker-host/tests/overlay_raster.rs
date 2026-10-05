//! Overlay raster test: the port's overlays against the textures the
//! reference uploaded, as the golden minter (`parity/batch-golden.mjs`)
//! reads them back: `<name>.node_<N>_overlayTex.png` next to
//! `<name>.graph.json`, whose pass for node_<N> names the effect and holds
//! the step values (`seed`, `density`) the asyncInit traced with.
//!
//! `NM_HOST_GOLDENS` lists the minter output directories to check
//! (separated by `:`); without it the test explains why and passes. Reports
//! per fixture the pixels that differ, max-abs-diff and SSIM as
//! `parity/compare.py` computes them, and asserts the measured bound of the
//! canvas model (see the residual note in `src/canvas.rs`): the same drawn
//! footprint, at most 0.1 % of drawn pixels different, SSIM >= 0.999.

use noisemaker_host::overlay::{JsValue, OverlayEffect, OverlayParams, render_async_overlay};
use std::path::{Path, PathBuf};

fn read_png(path: &Path) -> (u32, u32, Vec<u8>) {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgba, "{}", path.display());
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "{}", path.display());
    buf.truncate(info.buffer_size());
    (info.width, info.height, buf)
}

/// Global SSIM over Rec. 601 luma, as parity/compare.py computes it.
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

fn json_value(v: Option<&serde_json::Value>) -> JsValue {
    match v {
        None => JsValue::Undefined,
        Some(serde_json::Value::Null) => JsValue::Null,
        Some(serde_json::Value::Bool(b)) => JsValue::Bool(*b),
        Some(serde_json::Value::Number(n)) => JsValue::Number(n.as_f64().unwrap()),
        Some(serde_json::Value::String(s)) => JsValue::String(s.clone()),
        Some(_) => JsValue::Object,
    }
}

/// (effect, params) of the overlay `node_id` in a minted graph.
fn overlay_params(graph: &Path, node_id: &str) -> Option<(OverlayEffect, OverlayParams)> {
    let graph: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(graph).ok()?).ok()?;
    let pass = graph["passes"]
        .as_array()?
        .iter()
        .find(|p| p["nodeId"] == node_id && p["effectKey"].is_string())?;
    let effect = OverlayEffect::from_name(pass["effectKey"].as_str()?)?;
    let uniforms = &pass["uniforms"];
    Some((
        effect,
        OverlayParams {
            seed: json_value(uniforms.get("seed")),
            density: json_value(uniforms.get("density")),
        },
    ))
}

#[test]
fn overlays_match_the_reference_uploads() {
    let Some(dirs) = std::env::var_os("NM_HOST_GOLDENS") else {
        eprintln!("overlay_raster: NM_HOST_GOLDENS is not set; skipping the raster comparison");
        return;
    };
    let mut fixtures: Vec<(PathBuf, PathBuf, String)> = Vec::new();
    for dir in std::env::split_paths(&dirs) {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with("_overlayTex.png"))
            .collect();
        names.sort();
        for name in names {
            let stem = name.trim_end_matches(".png");
            let Some((program, texture)) = stem.split_once('.') else {
                continue;
            };
            let node_id = texture.trim_end_matches("_overlayTex").to_owned();
            fixtures.push((
                dir.join(&name),
                dir.join(format!("{program}.graph.json")),
                node_id,
            ));
        }
    }
    if fixtures.is_empty() {
        // NM_HOST_GOLDENS also serves the minted text textures
        eprintln!("overlay_raster: no *_overlayTex.png under NM_HOST_GOLDENS; nothing to compare");
        return;
    }

    let mut failures = Vec::new();
    let (mut total_drawn, mut total_diff, mut exact) = (0usize, 0usize, 0usize);
    for (png_path, graph_path, node_id) in &fixtures {
        let Some((effect, params)) = overlay_params(graph_path, node_id) else {
            failures.push(format!(
                "{}: no overlay pass {node_id} in its graph",
                png_path.display()
            ));
            continue;
        };
        let (w, h, golden) = read_png(png_path);
        let ours = render_async_overlay(effect, w, h, &params);
        let mut differing = 0usize;
        let mut max_abs = 0u8;
        let mut footprint_mismatch = 0usize;
        let mut drawn = 0usize;
        for (g, o) in golden
            .as_chunks::<4>()
            .0
            .iter()
            .zip(ours.data.as_chunks::<4>().0)
        {
            drawn += usize::from(g[3] > 0);
            footprint_mismatch += usize::from((g[3] > 0) != (o[3] > 0));
            let d = g.iter().zip(o).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            differing += usize::from(d > 0);
            max_abs = max_abs.max(d);
        }
        let ssim = global_ssim(&golden, &ours.data);
        total_drawn += drawn;
        total_diff += differing;
        exact += usize::from(differing == 0);
        let label = format!(
            "{} {}x{} seed={:?} density={:?} ({})",
            effect.func(),
            w,
            h,
            params.seed,
            params.density,
            png_path.file_name().unwrap().to_string_lossy()
        );
        eprintln!(
            "overlay_raster: {label}: {differing} of {} px differ ({drawn} drawn), max-abs-diff {max_abs}, SSIM {ssim:.6}",
            w * h
        );
        if footprint_mismatch > 0 || differing * 1000 > drawn.max(1) || ssim < 0.999 {
            failures.push(format!(
                "{label}: footprint mismatch {footprint_mismatch}, {differing} differing of {drawn} drawn, SSIM {ssim}"
            ));
        }
    }
    eprintln!(
        "overlay_raster: {} fixtures, {exact} byte-exact, {total_diff} differing px of {total_drawn} drawn",
        fixtures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
