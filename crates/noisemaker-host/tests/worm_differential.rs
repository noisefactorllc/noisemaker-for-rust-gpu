//! Worm-tracer differential test: the port's asyncInit drivers and tracer
//! against the reference's, recorded by `tools/reference-host.mjs worm` on a
//! mock 2D context. The sequence of canvas operations must be identical:
//! same styles, same coordinates to the bit, same upload points.
//!
//! The recorder evaluates Math.sin/cos/log correctly rounded, as Chromium's
//! V8 does (`reference-host.mjs math-check` verifies that against the
//! browser); Node's own V8 returns different values for a few percent of
//! arguments.
//!
//! Needs NM_REFERENCE_ROOT (a reference checkout) and node; without them
//! the test explains why and passes.

mod common;

use noisemaker_host::canvas::CallRecorder;
use noisemaker_host::overlay::{JsValue, OverlayEffect, OverlayParams, run_async_init};

struct Case {
    effect: OverlayEffect,
    width: u32,
    height: u32,
    params_json: &'static str,
    params: OverlayParams,
    cancel_after: Option<usize>,
}

fn case(
    effect: &str,
    width: u32,
    height: u32,
    params_json: &'static str,
    params: OverlayParams,
) -> Case {
    Case {
        effect: OverlayEffect::from_name(effect).unwrap(),
        width,
        height,
        params_json,
        params,
        cancel_after: None,
    }
}

fn num(seed: f64, density: f64) -> OverlayParams {
    OverlayParams::new(seed, density)
}

fn cases() -> Vec<Case> {
    let undefined = OverlayParams::default;
    let mut cases = vec![
        // initAsyncEffects runs with the pipeline's global uniforms ({}).
        case("fibers", 256, 256, "{}", undefined()),
        case("fibers", 512, 512, "{}", undefined()),
        case("scratches", 256, 256, "{}", undefined()),
        case("scratches", 512, 512, "{}", undefined()),
        case("strayHair", 256, 256, "{}", undefined()),
        case("strayHair", 512, 512, "{}", undefined()),
        // step values (checkAsyncRegen)
        case(
            "fibers",
            256,
            256,
            r#"{"seed":1,"density":1}"#,
            num(1.0, 1.0),
        ),
        case(
            "fibers",
            512,
            512,
            r#"{"seed":1,"density":1}"#,
            num(1.0, 1.0),
        ),
        case(
            "fibers",
            256,
            256,
            r#"{"seed":7,"density":0.25}"#,
            num(7.0, 0.25),
        ),
        case(
            "fibers",
            512,
            512,
            r#"{"seed":42,"density":0.6}"#,
            num(42.0, 0.6),
        ),
        case(
            "scratches",
            256,
            256,
            r#"{"seed":2,"density":0.8}"#,
            num(2.0, 0.8),
        ),
        case(
            "scratches",
            512,
            512,
            r#"{"seed":2,"density":0.8}"#,
            num(2.0, 0.8),
        ),
        case(
            "scratches",
            256,
            256,
            r#"{"seed":4,"density":0.3}"#,
            num(4.0, 0.3),
        ),
        case(
            "strayHair",
            256,
            256,
            r#"{"seed":3,"density":0.5}"#,
            num(3.0, 0.5),
        ),
        case(
            "strayHair",
            512,
            512,
            r#"{"seed":9,"density":1}"#,
            num(9.0, 1.0),
        ),
        case(
            "strayHair",
            512,
            512,
            r#"{"seed":100,"density":0}"#,
            num(100.0, 0.0),
        ),
        // non-square canvases
        case(
            "fibers",
            384,
            216,
            r#"{"seed":5,"density":0.5}"#,
            num(5.0, 0.5),
        ),
        case(
            "strayHair",
            300,
            200,
            r#"{"seed":2.5}"#,
            OverlayParams {
                seed: JsValue::Number(2.5),
                density: JsValue::Undefined,
            },
        ),
        // JavaScript value semantics of `params.seed || 1` and
        // `params.density !== undefined ? params.density : default`
        case(
            "fibers",
            256,
            256,
            r#"{"seed":0,"density":null}"#,
            OverlayParams {
                seed: JsValue::Number(0.0),
                density: JsValue::Null,
            },
        ),
        case(
            "scratches",
            256,
            256,
            r#"{"seed":"3","density":"0.75"}"#,
            OverlayParams {
                seed: JsValue::String("3".into()),
                density: JsValue::String("0.75".into()),
            },
        ),
        case(
            "scratches",
            256,
            256,
            r#"{"seed":-3,"density":2}"#,
            num(-3.0, 2.0),
        ),
        case(
            "strayHair",
            256,
            256,
            r#"{"seed":"","density":true}"#,
            OverlayParams {
                seed: JsValue::String(String::new()),
                density: JsValue::Bool(true),
            },
        ),
    ];
    // cancellation: isCancelled() true from its 4th call (inside layer 0)
    let mut cancelled = case(
        "fibers",
        256,
        256,
        r#"{"seed":1,"density":1}"#,
        num(1.0, 1.0),
    );
    cancelled.cancel_after = Some(3);
    cases.push(cancelled);
    let mut cancelled = case("scratches", 256, 256, "{}", undefined());
    cancelled.cancel_after = Some(40);
    cases.push(cancelled);
    cases
}

fn port_log(case: &Case) -> Vec<String> {
    let mut recorder = CallRecorder::new(case.width, case.height);
    let mut polls = 0usize;
    let limit = case.cancel_after.unwrap_or(usize::MAX);
    run_async_init(
        case.effect,
        &mut recorder,
        &case.params,
        &mut || {
            let cancelled = polls >= limit;
            polls += 1;
            cancelled
        },
        &mut |name, r: &mut CallRecorder| r.note(format!("update {name}")),
    );
    recorder.lines().to_vec()
}

#[test]
fn traced_canvas_calls_match_the_reference_exactly() {
    let Some(reference) = common::reference_env("worm_differential") else {
        return;
    };
    let mut failures = Vec::new();
    let mut total_lines = 0usize;
    let cases = cases();
    for case in &cases {
        let (w, h) = (case.width.to_string(), case.height.to_string());
        let cancel = case.cancel_after.map(|n| n.to_string());
        let mut args = vec![
            "worm",
            "--effect",
            case.effect.func(),
            "--width",
            &w,
            "--height",
            &h,
            "--params",
            case.params_json,
        ];
        if let Some(n) = &cancel {
            args.extend(["--cancel-after", n]);
        }
        let out = common::reference_host(&reference, &args);
        let reference_log: Vec<&str> = std::str::from_utf8(&out).unwrap().lines().collect();
        let ours = port_log(case);
        total_lines += reference_log.len();
        let label = format!(
            "{} {}x{} {}{}",
            case.effect.func(),
            case.width,
            case.height,
            case.params_json,
            cancel
                .map(|n| format!(" cancel-after {n}"))
                .unwrap_or_default()
        );
        let strokes = reference_log.iter().filter(|l| **l == "stroke").count();
        match ours
            .iter()
            .map(String::as_str)
            .zip(reference_log.iter().copied())
            .position(|(a, b)| a != b)
        {
            Some(i) => failures.push(format!(
                "{label}: first difference at line {i}: ours {:?} reference {:?}",
                ours[i], reference_log[i]
            )),
            None if ours.len() != reference_log.len() => failures.push(format!(
                "{label}: {} lines vs reference {}",
                ours.len(),
                reference_log.len()
            )),
            None => eprintln!(
                "worm_differential: {label}: {} calls, {strokes} strokes identical",
                ours.len()
            ),
        }
    }
    eprintln!(
        "worm_differential: {} cases, {total_lines} reference calls, {} mismatching cases",
        cases.len(),
        failures.len()
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
