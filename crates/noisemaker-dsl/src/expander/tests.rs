//! Tests of the expander port.
//!
//! Besides unit tests, the differential tests run the reference engine's own
//! `expand` and `compileGraph` on synthetic effect definitions and validated
//! plans (`differential/cases.mjs`, and seeded random ones from
//! `differential/fuzz.mjs`) and require this port to produce the same expansion,
//! graph and thrown errors, member order included. They reach the paths the
//! effect catalog and the DSL corpus cannot. They run when `NM_REFERENCE_ROOT`
//! points at the reference checkout and `node` is on PATH, and are skipped
//! otherwise.

use std::io::Write as _;
use std::process::{Command, Stdio};
use std::rc::Rc;

use super::*;
use crate::compiler::{CompileOptions, compile_graph_from_validated};
use crate::registry::EffectEntry;

const REFERENCE_HARNESS: &str = include_str!("differential/reference.mjs");
const SYNTHETIC_CASES: &str = include_str!("differential/cases.mjs");
const FUZZ_CASES: &str = include_str!("differential/fuzz.mjs");

/// The marker the harness scripts use for an `undefined` member or element.
const UNDEF: &str = "\u{0}undefined";

#[test]
fn patterns() {
    assert!(is_surface_ref("o0"));
    assert!(is_surface_ref("vol7"));
    assert!(is_surface_ref("rgba3"));
    assert!(!is_surface_ref("o8"));
    assert!(!is_surface_ref("o10"));
    assert!(!is_surface_ref("global_o0"));
    assert!(!is_surface_ref("mesh0"));
    assert!(is_particle_global("global_points_trail"));
    assert!(!is_particle_global("global_xyz_node_1"));
    assert_eq!(resolve_global_surface_ref("geo3"), "global_geo3");
    assert_eq!(resolve_global_surface_ref("none"), "none");
    assert_eq!(resolve_global_surface_ref("global_a"), "global_a");
    assert_eq!(resolve_global_surface_ref("myTex"), "myTex");
}

#[test]
fn resolve_enum_walks_standard_enums_only() {
    let registry = Registry::with_catalog();
    let std_enums = Value::Object(registry.std_enums());
    assert_eq!(
        resolve_enum(&std_enums, "oscKind.saw"),
        Some(Value::Number(2.0))
    );
    assert_eq!(
        resolve_enum(&std_enums, "channel.r"),
        Some(Value::Number(0.0))
    );
    // Intermediate values must be truthy: `channel.r.value` stops at 0.
    assert_eq!(resolve_enum(&std_enums, "channel.r.value"), None);
    assert_eq!(resolve_enum(&std_enums, "palette"), None);
    assert_eq!(resolve_enum(&std_enums, "synth.noise.type.simplex"), None);
    assert!(resolve_enum(&std_enums, "palette.brushedMetal").is_some());
}

#[test]
fn locale_compare_follows_icu_root_for_ascii() {
    use std::cmp::Ordering::*;
    assert_eq!(locale_compare("BLEND_MODE", "BLUR_LAYER"), Less);
    assert_eq!(locale_compare("a", "A"), Less);
    assert_eq!(locale_compare("aB", "Ab"), Less);
    assert_eq!(locale_compare("a_b", "A_a"), Greater);
    assert_eq!(locale_compare("A_B", "AB"), Less);
    assert_eq!(locale_compare("a\u{1}b", "ab"), Equal);
    assert_eq!(locale_compare("Z9", "a"), Greater);
    assert_eq!(locale_compare("9", "_"), Greater);
}

#[test]
fn expand_reports_missing_render_surface() {
    let registry = Registry::new();
    let input = crate::js!({"plans": [], "diagnostics": [], "render": null});
    let expansion = expand(&input, &registry, &ExpandOptions::default()).unwrap();
    assert_eq!(
        Value::Array(expansion.errors).to_json().unwrap(),
        r#"[{"message":"No render surface specified and no write() found - add render(oN) or write(oN)"}]"#
    );
    assert_eq!(expansion.render_surface, Value::Null);
    let branch = crate::js!({"plans": [{"type": "Branch"}], "diagnostics": [], "render": "o0"});
    assert_eq!(
        expand(&branch, &registry, &ExpandOptions::default()),
        Err(JsError::type_error("plan.chain is not iterable"))
    );
}

// --- Differential tests against the reference engine ------------------------------

fn reference_root(test: &str) -> Option<String> {
    match std::env::var("NM_REFERENCE_ROOT") {
        Ok(root) => Some(root),
        Err(_) => {
            eprintln!("skipping {test}: NM_REFERENCE_ROOT is not set");
            None
        }
    }
}

/// Run an ES module script with node and return its standard output.
fn run_node(script: &str, stdin: Option<&str>, env: &[(&str, String)]) -> String {
    let mut cmd = Command::new("node");
    cmd.args(["--input-type=module", "-e", script])
        .stdin(if stdin.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in env {
        cmd.env(key, value);
    }
    let mut child = cmd.spawn().expect("node is on PATH");
    if let Some(input) = stdin {
        child
            .stdin
            .take()
            .expect("piped stdin")
            .write_all(input.as_bytes())
            .expect("node reads the spec");
    }
    let output = child.wait_with_output().expect("node exits");
    assert!(
        output.status.success(),
        "node failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("node prints UTF-8")
}

/// Replace the harness's `undefined` markers with `undefined`.
fn revive(value: Value) -> Value {
    match value {
        Value::String(s) if s == UNDEF => Value::Undefined,
        Value::Array(a) => Value::Array(a.into_iter().map(revive).collect()),
        Value::Object(o) => Value::Object(o.into_iter().map(|(k, v)| (k, revive(v))).collect()),
        other => other,
    }
}

/// A value as `JSON.parse(JSON.stringify(value))` leaves it (the reference
/// harness's `plain`).
fn plain(value: &Value) -> Value {
    Value::from_json(&value.to_json().unwrap_or_else(|| "null".into()))
        .expect("serialized values parse")
}

/// This port's records for the spec's cases, in the reference harness's format:
/// `{name, expanded|expandedError, graph|graphError}`.
fn run_cases(spec: &Value) -> Vec<Value> {
    let mut registry = Registry::with_catalog();
    for effect in spec.get("effects").as_array().expect("effects") {
        let keys: Vec<String> = effect
            .get("keys")
            .as_array()
            .expect("keys")
            .iter()
            .map(|k| k.as_str().expect("string key").to_owned())
            .collect();
        let entry = Rc::new(EffectEntry {
            namespace: "test".into(),
            name: keys[0].clone(),
            def: effect.get("def").clone(),
        });
        for key in keys {
            registry.register_effect(key, entry.clone());
        }
    }
    let mut records = Vec::new();
    for case in spec.get("cases").as_array().expect("cases") {
        let mut record = Object::new();
        record.insert("name", case.get("name").clone());
        let input = case.get("input");
        let shader_overrides = case
            .get("options")
            .get("shaderOverrides")
            .as_object()
            .cloned()
            .unwrap_or_default();
        match expand(
            input,
            &registry,
            &ExpandOptions {
                shader_overrides: shader_overrides.clone(),
            },
        ) {
            Ok(expansion) => record.insert("expanded", plain(&expansion.to_value())),
            Err(e) => record.insert("expandedError", plain(&e.to_value())),
        };
        let source = case.get("source").as_str().unwrap_or_default();
        match compile_graph_from_validated(
            source,
            input,
            &registry,
            &CompileOptions { shader_overrides },
        ) {
            Ok(graph) => record.insert("graph", plain(&graph)),
            Err(e) => record.insert("graphError", plain(&e.to_value())),
        };
        records.push(Value::Object(record));
    }
    records
}

/// The first difference between `a` (reference) and `b` (port), member order
/// included.
fn first_difference(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let xk: Vec<&String> = x.keys().collect();
            let yk: Vec<&String> = y.keys().collect();
            if xk != yk {
                return Some(format!(
                    "{path}: members {xk:?} (reference) vs {yk:?} (port)"
                ));
            }
            x.iter().find_map(|(k, v)| {
                first_difference(v, y.get_or_undefined(k), &format!("{path}.{k}"))
            })
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!(
                    "{path}: length {} (reference) vs {} (port)",
                    x.len(),
                    y.len()
                ));
            }
            x.iter()
                .zip(y)
                .enumerate()
                .find_map(|(i, (v, w))| first_difference(v, w, &format!("{path}[{i}]")))
        }
        _ if a == b => None,
        _ => Some(format!("{path}: {a:?} (reference) vs {b:?} (port)")),
    }
}

fn check_against_reference(spec_json: &str, what: &str, root: &str) {
    let reference = run_node(
        REFERENCE_HARNESS,
        Some(spec_json),
        &[("NM_REFERENCE_ROOT", root.to_owned())],
    );
    let theirs: Vec<Value> = reference
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Value::from_json(l).expect("reference record"))
        .collect();
    let spec = revive(Value::from_json(spec_json).expect("spec JSON"));
    let ours = run_cases(&spec);
    assert_eq!(theirs.len(), ours.len(), "{what}: record count");
    let failures: Vec<String> = theirs
        .iter()
        .zip(&ours)
        .filter_map(|(r, o)| {
            first_difference(r, o, "$").map(|d| format!("{}: {d}", js_string(r.get("name"))))
        })
        .collect();
    assert!(
        failures.is_empty(),
        "{what}: {} of {} cases differ from the reference:\n{}",
        failures.len(),
        ours.len(),
        failures.join("\n")
    );
}

#[test]
fn differential_synthetic_cases() {
    let Some(root) = reference_root("differential_synthetic_cases") else {
        return;
    };
    let spec = run_node(SYNTHETIC_CASES, None, &[]);
    check_against_reference(&spec, "synthetic cases", &root);
}

#[test]
fn differential_random_cases() {
    let Some(root) = reference_root("differential_random_cases") else {
        return;
    };
    for seed in 1..=4 {
        let spec = run_node(
            FUZZ_CASES,
            None,
            &[
                ("NM_FUZZ_SEED", seed.to_string()),
                ("NM_FUZZ_COUNT", "250".into()),
            ],
        );
        check_against_reference(&spec, &format!("random cases (seed {seed})"), &root);
    }
}

/// `locale_compare` against V8's `localeCompare` on random ASCII strings.
#[test]
fn locale_compare_matches_reference() {
    if reference_root("locale_compare_matches_reference").is_none() {
        return;
    }
    let script = r#"
let s = 7
const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 }
const chars = '\t\n _-,;:!?.\'"()[]{}@*/\\&#%`^+<=>|~$0189aAbBzZ\u0001\u007f'
const word = () => { let w = ''; for (let i = 0, n = 1 + Math.floor(rnd() * 5); i < n; i++) w += chars[Math.floor(rnd() * chars.length)]; return w }
const out = []
for (let i = 0; i < 3000; i++) { const a = word(), b = rnd() < 0.2 ? a.toUpperCase() : word(); out.push([a, b, Math.sign(a.localeCompare(b))]) }
process.stdout.write(JSON.stringify(out))
"#;
    let pairs = Value::from_json(&run_node(script, None, &[])).expect("pairs");
    let mut failures = Vec::new();
    for pair in pairs.as_array().expect("array") {
        let a = pair.at(0).as_str().expect("a");
        let b = pair.at(1).as_str().expect("b");
        let want = pair.at(2).as_f64().expect("sign") as i32;
        let got = locale_compare(a, b) as i32;
        if got != want {
            failures.push(format!("{a:?} vs {b:?}: reference {want}, port {got}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
