//! Validator paths that no effect of the catalog reaches, checked against the
//! reference.
//!
//! `reference_cases.json` holds, for each program, the reference's AST and its
//! `validate(ast)` result (or thrown error), computed with the catalog registries
//! plus the registrations of [`registry`]: an op whose `r`, `g`, `b` parameters
//! take a packed color and whose `level` defaults from another parameter
//! (`defaultFrom`), an op with a surface, a member and a vec4/vec3 parameter
//! without defaults and a string parameter outside the allowlist, a deprecated
//! effect (S008) with a parameter alias, and the validator hook of [`hooks`].

use std::rc::Rc;

use crate::registry::Registry;
use crate::value::{Object, Value};

use super::{HookContext, HookResult, ValidatorHook, ValidatorHooks, validate_with_hooks};

const CASES: &str = include_str!("reference_cases.json");

/// The registrations the reference cases were computed with, on top of the
/// catalog.
fn registry() -> Registry {
    let mut reg = Registry::with_catalog();
    let ops = Value::from_json(
        r#"{
        "synth.rgbStarter": {"name": "rgbStarter", "args": [
            {"name": "r", "type": "float", "default": 0.1, "min": 0, "max": 1},
            {"name": "g", "type": "float", "default": 0.2, "min": 0, "max": 1},
            {"name": "b", "type": "float", "default": 0.3, "min": 0, "max": 1},
            {"name": "level", "type": "float", "default": 0.5, "defaultFrom": "g"}]},
        "filter.needsTex": {"name": "needsTex", "args": [
            {"name": "tex", "type": "surface"},
            {"name": "mode", "type": "member", "enum": "filter.needsTex.mode"},
            {"name": "label", "type": "string", "default": "x"},
            {"name": "tint", "type": "vec4", "default": [1, 0.5, 0.25, 1]},
            {"name": "tint3", "type": "vec4"},
            {"name": "offset", "type": "vec3"},
            {"name": "amount", "type": "float", "default": 2, "defaultFrom": "level"}]},
        "synth.oldThing": {"name": "oldThing", "args": [{"name": "size", "type": "float", "default": 1}]}
    }"#,
    )
    .expect("valid JSON");
    for (name, spec) in ops.as_object().expect("object").iter() {
        reg.register_op(name.clone(), spec.clone());
    }
    reg.register_effect_alias("synth.oldThing", "newThing");
    let mut aliases = Object::new();
    aliases.insert("sz", Value::from("size"));
    reg.register_param_aliases("synth.oldThing", &aliases);
    reg.register_starter_ops(&[
        "synth.rgbStarter",
        "rgbStarter",
        "synth.oldThing",
        "oldThing",
    ]);
    let modes = Value::from_json(
        r#"{"filter": {"needsTex": {"mode": {"soft": {"type": "Number", "value": 1}, "hard": {"type": "Number", "value": 2}}}}}"#,
    )
    .expect("valid JSON");
    reg.merge_into_enums(modes.as_object().expect("object"));
    reg
}

/// The reference cases' hook on `rgbStarter`: a call whose resolved `level` is
/// above 0.5 becomes a `custom.rgb` step (with a state and a diagnostic); any
/// other call keeps its step and gains `args.hookSeen`.
fn hooks() -> ValidatorHooks {
    let hook: Rc<ValidatorHook> = Rc::new(|ctx: &mut HookContext<'_, '_>| {
        if let Some(level) = ctx.args.get("level").as_f64()
            && level > 0.5
        {
            let t = ctx.allocate_temp();
            let mut step = Object::new();
            step.insert("op", Value::from("custom.rgb"));
            step.insert("args", ctx.args.clone());
            step.insert("from", ctx.from.clone());
            step.insert("temp", Value::Number(t));
            step.insert("hooked", Value::Bool(true));
            let starter_index = if ctx.starter.is_truthy() {
                ctx.starter.get("index").clone()
            } else {
                Value::Number(-1.0)
            };
            step.insert("starter", starter_index);
            ctx.add_step(&Value::Object(step));
            let mut state = Object::new();
            state.insert("kind", Value::from("rgbState"));
            state.insert("temp", Value::Number(t));
            state.insert("write", ctx.write_name.clone());
            ctx.add_state(&Value::Object(state));
            let original = ctx.original_call.clone();
            ctx.push_diagnostic("S002", &original, Some("hooked rgbStarter"))?;
            return Ok(Some(HookResult {
                handled: true,
                current: Value::Number(t),
            }));
        }
        ctx.args.set("hookSeen", Value::Bool(true));
        Ok(None)
    });
    let mut hooks = ValidatorHooks::new();
    hooks.register("rgbStarter", hook);
    hooks
}

/// The first difference between `a` (reference) and `b`, comparing values and
/// member order.
fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let kx: Vec<_> = x.keys().collect();
            let ky: Vec<_> = y.keys().collect();
            if kx != ky {
                return Some(format!("{path}: keys {kx:?} (reference) vs {ky:?}"));
            }
            x.iter()
                .find_map(|(k, v)| diff(v, y.get(k).expect("same keys"), &format!("{path}.{k}")))
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!(
                    "{path}: length {} (reference) vs {}",
                    x.len(),
                    y.len()
                ));
            }
            x.iter()
                .zip(y)
                .enumerate()
                .find_map(|(i, (p, q))| diff(p, q, &format!("{path}[{i}]")))
        }
        _ if a == b => None,
        _ => Some(format!("{path}: {a:?} (reference) vs {b:?}")),
    }
}

#[test]
fn custom_registry_cases_match_the_reference() {
    let reg = registry();
    let hooks = hooks();
    let cases = Value::from_json(CASES).expect("valid JSON");
    let mut checked = 0;
    for (name, case) in cases.as_object().expect("object").iter() {
        let actual = validate_with_hooks(case.get("ast"), &reg, &hooks);
        match (actual, case.get("result")) {
            (Ok(result), expected) if !expected.is_undefined() => {
                // Serialize as JSON.stringify does, then compare.
                let result =
                    Value::from_json(&result.to_json().expect("serializable")).expect("valid JSON");
                if let Some(d) = diff(expected, &result, "$") {
                    panic!("{name}: {d}");
                }
            }
            (Err(e), _) => {
                let expected = case.get("error");
                assert_eq!(&e.to_value(), expected, "{name}");
            }
            (Ok(result), _) => panic!(
                "{name}: expected the error {:?}, got {result:?}",
                case.get("error")
            ),
        }
        checked += 1;
    }
    assert_eq!(checked, 9);
}
