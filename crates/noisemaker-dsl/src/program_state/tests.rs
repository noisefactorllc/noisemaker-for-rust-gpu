//! Unit tests of the ProgramState port. The differential gate
//! (`parity/check_program_state.mjs`) compares it with the reference.

use std::cell::RefCell;
use std::rc::Rc;

use super::*;
use crate::compiler::{CompileOptions, compile_graph};

thread_local! {
    static REGISTRY: Rc<Registry> = Rc::new(Registry::with_catalog());
}

fn registry() -> Rc<Registry> {
    REGISTRY.with(Rc::clone)
}

/// A mock renderer loaded with `dsl` (the host compiled it).
fn host_for(dsl: &str, methods: MockMethods) -> MockHost {
    let reg = registry();
    let graph = compile_graph(dsl, &reg, &CompileOptions::default()).expect("graph compiles");
    MockHost {
        current_dsl: dsl.to_owned(),
        enums: Rc::new(Value::Object(reg.enums.clone())),
        convert: MockConvert::Canvas,
        pipeline: Some(MockPipeline::new(graph, methods)),
    }
}

fn events(state: &ProgramState<MockHost>, names: &[&'static str]) -> Rc<RefCell<Vec<String>>> {
    let log: Rc<RefCell<Vec<String>>> = Rc::default();
    for &name in names {
        let log = log.clone();
        state.on(
            name,
            listener(move |_, data: &Value| {
                log.borrow_mut()
                    .push(format!("{name}:{}", data.to_json().unwrap_or_default()));
                Ok(())
            }),
        );
    }
    log
}

#[test]
fn from_dsl_builds_step_states() {
    let dsl = "search synth, filter\nnoise(scaleX: 20).grade().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    let log = events(&state, &["structurechange", "load"]);
    state.from_dsl(dsl).unwrap();
    assert_eq!(state.get_step_keys(), ["step_0", "step_1", "step_2"]);
    assert_eq!(state.get_value("step_0", "scaleX"), Value::Number(20.0));
    assert_eq!(state.structure()[1].effect_key, "filter.grade");
    assert_eq!(state.structure()[2].effect_key, "_write");
    let log = log.borrow();
    assert_eq!(log.len(), 2);
    assert!(log[0].starts_with("structurechange:"));
    assert!(log[1].starts_with("load:"));
}

#[test]
fn set_value_validates_applies_and_emits() {
    let dsl = "search synth\nnoise().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    state.renderer_mut().unwrap().take_calls();
    let log = events(&state, &["change", "stepchange", "recompileNeeded"]);
    state
        .set_value("step_0", "scaleX", Value::from("33.5"))
        .unwrap();
    assert_eq!(state.get_value("step_0", "scaleX"), Value::Number(33.5));
    let passes = state
        .renderer_mut()
        .unwrap()
        .graph_passes()
        .unwrap()
        .clone();
    let pass = passes
        .iter()
        .find(|p| {
            p.get("id")
                .as_str()
                .is_some_and(|id| id.starts_with("node_0_"))
        })
        .unwrap();
    assert_eq!(pass.get("uniforms").get("scaleX"), &Value::Number(33.5));
    // `type` is a compile-time define of synth.noise: one recompileNeeded.
    state
        .batch(|s| {
            s.set_value("step_0", "type", Value::from(0.0))?;
            s.set_value("step_0", "octaves", Value::from(3.0))
        })
        .unwrap();
    let log = log.borrow();
    assert_eq!(log.len(), 3, "{log:?}");
    assert!(log[0].starts_with("change:"));
    assert!(log[1].starts_with("stepchange:"));
    assert_eq!(log[2], "recompileNeeded:");
}

#[test]
fn to_dsl_round_trip() {
    let dsl = "search synth\nnoise().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    state
        .set_value("step_0", "scaleX", Value::from(12.0))
        .unwrap();
    let out = state.to_dsl();
    assert!(out.contains("scaleX: 12"), "{out}");
    assert!(!state.would_change_structure(&out));
}

#[test]
fn search_directive_capture() {
    assert_eq!(
        search_namespaces("// x\nsearch  synth ,filter,  render \nnoise()"),
        Some(vec!["synth".into(), "filter".into(), "render ".into()])
    );
    assert_eq!(search_namespaces("research synth"), None);
    assert_eq!(
        search_namespaces("search\n\n synth"),
        Some(vec!["synth".into()])
    );
    assert_eq!(
        search_namespaces("search a,,b"),
        Some(vec!["a".into(), "".into(), "b".into()])
    );
}

#[test]
fn convert_and_resolve() {
    let reg = registry();
    let enums = Value::Object(reg.enums.clone());
    let spec = Value::from_json(r#"{"type": "int", "enum": "color"}"#).unwrap();
    assert_eq!(
        convert_parameter_for_uniform(&Value::from("rgb"), &spec, &enums).unwrap(),
        Value::Number(1.0)
    );
    let color = Value::from_json(r#"{"type": "color"}"#).unwrap();
    assert_eq!(
        convert_parameter_for_uniform(&Value::from("#ff0000"), &color, &enums).unwrap(),
        Value::from_json("[1, 0, 0]").unwrap()
    );
    assert_eq!(
        resolve_enum_value(&Value::from("color.hsv"), &enums),
        Value::Number(2.0)
    );
    assert_eq!(
        resolve_enum_value(&Value::from("nope.x"), &enums),
        Value::Null
    );
}

/// Console sink recording `level: args` lines.
#[derive(Default)]
struct Recorder(RefCell<Vec<String>>);

impl Console for Recorder {
    fn warn(&self, args: &[ConsoleArg]) {
        self.0.borrow_mut().push(format!("warn: {args:?}"));
    }
    fn error(&self, args: &[ConsoleArg]) {
        self.0.borrow_mut().push(format!("error: {args:?}"));
    }
}

#[test]
fn listener_errors_are_logged_and_do_not_stop_dispatch() {
    let console = Rc::new(Recorder::default());
    let previous = set_console(console.clone());
    let dsl = "search synth\nnoise().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    state.on(
        "change",
        listener(|_, _| Err(JsError::error("listener boom"))),
    );
    let log = events(&state, &["change"]);
    // A listener may call back into the state (re-entrancy).
    state.on(
        "change",
        listener(|s: &mut ProgramState<MockHost>, data: &Value| {
            if data.get("paramName").as_str() == Some("scaleX") {
                s.set_value("step_0", "scaleY", Value::from(5.0))?;
            }
            Ok(())
        }),
    );
    state
        .set_value("step_0", "scaleX", Value::from(9.0))
        .unwrap();
    set_console(previous);
    let log = log.borrow();
    assert_eq!(log.len(), 2, "{log:?}");
    assert!(log[0].contains("\"paramName\":\"scaleX\""));
    assert!(log[1].contains("\"paramName\":\"scaleY\""));
    let lines = console.0.borrow();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines[0].starts_with("error:") && lines[0].contains("[Emitter] Error in change handler:")
    );
}

#[test]
fn batches_flush_when_the_batch_fails() {
    let dsl = "search synth\nnoise().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    let log = events(&state, &["change", "stepchange", "recompileNeeded"]);
    let err = state
        .batch(|s| {
            s.set_value("step_0", "type", Value::from(2.0))?;
            s.set_value("step_0", "seed", Value::from("7.9"))?;
            Err(JsError::error("stop"))
        })
        .unwrap_err();
    assert_eq!(err, JsError::error("stop"));
    assert_eq!(state.batch_depth(), 0);
    assert_eq!(state.get_value("step_0", "seed"), Value::Number(7.0));
    let log = log.borrow();
    assert_eq!(log.len(), 2, "{log:?}");
    assert!(log[0].starts_with("stepchange:") && log[0].contains("\"seed\":7"));
    assert_eq!(log[1], "recompileNeeded:");
}

#[test]
fn deserialized_keys_are_strings() {
    let dsl = "search synth, filter\nnoise().write(o2).blur().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    state.set_write_target(0, "o3");
    state.set_write_step_target(1, "o4");
    state.set_media_input(0, Value::from_json(r#"{"type": "image"}"#).unwrap());
    assert!(state.to_dsl().contains("write(o4)"));
    let saved = state.serialize().unwrap();
    state.deserialize(&saved).unwrap();
    // `new Map(Object.entries(...))`: the keys are now strings, so the
    // numeric lookups (and the step-level override) no longer match.
    assert_eq!(state.get_media_input(0), Value::Undefined);
    assert_eq!(
        state.get_media_input("0"),
        Value::from_json(r#"{"type": "image"}"#).unwrap()
    );
    assert_eq!(state.get_write_target("0"), Value::from("o3"));
    let out = state.to_dsl();
    assert!(
        out.contains("write(o3)") && out.contains("write(o2)"),
        "{out}"
    );
}

#[test]
fn insert_and_delete_steps() {
    let dsl = "search synth\nnoise().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    let starter = state.insert_step(0.0, "synth/noise").unwrap();
    assert_eq!(
        starter.error.as_deref(),
        Some("Cannot insert starter effect 'synth/noise' mid-chain")
    );
    let inserted = state.insert_step(0.0, "filter/blur").unwrap();
    assert!(inserted.success);
    assert_eq!(inserted.new_step_index, Some(1.0));
    let new_dsl = inserted.new_dsl.unwrap();
    assert!(new_dsl.starts_with("search synth, filter"), "{new_dsl}");
    assert_eq!(state.structure()[1].effect_key, "filter.blur");
    state.renderer_mut().unwrap().current_dsl = new_dsl;
    let deleted = state.delete_step(0.0).unwrap();
    assert!(deleted.success);
    assert_eq!(deleted.deleted_surface_name, Value::from("o0"));
    assert!(state.structure().is_empty());
    assert_eq!(
        state.delete_step(5.0).unwrap().error.as_deref(),
        Some("step not found")
    );
}

#[test]
fn automation_bindings_round_trip_as_variables() {
    let dsl =
        "search synth\nlet o = osc(oscKind.sine, 0, 10)\nnoise(scaleX: o).write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::ALL));
    state.from_dsl(dsl).unwrap();
    let mut binding = Object::new();
    binding.insert("_varRef", Value::from("o"));
    state
        .set_value("step_0", "scaleX", Value::Object(binding))
        .unwrap();
    state
        .set_value("step_0", "scaleX", Value::from(30.0))
        .unwrap();
    assert_eq!(state.get_value("step_0", "scaleX"), Value::Number(30.0));
    let out = state.to_dsl();
    assert!(out.contains("scaleX: o"), "{out}");
}

#[test]
fn missing_pipeline_members_throw_like_the_reference() {
    let dsl = "search synth3d, filter3d, render\nnoise3d(volumeSize: x32).flow3d().render3d().write(o0)\nrender(o0)";
    let mut state = ProgramState::with_renderer(registry(), host_for(dsl, MockMethods::NONE));
    let err = state.from_dsl(dsl).unwrap_err();
    assert_eq!(
        err,
        JsError::type_error("pipeline.broadcastChainScopedParam is not a function")
    );
    // The state was rebuilt before the pipeline write failed.
    assert_eq!(state.step_count(), 4);
}
