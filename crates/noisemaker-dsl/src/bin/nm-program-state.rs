//! `nm-program-state` — run ProgramState scenarios through the Rust port.
//!
//! ```text
//! nm-program-state run <resolved.jsonl> --out <candidate.jsonl>
//! ```
//!
//! The input is the `--resolved` output of `tools/reference-program-state.mjs`:
//! `{"suite"}` header lines and one `{name, host, ops}` line per scenario, with
//! every op concrete. Each scenario runs on a fresh
//! [`ProgramState`] over a [`MockHost`] configured like the oracle's mock
//! renderer, and every op writes one record in the oracle's format
//! (`{s, i, op, r | x, ev, log, calls, u, st}`, values in the tagged JSON
//! encoding, unchanged parts as `{"$same": true}`), for
//! `parity/check_program_state.mjs` to compare line by line with the
//! reference's records.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::process::ExitCode;
use std::rc::Rc;

use noisemaker_dsl::compiler::{CompileOptions, compile_graph};
use noisemaker_dsl::program_state::{
    Console, ConsoleArg, Listener, MockConvert, MockHost, MockMethods, MockPipeline, ProgramState,
    convert_parameter_for_uniform, effects_to_value, extract_effects_from_dsl, resolve_enum_value,
    set_console,
};
use noisemaker_dsl::unparser::jsv::{json_stringify, member};
use noisemaker_dsl::{JsError, Object, Registry, Value};

type State = ProgramState<MockHost>;

const TAG: &str = "$js";
const EVENTS: [&str; 8] = [
    "change",
    "stepchange",
    "structurechange",
    "reset",
    "load",
    "recompileNeeded",
    "mediachange",
    "textchange",
];

// ---------------------------------------------------------------------------
// Tagged encoding
// ---------------------------------------------------------------------------

fn obj(members: Vec<(&str, Value)>) -> Value {
    let mut o = Object::new();
    for (k, v) in members {
        o.insert(k, v);
    }
    Value::Object(o)
}

fn tag(name: &str) -> Value {
    obj(vec![(TAG, Value::from(name))])
}

/// A JavaScript value in the tagged encoding.
fn enc(v: &Value) -> Value {
    match v {
        Value::Undefined => tag("undefined"),
        Value::Number(n) if n.is_nan() => tag("NaN"),
        Value::Number(n) if n.is_infinite() => tag(if *n > 0.0 { "Infinity" } else { "-Infinity" }),
        Value::Number(n) if *n == 0.0 && n.is_sign_negative() => tag("-0"),
        Value::Function(_) => tag("function"),
        Value::Array(items) => Value::Array(items.iter().map(enc).collect()),
        Value::Object(o) => {
            let members: Object = o.iter().map(|(k, v)| (k.clone(), enc(v))).collect();
            if members.contains_key(TAG) {
                obj(vec![
                    (TAG, Value::from("object")),
                    ("members", Value::Object(members)),
                ])
            } else {
                Value::Object(members)
            }
        }
        other => other.clone(),
    }
}

fn dec_members(v: &Value) -> Value {
    match v {
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), dec(v))).collect()),
        _ => Value::Object(Object::new()),
    }
}

/// A tagged value back to a JavaScript value.
fn dec(v: &Value) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(dec).collect()),
        Value::Object(o) => match o.get(TAG).and_then(Value::as_str) {
            None => dec_members(v),
            Some("undefined") => Value::Undefined,
            Some("NaN") => Value::Number(f64::NAN),
            Some("Infinity") => Value::Number(f64::INFINITY),
            Some("-Infinity") => Value::Number(f64::NEG_INFINITY),
            Some("-0") => Value::Number(-0.0),
            Some("function") => Value::Function("function () {}".into()),
            Some("object") => dec_members(v.get("members")),
            Some(other) => panic!("unknown value tag {other}"),
        },
        other => other.clone(),
    }
}

/// `{name, message}` for Error instances, `{thrown}` for other thrown values.
fn thrown_rec(e: &JsError) -> Value {
    match e {
        JsError::Error { name, message } => obj(vec![
            ("name", Value::from(name.as_str())),
            ("message", Value::from(message.as_str())),
        ]),
        JsError::Thrown(v) => obj(vec![("thrown", enc(v))]),
    }
}

/// A thrown `Error` of the scenario's description (`{name, message}`).
fn make_error(spec: &Value) -> JsError {
    JsError::Error {
        name: spec.get("name").as_str().unwrap_or("Error").to_owned(),
        message: spec.get("message").as_str().unwrap_or_default().to_owned(),
    }
}

fn json(v: &Value) -> String {
    json_stringify(v).unwrap_or_else(|| "null".into())
}

// ---------------------------------------------------------------------------
// Console capture
// ---------------------------------------------------------------------------

#[derive(Default)]
struct RecordingConsole {
    log: RefCell<Vec<Value>>,
}

impl RecordingConsole {
    fn push(&self, level: &str, args: &[ConsoleArg]) {
        let args = args
            .iter()
            .map(|a| match a {
                ConsoleArg::Value(v) => enc(v),
                ConsoleArg::Error(JsError::Error { name, message }) => obj(vec![
                    (TAG, Value::from("error")),
                    ("name", Value::from(name.as_str())),
                    ("message", Value::from(message.as_str())),
                ]),
                ConsoleArg::Error(JsError::Thrown(v)) => enc(v),
            })
            .collect();
        self.log.borrow_mut().push(obj(vec![
            ("level", Value::from(level)),
            ("args", Value::Array(args)),
        ]));
    }
}

impl Console for RecordingConsole {
    fn warn(&self, args: &[ConsoleArg]) {
        self.push("warn", args);
    }
    fn error(&self, args: &[ConsoleArg]) {
        self.push("error", args);
    }
}

// ---------------------------------------------------------------------------
// Scenario context
// ---------------------------------------------------------------------------

/// What listeners share with the scenario.
#[derive(Default)]
struct Shared {
    /// Events seen during the current op.
    events: RefCell<Vec<Value>>,
    /// `$same` compression: key -> last JSON.
    last: RefCell<HashMap<String, String>>,
    /// Listeners by id.
    listeners: RefCell<HashMap<String, Listener<State>>>,
}

impl Shared {
    fn same(&self, key: &str, encoded: Value) -> Value {
        let text = json(&encoded);
        let mut last = self.last.borrow_mut();
        if last.get(key) == Some(&text) {
            return obj(vec![("$same", Value::Bool(true))]);
        }
        last.insert(key.to_owned(), text);
        encoded
    }

    fn listener(&self, id: &str) -> Listener<State> {
        self.listeners
            .borrow()
            .get(id)
            .cloned()
            .unwrap_or_else(|| panic!("undefined listener {id}"))
    }
}

struct Globals {
    registry: Rc<Registry>,
    host_enums: Rc<Value>,
    std_enums: Rc<Value>,
    console: Rc<RecordingConsole>,
}

struct Ctx<'g> {
    g: &'g Globals,
    name: String,
    index: usize,
    state: State,
    /// The mock renderer while it is not attached to the state.
    detached: Option<MockHost>,
    methods: MockMethods,
    pass_json: Vec<String>,
    shared: Rc<Shared>,
}

fn host_config(host: &Value) -> (bool, String, String, Vec<String>) {
    let field = |k: &str| match host {
        Value::Object(o) => o.get(k).cloned(),
        _ => None,
    };
    let renderer = field("renderer").map(|v| v.is_truthy()).unwrap_or(true);
    let enums = field("enums")
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| "host".into());
    let convert = field("convert")
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_else(|| "canvas".into());
    let methods = match field("methods") {
        Some(Value::Array(m)) => m
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => [
            "broadcastChainScopedParam",
            "checkAsyncRegen",
            "recreateTextures",
            "collectDefaultUniforms",
            "setUniform",
        ]
        .iter()
        .map(|s| (*s).to_owned())
        .collect(),
    };
    (renderer, enums, convert, methods)
}

impl<'g> Ctx<'g> {
    fn new(g: &'g Globals, name: String, host: &Value) -> Self {
        let mut ctx = Ctx {
            g,
            name,
            index: 0,
            state: ProgramState::new(g.registry.clone()),
            detached: None,
            methods: MockMethods::ALL,
            pass_json: Vec::new(),
            shared: Rc::new(Shared::default()),
        };
        ctx.new_state(host);
        ctx
    }

    fn enums_for(&self, name: &str) -> Rc<Value> {
        match name {
            "host" => self.g.host_enums.clone(),
            "std" => self.g.std_enums.clone(),
            "empty" => Rc::new(Value::Object(Object::new())),
            "none" => Rc::new(Value::Undefined),
            other => panic!("unknown enums {other}"),
        }
    }

    /// `newState(host)`: a fresh renderer and ProgramState with the recorders.
    fn new_state(&mut self, host: &Value) {
        let (attach, enums, convert, methods) = host_config(host);
        self.methods = MockMethods::from_names(&methods);
        let mock = MockHost {
            current_dsl: String::new(),
            enums: self.enums_for(&enums),
            convert: match convert.as_str() {
                "canvas" => MockConvert::Canvas,
                "passthrough" => MockConvert::Passthrough,
                "absent" => MockConvert::Absent,
                other => panic!("unknown convert {other}"),
            },
            pipeline: None,
        };
        self.shared.events.borrow_mut().clear();
        self.shared.listeners.borrow_mut().clear();
        self.pass_json.clear();
        self.state = ProgramState::new(self.g.registry.clone());
        if attach {
            self.state.set_renderer(Some(mock));
            self.detached = None;
        } else {
            self.detached = Some(mock);
        }
        for event in EVENTS {
            let shared = self.shared.clone();
            self.state.on(
                event,
                Rc::new(move |_: &mut State, data: &Value| {
                    let d = shared.same(&format!("ev.{event}"), enc(data));
                    shared.events.borrow_mut().push(obj(vec![
                        ("l", Value::from("*")),
                        ("e", Value::from(event)),
                        ("d", d),
                    ]));
                    Ok(())
                }),
            );
        }
    }

    fn mock(&mut self) -> &mut MockHost {
        match self.state.renderer_mut() {
            Some(m) => m,
            None => self.detached.as_mut().expect("the mock renderer exists"),
        }
    }

    fn mock_ref(&self) -> &MockHost {
        match self.state.renderer() {
            Some(m) => m,
            None => self.detached.as_ref().expect("the mock renderer exists"),
        }
    }

    fn host_load(&mut self, dsl: &str, set_dsl: bool) -> Value {
        let methods = self.methods;
        let registry = self.g.registry.clone();
        let mock = self.mock();
        if set_dsl {
            mock.current_dsl = dsl.to_owned();
        }
        match compile_graph(dsl, &registry, &CompileOptions::default()) {
            Err(e) => {
                mock.pipeline = None;
                self.pass_json.clear();
                obj(vec![("error", thrown_rec(&e))])
            }
            Ok(graph) => {
                let passes = match graph.get("passes") {
                    Value::Array(p) => p.clone(),
                    _ => Vec::new(),
                };
                mock.pipeline = Some(MockPipeline::new(graph, methods));
                self.pass_json = passes
                    .iter()
                    .map(|p| json(&enc(p.get("uniforms"))))
                    .collect();
                let summaries = passes
                    .iter()
                    .map(|p| {
                        obj(vec![
                            ("id", member(p, "id")),
                            ("nodeId", member(p, "nodeId")),
                            ("effectKey", member(p, "effectKey")),
                            ("stepIndex", member(p, "stepIndex")),
                            ("uniforms", member(p, "uniforms")),
                            ("scopedParams", member(p, "scopedParams")),
                            ("inheritsVolumeSize", member(p, "inheritsVolumeSize")),
                            ("uniformAliases", member(p, "uniformAliases")),
                        ])
                    })
                    .collect();
                obj(vec![("passes", Value::Array(summaries))])
            }
        }
    }

    /// Top-level ops: host operations, then state ops.
    fn run_op(&mut self, op: &Value) -> Result<Value, JsError> {
        let name = op.get("op").as_str().unwrap_or_default();
        let s = |k: &str| op.get(k).as_str().unwrap_or_default().to_owned();
        match name {
            "host.load" => Ok(enc(&self.host_load(&s("dsl"), true))),
            "host.loadGraph" => Ok(enc(&self.host_load(&s("dsl"), false))),
            "host.setDsl" => {
                self.mock().current_dsl = s("dsl");
                Ok(enc(&Value::Undefined))
            }
            "host.clearPipeline" => {
                self.mock().pipeline = None;
                self.pass_json.clear();
                Ok(enc(&Value::Undefined))
            }
            "host.setMethods" => {
                let names: Vec<String> = match op.get("methods") {
                    Value::Array(m) => m
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_owned))
                        .collect(),
                    _ => Vec::new(),
                };
                self.methods = MockMethods::from_names(&names);
                Ok(enc(&Value::Undefined))
            }
            "host.setEnums" => {
                let enums = self.enums_for(&s("enums"));
                self.mock().enums = enums;
                Ok(enc(&Value::Undefined))
            }
            "setRenderer" => {
                // The scenario's mock renderer stays the same object while
                // detached, as the oracle's does.
                if op.get("renderer").is_truthy() {
                    if self.state.renderer().is_none() {
                        let mock = self.detached.take();
                        self.state.set_renderer(mock);
                    }
                } else if self.state.renderer().is_some() {
                    self.detached = take_renderer(&mut self.state);
                }
                Ok(enc(&Value::Undefined))
            }
            "newState" => {
                let host = op.get("host").clone();
                self.new_state(&host);
                Ok(enc(&Value::Undefined))
            }
            "convertParameterForUniform" => {
                let enums = self.enums_for(&s("enums"));
                Ok(enc(&convert_parameter_for_uniform(
                    &dec(op.get("value")),
                    &dec(op.get("spec")),
                    &enums,
                )?))
            }
            "resolveEnumValue" => {
                let enums = self.enums_for(&s("enums"));
                Ok(enc(&resolve_enum_value(&dec(op.get("path")), &enums)))
            }
            _ => run_state_op(&mut self.state, &self.shared, op),
        }
    }

    fn compress_calls(&self, calls: Vec<Value>) -> Vec<Value> {
        calls
            .into_iter()
            .map(|c| {
                let mut c = enc(&c);
                let f = c.get("fn").as_str().unwrap_or_default().to_owned();
                if f == "checkAsyncRegen" {
                    let key = format!("call.checkAsyncRegen.{}", json(c.get("nodeId")));
                    let v = self.shared.same(&key, c.get("stepValues").clone());
                    c.set("stepValues", v);
                } else if f == "recreateTextures" {
                    let v = self
                        .shared
                        .same("call.recreateTextures", c.get("uniforms").clone());
                    c.set("uniforms", v);
                }
                c
            })
            .collect()
    }

    fn uniform_deltas(&mut self) -> Vec<Value> {
        let passes: Vec<Value> = match self
            .mock_ref()
            .pipeline
            .as_ref()
            .map(|p| p.graph.get("passes"))
        {
            Some(Value::Array(p)) => p.clone(),
            _ => return Vec::new(),
        };
        let mut out = Vec::new();
        for (i, pass) in passes.iter().enumerate() {
            let u = enc(pass.get("uniforms"));
            let text = json(&u);
            if self.pass_json.get(i) != Some(&text) {
                out.push(obj(vec![("p", Value::from(i)), ("u", u)]));
                if i < self.pass_json.len() {
                    self.pass_json[i] = text;
                } else {
                    self.pass_json.resize(i, String::new());
                    self.pass_json.push(text);
                }
            }
        }
        out
    }

    fn snapshot(&self) -> Value {
        let st = &self.state;
        let shared = &self.shared;
        let steps: Vec<Value> = st
            .step_states()
            .iter()
            .map(|(key, s)| {
                let def = match &s.effect_def {
                    Some(entry) if entry.def.is_truthy() => obj(vec![
                        ("func", member(&entry.def, "func")),
                        ("namespace", member(&entry.def, "namespace")),
                    ]),
                    _ => Value::Null,
                };
                Value::Array(vec![
                    Value::from(key.as_str()),
                    obj(vec![
                        ("effectKey", Value::from(s.effect_key.as_str())),
                        ("def", def),
                        ("stepIndex", Value::Number(s.step_index)),
                        ("values", Value::Object(s.values.clone())),
                    ]),
                ])
            })
            .collect();
        let entries = |m: &noisemaker_dsl::program_state::JsMap<Value>| -> Value {
            Value::Array(
                m.iter()
                    .map(|(k, v)| Value::Array(vec![k.clone(), v.clone()]))
                    .collect(),
            )
        };
        let [wt, wst, rs, r3v, r3g, w3v, w3g] = st.routing_overrides();
        let routing = obj(vec![
            ("writeTargets", entries(wt)),
            ("writeStepTargets", entries(wst)),
            ("readSources", entries(rs)),
            ("read3dVol", entries(r3v)),
            ("read3dGeo", entries(r3g)),
            ("write3dVol", entries(w3v)),
            ("write3dGeo", entries(w3g)),
            ("renderTarget", st.get_render_target()),
        ]);
        let batch = obj(vec![
            ("depth", Value::from(st.batch_depth())),
            (
                "changes",
                Value::Array(st.batched_changes().iter().map(|c| c.to_value()).collect()),
            ),
            ("recompilePending", Value::Bool(st.recompile_pending())),
        ]);
        obj(vec![
            ("steps", shared.same("st.steps", enc(&Value::Array(steps)))),
            (
                "structure",
                shared.same("st.structure", enc(&effects_to_value(st.structure()))),
            ),
            (
                "compiled",
                shared.same(
                    "st.compiled",
                    enc(&st.get_compiled().cloned().unwrap_or(Value::Null)),
                ),
            ),
            ("routing", shared.same("st.routing", enc(&routing))),
            (
                "media",
                shared.same("st.media", enc(&entries(&st.get_all_media_inputs()))),
            ),
            (
                "text",
                shared.same("st.text", enc(&entries(&st.get_all_text_inputs()))),
            ),
            ("batch", enc(&batch)),
            (
                "dsl",
                match st.renderer() {
                    Some(r) => Value::from(r.current_dsl.as_str()),
                    None => Value::Null,
                },
            ),
        ])
    }

    /// Run one op and write its record.
    fn exec(&mut self, op: &Value, out: &mut impl Write) {
        self.g.console.log.borrow_mut().clear();
        let name = op.get("op").as_str().unwrap_or_default().to_owned();
        let outcome = catch_unwind(AssertUnwindSafe(|| self.run_op(op))).unwrap_or_else(|panic| {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_default();
            Err(JsError::Error {
                name: "RustPanic".into(),
                message,
            })
        });
        let mut rec = Object::new();
        rec.insert("s", Value::from(self.name.as_str()));
        rec.insert("i", Value::from(self.index));
        rec.insert("op", Value::from(name.as_str()));
        self.index += 1;
        match outcome {
            Ok(r) => {
                let r = if QUERY_OPS.contains(&name.as_str()) {
                    self.shared.same(&format!("res.{name}"), r)
                } else {
                    r
                };
                rec.insert("r", r);
            }
            Err(e) => {
                rec.insert("x", thrown_rec(&e));
            }
        }
        let events = std::mem::take(&mut *self.shared.events.borrow_mut());
        rec.insert("ev", Value::Array(events));
        let log = std::mem::take(&mut *self.g.console.log.borrow_mut());
        rec.insert("log", Value::Array(log));
        let calls = self.mock().take_calls();
        rec.insert("calls", Value::Array(self.compress_calls(calls)));
        let deltas = self.uniform_deltas();
        rec.insert("u", Value::Array(deltas));
        rec.insert("st", self.snapshot());
        writeln!(out, "{}", json(&Value::Object(rec))).expect("write record");
    }
}

/// Detach the renderer from the state, returning it.
fn take_renderer(state: &mut State) -> Option<MockHost> {
    let mock = state.renderer().cloned();
    state.set_renderer(None);
    mock
}

const QUERY_OPS: [&str; 4] = [
    "getStructure",
    "getCompiled",
    "getAllStepValues",
    "serialize",
];

fn effect_def_summary(def: Option<Value>) -> Value {
    match def {
        Some(def) => obj(vec![
            ("func", member(&def, "func")),
            ("namespace", member(&def, "namespace")),
        ]),
        None => Value::Null,
    }
}

fn number_arg(op: &Value, key: &str) -> f64 {
    match dec(op.get(key)) {
        Value::Number(n) => n,
        other => panic!("{key} must be a number, got {other:?}"),
    }
}

fn entries_value(m: &noisemaker_dsl::program_state::JsMap<Value>) -> Value {
    Value::Array(
        m.iter()
            .map(|(k, v)| Value::Array(vec![k.clone(), v.clone()]))
            .collect(),
    )
}

/// A listener running a scenario's action list.
fn make_listener(shared: &Rc<Shared>, id: &str, actions: &Value) -> Listener<State> {
    let shared = shared.clone();
    let id = id.to_owned();
    let actions = match actions {
        Value::Array(a) => a.clone(),
        _ => Vec::new(),
    };
    Rc::new(move |state: &mut State, data: &Value| {
        for action in &actions {
            let event = action.get("event").as_str().unwrap_or_default().to_owned();
            match action.get("do").as_str().unwrap_or_default() {
                "record" => shared
                    .events
                    .borrow_mut()
                    .push(obj(vec![("l", Value::from(id.as_str())), ("d", enc(data))])),
                "throw" => return Err(make_error(action)),
                "on" => state.on(
                    &event,
                    shared.listener(action.get("id").as_str().unwrap_or_default()),
                ),
                "off" => state.off(
                    &event,
                    &shared.listener(action.get("id").as_str().unwrap_or_default()),
                ),
                "once" => {
                    state.once(
                        &event,
                        shared.listener(action.get("id").as_str().unwrap_or_default()),
                    );
                }
                "removeAllListeners" => {
                    let e = action.get("event").as_str().map(str::to_owned);
                    state.remove_all_listeners(e.as_deref());
                }
                "op" => {
                    let nested = action.get("op");
                    let n = nested.get("op").clone();
                    match run_state_op(state, &shared, nested) {
                        Ok(r) => shared.events.borrow_mut().push(obj(vec![
                            ("l", Value::from(id.as_str())),
                            ("n", n),
                            ("r", r),
                        ])),
                        Err(e) => {
                            shared.events.borrow_mut().push(obj(vec![
                                ("l", Value::from(id.as_str())),
                                ("n", n),
                                ("x", thrown_rec(&e)),
                            ]));
                            if action.get("rethrow").is_truthy() {
                                return Err(e);
                            }
                        }
                    }
                }
                other => panic!("unknown listener action {other}"),
            }
        }
        Ok(())
    })
}

/// A ProgramState op; returns the encoded result.
fn run_state_op(state: &mut State, shared: &Rc<Shared>, op: &Value) -> Result<Value, JsError> {
    let name = op.get("op").as_str().unwrap_or_default();
    let s = |k: &str| op.get(k).as_str().unwrap_or_default().to_owned();
    let a = |k: &str| dec(op.get(k));
    let undefined = || Ok(enc(&Value::Undefined));
    match name {
        "fromDsl" => {
            state.from_dsl(&s("dsl"))?;
            undefined()
        }
        "toDsl" => Ok(enc(&Value::from(state.to_dsl()))),
        "wouldChangeStructure" => Ok(enc(&Value::Bool(state.would_change_structure(&s("dsl"))))),
        "getValue" => Ok(enc(&state.get_value(&s("stepKey"), &s("paramName")))),
        "setValue" => {
            state.set_value(&s("stepKey"), &s("paramName"), a("value"))?;
            undefined()
        }
        "getStepValues" => Ok(enc(&Value::Object(state.get_step_values(&s("stepKey"))))),
        "setStepValues" => {
            state.set_step_values(&s("stepKey"), &a("values"))?;
            undefined()
        }
        "batch" => {
            let nested = match op.get("ops") {
                Value::Array(ops) => ops.clone(),
                _ => Vec::new(),
            };
            let throw = op.get("throw").clone();
            let mut results = Vec::new();
            state.batch(|st| {
                for n in &nested {
                    results.push(run_state_op(st, shared, n)?);
                }
                if throw.is_truthy() {
                    return Err(make_error(&throw));
                }
                Ok(())
            })?;
            Ok(Value::Array(results))
        }
        "resetStep" => {
            state.reset_step(&s("stepKey"))?;
            undefined()
        }
        "setSkip" => {
            state.set_skip(&s("stepKey"), a("skip"))?;
            undefined()
        }
        "isSkipped" => Ok(enc(&Value::Bool(state.is_skipped(&s("stepKey"))))),
        "deleteStep" => Ok(enc(&state
            .delete_step(number_arg(op, "stepIndex"))?
            .to_value())),
        "insertStep" => Ok(enc(&state
            .insert_step(number_arg(op, "afterStepIndex"), &s("effectId"))?
            .to_value())),
        "getStructure" => Ok(enc(&effects_to_value(&state.get_structure()))),
        "getCompiled" => Ok(enc(&state.get_compiled().cloned().unwrap_or(Value::Null))),
        "getEffectDef" => Ok(enc(&effect_def_summary(
            state.get_effect_def(&s("stepKey")),
        ))),
        "stepCount" => Ok(enc(&Value::from(state.step_count()))),
        "getStepKeys" => Ok(enc(&Value::Array(
            state.get_step_keys().into_iter().map(Value::from).collect(),
        ))),
        "getAllStepValues" => Ok(enc(&Value::Object(state.get_all_step_values()))),
        "setWriteTarget" => {
            state.set_write_target(a("planIndex"), a("target"));
            undefined()
        }
        "getWriteTarget" => Ok(enc(&state.get_write_target(a("planIndex")))),
        "setWriteStepTarget" => {
            state.set_write_step_target(a("stepIndex"), a("target"));
            undefined()
        }
        "getWriteStepTarget" => Ok(enc(&state.get_write_step_target(a("stepIndex")))),
        "setReadSource" => {
            state.set_read_source(a("stepIndex"), a("source"));
            undefined()
        }
        "getReadSource" => Ok(enc(&state.get_read_source(a("stepIndex")))),
        "setRead3dVolume" => {
            state.set_read3d_volume(a("stepIndex"), a("volume"));
            undefined()
        }
        "setRead3dGeometry" => {
            state.set_read3d_geometry(a("stepIndex"), a("geometry"));
            undefined()
        }
        "setWrite3dVolume" => {
            state.set_write3d_volume(a("stepIndex"), a("volume"));
            undefined()
        }
        "setWrite3dGeometry" => {
            state.set_write3d_geometry(a("stepIndex"), a("geometry"));
            undefined()
        }
        "setRenderTarget" => {
            state.set_render_target(a("target"));
            undefined()
        }
        "getRenderTarget" => Ok(enc(&state.get_render_target())),
        "clearRoutingOverrides" => {
            state.clear_routing_overrides();
            undefined()
        }
        "setMediaInput" => {
            state.set_media_input(a("stepIndex"), a("metadata"));
            undefined()
        }
        "getMediaInput" => Ok(enc(&state.get_media_input(a("stepIndex")))),
        "removeMediaInput" => {
            state.remove_media_input(a("stepIndex"));
            undefined()
        }
        "getAllMediaInputs" => Ok(enc(&entries_value(&state.get_all_media_inputs()))),
        "setTextInput" => {
            state.set_text_input(a("stepIndex"), a("metadata"));
            undefined()
        }
        "getTextInput" => Ok(enc(&state.get_text_input(a("stepIndex")))),
        "removeTextInput" => {
            state.remove_text_input(a("stepIndex"));
            undefined()
        }
        "getAllTextInputs" => Ok(enc(&entries_value(&state.get_all_text_inputs()))),
        "applyToPipeline" => {
            state.apply_to_pipeline()?;
            undefined()
        }
        "serialize" => Ok(enc(&state.serialize()?)),
        "deserialize" => {
            state.deserialize(&a("data"))?;
            undefined()
        }
        "emit" => {
            state.emit(&s("event"), &a("data"));
            undefined()
        }
        "defineListener" => {
            let id = s("id");
            let l = make_listener(shared, &id, op.get("actions"));
            shared.listeners.borrow_mut().insert(id, l);
            undefined()
        }
        "on" => {
            state.on(&s("event"), shared.listener(&s("id")));
            undefined()
        }
        "off" => {
            state.off(&s("event"), &shared.listener(&s("id")));
            undefined()
        }
        "once" => {
            state.once(&s("event"), shared.listener(&s("id")));
            undefined()
        }
        "removeAllListeners" => {
            let event = op.get("event").as_str().map(str::to_owned);
            state.remove_all_listeners(event.as_deref());
            undefined()
        }
        "extractEffectsFromDsl" => Ok(enc(&effects_to_value(&extract_effects_from_dsl(
            &s("dsl"),
            state.registry(),
        )))),
        other => Err(JsError::error(format!("unknown op {other}"))),
    }
}

fn cmd_run(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut out = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            other => input = Some(other.to_owned()),
        }
        i += 1;
    }
    let (Some(input), Some(out_path)) = (input, out) else {
        return Err("usage: nm-program-state run <resolved.jsonl> --out <candidate.jsonl>".into());
    };
    let text = fs::read_to_string(&input).map_err(|e| format!("{input}: {e}"))?;
    let registry = Rc::new(Registry::with_catalog());
    let console = Rc::new(RecordingConsole::default());
    set_console(console.clone());
    let globals = Globals {
        host_enums: Rc::new(Value::Object(registry.enums.clone())),
        std_enums: Rc::new(Value::Object(registry.std_enums())),
        registry,
        console,
    };
    let file = fs::File::create(&out_path).map_err(|e| format!("{out_path}: {e}"))?;
    let mut out = std::io::BufWriter::new(file);
    // Panics are recorded per op; keep their default report off the output.
    std::panic::set_hook(Box::new(|info| {
        eprintln!("nm-program-state: panic: {info}")
    }));
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let scenario = Value::from_json(line).map_err(|e| e.to_string())?;
        if scenario.get("suite").is_truthy() {
            continue;
        }
        let name = scenario.get("name").as_str().unwrap_or_default().to_owned();
        let mut ctx = Ctx::new(&globals, name, scenario.get("host"));
        if let Value::Array(ops) = scenario.get("ops") {
            for op in ops {
                ctx.exec(op, &mut out);
            }
        }
    }
    out.flush().map_err(|e| e.to_string())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        _ => Err("usage: nm-program-state run <resolved.jsonl> --out <candidate.jsonl>".into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("nm-program-state: {message}");
            ExitCode::FAILURE
        }
    }
}
