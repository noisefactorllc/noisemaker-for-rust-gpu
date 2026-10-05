//! `nm-dsl` — the DSL tooling of `noisemaker-dsl` on the command line.
//!
//! ```text
//! nm-dsl unparse <program.dsl | compiled.json> [--host]   regenerate DSL source
//! nm-dsl steps <program.dsl | compiled.json>              listSteps() as JSON
//! nm-dsl validate-effect <definition.json> [--instance]   validateEffectDefinition() errors
//! nm-dsl format-error <program.dsl>                       formatDslError() of the compile error
//! nm-dsl cases <resolved.jsonl> --out <results.jsonl> [--rust-frontend]
//! ```
//!
//! `.dsl` inputs are compiled with this crate's frontend (lex, parse,
//! validate); `.json` inputs are compiled programs. `--host` unparses the way
//! the host's program state does (effect definitions from the registry, enum
//! names from the merged enum tree).
//!
//! `cases` runs a case file written by `tools/reference-dsl-tools.mjs` (a
//! `{"setup": [...]}` header, then one case per line) and writes one result
//! record per case in the same tagged JSON encoding, for
//! `parity/check_dsl_tools.mjs` to compare with the reference's records.
//! `--rust-frontend` compiles the cases' DSL sources with this crate's frontend
//! instead of using the reference's compile results.

use std::fs;
use std::process::ExitCode;
use std::rc::Rc;

use noisemaker_dsl::effect_validator::{Definition, validate_definition};
use noisemaker_dsl::error_formatter::{format_dsl_error, is_dsl_syntax_error};
use noisemaker_dsl::registry::EffectEntry;
use noisemaker_dsl::transform::{
    get_compatible_replacements, list_steps, predict_replacement, replace_effect,
};
use noisemaker_dsl::unparser::jsv::{
    cannot_read, not_a_function, object_member, to_property_key, to_string,
};
use noisemaker_dsl::unparser::{
    CustomFormatter, EffectDefLookup, UnparseOptions, apply_parameter_updates, format_value,
    unparse, unparse_call, unparse_chain,
};
use noisemaker_dsl::{JsError, Object, Registry, Stage, Value, run_stage};

const TAG: &str = "$js";

// ---------------------------------------------------------------------------
// Tagged JSON encoding (tools/reference-dsl-tools.mjs)
// ---------------------------------------------------------------------------

fn tag(name: &str) -> Value {
    let mut o = Object::new();
    o.insert(TAG, Value::from(name));
    Value::Object(o)
}

/// A JavaScript value as tagged JSON.
fn encode(v: &Value) -> Value {
    match v {
        Value::Undefined => tag("undefined"),
        Value::Number(n) if n.is_nan() => tag("NaN"),
        Value::Number(n) if n.is_infinite() => tag(if *n > 0.0 { "Infinity" } else { "-Infinity" }),
        Value::Number(n) if *n == 0.0 && n.is_sign_negative() => tag("-0"),
        Value::Function(source) => {
            let mut o = Object::new();
            o.insert(TAG, Value::from("function"));
            o.insert("source", Value::from(source.as_str()));
            Value::Object(o)
        }
        Value::Array(items) => Value::Array(items.iter().map(encode).collect()),
        Value::Object(members) => {
            let encoded: Object = members
                .iter()
                .map(|(k, v)| (k.clone(), encode(v)))
                .collect();
            if encoded.contains_key(TAG) {
                let mut o = Object::new();
                o.insert(TAG, Value::from("object"));
                o.insert("members", Value::Object(encoded));
                Value::Object(o)
            } else {
                Value::Object(encoded)
            }
        }
        other => other.clone(),
    }
}

fn decode_members(v: &Value) -> Value {
    match v {
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), decode(v))).collect()),
        _ => Value::Object(Object::new()),
    }
}

/// Tagged JSON back to a JavaScript value.
fn decode(v: &Value) -> Value {
    match v {
        Value::Array(items) => Value::Array(items.iter().map(decode).collect()),
        Value::Object(o) => match o.get(TAG).and_then(Value::as_str) {
            None => decode_members(v),
            Some(t) => match t {
                "undefined" => Value::Undefined,
                "NaN" => Value::Number(f64::NAN),
                "Infinity" => Value::Number(f64::INFINITY),
                "-Infinity" => Value::Number(f64::NEG_INFINITY),
                "-0" => Value::Number(-0.0),
                "function" => {
                    Value::Function(v.get("source").as_str().unwrap_or_default().to_owned())
                }
                // Typed arrays read like arrays of their (already rounded) elements.
                "Float32Array" | "Float64Array" => match v.get("values") {
                    Value::Array(values) => Value::Array(values.iter().map(decode).collect()),
                    _ => Value::Array(Vec::new()),
                },
                // A Map has no enumerable string-keyed members.
                "Map" => Value::Object(Object::new()),
                "object" => decode_members(v.get("members")),
                "error" => {
                    let mut e = Object::new();
                    e.insert("name", v.get("name").clone());
                    e.insert("message", v.get("message").clone());
                    Value::Object(e)
                }
                "effectInstance" => decode_members(v.get("props")),
                "effectSubclass" => Value::Function("class extends Effect {}".into()),
                other => panic!("unknown value tag {other}"),
            },
        },
        other => other.clone(),
    }
}

/// A thrown value from its tagged encoding.
fn decode_thrown(v: &Value) -> JsError {
    if v.get(TAG).as_str() == Some("error") {
        JsError::Error {
            name: v.get("name").as_str().unwrap_or("Error").to_owned(),
            message: v.get("message").as_str().unwrap_or_default().to_owned(),
        }
    } else {
        JsError::Thrown(decode(v))
    }
}

/// `{name, message}` for Error instances, `{thrown}` for other thrown values.
fn error_record(e: &JsError) -> Value {
    match e {
        JsError::Error { name, message } => {
            let mut o = Object::new();
            o.insert("name", Value::from(name.as_str()));
            o.insert("message", Value::from(message.as_str()));
            Value::Object(o)
        }
        JsError::Thrown(v) => {
            let mut o = Object::new();
            o.insert("thrown", encode(v));
            Value::Object(o)
        }
    }
}

// ---------------------------------------------------------------------------
// Registry setup and harness options
// ---------------------------------------------------------------------------

/// Apply a suite's registry mutations (the reference test files' own
/// registrations) on top of the host registration.
fn apply_setup(registry: &mut Registry, setup: &Value) {
    let Value::Array(steps) = setup else {
        return;
    };
    for step in steps {
        let name = || step.get("name").as_str().unwrap_or_default().to_owned();
        match step.get("op").as_str() {
            Some("registerOp") => registry.register_op(name(), decode(step.get("spec"))),
            Some("registerStarterOps") => {
                if let Value::Array(names) = decode(step.get("names")) {
                    let names: Vec<String> = names
                        .iter()
                        .filter_map(|n| n.as_str().map(str::to_owned))
                        .collect();
                    registry.register_starter_ops(&names);
                }
            }
            Some("registerEffect") => {
                let def = decode(step.get("definition"));
                let entry = EffectEntry {
                    namespace: def.get("namespace").as_str().unwrap_or_default().to_owned(),
                    name: def.get("name").as_str().unwrap_or_default().to_owned(),
                    def,
                };
                registry.register_effect(name(), Rc::new(entry));
            }
            Some("registerParamAliases") => {
                if let Value::Object(aliases) = decode(step.get("aliases")) {
                    registry.register_param_aliases(
                        step.get("opName").as_str().unwrap_or_default(),
                        &aliases,
                    );
                }
            }
            Some("registerEffectAlias") => registry.register_effect_alias(
                step.get("opName").as_str().unwrap_or_default(),
                step.get("newName").as_str().unwrap_or_default(),
            ),
            Some("mergeIntoEnums") => {
                if let Value::Object(source) = decode(step.get("source")) {
                    registry.merge_into_enums(&source);
                }
            }
            other => panic!("unknown setup op {other:?}"),
        }
    }
}

/// The named enum trees: the standard enums, and the host's merged tree.
struct Enums {
    std: Rc<Value>,
    host: Rc<Value>,
}

impl Enums {
    fn new(registry: &Registry) -> Self {
        Enums {
            std: Rc::new(Value::Object(registry.std_enums())),
            host: Rc::new(Value::Object(registry.enums.clone())),
        }
    }

    /// `{"$enums": "std" | "host"}` or a literal enum tree.
    fn resolve(&self, v: &Value) -> Rc<Value> {
        match v.get("$enums").as_str() {
            Some("std") => self.std.clone(),
            Some("host") => self.host.clone(),
            Some(other) => panic!("unknown enums source {other}"),
            None => Rc::new(decode(v)),
        }
    }
}

/// `getEffect(name)` with JavaScript Map semantics (only strings match).
fn get_effect(registry: &Registry, name: &str) -> Option<Value> {
    registry.get_effect(name).map(|e| e.def.clone())
}

fn effect_def_lookup<'a>(registry: &'a Registry, spec: &Value) -> EffectDefLookup<'a> {
    match spec.get("kind").as_str() {
        // createEffectDefCallback(getEffect) (demo-ui.js), the ProgramState.toDsl lookup.
        Some("registry") => Rc::new(move |name: &Value, namespace: &Value| {
            if let Value::String(n) = name
                && let Some(def) = get_effect(registry, n)
            {
                return Ok(def);
            }
            let effect_name = match name {
                Value::String(n) => n.as_str(),
                Value::Undefined | Value::Null => return Err(cannot_read(name, "includes")),
                _ => return Err(not_a_function("effectName.includes")),
            };
            if effect_name.contains('.')
                && let Some(def) = get_effect(registry, &effect_name.replacen('.', "/", 1))
            {
                return Ok(def);
            }
            if namespace.is_truthy() {
                let ns = to_string(namespace)?;
                if let Some(def) = get_effect(registry, &format!("{ns}/{effect_name}"))
                    .or_else(|| get_effect(registry, &format!("{ns}.{effect_name}")))
                {
                    return Ok(def);
                }
            }
            Ok(Value::Null)
        }),
        // ProgramState.deleteStep / insertStep lookup.
        Some("registrySimple") => Rc::new(move |name: &Value, namespace: &Value| {
            let direct = match name {
                Value::String(n) => get_effect(registry, n),
                _ => None,
            };
            if let Some(def) = direct {
                return Ok(def);
            }
            if namespace.is_truthy() {
                let ns = to_string(namespace)?;
                let n = to_string(name)?;
                if let Some(def) = get_effect(registry, &format!("{ns}/{n}"))
                    .or_else(|| get_effect(registry, &format!("{ns}.{n}")))
                {
                    return Ok(def);
                }
            }
            Ok(Value::Undefined)
        }),
        Some("map") => {
            let defs = decode(spec.get("defs"));
            Rc::new(move |name: &Value, _namespace: &Value| {
                let key = to_property_key(name)?;
                Ok(defs
                    .as_object()
                    .and_then(|o| o.get(&key))
                    .cloned()
                    .unwrap_or(Value::Null))
            })
        }
        Some("constant") => {
            let def = decode(spec.get("def"));
            Rc::new(move |_name: &Value, _namespace: &Value| Ok(def.clone()))
        }
        Some("ops") => Rc::new(move |name: &Value, _namespace: &Value| {
            let spec = object_member(&registry.ops, &to_property_key(name)?);
            Ok(if spec.is_truthy() { spec } else { Value::Null })
        }),
        other => panic!("unknown getEffectDef kind {other:?}"),
    }
}

fn custom_formatter<'a>(enums: &Enums, spec: &Value) -> CustomFormatter<'a> {
    match spec.get("kind").as_str() {
        // (value, spec) => formatValue(value, spec, { enums })
        Some("formatValue") => {
            let options = UnparseOptions {
                enums: enums.resolve(spec.get("enums")),
                ..UnparseOptions::default()
            };
            Rc::new(move |value: &Value, s: &Value| {
                format_value(value, s, &options, &Value::Undefined)
            })
        }
        other => panic!("unknown customFormatter kind {other:?}"),
    }
}

/// The options object a case describes as data.
fn build_options<'a>(registry: &'a Registry, enums: &Enums, spec: &Value) -> UnparseOptions<'a> {
    let mut options = UnparseOptions::default();
    let Value::Object(members) = spec else {
        return options;
    };
    if let Some(legacy) = members.get("$legacyFormatter") {
        // The legacy third argument of formatValue: a bare customFormatter.
        options.custom_formatter = Some(custom_formatter(enums, legacy));
        return options;
    }
    for (key, value) in members.iter() {
        match key.as_str() {
            "getEffectDef" => options.get_effect_def = Some(effect_def_lookup(registry, value)),
            "customFormatter" => options.custom_formatter = Some(custom_formatter(enums, value)),
            "enums" => options.enums = enums.resolve(value),
            "omitSearchDirective" => options.omit_search_directive = Rc::new(decode(value)),
            "multilineKwargs" => options.multiline_kwargs = Rc::new(decode(value)),
            "indent" => options.indent = Rc::new(decode(value)),
            "specs" => options.specs = Rc::new(decode(value)),
            "schemaSpecs" => options.schema_specs = Rc::new(decode(value)),
            _ => {}
        }
    }
    options
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// Compile a program with this crate's frontend (`validate(parse(lex(src)))`).
fn compile(src: &str, registry: &Registry) -> Result<Value, JsError> {
    run_stage(Stage::Validated, src, registry)
}

struct Runner<'a> {
    registry: &'a Registry,
    enums: Enums,
    rust_frontend: bool,
}

impl Runner<'_> {
    /// A decoded argument (`undefined` when the case omits it); with
    /// `--rust-frontend`, arguments the reference compiled from DSL are
    /// compiled here instead (programs the reference compiled with parse
    /// options keep the reference's result).
    fn arg(&self, case: &Value, name: &str) -> Result<Value, JsError> {
        if self.rust_frontend
            && let Value::Object(from) = case.get("compiledFrom").get(name)
            && from.get("options").is_none_or(Value::is_nullish)
            && let Some(Value::String(dsl)) = from.get("dsl")
        {
            return compile(dsl, self.registry);
        }
        Ok(match case.as_object().and_then(|o| o.get(name)) {
            Some(v) => decode(v),
            None => Value::Undefined,
        })
    }

    fn options<'r>(&'r self, case: &Value, name: &str) -> UnparseOptions<'r> {
        build_options(self.registry, &self.enums, case.get(name))
    }

    fn run(&self, case: &Value) -> Result<Value, JsError> {
        let registry = self.registry;
        let a = |name: &str| self.arg(case, name);
        match case.get("op").as_str().unwrap_or_default() {
            "unparse" => Ok(unparse(
                &a("compiled")?,
                &a("overrides")?,
                &self.options(case, "options"),
                registry,
            )?
            .into()),
            "applyParameterUpdates" => {
                let dsl = a("dsl")?;
                let dsl = dsl.as_str().unwrap_or_default();
                let compile_fn = |src: &str| -> Result<Value, JsError> {
                    if self.rust_frontend && !case.get("compiledLiteral").is_truthy() {
                        return compile(src, registry);
                    }
                    match case.as_object().and_then(|o| o.get("compileError")) {
                        Some(e) => Err(decode_thrown(e)),
                        None => Ok(decode(case.get("compiled"))),
                    }
                };
                Ok(apply_parameter_updates(dsl, compile_fn, &a("updates")?, registry)?.into())
            }
            "formatValue" => format_value(
                &a("value")?,
                &a("spec")?,
                &self.options(case, "options"),
                &a("sourceForm")?,
            ),
            "unparseCall" => Ok(unparse_call(&a("call")?, &self.options(case, "options"))?.into()),
            "unparseChain" => {
                Ok(unparse_chain(&a("chain")?, &self.options(case, "options"))?.into())
            }
            "listSteps" => list_steps(&a("compiled")?, &a("options")?, registry),
            "replaceEffect" => replace_effect(
                &a("compiled")?,
                &a("stepIndex")?,
                &a("newEffectName")?,
                &a("newArgs")?,
                &a("options")?,
                registry,
            ),
            "replaceEffectUnparse" => {
                let result = replace_effect(
                    &a("compiled")?,
                    &a("stepIndex")?,
                    &a("newEffectName")?,
                    &a("newArgs")?,
                    &a("options")?,
                    registry,
                )?;
                let success = result.get("success").clone();
                let dsl = if success.is_truthy() {
                    Value::from(unparse(
                        result.get("program"),
                        &Value::Object(Object::new()),
                        &self.options(case, "unparseOptions"),
                        registry,
                    )?)
                } else {
                    Value::Undefined
                };
                let mut o = Object::new();
                o.insert("success", success);
                o.insert("error", result.get("error").clone());
                o.insert("dsl", dsl);
                Ok(Value::Object(o))
            }
            "getCompatibleReplacements" => get_compatible_replacements(
                &a("compiled")?,
                &a("stepIndex")?,
                &a("options")?,
                registry,
            ),
            "predictReplacement" => predict_replacement(
                &a("resolvedName")?,
                &a("spec")?,
                &a("newArgs")?,
                &a("oldInstance")?,
                &a("options")?,
                registry,
            ),
            "formatDslError" => {
                let error = decode_thrown(case.get("error"));
                Ok(format_dsl_error(&a("source")?, &error, &a("options")?)?.into())
            }
            "isDslSyntaxError" => Ok(is_dsl_syntax_error(&decode_thrown(case.get("error"))).into()),
            "validateEffectDefinition" => {
                let definition = case.get("definition");
                let methods: Vec<String> = match definition.get("prototypeMethods") {
                    Value::Array(m) => m
                        .iter()
                        .filter_map(|n| n.as_str().map(str::to_owned))
                        .collect(),
                    _ => Vec::new(),
                };
                let errors = match definition.get(TAG).as_str() {
                    Some("effectInstance") => {
                        let own = decode_members(definition.get("props"));
                        validate_definition(Definition::Instance {
                            own: &own,
                            prototype_methods: &methods,
                        })?
                    }
                    Some("effectSubclass") => {
                        let statics = decode_members(definition.get("statics"));
                        let statics = statics.as_object().cloned().unwrap_or_default();
                        validate_definition(Definition::Subclass {
                            prototype_methods: &methods,
                            statics: &statics,
                        })?
                    }
                    _ => validate_definition(Definition::Plain(&decode(definition)))?,
                };
                Ok(Value::Array(errors.into_iter().map(Value::from).collect()))
            }
            other => panic!("unknown op {other}"),
        }
    }
}

fn cmd_cases(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut out = None;
    let mut rust_frontend = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out = args.get(i).cloned();
            }
            "--rust-frontend" => rust_frontend = true,
            other => input = Some(other.to_owned()),
        }
        i += 1;
    }
    let (Some(input), Some(out)) = (input, out) else {
        return Err(
            "usage: nm-dsl cases <resolved.jsonl> --out <results.jsonl> [--rust-frontend]".into(),
        );
    };
    let text = fs::read_to_string(&input).map_err(|e| format!("{input}: {e}"))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = Value::from_json(lines.next().unwrap_or("{}")).map_err(|e| e.to_string())?;
    let mut registry = Registry::with_catalog();
    apply_setup(&mut registry, header.get("setup"));
    let runner = Runner {
        enums: Enums::new(&registry),
        registry: &registry,
        rust_frontend,
    };
    let mut output = String::new();
    for line in lines {
        let case = Value::from_json(line).map_err(|e| e.to_string())?;
        let mut record = Object::new();
        record.insert("id", case.get("id").clone());
        record.insert("category", case.get("category").clone());
        match runner.run(&case) {
            Ok(v) => record.insert("result", encode(&v)),
            Err(e) => record.insert("error", error_record(&e)),
        };
        output.push_str(&Value::Object(record).to_json().unwrap_or_default());
        output.push('\n');
    }
    fs::write(&out, output).map_err(|e| format!("{out}: {e}"))
}

// ---------------------------------------------------------------------------
// Single-file commands
// ---------------------------------------------------------------------------

/// A compiled program from a `.json` file or a `.dsl` source.
fn load_program(path: &str, registry: &Registry) -> Result<Value, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    if path.ends_with(".json") {
        return Value::from_json(&text).map_err(|e| format!("{path}: {e}"));
    }
    compile(&text, registry).map_err(|e| {
        format_dsl_error(&Value::from(text.as_str()), &e, &Value::Undefined)
            .unwrap_or_else(|_| e.to_string())
    })
}

/// The options `ProgramState.toDsl` passes: registry definitions and enum names
/// from the host's merged enum tree.
fn host_options() -> Value {
    Value::from_json(r#"{"getEffectDef": {"kind": "registry"}, "customFormatter": {"kind": "formatValue", "enums": {"$enums": "host"}}}"#)
        .expect("valid options")
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or_default();
    let rest = args.get(1..).unwrap_or_default();
    let positional: Vec<&String> = rest.iter().filter(|a| !a.starts_with("--")).collect();
    let flag = |name: &str| rest.iter().any(|a| a == name);
    match command {
        "cases" => cmd_cases(rest),
        "unparse" | "steps" => {
            let registry = Registry::with_catalog();
            let path = positional.first().ok_or("missing program path")?;
            let program = load_program(path, &registry)?;
            if command == "steps" {
                let steps = list_steps(&program, &Value::Undefined, &registry)
                    .map_err(|e| e.to_string())?;
                println!("{}", encode(&steps).to_json_pretty().unwrap_or_default());
                return Ok(());
            }
            let enums = Enums::new(&registry);
            let options = if flag("--host") {
                build_options(&registry, &enums, &host_options())
            } else {
                UnparseOptions::default()
            };
            let text = unparse(&program, &Value::Undefined, &options, &registry)
                .map_err(|e| e.to_string())?;
            println!("{text}");
            Ok(())
        }
        "validate-effect" => {
            let path = positional.first().ok_or("missing definition path")?;
            let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            let def = Value::from_json(&text).map_err(|e| format!("{path}: {e}"))?;
            let errors = if flag("--instance") {
                validate_definition(Definition::Instance {
                    own: &def,
                    prototype_methods: &[],
                })
            } else {
                validate_definition(Definition::Plain(&def))
            }
            .map_err(|e| e.to_string())?;
            println!(
                "{}",
                Value::Array(errors.into_iter().map(Value::from).collect())
                    .to_json_pretty()
                    .unwrap_or_default()
            );
            Ok(())
        }
        "format-error" => {
            let path = positional.first().ok_or("missing program path")?;
            let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
            match compile(&text, &Registry::with_catalog()) {
                Ok(_) => println!("ok"),
                Err(e) => println!(
                    "{}",
                    format_dsl_error(&Value::from(text.as_str()), &e, &Value::Undefined)
                        .map_err(|e| e.to_string())?
                ),
            }
            Ok(())
        }
        _ => Err("usage: nm-dsl unparse|steps|validate-effect|format-error|cases ...".into()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("nm-dsl: {message}");
            ExitCode::FAILURE
        }
    }
}
