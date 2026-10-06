//! `nm-api` — the candidate side of the public-API gates.
//!
//! ```text
//! nm-api cases <cases.jsonl> --out <results.jsonl>        parity/check_api.mjs
//! nm-api portable <scenarios.jsonl> --out <results.jsonl> parity/check_portable.mjs
//! ```
//!
//! `cases` runs the API cases `tools/reference-api.mjs api` wrote, in order,
//! on one registry with the catalog loaded (the reference runs them in one
//! realm): the tags and namespace API (with the parser and validator against
//! registered namespaces), the cosine palettes, the block-category constants,
//! the Effect constructor and parameter categories, the `renderer/canvas.js`
//! helpers, the effect/op/starter/enum registries, the resource allocator
//! and the renderer's manifest and effect-string queries. `portable` runs each `registerPortableEffect` scenario on a
//! fresh registry (the reference: a fresh realm) and reports every step's
//! outcome, the registry lookups of its name and the final registry state.
//! Values use the tagged JSON encoding of `noisemaker_dsl::tagged`.

use std::fs;
use std::process::ExitCode;
use std::rc::Rc;

use noisemaker_dsl::canvas::{
    clone_param_value, get_vol_geo_params, has_explicit_tex_param, has_tex_surface_param,
    is_3d_generator, is_3d_processor, is_starter_effect, is_valid_identifier, needs_input_tex3d,
    sanitize_enum_name,
};
use noisemaker_dsl::constants::{
    DEFAULT_BLOCK_CATEGORY_TYPE, STARTER_BLOCK_CATEGORIES, block_category_types_value,
    get_block_category_type, is_starter_block_category,
};
use noisemaker_dsl::effect::{
    DEFAULT_CATEGORY, GroupOptions, effect_instance, get_categories, get_uniform_category,
    group_globals_by_category,
};
use noisemaker_dsl::palettes::{palettes, sample_palette};
use noisemaker_dsl::program_state::{Console, ConsoleArg, resolve_enum_value, set_console};
use noisemaker_dsl::registry::EffectEntry;
use noisemaker_dsl::resources::{allocate_resources, analyze_liveness};
use noisemaker_dsl::strings::EffectStrings;
use noisemaker_dsl::tagged::{decode, encode, error_record};
use noisemaker_dsl::tags::{
    BUILTIN_NAMESPACE, IO_FUNCTIONS, VALID_TAGS, get_tag_definition, is_io_function, is_valid_tag,
    tag_definitions_value, validate_tags,
};
use noisemaker_dsl::unparser::jsv::{get_opt, json_stringify};
use noisemaker_dsl::validator::aliases::check_effect_alias;
use noisemaker_dsl::{JsError, Object, PHASE, Registry, Stage, VERSION, Value, run_stage};

fn strings<'a>(items: impl IntoIterator<Item = &'a str>) -> Value {
    Value::Array(items.into_iter().map(Value::from).collect())
}

/// `JSON.parse(JSON.stringify(value))`: a stage output as the reference
/// oracle serializes it.
fn plain(value: &Value) -> Value {
    json_stringify(value)
        .and_then(|text| Value::from_json(&text).ok())
        .unwrap_or(Value::Undefined)
}

/// An effect registry value as the gates compare it.
fn describe(entry: Option<&Rc<EffectEntry>>) -> Value {
    let Some(entry) = entry else {
        return Value::Undefined;
    };
    let mut o = Object::new();
    for key in ["namespace", "func", "name"] {
        o.insert(key, entry.def.get(key).clone());
    }
    Value::Object(o)
}

fn str_arg<'a>(args: &'a Object, key: &str) -> &'a str {
    args.get(key).and_then(Value::as_str).unwrap_or_default()
}

fn arg(args: &Object, key: &str) -> Value {
    args.get(key).cloned().unwrap_or_default()
}

fn constant(registry: &Registry, name: &str) -> Value {
    match name {
        "TAG_DEFINITIONS" => tag_definitions_value(),
        "VALID_TAGS" => strings(VALID_TAGS.iter().copied()),
        "NAMESPACE_DESCRIPTIONS" => registry.namespace_descriptions_value(),
        "VALID_NAMESPACES" => strings(registry.valid_namespaces()),
        "BUILTIN_NAMESPACE" => Value::from(BUILTIN_NAMESPACE),
        "IO_FUNCTIONS" => strings(IO_FUNCTIONS.iter().copied()),
        "PALETTES" => Value::Object(palettes().clone()),
        "STARTER_BLOCK_CATEGORIES" => strings(STARTER_BLOCK_CATEGORIES.iter().copied()),
        "BLOCK_CATEGORY_TYPES" => block_category_types_value(),
        "DEFAULT_BLOCK_CATEGORY_TYPE" => Value::from(DEFAULT_BLOCK_CATEGORY_TYPE),
        "DEFAULT_CATEGORY" => Value::from(DEFAULT_CATEGORY),
        "VERSION" => Value::from(VERSION),
        "PHASE" => Value::Number(f64::from(PHASE)),
        "stdEnums" => Value::Object(registry.enums.clone()),
        other => panic!("unknown constant {other}"),
    }
}

/// The definition a canvas.js helper case names: a catalog effect's, or the
/// given one.
fn helper_definition(registry: &Registry, args: &Object) -> Value {
    match args.get("catalog").and_then(Value::as_str) {
        Some(id) => registry
            .get_effect(id)
            .map(|e| e.def.clone())
            .unwrap_or_default(),
        None => arg(args, "definition"),
    }
}

/// `registerEffect(name, definition)` with a plain definition object.
fn register_plain_effect(registry: &mut Registry, name: &str, def: Value) {
    let field = |key: &str| def.get(key).as_str().unwrap_or_default().to_owned();
    let entry = EffectEntry {
        namespace: field("namespace"),
        name: field("name"),
        def,
    };
    registry.register_effect(name, Rc::new(entry));
}

/// A renderer's effect strings after `setLocale(args.locale)`.
fn locale_strings(args: &Object) -> EffectStrings {
    let mut effect_strings = EffectStrings::default();
    effect_strings.set_locale(args.get("locale").and_then(Value::as_str));
    effect_strings
}

fn string_list(args: &Object, key: &str) -> Vec<String> {
    match arg(args, key) {
        Value::Array(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn passes_arg(args: &Object) -> Vec<Value> {
    match arg(args, "passes") {
        Value::Array(passes) => passes,
        _ => Vec::new(),
    }
}

fn run_api_case(registry: &mut Registry, op: &str, args: &Object) -> Result<Value, JsError> {
    let s = |key: &str| str_arg(args, key);
    Ok(match op {
        "constant" => constant(registry, s("name")),
        "isValidTag" => Value::Bool(is_valid_tag(s("tagId"))),
        "getTagDefinition" => get_tag_definition(s("tagId")),
        "validateTags" => validate_tags(&arg(args, "tags")).to_value(),
        "isIOFunction" => Value::Bool(is_io_function(s("funcName"))),
        "isValidNamespace" => Value::Bool(registry.is_valid_namespace(s("id"))),
        "getNamespaceDescription" => registry
            .get_namespace_description(s("id"))
            .map_or(Value::Null, |d| d.to_value()),
        "registerNamespace" => registry
            .register_namespace_value(&arg(args, "id"), &arg(args, "descriptor"))?
            .to_value(),
        "unregisterNamespace" => Value::Bool(registry.unregister_namespace(s("id"))?),
        "parse" => plain(&run_stage(Stage::Ast, s("src"), registry)?),
        "compile" => plain(&run_stage(Stage::Validated, s("src"), registry)?),
        "samplePalette" => {
            let t = arg(args, "t").as_f64().unwrap_or(f64::NAN);
            let rgb = sample_palette(s("name"), t)?;
            Value::Array(rgb.iter().map(|c| Value::Number(*c)).collect())
        }
        "samplePaletteSweep" => {
            let Value::Array(ts) = arg(args, "ts") else {
                return Err(JsError::type_error("a.ts.map is not a function"));
            };
            let mut out = Vec::with_capacity(ts.len());
            for t in ts {
                let rgb = sample_palette(s("name"), t.as_f64().unwrap_or(f64::NAN))?;
                out.push(Value::Array(
                    rgb.iter().map(|c| Value::Number(*c)).collect(),
                ));
            }
            Value::Array(out)
        }
        "isStarterBlockCategory" => Value::Bool(is_starter_block_category(&arg(args, "category"))),
        "getBlockCategoryType" => get_block_category_type(&arg(args, "category")),
        "newEffect" => match arg(args, "config") {
            Value::Object(config) => Value::Object(effect_instance(&config)),
            other => panic!("newEffect takes an object config, got {other:?}"),
        },
        "getUniformCategory" => get_uniform_category(&arg(args, "spec")),
        "groupGlobalsByCategory" => {
            let options = GroupOptions {
                include_hidden: get_opt(&arg(args, "options"), "includeHidden").is_truthy(),
            };
            Value::Object(group_globals_by_category(&arg(args, "globals"), options)?)
        }
        "getCategories" => strings(
            get_categories(&arg(args, "globals"))?
                .iter()
                .map(String::as_str),
        ),
        "cloneParamValue" => clone_param_value(&arg(args, "value")),
        "isValidIdentifier" => Value::Bool(is_valid_identifier(s("name"))),
        "sanitizeEnumName" => sanitize_enum_name(s("name")).map_or(Value::Null, Value::from),
        "hasTexSurfaceParam" => {
            Value::Bool(has_tex_surface_param(&helper_definition(registry, args)))
        }
        "hasExplicitTexParam" => {
            Value::Bool(has_explicit_tex_param(&helper_definition(registry, args)))
        }
        "getVolGeoParams" => get_vol_geo_params(&helper_definition(registry, args)).to_value(),
        "needsInputTex3d" => Value::Bool(needs_input_tex3d(&helper_definition(registry, args))),
        "is3dGenerator" => Value::Bool(is_3d_generator(&helper_definition(registry, args))),
        "is3dProcessor" => Value::Bool(is_3d_processor(&helper_definition(registry, args))),
        "isStarterEffect" => Value::Bool(is_starter_effect(&helper_definition(registry, args))),
        "registerEffect" => {
            register_plain_effect(registry, s("name"), arg(args, "definition"));
            Value::Undefined
        }
        "unregisterEffect" => Value::Bool(registry.unregister_effect(s("name"))),
        "getEffect" => describe(registry.get_effect(s("name"))),
        "getAllEffects" => Value::Array(
            registry
                .all_effects()
                .iter()
                .map(|(k, e)| Value::Array(vec![Value::from(k.as_str()), describe(Some(e))]))
                .collect(),
        ),
        "registerOp" => {
            registry.register_op(s("name"), arg(args, "spec"));
            Value::Undefined
        }
        "op" => registry.ops.get(s("name")).cloned().unwrap_or_default(),
        "registerStarterOps" => {
            if let Value::Array(names) = arg(args, "names") {
                let names: Vec<&str> = names.iter().filter_map(Value::as_str).collect();
                registry.register_starter_ops(&names);
            }
            Value::Undefined
        }
        "isStarterOp" => Value::Bool(registry.is_starter_op(s("name"))),
        "mergeIntoEnums" => {
            if let Value::Object(source) = arg(args, "source") {
                registry.merge_into_enums(&source);
            }
            Value::Undefined
        }
        "getEffectsFromManifest" => {
            let include_hidden = get_opt(&arg(args, "options"), "includeHidden").is_truthy();
            strings(
                registry
                    .effects_from_manifest(s("namespace"), include_hidden)
                    .iter()
                    .map(String::as_str),
            )
        }
        "setLocale" => locale_strings(args)
            .locale()
            .map_or(Value::Null, Value::from),
        "getEffectDescriptionSweep" => {
            let effect_strings = locale_strings(args);
            Value::Array(
                string_list(args, "ids")
                    .iter()
                    .map(|id| registry.effect_description(id, &effect_strings))
                    .collect(),
            )
        }
        "localizeSweep" => {
            let effect_strings = locale_strings(args);
            // localize(id, fallback = null)
            let fallback = match arg(args, "fallback") {
                Value::Undefined => Value::Null,
                f => f,
            };
            Value::Array(
                string_list(args, "ids")
                    .iter()
                    .map(|id| effect_strings.localize(id, fallback.clone()))
                    .collect(),
            )
        }
        "analyzeLiveness" => Value::Array(
            analyze_liveness(&passes_arg(args))?
                .into_iter()
                .map(|(k, l)| {
                    let mut o = Object::new();
                    o.insert("start", Value::Number(l.start as f64));
                    o.insert("end", Value::Number(l.end as f64));
                    Value::Array(vec![Value::from(k), Value::Object(o)])
                })
                .collect(),
        ),
        "allocateResources" => Value::Array(
            allocate_resources(&passes_arg(args))?
                .iter()
                .map(|(k, v)| Value::Array(vec![Value::from(k.as_str()), v.clone()]))
                .collect(),
        ),
        other => panic!("unknown op {other}"),
    })
}

fn outcome(result: Result<Value, JsError>) -> Object {
    let mut o = Object::new();
    match result {
        Ok(v) => o.insert("result", encode(&v)),
        Err(e) => o.insert("error", error_record(&e)),
    };
    o
}

fn read_lines(path: &str) -> Result<Vec<Value>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Value::from_json(l).map_err(|e| format!("{path}: {e}")))
        .collect()
}

fn write_lines(path: &str, records: &[Value]) -> Result<(), String> {
    let mut out = String::new();
    for r in records {
        out.push_str(&r.to_json().unwrap_or_default());
        out.push('\n');
    }
    fs::write(path, out).map_err(|e| format!("{path}: {e}"))
}

fn cmd_cases(input: &str, out: &str) -> Result<(), String> {
    let mut registry = Registry::with_catalog();
    let mut records = Vec::new();
    for case in read_lines(input)? {
        let args: Object = case
            .get("args")
            .as_object()
            .map(|a| a.iter().map(|(k, v)| (k.clone(), decode(v))).collect())
            .unwrap_or_default();
        let op = case.get("op").as_str().unwrap_or_default().to_owned();
        let mut record = Object::new();
        record.insert("id", case.get("id").clone());
        record.insert("category", case.get("category").clone());
        record.assign(&outcome(run_api_case(&mut registry, &op, &args)));
        records.push(Value::Object(record));
    }
    write_lines(out, &records)
}

// ---------------------------------------------------------------------------
// Portable scenarios
// ---------------------------------------------------------------------------

/// The registry lookups of a Portable function name after a step.
fn observe(registry: &Registry, func: &str) -> Value {
    let user = format!("user.{func}");
    let enums = Value::Object(registry.enums.clone());
    let aliases: Object = registry
        .param_aliases
        .get(&user)
        .map(|a| {
            a.iter()
                .map(|(k, v)| (k.clone(), Value::from(v.as_str())))
                .collect()
        })
        .unwrap_or_default();
    let mut o = Object::new();
    o.insert("bare", describe(registry.get_effect(func)));
    o.insert("dot", describe(registry.get_effect(&user)));
    o.insert(
        "slash",
        describe(registry.get_effect(&format!("user/{func}"))),
    );
    o.insert("op", registry.ops.get(&user).cloned().unwrap_or_default());
    o.insert("enums", get_opt(&get_opt(&enums, "user"), func));
    o.insert("starter", Value::Bool(registry.is_starter_op(&user)));
    o.insert("starterBare", Value::Bool(registry.is_starter_op(func)));
    o.insert("paramAliases", Value::Object(aliases));
    o.insert(
        "effectAlias",
        check_effect_alias(registry, &user).map_or(Value::Null, Value::from),
    );
    Value::Object(o)
}

/// The registry state at the end of a scenario.
fn digest(registry: &Registry) -> Value {
    let keys: Vec<&str> = registry.ops.keys().map(String::as_str).collect();
    let effect_keys: Vec<&str> = registry.effects.keys().map(String::as_str).collect();
    let mut user_ops = Object::new();
    for k in keys.iter().filter(|k| k.starts_with("user.")) {
        user_ops.insert(*k, registry.ops.get(k).cloned().unwrap_or_default());
    }
    let param_aliases: Vec<Value> = keys
        .iter()
        .filter_map(|k| {
            let aliases = registry.param_aliases.get(*k).filter(|a| !a.is_empty())?;
            let o: Object = aliases
                .iter()
                .map(|(a, t)| (a.clone(), Value::from(t.as_str())))
                .collect();
            Some(Value::Array(vec![Value::from(*k), Value::Object(o)]))
        })
        .collect();
    let effect_aliases: Vec<Value> = keys
        .iter()
        .filter_map(|k| {
            check_effect_alias(registry, k)
                .map(|m| Value::Array(vec![Value::from(*k), Value::from(m)]))
        })
        .collect();
    let mut o = Object::new();
    o.insert(
        "effects",
        Value::Array(
            registry
                .effects
                .iter()
                .map(|(k, e)| Value::Array(vec![Value::from(k.as_str()), describe(Some(e))]))
                .collect(),
        ),
    );
    o.insert("ops", strings(keys.iter().copied()));
    o.insert("userOps", Value::Object(user_ops));
    o.insert("enums", Value::Object(registry.enums.clone()));
    o.insert(
        "starters",
        strings(
            keys.iter()
                .chain(effect_keys.iter())
                .copied()
                .filter(|n| registry.is_starter_op(n)),
        ),
    );
    o.insert("paramAliases", Value::Array(param_aliases));
    o.insert("effectAliases", Value::Array(effect_aliases));
    o.insert(
        "loaded",
        Value::Array(
            registry
                .loaded
                .iter()
                .skip(noisemaker_effects::EFFECTS.len())
                .map(|e| Value::from(e.id()))
                .collect(),
        ),
    );
    Value::Object(o)
}

fn run_scenario(scenario: &Value) -> Value {
    let mut registry = Registry::with_catalog();
    if let Value::Array(setup) = scenario.get("setup") {
        for step in setup {
            let name = step.get("name").as_str().unwrap_or_default();
            match step.get("op").as_str() {
                Some("mergeIntoEnums") => {
                    if let Value::Object(source) = decode(step.get("source")) {
                        registry.merge_into_enums(&source);
                    }
                }
                Some("registerEffect") => {
                    register_plain_effect(&mut registry, name, decode(step.get("definition")))
                }
                Some("registerOp") => registry.register_op(name, decode(step.get("spec"))),
                other => panic!("unknown setup op {other:?}"),
            }
        }
    }
    let mut outcomes = Vec::new();
    if let Value::Array(steps) = scenario.get("steps") {
        for step in steps {
            let dsl = step.get("dsl").as_str().unwrap_or_default();
            let mut record = match step.get("op").as_str() {
                Some("register") => {
                    let result = registry
                        .register_portable_effect(&decode(step.get("definition")))
                        .map(|entry| {
                            let mut o = Object::new();
                            o.insert("namespace", Value::from(entry.namespace.as_str()));
                            o.insert("name", Value::from(entry.name.as_str()));
                            o.insert("instance", entry.def.clone());
                            Value::Object(o)
                        });
                    outcome(result)
                }
                Some("compile") => {
                    outcome(run_stage(Stage::Validated, dsl, &registry).map(|v| plain(&v)))
                }
                Some("graph") => {
                    outcome(noisemaker_dsl::compiler::dump_graph(dsl, &registry).map(|v| plain(&v)))
                }
                // The renderer's enum tree after its registration (the
                // scenarios merge nothing after it).
                Some("resolveEnumValue") => outcome(Ok(resolve_enum_value(
                    step.get("path"),
                    &Value::Object(registry.enums.clone()),
                ))),
                other => panic!("unknown step {other:?}"),
            };
            if let Some(probe) = step.get("probe").as_str() {
                record.insert("observed", encode(&observe(&registry, probe)));
            }
            outcomes.push(Value::Object(record));
        }
    }
    let mut result = Object::new();
    result.insert("steps", Value::Array(outcomes));
    result.insert("state", encode(&digest(&registry)));
    Value::Object(result)
}

fn cmd_portable(input: &str, out: &str) -> Result<(), String> {
    let mut records = Vec::new();
    for scenario in read_lines(input)? {
        let mut record = Object::new();
        record.insert("id", scenario.get("id").clone());
        record.insert("category", scenario.get("category").clone());
        record.insert("result", run_scenario(&scenario));
        records.push(Value::Object(record));
    }
    write_lines(out, &records)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = || "usage: nm-api cases|portable <input.jsonl> --out <results.jsonl>".to_owned();
    let command = args.first().ok_or_else(usage)?;
    let mut input = None;
    let mut out = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--out" {
            i += 1;
            out = args.get(i).cloned();
        } else {
            input = Some(args[i].clone());
        }
        i += 1;
    }
    let (Some(input), Some(out)) = (input, out) else {
        return Err(usage());
    };
    match command.as_str() {
        "cases" => cmd_cases(&input, &out),
        "portable" => cmd_portable(&input, &out),
        _ => Err(usage()),
    }
}

/// The reference's console output goes to its oracle's ignored stderr; the
/// gates compare values, not warnings.
struct QuietConsole;

impl Console for QuietConsole {
    fn warn(&self, _args: &[ConsoleArg]) {}
    fn error(&self, _args: &[ConsoleArg]) {}
}

fn main() -> ExitCode {
    set_console(Rc::new(QuietConsole));
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("nm-api: {message}");
            ExitCode::FAILURE
        }
    }
}
