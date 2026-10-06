//! Port of `CanvasRenderer.registerPortableEffect` (`renderer/canvas.js`):
//! user-defined effects ("Portable" definitions: plain JSON with their own
//! shader sources) registered at runtime into the `user` namespace.
//!
//! A Portable definition is the JSON form of an effect definition (`func` or
//! `name`, `globals`, `passes`, `textures`, ...) with its shaders already
//! loaded under `shaders[program].wgsl` (and/or `.glsl`). Registration checks
//! the registration inputs (not shader compilation or backend support) and
//! fails with the reference's `Portable effect: ...` messages before it
//! changes any registry; then it registers the effect exactly as the host
//! registers a catalog effect, under `user.<func>` and `user/<func>`, keeping
//! any built-in effect that already owns the bare name `<func>`, merging its
//! choice enums under `user.<func>.<param>` and making `user.<func>` a
//! starter when the definition says so or none of its passes reads a
//! pipeline input. A name can be registered once per registry: the
//! reference's registries are shared within a JavaScript realm, and a
//! [`Registry`] is one realm's registries.

use std::rc::Rc;

use crate::canvas::is_valid_identifier;
use crate::effect::effect_instance;
use crate::error::JsError;
use crate::js::{is_finite_number, trim};
use crate::registry::{EffectEntry, Registry};
use crate::unparser::jsv::{entries, get_opt, values};
use crate::value::{Object, Value};

/// The keys a Portable definition may not use anywhere
/// (`Object.getOwnPropertyNames(Object.prototype)` and `prototype`): the
/// operator and enum registries are object trees, and these keys would reach
/// their prototypes.
pub const PORTABLE_RESERVED_KEYS: &[&str] = &[
    "constructor",
    "__defineGetter__",
    "__defineSetter__",
    "hasOwnProperty",
    "__lookupGetter__",
    "__lookupSetter__",
    "isPrototypeOf",
    "propertyIsEnumerable",
    "toString",
    "valueOf",
    "__proto__",
    "toLocaleString",
    "prototype",
];

/// The pass inputs that make a Portable effect a chain filter rather than a
/// starter (`pipelineInputs` of `registerPortableEffect`).
pub const PORTABLE_PIPELINE_INPUTS: &[&str] = &[
    "inputTex",
    "inputTex3d",
    "inputGeo",
    "inputXyz",
    "inputVel",
    "inputRgba",
    "src",
    "o0",
    "o1",
    "o2",
    "o3",
    "o4",
    "o5",
    "o6",
    "o7",
];

/// The namespace Portable effects are registered in.
pub const PORTABLE_NAMESPACE: &str = "user";

fn fail(message: impl AsRef<str>) -> JsError {
    JsError::error(format!("Portable effect: {}", message.as_ref()))
}

/// `value !== null && typeof value === 'object' && !Array.isArray(value)`.
fn is_record(value: &Value) -> bool {
    matches!(value, Value::Object(_))
}

/// `typeof source === 'string' && source.trim().length > 0`.
fn has_source(source: &Value) -> bool {
    matches!(source, Value::String(s) if !trim(s).is_empty())
}

/// The first reserved key of the definition tree, visited as the reference
/// visits it (a stack of objects; each object's keys in order).
fn first_reserved_key(definition: &Value) -> Option<String> {
    let mut pending: Vec<&Value> = vec![definition];
    while let Some(value) = pending.pop() {
        let children: Vec<(String, &Value)> = match value {
            Value::Object(o) => o.iter().map(|(k, v)| (k.clone(), v)).collect(),
            Value::Array(a) => a
                .iter()
                .enumerate()
                .map(|(i, v)| (i.to_string(), v))
                .collect(),
            _ => continue,
        };
        for (key, child) in children {
            if PORTABLE_RESERVED_KEYS.contains(&key.as_str()) {
                return Some(key);
            }
            if matches!(child, Value::Object(_) | Value::Array(_)) {
                pending.push(child);
            }
        }
    }
    None
}

/// `shaders[program][language]` of a validated pass.
fn shader_source(shaders: &Value, program: &str, language: &str) -> Value {
    get_opt(&get_opt(shaders, program), language)
}

/// The checks of `registerPortableEffect`, in its order, before any registry
/// changes; returns the effect's function name.
fn check_definition(registry: &Registry, definition: &Value) -> Result<String, JsError> {
    if !is_record(definition) {
        return Err(fail("expected a definition object"));
    }
    let member = |key: &str| get_opt(definition, key);
    let namespace = member("namespace");
    let passes = member("passes");
    let shaders = member("shaders");
    let globals = member("globals");
    let starter = member("starter");
    let func = match member("func") {
        Value::Undefined | Value::Null => member("name"),
        func => func,
    };
    let func = match &func {
        Value::String(f) if is_valid_identifier(f) => f.clone(),
        _ => return Err(fail("func must be a DSL identifier")),
    };
    if PORTABLE_RESERVED_KEYS.contains(&func.as_str()) {
        return Err(fail(format!("reserved func {func}")));
    }
    if let Some(key) = first_reserved_key(definition) {
        return Err(fail(format!("reserved metadata key {key}")));
    }
    if !namespace.is_undefined() && namespace.as_str() != Some(PORTABLE_NAMESPACE) {
        return Err(fail("namespace must be user"));
    }
    if !starter.is_undefined() && !matches!(starter, Value::Bool(_)) {
        return Err(fail("starter must be boolean"));
    }
    let passes = match &passes {
        Value::Array(p) if !p.is_empty() => p,
        _ => return Err(fail("passes must be a nonempty array")),
    };
    if !is_record(&shaders) {
        return Err(fail("loaded shaders are required"));
    }
    for pass in passes {
        let program = match get_opt(pass, "program") {
            Value::String(p) if is_record(pass) && !p.is_empty() => p,
            _ => return Err(fail("each pass must name a program")),
        };
        for field in ["inputs", "outputs"] {
            let bindings = get_opt(pass, field);
            if !bindings.is_undefined()
                && (!is_record(&bindings) || values(&bindings).iter().any(|v| !has_source(v)))
            {
                return Err(fail(format!(
                    "pass {field} must map names to nonempty texture references"
                )));
            }
        }
        let source = get_opt(&shaders, &program);
        if !is_record(&source)
            || ![get_opt(&source, "glsl"), get_opt(&source, "wgsl")]
                .iter()
                .any(has_source)
        {
            return Err(fail(format!("missing shader source for {program}")));
        }
    }
    let program_of = |pass: &Value| {
        get_opt(pass, "program")
            .as_str()
            .unwrap_or_default()
            .to_owned()
    };
    for language in ["glsl", "wgsl"] {
        if passes
            .iter()
            .any(|pass| has_source(&shader_source(&shaders, &program_of(pass), language)))
        {
            for pass in passes {
                let program = program_of(pass);
                if !has_source(&shader_source(&shaders, &program, language)) {
                    return Err(fail(format!(
                        "missing {language} shader source for {program}"
                    )));
                }
            }
        }
    }
    if !globals.is_undefined()
        && (!is_record(&globals) || values(&globals).iter().any(|spec| !is_record(spec)))
    {
        return Err(fail("globals must contain parameter objects"));
    }
    for (key, spec) in entries(&globals) {
        let choices = get_opt(&spec, "choices");
        if choices.is_undefined() {
            continue;
        }
        let strings = get_opt(&spec, "type").as_str() == Some("string");
        let bad_value = |value: &Value| {
            !value.is_null()
                && if strings {
                    !matches!(value, Value::String(_))
                } else {
                    !is_finite_number(value)
                }
        };
        if !is_record(&choices) || values(&choices).iter().any(bad_value) {
            return Err(fail(format!(
                "choices for {key} must map names to {} or null",
                if strings { "strings" } else { "numbers" }
            )));
        }
    }
    let aliases = member("paramAliases");
    if !aliases.is_undefined() {
        let declared = |target: &Value| match (&globals, target) {
            (Value::Object(g), Value::String(t)) => g.contains_key(t),
            _ => false,
        };
        if !is_record(&aliases) || values(&aliases).iter().any(|t| !declared(t)) {
            return Err(fail("paramAliases must map names to declared globals"));
        }
    }
    if registry
        .get_effect(&format!("{PORTABLE_NAMESPACE}.{func}"))
        .is_some()
        || registry
            .get_effect(&format!("{PORTABLE_NAMESPACE}/{func}"))
            .is_some()
    {
        return Err(fail(format!("user.{func} is already registered")));
    }
    Ok(func)
}

/// The `starter` of a validated definition: the explicit value, else whether
/// no pass reads a pipeline input.
fn infer_starter(definition: &Value) -> bool {
    if let Value::Bool(starter) = get_opt(definition, "starter") {
        return starter;
    }
    let Value::Array(passes) = get_opt(definition, "passes") else {
        return true;
    };
    !passes.iter().any(|pass| {
        values(&get_opt(pass, "inputs")).iter().any(|input| {
            input
                .as_str()
                .is_some_and(|i| PORTABLE_PIPELINE_INPUTS.contains(&i))
        })
    })
}

impl Registry {
    /// `registerPortableEffect(definition)`: validate a Portable definition
    /// (its shader sources already loaded under `shaders`) and register it as
    /// `user.<func>`. Returns the registered effect (namespace `user`, name
    /// `<func>`, its `Effect` instance with `shaders` and `starter`), which
    /// is also appended to [`Registry::loaded`].
    ///
    /// Fails with `Error: Portable effect: <reason>` and no registry change
    /// when the definition is not an object, `func` (or `name`) is not a DSL
    /// identifier or is reserved, a key anywhere in it is reserved
    /// ([`PORTABLE_RESERVED_KEYS`]), `namespace` is not `user`, `starter` is
    /// not a boolean, `passes` is empty or names no program, a pass's
    /// `inputs`/`outputs` map to empty texture references, a program has no
    /// shader source (or only some programs have a source in one language),
    /// `globals` holds a non-object, a parameter's `choices` map to values of
    /// the wrong type, `paramAliases` name undeclared parameters, or
    /// `user.<func>` is already registered.
    pub fn register_portable_effect(
        &mut self,
        definition: &Value,
    ) -> Result<Rc<EffectEntry>, JsError> {
        let func = check_definition(self, definition)?;
        let Value::Object(fields) = definition else {
            unreachable!("check_definition accepts records only")
        };
        // new Effect({ ...definition, func, namespace: 'user' })
        let mut config: Object = fields.clone();
        config.insert("func", Value::from(func.as_str()));
        config.insert("namespace", Value::from(PORTABLE_NAMESPACE));
        let mut instance = effect_instance(&config);
        instance.insert("shaders", get_opt(definition, "shaders"));
        let starter = infer_starter(definition);
        instance.insert("starter", Value::Bool(starter));
        let entry = Rc::new(EffectEntry {
            namespace: PORTABLE_NAMESPACE.to_owned(),
            name: func.clone(),
            def: Value::Object(instance),
        });
        // Portable effects belong to user.*; preserve a built-in's bare lookup.
        let previous_bare = self.get_effect(&func).cloned();
        self.register_effect_with_runtime(&entry);
        match previous_bare {
            None => {
                self.unregister_effect(&func);
            }
            Some(previous) => self.register_effect(func.clone(), previous),
        }
        if starter {
            self.register_starter_ops(&[format!("{PORTABLE_NAMESPACE}.{func}")]);
        }
        self.loaded.push(entry.clone());
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(func: &str, overrides: &str) -> Value {
        let mut def = Value::from_json(&format!(
            r##"{{
                "namespace": "user", "name": "{func}", "func": "{func}", "globals": {{}},
                "passes": [{{"name": "main", "program": "main", "inputs": {{}}, "outputs": {{"fragColor": "outputTex"}}}}],
                "shaders": {{"main": {{"wgsl": "@fragment fn main() -> @location(0) vec4<f32> {{ return vec4<f32>(1.0); }}"}}}}
            }}"##
        ))
        .unwrap();
        if let Value::Object(extra) = Value::from_json(overrides).unwrap() {
            for (k, v) in extra.iter() {
                def.set(k, v.clone());
            }
        }
        def
    }

    #[test]
    fn registers_into_user() {
        let mut reg = Registry::with_catalog();
        let entry = reg
            .register_portable_effect(&definition(
                "portableParams",
                r#"{"globals": {"mode": {"type": "int", "default": 0, "uniform": "mode", "choices": {"Modes:": -1, "Soft Light": 3}}},
                    "paramAliases": {"oldMode": "mode"}}"#,
            ))
            .unwrap();
        assert_eq!(entry.id(), "user/portableParams");
        assert!(reg.get_effect("user.portableParams").is_some());
        assert!(reg.get_effect("portableParams").is_none());
        assert!(reg.ops.contains_key("user.portableParams"));
        assert!(reg.is_starter_op("user.portableParams"));
        assert!(!reg.is_starter_op("portableParams"));
        let soft = reg
            .enums
            .get("user")
            .unwrap()
            .get("portableParams")
            .get("mode")
            .get("SoftLight");
        assert_eq!(soft.get("value").as_f64(), Some(3.0));
        assert_eq!(
            reg.param_aliases
                .get("user.portableParams")
                .and_then(|a| a.get("oldMode"))
                .map(String::as_str),
            Some("mode")
        );
        let err = reg
            .register_portable_effect(&definition("portableParams", "{}"))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Error: Portable effect: user.portableParams is already registered"
        );
    }

    #[test]
    fn keeps_bare_builtin() {
        let mut reg = Registry::with_catalog();
        let mut def = definition("noise", "{}");
        def.as_object_mut().unwrap().remove("func");
        let entry = reg.register_portable_effect(&def).unwrap();
        assert_eq!(reg.get_effect("noise").unwrap().namespace, "synth");
        assert!(Rc::ptr_eq(reg.get_effect("user.noise").unwrap(), &entry));
        assert_eq!(entry.def.get("func").as_str(), Some("noise"));
    }

    #[test]
    fn rejects_before_registering() {
        let mut reg = Registry::with_catalog();
        let before = reg.effects.len();
        for (def, message) in [
            (Value::Null, "expected a definition object"),
            (
                definition("bad-name", "{}"),
                "func must be a DSL identifier",
            ),
            (definition("valueOf", "{}"), "reserved func valueOf"),
            (
                definition("x", r#"{"namespace": "synth"}"#),
                "namespace must be user",
            ),
            (
                definition("x", r#"{"starter": "false"}"#),
                "starter must be boolean",
            ),
            (
                definition("x", r#"{"passes": []}"#),
                "passes must be a nonempty array",
            ),
            (
                definition("x", r#"{"shaders": {}}"#),
                "missing shader source for main",
            ),
            (
                definition("x", r#"{"globals": {"m": {"choices": {"toString": 1}}}}"#),
                "reserved metadata key toString",
            ),
            (
                definition("x", r#"{"globals": {"m": {"choices": {"a": "b"}}}}"#),
                "choices for m must map names to numbers or null",
            ),
            (
                definition("x", r#"{"paramAliases": {"old": "absent"}}"#),
                "paramAliases must map names to declared globals",
            ),
        ] {
            let err = reg.register_portable_effect(&def).unwrap_err();
            assert_eq!(
                err.to_string(),
                format!("Error: Portable effect: {message}")
            );
        }
        assert_eq!(reg.effects.len(), before);
    }
}
