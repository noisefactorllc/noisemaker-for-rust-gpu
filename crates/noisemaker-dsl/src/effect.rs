//! Port of `runtime/effect.js`: the `Effect` base class as data, and the
//! grouping of an effect's parameters into UI categories.
//!
//! An effect instance is plain data here (the definition object a
//! [`crate::registry::EffectEntry`] holds): [`effect_instance`] builds what
//! `new Effect(config)` builds. The lifecycle hooks the reference class
//! carries (`onInit`, `onUpdate`, `onDestroy`, `asyncInit`) are native code in
//! the port: `noisemaker_gpu::hooks` (`EffectLifecycle`, `AsyncInitEffect`)
//! defines them and `noisemaker_gpu::effects` implements the catalog's; an
//! effect without native hooks behaves as the base class's no-op hooks do.

use crate::error::JsError;
use crate::unparser::jsv::{entries, get, get_opt, object_member, to_property_key};
use crate::value::{Object, Value};

/// `DEFAULT_CATEGORY`: the category of a parameter without `ui.category`.
pub const DEFAULT_CATEGORY: &str = "general";

/// The config fields `new Effect(config)` copies onto the instance when they
/// are truthy, in the constructor's order.
pub const EFFECT_CONFIG_FIELDS: &[&str] = &[
    "name",
    "namespace",
    "func",
    "description",
    "tags",
    "globals",
    "passes",
    "textures",
    "textures3d",
    "shaders",
    "externalTexture",
    "externalMesh",
    "builtinMeshes",
    "outputTex3d",
    "outputGeo",
    "uniformLayout",
    "uniformLayouts",
    "paramAliases",
    "openCategories",
    "defaultProgram",
];

/// Config lifecycle hooks and the instance members the constructor stores
/// them in.
pub const EFFECT_CONFIG_HOOKS: &[(&str, &str)] = &[
    ("onInit", "_configOnInit"),
    ("onUpdate", "_configOnUpdate"),
    ("onDestroy", "_configOnDestroy"),
    ("asyncInit", "_configAsyncInit"),
];

/// `new Effect(config)`: the instance's own members. `state` and `uniforms`
/// start empty; the [`EFFECT_CONFIG_FIELDS`] are copied when truthy (sharing
/// the config's values), `hidden` becomes `true` when truthy, `deprecatedBy`
/// is copied when truthy, and config hooks are kept under their
/// `_config<Hook>` members.
pub fn effect_instance(config: &Object) -> Object {
    let mut instance = Object::new();
    instance.insert("state", Value::object());
    instance.insert("uniforms", Value::object());
    let read = |key: &str| object_member(config, key);
    for field in EFFECT_CONFIG_FIELDS {
        let value = read(field);
        if value.is_truthy() {
            instance.insert(*field, value);
        }
    }
    if read("hidden").is_truthy() {
        instance.insert("hidden", Value::Bool(true));
    }
    let deprecated_by = read("deprecatedBy");
    if deprecated_by.is_truthy() {
        instance.insert("deprecatedBy", deprecated_by);
    }
    for (hook, member) in EFFECT_CONFIG_HOOKS {
        let value = read(hook);
        if value.is_truthy() {
            instance.insert(*member, value);
        }
    }
    instance
}

/// `getUniformCategory(spec)`: `spec?.ui?.category || DEFAULT_CATEGORY` (the
/// category value itself, whatever its type).
pub fn get_uniform_category(spec: &Value) -> Value {
    let category = get_opt(&get_opt(spec, "ui"), "category");
    if category.is_truthy() {
        category
    } else {
        Value::from(DEFAULT_CATEGORY)
    }
}

/// Options of [`group_globals_by_category`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GroupOptions {
    /// Include parameters whose control is hidden (`ui.control === false` or
    /// `ui.hidden === true`).
    pub include_hidden: bool,
}

/// `groupGlobalsByCategory(globals, {includeHidden})`: category → the
/// `[key, spec]` pairs of its parameters. Categories come in order of first
/// occurrence with `general` first (as object keys, so integer-like category
/// names precede the others, as in the reference's result object). A
/// category named after an `Object.prototype` member throws the reference's
/// `TypeError`, as does a `null` parameter spec unless `include_hidden`.
pub fn group_globals_by_category(
    globals: &Value,
    options: GroupOptions,
) -> Result<Object, JsError> {
    let mut categories = Object::new();
    let mut order: Vec<Value> = Vec::new();
    if !globals.is_truthy() {
        return Ok(categories);
    }
    for (key, spec) in entries(globals) {
        if !options.include_hidden {
            let ui = get(&spec, "ui")?;
            let control = get_opt(&ui, "control");
            let hidden = get_opt(&ui, "hidden");
            if control == Value::Bool(false) || hidden == Value::Bool(true) {
                continue;
            }
        }
        let category = get_uniform_category(&spec);
        let slot = to_property_key(&category)?;
        if !object_member(&categories, &slot).is_truthy() {
            categories.insert(slot.clone(), Value::Array(Vec::new()));
            if category.as_str() != Some(DEFAULT_CATEGORY) {
                order.push(category.clone());
            }
        }
        match categories.get_mut(&slot) {
            Some(Value::Array(members)) => {
                members.push(Value::Array(vec![Value::from(key.as_str()), spec]));
            }
            _ => {
                return Err(JsError::type_error(
                    "categories[category].push is not a function",
                ));
            }
        }
    }
    if categories.contains_key(DEFAULT_CATEGORY) {
        order.insert(0, Value::from(DEFAULT_CATEGORY));
    }
    let mut ordered = Object::new();
    for category in &order {
        let slot = to_property_key(category)?;
        let members = categories.get(&slot).cloned().unwrap_or_default();
        ordered.insert(slot, members);
    }
    Ok(ordered)
}

/// `getCategories(globals)`: the category names of
/// [`group_globals_by_category`] (hidden parameters excluded).
pub fn get_categories(globals: &Value) -> Result<Vec<String>, JsError> {
    Ok(group_globals_by_category(globals, GroupOptions::default())?
        .keys()
        .cloned()
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_fields() {
        let config = Value::from_json(
            r#"{"func": "f", "hidden": 1, "tags": [], "description": "", "extra": 3, "onInit": "x"}"#,
        )
        .unwrap();
        let instance = effect_instance(config.as_object().unwrap());
        let keys: Vec<&str> = instance.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "state",
                "uniforms",
                "func",
                "tags",
                "hidden",
                "_configOnInit"
            ]
        );
        assert_eq!(instance.get("hidden"), Some(&Value::Bool(true)));
    }

    #[test]
    fn categories() {
        let globals = Value::from_json(
            r#"{
                "a": {"ui": {"category": "effect"}},
                "b": {},
                "c": {"ui": {"control": false}},
                "d": {"ui": {"category": 7}},
                "e": {"ui": {"category": "effect", "hidden": true}}
            }"#,
        )
        .unwrap();
        assert_eq!(
            get_categories(&globals).unwrap(),
            ["7", "general", "effect"]
        );
        let all = group_globals_by_category(
            &globals,
            GroupOptions {
                include_hidden: true,
            },
        )
        .unwrap();
        assert_eq!(
            all.get("effect").map(|v| v.as_array().map(Vec::len)),
            Some(Some(2))
        );
        let bad = Value::from_json(r#"{"a": {"ui": {"category": "toString"}}}"#).unwrap();
        assert!(get_categories(&bad).is_err());
    }
}
