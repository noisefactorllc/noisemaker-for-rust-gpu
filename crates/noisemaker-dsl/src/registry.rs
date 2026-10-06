//! The engine's registries: effects, operators, starter operators, enums, aliases
//! and namespaces.
//!
//! The reference keeps these as module-level singletons (`runtime/registry.js`,
//! `lang/ops.js`, `lang/enums.js`, the starter-op set in `lang/validator.js`,
//! `lang/paramAliases.js`, `lang/effectAliases.js`, `runtime/tags.js`) that its
//! host, `CanvasRenderer`, fills while it loads effects (and registers
//! Portable effects: [`Registry::register_portable_effect`], `crate::portable`). [`Registry`] holds the same
//! state as one value, and [`Registry::with_catalog`] performs the host's loading
//! sequence for every effect of the embedded catalog:
//!
//! 1. `loadManifest`: register the manifest's starter effects (bare and namespaced
//!    names), then merge the standard enums;
//! 2. per effect, in catalog order (namespace, then effect directory):
//!    `loadEffectDefinition` + `loadEffectShaders` (attach `shaders[program].wgsl`),
//!    `registerEffectWithRuntime` (effect lookup keys, the operator spec, parameter
//!    and effect aliases, choice enums), `mergeIntoEnums(choices)` and
//!    `registerStarterOpForEffect`.

use std::rc::Rc;

use indexmap::{IndexMap, IndexSet};

use crate::value::{Object, Value};

/// A registered effect: the definition instance (the reference `Effect` object,
/// as data) with its loaded shaders attached under `shaders`.
#[derive(Debug, Clone)]
pub struct EffectEntry {
    /// Namespace directory of the effect (the effect id's first segment). This is
    /// the registration namespace even when the definition omits `namespace`.
    pub namespace: String,
    /// Effect directory name (the effect id's second segment).
    pub name: String,
    /// The definition object (`instance`), including `shaders`.
    pub def: Value,
}

impl EffectEntry {
    /// `<namespace>/<name>`.
    pub fn id(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }

    /// `instance.func`.
    pub fn func(&self) -> Option<&str> {
        self.def.get("func").as_str()
    }
}

pub use crate::tags::{BUILTIN_NAMESPACES, IO_FUNCTIONS, RESERVED_FUNCTION_NAMES};

/// The engine registries.
#[derive(Debug, Clone, Default)]
pub struct Registry {
    /// `registerEffect(name, definition)`: lookup key -> effect. Each effect is
    /// registered under `func`, `ns.func`, `ns/name` and `ns.name`.
    pub effects: IndexMap<String, Rc<EffectEntry>>,
    /// Effects in load order (the host's `_loadedEffects`).
    pub loaded: Vec<Rc<EffectEntry>>,
    /// `registerOp(name, spec)` — the `ops` object of `lang/ops.js`.
    pub ops: Object,
    /// `STARTER_OPS`.
    pub starter_ops: IndexSet<String>,
    /// The merged enum tree (`mergeIntoEnums` target).
    pub enums: Object,
    /// `registerParamAliases`: op name -> { oldName: newName }.
    pub param_aliases: IndexMap<String, IndexMap<String, String>>,
    /// `registerEffectAlias`: old op name -> replacement name.
    pub effect_aliases: IndexMap<String, String>,
    /// `VALID_NAMESPACES` with descriptions, in registration order. Read it
    /// through [`Registry::valid_namespaces`] and
    /// [`Registry::namespace_descriptions`]; it changes only through
    /// [`Registry::register_namespace`] and [`Registry::unregister_namespace`]
    /// (`crate::tags`).
    pub(crate) namespaces: IndexMap<String, String>,
    /// The palette table (`share/palettes.json`), in file order.
    pub palettes: Object,
    /// The effect manifest (`shaders/effects/manifest.json`).
    pub manifest: Object,
}

pub use crate::canvas::{is_starter_effect, is_valid_identifier, sanitize_enum_name};

/// `deepMerge` of `lang/enums.js`.
pub fn deep_merge_enums(target: &mut Object, source: &Object) {
    for (key, source_val) in source.iter() {
        if key == "__proto__" || key == "constructor" || key == "prototype" {
            continue;
        }
        let recurse = matches!(source_val, Value::Object(s) if !s.contains_key("type"))
            && matches!(target.get(key), Some(Value::Object(_)));
        if recurse {
            let Value::Object(src) = source_val else {
                unreachable!()
            };
            if let Some(Value::Object(t)) = target.get_mut(key) {
                deep_merge_enums(t, src);
            }
        } else {
            target.insert(key.clone(), source_val.clone());
        }
    }
}

impl Registry {
    /// Empty registries with the built-in namespaces (no enums, no effects).
    pub fn new() -> Self {
        let mut reg = Registry::default();
        for (id, description) in BUILTIN_NAMESPACES {
            reg.namespaces
                .insert((*id).to_owned(), (*description).to_owned());
        }
        reg
    }

    /// The registries after the host loaded every effect of the embedded catalog.
    pub fn with_catalog() -> Self {
        let mut reg = Registry::new();
        let manifest = Value::from_json(noisemaker_effects::MANIFEST_JSON)
            .expect("catalog manifest.json is valid JSON");
        let palettes = Value::from_json(
            std::str::from_utf8(
                noisemaker_effects::share_file("share/palettes.json")
                    .expect("catalog ships share/palettes.json"),
            )
            .expect("palettes.json is UTF-8"),
        )
        .expect("palettes.json is valid JSON");
        reg.manifest = manifest.as_object().cloned().unwrap_or_default();
        reg.palettes = palettes.as_object().cloned().unwrap_or_default();

        // loadManifest: starter ops from the manifest, then the standard enums.
        let mut starter_names = Vec::new();
        for (effect_id, entry) in reg.manifest.iter() {
            if entry.get("starter").is_truthy() {
                let parts: Vec<&str> = effect_id.split('/').collect();
                if parts.len() == 2 {
                    starter_names.push(parts[1].to_owned());
                    starter_names.push(format!("{}.{}", parts[0], parts[1]));
                }
            }
        }
        reg.register_starter_ops(&starter_names);
        let std = reg.std_enums();
        reg.merge_into_enums(&std);

        for source in noisemaker_effects::EFFECTS {
            let def = Value::from_json(source.definition_json).unwrap_or_else(|e| {
                panic!(
                    "catalog definition {}/{} is invalid JSON: {e}",
                    source.namespace, source.name
                )
            });
            reg.load_effect(source.namespace, source.name, def, |program| {
                source.wgsl_program(program)
            });
        }
        reg
    }

    /// `stdEnums` (lang/std_enums.js), built over this registry's palette table.
    pub fn std_enums(&self) -> Object {
        fn entries(pairs: &[(&str, f64)]) -> Value {
            let mut o = Object::new();
            for (name, value) in pairs {
                let mut e = Object::new();
                e.insert("type", Value::from("Number"));
                e.insert("value", Value::Number(*value));
                o.insert(*name, Value::Object(e));
            }
            Value::Object(o)
        }
        let mut palette = Object::new();
        for (index, name) in self.palettes.keys().enumerate() {
            let mut e = Object::new();
            e.insert("type", Value::from("Number"));
            e.insert("value", Value::Number(index as f64));
            palette.insert(name.clone(), Value::Object(e));
        }
        let mut std = Object::new();
        std.insert(
            "channel",
            entries(&[("r", 0.0), ("g", 1.0), ("b", 2.0), ("a", 3.0)]),
        );
        std.insert(
            "color",
            entries(&[("mono", 0.0), ("rgb", 1.0), ("hsv", 2.0)]),
        );
        std.insert(
            "oscType",
            entries(&[
                ("sine", 0.0),
                ("linear", 1.0),
                ("sawtooth", 2.0),
                ("sawtoothInv", 3.0),
                ("square", 4.0),
                ("noise1d", 5.0),
                ("noise2d", 6.0),
            ]),
        );
        std.insert(
            "oscKind",
            entries(&[
                ("sine", 0.0),
                ("tri", 1.0),
                ("saw", 2.0),
                ("sawInv", 3.0),
                ("square", 4.0),
                ("noise", 5.0),
                ("noise1d", 5.0),
                ("noise2d", 6.0),
            ]),
        );
        std.insert(
            "midiMode",
            entries(&[
                ("noteChange", 0.0),
                ("gateNote", 1.0),
                ("gateVelocity", 2.0),
                ("triggerNote", 3.0),
                ("velocity", 4.0),
                ("cc", 5.0),
                ("cc14", 6.0),
                ("nrpn", 7.0),
                ("pitchBend", 8.0),
                ("pressure", 9.0),
                ("polyPressure", 10.0),
            ]),
        );
        std.insert("midiZone", entries(&[("lower", 0.0), ("upper", 1.0)]));
        std.insert(
            "audioBand",
            entries(&[
                ("low", 0.0),
                ("mid", 1.0),
                ("high", 2.0),
                ("vol", 3.0),
                ("raw", 4.0),
            ]),
        );
        std.insert("palette", Value::Object(palette));
        std
    }

    /// `mergeIntoEnums(source)`.
    pub fn merge_into_enums(&mut self, source: &Object) {
        deep_merge_enums(&mut self.enums, source);
    }

    /// `registerStarterOps(names)`.
    pub fn register_starter_ops<S: AsRef<str>>(&mut self, names: &[S]) {
        for name in names {
            let name = name.as_ref();
            if !name.is_empty() {
                self.starter_ops.insert(name.to_owned());
            }
        }
    }

    /// `isStarterOp(name)` (lang/validator.js).
    pub fn is_starter_op(&self, name: &str) -> bool {
        if name == "particles" || name == "render.particles" {
            return false;
        }
        if self.starter_ops.contains(name) {
            return true;
        }
        let parts: Vec<&str> = name.split('.').collect();
        if parts.len() > 1 {
            let canonical = parts[parts.len() - 1];
            if self.starter_ops.contains(canonical) {
                let suffix = format!(".{canonical}");
                return !self.starter_ops.iter().any(|op| op.ends_with(&suffix));
            }
        }
        false
    }

    /// `registerEffect(name, definition)`.
    pub fn register_effect(&mut self, key: impl Into<String>, entry: Rc<EffectEntry>) {
        self.effects.insert(key.into(), entry);
    }

    /// `getEffect(name)`.
    pub fn get_effect(&self, key: &str) -> Option<&Rc<EffectEntry>> {
        self.effects.get(key)
    }

    /// `unregisterEffect(name)`: `true` when `name` was registered. The other
    /// lookup keys keep their order.
    pub fn unregister_effect(&mut self, key: &str) -> bool {
        self.effects.shift_remove(key).is_some()
    }

    /// `getAllEffects()`: every lookup key and its effect, in registration
    /// order (a key registered again keeps its first position).
    pub fn all_effects(&self) -> &IndexMap<String, Rc<EffectEntry>> {
        &self.effects
    }

    /// `registerOp(name, spec)`.
    pub fn register_op(&mut self, name: impl Into<String>, spec: Value) {
        self.ops.insert(name, spec);
    }

    /// `registerParamAliases(opName, aliases)`.
    pub fn register_param_aliases(&mut self, op_name: &str, aliases: &Object) {
        let slot = self.param_aliases.entry(op_name.to_owned()).or_default();
        for (old, new) in aliases.iter() {
            if let Some(new) = new.as_str() {
                slot.insert(old.clone(), new.to_owned());
            }
        }
    }

    /// `registerEffectAlias(oldOpName, newName)`.
    pub fn register_effect_alias(&mut self, old_op_name: &str, new_name: &str) {
        self.effect_aliases
            .insert(old_op_name.to_owned(), new_name.to_owned());
    }

    /// The host's per-effect loading sequence (`loadEffect`): attach shaders,
    /// `registerEffectWithRuntime`, merge its choice enums, `registerStarterOpForEffect`.
    ///
    /// `wgsl` returns the WGSL source of a program of this effect, if any.
    pub fn load_effect<'a>(
        &mut self,
        namespace: &str,
        name: &str,
        mut def: Value,
        wgsl: impl Fn(&str) -> Option<&'a str>,
    ) -> Rc<EffectEntry> {
        // loadEffectShaders: one bucket per pass program; the WGSL is attached when
        // the manifest lists it. No manifest entry (or no passes) -> no shaders.
        let effect_id = format!("{namespace}/{name}");
        let manifest_entry = self.manifest.get(&effect_id).cloned();
        if let (Value::Array(passes), Some(manifest_entry)) =
            (def.get("passes").clone(), manifest_entry)
        {
            let mut shaders = match def.get("shaders") {
                Value::Object(o) => o.clone(),
                _ => Object::new(),
            };
            for pass in &passes {
                let Some(prog) = pass.get("program").as_str().filter(|p| !p.is_empty()) else {
                    continue;
                };
                if !shaders.contains_key(prog) {
                    shaders.insert(prog, Value::object());
                }
                if manifest_entry.get("wgsl").get(prog).is_truthy()
                    && let Some(source) = wgsl(prog)
                    && let Some(Value::Object(bucket)) = shaders.get_mut(prog)
                {
                    bucket.insert("wgsl", Value::from(source));
                }
            }
            def.set("shaders", Value::Object(shaders));
        }

        let entry = Rc::new(EffectEntry {
            namespace: namespace.to_owned(),
            name: name.to_owned(),
            def,
        });
        self.register_effect_with_runtime(&entry);
        self.register_starter_op_for_effect(&entry);
        self.loaded.push(entry.clone());
        entry
    }

    /// `registerEffectWithRuntime(effect)` followed by the host's
    /// `mergeIntoEnums(choicesToRegister)`.
    pub(crate) fn register_effect_with_runtime(&mut self, entry: &Rc<EffectEntry>) {
        let namespace = entry.namespace.as_str();
        let effect_name = entry.name.as_str();
        let def = &entry.def;
        let func_value = def.get("func").clone();
        let func_key = match &func_value {
            Value::String(s) => s.clone(),
            other => crate::js::value_to_property_key(other),
        };
        self.register_effect(func_key.clone(), entry.clone());
        self.register_effect(format!("{namespace}.{func_key}"), entry.clone());
        self.register_effect(format!("{namespace}/{effect_name}"), entry.clone());
        self.register_effect(format!("{namespace}.{effect_name}"), entry.clone());

        if !func_value.is_truthy() {
            return;
        }
        let func = func_key;
        let mut choices_to_register = Object::new();
        let mut args = Vec::new();
        if let Value::Object(globals) = def.get("globals") {
            for (key, spec) in globals.iter() {
                let mut enum_path = spec.get("enum").clone();
                if !enum_path.is_truthy() {
                    enum_path = spec.get("enumPath").clone();
                }
                if spec.get("choices").is_truthy() && !enum_path.is_truthy() {
                    enum_path = Value::String(format!("{namespace}.{func}.{key}"));
                    let mut members = Object::new();
                    if let Value::Object(choices) = spec.get("choices") {
                        for (choice_name, val) in choices.iter() {
                            if choice_name.ends_with(':') {
                                continue;
                            }
                            let mut e = Object::new();
                            e.insert("type", Value::from("Number"));
                            e.insert("value", val.clone());
                            members.insert(choice_name.clone(), Value::Object(e.clone()));
                            if let Some(sanitized) = sanitize_enum_name(choice_name)
                                && &sanitized != choice_name
                            {
                                members.insert(sanitized, Value::Object(e));
                            }
                        }
                    }
                    choices_to_register
                        .object_entry(namespace)
                        .object_entry(&func)
                        .insert(key.clone(), Value::Object(members));
                }
                let spec_type = spec.get("type").clone();
                let arg_type = if spec_type.as_str() == Some("vec4") {
                    Value::from("color")
                } else {
                    spec_type
                };
                let mut arg = Object::new();
                arg.insert("name", Value::from(key.as_str()));
                arg.insert("type", arg_type);
                arg.insert("default", spec.get("default").clone());
                arg.insert("enum", enum_path.clone());
                arg.insert("enumPath", enum_path);
                arg.insert("min", spec.get("min").clone());
                arg.insert("max", spec.get("max").clone());
                arg.insert("uniform", spec.get("uniform").clone());
                arg.insert("choices", spec.get("choices").clone());
                args.push(Value::Object(arg));
            }
        }
        let mut op_spec = Object::new();
        op_spec.insert("name", func_value.clone());
        op_spec.insert("args", Value::Array(args));
        let op_name = format!("{namespace}.{func}");
        self.register_op(op_name.clone(), Value::Object(op_spec));

        if let Value::Object(aliases) = def.get("paramAliases") {
            self.register_param_aliases(&op_name, aliases);
        }
        // registerEffectAlias stores the value; checkEffectAlias prints it
        // through a template literal.
        let deprecated_by = def.get("deprecatedBy");
        if def.get("hidden").is_truthy() && deprecated_by.is_truthy() {
            let replacement = crate::js::value_to_property_key(deprecated_by);
            self.register_effect_alias(&op_name, &replacement);
        }
        if !choices_to_register.is_empty() {
            self.merge_into_enums(&choices_to_register);
        }
    }

    /// `registerStarterOpForEffect(effect)`.
    fn register_starter_op_for_effect(&mut self, entry: &Rc<EffectEntry>) {
        if !is_starter_effect(&entry.def) {
            return;
        }
        let func = match entry.def.get("func") {
            Value::String(s) if !s.is_empty() => s.clone(),
            _ => entry.name.clone(),
        };
        let names = [func.clone(), format!("{}.{func}", entry.namespace)];
        self.register_starter_ops(&names);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_registration() {
        let reg = Registry::with_catalog();
        assert_eq!(reg.loaded.len(), 210);
        assert!(reg.ops.contains_key("synth.noise"));
        assert!(reg.is_starter_op("synth.noise"));
        assert!(!reg.is_starter_op("filter.blur"));
        let noise = reg.get_effect("synth.noise").unwrap();
        assert!(
            noise
                .def
                .get("shaders")
                .get("noise")
                .get("wgsl")
                .as_str()
                .is_some()
        );
        // Choice enums are merged under <namespace>.<func>.<param>.
        assert_eq!(
            reg.enums
                .get("synth")
                .unwrap()
                .get("noise")
                .get("type")
                .get("simplex")
                .get("value")
                .as_f64(),
            Some(10.0)
        );
        assert_eq!(
            reg.enums
                .get("palette")
                .unwrap()
                .get("none")
                .get("value")
                .as_f64(),
            Some(0.0)
        );
    }

    #[test]
    fn sanitize() {
        assert_eq!(
            sanitize_enum_name("Cell Scale").as_deref(),
            Some("CellScale")
        );
        assert_eq!(sanitize_enum_name("a  b c").as_deref(), Some("aBC"));
        assert_eq!(sanitize_enum_name("3d"), None);
        assert_eq!(sanitize_enum_name("x-y"), Some("xy".into()));
    }
}
