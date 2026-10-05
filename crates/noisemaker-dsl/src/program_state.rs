//! Port of `demo/shaders/lib/program-state.js`: ProgramState, the step-parameter
//! state layer between the DSL and the renderer, which the reference exports
//! publicly (`shaders/src/index.js` re-exports `ProgramState`, `Emitter` and
//! `extractEffectsFromDsl`) for the demo and downstream apps.
//!
//! ProgramState is the single source of truth for effect parameter values. It
//! loads them from DSL text ([`ProgramState::from_dsl`]), keeps them per step
//! (`step_<globalStepIndex>`), validates edits, writes them to the live
//! pipeline's pass uniforms, regenerates DSL from them ([`ProgramState::to_dsl`]),
//! edits the program's structure ([`ProgramState::insert_step`],
//! [`ProgramState::delete_step`]), keeps routing overrides and media/text
//! metadata, serializes, and emits events as state changes:
//!
//! | event | data |
//! |---|---|
//! | `change` | `{ stepKey, paramName, value, previousValue }`, or `{ type: 'routing', key, planIndex \| stepIndex, value }` |
//! | `stepchange` | `{ stepKey, values, previousValues }` (one per step of a batch) |
//! | `structurechange` | `{ structure, previousStructure }` |
//! | `reset` | `{ stepKey }` |
//! | `load` | `{ structure }` |
//! | `recompileNeeded` | `undefined` (a compile-time `define` parameter changed) |
//! | `mediachange`, `textchange` | `{ stepIndex, metadata }` |
//!
//! The renderer is a [`ProgramHost`] (the GPU crate implements it for the real
//! pipeline; [`MockHost`] is a pure-data one). Values are JavaScript values
//! ([`Value`]) because the reference's behavior depends on their JavaScript
//! types throughout (validation coerces with `parseFloat`/`parseInt`,
//! regeneration compares with `JSON.stringify`, `Map` keys distinguish `0` from
//! `"0"`).
//!
//! Where the reference throws (a renderer member that is missing, an
//! out-of-range palette value, a malformed serialized state), the method
//! returns the `JsError`, after the same partial effects. One difference is
//! inherent to the data model: a [`Value`] has no identity, so the only
//! reference-identity comparison here (`_valuesEqual(a, b)` short-circuiting on
//! `a === b` for the same object passed back to `setValue`) treats two
//! objects as distinct; vector values compare element-wise as in the
//! reference. It only decides `recompileNeeded` for `define` parameters, and
//! every `define` parameter of the catalog is an int or a boolean, so it never
//! changes an outcome there.

pub mod console;
pub mod dsl_utils;
pub mod emitter;
pub mod host;
pub mod js_map;
pub mod mock;

#[cfg(test)]
mod tests;

use std::rc::Rc;

use indexmap::IndexMap;

pub use console::{Console, ConsoleArg, StderrConsole, set_console};
pub use dsl_utils::{EffectInfo, compile, effects_to_value, extract_effects_from_dsl};
pub use emitter::{Emitter, Listener, listener};
pub use host::{
    ProgramHost, convert_parameter_for_uniform, resolve_enum_value, write_uniform_aliases,
};
pub use js_map::JsMap;
pub use mock::{MockConvert, MockHost, MockMethods, MockPipeline};

use crate::JsError;
use crate::js::{math_max, math_min, number_to_string};
use crate::palette::expand_palette_value;
use crate::registry::{EffectEntry, Registry, is_starter_effect};
use crate::unparser::jsv::{
    cannot_read, entries, entries_strict, get, has_property, iterate, json_stringify, member,
    not_a_function, object_member, set_plain, spread_into, strict_equals, to_number,
    to_property_key, to_string,
};
use crate::unparser::{CustomFormatter, EffectDefLookup, UnparseOptions, format_value, unparse};
use crate::value::{Object, Value};

use dsl_utils::is_automation_config;
use host::{
    assign_member, hex_to_rgb, in_object, key_of, parse_digits, parse_float_value, parse_int_value,
    pass_node_digits, step_key_digits,
};

/// The state of one step (`StepState` of the reference).
#[derive(Debug, Clone)]
pub struct StepState {
    /// Effect identifier (`"synth.noise"`), `""` for a step created by
    /// `setValue` on an unknown key.
    pub effect_key: String,
    /// `getEffect(effectKey)`: the registered effect, if any.
    pub effect_def: Option<Rc<EffectEntry>>,
    /// The step's global index.
    pub step_index: f64,
    /// Parameter values (automation bindings included).
    pub values: Object,
}

impl StepState {
    /// `stepState.effectDef` as the reference reads it (`undefined` when none).
    fn def(&self) -> Value {
        match &self.effect_def {
            Some(entry) => entry.def.clone(),
            None => Value::Undefined,
        }
    }

    /// `stepState.effectDef?.globals` (an own member of the definition; no
    /// `Object.prototype` member is named `globals`).
    fn globals(&self) -> &Value {
        static UNDEFINED: Value = Value::Undefined;
        match &self.effect_def {
            Some(entry) => entry.def.get("globals"),
            None => &UNDEFINED,
        }
    }
}

/// A queued `change` of a batch.
#[derive(Debug, Clone, PartialEq)]
pub struct Change {
    pub step_key: String,
    pub param_name: String,
    pub value: Value,
    pub previous_value: Value,
}

impl Change {
    /// `{ stepKey, paramName, value, previousValue }`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("stepKey", Value::from(self.step_key.as_str()));
        o.insert("paramName", Value::from(self.param_name.as_str()));
        o.insert("value", self.value.clone());
        o.insert("previousValue", self.previous_value.clone());
        Value::Object(o)
    }
}

/// The result of [`ProgramState::delete_step`].
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteStepResult {
    pub success: bool,
    /// The regenerated program (success).
    pub new_dsl: Option<String>,
    /// The surface the removed chain wrote to, or `null` (success).
    pub deleted_surface_name: Value,
    /// Why the step was not deleted (failure).
    pub error: Option<String>,
}

impl DeleteStepResult {
    fn failure(error: impl Into<String>) -> Self {
        DeleteStepResult {
            success: false,
            new_dsl: None,
            deleted_surface_name: Value::Null,
            error: Some(error.into()),
        }
    }

    /// `{ success: true, newDsl, deletedSurfaceName }` or `{ success: false, error }`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("success", Value::Bool(self.success));
        if self.success {
            o.insert(
                "newDsl",
                Value::from(self.new_dsl.clone().unwrap_or_default()),
            );
            o.insert("deletedSurfaceName", self.deleted_surface_name.clone());
        } else {
            o.insert("error", Value::from(self.error.clone().unwrap_or_default()));
        }
        Value::Object(o)
    }
}

/// The result of [`ProgramState::insert_step`].
#[derive(Debug, Clone, PartialEq)]
pub struct InsertStepResult {
    pub success: bool,
    /// The regenerated program (success).
    pub new_dsl: Option<String>,
    /// The inserted step's global index (success).
    pub new_step_index: Option<f64>,
    /// Why the step was not inserted (failure).
    pub error: Option<String>,
}

impl InsertStepResult {
    fn failure(error: impl Into<String>) -> Self {
        InsertStepResult {
            success: false,
            new_dsl: None,
            new_step_index: None,
            error: Some(error.into()),
        }
    }

    /// `{ success: true, newDsl, newStepIndex }` or `{ success: false, error }`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("success", Value::Bool(self.success));
        if self.success {
            o.insert(
                "newDsl",
                Value::from(self.new_dsl.clone().unwrap_or_default()),
            );
            o.insert(
                "newStepIndex",
                Value::Number(self.new_step_index.unwrap_or(f64::NAN)),
            );
        } else {
            o.insert("error", Value::from(self.error.clone().unwrap_or_default()));
        }
        Value::Object(o)
    }
}

/// `value && typeof value === 'object' && value._varRef`: an automation
/// binding to a variable (`{ _varRef, value }`).
fn is_var_ref_binding(value: &Value) -> bool {
    matches!(value, Value::Object(_) | Value::Array(_)) && member(value, "_varRef").is_truthy()
}

/// The automation check of `_applyToPipeline`: a variable binding or an
/// oscillator/MIDI/audio config.
fn is_automation_controlled(value: &Value) -> bool {
    is_var_ref_binding(value) || is_automation_config(value)
}

/// `_cloneValue(value)`: arrays and objects copied one level deep (an object
/// copy keeps its enumerable members only, as a spread does).
fn clone_value(value: &Value) -> Value {
    match value {
        Value::Object(o) => {
            let mut copy = Object::new();
            copy.assign(o);
            Value::Object(copy)
        }
        other => other.clone(),
    }
}

/// `{ ...obj }`.
fn spread_object(obj: &Object) -> Object {
    let mut copy = Object::new();
    copy.assign(obj);
    copy
}

/// `x === undefined ? undefined : ...`: `spec.min`/`spec.max` clamping of
/// `_validateValue` (`Math.max(spec.min, value)` then `Math.min(spec.max, value)`).
fn clamp(spec: &Value, value: Value) -> Result<Value, JsError> {
    let mut value = value;
    let min = member(spec, "min");
    if !min.is_undefined() {
        let a = to_number(&min)?;
        let b = to_number(&value)?;
        value = Value::Number(math_max(a, b));
    }
    let max = member(spec, "max");
    if !max.is_undefined() {
        let a = to_number(&max)?;
        let b = to_number(&value)?;
        value = Value::Number(math_min(a, b));
    }
    Ok(value)
}

/// `spec.default || fallback`.
fn default_or(spec: &Value, fallback: Vec<f64>) -> Value {
    let d = member(spec, "default");
    if d.is_truthy() {
        d
    } else {
        Value::Array(fallback.into_iter().map(Value::Number).collect())
    }
}

/// `value.slice(0, n).map(v => parseFloat(v) || 0)`.
fn parse_components(items: &[Value], n: usize) -> Result<Value, JsError> {
    let mut out = Vec::with_capacity(n.min(items.len()));
    for v in items.iter().take(n) {
        let f = parse_float_value(v)?;
        out.push(Value::Number(if f.is_nan() || f == 0.0 { 0.0 } else { f }));
    }
    Ok(Value::Array(out))
}

/// `_validateValue(value, spec)`: coerce a value to its parameter type.
fn validate_value(value: Value, spec: &Value) -> Result<Value, JsError> {
    if !spec.is_truthy() {
        return Ok(value);
    }
    // Preserve automation configs (oscillator, midi, audio) untouched
    if is_automation_config(&value) {
        return Ok(value);
    }
    match member(spec, "type").as_str() {
        Some("float") => {
            let mut v = Value::Number(parse_float_value(&value)?);
            if v.as_f64().is_some_and(f64::is_nan) {
                let d = member(spec, "default");
                v = if d.is_nullish() {
                    Value::Number(0.0)
                } else {
                    d
                };
            }
            clamp(spec, v)
        }
        Some("int") => {
            let mut v = Value::Number(parse_int_value(&value, 10)?);
            if v.as_f64().is_some_and(f64::is_nan) {
                let d = member(spec, "default");
                v = if d.is_nullish() {
                    Value::Number(0.0)
                } else {
                    d
                };
            }
            clamp(spec, v)
        }
        Some("boolean") => Ok(Value::Bool(value.is_truthy())),
        Some("vec2") => match &value {
            Value::Array(items) => parse_components(items, 2),
            _ => Ok(default_or(spec, vec![0.0, 0.0])),
        },
        Some("vec3") => match &value {
            Value::Array(items) => parse_components(items, 3),
            _ => Ok(default_or(spec, vec![0.0, 0.0, 0.0])),
        },
        Some("vec4") => match &value {
            Value::Array(items) => parse_components(items, 4),
            _ => Ok(default_or(spec, vec![0.0, 0.0, 0.0, 0.0])),
        },
        Some("color") => match &value {
            // Color type: ensure value is a vec3 array
            Value::Array(items) => parse_components(items, 3),
            // Handle hex strings (from UI) by converting to array
            Value::String(s) if s.starts_with('#') => Ok(hex_to_rgb(s)),
            _ => Ok(default_or(spec, vec![0.0, 0.0, 0.0])),
        },
        _ => Ok(value),
    }
}

/// `_valuesEqual(a, b)`: `===`, or element-wise `===` of two arrays.
fn values_equal(a: &Value, b: &Value) -> bool {
    if strict_equals(a, b) {
        return true;
    }
    if let (Value::Array(x), Value::Array(y)) = (a, b) {
        return x.len() == y.len() && x.iter().zip(y).all(|(p, q)| strict_equals(p, q));
    }
    false
}

/// `_structuresMatch(a, b)`: same effects in the same order.
fn structures_match(a: &[EffectInfo], b: &[EffectInfo]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.effect_key == y.effect_key)
}

/// `this._structure[stepIndex]`.
fn structure_at(structure: &[EffectInfo], index: f64) -> Option<&EffectInfo> {
    if index >= 0.0 && index.fract() == 0.0 && index < structure.len() as f64 {
        structure.get(index as usize)
    } else {
        None
    }
}

/// `_createEmptyStepState(stepKey)`.
fn create_empty_step_state(step_key: &str) -> StepState {
    let step_index = step_key_digits(step_key).map_or(0.0, parse_digits);
    StepState {
        effect_key: String::new(),
        effect_def: None,
        step_index,
        values: Object::new(),
    }
}

/// `typeof plan.write === 'object' ? plan.write.name : plan.write`.
fn surface_name(write: &Value) -> Value {
    match write {
        Value::Object(_) | Value::Array(_) => member(write, "name"),
        other => other.clone(),
    }
}

/// `{ kind: target.startsWith('o') ? 'output' : 'feedback', name: target }`.
fn surface_ref(target: &Value) -> Result<Value, JsError> {
    let is_output = match target {
        Value::String(s) => s.starts_with('o'),
        Value::Undefined | Value::Null => return Err(cannot_read(target, "startsWith")),
        _ => return Err(not_a_function("target.startsWith")),
    };
    let mut r = Object::new();
    r.insert(
        "kind",
        Value::from(if is_output { "output" } else { "feedback" }),
    );
    r.insert("name", target.clone());
    Ok(Value::Object(r))
}

/// `{ kind, name }`.
fn kind_ref(kind: &str, name: &Value) -> Value {
    let mut r = Object::new();
    r.insert("kind", Value::from(kind));
    r.insert("name", name.clone());
    Value::Object(r)
}

/// `container[key]`, mutably, for the objects and arrays a compiled program
/// nests (`None` when absent or not a container).
fn child_mut<'a>(container: &'a mut Value, key: &str) -> Option<&'a mut Value> {
    match container {
        Value::Object(o) => o.get_mut(key),
        Value::Array(items) => {
            if crate::value::is_array_index(key) {
                key.parse::<usize>().ok().and_then(|i| items.get_mut(i))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `container[key] = value` where `container` is `parent[parent_key]`, read
/// as the reference reads it (`parent[parent_key].key = value`).
fn assign_in(parent: &mut Value, parent_key: &str, key: &str, value: Value) -> Result<(), JsError> {
    match child_mut(parent, parent_key) {
        Some(target) => assign_member(target, key, value),
        None => {
            // Missing members read as undefined (assignment throws); inherited
            // ones (prototype methods, `length`) as what they are.
            let mut target = member(parent, parent_key);
            assign_member(&mut target, key, value)
        }
    }
}

/// The search namespaces of `currentDsl.match(/^search\s+(\S.*?)$/m)`,
/// `split(/\s*,\s*/)`, or `None` when no line starts a search directive.
fn search_namespaces(src: &str) -> Option<Vec<String>> {
    fn is_line_terminator(c: char) -> bool {
        matches!(c, '\n' | '\r' | '\u{2028}' | '\u{2029}')
    }
    fn is_space(c: char) -> bool {
        matches!(
            c,
            '\u{9}'
                | '\u{a}'
                | '\u{b}'
                | '\u{c}'
                | '\u{d}'
                | ' '
                | '\u{a0}'
                | '\u{1680}'
                | '\u{2000}'
                ..='\u{200a}'
                    | '\u{2028}'
                    | '\u{2029}'
                    | '\u{202f}'
                    | '\u{205f}'
                    | '\u{3000}'
                    | '\u{feff}'
        )
    }
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let word: Vec<char> = "search".chars().collect();
    let mut capture: Option<String> = None;
    for start in 0..n {
        // ^ (multiline): the input start or just after a line terminator.
        if start > 0 && !is_line_terminator(chars[start - 1]) {
            continue;
        }
        if start + word.len() > n || chars[start..start + word.len()] != word[..] {
            continue;
        }
        // \s+ (greedy), then \S: the first non-space character after a
        // non-empty whitespace run.
        let mut i = start + word.len();
        let ws = i;
        while i < n && is_space(chars[i]) {
            i += 1;
        }
        if i == ws || i >= n {
            continue;
        }
        // (\S.*?)$: lazily up to the first line end ('.' never crosses one).
        let mut end = i + 1;
        while end < n && !is_line_terminator(chars[end]) {
            end += 1;
        }
        capture = Some(chars[i..end].iter().collect());
        break;
    }
    let capture: Vec<char> = capture?.chars().collect();
    // .split(/\s*,\s*/)
    let mut parts = Vec::new();
    let mut last = 0;
    let mut i = 0;
    while i < capture.len() {
        let mut j = i;
        while j < capture.len() && is_space(capture[j]) {
            j += 1;
        }
        if j < capture.len() && capture[j] == ',' {
            let mut k = j + 1;
            while k < capture.len() && is_space(capture[k]) {
                k += 1;
            }
            parts.push(capture[last..i].iter().collect());
            last = k;
            i = k;
            continue;
        }
        i += 1;
    }
    parts.push(capture[last..].iter().collect());
    Some(parts)
}

/// `getEffect(name)` (only string keys can match the registry's string keys);
/// a falsy registered value reads as a miss for the `def ||` chains.
fn get_effect_value(registry: &Registry, name: &Value) -> Option<Value> {
    match name {
        Value::String(n) => registry
            .get_effect(n)
            .map(|e| e.def.clone())
            .filter(Value::is_truthy),
        _ => None,
    }
}

/// `getEffect(key)` for a string key.
fn get_effect_str(registry: &Registry, key: &str) -> Option<Value> {
    registry
        .get_effect(key)
        .map(|e| e.def.clone())
        .filter(Value::is_truthy)
}

/// The `getEffectDefCallback` of `toDsl`: direct lookup, then `ns/name` for a
/// dotted name, then the namespace forms.
fn to_dsl_effect_def(registry: &Registry) -> EffectDefLookup<'_> {
    Rc::new(move |name: &Value, namespace: &Value| {
        // Try direct lookup first
        if let Some(def) = get_effect_value(registry, name) {
            return Ok(def);
        }
        let effect_name = match name {
            Value::String(n) => n.as_str(),
            Value::Undefined | Value::Null => return Err(cannot_read(name, "includes")),
            _ => return Err(not_a_function("effectName.includes")),
        };
        // Try with "/" instead of "." (e.g., "filter/grade")
        if effect_name.contains('.')
            && let Some(def) = get_effect_str(registry, &effect_name.replacen('.', "/", 1))
        {
            return Ok(def);
        }
        // If namespace provided separately, try combining
        if namespace.is_truthy() {
            let ns = to_string(namespace)?;
            if let Some(def) = get_effect_str(registry, &format!("{ns}/{effect_name}"))
                .or_else(|| get_effect_str(registry, &format!("{ns}.{effect_name}")))
            {
                return Ok(def);
            }
        }
        Ok(Value::Null)
    })
}

/// The `getEffectDef` of `deleteStep`/`insertStep`'s regeneration.
fn edit_effect_def(registry: &Registry) -> EffectDefLookup<'_> {
    Rc::new(move |name: &Value, namespace: &Value| {
        let def = get_effect_value(registry, name);
        if def.is_none() && namespace.is_truthy() {
            let ns = to_string(namespace)?;
            let n = to_string(name)?;
            if let Some(def) = get_effect_str(registry, &format!("{ns}/{n}"))
                .or_else(|| get_effect_str(registry, &format!("{ns}.{n}")))
            {
                return Ok(def);
            }
            return Ok(Value::Undefined);
        }
        Ok(def.unwrap_or(Value::Undefined))
    })
}

/// `err.message` of a caught error.
fn error_message(err: &JsError) -> Result<String, JsError> {
    match err {
        JsError::Error { message, .. } => Ok(message.clone()),
        JsError::Thrown(v) => to_string(&member(v, "message")),
    }
}

/// ProgramState: the effect parameter state of a program.
///
/// `H` is the renderer ([`ProgramHost`]); the default boxes any host.
pub struct ProgramState<H: ProgramHost = Box<dyn ProgramHost>> {
    renderer: Option<H>,
    registry: Rc<Registry>,
    emitter: Emitter<ProgramState<H>>,

    step_states: IndexMap<String, StepState>,
    structure: Vec<EffectInfo>,

    write_target_overrides: JsMap<Value>,
    write_step_target_overrides: JsMap<Value>,
    read_source_overrides: JsMap<Value>,
    read3d_vol_overrides: JsMap<Value>,
    read3d_geo_overrides: JsMap<Value>,
    write3d_vol_overrides: JsMap<Value>,
    write3d_geo_overrides: JsMap<Value>,
    render_target_override: Value,

    media_inputs: JsMap<Value>,
    text_inputs: JsMap<Value>,

    compiled: Option<Value>,

    batch_depth: usize,
    batched_changes: Vec<Change>,
    recompile_pending: bool,
}

impl<H: ProgramHost> std::fmt::Debug for ProgramState<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProgramState")
            .field("has_renderer", &self.renderer.is_some())
            .field("step_states", &self.step_states)
            .field("structure", &self.structure)
            .field("listeners", &self.emitter)
            .finish_non_exhaustive()
    }
}

impl<H: ProgramHost> ProgramState<H> {
    /// `new ProgramState({ renderer })` without a renderer. `registry` provides
    /// the effect definitions (`getEffect`) and the DSL frontend's registries.
    pub fn new(registry: Rc<Registry>) -> Self {
        ProgramState {
            renderer: None,
            registry,
            emitter: Emitter::new(),
            step_states: IndexMap::new(),
            structure: Vec::new(),
            write_target_overrides: JsMap::new(),
            write_step_target_overrides: JsMap::new(),
            read_source_overrides: JsMap::new(),
            read3d_vol_overrides: JsMap::new(),
            read3d_geo_overrides: JsMap::new(),
            write3d_vol_overrides: JsMap::new(),
            write3d_geo_overrides: JsMap::new(),
            render_target_override: Value::Null,
            media_inputs: JsMap::new(),
            text_inputs: JsMap::new(),
            compiled: None,
            batch_depth: 0,
            batched_changes: Vec::new(),
            recompile_pending: false,
        }
    }

    /// `new ProgramState({ renderer })`.
    pub fn with_renderer(registry: Rc<Registry>, renderer: H) -> Self {
        let mut state = Self::new(registry);
        state.renderer = Some(renderer);
        state
    }

    /// The registry the state resolves effects in.
    pub fn registry(&self) -> &Rc<Registry> {
        &self.registry
    }

    /// Replace the registry (after the host registered more effects).
    pub fn set_registry(&mut self, registry: Rc<Registry>) {
        self.registry = registry;
    }

    // =====================================================================
    // Events (Emitter)
    // =====================================================================

    /// `on(event, callback)`.
    pub fn on(&self, event: &str, callback: Listener<Self>) {
        self.emitter.on(event, callback);
    }

    /// `off(event, callback)`.
    pub fn off(&self, event: &str, callback: &Listener<Self>) {
        self.emitter.off(event, callback);
    }

    /// `removeAllListeners(event)`.
    pub fn remove_all_listeners(&self, event: Option<&str>) {
        self.emitter.remove_all_listeners(event);
    }

    /// `emit(event, data)`.
    pub fn emit(&mut self, event: &str, data: &Value) {
        let emitter = self.emitter.share();
        emitter.emit(self, event, data);
    }

    /// The emitter (listener table) of this state.
    pub fn emitter(&self) -> &Emitter<Self> {
        &self.emitter
    }

    // =====================================================================
    // Parameter Access Methods
    // =====================================================================

    /// `getValue(stepKey, paramName)`: a parameter value (automation bindings
    /// unwrapped to their value), `undefined` when unknown.
    pub fn get_value(&self, step_key: &str, param_name: &str) -> Value {
        let Some(state) = self.step_states.get(step_key) else {
            return Value::Undefined;
        };
        let value = object_member(&state.values, param_name);
        // Unwrap automation bindings (oscillator, midi, audio) to return actual value
        if is_var_ref_binding(&value) {
            return member(&value, "value");
        }
        value
    }

    /// `setValue(stepKey, paramName, value)`: validate, store (keeping an
    /// automation binding), apply to the pipeline and emit `change` (queued in
    /// a batch). A changed compile-time `define` parameter flags
    /// `recompileNeeded`.
    pub fn set_value(
        &mut self,
        step_key: &str,
        param_name: &str,
        value: Value,
    ) -> Result<(), JsError> {
        if !self.step_states.contains_key(step_key) {
            let state = create_empty_step_state(step_key);
            self.step_states.insert(step_key.to_owned(), state);
        }

        let previous_value = self.get_value(step_key, param_name);

        // Validate and coerce value
        let state = &self.step_states[step_key];
        let spec = member(state.globals(), param_name);
        let mut value = value;
        if spec.is_truthy() {
            value = validate_value(value, &spec)?;
        }

        // Preserve automation binding (_varRef) if present (for oscillator, midi, audio)
        let current_value = object_member(&state.values, param_name);
        let state = self
            .step_states
            .get_mut(step_key)
            .expect("the step state exists");
        if is_var_ref_binding(&current_value) {
            let mut binding = Object::new();
            spread_into(&mut binding, &current_value);
            binding.insert("value", value.clone());
            set_plain(&mut state.values, param_name, Value::Object(binding));
        } else {
            set_plain(&mut state.values, param_name, value.clone());
        }

        // Compile-time defines are baked into the shader source by the
        // expander: flag the recompile (coalesced across a batch) when such a
        // param actually changes value.
        if member(&spec, "define").is_truthy() && !values_equal(&previous_value, &value) {
            self.recompile_pending = true;
        }

        // Apply to pipeline immediately
        self.apply_to_pipeline_internal()?;

        // Emit change event (or batch)
        self.emit_change(Change {
            step_key: step_key.to_owned(),
            param_name: param_name.to_owned(),
            value,
            previous_value,
        });
        Ok(())
    }

    /// `getStepValues(stepKey)`: every parameter value of a step (automation
    /// bindings unwrapped).
    pub fn get_step_values(&self, step_key: &str) -> Object {
        let mut result = Object::new();
        let Some(state) = self.step_states.get(step_key) else {
            return result;
        };
        for (key, value) in state.values.iter() {
            if is_var_ref_binding(value) {
                set_plain(&mut result, key, member(value, "value"));
            } else {
                set_plain(&mut result, key, value.clone());
            }
        }
        result
    }

    /// `setStepValues(stepKey, values)`: `setValue` for each member of `values`,
    /// batched into one `stepchange`.
    pub fn set_step_values(&mut self, step_key: &str, values: &Value) -> Result<(), JsError> {
        self.batch(|state| {
            for (param_name, value) in entries_strict(values)? {
                state.set_value(step_key, &param_name, value)?;
            }
            Ok(())
        })
    }

    // =====================================================================
    // Batching Support
    // =====================================================================

    /// `batch(fn)`: run `f` with `change` events queued; when the outermost
    /// batch ends (also when `f` fails), the queued changes are emitted as one
    /// `stepchange` per step, then a pending `recompileNeeded`.
    pub fn batch<F>(&mut self, f: F) -> Result<(), JsError>
    where
        F: FnOnce(&mut Self) -> Result<(), JsError>,
    {
        self.batch_depth += 1;
        let result = f(self);
        self.batch_depth -= 1;
        if self.batch_depth == 0 && !self.batched_changes.is_empty() {
            self.flush_batched_changes();
        }
        result
    }

    /// `_emitChange(change)`.
    fn emit_change(&mut self, change: Change) {
        if self.batch_depth > 0 {
            self.batched_changes.push(change);
        } else {
            self.emit("change", &change.to_value());
            self.flush_pending_recompile();
        }
    }

    /// `_flushBatchedChanges()`.
    fn flush_batched_changes(&mut self) {
        // Group by stepKey
        let mut by_step: IndexMap<String, (Object, Object)> = IndexMap::new();
        for change in &self.batched_changes {
            let group = by_step
                .entry(change.step_key.clone())
                .or_insert_with(|| (Object::new(), Object::new()));
            set_plain(&mut group.0, &change.param_name, change.value.clone());
            set_plain(
                &mut group.1,
                &change.param_name,
                change.previous_value.clone(),
            );
        }

        // Emit stepchange for each affected step
        for (step_key, (values, previous_values)) in by_step {
            let mut data = Object::new();
            data.insert("stepKey", Value::from(step_key));
            data.insert("values", Value::Object(values));
            data.insert("previousValues", Value::Object(previous_values));
            self.emit("stepchange", &Value::Object(data));
        }

        self.batched_changes.clear();

        // A single recompileNeeded after the batch's stepchange events.
        self.flush_pending_recompile();
    }

    /// `_flushPendingRecompile()`.
    fn flush_pending_recompile(&mut self) {
        if !self.recompile_pending {
            return;
        }
        self.recompile_pending = false;
        self.emit("recompileNeeded", &Value::Undefined);
    }

    // =====================================================================
    // DSL Synchronization
    // =====================================================================

    /// `fromDsl(dslText)`: load the program's structure and values. A changed
    /// structure rebuilds the step states (keeping, per effect occurrence, the
    /// values the DSL does not specify) and emits `structurechange`; an
    /// unchanged one takes the DSL's argument values. Then applies to the
    /// pipeline and emits `load`.
    pub fn from_dsl(&mut self, dsl_text: &str) -> Result<(), JsError> {
        self.load_dsl(Some(dsl_text))
    }

    /// `fromDsl(value)` for any value (`deserialize` passes `data.dsl` as it
    /// is). A non-string program has no effects (`extractEffectsFromDsl`
    /// returns `[]` for it) and does not compile: the reference's `compile`
    /// reads it through `length` and indexing and throws (no `search`
    /// directive for numbers, booleans and objects; malformed tokens for
    /// arrays, even an array of the program's characters).
    pub fn from_dsl_value(&mut self, dsl: &Value) -> Result<(), JsError> {
        match dsl {
            Value::String(s) => self.load_dsl(Some(s)),
            _ => self.load_dsl(None),
        }
    }

    fn load_dsl(&mut self, dsl_text: Option<&str>) -> Result<(), JsError> {
        let previous_structure = self.structure.clone();

        // Cache compiled program for getCompiled()
        self.compiled = dsl_text.and_then(|src| compile(src, &self.registry).ok());

        // Parse DSL to extract effects (an empty array is valid: it clears the
        // structure). `if (!effects)` never holds: the extraction always
        // returns an array.
        let effects = match dsl_text {
            Some(src) => extract_effects_from_dsl(src, &self.registry),
            None => Vec::new(),
        };

        // Check if structure changed
        let structure_changed = !structures_match(&previous_structure, &effects);

        // Preserve values by occurrence for structure changes
        let preserved_values = self.preserve_values_by_occurrence();

        // Update structure
        self.structure = effects.clone();

        // Rebuild step states
        if structure_changed {
            self.rebuild_step_states(&effects, &preserved_values)?;
            let mut data = Object::new();
            data.insert("structure", effects_to_value(&effects));
            data.insert("previousStructure", effects_to_value(&previous_structure));
            self.emit("structurechange", &Value::Object(data));
        } else {
            // Just update values from DSL args
            self.update_values_from_dsl(&effects);
        }

        // Apply to pipeline
        self.apply_to_pipeline_internal()?;

        let mut data = Object::new();
        data.insert("structure", effects_to_value(&effects));
        self.emit("load", &Value::Object(data));
        Ok(())
    }

    /// `toDsl()`: the renderer's current program regenerated with this state's
    /// parameter values and routing overrides (`""` without a renderer program;
    /// the current program unchanged, with a console warning, when it cannot be
    /// regenerated).
    pub fn to_dsl(&self) -> String {
        let current = match &self.renderer {
            Some(r) => r.current_dsl(),
            None => String::new(),
        };
        if current.is_empty() {
            return String::new();
        }
        match self.to_dsl_inner(&current) {
            Ok(dsl) => dsl,
            Err(err) => {
                console::warn(&["[ProgramState] Failed to generate DSL:".into(), err.into()]);
                current
            }
        }
    }

    fn to_dsl_inner(&self, current: &str) -> Result<String, JsError> {
        let mut compiled = compile(current, &self.registry)?;
        if !member(&compiled, "plans").is_truthy() {
            return Ok(current.to_owned());
        }

        // Apply parameter overrides from state
        let overrides = self.build_parameter_overrides()?;

        // Apply routing overrides
        self.apply_routing_overrides_to_compiled(&mut compiled)?;

        // Unparse back to DSL text with custom formatter for arrays/colors/etc.
        let renderer_enums = match &self.renderer {
            Some(r) => r.enums(),
            None => Rc::new(Value::Undefined),
        };
        let enums = if renderer_enums.is_truthy() {
            renderer_enums
        } else {
            Rc::new(Value::Object(Object::new()))
        };
        let format_options = UnparseOptions {
            enums,
            ..UnparseOptions::default()
        };
        let custom_formatter: CustomFormatter<'_> = Rc::new(move |value: &Value, spec: &Value| {
            format_value(value, spec, &format_options, &Value::Undefined)
        });
        let options = UnparseOptions {
            custom_formatter: Some(custom_formatter),
            get_effect_def: Some(to_dsl_effect_def(&self.registry)),
            ..UnparseOptions::default()
        };
        unparse(&compiled, &overrides, &options, &self.registry)
    }

    /// `wouldChangeStructure(dslText)`: whether loading `dsl_text` would change
    /// the effect structure.
    pub fn would_change_structure(&self, dsl_text: &str) -> bool {
        let effects = extract_effects_from_dsl(dsl_text, &self.registry);
        !structures_match(&self.structure, &effects)
    }

    // =====================================================================
    // Step Operations
    // =====================================================================

    /// `resetStep(stepKey)`: restore a step's defaults (keeping its skip flag),
    /// apply, emit `reset`.
    pub fn reset_step(&mut self, step_key: &str) -> Result<(), JsError> {
        let Some(state) = self.step_states.get(step_key) else {
            return Ok(());
        };
        let globals = state.globals().clone();
        if !globals.is_truthy() {
            return Ok(());
        }

        // Preserve skip flag
        let was_skipped = object_member(&state.values, "_skip");

        // Reset to defaults
        let mut new_values = Object::new();
        for (param_name, spec) in entries(&globals) {
            let default = get(&spec, "default")?;
            if !default.is_undefined() {
                set_plain(&mut new_values, &param_name, clone_value(&default));
            }
        }

        // Restore skip flag
        if was_skipped.is_truthy() {
            new_values.insert("_skip", Value::Bool(true));
        }

        self.step_states
            .get_mut(step_key)
            .expect("the step state exists")
            .values = new_values;

        self.apply_to_pipeline_internal()?;
        let mut data = Object::new();
        data.insert("stepKey", Value::from(step_key));
        self.emit("reset", &Value::Object(data));
        Ok(())
    }

    /// `setSkip(stepKey, skip)`: set a step's bypass flag, apply, emit `change`.
    pub fn set_skip(&mut self, step_key: &str, skip: impl Into<Value>) -> Result<(), JsError> {
        let skip = skip.into();
        let Some(state) = self.step_states.get_mut(step_key) else {
            return Ok(());
        };
        let previous_value = object_member(&state.values, "_skip");
        state.values.insert("_skip", skip.clone());
        self.apply_to_pipeline_internal()?;
        let mut data = Object::new();
        data.insert("stepKey", Value::from(step_key));
        data.insert("paramName", Value::from("_skip"));
        data.insert("value", skip);
        data.insert("previousValue", previous_value);
        self.emit("change", &Value::Object(data));
        Ok(())
    }

    /// `isSkipped(stepKey)`.
    pub fn is_skipped(&self, step_key: &str) -> bool {
        self.step_states
            .get(step_key)
            .is_some_and(|s| strict_equals(&object_member(&s.values, "_skip"), &Value::Bool(true)))
    }

    /// `deleteStep(stepIndex)`: remove the step at a global index from the
    /// renderer's program (a chain's starter removes the whole chain, as does
    /// removing its last non-write step), regenerate the DSL and load it.
    pub fn delete_step(&mut self, step_index: f64) -> Result<DeleteStepResult, JsError> {
        let current_dsl = match &self.renderer {
            Some(r) => r.current_dsl(),
            None => String::new(),
        };
        if current_dsl.is_empty() {
            return Ok(DeleteStepResult::failure("no DSL available"));
        }

        let mut compiled = match compile(&current_dsl, &self.registry) {
            Ok(c) => c,
            Err(err) => {
                return Ok(DeleteStepResult::failure(format!(
                    "DSL syntax error: {}",
                    error_message(&err)?
                )));
            }
        };
        if !member(&compiled, "plans").is_truthy() {
            return Ok(DeleteStepResult::failure("compilation failed"));
        }

        // Preserve search namespaces
        if let Some(namespaces) = search_namespaces(&current_dsl) {
            compiled.set(
                "searchNamespaces",
                Value::Array(namespaces.into_iter().map(Value::from).collect()),
            );
        }

        let mut global_step_index = 0.0;
        let mut found = false;
        let mut deleted_surface_name = Value::Null;

        // `for (p...) { const plan = compiled.plans[p]; if (!plan.chain) continue;
        // for (s...) ... }`: nothing changes until the step is found, so the
        // lengths are read once per loop.
        let plan_count = to_number(&member(compiled.get("plans"), "length"))?;
        let mut p = 0usize;
        while (p as f64) < plan_count {
            let chain_len = {
                let plan = compiled.get("plans").at(p);
                if plan.is_nullish() {
                    return Err(cannot_read(plan, "chain"));
                }
                let chain = plan.get("chain");
                if !chain.is_truthy() {
                    p += 1;
                    continue;
                }
                to_number(&member(chain, "length"))?
            };
            let mut s = 0usize;
            while (s as f64) < chain_len {
                if global_step_index == step_index {
                    let plans = compiled
                        .get_mut("plans")
                        .and_then(Value::as_array_mut)
                        .ok_or_else(|| not_a_function("compiled.plans.splice"))?;
                    let plan_ref = &mut plans[p];
                    let deleted_step = plan_ref.get("chain").at(s).clone();

                    // Deleting a starter effect should remove the entire chain
                    if s == 0
                        && deleted_step.is_truthy()
                        && !member(&deleted_step, "builtin").is_truthy()
                    {
                        let ns = member(&deleted_step, "namespace");
                        let namespace = {
                            let a = member(&ns, "namespace");
                            if a.is_truthy() {
                                a
                            } else {
                                let b = member(&ns, "resolved");
                                if b.is_truthy() { b } else { Value::Null }
                            }
                        };
                        let op = member(&deleted_step, "op");
                        let def = match get_effect_value(&self.registry, &op) {
                            Some(d) => Some(d),
                            None if namespace.is_truthy() => get_effect_str(
                                &self.registry,
                                &format!("{}/{}", to_string(&namespace)?, to_string(&op)?),
                            ),
                            None => None,
                        };
                        let deleted_is_starter = def.as_ref().is_some_and(is_starter_effect);
                        if deleted_is_starter {
                            // Track the surface this chain was writing to
                            let write = member(plan_ref, "write");
                            if write.is_truthy() {
                                deleted_surface_name = surface_name(&write);
                            }
                            plans.remove(p);
                            found = true;
                            break;
                        }
                    }

                    let chain = plan_ref
                        .get_mut("chain")
                        .and_then(Value::as_array_mut)
                        .ok_or_else(|| not_a_function("plan.chain.splice"))?;
                    if s < chain.len() {
                        chain.remove(s);
                    }

                    let chain_empty = chain.is_empty();
                    let has_non_write_step = chain.iter().any(|step| {
                        !(member(step, "builtin").is_truthy()
                            && strict_equals(&member(step, "op"), &Value::from("_write")))
                    });
                    if chain_empty || !has_non_write_step {
                        // Track the surface this chain was writing to (a plan
                        // with only _write nodes left goes too)
                        let write = member(plan_ref, "write");
                        if write.is_truthy() {
                            deleted_surface_name = surface_name(&write);
                        }
                        plans.remove(p);
                    }
                    found = true;
                    break;
                }
                global_step_index += 1.0;
                s += 1;
            }
            if found {
                break;
            }
            p += 1;
        }

        if !found {
            return Ok(DeleteStepResult::failure("step not found"));
        }

        // Regenerate DSL
        let new_dsl = {
            let options = UnparseOptions {
                get_effect_def: Some(edit_effect_def(&self.registry)),
                ..UnparseOptions::default()
            };
            unparse(
                &compiled,
                &Value::Object(Object::new()),
                &options,
                &self.registry,
            )?
        };

        // Update state (this will emit structurechange)
        self.from_dsl(&new_dsl)?;

        Ok(DeleteStepResult {
            success: true,
            new_dsl: Some(new_dsl),
            deleted_surface_name,
            error: None,
        })
    }

    /// `insertStep(afterStepIndex, effectId)`: insert an effect (`"ns/name"` or
    /// `"name"`) after a global step index (`-1`: at the start of the first
    /// chain; before a terminal `_write`), regenerate the DSL and load it.
    pub fn insert_step(
        &mut self,
        after_step_index: f64,
        effect_id: &str,
    ) -> Result<InsertStepResult, JsError> {
        let current_dsl = match &self.renderer {
            Some(r) => r.current_dsl(),
            None => String::new(),
        };
        if current_dsl.is_empty() {
            return Ok(InsertStepResult::failure("no DSL available"));
        }

        let mut compiled = match compile(&current_dsl, &self.registry) {
            Ok(c) => c,
            Err(err) => {
                return Ok(InsertStepResult::failure(format!(
                    "DSL syntax error: {}",
                    error_message(&err)?
                )));
            }
        };
        if !member(&compiled, "plans").is_truthy() {
            return Ok(InsertStepResult::failure("compilation failed"));
        }

        // Preserve search namespaces
        if let Some(namespaces) = search_namespaces(&current_dsl) {
            compiled.set(
                "searchNamespaces",
                Value::Array(namespaces.into_iter().map(Value::from).collect()),
            );
        }

        // Parse effect ID to get namespace and name (an empty namespace is
        // falsy wherever the reference tests it)
        let (namespace, effect_name) = match effect_id.find('/') {
            Some(i) => (
                Some(&effect_id[..i]).filter(|ns| !ns.is_empty()),
                &effect_id[i + 1..],
            ),
            None => (None, effect_id),
        };

        // Check if effect is a starter (can only begin chains)
        let effect_def = get_effect_str(&self.registry, effect_id)
            .or_else(|| get_effect_str(&self.registry, effect_name))
            .or_else(|| {
                namespace
                    .and_then(|ns| get_effect_str(&self.registry, &format!("{ns}.{effect_name}")))
            });
        if effect_def.as_ref().is_some_and(is_starter_effect) {
            return Ok(InsertStepResult::failure(format!(
                "Cannot insert starter effect '{effect_id}' mid-chain"
            )));
        }

        // Ensure namespace is in search directives (a non-empty namespace)
        if let Some(ns) = namespace {
            let search = member(&compiled, "searchNamespaces");
            let includes = match &search {
                Value::Array(items) => items
                    .iter()
                    .any(|v| crate::unparser::jsv::same_value_zero(v, &Value::from(ns))),
                Value::String(s) => s.contains(ns),
                _ => false,
            };
            if !search.is_truthy() || !includes {
                if !search.is_truthy() {
                    compiled.set("searchNamespaces", Value::Array(Vec::new()));
                }
                match compiled.get_mut("searchNamespaces") {
                    Some(Value::Array(items)) => items.push(Value::from(ns)),
                    _ => return Err(not_a_function("compiled.searchNamespaces.push")),
                }
            }
        }

        // Find the target step location
        let mut global_step_index = 0.0;
        let mut target_plan_index: f64 = -1.0;
        let mut target_chain_index: f64 = -1.0;
        let plans_value = get(&compiled, "plans")?;
        let plan_list = iterate(&plans_value, "compiled.plans").unwrap_or_default();
        'plans: for (p, plan) in plan_list.iter().enumerate() {
            let chain = get(plan, "chain")?;
            if !chain.is_truthy() {
                continue;
            }
            let len = to_number(&member(&chain, "length"))?;
            let mut s = 0usize;
            while (s as f64) < len {
                if global_step_index == after_step_index {
                    target_plan_index = p as f64;
                    target_chain_index = s as f64;
                    break 'plans;
                }
                global_step_index += 1.0;
                s += 1;
            }
        }

        if target_plan_index < 0.0 && after_step_index >= 0.0 {
            return Ok(InsertStepResult::failure(format!(
                "Step index {} not found",
                number_to_string(after_step_index)
            )));
        }

        // Handle special case: afterStepIndex = -1 means insert at beginning
        if after_step_index < 0.0 {
            target_plan_index = 0.0;
            target_chain_index = -1.0; // Will insert at position 0
        }

        let target_chain = if target_plan_index >= 0.0 {
            member(
                &member(&plans_value, &number_to_string(target_plan_index)),
                "chain",
            )
        } else {
            Value::Undefined
        };
        if !target_chain.is_truthy() {
            return Ok(InsertStepResult::failure("Target chain not found"));
        }

        // Find max temp index for new step
        let mut max_temp = 0.0;
        for plan in &plan_list {
            let chain = get(plan, "chain")?;
            if !chain.is_truthy() {
                continue;
            }
            for step in iterate(&chain, "plan.chain")? {
                if let Value::Number(t) = get(&step, "temp")?
                    && t > max_temp
                {
                    max_temp = t;
                }
            }
        }

        // Create the new step AST node
        let mut new_step = Object::new();
        new_step.insert("op", Value::from(effect_name));
        new_step.insert("args", Value::Object(Object::new()));
        new_step.insert("temp", Value::Number(max_temp + 1.0));
        // Add namespace if present
        if let Some(ns) = namespace {
            let mut n = Object::new();
            n.insert("namespace", Value::from(ns));
            new_step.insert("namespace", Value::Object(n));
        }

        // Determine insert position (after target, before any _write)
        let mut insert_position = target_chain_index + 1.0;
        let target_step = member(&target_chain, &number_to_string(target_chain_index));
        if member(&target_step, "builtin").is_truthy()
            && strict_equals(&member(&target_step, "op"), &Value::from("_write"))
        {
            insert_position = target_chain_index;
        }

        // Insert the new step
        let target_plan = target_plan_index as usize;
        match compiled
            .get_mut("plans")
            .and_then(|plans| child_mut(plans, &target_plan.to_string()))
            .and_then(|plan| plan.get_mut("chain"))
        {
            Some(Value::Array(chain)) => {
                let at = (insert_position.max(0.0) as usize).min(chain.len());
                chain.insert(at, Value::Object(new_step));
            }
            _ => return Err(not_a_function("targetChain.splice")),
        }

        // Regenerate DSL
        let new_dsl = {
            let options = UnparseOptions {
                get_effect_def: Some(edit_effect_def(&self.registry)),
                ..UnparseOptions::default()
            };
            unparse(
                &compiled,
                &Value::Object(Object::new()),
                &options,
                &self.registry,
            )?
        };

        // Update state (this will emit structurechange)
        self.from_dsl(&new_dsl)?;

        // Calculate the new step's global index
        let mut new_step_index = 0.0;
        let plans = get(&compiled, "plans")?;
        for p in 0..target_plan {
            let len = member(&member(&plans.at(p).clone(), "chain"), "length");
            new_step_index += if len.is_truthy() {
                to_number(&len)?
            } else {
                0.0
            };
        }
        new_step_index += insert_position;

        Ok(InsertStepResult {
            success: true,
            new_dsl: Some(new_dsl),
            new_step_index: Some(new_step_index),
            error: None,
        })
    }

    // =====================================================================
    // Structure Access Methods
    // =====================================================================

    /// `getStructure()`: the effect chain (a copy).
    pub fn get_structure(&self) -> Vec<EffectInfo> {
        self.structure.clone()
    }

    /// The effect chain.
    pub fn structure(&self) -> &[EffectInfo] {
        &self.structure
    }

    /// `getCompiled()`: the compiled program of the last `fromDsl` (`None`:
    /// `null`).
    pub fn get_compiled(&self) -> Option<&Value> {
        self.compiled.as_ref()
    }

    /// `getEffectDef(stepKey)`: the step's effect definition (`None`: `null`).
    pub fn get_effect_def(&self, step_key: &str) -> Option<Value> {
        let def = self.step_states.get(step_key)?.def();
        def.is_truthy().then_some(def)
    }

    /// The step's registered effect.
    pub fn effect_entry(&self, step_key: &str) -> Option<&Rc<EffectEntry>> {
        self.step_states.get(step_key)?.effect_def.as_ref()
    }

    /// `stepCount`.
    pub fn step_count(&self) -> usize {
        self.step_states.len()
    }

    /// `getStepKeys()`.
    pub fn get_step_keys(&self) -> Vec<String> {
        self.step_states.keys().cloned().collect()
    }

    /// The step states, in insertion order.
    pub fn step_states(&self) -> &IndexMap<String, StepState> {
        &self.step_states
    }

    /// `getAllStepValues()`: every step's values keyed by step key (the shape
    /// `applyStepParameterValues` takes).
    pub fn get_all_step_values(&self) -> Object {
        let mut result = Object::new();
        for (step_key, state) in &self.step_states {
            set_plain(
                &mut result,
                step_key,
                Value::Object(spread_object(&state.values)),
            );
        }
        result
    }

    // =====================================================================
    // Routing Override Methods
    // =====================================================================

    fn emit_routing(&mut self, key: &str, index_name: Option<&str>, index: Value, value: Value) {
        let mut data = Object::new();
        data.insert("type", Value::from("routing"));
        data.insert("key", Value::from(key));
        if let Some(name) = index_name {
            data.insert(name, index);
        }
        data.insert("value", value);
        self.emit("change", &Value::Object(data));
    }

    /// `setWriteTarget(planIndex, target)`: the end-of-chain write target of a plan.
    pub fn set_write_target(&mut self, plan_index: impl Into<Value>, target: impl Into<Value>) {
        let (plan_index, target) = (plan_index.into(), target.into());
        self.write_target_overrides
            .set(plan_index.clone(), target.clone());
        self.emit_routing("writeTarget", Some("planIndex"), plan_index, target);
    }

    /// `getWriteTarget(planIndex)`.
    pub fn get_write_target(&self, plan_index: impl Into<Value>) -> Value {
        self.write_target_overrides
            .get(&plan_index.into())
            .cloned()
            .unwrap_or(Value::Undefined)
    }

    /// `setWriteStepTarget(stepIndex, target)`: the target of a mid-chain write step.
    pub fn set_write_step_target(
        &mut self,
        step_index: impl Into<Value>,
        target: impl Into<Value>,
    ) {
        let (step_index, target) = (step_index.into(), target.into());
        self.write_step_target_overrides
            .set(step_index.clone(), target.clone());
        self.emit_routing("writeStepTarget", Some("stepIndex"), step_index, target);
    }

    /// `getWriteStepTarget(stepIndex)`.
    pub fn get_write_step_target(&self, step_index: impl Into<Value>) -> Value {
        self.write_step_target_overrides
            .get(&step_index.into())
            .cloned()
            .unwrap_or(Value::Undefined)
    }

    /// `setReadSource(stepIndex, source)`: the source of a read step.
    pub fn set_read_source(&mut self, step_index: impl Into<Value>, source: impl Into<Value>) {
        let (step_index, source) = (step_index.into(), source.into());
        self.read_source_overrides
            .set(step_index.clone(), source.clone());
        self.emit_routing("readSource", Some("stepIndex"), step_index, source);
    }

    /// `getReadSource(stepIndex)`.
    pub fn get_read_source(&self, step_index: impl Into<Value>) -> Value {
        self.read_source_overrides
            .get(&step_index.into())
            .cloned()
            .unwrap_or(Value::Undefined)
    }

    /// `setRead3dVolume(stepIndex, volume)` (no event).
    pub fn set_read3d_volume(&mut self, step_index: impl Into<Value>, volume: impl Into<Value>) {
        self.read3d_vol_overrides
            .set(step_index.into(), volume.into());
    }

    /// `setRead3dGeometry(stepIndex, geometry)` (no event).
    pub fn set_read3d_geometry(
        &mut self,
        step_index: impl Into<Value>,
        geometry: impl Into<Value>,
    ) {
        self.read3d_geo_overrides
            .set(step_index.into(), geometry.into());
    }

    /// `setWrite3dVolume(stepIndex, volume)` (no event).
    pub fn set_write3d_volume(&mut self, step_index: impl Into<Value>, volume: impl Into<Value>) {
        self.write3d_vol_overrides
            .set(step_index.into(), volume.into());
    }

    /// `setWrite3dGeometry(stepIndex, geometry)` (no event).
    pub fn set_write3d_geometry(
        &mut self,
        step_index: impl Into<Value>,
        geometry: impl Into<Value>,
    ) {
        self.write3d_geo_overrides
            .set(step_index.into(), geometry.into());
    }

    /// `setRenderTarget(target)`.
    pub fn set_render_target(&mut self, target: impl Into<Value>) {
        let target = target.into();
        self.render_target_override = target.clone();
        self.emit_routing("renderTarget", None, Value::Undefined, target);
    }

    /// `getRenderTarget()` (`null` when not overridden).
    pub fn get_render_target(&self) -> Value {
        self.render_target_override.clone()
    }

    /// `clearRoutingOverrides()`.
    pub fn clear_routing_overrides(&mut self) {
        self.write_target_overrides.clear();
        self.write_step_target_overrides.clear();
        self.read_source_overrides.clear();
        self.read3d_vol_overrides.clear();
        self.read3d_geo_overrides.clear();
        self.write3d_vol_overrides.clear();
        self.write3d_geo_overrides.clear();
        self.render_target_override = Value::Null;
    }

    /// The routing override maps, in the reference's order: write targets,
    /// write step targets, read sources, read3d volumes, read3d geometries,
    /// write3d volumes, write3d geometries.
    pub fn routing_overrides(&self) -> [&JsMap<Value>; 7] {
        [
            &self.write_target_overrides,
            &self.write_step_target_overrides,
            &self.read_source_overrides,
            &self.read3d_vol_overrides,
            &self.read3d_geo_overrides,
            &self.write3d_vol_overrides,
            &self.write3d_geo_overrides,
        ]
    }

    // =====================================================================
    // Media Metadata
    // =====================================================================

    /// `setMediaInput(stepIndex, metadata)`: emit `mediachange`.
    pub fn set_media_input(&mut self, step_index: impl Into<Value>, metadata: Value) {
        let step_index = step_index.into();
        self.media_inputs.set(step_index.clone(), metadata.clone());
        self.emit_metadata("mediachange", step_index, metadata);
    }

    /// `getMediaInput(stepIndex)`.
    pub fn get_media_input(&self, step_index: impl Into<Value>) -> Value {
        self.media_inputs
            .get(&step_index.into())
            .cloned()
            .unwrap_or(Value::Undefined)
    }

    /// `removeMediaInput(stepIndex)`: emit `mediachange` with `metadata: null`.
    pub fn remove_media_input(&mut self, step_index: impl Into<Value>) {
        let step_index = step_index.into();
        self.media_inputs.delete(&step_index);
        self.emit_metadata("mediachange", step_index, Value::Null);
    }

    /// `getAllMediaInputs()` (a copy).
    pub fn get_all_media_inputs(&self) -> JsMap<Value> {
        self.media_inputs.clone()
    }

    /// `setTextInput(stepIndex, metadata)`: emit `textchange`.
    pub fn set_text_input(&mut self, step_index: impl Into<Value>, metadata: Value) {
        let step_index = step_index.into();
        self.text_inputs.set(step_index.clone(), metadata.clone());
        self.emit_metadata("textchange", step_index, metadata);
    }

    /// `getTextInput(stepIndex)`.
    pub fn get_text_input(&self, step_index: impl Into<Value>) -> Value {
        self.text_inputs
            .get(&step_index.into())
            .cloned()
            .unwrap_or(Value::Undefined)
    }

    /// `removeTextInput(stepIndex)`: emit `textchange` with `metadata: null`.
    pub fn remove_text_input(&mut self, step_index: impl Into<Value>) {
        let step_index = step_index.into();
        self.text_inputs.delete(&step_index);
        self.emit_metadata("textchange", step_index, Value::Null);
    }

    /// `getAllTextInputs()` (a copy).
    pub fn get_all_text_inputs(&self) -> JsMap<Value> {
        self.text_inputs.clone()
    }

    fn emit_metadata(&mut self, event: &str, step_index: Value, metadata: Value) {
        let mut data = Object::new();
        data.insert("stepIndex", step_index);
        data.insert("metadata", metadata);
        self.emit(event, &Value::Object(data));
    }

    // =====================================================================
    // Pipeline Integration
    // =====================================================================

    /// `setRenderer(renderer)`.
    pub fn set_renderer(&mut self, renderer: Option<H>) {
        self.renderer = renderer;
    }

    /// The renderer.
    pub fn renderer(&self) -> Option<&H> {
        self.renderer.as_ref()
    }

    /// The renderer, mutably.
    pub fn renderer_mut(&mut self) -> Option<&mut H> {
        self.renderer.as_mut()
    }

    /// `applyToPipeline()`: write every step's values to its passes' uniforms
    /// (also called after every state change).
    pub fn apply_to_pipeline(&mut self) -> Result<(), JsError> {
        self.apply_to_pipeline_internal()
    }

    /// `_applyToPipeline()`.
    fn apply_to_pipeline_internal(&mut self) -> Result<(), JsError> {
        let Some(renderer) = self.renderer.as_mut() else {
            return Ok(());
        };
        if renderer.graph_passes().is_none() {
            return Ok(());
        }
        let step_states = &self.step_states;

        // Whether any chain-scoped param changed, so texture dimensions are
        // re-resolved (e.g. screenDivide:'zoom' → zoom_chain_N).
        let mut scoped_param_changed = false;

        for (step_key, state) in step_states {
            let Some(digits) = step_key_digits(step_key) else {
                continue;
            };
            let step_index = parse_digits(digits);

            // Find passes for this step
            let mut step_passes = Vec::new();
            {
                let passes = renderer
                    .graph_passes()
                    .ok_or_else(|| cannot_read(&Value::Undefined, "passes"))?;
                for (i, pass) in passes.iter().enumerate() {
                    if pass.is_nullish() {
                        return Err(cannot_read(pass, "id"));
                    }
                    let id = pass.get("id");
                    if !id.is_truthy() {
                        continue;
                    }
                    if let Some(d) = pass_node_digits(id)?
                        && parse_digits(&d) == step_index
                    {
                        step_passes.push(i);
                    }
                }
            }
            if step_passes.is_empty() {
                continue;
            }

            let globals = state.globals();

            // The pass members read here (`uniforms`, `inheritsVolumeSize`,
            // `scopedParams`, `nodeId`, `effectKey`) are own data members of a
            // graph pass, so they are read in place.
            for &pass_index in &step_passes {
                macro_rules! pass {
                    () => {
                        renderer
                            .graph_passes()
                            .and_then(|p| p.get_mut(pass_index))
                            .ok_or_else(|| cannot_read(&Value::Undefined, "uniforms"))?
                    };
                }
                if !pass!().get("uniforms").is_truthy() {
                    continue;
                }

                // Deferred palette expansion: collected during the param loop
                // and applied after it, so default param values do not
                // overwrite the expanded values.
                let mut palette_expansion: Option<Object> = None;

                for (param_name, value) in state.values.iter() {
                    if value.is_nullish() {
                        continue;
                    }
                    if param_name.starts_with('_') {
                        continue; // Skip internal flags
                    }
                    // Skip automation-controlled params (oscillator, midi, audio manage the value)
                    if is_automation_controlled(value) {
                        continue;
                    }

                    let spec = member(globals, param_name);
                    let uniform_name = {
                        let u = member(&spec, "uniform");
                        if u.is_truthy() {
                            u
                        } else {
                            Value::from(param_name.as_str())
                        }
                    };

                    // Convert value for uniform if renderer has the method
                    let converted = if renderer.has_convert_parameter_for_uniform() {
                        renderer.convert_parameter_for_uniform(value, &spec)?
                    } else {
                        value.clone()
                    };

                    // A pass may feed a renamed shader uniform from this param.
                    write_uniform_aliases(pass!(), param_name, &uniform_name, &converted)?;

                    let uniform_key = key_of(&uniform_name)?;
                    if in_object(&uniform_key, pass!().get("uniforms"))? {
                        // Consumer passes inherit volumeSize from the upstream
                        // source emitter: their step-state default is stale.
                        if strict_equals(&uniform_name, &Value::from("volumeSize"))
                            && pass!().get("inheritsVolumeSize").is_truthy()
                        {
                            continue;
                        }

                        let pass = pass!();
                        let uniforms = pass
                            .get_mut("uniforms")
                            .ok_or_else(|| cannot_read(&Value::Undefined, &uniform_key))?;
                        assign_member(uniforms, &uniform_key, converted.clone())?;

                        // Propagate to the chain-scoped variant and broadcast it
                        // to the other chain members.
                        let scoped_params = pass.get("scopedParams");
                        if scoped_params.is_truthy() {
                            let scoped = member(scoped_params, &uniform_key);
                            if scoped.is_truthy() {
                                let scoped_key = key_of(&scoped)?;
                                let uniforms = pass
                                    .get_mut("uniforms")
                                    .ok_or_else(|| cannot_read(&Value::Undefined, &scoped_key))?;
                                let current = member(uniforms, &uniform_key);
                                assign_member(uniforms, &scoped_key, current)?;
                                scoped_param_changed = true;
                                if !renderer.has_broadcast_chain_scoped_param() {
                                    return Err(not_a_function(
                                        "pipeline.broadcastChainScopedParam",
                                    ));
                                }
                                renderer.broadcast_chain_scoped_param(
                                    pass_index,
                                    &uniform_key,
                                    &scoped_key,
                                )?;
                            }
                        }
                    }

                    // Legacy classicNoisedeck palette expansion
                    if strict_equals(&member(&spec, "type"), &Value::from("palette")) {
                        palette_expansion = expand_palette_value(&converted)?;
                    }
                }

                // Apply palette expansion after all params are written
                if let Some(expansion) = palette_expansion {
                    for (u_name, u_value) in expansion.iter() {
                        let pass = pass!();
                        if in_object(u_name, pass.get("uniforms"))? {
                            let uniforms = pass
                                .get_mut("uniforms")
                                .ok_or_else(|| cannot_read(&Value::Undefined, u_name))?;
                            assign_member(uniforms, u_name, u_value.clone())?;
                        }
                    }
                }
            }

            // Trigger async effect regen if any non-alpha param changed for this node
            if renderer.has_check_async_regen() {
                let (node_id, effect_key) = {
                    let passes = renderer
                        .graph_passes()
                        .ok_or_else(|| cannot_read(&Value::Undefined, "passes"))?;
                    let first = &passes[step_passes[0]];
                    (first.get("nodeId").clone(), first.get("effectKey").clone())
                };
                if node_id.is_truthy() && effect_key.is_truthy() {
                    renderer.check_async_regen(&node_id, &effect_key, &state.values)?;
                }
            }
        }

        // A chain-scoped param drives screenDivide-sized buffers: recompute
        // affected texture dimensions.
        if scoped_param_changed && renderer.has_recreate_textures() {
            if !renderer.has_collect_default_uniforms() {
                return Err(not_a_function("pipeline.collectDefaultUniforms"));
            }
            let uniforms = renderer.collect_default_uniforms()?;
            renderer.recreate_textures(uniforms)?;
        }

        // stateSize → scoped pipeline.setUniform('stateSize_node_N', ...).
        if renderer.has_set_uniform() {
            for (step_key, state) in step_states {
                // `'stateSize' in stepState.values` (not an Object.prototype member)
                if state.values.contains_key("stateSize")
                    && let Some(digits) = step_key_digits(step_key)
                {
                    let value = object_member(&state.values, "stateSize");
                    renderer.set_uniform(&format!("stateSize_node_{digits}"), &value)?;
                }
            }
        }
        Ok(())
    }

    // =====================================================================
    // Serialization
    // =====================================================================

    /// `serialize()`: `{ version: 1, dsl, stepStates, overrides, mediaInputs,
    /// textInputs }`.
    pub fn serialize(&self) -> Result<Value, JsError> {
        let mut step_states = Object::new();
        for (key, state) in &self.step_states {
            let mut s = Object::new();
            s.insert("effectKey", Value::from(state.effect_key.as_str()));
            s.insert("values", Value::Object(spread_object(&state.values)));
            s.insert("_skip", object_member(&state.values, "_skip"));
            set_plain(&mut step_states, key, Value::Object(s));
        }

        let mut overrides = Object::new();
        let names = [
            "writeTargets",
            "writeStepTargets",
            "readSources",
            "read3dVol",
            "read3dGeo",
            "write3dVol",
            "write3dGeo",
        ];
        for (name, map) in names.iter().zip(self.routing_overrides()) {
            overrides.insert(*name, Value::Object(map.to_object()?));
        }
        overrides.insert("renderTarget", self.render_target_override.clone());

        let mut out = Object::new();
        out.insert("version", Value::Number(1.0));
        out.insert(
            "dsl",
            Value::from(
                self.renderer
                    .as_ref()
                    .map(|r| r.current_dsl())
                    .unwrap_or_default(),
            ),
        );
        out.insert("stepStates", Value::Object(step_states));
        out.insert("overrides", Value::Object(overrides));
        out.insert("mediaInputs", Value::Object(self.media_inputs.to_object()?));
        out.insert("textInputs", Value::Object(self.text_inputs.to_object()?));
        Ok(Value::Object(out))
    }

    /// `deserialize(data)`: restore overrides and metadata (their keys become
    /// strings), load `data.dsl`, then merge the saved values over the loaded
    /// ones, apply and emit `load`.
    pub fn deserialize(&mut self, data: &Value) -> Result<(), JsError> {
        let version = get(data, "version")?;
        if !strict_equals(&version, &Value::Number(1.0)) {
            console::warn(&[
                "[ProgramState] Unknown serialization version:".into(),
                version.into(),
            ]);
        }

        // Restore overrides
        let overrides = member(data, "overrides");
        let restore = |key: &str| {
            let v = member(&overrides, key);
            if v.is_truthy() {
                JsMap::from_entries_of(&v)
            } else {
                JsMap::new()
            }
        };
        self.write_target_overrides = restore("writeTargets");
        self.write_step_target_overrides = restore("writeStepTargets");
        self.read_source_overrides = restore("readSources");
        self.read3d_vol_overrides = restore("read3dVol");
        self.read3d_geo_overrides = restore("read3dGeo");
        self.write3d_vol_overrides = restore("write3dVol");
        self.write3d_geo_overrides = restore("write3dGeo");
        let render_target = member(&overrides, "renderTarget");
        self.render_target_override = if render_target.is_truthy() {
            render_target
        } else {
            Value::Null
        };

        // Restore media/text metadata
        let restore_data = |key: &str| {
            let v = member(data, key);
            if v.is_truthy() {
                JsMap::from_entries_of(&v)
            } else {
                JsMap::new()
            }
        };
        self.media_inputs = restore_data("mediaInputs");
        self.text_inputs = restore_data("textInputs");

        // Load DSL (this rebuilds stepStates from structure)
        let dsl = member(data, "dsl");
        if dsl.is_truthy() {
            self.from_dsl_value(&dsl)?;
        }

        // Override stepState values from serialized data
        let saved_states = member(data, "stepStates");
        let saved_states = if saved_states.is_truthy() {
            entries(&saved_states)
        } else {
            Vec::new()
        };
        for (key, saved_state) in saved_states {
            if self.step_states.contains_key(&key) {
                let saved_values = get(&saved_state, "values")?;
                let state = self
                    .step_states
                    .get_mut(&key)
                    .expect("the step state exists");
                let mut merged = spread_object(&state.values);
                spread_into(&mut merged, &saved_values);
                state.values = merged;
            }
        }

        self.apply_to_pipeline_internal()?;
        let mut data_out = Object::new();
        data_out.insert("structure", effects_to_value(&self.structure));
        self.emit("load", &Value::Object(data_out));
        Ok(())
    }

    // =====================================================================
    // Internal state (for inspection)
    // =====================================================================

    /// The current batch nesting depth.
    pub fn batch_depth(&self) -> usize {
        self.batch_depth
    }

    /// The changes queued by the current batch.
    pub fn batched_changes(&self) -> &[Change] {
        &self.batched_changes
    }

    /// Whether a `recompileNeeded` is pending.
    pub fn recompile_pending(&self) -> bool {
        self.recompile_pending
    }

    // =====================================================================
    // Helpers
    // =====================================================================

    /// `_preserveValuesByOccurrence()`: effectKey -> each occurrence's values.
    fn preserve_values_by_occurrence(&self) -> IndexMap<String, Vec<Object>> {
        let mut preserved: IndexMap<String, Vec<Object>> = IndexMap::new();
        for state in self.step_states.values() {
            preserved
                .entry(state.effect_key.clone())
                .or_default()
                .push(spread_object(&state.values));
        }
        preserved
    }

    /// `_rebuildStepStates(effects, preservedValues)`: defaults, then DSL
    /// arguments, then the preserved values of the same effect occurrence for
    /// parameters the DSL does not specify (never `_skip`, never over an
    /// automation binding from the DSL).
    fn rebuild_step_states(
        &mut self,
        effects: &[EffectInfo],
        preserved_values: &IndexMap<String, Vec<Object>>,
    ) -> Result<(), JsError> {
        let mut new_step_states: IndexMap<String, StepState> = IndexMap::new();
        let mut occurrence_counts: IndexMap<String, usize> = IndexMap::new();

        for effect in effects {
            let step_key = format!("step_{}", effect.step_index);
            let effect_key = effect.effect_key.clone();

            // Get effect definition
            let effect_def = self.registry.get_effect(&effect_key).cloned();

            // Track occurrence of this effect type
            let occurrence = occurrence_counts.get(&effect_key).copied().unwrap_or(0);
            occurrence_counts.insert(effect_key.clone(), occurrence + 1);

            // Try to restore preserved values for this occurrence
            let empty = Object::new();
            let preserved_vals = preserved_values
                .get(&effect_key)
                .and_then(|v| v.get(occurrence))
                .unwrap_or(&empty);

            let mut values = Object::new();

            // Start with defaults from effect definition
            let globals = match &effect_def {
                Some(entry) => member(&entry.def, "globals"),
                None => Value::Undefined,
            };
            if globals.is_truthy() {
                for (param_name, spec) in entries(&globals) {
                    let default = get(&spec, "default")?;
                    if !default.is_undefined() {
                        set_plain(&mut values, &param_name, clone_value(&default));
                    }
                }
            }

            // Override with values from DSL args
            for (param_name, value) in effect.args.iter() {
                set_plain(&mut values, param_name, clone_value(value));
            }

            // Override with preserved values ONLY for params NOT specified in DSL
            let args_value = Value::Object(effect.args.clone());
            for (param_name, value) in preserved_vals.iter() {
                // Skip if DSL explicitly specifies this param (use DSL value instead)
                if has_property(&args_value, param_name) {
                    continue;
                }
                // Never restore _skip from preserved values (structural flag)
                if param_name == "_skip" {
                    continue;
                }
                if param_name.starts_with('_') || !value.is_undefined() {
                    // Don't overwrite automation bindings from DSL with preserved scalar values
                    let dsl_arg = object_member(&effect.args, param_name);
                    if is_automation_config(&dsl_arg) {
                        continue;
                    }
                    set_plain(&mut values, param_name, clone_value(value));
                }
            }

            new_step_states.insert(
                step_key,
                StepState {
                    effect_key,
                    effect_def,
                    step_index: effect.step_index as f64,
                    values,
                },
            );
        }

        self.step_states = new_step_states;
        Ok(())
    }

    /// `_updateValuesFromDsl(effects)`: take the DSL's argument values that
    /// differ (by `JSON.stringify`) from the current ones.
    fn update_values_from_dsl(&mut self, effects: &[EffectInfo]) {
        for effect in effects {
            let step_key = format!("step_{}", effect.step_index);
            let Some(state) = self.step_states.get_mut(&step_key) else {
                continue;
            };
            for (param_name, value) in effect.args.iter() {
                let current = object_member(&state.values, param_name);
                if json_stringify(&current) != json_stringify(value) {
                    set_plain(&mut state.values, param_name, clone_value(value));
                }
            }
        }
    }

    /// `_buildParameterOverrides()`: per global step index, the parameters to
    /// regenerate (internal flags but `_skip`, hidden and button controls and
    /// parameters automated in the DSL are left out; bindings regenerate as
    /// their variable).
    fn build_parameter_overrides(&self) -> Result<Value, JsError> {
        let mut overrides = Object::new();
        for (step_key, state) in &self.step_states {
            let Some(digits) = step_key_digits(step_key) else {
                continue;
            };
            let step_index = parse_digits(digits);

            // Original effect info, to check for automated parameters
            let effect_info = structure_at(&self.structure, step_index);

            let mut step_overrides = Object::new();
            let globals = state.globals();
            for (param_name, value) in state.values.iter() {
                // Skip internal flags EXCEPT _skip which is a DSL argument
                if param_name.starts_with('_') && param_name != "_skip" {
                    continue;
                }

                let spec = member(globals, param_name);
                let control = member(&member(&spec, "ui"), "control");
                // Skip internal-only params (no UI control, set programmatically)
                if strict_equals(&control, &Value::Bool(false)) {
                    continue;
                }
                // Skip button params (actions like resetState, not persistent state)
                if strict_equals(&control, &Value::from("button")) {
                    continue;
                }

                // Skip parameters automated in the original DSL
                let raw_kwarg = match effect_info {
                    Some(info) => member(&info.raw_kwargs, param_name),
                    None => Value::Undefined,
                };
                let is_automated_in_dsl = matches!(raw_kwarg, Value::Object(_) | Value::Array(_))
                    && matches!(
                        member(&raw_kwarg, "type").as_str(),
                        Some("Oscillator" | "Midi" | "Audio")
                    );
                if is_automated_in_dsl {
                    continue;
                }

                // Unwrap automation bindings for DSL (use varRef, not value)
                if is_var_ref_binding(value) {
                    let mut binding = Object::new();
                    binding.insert("_varRef", member(value, "_varRef"));
                    set_plain(&mut step_overrides, param_name, Value::Object(binding));
                } else {
                    set_plain(&mut step_overrides, param_name, value.clone());
                }
            }

            set_plain(
                &mut overrides,
                &number_to_string(step_index),
                Value::Object(step_overrides),
            );
        }
        Ok(Value::Object(overrides))
    }

    /// `_applyRoutingOverridesToCompiled(compiled)`.
    fn apply_routing_overrides_to_compiled(&self, compiled: &mut Value) -> Result<(), JsError> {
        if !member(compiled, "plans").is_truthy() {
            return Ok(());
        }

        // Apply write target overrides (end of chain)
        for (plan_index, target) in self.write_target_overrides.iter() {
            let key = to_property_key(plan_index)?;
            let plan = member(&member(compiled, "plans"), &key);
            if !plan.is_truthy() {
                continue;
            }
            let r = surface_ref(target)?;
            let plans = compiled.get_mut("plans").expect("plans is truthy");
            assign_in(plans, &key, "write", r.clone())?;

            // Also update the terminal _write step in the chain, since the
            // unparser uses its args.tex instead of plan.write when the chain
            // ends with a _write step.
            let chain = member(&plan, "chain");
            if chain.is_truthy() {
                let last_key = number_to_string(to_number(&member(&chain, "length"))? - 1.0);
                let last_step = member(&chain, &last_key);
                if member(&last_step, "builtin").is_truthy()
                    && strict_equals(&member(&last_step, "op"), &Value::from("_write"))
                    && member(&last_step, "args").is_truthy()
                    && let Some(step) = child_mut(plans, &key)
                        .and_then(|p| p.get_mut("chain"))
                        .and_then(|c| child_mut(c, &last_key))
                {
                    assign_in(step, "args", "tex", r)?;
                }
            }
        }

        // Builtin steps of every plan, with their global step index.
        let plan_count = match member(compiled, "plans") {
            Value::Array(plans) => plans.len(),
            other => iterate(&other, "compiled.plans")?.len(),
        };
        let for_each_builtin = |compiled: &mut Value,
                                op_name: &str,
                                f: &mut dyn FnMut(&mut Value, f64) -> Result<(), JsError>|
         -> Result<(), JsError> {
            let mut global_step_index = 0.0;
            for p in 0..plan_count {
                let plans = compiled.get_mut("plans").expect("plans is truthy");
                let Some(plan) = child_mut(plans, &p.to_string()) else {
                    // A missing plan reads `undefined.chain`.
                    return Err(cannot_read(&Value::Undefined, "chain"));
                };
                let chain = get(plan, "chain")?;
                if !chain.is_truthy() {
                    continue;
                }
                let steps = iterate(&chain, "plan.chain")?;
                for (s, step) in steps.iter().enumerate() {
                    if get(step, "builtin")?.is_truthy()
                        && strict_equals(&member(step, "op"), &Value::from(op_name))
                    {
                        let step_mut = plan
                            .get_mut("chain")
                            .and_then(|c| child_mut(c, &s.to_string()))
                            .expect("the step exists");
                        f(step_mut, global_step_index)?;
                    }
                    global_step_index += 1.0;
                }
            }
            Ok(())
        };

        // Apply mid-chain write step target overrides
        if !self.write_step_target_overrides.is_empty() {
            let overrides = &self.write_step_target_overrides;
            for_each_builtin(compiled, "_write", &mut |step, index| {
                if let Some(target) = overrides.get(&Value::Number(index)) {
                    let r = surface_ref(target)?;
                    assign_in(step, "args", "tex", r)?;
                }
                Ok(())
            })?;
        }

        // Apply read source overrides
        if !self.read_source_overrides.is_empty() {
            let overrides = &self.read_source_overrides;
            for_each_builtin(compiled, "_read", &mut |step, index| {
                if let Some(source) = overrides.get(&Value::Number(index)) {
                    let r = surface_ref(source)?;
                    assign_in(step, "args", "tex", r)?;
                }
                Ok(())
            })?;
        }

        // Apply read3d volume and geometry overrides
        if !self.read3d_vol_overrides.is_empty() || !self.read3d_geo_overrides.is_empty() {
            let (vol, geo) = (&self.read3d_vol_overrides, &self.read3d_geo_overrides);
            for_each_builtin(compiled, "_read3d", &mut |step, index| {
                if let Some(name) = vol.get(&Value::Number(index)) {
                    assign_in(step, "args", "tex3d", kind_ref("vol", name))?;
                }
                if let Some(name) = geo.get(&Value::Number(index)) {
                    assign_in(step, "args", "geo", kind_ref("geo", name))?;
                }
                Ok(())
            })?;
        }

        // Apply write3d volume and geometry overrides
        if !self.write3d_vol_overrides.is_empty() || !self.write3d_geo_overrides.is_empty() {
            let (vol, geo) = (&self.write3d_vol_overrides, &self.write3d_geo_overrides);
            for_each_builtin(compiled, "_write3d", &mut |step, index| {
                if let Some(name) = vol.get(&Value::Number(index)) {
                    assign_in(step, "args", "tex3d", kind_ref("vol", name))?;
                }
                if let Some(name) = geo.get(&Value::Number(index)) {
                    assign_in(step, "args", "geo", kind_ref("geo", name))?;
                }
                Ok(())
            })?;
        }

        // Apply render target override
        if self.render_target_override.is_truthy() {
            let render = member(compiled, "render");
            if matches!(render, Value::String(_)) {
                compiled.set("render", self.render_target_override.clone());
            } else if render.is_truthy() {
                assign_in(
                    compiled,
                    "render",
                    "target",
                    self.render_target_override.clone(),
                )?;
            }
        }
        Ok(())
    }
}

impl<H: ProgramHost + 'static> ProgramState<H> {
    /// `once(event, callback)`. Returns the self-removing wrapper.
    pub fn once(&self, event: &str, callback: Listener<Self>) -> Listener<Self> {
        self.emitter.once(event, callback)
    }
}
