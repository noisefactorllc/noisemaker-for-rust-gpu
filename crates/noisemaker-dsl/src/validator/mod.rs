//! Port of `lang/validator.js`: the semantic validator of the Polymorphic DSL.
//!
//! [`validate`] turns the parser's AST into the planned chains the expander
//! consumes — `{plans, diagnostics, render, vars, searchNamespaces}` — exactly as
//! the reference does, including its diagnostics (codes, messages, identifiers,
//! locations), its quirks and the `TypeError`s its code raises on some inputs.
//!
//! The port keeps the reference's function boundaries: every inner function of
//! `validate` is a method of [`Validator`] with the same name in snake case. The
//! reference manipulates one shared, mutable object graph (see `heap.rs`); the port
//! does the same and serializes the graph once at the end, so every output shows
//! the final state of every shared object, as `JSON.stringify` of the reference's
//! result does.
//!
//! Helpers ported alongside: `lang/enumPaths.js` ([`enum_paths`]),
//! `lang/stringLiterals.js` ([`string_literals`]), `resolveParamAliases` and
//! `checkEffectAlias` ([`aliases`]). The registries the reference imports (`ops`,
//! `enums`, the starter ops, the alias tables) are the [`Registry`]; `stdEnums`
//! is [`Registry::std_enums`]. `new Function` syntax checks are [`func_syntax`].

pub mod aliases;
pub mod enum_paths;
pub mod func_syntax;
mod heap;
#[cfg(test)]
mod reference_tests;
pub mod string_literals;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use indexmap::IndexMap;

use crate::diagnostics;
use crate::error::JsError;
use crate::registry::Registry;
use crate::value::{Object, Value};

use enum_paths::{
    apply_enum_prefix, normalize_member_path, normalize_member_path_v, path_starts_with,
};
use func_syntax::{FuncSyntax, check_function_body};
use heap::{V, greater_than, less_than, utf16_prefix};

/// STRICT ALLOWLIST FOR STRING PARAMETERS (`ALLOWED_STRING_PARAMS`):
/// `"<func>.<param>"`. Do not expand it.
const ALLOWED_STRING_PARAMS: &[&str] = &[
    "text.text",
    "text.font",
    "text.justify",
    "text.style",
    "midi.name",
    "midi.id",
    "audio.name",
    "audio.id",
];

/// `stateSurfaces`.
const STATE_SURFACES: &[&str] = &["time", "frame", "mouse", "resolution", "seed", "a"];

/// `stateValues`.
const STATE_VALUES: &[&str] = &[
    "time",
    "frame",
    "mouse",
    "resolution",
    "seed",
    "a",
    "u1",
    "u2",
    "u3",
    "u4",
    "s1",
    "s2",
    "b1",
    "b2",
    "a1",
    "a2",
    "deltaTime",
];

/// `SURFACE_PASSTHROUGH_CALLS`.
const SURFACE_PASSTHROUGH_CALLS: &[&str] = &["read"];

/// `MAX_AUTOMATION_DEPTH`.
const MAX_AUTOMATION_DEPTH: i64 = 8;

/// `AUTOMATION_FIELDS[type]`.
fn automation_fields(ty: &V) -> Option<&'static [&'static str]> {
    match ty.as_str()? {
        "Oscillator" => Some(&["oscType", "min", "max", "speed", "offset", "seed"]),
        "Midi" => Some(&[
            "channel",
            "mode",
            "min",
            "max",
            "sensitivity",
            "name",
            "id",
            "cc",
            "nrpn",
            "zone",
            "members",
        ]),
        "Audio" => Some(&["band", "min", "max", "channel", "name", "id"]),
        _ => None,
    }
}

fn has_str(set: &[&str], v: &V) -> bool {
    v.as_str().is_some_and(|s| set.contains(&s))
}

/// `toBoolean(value)`.
fn to_boolean(value: &V) -> bool {
    match value {
        V::Num(n) => *n != 0.0,
        other => other.truthy(),
    }
}

/// `clamp(value, min, max)` (`min`/`max` apply only when they are numbers).
fn clamp(value: &V, min: &Value, max: &Value) -> V {
    if let Value::Number(min) = min
        && less_than(value, *min)
    {
        return V::Num(*min);
    }
    if let Value::Number(max) = max
        && greater_than(value, *max)
    {
        return V::Num(*max);
    }
    value.clone()
}

/// `Number.isInteger(value)`.
fn is_integer(value: &V) -> bool {
    matches!(value, V::Num(n) if n.is_finite() && n.fract() == 0.0)
}

/// The elements of `value` for an array method call (`value.map(...)`): the
/// TypeError V8 raises for a missing or non-array value.
fn array_for(value: &V, expr: &str, method: &str) -> Result<Vec<V>, JsError> {
    match value {
        V::Arr(a) => Ok(a.borrow().clone()),
        V::Undefined | V::Null => value.get(method).map(|_| Vec::new()),
        _ => Err(JsError::type_error(format!(
            "{expr}.{method} is not a function"
        ))),
    }
}

/// `Object.entries(value)` (own enumerable string-keyed properties).
fn object_entries(value: &V) -> Result<Vec<(String, V)>, JsError> {
    match value {
        V::Obj(o) => Ok(o.borrow().entries()),
        V::Arr(a) => Ok(a
            .borrow()
            .iter()
            .enumerate()
            .map(|(i, v)| (i.to_string(), v.clone()))
            .collect()),
        V::Str(_) => match value.spread() {
            V::Obj(o) => Ok(o.borrow().entries()),
            _ => unreachable!(),
        },
        V::Undefined | V::Null => Err(JsError::type_error(
            "Cannot convert undefined or null to object",
        )),
        _ => Ok(Vec::new()),
    }
}

/// `Object.keys(value)`.
fn object_keys(value: &V) -> Result<Vec<String>, JsError> {
    Ok(object_entries(value)?.into_iter().map(|(k, _)| k).collect())
}

/// `path.join(sep)` for a JavaScript array.
fn join(path: &V, sep: &str) -> String {
    match path {
        V::Arr(a) => a
            .borrow()
            .iter()
            .map(|e| {
                if e.is_nullish() {
                    String::new()
                } else {
                    e.to_js_string()
                }
            })
            .collect::<Vec<_>>()
            .join(sep),
        other => other.to_js_string(),
    }
}

/// `for (const x of value)` and array destructuring: the elements of an array,
/// the code points of a string. Anything else throws V8's TypeError, whose text
/// depends on the site: `rendered` is the iterated expression where V8 prints it
/// (`calls is not iterable`); `None` where V8 prints the value instead (`number
/// 5 is not iterable (cannot read property Symbol(Symbol.iterator))`).
fn iterate(value: &V, rendered: Option<&str>) -> Result<Vec<V>, JsError> {
    match value {
        V::Arr(a) => Ok(a.borrow().clone()),
        V::Str(s) => Ok(s
            .chars()
            .map(|c| V::str(c.encode_utf8(&mut [0; 4])))
            .collect()),
        _ => Err(JsError::type_error(match rendered {
            Some(expr) => format!("{expr} is not iterable"),
            None => {
                let shown = match value {
                    V::Undefined => "undefined".to_owned(),
                    V::Null => "object null".to_owned(),
                    V::Bool(b) => format!("boolean {b}"),
                    V::Num(n) => format!("number {}", crate::js::number_to_string(*n)),
                    V::Func(_) => "function".to_owned(),
                    _ => "object".to_owned(),
                };
                format!("{shown} is not iterable (cannot read property Symbol(Symbol.iterator))")
            }
        })),
    }
}

/// `value?.join(sep)`: `undefined` for a nullish value, the joined elements of
/// an array; anything else has no `join` method.
fn optional_join(value: &V, expr: &str, sep: &str) -> Result<V, JsError> {
    match value {
        V::Undefined | V::Null => Ok(V::Undefined),
        V::Arr(_) => Ok(V::str(&join(value, sep))),
        _ => Err(JsError::type_error(format!(
            "{expr}?.join is not a function"
        ))),
    }
}

/// `` `${value.slice(0, n)}` `` and the like: the leading part of a string or
/// array, as a template literal shows it.
fn slice_text(value: &V, expr: &str, n: usize) -> Result<V, JsError> {
    match value.slice(expr, Some(n))? {
        V::Str(s) => Ok(V::Str(s)),
        other => Ok(V::str(&other.to_js_string())),
    }
}

/// `/^vol[0-7]$/.test(value)` style checks (`RegExp.prototype.test` converts its
/// argument with `ToString`).
fn is_ref_name(value: &V, prefix: &str) -> bool {
    let s = value.to_js_string();
    s.len() == prefix.len() + 1
        && s.starts_with(prefix)
        && matches!(s.as_bytes()[prefix.len()], b'0'..=b'7')
}

fn strs_to_v(path: &[String]) -> V {
    V::array_from(path.iter().map(|s| V::str(s)).collect())
}

/// `def.choices[name]` (an ordinary object's property read).
fn choices_get(choices: &Value, name: &V) -> V {
    match choices {
        Value::Object(_) => V::from_value(choices).get_opt(&name.to_js_string()),
        Value::Undefined | Value::Null => V::Undefined,
        other => V::from_value(other).get_opt(&name.to_js_string()),
    }
}

/// `getStarterInfo(node)`'s query name: `<namespace.resolved>.<name>` when the
/// call carries a resolved namespace, else its bare name.
fn starter_query_name(node: &V) -> Result<V, JsError> {
    let mut name = node.get("name")?;
    let ns = node.get("namespace")?;
    if ns.truthy() {
        let resolved = ns.get("resolved")?;
        if resolved.truthy() {
            name = V::str(&format!(
                "{}.{}",
                resolved.to_js_string(),
                node.get("name")?.to_js_string()
            ));
        }
    }
    Ok(name)
}

/// `toSurface(arg)`.
fn to_surface(arg: &V) -> Result<V, JsError> {
    if !arg.truthy() {
        return Ok(V::Null);
    }
    let ty = arg.get("type")?;
    let kind = match ty.as_str() {
        Some("OutputRef") => "output",
        Some("SourceRef") => "source",
        Some("XyzRef") => "xyz",
        Some("VelRef") => "vel",
        Some("RgbaRef") => "rgba",
        Some("MeshRef") => "mesh",
        _ => "",
    };
    if !kind.is_empty() {
        return Ok(V::object_from(vec![
            ("kind", V::str(kind)),
            ("name", arg.get("name")?),
        ]));
    }
    if ty.is_str("Ident") && arg.get("name")?.is_str("none") {
        return Ok(V::object_from(vec![
            ("kind", V::str("output")),
            ("name", V::str("none")),
        ]));
    }
    if ty.is_str("Ident") && has_str(STATE_SURFACES, &arg.get("name")?) {
        return Ok(V::object_from(vec![
            ("kind", V::str("state")),
            ("name", arg.get("name")?),
        ]));
    }
    Ok(V::Null)
}

/// `callToSurface(node)`.
fn call_to_surface(node: &V) -> Result<V, JsError> {
    if !node.truthy() || node.type_of() != "object" {
        return Ok(V::Null);
    }
    let ty = node.get("type")?;
    if ty.is_str("Chain") {
        let chain = node.get("chain")?;
        if let Some(elems) = chain.elements()
            && elems.len() == 1
        {
            return call_to_surface(&elems[0]);
        }
    }
    if !ty.is_str("Call") || !has_str(SURFACE_PASSTHROUGH_CALLS, &node.get("name")?) {
        return Ok(V::Null);
    }
    let mut target = V::Null;
    let args = node.get("args")?;
    if let Some(a) = args.elements()
        && !a.is_empty()
    {
        target = a[0].clone();
    }
    if !target.truthy() {
        let kwargs = node.get("kwargs")?;
        if kwargs.truthy() && kwargs.type_of() == "object" {
            target = kwargs.get("tex")?;
        }
    }
    if !target.truthy() {
        return Ok(V::Null);
    }
    to_surface(&target)
}

/// `firstChainCall(node)`.
fn first_chain_call(node: &V) -> Result<V, JsError> {
    if !node.truthy() || node.type_of() != "object" {
        return Ok(V::Null);
    }
    let ty = node.get("type")?;
    if ty.is_str("Call") {
        return Ok(node.clone());
    }
    if ty.is_str("Chain") {
        let chain = node.get("chain")?;
        let head = if chain.truthy() {
            chain.get("0")?
        } else {
            chain
        };
        return Ok(if head.truthy() && head.get("type")?.is_str("Call") {
            head
        } else {
            V::Null
        });
    }
    Ok(V::Null)
}

/// `buildNamespaceSnapshot(callNamespace)`.
fn build_namespace_snapshot(ns: &V) -> Result<V, JsError> {
    if !ns.truthy() || ns.type_of() != "object" {
        return Ok(V::Null);
    }
    let string_or_null = |v: V| if v.as_str().is_some() { v } else { V::Null };
    let call = V::object_from(vec![
        ("name", string_or_null(ns.get("name")?)),
        ("resolved", string_or_null(ns.get("resolved")?)),
        ("explicit", V::Bool(ns.get("explicit")?.truthy())),
        ("source", string_or_null(ns.get("source")?)),
    ]);
    let snapshot = V::object_from(vec![("call", call.clone())]);
    let search_order = ns.get("searchOrder")?;
    if search_order.is_array() {
        call.set(
            "searchOrder",
            search_order.slice("callNamespace.searchOrder", None)?,
        )?;
    }
    if ns.get("fromOverride")?.truthy() {
        call.set("fromOverride", V::Bool(true))?;
    }
    let resolved = ns.get("resolved")?;
    if resolved.truthy() {
        snapshot.set("resolved", resolved)?;
    }
    Ok(snapshot)
}

/// `new Function('state', \`with(state){ return ${src}; }\`)`: whether V8 accepts
/// the body. Bodies nested deeper than the checker decides are an error (see
/// [`func_syntax`]) rather than a guess.
fn compile_func(src: &V) -> Result<bool, JsError> {
    let body = format!("with(state){{ return {}; }}", src.to_js_string());
    match check_function_body("state", &body) {
        FuncSyntax::Valid => Ok(true),
        FuncSyntax::Invalid(_) => Ok(false),
        FuncSyntax::Undecidable(why) => Err(JsError::error(format!(
            "function body not decided ({why}): () => {}",
            utf16_prefix(&src.to_js_string(), 50)
        ))),
    }
}

/// A function value holding the source text of the reference closure it
/// stands for (what `Function.prototype.toString` returns for it). Nothing ever
/// calls these closures; consumers that print compiled arguments see the
/// reference's text.
fn closure(text: &str) -> V {
    V::Func(Rc::from(text))
}

/// The function `new Function('state', \`with(state){ return ${src}; }\`)`
/// returns, as its `toString()` prints it.
fn compiled_func(src: &V) -> V {
    V::Func(Rc::from(
        format!(
            "function anonymous(state\n) {{\nwith(state){{ return {}; }}\n}}",
            src.to_js_string()
        )
        .as_str(),
    ))
}

// ------------------------------------------------------------ validator hooks

/// What a validator hook receives (`hook({call, originalCall, args, writeName,
/// from, allocateTemp, addStep, addState, pushDiagnostic, states, starter})`).
///
/// The values are snapshots of the reference's live objects; `args` is the
/// argument object the step will carry, and changes a hook makes to it are kept.
pub struct HookContext<'a, 'r> {
    /// The resolved call (`call`).
    pub call: Value,
    /// The AST node of the call (`originalCall`).
    pub original_call: Value,
    /// The resolved arguments (`args`).
    pub args: Value,
    /// The statement's `write()` target name (`writeName`).
    pub write_name: Value,
    /// The step input (`from`).
    pub from: Value,
    /// `getStarterInfo(originalCall)`: `{call, index}` or null (`starter`).
    pub starter: Value,
    validator: &'a mut Validator<'r>,
    chain: &'a mut Vec<V>,
    states: &'a mut Vec<V>,
}

impl HookContext<'_, '_> {
    /// `allocateTemp()`: the next temp index.
    pub fn allocate_temp(&mut self) -> f64 {
        let idx = self.validator.temp_index;
        self.validator.temp_index += 1.0;
        idx
    }

    /// `addStep(step)`: appends an object step to the statement's chain.
    pub fn add_step(&mut self, step: &Value) {
        if matches!(step, Value::Object(_) | Value::Array(_)) {
            self.chain.push(V::from_value(step));
        }
    }

    /// `addState(state)`: appends an object to the plan's `states`.
    pub fn add_state(&mut self, state: &Value) {
        if matches!(state, Value::Object(_) | Value::Array(_)) {
            self.states.push(V::from_value(state));
        }
    }

    /// The plan's `states` so far.
    pub fn states(&self) -> Vec<Value> {
        self.states.iter().map(V::to_value).collect()
    }

    /// `pushDiagnostic(code, node, message)`.
    pub fn push_diagnostic(
        &mut self,
        code: &str,
        node: &Value,
        message: Option<&str>,
    ) -> Result<(), JsError> {
        self.validator
            .push_diag(code, &V::from_value(node), message.map(str::to_owned))
    }
}

/// A hook's verdict (`{handled, current}`): when `handled`, the call adds no step
/// of its own, and a non-null `current` becomes the chain's current temp.
#[derive(Debug, Clone, Default)]
pub struct HookResult {
    pub handled: bool,
    pub current: Value,
}

/// A validator hook (`registerValidatorHook(name, hook)`): called for every call
/// of an effect named `name` after its arguments are resolved.
pub type ValidatorHook = dyn Fn(&mut HookContext<'_, '_>) -> Result<Option<HookResult>, JsError>;

/// The registered validator hooks (`validatorHooks`). The reference host
/// registers none; [`validate`] runs with an empty set.
#[derive(Clone, Default)]
pub struct ValidatorHooks {
    hooks: IndexMap<String, Rc<ValidatorHook>>,
}

impl ValidatorHooks {
    pub fn new() -> Self {
        Self::default()
    }

    /// `registerValidatorHook(name, hook)`.
    pub fn register(&mut self, name: impl Into<String>, hook: Rc<ValidatorHook>) {
        self.hooks.insert(name.into(), hook);
    }

    fn get(&self, name: &str) -> Option<Rc<ValidatorHook>> {
        self.hooks.get(name).cloned()
    }
}

impl std::fmt::Debug for ValidatorHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.hooks.keys()).finish()
    }
}

// ------------------------------------------------------------ validate

/// `validate(ast)`: `{plans, diagnostics, render, vars, searchNamespaces}` (and
/// `trailingComments`), with no validator hooks registered.
pub fn validate(ast: &Value, registry: &Registry) -> Result<Value, JsError> {
    validate_with_hooks(ast, registry, &ValidatorHooks::default())
}

/// `validate(ast)` with the given validator hooks registered.
pub fn validate_with_hooks(
    ast: &Value,
    registry: &Registry,
    hooks: &ValidatorHooks,
) -> Result<Value, JsError> {
    let ast = V::from_value(ast);
    let mut validator = Validator {
        registry,
        hooks,
        std_enums: registry.std_enums(),
        diagnostics: Vec::new(),
        symbols: HashMap::new(),
        program_search_order: V::Undefined,
        temp_index: 0.0,
        reported_automation_cycles: HashSet::new(),
    };
    Ok(validator.validate(&ast)?.to_value())
}

/// The per-statement state of `compileChainStatement` that `processChain`
/// closes over.
struct ChainCtx {
    chain: Vec<V>,
    states: Vec<V>,
    write_name: V,
}

/// The state of one `validate(ast)` call (the variables its inner functions close
/// over).
pub struct Validator<'r> {
    registry: &'r Registry,
    hooks: &'r ValidatorHooks,
    std_enums: Object,
    /// `diagnosticsList`.
    diagnostics: Vec<V>,
    /// `symbols` (a Map keyed by variable name).
    symbols: HashMap<String, V>,
    /// `programSearchOrder`.
    program_search_order: V,
    /// `tempIndex`.
    temp_index: f64,
    /// `reportedAutomationCycles`.
    reported_automation_cycles: HashSet<String>,
}

/// `options` of `resolveAutomationNumber`.
#[derive(Default)]
struct NumberOptions<'a> {
    allow_boolean: bool,
    allow_automation: bool,
    /// `options.allowMember !== false`.
    disallow_member: bool,
    clamp01: bool,
    integer: bool,
    min: Option<f64>,
    max: Option<f64>,
    /// `options.onInvalid`: the flag it clears.
    on_invalid: Option<&'a mut bool>,
}

impl NumberOptions<'_> {
    fn on_invalid(&mut self) {
        if let Some(flag) = self.on_invalid.as_deref_mut() {
            *flag = false;
        }
    }
}

impl<'r> Validator<'r> {
    // -------------------------------------------------------- diagnostics

    /// `pushDiag(code, node, message = diagnostics[code].message)`.
    fn push_diag(&mut self, code: &str, node: &V, message: Option<String>) -> Result<(), JsError> {
        let info = diagnostics::lookup(code);
        let message = match message {
            Some(m) => m,
            None => match info {
                Some(i) => i.message.to_owned(),
                None => {
                    return Err(JsError::type_error(
                        "Cannot read properties of undefined (reading 'message')",
                    ));
                }
            },
        };
        // Enrich message with identifier/location context when available.
        let mut enriched = message.clone();
        let ident_name = Self::extract_identifier_name(node)?;
        if ident_name.truthy() {
            let ident = ident_name.to_js_string();
            if !message.contains(&ident) && !message.contains('\'') {
                enriched = format!("{message}: '{ident}'");
            }
        }
        let mut location = V::Null;
        let loc = node.get_opt("loc");
        if loc.truthy() {
            let line = loc.get("line")?;
            let mut column = loc.get("column")?;
            if column.is_nullish() {
                column = loc.get("col")?;
            }
            location = V::object_from(vec![("line", line), ("column", column)]);
        }
        let Some(info) = info else {
            return Err(JsError::type_error(
                "Cannot read properties of undefined (reading 'severity')",
            ));
        };
        let mut entries = vec![
            ("code", V::str(code)),
            ("message", V::str(&enriched)),
            ("severity", V::str(info.severity)),
            ("nodeId", node.get_opt("id")),
        ];
        if location.truthy() {
            entries.push(("location", location));
        }
        if ident_name.truthy() {
            entries.push(("identifier", ident_name));
        }
        self.diagnostics.push(V::object_from(entries));
        Ok(())
    }

    /// `extractIdentifierName(node)`.
    fn extract_identifier_name(node: &V) -> Result<V, JsError> {
        if !node.truthy() {
            return Ok(V::Null);
        }
        let ty = node.get("type")?;
        if ty.is_str("Ident") {
            return node.get("name");
        }
        if ty.is_str("Member") {
            let path = node.get("path")?;
            if path.is_array() {
                return Ok(V::str(&join(&path, ".")));
            }
        }
        if ty.is_str("Call") {
            return node.get("name");
        }
        if ty.is_str("Func") {
            let src = node.get("src")?;
            if src.truthy() {
                let head = slice_text(&src, "node.src", 30)?.to_js_string();
                let ellipsis = if src.get("length")?.to_number() > 30.0 {
                    "..."
                } else {
                    ""
                };
                return Ok(V::str(&format!("{{{head}{ellipsis}}}")));
            }
        }
        // Fallback: try to extract any name-like property.
        let name = node.get("name")?;
        if name.truthy() {
            return Ok(name);
        }
        let value = node.get("value")?;
        if value.truthy() {
            return Ok(V::str(&value.to_js_string()));
        }
        Ok(V::str(&format!(
            "[{}]",
            if ty.truthy() {
                ty.to_js_string()
            } else {
                "unknown".to_owned()
            }
        )))
    }

    // -------------------------------------------------------- registries

    /// `ops[name]` for a qualified op name.
    fn op_spec(&self, name: &str) -> Option<&'r Value> {
        self.registry.ops.get(name).filter(|spec| spec.is_truthy())
    }

    /// `!!ops[name]` for any name (an ordinary object: `Object.prototype`'s
    /// members are found too).
    fn ops_has(&self, name: &V) -> bool {
        let key = name.to_js_string();
        if self.registry.ops.get(&key).is_some_and(Value::is_truthy) {
            return true;
        }
        V::new_object().get_opt(&key).truthy()
    }

    /// `isStarterOp(name)` for any value.
    fn is_starter_op(&self, name: &V) -> bool {
        name.as_str()
            .is_some_and(|n| self.registry.is_starter_op(n))
    }

    // -------------------------------------------------------- enums

    /// `resolveEnum(path)`.
    fn resolve_enum(&self, path: &V) -> Result<V, JsError> {
        match path.elements() {
            Some(parts) if !parts.is_empty() => self.resolve_enum_parts(&parts),
            _ => Ok(V::Undefined),
        }
    }

    fn resolve_enum_strs(&self, path: &[String]) -> Result<V, JsError> {
        if path.is_empty() {
            return Ok(V::Undefined);
        }
        let parts: Vec<V> = path.iter().map(|s| V::str(s)).collect();
        self.resolve_enum_parts(&parts)
    }

    fn resolve_enum_parts(&self, parts: &[V]) -> Result<V, JsError> {
        /// The walk's current value: in the validator's object graph, or in the
        /// registry's enum tree (converted only when returned).
        enum Cur<'a> {
            Heap(V),
            Reg(&'a Value),
        }
        let head = &parts[0];
        let mut cur = match head.as_str().and_then(|h| self.symbols.get(h)) {
            Some(sym) => {
                let mut c = sym.clone();
                if c.truthy() {
                    let t = c.get("type")?;
                    if t.is_str("Number") || t.is_str("Boolean") {
                        c = c.get("value")?;
                    }
                }
                Cur::Heap(c)
            }
            None => {
                let key = head.to_js_string();
                if let Some(v) = self.registry.enums.get(&key) {
                    Cur::Reg(v)
                } else if let Some(v) = self.std_enums.get(&key) {
                    Cur::Reg(v)
                } else {
                    return Ok(V::Undefined);
                }
            }
        };
        for part in &parts[1..] {
            let key = part.to_js_string();
            cur = match cur {
                Cur::Heap(c) => {
                    if c.truthy() && c.has_own(&key) {
                        Cur::Heap(c.get(&key)?)
                    } else {
                        return Ok(V::Undefined);
                    }
                }
                Cur::Reg(c) => {
                    if !c.is_truthy() {
                        return Ok(V::Undefined);
                    }
                    match c {
                        Value::Object(o) => match o.get(&key) {
                            Some(v) => Cur::Reg(v),
                            None => return Ok(V::Undefined),
                        },
                        other => {
                            let h = V::from_value(other);
                            if h.has_own(&key) {
                                Cur::Heap(h.get(&key)?)
                            } else {
                                return Ok(V::Undefined);
                            }
                        }
                    }
                }
            };
        }
        let cur = match cur {
            Cur::Heap(c) => c,
            Cur::Reg(c) => V::from_value(c),
        };
        if cur.truthy() {
            let t = cur.get("type")?;
            if t.is_str("Number") || t.is_str("Boolean") {
                return cur.get("value");
            }
        }
        Ok(cur)
    }

    /// `isOwnChoice(def, name)`: a bare name the parameter defines itself, as an
    /// inline choice or a member of its enum, means that value even where it
    /// shadows a state value such as `seed` or `a`.
    fn is_own_choice(&self, def: &Value, name: &V) -> Result<bool, JsError> {
        let choices = def.get("choices");
        if choices.is_truthy() && matches!(choices_get(choices, name), V::Num(_)) {
            return Ok(true);
        }
        let enum_path = if def.get("enumPath").is_truthy() {
            def.get("enumPath")
        } else {
            def.get("enum")
        };
        if !enum_path.is_truthy() {
            return Ok(false);
        }
        let prefix = normalize_member_path(enum_path);
        let path = apply_enum_prefix(&[name.to_js_string()], prefix.as_deref());
        Ok(matches!(self.resolve_enum_strs(&path)?, V::Num(_)))
    }

    /// `canResolveOpName(name)`: whether a bare op name resolves via the search
    /// order.
    fn can_resolve_op_name(&self, name: &V) -> Result<bool, JsError> {
        for ns in self.program_search_order_items()? {
            if self
                .registry
                .ops
                .get(&format!("{}.{}", ns.to_js_string(), name.to_js_string()))
                .is_some_and(Value::is_truthy)
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// `for (const ns of programSearchOrder)`.
    fn program_search_order_items(&self) -> Result<Vec<V>, JsError> {
        iterate(&self.program_search_order, Some("programSearchOrder"))
    }

    // -------------------------------------------------------- calls and chains

    /// `resolveCall(call)`: a call of a variable holding a call or an identifier.
    fn resolve_call(&self, call: &V) -> Result<V, JsError> {
        let name = call.get("name")?;
        if let Some(val) = name.as_str().and_then(|n| self.symbols.get(n)) {
            let val = val.clone();
            let vty = val.get("type")?;
            if vty.is_str("Ident") {
                let out = call.spread();
                out.set("name", val.get("name")?)?;
                return Ok(out);
            }
            if vty.is_str("Call") {
                let val_args = val.get("args")?;
                let merged_args = if val_args.truthy() {
                    val_args.slice("val.args", None)?
                } else {
                    V::array_from(Vec::new())
                };
                // Positional arguments are appended to the stored arguments.
                let call_args = call.get("args")?;
                let call_args = if call_args.truthy() {
                    call_args
                } else {
                    V::array_from(Vec::new())
                };
                let mut i = 0usize;
                while less_than(&V::num(i as f64), call_args.get("length")?.to_number()) {
                    // `mergedArgs.push(callArgs[i])` (a string's `slice()` has no `push`)
                    let V::Arr(merged) = &merged_args else {
                        return Err(JsError::type_error("mergedArgs.push is not a function"));
                    };
                    let arg = call_args.get(&i.to_string())?;
                    merged.borrow_mut().push(arg);
                    i += 1;
                }
                let val_kwargs = val.get("kwargs")?;
                let mut merged_kw = if val_kwargs.truthy() {
                    Some(val_kwargs.spread())
                } else {
                    None
                };
                let call_kwargs = call.get("kwargs")?;
                if call_kwargs.truthy() {
                    let kw = merged_kw.get_or_insert_with(V::new_object);
                    for (k, v) in object_entries(&call_kwargs)? {
                        kw.set(&k, v)?;
                    }
                }
                let merged = V::object_from(vec![
                    ("type", V::str("Call")),
                    ("name", val.get("name")?),
                    ("args", merged_args),
                ]);
                if let Some(kw) = merged_kw {
                    merged.set("kwargs", kw)?;
                }
                let call_ns = call.get("namespace")?;
                if call_ns.truthy() {
                    merged.set("namespace", call_ns.spread())?;
                } else {
                    let val_ns = val.get("namespace")?;
                    if val_ns.truthy() {
                        merged.set("namespace", val_ns.spread())?;
                    }
                }
                return Ok(merged);
            }
        }
        Ok(call.clone())
    }

    /// `getStarterInfo(node)`: `{call, index}` of the first starter call, or null.
    fn get_starter_info(&self, node: &V) -> Result<V, JsError> {
        if !node.truthy() || node.type_of() != "object" {
            return Ok(V::Null);
        }
        let ty = node.get("type")?;
        if ty.is_str("Call") {
            let name = starter_query_name(node)?;
            return Ok(if self.is_starter_op(&name) {
                V::object_from(vec![("call", node.clone()), ("index", V::num(0.0))])
            } else {
                V::Null
            });
        }
        if ty.is_str("Chain") {
            let chain = node.get("chain")?;
            if let Some(elems) = chain.elements() {
                for (i, entry) in elems.iter().enumerate() {
                    if entry.truthy() && entry.get("type")?.is_str("Call") {
                        let name = starter_query_name(entry)?;
                        if self.is_starter_op(&name) {
                            return Ok(V::object_from(vec![
                                ("call", entry.clone()),
                                ("index", V::num(i as f64)),
                            ]));
                        }
                    }
                }
            }
        }
        Ok(V::Null)
    }

    /// `isStarterChain(node)`.
    fn is_starter_chain(&self, node: &V) -> Result<bool, JsError> {
        if !node.truthy() || !node.get("type")?.is_str("Chain") {
            return Ok(false);
        }
        let starter = self.get_starter_info(node)?;
        Ok(starter.truthy() && matches!(starter.get("index")?, V::Num(n) if n == 0.0))
    }

    /// `substitute(node, resolving)`: variables replaced by (clones of) their
    /// values, recursively through automation descriptors, chains and calls.
    fn substitute(&mut self, node: &V, resolving: &[String]) -> Result<V, JsError> {
        if !node.truthy() {
            return Ok(node.clone());
        }
        let ty = node.get("type")?;
        if ty.is_str("Ident") {
            let name = node.get("name")?;
            if let Some(n) = name.as_str()
                && let Some(cycle_start) = resolving.iter().position(|r| r == n)
            {
                let mut cycle: Vec<String> = resolving[cycle_start..].to_vec();
                cycle.push(n.to_owned());
                let cycle_key = cycle.join(" -> ");
                if self.reported_automation_cycles.insert(cycle_key.clone()) {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!("Automation cycle detected: {cycle_key}")),
                    )?;
                }
                return Ok(V::object_from(vec![
                    ("type", V::str("Number")),
                    ("value", V::num(0.0)),
                    ("_automationInvalid", V::Bool(true)),
                ]));
            }
            if let Some(n) = name.as_str()
                && let Some(value) = self.symbols.get(n).cloned()
            {
                let mut nested = resolving.to_vec();
                nested.push(n.to_owned());
                let result = self.substitute(&clone(&value), &nested)?;
                if result.truthy() && result.type_of() == "object" {
                    result.set("_varRef", name)?;
                }
                return Ok(result);
            }
        }
        if let Some(fields) = automation_fields(&ty) {
            let mapped = node.spread();
            for field in fields {
                let value = node.get(field)?;
                if !value.is_undefined() {
                    let substituted = self.substitute(&value, resolving)?;
                    mapped.set(field, substituted)?;
                }
            }
            return Ok(mapped);
        }
        if ty.is_str("Chain") {
            let mut mapped = Vec::new();
            for c in array_for(&node.get("chain")?, "node.chain", "map")? {
                let mapped_call = self.substitute_call(&c, "c", resolving)?;
                mapped.push(self.resolve_call(&mapped_call)?);
            }
            return Ok(V::object_from(vec![
                ("type", V::str("Chain")),
                ("chain", V::array_from(mapped)),
            ]));
        }
        if ty.is_str("Call") {
            let mapped_call = self.substitute_call(node, "node", resolving)?;
            return self.resolve_call(&mapped_call);
        }
        Ok(node.clone())
    }

    /// The `mappedCall` of `substitute` for one call: `{type:'Call', name, args,
    /// kwargs?}` with its arguments substituted.
    fn substitute_call(&mut self, c: &V, expr: &str, resolving: &[String]) -> Result<V, JsError> {
        let mut mapped_args = Vec::new();
        for a in array_for(&c.get("args")?, &format!("{expr}.args"), "map")? {
            mapped_args.push(self.substitute(&a, resolving)?);
        }
        let mapped_call = V::object_from(vec![
            ("type", V::str("Call")),
            ("name", c.get("name")?),
            ("args", V::array_from(mapped_args)),
        ]);
        let kwargs = c.get("kwargs")?;
        if kwargs.truthy() {
            let kw = V::new_object();
            for (k, v) in object_entries(&kwargs)? {
                let substituted = self.substitute(&v, resolving)?;
                kw.set(&k, substituted)?;
            }
            mapped_call.set("kwargs", kw)?;
        }
        Ok(mapped_call)
    }

    // -------------------------------------------------------- variables

    /// The `ast.vars` loop: binds every variable in `symbols`.
    fn bind_vars(&mut self, vars: &V) -> Result<(), JsError> {
        let Some(vars) = vars.elements() else {
            return Ok(());
        };
        for v in vars {
            // `substitute(clone(v.expr), [v.name])`: `v.expr` is read first.
            let v_expr = clone(&v.get("expr")?);
            let v_name = v.get("name")?;
            let resolving = vec![v_name.to_js_string()];
            let expr = self.substitute(&v_expr, &resolving)?;
            if expr.truthy() && self.is_starter_chain(&expr)? {
                let head = first_chain_call(&expr)?;
                if head.truthy() {
                    self.push_diag("S006", &head, None)?;
                }
            }
            if expr.is_nullish() || {
                let ty = expr.get("type")?;
                let name = if ty.is_str("Ident") {
                    expr.get("name")?
                } else {
                    V::Undefined
                };
                ty.is_str("Ident") && (name.is_str("null") || name.is_str("undefined"))
            } {
                self.push_diag("S004", &v, None)?;
                continue;
            }
            let ty = expr.get("type")?;
            if ty.is_str("Ident") {
                let name = expr.get("name")?;
                let known = name.as_str().is_some_and(|n| self.symbols.contains_key(n))
                    || has_str(STATE_VALUES, &name)
                    || self.ops_has(&name)
                    || self.can_resolve_op_name(&name)?;
                if !known {
                    self.push_diag("S003", &expr, None)?;
                    continue;
                }
            }
            let key = v_name.to_js_string();
            if ty.is_str("Chain")
                && expr
                    .get("chain")?
                    .get("length")?
                    .strict_equals(&V::num(1.0))
            {
                let head = expr.get("chain")?.get("0")?;
                self.symbols.insert(key, head);
            } else if ty.is_str("Member") {
                let resolved = self.resolve_enum(&expr.get("path")?)?;
                if let V::Num(n) = resolved {
                    self.symbols.insert(
                        key,
                        V::object_from(vec![("type", V::str("Number")), ("value", V::num(n))]),
                    );
                } else if !resolved.is_undefined() {
                    self.symbols.insert(key, resolved);
                } else {
                    self.symbols.insert(key, expr);
                }
            } else {
                self.symbols.insert(key, expr);
            }
        }
        Ok(())
    }

    /// `evalExpr(node)`.
    fn eval_expr(&mut self, node: &V) -> Result<V, JsError> {
        let expr = self.substitute(&clone(node), &[])?;
        if expr.truthy() && self.is_starter_chain(&expr)? {
            let head = first_chain_call(&expr)?;
            if head.truthy() {
                self.push_diag("S006", &head, None)?;
            }
        }
        if expr.truthy() && expr.get("type")?.is_str("Member") {
            let resolved = self.resolve_enum(&expr.get("path")?)?;
            if let V::Num(n) = resolved {
                return Ok(V::object_from(vec![
                    ("type", V::str("Number")),
                    ("value", V::num(n)),
                ]));
            }
            if !resolved.is_undefined() {
                return Ok(resolved);
            }
        }
        Ok(expr)
    }

    // -------------------------------------------------------- automation

    /// `resolveAutomationEnum(node, enumName, fallback, validValues,
    /// descriptorName, fieldName)`.
    #[allow(clippy::too_many_arguments)]
    fn resolve_automation_enum(
        &mut self,
        node: &V,
        enum_name: &str,
        fallback: V,
        valid_values: &[f64],
        descriptor_name: &str,
        field_name: &str,
    ) -> Result<V, JsError> {
        let ty = node.get_opt("type");
        let mut resolved = V::Undefined;
        if ty.is_str("Number") {
            resolved = node.get("value")?;
        } else if ty.is_str("Member") {
            resolved = self.resolve_enum(&node.get("path")?)?;
        } else if ty.is_str("Ident") {
            resolved = self.resolve_enum_parts(&[V::str(enum_name), node.get("name")?])?;
        }
        if resolved.truthy() && resolved.get("type")?.is_str("Number") {
            resolved = resolved.get("value")?;
        }
        if let V::Num(n) = resolved
            && valid_values.contains(&n)
        {
            return Ok(resolved);
        }
        if ty.is_str("String") {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "String literal not allowed for {descriptor_name}() {field_name}"
                )),
            )?;
        } else {
            let message = if descriptor_name == "audio" && field_name == "band" {
                format!(
                    "audio() band must resolve to an integer from 0 to 4 (got {})",
                    resolved.to_js_string()
                )
            } else {
                format!("{descriptor_name}() {field_name} must resolve to a supported enum value")
            };
            self.push_diag("S002", node, Some(message))?;
        }
        Ok(fallback)
    }

    /// `resolveAutomationString(node, descriptorName, fieldName)`.
    fn resolve_automation_string(
        &mut self,
        node: &V,
        descriptor_name: &str,
        field_name: &str,
    ) -> Result<V, JsError> {
        if !node.truthy() {
            return Ok(V::Undefined);
        }
        let allowlist_key = format!("{descriptor_name}.{field_name}");
        if !ALLOWED_STRING_PARAMS.contains(&allowlist_key.as_str()) {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "String parameter '{allowlist_key}' is not allowlisted"
                )),
            )?;
            return Ok(V::Undefined);
        }
        if !node.get("type")?.is_str("String") {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "{descriptor_name}() {field_name} requires a quoted string"
                )),
            )?;
            return Ok(V::Undefined);
        }
        let value = node.get("value")?;
        if value.get("length")?.strict_equals(&V::num(0.0)) {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "{descriptor_name}() {field_name} must not be empty"
                )),
            )?;
            return Ok(V::Undefined);
        }
        Ok(V::str(
            &string_literals::decode_json_string_literal_content(&value.to_js_string()),
        ))
    }

    /// `resolveAutomationNumber(node, descriptorName, fieldName, fallback,
    /// options, depth)`.
    fn resolve_automation_number(
        &mut self,
        node: &V,
        descriptor_name: &str,
        field_name: &str,
        fallback: V,
        mut options: NumberOptions<'_>,
        depth: i64,
    ) -> Result<V, JsError> {
        if !node.truthy() {
            return Ok(fallback);
        }
        macro_rules! reject {
            ($code:expr, $message:expr) => {{
                options.on_invalid();
                self.push_diag($code, node, Some($message))?;
                return Ok(fallback);
            }};
        }
        let ty = node.get("type")?;
        let value = if ty.is_str("Number") {
            node.get("value")?
        } else if options.allow_boolean && ty.is_str("Boolean") {
            V::num(if node.get("value")?.truthy() {
                1.0
            } else {
                0.0
            })
        } else if ty.is_str("Member") && !options.disallow_member {
            let resolved = self.resolve_enum(&node.get("path")?)?;
            if resolved.truthy() && resolved.get("type")?.is_str("Number") {
                resolved.get("value")?
            } else {
                resolved
            }
        } else if automation_fields(&ty).is_some() && options.allow_automation {
            let compiled = self.compile_automation_descriptor(node, depth + 1)?;
            if compiled.get_opt("_invalid").truthy() {
                options.on_invalid();
            }
            return Ok(compiled);
        } else if ty.is_str("String") {
            reject!(
                "S001",
                format!("String literal not allowed for {descriptor_name}() {field_name}")
            );
        } else if ty.is_str("Ident") {
            reject!(
                "S003",
                format!(
                    "Undefined automation source '{}' for {descriptor_name}() {field_name}",
                    node.get("name")?.to_js_string()
                )
            );
        } else {
            reject!(
                "S002",
                format!(
                    "{descriptor_name}() {field_name} must be a number{}",
                    if options.allow_automation {
                        " or automation source"
                    } else {
                        ""
                    }
                )
            );
        };

        let n = match value {
            V::Num(n) if n.is_finite() => n,
            _ => reject!(
                "S002",
                format!("{descriptor_name}() {field_name} must resolve to a finite number")
            ),
        };
        if options.integer && n.fract() != 0.0 {
            reject!(
                "S002",
                format!("{descriptor_name}() {field_name} must be an integer")
            );
        }
        if let Some(min) = options.min
            && n < min
        {
            reject!(
                "S002",
                format!(
                    "{descriptor_name}() {field_name} must be at least {} (got {})",
                    crate::js::number_to_string(min),
                    crate::js::number_to_string(n)
                )
            );
        }
        if let Some(max) = options.max
            && n > max
        {
            reject!(
                "S002",
                format!(
                    "{descriptor_name}() {field_name} must be at most {} (got {})",
                    crate::js::number_to_string(max),
                    crate::js::number_to_string(n)
                )
            );
        }
        if options.clamp01 {
            // Math.max(0, Math.min(1, value))
            return Ok(V::num(if n > 1.0 {
                1.0
            } else if n > 0.0 {
                n
            } else {
                0.0
            }));
        }
        Ok(V::Num(n))
    }

    /// `compileAutomationDescriptor(node, depth)`.
    fn compile_automation_descriptor(&mut self, node: &V, depth: i64) -> Result<V, JsError> {
        if depth > MAX_AUTOMATION_DEPTH {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "Automation nesting exceeds the maximum depth of {MAX_AUTOMATION_DEPTH}"
                )),
            )?;
            return Ok(V::num(0.0));
        }
        let ty = node.get("type")?;
        let level = || NumberOptions {
            allow_boolean: true,
            allow_automation: true,
            ..NumberOptions::default()
        };
        let unit = || NumberOptions {
            allow_boolean: true,
            allow_automation: true,
            clamp01: true,
            ..NumberOptions::default()
        };

        if ty.is_str("Oscillator") {
            let osc_type = self.resolve_automation_enum(
                &node.get("oscType")?,
                "oscKind",
                V::num(0.0),
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "osc",
                "type",
            )?;
            let min = self.resolve_automation_number(
                &node.get("min")?,
                "osc",
                "min",
                V::num(0.0),
                unit(),
                depth,
            )?;
            let max = self.resolve_automation_number(
                &node.get("max")?,
                "osc",
                "max",
                V::num(1.0),
                unit(),
                depth,
            )?;
            let speed = self.resolve_automation_number(
                &node.get("speed")?,
                "osc",
                "speed",
                V::num(1.0),
                level(),
                depth,
            )?;
            let offset = self.resolve_automation_number(
                &node.get("offset")?,
                "osc",
                "offset",
                V::num(0.0),
                level(),
                depth,
            )?;
            let seed = self.resolve_automation_number(
                &node.get("seed")?,
                "osc",
                "seed",
                V::num(1.0),
                level(),
                depth,
            )?;
            let mut entries = vec![
                ("type", V::str("Oscillator")),
                ("oscType", osc_type),
                ("min", min),
                ("max", max),
                ("speed", speed),
                ("offset", offset),
                ("seed", seed),
                ("_ast", node.clone()),
            ];
            let var_ref = node.get("_varRef")?;
            if var_ref.truthy() {
                entries.push(("_varRef", var_ref));
            }
            return Ok(V::object_from(entries));
        }

        if ty.is_str("Midi") {
            let mode = self.resolve_automation_enum(
                &node.get("mode")?,
                "midiMode",
                V::num(4.0),
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "midi",
                "mode",
            )?;
            let node_zone = node.get("zone")?;
            let has_zone = !node_zone.is_undefined();
            let zone = if has_zone {
                self.resolve_automation_enum(
                    &node_zone,
                    "midiZone",
                    V::Undefined,
                    &[0.0, 1.0],
                    "midi",
                    "zone",
                )?
            } else {
                V::Undefined
            };
            let mut valid_selection = !has_zone || !zone.is_undefined();
            let node_members = node.get("members")?;
            let mut members = V::Undefined;
            if !node_members.is_undefined() {
                members = self.resolve_automation_number(
                    &node_members,
                    "midi",
                    "members",
                    V::Undefined,
                    NumberOptions {
                        integer: true,
                        min: Some(1.0),
                        max: Some(15.0),
                        disallow_member: true,
                        on_invalid: Some(&mut valid_selection),
                        ..NumberOptions::default()
                    },
                    depth,
                )?;
                if !has_zone {
                    valid_selection = false;
                }
            }
            let node_channel = node.get("channel")?;
            if has_zone && !node_channel.is_undefined() {
                valid_selection = false;
            }
            let mut valid_channel = true;
            let channel = if has_zone {
                V::Undefined
            } else {
                self.resolve_automation_number(
                    &node_channel,
                    "midi",
                    "channel",
                    V::num(1.0),
                    NumberOptions {
                        integer: true,
                        min: Some(1.0),
                        max: Some(16.0),
                        disallow_member: true,
                        on_invalid: Some(&mut valid_channel),
                        ..NumberOptions::default()
                    },
                    depth,
                )?
            };
            let mut valid_cc = true;
            let node_cc = node.get("cc")?;
            let mode_is = |m: f64| matches!(mode, V::Num(n) if n == m);
            let cc = if !node_cc.is_undefined() || mode_is(5.0) || mode_is(6.0) {
                self.resolve_automation_number(
                    &node_cc,
                    "midi",
                    "cc",
                    V::num(1.0),
                    NumberOptions {
                        integer: true,
                        min: Some(0.0),
                        max: Some(if mode_is(6.0) { 31.0 } else { 127.0 }),
                        disallow_member: true,
                        on_invalid: Some(&mut valid_cc),
                        ..NumberOptions::default()
                    },
                    depth,
                )?
            } else {
                V::Undefined
            };
            let mut nrpn = V::Undefined;
            let node_nrpn = node.get("nrpn")?;
            if !node_nrpn.is_undefined() || mode_is(7.0) {
                if node_nrpn.is_undefined() {
                    self.push_diag(
                        "S002",
                        node,
                        Some("midi() nrpn mode requires a parameter number".to_owned()),
                    )?;
                    valid_selection = false;
                }
                nrpn = self.resolve_automation_number(
                    &node_nrpn,
                    "midi",
                    "nrpn",
                    V::Undefined,
                    NumberOptions {
                        integer: true,
                        min: Some(0.0),
                        max: Some(16382.0),
                        disallow_member: true,
                        on_invalid: Some(&mut valid_selection),
                        ..NumberOptions::default()
                    },
                    depth,
                )?;
            }
            let mut entries = vec![
                ("type", V::str("Midi")),
                ("channel", channel),
                ("mode", mode),
            ];
            if !cc.is_undefined() {
                entries.push(("cc", cc));
            }
            if !nrpn.is_undefined() {
                entries.push(("nrpn", nrpn));
            }
            if has_zone {
                entries.push(("zone", zone));
            }
            if !node_members.is_undefined() {
                entries.push(("members", members));
            }
            if !valid_cc || !valid_channel || !valid_selection {
                entries.push(("_invalid", V::Bool(true)));
            }
            let min = self.resolve_automation_number(
                &node.get("min")?,
                "midi",
                "min",
                V::num(0.0),
                unit(),
                depth,
            )?;
            entries.push(("min", min));
            let max = self.resolve_automation_number(
                &node.get("max")?,
                "midi",
                "max",
                V::num(1.0),
                unit(),
                depth,
            )?;
            entries.push(("max", max));
            let sensitivity = self.resolve_automation_number(
                &node.get("sensitivity")?,
                "midi",
                "sensitivity",
                V::num(1.0),
                level(),
                depth,
            )?;
            entries.push(("sensitivity", sensitivity));
            let name = self.resolve_automation_string(&node.get("name")?, "midi", "name")?;
            entries.push(("name", name));
            let id = self.resolve_automation_string(&node.get("id")?, "midi", "id")?;
            entries.push(("id", id));
            entries.push(("_ast", node.clone()));
            let var_ref = node.get("_varRef")?;
            if var_ref.truthy() {
                entries.push(("_varRef", var_ref));
            }
            return Ok(V::object_from(entries));
        }

        if ty.is_str("Audio") {
            let band = self.resolve_automation_enum(
                &node.get("band")?,
                "audioBand",
                V::Undefined,
                &[0.0, 1.0, 2.0, 3.0, 4.0],
                "audio",
                "band",
            )?;
            let mut valid_min = true;
            let mut valid_max = true;
            let min = self.resolve_automation_number(
                &node.get("min")?,
                "audio",
                "min",
                V::num(0.0),
                NumberOptions {
                    allow_automation: true,
                    disallow_member: true,
                    clamp01: true,
                    on_invalid: Some(&mut valid_min),
                    ..NumberOptions::default()
                },
                depth,
            )?;
            let max = self.resolve_automation_number(
                &node.get("max")?,
                "audio",
                "max",
                V::num(1.0),
                NumberOptions {
                    allow_automation: true,
                    disallow_member: true,
                    clamp01: true,
                    on_invalid: Some(&mut valid_max),
                    ..NumberOptions::default()
                },
                depth,
            )?;
            let mut channel = V::Undefined;
            let mut valid_channel = true;
            let node_channel = node.get("channel")?;
            if !node_channel.is_undefined() {
                let value = node_channel.get("value")?;
                let in_range = node_channel.get("type")?.is_str("Number")
                    && is_integer(&value)
                    && matches!(value, V::Num(n) if (1.0..=32.0).contains(&n));
                if in_range {
                    channel = value;
                } else {
                    valid_channel = false;
                    if node_channel.get("type")?.is_str("String") {
                        self.push_diag(
                            "S001",
                            &node_channel,
                            Some("String literal not allowed for audio() channel".to_owned()),
                        )?;
                    } else {
                        // node.channel.value ?? node.channel.name ?? node.channel.type
                        let mut shown = node_channel.get("value")?;
                        if shown.is_nullish() {
                            shown = node_channel.get("name")?;
                        }
                        if shown.is_nullish() {
                            shown = node_channel.get("type")?;
                        }
                        self.push_diag(
                            "S002",
                            &node_channel,
                            Some(format!(
                                "audio() channel must be a positive integer from 1 to 32 (got {})",
                                shown.to_js_string()
                            )),
                        )?;
                    }
                }
            }
            let node_name = node.get("name")?;
            let node_id = node.get("id")?;
            let name = self.resolve_automation_string(&node_name, "audio", "name")?;
            let id = self.resolve_automation_string(&node_id, "audio", "id")?;
            let valid_name = node_name.is_undefined() || !name.is_undefined();
            let valid_id = node_id.is_undefined() || !id.is_undefined();
            let invalid = band.is_undefined()
                || !valid_min
                || !valid_max
                || !valid_name
                || !valid_id
                || !valid_channel;
            let mut entries = vec![
                ("type", V::str("Audio")),
                ("band", band),
                ("min", min),
                ("max", max),
                ("channel", channel),
                ("name", name),
                ("id", id),
                ("_invalid", V::Bool(invalid)),
                ("_ast", node.clone()),
            ];
            let var_ref = node.get("_varRef")?;
            if var_ref.truthy() {
                entries.push(("_varRef", var_ref));
            }
            return Ok(V::object_from(entries));
        }

        Ok(V::num(0.0))
    }

    // -------------------------------------------------------- control flow

    /// `evalCondition(node)`.
    fn eval_condition(&mut self, node: &V) -> Result<V, JsError> {
        let expr = self.eval_expr(node)?;
        if !expr.truthy() {
            return Ok(V::Bool(false));
        }
        let ty = expr.get("type")?;
        if ty.is_str("Number") {
            return Ok(V::Bool(to_boolean(&expr.get("value")?)));
        }
        if ty.is_str("Boolean") {
            return Ok(V::Bool(expr.get("value")?.truthy()));
        }
        if ty.is_str("Func") {
            let src = expr.get("src")?;
            if compile_func(&src)? {
                return Ok(V::object_from(vec![(
                    "fn",
                    closure("(state) => toBoolean(fn(state))"),
                )]));
            }
            let shown = optional_src_snippet(&src, "expr.src")?;
            self.push_diag(
                "S001",
                &expr,
                Some(format!("Invalid function expression: '{shown}'")),
            )?;
            return Ok(V::Bool(false));
        }
        if ty.is_str("Ident") {
            let name = expr.get("name")?;
            if let Some(value) = name.as_str().and_then(|n| self.symbols.get(n)).cloned() {
                return self.eval_condition(&value);
            }
            if has_str(STATE_VALUES, &name) {
                return Ok(V::object_from(vec![(
                    "fn",
                    closure("(state)=>toBoolean(state[key])"),
                )]));
            }
            self.push_diag("S003", &expr, None)?;
            return Ok(V::Bool(false));
        }
        if ty.is_str("Member") {
            let path = expr.get("path")?;
            let cur = self.resolve_enum(&path)?;
            if matches!(cur, V::Num(_)) || !cur.is_undefined() {
                return Ok(V::Bool(to_boolean(&cur)));
            }
            let joined = optional_join(&path, "expr.path", ".")?;
            let shown = if joined.truthy() {
                joined.to_js_string()
            } else {
                "unknown".to_owned()
            };
            self.push_diag("S001", &expr, Some(format!("Unknown enum path: '{shown}'")))?;
            return Ok(V::Bool(false));
        }
        Ok(V::Bool(false))
    }

    // -------------------------------------------------------- statements

    /// `compileChainStatement(stmt)`.
    fn compile_chain_statement(&mut self, stmt: &V) -> Result<V, JsError> {
        // Check for S006: Starter chain missing write() or write3d().
        let stmt_chain = stmt.get("chain")?;
        let chain_node = V::object_from(vec![
            ("type", V::str("Chain")),
            ("chain", stmt_chain.clone()),
        ]);
        let write = stmt.get("write")?;
        let write3d = stmt.get("write3d")?;
        let has_write = if write.truthy() {
            write.clone()
        } else {
            write3d.clone()
        };
        if !has_write.truthy() && self.is_starter_chain(&chain_node)? {
            self.push_diag("S006", &stmt_chain.get("0")?, None)?;
        }
        // write or write3d target must be explicit
        if !has_write.truthy() {
            self.push_diag(
                "S001",
                &stmt_chain.get("0")?,
                Some("Chain must have explicit write() or write3d() target".to_owned()),
            )?;
            return Ok(V::Null);
        }
        let write_name = if write.truthy() {
            write.get("name")?
        } else {
            V::Null
        };
        let write3d_target = if write3d.truthy() {
            let target = |field: &str, kind: &str| -> Result<V, JsError> {
                let node = write3d.get(field)?;
                let name = node.get_opt("name");
                let name = if name.truthy() { name } else { node };
                Ok(V::object_from(vec![("kind", V::str(kind)), ("name", name)]))
            };
            V::object_from(vec![
                ("tex3d", target("tex3d", "vol")?),
                ("geo", target("geo", "geo")?),
            ])
        } else {
            V::Null
        };
        let mut ctx = ChainCtx {
            chain: Vec::new(),
            states: Vec::new(),
            write_name,
        };
        let final_index = self.process_chain(&mut ctx, &stmt_chain, V::Null, false)?;
        let write_surf = if write.truthy() {
            V::object_from(vec![
                ("kind", V::str("output")),
                ("name", write.get("name")?),
            ])
        } else {
            V::Null
        };
        let plan = V::object_from(vec![
            ("chain", V::array_from(ctx.chain)),
            ("write", write_surf),
            ("write3d", write3d_target),
            ("final", final_index),
            ("states", V::array_from(ctx.states)),
        ]);
        // Preserve plan-level leading comments (from the statement).
        let leading = stmt.get("leadingComments")?;
        if leading.truthy() {
            plan.set("leadingComments", leading)?;
        }
        Ok(plan)
    }

    /// A built-in step (`{op, args, from, temp, builtin: true}`).
    fn builtin_step(
        &mut self,
        ctx: &mut ChainCtx,
        op: &str,
        args: V,
        from: V,
        original: &V,
    ) -> Result<V, JsError> {
        let idx = self.next_temp();
        let step = V::object_from(vec![
            ("op", V::str(op)),
            ("args", args),
            ("from", from),
            ("temp", V::num(idx)),
            ("builtin", V::Bool(true)),
        ]);
        let leading = original.get("leadingComments")?;
        if leading.truthy() {
            step.set("leadingComments", leading)?;
        }
        ctx.chain.push(step);
        Ok(V::num(idx))
    }

    /// `tempIndex++`.
    fn next_temp(&mut self) -> f64 {
        let idx = self.temp_index;
        self.temp_index += 1.0;
        idx
    }

    /// A read3d/write3d reference: `{kind, name}` when `node?.name`, else null.
    fn ref_3d(node: &V, kind_if: &str, typed_kind: &str, other_kind: &str) -> Result<V, JsError> {
        let name = node.get_opt("name");
        if !name.truthy() {
            return Ok(V::Null);
        }
        let kind = if node.get("type")?.is_str(kind_if) {
            typed_kind
        } else {
            other_kind
        };
        Ok(V::object_from(vec![
            ("kind", V::str(kind)),
            ("name", node.get("name")?),
        ]))
    }

    /// `processChain(calls, input, options)`.
    fn process_chain(
        &mut self,
        ctx: &mut ChainCtx,
        calls: &V,
        input: V,
        allow_starterless: bool,
    ) -> Result<V, JsError> {
        let mut current = input;
        let calls = iterate(calls, Some("calls"))?;
        for original in calls {
            let otype = original.get("type")?;

            // Read: the pipeline built-in that reads 2D surfaces (a starter node).
            if otype.is_str("Read") {
                if !current.is_null() {
                    self.push_diag(
                        "S001",
                        &original,
                        Some(
                            "read() is a starter node and cannot be chained inline. Use standalone read() to start a new chain."
                                .to_owned(),
                        ),
                    )?;
                    continue;
                }
                let surface = to_surface(&original.get("surface")?)?;
                if !surface.truthy() {
                    self.push_diag(
                        "S001",
                        &original,
                        Some("read() requires a valid surface reference".to_owned()),
                    )?;
                    continue;
                }
                let step_args = V::object_from(vec![("tex", surface)]);
                if original.get("_skip")?.strict_equals(&V::Bool(true)) {
                    step_args.set("_skip", V::Bool(true))?;
                }
                current = self.builtin_step(ctx, "_read", step_args, V::Null, &original)?;
                continue;
            }

            // Read3D with a geometry: the two-argument starter form.
            if otype.is_str("Read3D") && original.get("geo")?.truthy() {
                if !current.is_null() {
                    self.push_diag(
                        "S001",
                        &original,
                        Some(
                            "read3d() is a starter node and cannot be chained inline. Use standalone read3d() to start a new chain."
                                .to_owned(),
                        ),
                    )?;
                    continue;
                }
                let tex3d = Self::ref_3d(&original.get("tex3d")?, "VolRef", "vol", "tex3d")?;
                let geo = Self::ref_3d(&original.get("geo")?, "GeoRef", "geo", "geo")?;
                if !tex3d.truthy() || !geo.truthy() {
                    self.push_diag(
                        "S001",
                        &original,
                        Some("read3d() as starter requires tex3d and geo references".to_owned()),
                    )?;
                    continue;
                }
                let step_args = V::object_from(vec![("tex3d", tex3d), ("geo", geo)]);
                if original.get("_skip")?.strict_equals(&V::Bool(true)) {
                    step_args.set("_skip", V::Bool(true))?;
                }
                current = self.builtin_step(ctx, "_read3d", step_args, V::Null, &original)?;
                continue;
            }

            // Write: writes to a surface and passes through (chainable).
            if otype.is_str("Write") {
                let surface = to_surface(&original.get("surface")?)?;
                if !surface.truthy() {
                    self.push_diag(
                        "S001",
                        &original,
                        Some("write() requires a valid surface reference".to_owned()),
                    )?;
                    continue;
                }
                if current.is_null() {
                    self.push_diag(
                        "S005",
                        &original,
                        Some("write() requires an input - cannot be first in chain".to_owned()),
                    )?;
                    continue;
                }
                let args = V::object_from(vec![("tex", surface)]);
                current = self.builtin_step(ctx, "_write", args, current, &original)?;
                continue;
            }

            // Write3D: writes 3D volumes and geometry (chainable).
            if otype.is_str("Write3D") {
                let tex3d = Self::ref_3d(&original.get("tex3d")?, "VolRef", "vol", "tex3d")?;
                let geo = Self::ref_3d(&original.get("geo")?, "GeoRef", "geo", "geo")?;
                if !tex3d.truthy() || !geo.truthy() {
                    self.push_diag(
                        "S001",
                        &original,
                        Some("write3d() requires tex3d and geo references".to_owned()),
                    )?;
                    continue;
                }
                if current.is_null() {
                    self.push_diag(
                        "S005",
                        &original,
                        Some("write3d() requires an input - cannot be first in chain".to_owned()),
                    )?;
                    continue;
                }
                let args = V::object_from(vec![("tex3d", tex3d), ("geo", geo)]);
                current = self.builtin_step(ctx, "_write3d", args, current, &original)?;
                continue;
            }

            // Subchain: a first-class grouping of contiguous effects.
            if otype.is_str("Subchain") {
                // Surface parser-attached subchain-argument reports (GAP-027) once
                // per subchain node, in source order.
                if let Some(reports) = original.get("subchainArgumentDiagnostics")?.elements() {
                    for report in reports {
                        let mut entries = vec![
                            ("code", report.get("code")?),
                            ("message", report.get("message")?),
                            ("severity", report.get("severity")?),
                            ("nodeId", original.get_opt("id")),
                        ];
                        let location = report.get("location")?;
                        if location.truthy() {
                            entries.push(("location", location));
                        }
                        self.diagnostics.push(V::object_from(entries));
                    }
                }
                if current.is_null() {
                    self.push_diag(
                        "S005",
                        &original,
                        Some("subchain() requires an input - cannot be first in chain".to_owned()),
                    )?;
                    continue;
                }
                let marker_args = || -> Result<V, JsError> {
                    let name = original.get("name")?;
                    let id = original.get("id")?;
                    Ok(V::object_from(vec![
                        ("name", if name.truthy() { name } else { V::Null }),
                        ("id", if id.truthy() { id } else { V::Null }),
                    ]))
                };
                current =
                    self.builtin_step(ctx, "_subchain_begin", marker_args()?, current, &original)?;
                // The body reuses all the argument resolution and validation.
                current = self.process_chain(ctx, &original.get("body")?, current, false)?;
                let end_idx = self.next_temp();
                ctx.chain.push(V::object_from(vec![
                    ("op", V::str("_subchain_end")),
                    ("args", marker_args()?),
                    ("from", current),
                    ("temp", V::num(end_idx)),
                    ("builtin", V::Bool(true)),
                ]));
                current = V::num(end_idx);
                continue;
            }

            if let Some(next) = self.process_call(ctx, &original, &current, allow_starterless)? {
                current = next;
            }
        }
        Ok(current)
    }

    /// One effect call of `processChain`: returns the new `current`, or `None`
    /// when the call is skipped (`continue` without a step).
    fn process_call(
        &mut self,
        ctx: &mut ChainCtx,
        original: &V,
        current: &V,
        allow_starterless: bool,
    ) -> Result<Option<V>, JsError> {
        let call = self.resolve_call(&original.spread())?;
        let call_name = call.get("name")?;
        let call_ns = call.get("namespace")?;
        let effective_namespace = if call_ns.truthy() {
            call_ns.clone()
        } else {
            V::object_from(vec![("searchOrder", self.program_search_order.clone())])
        };

        let mut candidate_names: Vec<String> = Vec::new();
        if call_ns.truthy() {
            let resolved = call_ns.get("resolved")?;
            if resolved.truthy() {
                candidate_names.push(format!(
                    "{}.{}",
                    resolved.to_js_string(),
                    call_name.to_js_string()
                ));
            }
        }
        if let Some(search_order) = effective_namespace.get("searchOrder")?.elements() {
            for ns in search_order {
                candidate_names.push(format!(
                    "{}.{}",
                    ns.to_js_string(),
                    call_name.to_js_string()
                ));
            }
        }
        let mut found: Option<(String, &'r Value)> = None;
        for candidate in candidate_names {
            if !candidate.is_empty()
                && let Some(spec) = self.op_spec(&candidate)
            {
                found = Some((candidate, spec));
                break;
            }
        }
        let Some((op_name, spec)) = found else {
            self.push_diag(
                "S001",
                original,
                Some(format!("Unknown effect: '{}'", call_name.to_js_string())),
            )?;
            return Ok(None);
        };
        // Check for deprecated effect aliases.
        if let Some(warning) = aliases::check_effect_alias(self.registry, &op_name) {
            self.push_diag("S008", original, Some(warning))?;
        }
        if op_name == "prev" {
            let idx = self.next_temp();
            let args = V::object_from(vec![(
                "tex",
                V::object_from(vec![
                    ("kind", V::str("output")),
                    ("name", ctx.write_name.clone()),
                ]),
            )]);
            let namespace_snapshot = build_namespace_snapshot(&call_ns)?;
            let step = V::object_from(vec![
                ("op", V::str(&op_name)),
                ("args", args),
                ("from", current.clone()),
                ("temp", V::num(idx)),
            ]);
            if namespace_snapshot.truthy() {
                step.set("namespace", namespace_snapshot)?;
            }
            let leading = original.get("leadingComments")?;
            if leading.truthy() {
                step.set("leadingComments", leading)?;
            }
            ctx.chain.push(step);
            return Ok(Some(V::num(idx)));
        }
        let is_starter = self.registry.is_starter_op(&op_name);
        let starterless_root = current.is_null();
        let allow_passthrough_root =
            allow_starterless && SURFACE_PASSTHROUGH_CALLS.contains(&op_name.as_str());
        if starterless_root && !is_starter && !allow_passthrough_root {
            self.push_diag("S005", original, None)?;
            return Ok(None);
        }
        // Use the already-resolved isStarter, not getStarterInfo (bare name).
        let starter_has_input = is_starter && !current.is_null();
        let from_input = if starter_has_input {
            V::Null
        } else {
            current.clone()
        };
        if starter_has_input {
            self.push_diag("S005", original, None)?;
        }
        let args = V::new_object();
        // Sidecar map remembering the source form of each arg ('array' for a
        // literal `[…]`); absent until something sets a form.
        let mut arg_sources: Option<V> = None;
        let kw = call.get("kwargs")?;
        // Resolve deprecated param aliases.
        if kw.truthy() {
            for warning in aliases::resolve_param_aliases(self.registry, &op_name, &kw)? {
                self.push_diag("S007", &call, Some(warning))?;
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        let spec_args = spec.get("args");
        let spec_args: &[Value] = match spec_args {
            Value::Array(a) => a,
            _ => &[],
        };
        let call_args = call.get("args")?;
        let mut i = 0usize;
        while i < spec_args.len() {
            let def = &spec_args[i];
            let def_name = def.get("name");
            let arg_key = crate::js::value_to_property_key(def_name);
            let kw_value = if kw.truthy() {
                kw.get(&arg_key)?
            } else {
                V::Undefined
            };
            let node = if kw.truthy() && !kw_value.is_undefined() {
                kw_value.clone()
            } else {
                call_args.get(&i.to_string())?
            };
            let node = self.substitute(&node, &[])?;
            let ntype = if node.truthy() {
                node.get("type")?
            } else {
                V::Undefined
            };
            let def_type = def.get("type");
            if !kw.truthy()
                && node.truthy()
                && ntype.is_str("Color")
                && def_type.as_str() != Some("color")
                && def_name.as_str() == Some("r")
                && spec_args
                    .get(i + 1)
                    .map(|d| d.get("name").as_str() == Some("g"))
                    .unwrap_or(false)
                && spec_args
                    .get(i + 2)
                    .map(|d| d.get("name").as_str() == Some("b"))
                    .unwrap_or(false)
            {
                // `const [r, g, b] = node.value`
                let rgb = iterate(&node.get("value")?, None)?;
                let component = |k: usize| rgb.get(k).cloned().unwrap_or_default();
                args.set(&arg_key, component(0))?;
                let g_key = crate::js::value_to_property_key(spec_args[i + 1].get("name"));
                args.set(&g_key, component(1))?;
                let b_key = crate::js::value_to_property_key(spec_args[i + 2].get("name"));
                args.set(&b_key, component(2))?;
                i += 3;
                continue;
            }
            if kw.truthy() && !kw_value.is_undefined() {
                seen.insert(arg_key.clone());
            }
            let resolved = self.resolve_arg(
                ctx,
                &call,
                original,
                &op_name,
                spec_args,
                def,
                &arg_key,
                &node,
                &args,
                &mut arg_sources,
            )?;
            if let Some(value) = resolved {
                args.set(&arg_key, value)?;
            }
            i += 1;
        }

        // Handle _skip meta-argument (skip this step in the pipeline).
        if kw.truthy() && !kw.get("_skip")?.is_undefined() {
            let skip_node = kw.get("_skip")?;
            if skip_node.truthy() && skip_node.get("type")?.is_str("Boolean") {
                args.set("_skip", skip_node.get("value")?)?;
            } else {
                args.set("_skip", V::Bool(false))?;
            }
            seen.insert("_skip".to_owned());
        }

        if kw.truthy() {
            for key in object_keys(&kw)? {
                if !seen.contains(&key) {
                    self.push_diag(
                        "S001",
                        &kw.get(&key)?,
                        Some(format!(
                            "Unknown argument '{key}' for {}()",
                            call_name.to_js_string()
                        )),
                    )?;
                }
            }
        }

        let hook = call_name.as_str().and_then(|n| self.hooks.get(n));
        if let Some(hook) = hook {
            let starter_info = self.get_starter_info(original)?;
            let args_before = args.to_value();
            let mut hook_ctx = HookContext {
                call: call.to_value(),
                original_call: original.to_value(),
                args: args_before.clone(),
                write_name: ctx.write_name.to_value(),
                from: from_input.to_value(),
                starter: starter_info.to_value(),
                validator: self,
                chain: &mut ctx.chain,
                states: &mut ctx.states,
            };
            let hook_result = hook(&mut hook_ctx)?;
            let args_after = std::mem::take(&mut hook_ctx.args);
            drop(hook_ctx);
            if args_after != args_before
                && let V::Obj(target) = &args
                && let V::Obj(fresh) = V::from_value(&args_after)
            {
                *target.borrow_mut() = fresh.borrow().clone();
            }
            if let Some(result) = hook_result
                && result.handled
            {
                if !result.current.is_nullish() {
                    return Ok(Some(V::from_value(&result.current)));
                }
                return Ok(None);
            }
        }

        let idx = self.next_temp();
        let namespace_snapshot = build_namespace_snapshot(&call_ns)?;
        let step = V::object_from(vec![
            ("op", V::str(&op_name)),
            ("args", args),
            ("from", from_input),
            ("temp", V::num(idx)),
        ]);
        if namespace_snapshot.truthy() {
            step.set("namespace", namespace_snapshot)?;
        }
        let leading = original.get("leadingComments")?;
        if leading.truthy() {
            step.set("leadingComments", leading)?;
        }
        // Preserve raw kwargs from the original AST for automation UI extraction.
        let raw_kwargs = original.get("kwargs")?;
        if raw_kwargs.truthy() && !object_keys(&raw_kwargs)?.is_empty() {
            step.set("rawKwargs", raw_kwargs)?;
        }
        // Sidecar source-form metadata for unparser round-trip.
        if let Some(sources) = arg_sources {
            step.set("argSources", sources)?;
        }
        ctx.chain.push(step);
        Ok(Some(V::num(idx)))
    }

    /// The value of one parameter (the body of the `specArgs` loop after the
    /// packed-color form): `Some(value)` for `args[argKey] = value`; `None` when
    /// the reference assigns nothing.
    #[allow(clippy::too_many_arguments)]
    fn resolve_arg(
        &mut self,
        ctx: &mut ChainCtx,
        call: &V,
        original: &V,
        op_name: &str,
        spec_args: &[Value],
        def: &Value,
        arg_key: &str,
        node: &V,
        args: &V,
        arg_sources: &mut Option<V>,
    ) -> Result<Option<V>, JsError> {
        let ntype = if node.truthy() {
            node.get("type")?
        } else {
            V::Undefined
        };
        let def_name = V::from_value(def.get("name"));
        let def_default = def.get("default");
        let call_name = call.get("name")?;
        let shown_call = call_name.to_js_string();
        let shown_def = def_name.to_js_string();
        let is_string_node = node.truthy() && ntype.is_str("String");
        let out_of_range = || format!("Argument out of range for '{shown_def}' in {shown_call}()");

        // Array literal: the additive `[…]` input form, passed through as a numeric
        // array with its source form recorded.
        if node.truthy() && ntype.is_str("ArrayLiteral") {
            let mut value = Vec::new();
            let elements = iterate(&node.get("elements")?, Some("node.elements"))?;
            for el in elements {
                if el.get("type")?.is_str("Number") {
                    value.push(el.get("value")?);
                } else {
                    self.push_diag(
                        "S002",
                        &el,
                        Some(format!(
                            "Array element must be a number for '{shown_def}' in {shown_call}()"
                        )),
                    )?;
                    value.push(V::num(0.0));
                }
            }
            let sources = arg_sources.get_or_insert_with(V::new_object);
            sources.set(arg_key, V::str("array"))?;
            return Ok(Some(V::array_from(value)));
        }

        match def.get("type").as_str() {
            Some("surface") => {
                if is_string_node {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "String literal not allowed for surface parameter '{shown_def}'"
                        )),
                    )?;
                    return Ok(Some(if def_default.is_truthy() {
                        to_surface(&ident_node(def_default))?
                    } else {
                        V::Null
                    }));
                }
                let mut surf = V::Null;
                let mut invalid_starter_chain = false;
                let starter = if node.truthy() {
                    self.get_starter_info(node)?
                } else {
                    V::Null
                };
                // Handle Read nodes (the parser's node for read() calls).
                if node.truthy() && ntype.is_str("Read") {
                    let surface = node.get("surface")?;
                    if surface.truthy() {
                        surf = to_surface(&surface)?;
                    }
                }
                let inline_surface = if surf.truthy() {
                    surf.clone()
                } else {
                    call_to_surface(node)?
                };
                if inline_surface.truthy() {
                    surf = inline_surface;
                } else if node.truthy() && ntype.is_str("Chain") {
                    let idx = self.process_chain(ctx, &node.get("chain")?, V::Null, true)?;
                    if !idx.is_nullish() {
                        surf = V::object_from(vec![("kind", V::str("temp")), ("index", idx)]);
                    }
                } else if node.truthy() && ntype.is_str("Call") {
                    let idx =
                        self.process_chain(ctx, &V::array_from(vec![node.clone()]), V::Null, true)?;
                    if !idx.is_nullish() {
                        surf = V::object_from(vec![("kind", V::str("temp")), ("index", idx)]);
                    }
                } else if starter.truthy() {
                    self.push_diag("S005", &starter.get("call")?, None)?;
                    invalid_starter_chain = true;
                } else {
                    surf = to_surface(node)?;
                }
                if !surf.truthy() {
                    if invalid_starter_chain {
                        return Ok(Some(surf));
                    }
                    // Only report an error if there's no default to fall back to.
                    if !def_default.is_truthy() {
                        if !node.truthy() {
                            self.push_diag(
                                "S001",
                                call,
                                Some(format!(
                                    "Missing required surface argument '{shown_def}' for {shown_call}()"
                                )),
                            )?;
                        } else if ntype.is_str("Ident")
                            && !node
                                .get("name")?
                                .as_str()
                                .is_some_and(|n| self.symbols.contains_key(n))
                        {
                            self.push_diag(
                                "S003",
                                node,
                                Some(format!(
                                    "Undefined variable '{}' for '{shown_def}' in {shown_call}()",
                                    node.get("name")?.to_js_string()
                                )),
                            )?;
                        } else {
                            // node.name || node.path?.join('.') || node.value || node.type || 'invalid'
                            let mut shown = node.get("name")?;
                            if !shown.truthy() {
                                shown = optional_join(&node.get("path")?, "node.path", ".")?;
                            }
                            if !shown.truthy() {
                                shown = node.get("value")?;
                            }
                            if !shown.truthy() {
                                shown = node.get("type")?;
                            }
                            if !shown.truthy() {
                                shown = V::str("invalid");
                            }
                            self.push_diag(
                                "S001",
                                node,
                                Some(format!(
                                    "Invalid surface reference '{}' for '{shown_def}' in {shown_call}()",
                                    shown.to_js_string()
                                )),
                            )?;
                        }
                    }
                    // Fall back to the default surface when resolution fails.
                    if def_default.is_truthy() {
                        let s = to_surface(&ident_node(def_default))?;
                        surf = if s.truthy() {
                            s
                        } else {
                            V::object_from(vec![
                                ("kind", V::str("pipeline")),
                                ("name", V::from_value(def_default)),
                            ])
                        };
                    }
                }
                Ok(Some(surf))
            }
            Some("color") => {
                if is_string_node {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "String literal not allowed for color parameter '{shown_def}'"
                        )),
                    )?;
                    return Ok(Some(V::from_value(def_default)));
                }
                let value = if node.truthy() && ntype.is_str("Color") {
                    // Keep hex colors as hex strings (e.g., "#ff0000").
                    let hex = node.get("hex")?;
                    if hex.truthy() {
                        hex
                    } else {
                        node.get("value")?
                    }
                } else {
                    if node.truthy() && ntype.truthy() && !ntype.is_str("Ident") {
                        self.push_diag("S002", node, Some(out_of_range()))?;
                    }
                    V::from_value(def_default)
                };
                Ok(Some(value))
            }
            Some(vec @ ("vec3" | "vec4")) => {
                let (n, zero): (usize, Vec<f64>) = if vec == "vec3" {
                    (3, vec![0.0, 0.0, 0.0])
                } else {
                    (4, vec![0.0, 0.0, 0.0, 1.0])
                };
                let default_copy = || -> Result<V, JsError> {
                    if def_default.is_truthy() {
                        V::from_value(def_default).slice("def.default", None)
                    } else {
                        Ok(V::array_from(zero.iter().map(|z| V::num(*z)).collect()))
                    }
                };
                if is_string_node {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "String literal not allowed for {vec} parameter '{shown_def}'"
                        )),
                    )?;
                    return Ok(Some(default_copy()?));
                }
                let node_args = if node.truthy() {
                    node.get("args")?
                } else {
                    V::Undefined
                };
                let value = if node.truthy()
                    && ntype.is_str("Call")
                    && node.get("name")?.is_str(vec)
                    && node_args.truthy()
                    && node_args.get("length")?.strict_equals(&V::num(n as f64))
                {
                    let mut value = Vec::new();
                    for arg in array_for(&node_args, "node.args", "map")? {
                        if arg.get("type")?.is_str("Number") {
                            value.push(arg.get("value")?);
                        } else {
                            self.push_diag("S002", &arg, Some(out_of_range()))?;
                            value.push(V::num(0.0));
                        }
                    }
                    V::array_from(value)
                } else if node.truthy() && ntype.is_str("Color") {
                    // `node.value.slice(0, 3)` (vec3), `node.value.slice()` (vec4)
                    let end = if vec == "vec3" { Some(3) } else { None };
                    node.get("value")?.slice("node.value", end)?
                } else {
                    if node.truthy() && ntype.truthy() && !ntype.is_str("Ident") {
                        self.push_diag("S002", node, Some(out_of_range()))?;
                    }
                    default_copy()?
                };
                Ok(Some(value))
            }
            Some("boolean") => {
                let default_bool =
                    || V::Bool(!def_default.is_undefined() && def_default.is_truthy());
                if is_string_node {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "String literal not allowed for boolean parameter '{shown_def}'"
                        )),
                    )?;
                    return Ok(Some(default_bool()));
                }
                let value = if node.truthy() && ntype.is_str("Boolean") {
                    V::Bool(node.get("value")?.truthy())
                } else if node.truthy() && ntype.is_str("Number") {
                    V::Bool(!node.get("value")?.strict_equals(&V::num(0.0)))
                } else if node.truthy() && ntype.is_str("Func") {
                    let src = node.get("src")?;
                    if compile_func(&src)? {
                        V::object_from(vec![("fn", closure("(state) => !!fn(state)"))])
                    } else {
                        let shown = func_snippet(&src)?;
                        self.push_diag(
                            "S001",
                            node,
                            Some(format!("Invalid function for '{shown_def}': '{shown}'")),
                        )?;
                        default_bool()
                    }
                } else if node.truthy()
                    && ntype.is_str("Ident")
                    && has_str(STATE_VALUES, &node.get("name")?)
                {
                    V::object_from(vec![("fn", closure("(state) => !!state[key]"))])
                } else {
                    if node.truthy()
                        && ntype.is_str("Ident")
                        && !has_str(STATE_VALUES, &node.get("name")?)
                    {
                        self.push_diag("S003", node, None)?;
                    } else if node.truthy() && ntype.truthy() && !ntype.is_str("Ident") {
                        self.push_diag("S002", node, Some(out_of_range()))?;
                    }
                    default_bool()
                };
                Ok(Some(value))
            }
            Some("member") => {
                self.resolve_member_arg(call, def, &def_name, node, &ntype, is_string_node)
            }
            Some("volume") | Some("geometry") => {
                let (kind, noun, prefix) = if def.get("type").as_str() == Some("volume") {
                    ("vol", "volume", "vol")
                } else {
                    ("geo", "geometry", "geo")
                };
                let default_ref = || {
                    if def_default.is_truthy() {
                        V::object_from(vec![
                            ("kind", V::str(kind)),
                            ("name", V::from_value(def_default)),
                        ])
                    } else {
                        V::Null
                    }
                };
                if is_string_node {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "String literal not allowed for {noun} parameter '{shown_def}'"
                        )),
                    )?;
                    return Ok(Some(default_ref()));
                }
                let mut value = V::Null;
                let ref_type = if kind == "vol" { "VolRef" } else { "GeoRef" };
                if node.truthy()
                    && ntype.is_str("Read3D")
                    && node.get("tex3d")?.truthy()
                    && !node.get("geo")?.truthy()
                {
                    // read3d(vol0) / read3d(geo0) in parameter position.
                    let ref_name = node.get("tex3d")?.get("name")?;
                    if is_ref_name(&ref_name, prefix) {
                        value = V::object_from(vec![("kind", V::str(kind)), ("name", ref_name)]);
                    } else {
                        self.push_diag(
                            "S001",
                            node,
                            Some(format!(
                                "Invalid {noun} reference '{}' in read3d() for '{shown_def}' - expected {prefix}0-{prefix}7",
                                ref_name.to_js_string()
                            )),
                        )?;
                        value = default_ref();
                    }
                } else if node.truthy() && ntype.is_str(ref_type) {
                    value =
                        V::object_from(vec![("kind", V::str(kind)), ("name", node.get("name")?)]);
                } else if node.truthy() && ntype.is_str("Ident") {
                    let name = node.get("name")?;
                    if name.is_str("none") {
                        value =
                            V::object_from(vec![("kind", V::str(kind)), ("name", V::str("none"))]);
                    } else if is_ref_name(&name, prefix) {
                        value = V::object_from(vec![("kind", V::str(kind)), ("name", name)]);
                    } else {
                        self.push_diag(
                            "S001",
                            node,
                            Some(format!(
                                "Invalid {noun} reference '{}' for '{shown_def}' - expected {prefix}0-{prefix}7 or none",
                                name.to_js_string()
                            )),
                        )?;
                        value = default_ref();
                    }
                } else if !node.truthy() && def_default.is_truthy() {
                    value = default_ref();
                }
                Ok(Some(value))
            }
            Some("string") => {
                // STRICT STRING PARAMETER VALIDATION
                let func_name = if op_name.contains('.') {
                    op_name.rsplit('.').next().unwrap_or("")
                } else {
                    op_name
                };
                let allowlist_key = format!("{func_name}.{shown_def}");
                if !ALLOWED_STRING_PARAMS.contains(&allowlist_key.as_str()) {
                    self.push_diag(
                        "S001",
                        if node.truthy() { node } else { original },
                        Some(format!(
                            "String parameter '{shown_def}' on effect '{func_name}' is NOT in the allowed string params list. String params are strictly controlled - use enums or choices instead."
                        )),
                    )?;
                    return Ok(Some(V::from_value(def_default)));
                }
                // String type parameters only accept String AST nodes.
                let choices = def.get("choices");
                let value = if is_string_node {
                    node.get("value")?
                } else if node.truthy() && ntype.is_str("Ident") && choices.is_truthy() {
                    // Allow bare identifiers if they match a choice key.
                    let name = node.get("name")?;
                    let choice = choices_get(choices, &name);
                    if !choice.is_undefined() {
                        choice
                    } else {
                        self.push_diag(
                            "S001",
                            node,
                            Some(format!(
                                "Invalid choice '{}' for string parameter '{shown_def}'",
                                name.to_js_string()
                            )),
                        )?;
                        V::from_value(def_default)
                    }
                } else if node.truthy() {
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "String parameter '{shown_def}' requires a quoted string literal, got {}",
                            ntype.to_js_string()
                        )),
                    )?;
                    V::from_value(def_default)
                } else {
                    V::from_value(def_default)
                };
                Ok(Some(value))
            }
            _ => self.resolve_numeric_arg(
                spec_args,
                def,
                &def_name,
                node,
                &ntype,
                is_string_node,
                call,
                args,
            ),
        }
    }

    /// The `def.type === 'member'` branch.
    fn resolve_member_arg(
        &mut self,
        call: &V,
        def: &Value,
        def_name: &V,
        node: &V,
        ntype: &V,
        is_string_node: bool,
    ) -> Result<Option<V>, JsError> {
        let def_default = def.get("default");
        if is_string_node {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "String literal not allowed for member/enum parameter '{}'",
                    def_name.to_js_string()
                )),
            )?;
            return Ok(Some(V::from_value(def_default)));
        }
        let enum_value = if def.get("enumPath").is_truthy() {
            def.get("enumPath")
        } else {
            def.get("enum")
        };
        let prefix = normalize_member_path(enum_value);
        let mut path: Option<Vec<String>> = None;
        if node.truthy() && ntype.is_str("Member") {
            path = normalize_member_path_v(&node.get("path")?);
        } else if node.truthy() && (ntype.is_str("Number") || ntype.is_str("Boolean")) {
            let value = node.get("value")?;
            return Ok(Some(if ntype.is_str("Boolean") {
                V::num(if value.truthy() { 1.0 } else { 0.0 })
            } else {
                value
            }));
        } else if node.truthy()
            && ntype.is_str("Ident")
            && has_str(STATE_VALUES, &node.get("name")?)
            && !self.is_own_choice(def, &node.get("name")?)?
        {
            return Ok(Some(V::object_from(vec![(
                "fn",
                closure("(state) => state[key]"),
            )])));
        } else if node.truthy() && ntype.is_str("Ident") {
            path = Some(vec![node.get("name")?.to_js_string()]);
        }
        if path.is_none() {
            path = normalize_member_path(def_default);
        }
        let unwrap = |resolved: V| -> Result<V, JsError> {
            let mut resolved = resolved;
            if resolved.truthy() && resolved.get("type")?.is_str("Number") {
                resolved = resolved.get("value")?;
            }
            if resolved.truthy() && resolved.get("type")?.is_str("Boolean") {
                resolved = V::num(if resolved.get("value")?.truthy() {
                    1.0
                } else {
                    0.0
                });
            }
            Ok(resolved)
        };
        let mut resolved = match &path {
            Some(p) => unwrap(self.resolve_enum_strs(p)?)?,
            None => V::Undefined,
        };
        if !matches!(resolved, V::Num(_)) {
            let mut prefixed = apply_enum_prefix(path.as_deref().unwrap_or(&[]), prefix.as_deref());
            if let Some(prefix) = &prefix
                && !path_starts_with(&prefixed, Some(prefix))
            {
                self.push_diag(
                    "S001",
                    if node.truthy() { node } else { call },
                    Some(format!(
                        "Invalid enum value for '{}': expected path starting with '{}'",
                        def_name.to_js_string(),
                        prefix.join(".")
                    )),
                )?;
                prefixed = prefix.clone();
            }
            path = Some(prefixed);
            resolved = unwrap(self.resolve_enum_strs(path.as_deref().unwrap_or(&[]))?)?;
        }
        if !matches!(resolved, V::Num(_)) {
            let fallback = normalize_member_path(def_default);
            let fallback_value = match &fallback {
                Some(p) => unwrap(self.resolve_enum_strs(p)?)?,
                None => V::Undefined,
            };
            resolved = if matches!(fallback_value, V::Num(_)) {
                fallback_value
            } else {
                V::num(0.0)
            };
        }
        if node.truthy()
            && ntype.is_str("Member")
            && let Some(p) = &path
        {
            node.set("path", strs_to_v(p))?;
        }
        Ok(Some(resolved))
    }

    /// The numeric (default) branch of the `specArgs` loop.
    #[allow(clippy::too_many_arguments)]
    fn resolve_numeric_arg(
        &mut self,
        spec_args: &[Value],
        def: &Value,
        def_name: &V,
        node: &V,
        ntype: &V,
        is_string_node: bool,
        call: &V,
        args: &V,
    ) -> Result<Option<V>, JsError> {
        let def_default = def.get("default");
        let (def_min, def_max) = (def.get("min"), def.get("max"));
        let shown_def = def_name.to_js_string();
        let shown_call = call.get("name")?.to_js_string();
        // Numeric types reject String nodes.
        if is_string_node {
            self.push_diag(
                "S001",
                node,
                Some(format!(
                    "String literal not allowed for numeric parameter '{shown_def}' - strings are only valid for type: \"string\" parameters"
                )),
            )?;
            return Ok(Some(V::from_value(def_default)));
        }
        let func_with_bounds = |f: V| {
            V::object_from(vec![
                ("fn", f),
                ("min", V::from_value(def_min)),
                ("max", V::from_value(def_max)),
            ])
        };
        let value = if node.truthy() && (ntype.is_str("Number") || ntype.is_str("Boolean")) {
            let value = if ntype.is_str("Boolean") {
                V::num(if node.get("value")?.truthy() {
                    1.0
                } else {
                    0.0
                })
            } else {
                node.get("value")?
            };
            let clamped = clamp(&value, def_min, def_max);
            if !clamped.strict_equals(&value) {
                self.push_diag(
                    "S002",
                    node,
                    Some(format!(
                        "Argument out of range for '{shown_def}' in {shown_call}() (got {}, clamped to {})",
                        value.to_js_string(),
                        clamped.to_js_string()
                    )),
                )?;
            }
            let mut value = clamped;
            // Preserve the variable reference marker for the unparser round trip.
            let var_ref = node.get("_varRef")?;
            if var_ref.truthy() {
                value = V::object_from(vec![("_varRef", var_ref), ("value", value)]);
            }
            value
        } else if node.truthy() && ntype.is_str("Func") {
            let src = node.get("src")?;
            if compile_func(&src)? {
                func_with_bounds(compiled_func(&src))
            } else {
                let shown = func_snippet(&src)?;
                self.push_diag(
                    "S001",
                    node,
                    Some(format!("Invalid function for '{shown_def}': '{shown}'")),
                )?;
                V::from_value(def_default)
            }
        } else if node.truthy()
            && (ntype.is_str("Oscillator") || ntype.is_str("Midi") || ntype.is_str("Audio"))
        {
            self.compile_automation_descriptor(node, 0)?
        } else if node.truthy() && ntype.is_str("Member") {
            let path = node.get("path")?;
            let cur = self.resolve_enum(&path)?;
            match cur {
                V::Num(_) => {
                    let value = clamp(&cur, def_min, def_max);
                    if !value.strict_equals(&cur) {
                        self.push_diag(
                            "S002",
                            node,
                            Some(format!(
                                "Argument out of range for '{shown_def}' in {shown_call}() (got {}, clamped to {})",
                                cur.to_js_string(),
                                value.to_js_string()
                            )),
                        )?;
                    }
                    value
                }
                V::Bool(b) => {
                    let num = V::num(if b { 1.0 } else { 0.0 });
                    let value = clamp(&num, def_min, def_max);
                    if !value.strict_equals(&num) {
                        self.push_diag(
                            "S002",
                            node,
                            Some(format!(
                                "Argument out of range for '{shown_def}' in {shown_call}() (got {}, clamped to {})",
                                num.to_js_string(),
                                value.to_js_string()
                            )),
                        )?;
                    }
                    value
                }
                _ => {
                    // node?.path?.join('.') || node?.name || 'unknown'
                    let mut shown = optional_join(&path, "node?.path", ".")?;
                    if !shown.truthy() {
                        shown = node.get("name")?;
                    }
                    let shown = if shown.truthy() {
                        shown.to_js_string()
                    } else {
                        "unknown".to_owned()
                    };
                    self.push_diag(
                        "S001",
                        node,
                        Some(format!(
                            "Cannot resolve enum value for '{shown_def}': '{shown}'"
                        )),
                    )?;
                    V::from_value(def_default)
                }
            }
        } else if node.truthy()
            && ntype.is_str("Ident")
            && has_str(STATE_VALUES, &node.get("name")?)
            && !self.is_own_choice(def, &node.get("name")?)?
        {
            V::object_from(vec![
                ("fn", closure("(state) => state[key]")),
                ("min", V::from_value(def_min)),
                ("max", V::from_value(def_max)),
                ("_ast", node.clone()),
            ])
        } else if node.truthy() && ntype.is_str("Ident") && def.get("enum").is_truthy() {
            // Try to resolve the bare identifier within the param's enum path.
            let prefix = normalize_member_path(def.get("enum"));
            let name = node.get("name")?.to_js_string();
            let path: Vec<String> = match prefix {
                Some(mut p) => {
                    p.push(name);
                    p
                }
                None => vec![name],
            };
            let resolved = self.resolve_enum_strs(&path)?;
            if matches!(resolved, V::Num(_)) {
                clamp(&resolved, def_min, def_max)
            } else if resolved.truthy() && resolved.get("type")?.is_str("Number") {
                clamp(&resolved.get("value")?, def_min, def_max)
            } else {
                self.push_diag("S003", node, None)?;
                V::from_value(def_default)
            }
        } else if node.truthy() && ntype.is_str("Ident") && def.get("choices").is_truthy() {
            // Try to resolve the bare identifier against inline choices.
            let choice = choices_get(def.get("choices"), &node.get("name")?);
            if matches!(choice, V::Num(_)) {
                clamp(&choice, def_min, def_max)
            } else {
                self.push_diag("S003", node, None)?;
                V::from_value(def_default)
            }
        } else {
            if node.truthy() && ntype.is_str("Ident") && !has_str(STATE_VALUES, &node.get("name")?)
            {
                self.push_diag("S003", node, None)?;
            } else if node.truthy() && ntype.truthy() && !ntype.is_str("Ident") {
                self.push_diag(
                    "S002",
                    node,
                    Some(format!(
                        "Argument out of range for '{shown_def}' in {shown_call}()"
                    )),
                )?;
            }
            let default_from = def.get("defaultFrom");
            if default_from.is_truthy() {
                // Look up the referenced arg by its DSL name (def.name).
                let reference = spec_args
                    .iter()
                    .find(|d| values_strict_equal(d.get("name"), default_from));
                let ref_key = match reference {
                    Some(r) => crate::js::value_to_property_key(r.get("name")),
                    None => crate::js::value_to_property_key(default_from),
                };
                let current = args.get(&ref_key)?;
                if !current.is_undefined() {
                    current
                } else {
                    V::from_value(def_default)
                }
            } else {
                V::from_value(def_default)
            }
        };
        Ok(Some(value))
    }

    // -------------------------------------------------------- blocks

    /// `compileBlock(body)`.
    fn compile_block(&mut self, body: &V) -> Result<V, JsError> {
        let mut result = Vec::new();
        let items = if body.truthy() {
            iterate(body, Some("(body || [])"))?
        } else {
            Vec::new()
        };
        for s in items {
            let compiled = self.compile_stmt(&s)?;
            if compiled.truthy() {
                result.push(compiled);
            }
        }
        Ok(V::array_from(result))
    }

    /// `compileStmt(stmt)`.
    fn compile_stmt(&mut self, stmt: &V) -> Result<V, JsError> {
        let ty = stmt.get("type")?;
        if ty.is_str("IfStmt") {
            let cond = self.eval_condition(&stmt.get("condition")?)?;
            let then_branch = self.compile_block(&stmt.get("then")?)?;
            let mut elif = Vec::new();
            let elifs = stmt.get("elif")?;
            let elifs = if elifs.truthy() {
                iterate(&elifs, None)?
            } else {
                Vec::new()
            };
            for e in elifs {
                let cond = self.eval_condition(&e.get("condition")?)?;
                let then = self.compile_block(&e.get("then")?)?;
                elif.push(V::object_from(vec![("cond", cond), ("then", then)]));
            }
            let else_branch = self.compile_block(&stmt.get("else")?)?;
            return Ok(V::object_from(vec![
                ("type", V::str("Branch")),
                ("cond", cond),
                ("then", then_branch),
                ("elif", V::array_from(elif)),
                ("else", else_branch),
            ]));
        }
        if ty.is_str("Break") {
            return Ok(V::object_from(vec![("type", V::str("Break"))]));
        }
        if ty.is_str("Continue") {
            return Ok(V::object_from(vec![("type", V::str("Continue"))]));
        }
        if ty.is_str("Return") {
            let node = V::object_from(vec![("type", V::str("Return"))]);
            let value = stmt.get("value")?;
            if value.truthy() {
                let evaluated = self.eval_expr(&value)?;
                node.set("value", evaluated)?;
            }
            return Ok(node);
        }
        self.compile_chain_statement(stmt)
    }

    /// The body of `validate(ast)`.
    fn validate(&mut self, ast: &V) -> Result<V, JsError> {
        let mut plans = Vec::new();
        let render_node = ast.get("render")?;
        let render = if render_node.truthy() {
            render_node.get("name")?
        } else {
            V::Null
        };

        let namespace = ast.get("namespace")?;
        self.program_search_order = namespace.get_opt("searchOrder");
        let search_order = self.program_search_order.clone();
        if !search_order.truthy() || search_order.get("length")?.strict_equals(&V::num(0.0)) {
            return Err(JsError::error(
                "Missing required 'search' directive. Every program must start with 'search <namespace>, ...' to specify namespace search order.",
            ));
        }

        let vars = ast.get("vars")?;
        if vars.is_array() {
            self.bind_vars(&vars)?;
        }

        let stmts = ast.get("plans")?;
        let stmts = if stmts.truthy() {
            iterate(&stmts, None)?
        } else {
            Vec::new()
        };
        for stmt in stmts {
            let compiled = self.compile_stmt(&stmt)?;
            if compiled.truthy() {
                plans.push(compiled);
            }
        }

        // Include original variable declarations for unparsing.
        let vars = if vars.truthy() {
            vars
        } else {
            V::array_from(Vec::new())
        };
        // Include the search order for transform operations.
        let search_namespaces = search_order;
        let result = V::object_from(vec![
            ("plans", V::array_from(plans)),
            (
                "diagnostics",
                V::array_from(std::mem::take(&mut self.diagnostics)),
            ),
            ("render", render),
            ("vars", vars),
            ("searchNamespaces", search_namespaces),
        ]);
        // Preserve trailing comments from the program.
        let trailing = ast.get("trailingComments")?;
        if trailing.truthy() {
            result.set("trailingComments", trailing)?;
        }
        Ok(result)
    }
}

/// `clone(node)`: `JSON.parse(JSON.stringify(node))` for objects.
fn clone(node: &V) -> V {
    if node.truthy() && node.type_of() == "object" {
        node.json_clone()
    } else {
        node.clone()
    }
}

/// `{ type: 'Ident', name }` for a parameter default.
fn ident_node(name: &Value) -> V {
    V::object_from(vec![
        ("type", V::str("Ident")),
        ("name", V::from_value(name)),
    ])
}

/// `node.src?.slice(0, 50) || 'unknown'`.
fn func_snippet(src: &V) -> Result<String, JsError> {
    optional_src_snippet(src, "node.src")
}

/// `` `${<expr>?.slice(0, 50) || 'unknown'}` ``.
fn optional_src_snippet(src: &V, expr: &str) -> Result<String, JsError> {
    if src.is_nullish() {
        return Ok("unknown".to_owned());
    }
    let head = src.slice(&format!("{expr}?"), Some(50))?;
    Ok(if head.truthy() {
        head.to_js_string()
    } else {
        "unknown".to_owned()
    })
}

/// `a === b` for registry values.
fn values_strict_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Undefined, Value::Undefined) | (Value::Null, Value::Null) => true,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Number(x), Value::Number(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_search_directive_throws() {
        let reg = Registry::new();
        let ast = Value::from_json(r#"{"type":"Program","plans":[],"render":null}"#).unwrap();
        let err = validate(&ast, &reg).unwrap_err();
        assert!(
            err.to_string()
                .contains("Missing required 'search' directive")
        );
    }
}
