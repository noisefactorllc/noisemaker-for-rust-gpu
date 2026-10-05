//! Port of `lang/transform.js`: programmatic edits of compiled programs —
//! [`list_steps`], [`replace_effect`], [`get_compatible_replacements`] and the
//! replacement preflight [`predict_replacement`].
//!
//! The reference reads module registries (`ops`, the starter-op set, the runtime
//! effect registry, the parameter-alias registry); the port takes them from an
//! explicit [`Registry`]. Arguments the reference reads as plain JavaScript
//! data (`options`, `newArgs`, effect names, step indices) are [`Value`]s with
//! the reference's coercion rules, and results are the reference's result
//! objects (`undefined` members included).

use std::rc::Rc;

use crate::error::JsError;
use crate::registry::{EffectEntry, Registry};
use crate::unparser::jsv::{
    self, cannot_read, entries, entries_strict, get, get_opt, in_operator, iterate, iterate_anon,
    keys, math_round, member, not_a_function, object_member, same_value_zero, set_plain,
    strict_equals, to_number, to_property_key, to_string, values,
};
use crate::value::{Object, Value};

static NULL: Value = Value::Null;

/// Build a plain object from `(key, value)` pairs in order.
fn obj<const N: usize>(members: [(&str, Value); N]) -> Value {
    let mut o = Object::new();
    for (k, v) in members {
        o.insert(k, v);
    }
    Value::Object(o)
}

/// `{ success: false, error }`.
fn failure(error: String) -> Value {
    obj([
        ("success", Value::Bool(false)),
        ("error", Value::String(error)),
    ])
}

/// `a || b`.
fn or(a: Value, b: impl FnOnce() -> Value) -> Value {
    if a.is_truthy() { a } else { b() }
}

/// `options = {}` default.
fn default_object(v: &Value) -> Value {
    if v.is_undefined() {
        Value::Object(Object::new())
    } else {
        v.clone()
    }
}

/// `s.<method>(...)` on a value that must be a string.
fn str_of<'v>(s: &'v Value, expr: &str, method: &str) -> Result<&'v str, JsError> {
    match s {
        Value::String(text) => Ok(text),
        Value::Undefined | Value::Null => Err(cannot_read(s, method)),
        _ => Err(not_a_function(&format!("{expr}.{method}"))),
    }
}

/// `x.length > 0`.
fn length_positive(x: &Value) -> Result<bool, JsError> {
    Ok(to_number(&member(x, "length"))? > 0.0)
}

/// `ops[name]` (own members, else `Object.prototype` members).
fn op_spec(registry: &Registry, name: &Value) -> Result<Value, JsError> {
    Ok(object_member(&registry.ops, &to_property_key(name)?))
}

/// `getEffect(name)`: the registered effect. Map keys compare with
/// SameValueZero, so only string names can match.
fn effect_entry(registry: &Registry, name: &Value) -> Option<Rc<EffectEntry>> {
    match name {
        Value::String(s) => registry.get_effect(s).cloned(),
        _ => None,
    }
}

/// `getEffectInstance(name)` (`getEffect(name) || null`) as a borrowed value.
fn instance_of(entry: &Option<Rc<EffectEntry>>) -> &Value {
    entry.as_ref().map(|e| &e.def).unwrap_or(&NULL)
}

/// `container[key]` for assignment through a cloned program (objects by key,
/// arrays by index).
fn index_mut<'v>(container: &'v mut Value, key: &str) -> Option<&'v mut Value> {
    match container {
        Value::Object(o) => o.get_mut(key),
        Value::Array(a) => key.parse::<usize>().ok().and_then(|i| a.get_mut(i)),
        _ => None,
    }
}

/// `deepClone(obj)`: plain objects and arrays are copied recursively.
fn deep_clone(value: &Value) -> Value {
    match value {
        Value::Array(items) => Value::Array(items.iter().map(deep_clone).collect()),
        Value::Object(o) => {
            let mut cloned = Object::new();
            for (k, v) in o.iter() {
                set_plain(&mut cloned, k, deep_clone(v));
            }
            Value::Object(cloned)
        }
        other => other.clone(),
    }
}

/// The location `findStepByIndex` returns.
struct StepLocation {
    plan_index: usize,
    chain_index: usize,
    step: Value,
}

/// `findStepByIndex(compiled, stepIndex)`: the non-builtin step whose `temp` is
/// `stepIndex`.
fn find_step_by_index(
    compiled: &Value,
    step_index: &Value,
) -> Result<Option<StepLocation>, JsError> {
    let plans = get_opt(compiled, "plans");
    if !plans.is_truthy() {
        return Ok(None);
    }
    let plan_count = to_number(&member(&plans, "length"))?;
    let mut plan_index = 0usize;
    while (plan_index as f64) < plan_count {
        let plan = member(&plans, &plan_index.to_string());
        let chain = get_opt(&plan, "chain");
        if chain.is_truthy() {
            let chain_len = to_number(&member(&chain, "length"))?;
            let mut chain_index = 0usize;
            while (chain_index as f64) < chain_len {
                let step = member(&chain, &chain_index.to_string());
                if !get(&step, "builtin")?.is_truthy()
                    && strict_equals(&get(&step, "temp")?, step_index)
                {
                    return Ok(Some(StepLocation {
                        plan_index,
                        chain_index,
                        step,
                    }));
                }
                chain_index += 1;
            }
        }
        plan_index += 1;
    }
    Ok(None)
}

/// `isStarterOp(name)` for any value (non-strings are never starters).
fn is_starter_op(registry: &Registry, name: &Value) -> bool {
    matches!(name, Value::String(s) if registry.is_starter_op(s))
}

/// `checkIsStarter(effectName, searchOrder)`: the name itself, or a bare name
/// under any search namespace, is a registered starter.
fn check_is_starter(
    registry: &Registry,
    effect_name: &Value,
    search_order: &Value,
) -> Result<bool, JsError> {
    // Direct check
    if is_starter_op(registry, effect_name) {
        return Ok(true);
    }
    // Check with each namespace prefix if bare name
    let name = str_of(effect_name, "effectName", "includes")?;
    if !name.contains('.') && length_positive(search_order)? {
        for ns in iterate(search_order, "searchOrder")? {
            let candidate = Value::String(format!("{}.{name}", to_string(&ns)?));
            if is_starter_op(registry, &candidate) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// `getEffectSpec(effectName, searchOrder)`: the op spec of the name, or of the
/// bare name under the first search namespace that has it.
fn get_effect_spec(
    registry: &Registry,
    effect_name: &Value,
    search_order: &Value,
) -> Result<Value, JsError> {
    // Direct lookup
    let direct = op_spec(registry, effect_name)?;
    if direct.is_truthy() {
        return Ok(direct);
    }
    // Try with namespace prefixes
    let name = str_of(effect_name, "effectName", "includes")?;
    if !name.contains('.') && length_positive(search_order)? {
        for ns in iterate(search_order, "searchOrder")? {
            let namespaced = Value::String(format!("{}.{name}", to_string(&ns)?));
            let spec = op_spec(registry, &namespaced)?;
            if spec.is_truthy() {
                return Ok(spec);
            }
        }
    }
    Ok(Value::Null)
}

/// `typeMatches(value, type)`: only clear mismatches are flagged.
fn type_matches(value: &Value, declared: &Value) -> bool {
    if value.is_nullish() {
        return true;
    }
    match declared.as_str() {
        Some("float" | "int" | "number") => matches!(value, Value::Number(_)),
        Some("color") => matches!(value, Value::String(s) if s.starts_with('#')),
        Some("bool" | "boolean") => matches!(value, Value::Bool(_)),
        Some("surface" | "tex") => matches!(value, Value::String(_)),
        _ => true,
    }
}

/// A JavaScript `Set`/`Map` key list (SameValueZero, insertion order).
fn set_add(set: &mut Vec<Value>, v: Value) {
    if !set.iter().any(|k| same_value_zero(k, &v)) {
        set.push(v);
    }
}

fn set_has(set: &[Value], v: &Value) -> bool {
    set.iter().any(|k| same_value_zero(k, v))
}

/// `collectAcceptedArgNames(spec, instance, aliases)`: spec argument names,
/// instance globals and both sides of the alias map.
fn collect_accepted_arg_names(
    spec: &Value,
    instance: &Value,
    aliases: &Value,
) -> Result<Vec<Value>, JsError> {
    let mut accepted = Vec::new();
    for def in iterate_anon(&or(get_opt(spec, "args"), || Value::Array(Vec::new())))? {
        let name = get_opt(&def, "name");
        if name.is_truthy() {
            set_add(&mut accepted, name);
        }
    }
    let globals = get_opt(instance, "globals");
    if globals.is_truthy() {
        for key in keys(&globals) {
            set_add(&mut accepted, key.into());
        }
    }
    for name in keys(aliases) {
        set_add(&mut accepted, name.into());
    }
    for name in values(aliases) {
        set_add(&mut accepted, name);
    }
    Ok(accepted)
}

/// `getParamAliases(resolvedName)`: a copy of the registered alias map.
fn alias_map_for(registry: &Registry, resolved_name: &Value) -> Result<Value, JsError> {
    let key = to_property_key(resolved_name)?;
    let mut out = Object::new();
    if let Some(aliases) = registry.param_aliases.get(&key) {
        for (old, new) in aliases {
            out.insert(old.clone(), Value::from(new.as_str()));
        }
    }
    Ok(Value::Object(out))
}

/// `aliases[key] || key`.
fn canonical_name(aliases: &Value, key: &str) -> Value {
    or(member(aliases, key), || key.into())
}

/// `predictBackendSupport(instance, resolvedName, manifest)`.
fn predict_backend_support(
    instance: &Value,
    resolved_name: &Value,
    manifest: &Value,
) -> Result<Value, JsError> {
    if !manifest.is_truthy() || !instance.is_truthy() || !get(instance, "passes")?.is_truthy() {
        return Ok(Value::Undefined);
    }
    // `resolvedName.split('.')`
    let name_parts = || -> Result<Vec<String>, JsError> {
        Ok(str_of(resolved_name, "resolvedName", "split")?
            .split('.')
            .map(str::to_owned)
            .collect())
    };
    let namespace = {
        let ns = get(instance, "namespace")?;
        if ns.is_truthy() {
            ns
        } else {
            Value::from(name_parts()?[0].as_str())
        }
    };
    let last = name_parts()?.pop().map(Value::from).unwrap_or_default();
    let candidates = [get(instance, "name")?, get(instance, "func")?, last];
    let mut entry = Value::Null;
    for display_name in &candidates {
        if !display_name.is_truthy() {
            continue;
        }
        let key = format!("{}/{}", to_string(&namespace)?, to_string(display_name)?);
        let found = member(manifest, &key);
        if found.is_truthy() {
            entry = found;
            break;
        }
    }
    if !entry.is_truthy() {
        return Ok(obj([("webgl2", false.into()), ("webgpu", false.into())]));
    }
    let passes = get(instance, "passes")?;
    let Value::Array(pass_list) = &passes else {
        return Err(not_a_function("instance.passes.map"));
    };
    let programs: Vec<Value> = pass_list
        .iter()
        .map(|p| get_opt(p, "program"))
        .filter(Value::is_truthy)
        .collect();
    let cover = |table: &Value| -> Result<Value, JsError> {
        if !table.is_truthy() {
            return Ok(false.into());
        }
        if programs.is_empty() {
            return Ok(Value::Undefined);
        }
        let mut hits = 0usize;
        for p in &programs {
            if in_operator(p, table)? {
                hits += 1;
            }
        }
        if hits == 0 {
            return Ok(false.into());
        }
        Ok(if hits == programs.len() {
            true.into()
        } else {
            "partial".into()
        })
    };
    let webgl2 = cover(&get(&entry, "glsl")?)?;
    let webgpu = cover(&get(&entry, "wgsl")?)?;
    Ok(obj([("webgl2", webgl2), ("webgpu", webgpu)]))
}

/// `predictSamplerTopology(instance)`: internal textures and the distinct string
/// pass inputs.
fn predict_sampler_topology(instance: &Value) -> Result<Value, JsError> {
    if !instance.is_truthy() {
        return Ok(Value::Undefined);
    }
    let textures = get(instance, "textures")?;
    let internal_textures: Vec<Value> = if textures.is_truthy() {
        keys(&textures).into_iter().map(Value::from).collect()
    } else {
        Vec::new()
    };
    let mut pass_inputs: Vec<Value> = Vec::new();
    for pass in iterate_anon(&or(get(instance, "passes")?, || Value::Array(Vec::new())))? {
        let inputs = or(get_opt(&pass, "inputs"), || Value::Object(Object::new()));
        for value in values(&inputs) {
            if matches!(value, Value::String(_)) && !set_has(&pass_inputs, &value) {
                pass_inputs.push(value);
            }
        }
    }
    Ok(obj([
        ("internalTextures", Value::Array(internal_textures)),
        ("passInputs", Value::Array(pass_inputs)),
    ]))
}

/// `predictPassesAndOutputs(instance)`: the pass list and the 3D outputs.
fn predict_passes_and_outputs(instance: &Value) -> Result<Value, JsError> {
    if !instance.is_truthy() {
        return Ok(Value::Undefined);
    }
    let passes = or(get(instance, "passes")?, || Value::Array(Vec::new()));
    let Value::Array(pass_list) = &passes else {
        return Err(not_a_function("(instance.passes || []).map"));
    };
    let copy = |v: &Value| -> Value {
        if v.is_truthy() {
            let mut o = Object::new();
            jsv::spread_into(&mut o, v);
            Value::Object(o)
        } else {
            Value::Object(Object::new())
        }
    };
    let passes: Vec<Value> = pass_list
        .iter()
        .map(|pass| {
            obj([
                ("name", get_opt(pass, "name")),
                ("program", get_opt(pass, "program")),
                ("inputs", copy(&get_opt(pass, "inputs"))),
                ("outputs", copy(&get_opt(pass, "outputs"))),
                ("drawBuffers", get_opt(pass, "drawBuffers")),
            ])
        })
        .collect();
    let nullish_or_null = |v: Value| if v.is_nullish() { Value::Null } else { v };
    let outputs = obj([
        ("geo", nullish_or_null(get(instance, "outputGeo")?)),
        ("tex3d", nullish_or_null(get(instance, "outputTex3d")?)),
    ]);
    Ok(obj([
        ("passes", Value::Array(passes)),
        ("outputs", outputs),
    ]))
}

/// `predictReplacement(resolvedName, spec, newArgs, oldInstance, options)`:
/// predict a candidate replacement's compatibility before mutation.
///
/// Covered dimensions: shader availability, arguments (unknown/missing), types,
/// ranges (min/max/choices), passes, outputs, sampler topology and backend
/// support (from `options.manifest`). Unavailable dimensions stay `undefined`.
pub fn predict_replacement(
    resolved_name: &Value,
    spec: &Value,
    new_args: &Value,
    old_instance: &Value,
    options: &Value,
    registry: &Registry,
) -> Result<Value, JsError> {
    let options = default_object(options);
    let entry = effect_entry(registry, resolved_name);
    let instance = instance_of(&entry);
    let mut available = Value::Undefined;
    let mut issues: Vec<Value> = Vec::new();

    // Shader availability
    if instance.is_truthy() {
        available = true.into();
    } else if !registry.effects.is_empty() {
        available = false.into();
        issues.push(obj([
            ("dimension", "shader-availability".into()),
            (
                "message",
                format!(
                    "No registered effect definition for '{}'",
                    to_string(resolved_name)?
                )
                .into(),
            ),
        ]));
    }

    let aliases = alias_map_for(registry, resolved_name)?;
    let accepted = collect_accepted_arg_names(spec, instance, &aliases)?;

    let provided = or(new_args.clone(), || Value::Object(Object::new()));
    let provided_keys = keys(&provided);
    let mut provided_canonical = Vec::new();
    for key in &provided_keys {
        set_add(&mut provided_canonical, canonical_name(&aliases, key));
    }
    let mut unknown: Vec<Value> = Vec::new();
    for key in &provided_keys {
        let canonical = canonical_name(&aliases, key);
        if !set_has(&accepted, &canonical) {
            unknown.push(key.as_str().into());
        }
    }
    // knownDefs: spec args, then instance globals not already declared.
    let mut known_defs: Vec<(Value, Value)> = Vec::new();
    let map_set = |map: &mut Vec<(Value, Value)>, k: Value, v: Value| match map
        .iter_mut()
        .find(|(key, _)| same_value_zero(key, &k))
    {
        Some(slot) => slot.1 = v,
        None => map.push((k, v)),
    };
    for def in iterate_anon(&or(get_opt(spec, "args"), || Value::Array(Vec::new())))? {
        let name = get(&def, "name")?;
        map_set(&mut known_defs, name, def.clone());
    }
    let globals = get_opt(instance, "globals");
    if globals.is_truthy() {
        for (key, def) in entries(&globals) {
            let key = Value::from(key);
            if !known_defs.iter().any(|(k, _)| same_value_zero(k, &key)) {
                map_set(&mut known_defs, key, def);
            }
        }
    }
    let mut missing: Vec<Value> = Vec::new();
    for (key, def) in &known_defs {
        if get_opt(def, "default").is_undefined() && !set_has(&provided_canonical, key) {
            missing.push(key.clone());
        }
    }
    if !unknown.is_empty() {
        issues.push(obj([
            ("dimension", "arguments".into()),
            (
                "message",
                format!(
                    "Unknown argument(s) for '{}': {}",
                    to_string(resolved_name)?,
                    jsv::join(&unknown, ", ")?
                )
                .into(),
            ),
        ]));
    }

    // Type and range checks
    let mut types: Vec<Value> = Vec::new();
    let mut ranges: Vec<Value> = Vec::new();
    for (key, value) in entries(&provided) {
        let canonical = canonical_name(&aliases, &key);
        let Some(def) = known_defs
            .iter()
            .find(|(k, _)| same_value_zero(k, &canonical))
            .map(|(_, d)| d.clone())
        else {
            continue;
        };
        if !def.is_truthy() {
            continue;
        }
        let def_type = get(&def, "type")?;
        let declared_type = if strict_equals(&def_type, &"color".into()) {
            Value::from("color")
        } else {
            def_type
        };
        if !type_matches(&value, &declared_type) {
            types.push(obj([
                ("arg", key.as_str().into()),
                ("expected", declared_type.clone()),
                ("actual", value.type_of().into()),
            ]));
        }
        if let Value::Number(n) = value {
            let min = get(&def, "min")?;
            if !min.is_undefined() && n < to_number(&min)? {
                ranges.push(obj([
                    ("arg", canonical.clone()),
                    ("value", value.clone()),
                    ("min", min.clone()),
                    ("max", get(&def, "max")?),
                ]));
            }
            let max = get(&def, "max")?;
            if !max.is_undefined() && n > to_number(&max)? {
                ranges.push(obj([
                    ("arg", canonical.clone()),
                    ("value", value.clone()),
                    ("min", get(&def, "min")?),
                    ("max", max.clone()),
                ]));
            }
        }
        let choices = get(&def, "choices")?;
        if choices.is_truthy() && matches!(value, Value::Number(_)) {
            let allowed = values(&choices);
            if !allowed.iter().any(|a| same_value_zero(a, &value)) {
                ranges.push(obj([
                    ("arg", canonical.clone()),
                    ("value", value.clone()),
                    ("choices", Value::Array(allowed)),
                ]));
            }
        }
    }
    if !types.is_empty() {
        let mut messages = Vec::new();
        for t in &types {
            messages.push(format!(
                "Argument '{}' for '{}' expects {}, got {}",
                to_string(&get_opt(t, "arg"))?,
                to_string(resolved_name)?,
                to_string(&get_opt(t, "expected"))?,
                to_string(&get_opt(t, "actual"))?
            ));
        }
        issues.push(obj([
            ("dimension", "types".into()),
            ("message", messages.join("; ").into()),
        ]));
    }
    if !ranges.is_empty() {
        let mut messages = Vec::new();
        for r in &ranges {
            let choices = get_opt(r, "choices");
            if choices.is_truthy() {
                messages.push(format!(
                    "Argument '{}' for '{}' value {} is not one of {}",
                    to_string(&get_opt(r, "arg"))?,
                    to_string(resolved_name)?,
                    to_string(&get_opt(r, "value"))?,
                    match &choices {
                        Value::Array(items) => jsv::join(items, ", ")?,
                        _ => unreachable!("choices are an array"),
                    }
                ));
            } else {
                messages.push(format!(
                    "Argument '{}' for '{}' value {} outside range [{}, {}]",
                    to_string(&get_opt(r, "arg"))?,
                    to_string(resolved_name)?,
                    to_string(&get_opt(r, "value"))?,
                    to_string(&get_opt(r, "min"))?,
                    to_string(&get_opt(r, "max"))?
                ));
            }
        }
        issues.push(obj([
            ("dimension", "ranges".into()),
            ("message", messages.join("; ").into()),
        ]));
    }

    // Structure predictions (informative when data is available)
    let passes = predict_passes_and_outputs(instance)?;
    let mut sampler_topology = predict_sampler_topology(instance)?;
    if sampler_topology.is_truthy() && old_instance.is_truthy() {
        let old_topology = predict_sampler_topology(old_instance)?;
        if old_topology.is_truthy() {
            let changed_from = obj([
                (
                    "internalTextures",
                    get_opt(&old_topology, "internalTextures"),
                ),
                ("passInputs", get_opt(&old_topology, "passInputs")),
            ]);
            sampler_topology.set("changedFrom", changed_from);
        }
    }
    let backend_support =
        predict_backend_support(instance, resolved_name, &get(&options, "manifest")?)?;

    Ok(obj([
        ("effect", resolved_name.clone()),
        ("available", available),
        (
            "arguments",
            obj([
                ("unknown", Value::Array(unknown)),
                ("missing", Value::Array(missing)),
            ]),
        ),
        ("types", Value::Array(types)),
        ("ranges", Value::Array(ranges)),
        ("passes", passes),
        ("outputs", Value::Undefined),
        ("samplerTopology", sampler_topology),
        ("backendSupport", backend_support),
        ("issues", Value::Array(issues)),
    ]))
}

/// `options.searchOrder || compiled.searchNamespaces || []`.
fn search_order_of(options: &Value, compiled: &Value) -> Result<Value, JsError> {
    let order = get(options, "searchOrder")?;
    if order.is_truthy() {
        return Ok(order);
    }
    Ok(or(get(compiled, "searchNamespaces")?, || {
        Value::Array(Vec::new())
    }))
}

/// `step.from === null || step.from === undefined`.
fn has_no_predecessor(step: &Value) -> Result<bool, JsError> {
    Ok(get(step, "from")?.is_nullish())
}

/// `replaceEffect(compiled, stepIndex, newEffectName, newArgs, options)`: replace
/// the effect of the step whose `temp` is `stepIndex`, like for like (a starter
/// position takes only starters, any other position only non-starters).
///
/// Returns `{success, program?, prediction?, error?}`; the input program is not
/// modified. `options`: `searchOrder`, `preflight` (refuse candidates with
/// predicted issues), `manifest`.
pub fn replace_effect(
    compiled: &Value,
    step_index: &Value,
    new_effect_name: &Value,
    new_args: &Value,
    options: &Value,
    registry: &Registry,
) -> Result<Value, JsError> {
    let new_args = default_object(new_args);
    let options = default_object(options);
    if !get_opt(compiled, "plans").is_truthy() {
        return Ok(failure("Invalid compiled program: missing plans".into()));
    }

    let search_order = search_order_of(&options, compiled)?;

    // Find the step to replace
    let Some(location) = find_step_by_index(compiled, step_index)? else {
        return Ok(failure(format!(
            "Step with index {} not found",
            to_string(step_index)?
        )));
    };
    let StepLocation {
        plan_index,
        chain_index,
        step,
    } = location;
    let old_effect_name = get(&step, "op")?;

    // A step is in "starter position" if it is the first step of its chain, or
    // an inline surface producer (a starter without a pipeline predecessor).
    let current_is_starter = check_is_starter(registry, &old_effect_name, &search_order)?;
    let is_starter_position =
        chain_index == 0 || (current_is_starter && has_no_predecessor(&step)?);

    // Check if new effect is a starter
    let new_is_starter = check_is_starter(registry, new_effect_name, &search_order)?;

    // Verify the new effect exists
    let new_spec = get_effect_spec(registry, new_effect_name, &search_order)?;
    if !new_spec.is_truthy() {
        return Ok(failure(format!(
            "Effect '{}' not found",
            to_string(new_effect_name)?
        )));
    }

    // Enforce like-for-like replacement
    if is_starter_position && !new_is_starter {
        return Ok(failure(format!(
            "Cannot replace starter effect '{}' with non-starter effect '{}'. The first effect in a chain must be a starting effect.",
            to_string(&old_effect_name)?,
            to_string(new_effect_name)?
        )));
    }
    if !is_starter_position && new_is_starter {
        return Ok(failure(format!(
            "Cannot replace non-starter effect '{}' with starter effect '{}'. Starting effects can only appear at the beginning of a chain.",
            to_string(&old_effect_name)?,
            to_string(new_effect_name)?
        )));
    }

    // Clone the program for immutability
    let mut new_program = deep_clone(compiled);

    // Build new args with defaults from effect spec (DSL parameter names)
    let mut final_args = Object::new();
    let spec_args = or(get(&new_spec, "args")?, || Value::Array(Vec::new()));
    for def in iterate(&spec_args, "specArgs")? {
        let default = get(&def, "default")?;
        if !default.is_undefined() {
            set_plain(
                &mut final_args,
                &to_property_key(&get(&def, "name")?)?,
                default,
            );
        }
    }

    // Apply provided args (with rounding for floats - max 3 decimal places)
    for (key, value) in entries_strict(&new_args)? {
        let value = match value {
            Value::Number(n) if !Value::Number(n).is_integer() => {
                Value::Number(math_round(n * 1000.0) / 1000.0)
            }
            other => other,
        };
        set_plain(&mut final_args, &key, value);
    }

    // Get the resolved effect name (with namespace if needed)
    let mut resolved_new_name = new_effect_name.clone();
    let mut effect_namespace = Value::Null;
    let name = str_of(new_effect_name, "newEffectName", "includes")?;
    if name.contains('.') {
        // Already namespaced - extract namespace
        effect_namespace = name.split('.').next().unwrap_or_default().into();
        // Verify it exists
        if !op_spec(registry, new_effect_name)?.is_truthy() {
            return Ok(failure(format!("Effect '{name}' not found")));
        }
    } else {
        // Try to find the namespaced version
        for ns in iterate(&search_order, "searchOrder")? {
            let namespaced = Value::String(format!("{}.{name}", to_string(&ns)?));
            if op_spec(registry, &namespaced)?.is_truthy() {
                resolved_new_name = namespaced;
                effect_namespace = ns;
                break;
            }
        }
        // If not found in search order, try all registered ops
        if !effect_namespace.is_truthy() {
            let suffix = format!(".{name}");
            for op_name in registry.ops.keys() {
                if op_name.ends_with(&suffix) {
                    resolved_new_name = op_name.as_str().into();
                    effect_namespace = op_name.split('.').next().unwrap_or_default().into();
                    break;
                }
            }
        }
    }

    // Preflight prediction BEFORE any mutation, using the resolved name.
    let old_entry = effect_entry(registry, &old_effect_name);
    let prediction = predict_replacement(
        &resolved_new_name,
        &new_spec,
        &new_args,
        instance_of(&old_entry),
        &options,
        registry,
    )?;
    let issues = get_opt(&prediction, "issues");
    if strict_equals(&get(&options, "preflight")?, &true.into()) && length_positive(&issues)? {
        let messages: Vec<Value> = iterate(&issues, "prediction.issues")?
            .iter()
            .map(|issue| get_opt(issue, "message"))
            .collect();
        return Ok(obj([
            ("success", false.into()),
            (
                "error",
                format!(
                    "Replacement preflight failed: {}",
                    jsv::join(&messages, "; ")?
                )
                .into(),
            ),
            ("prediction", prediction),
        ]));
    }

    // Ensure the namespace is in searchNamespaces so unparser can strip it
    if effect_namespace.is_truthy() {
        let namespaces = get(&new_program, "searchNamespaces")?;
        let includes = match &namespaces {
            Value::Array(items) => items.iter().any(|n| same_value_zero(n, &effect_namespace)),
            Value::String(s) => s.contains(to_string(&effect_namespace)?.as_str()),
            Value::Undefined | Value::Null => return Err(cannot_read(&namespaces, "includes")),
            _ => {
                return Err(not_a_function("newProgram.searchNamespaces.includes"));
            }
        };
        if !includes {
            let mut extended = iterate(&namespaces, "newProgram.searchNamespaces")?;
            extended.push(effect_namespace.clone());
            new_program.set("searchNamespaces", Value::Array(extended));
        }
    }

    // Update the step: op, args, and a clean namespace (clears stale from() overrides)
    let namespace = if effect_namespace.is_truthy() {
        obj([("resolved", effect_namespace.clone())])
    } else {
        Value::Null
    };
    let new_step = index_mut(&mut new_program, "plans")
        .and_then(|plans| index_mut(plans, &plan_index.to_string()))
        .and_then(|plan| index_mut(plan, "chain"))
        .and_then(|chain| index_mut(chain, &chain_index.to_string()));
    match new_step {
        Some(Value::Object(step)) => {
            set_plain(step, "op", resolved_new_name);
            set_plain(step, "args", Value::Object(final_args));
            set_plain(step, "namespace", namespace);
        }
        // Arrays and functions take the members as properties no Value can hold.
        Some(Value::Array(_) | Value::Function(_)) | None => {}
        Some(other) => {
            return Err(JsError::type_error(format!(
                "Cannot create property 'op' on {} '{}'",
                other.type_of(),
                to_string(other)?
            )));
        }
    }

    Ok(obj([
        ("success", true.into()),
        ("program", new_program),
        ("prediction", prediction),
    ]))
}

/// `listSteps(compiled, options)`: every non-builtin step with its position and
/// replacement constraints.
pub fn list_steps(
    compiled: &Value,
    options: &Value,
    registry: &Registry,
) -> Result<Value, JsError> {
    let options = default_object(options);
    let plans = get_opt(compiled, "plans");
    if !plans.is_truthy() {
        return Ok(Value::Array(Vec::new()));
    }
    let search_order = search_order_of(&options, compiled)?;
    let mut steps = Vec::new();
    let plan_count = to_number(&member(&plans, "length"))?;
    let mut plan_index = 0usize;
    while (plan_index as f64) < plan_count {
        let plan = member(&plans, &plan_index.to_string());
        let chain = get_opt(&plan, "chain");
        if chain.is_truthy() {
            let chain_len = to_number(&member(&chain, "length"))?;
            let mut chain_index = 0usize;
            while (chain_index as f64) < chain_len {
                let step = member(&chain, &chain_index.to_string());
                if !get(&step, "builtin")?.is_truthy() {
                    let op = get(&step, "op")?;
                    let is_starter = check_is_starter(registry, &op, &search_order)?;
                    let is_starter_position =
                        chain_index == 0 || (is_starter && has_no_predecessor(&step)?);
                    steps.push(obj([
                        ("stepIndex", get(&step, "temp")?),
                        ("planIndex", Value::from(plan_index)),
                        ("chainIndex", Value::from(chain_index)),
                        ("effectName", op),
                        ("isStarter", is_starter.into()),
                        ("isStarterPosition", is_starter_position.into()),
                        ("canReplaceWithStarter", is_starter_position.into()),
                        ("canReplaceWithNonStarter", (!is_starter_position).into()),
                        (
                            "args",
                            or(get(&step, "args")?, || Value::Object(Object::new())),
                        ),
                    ]));
                }
                chain_index += 1;
            }
        }
        plan_index += 1;
    }
    Ok(Value::Array(steps))
}

/// `getCompatibleReplacements(compiled, stepIndex, options)`: every registered
/// op classified as a legal (`compatible`) or illegal (`incompatible`)
/// replacement of the step, with a per-candidate `predictions` map. With
/// `options.preflight === true`, candidates with predicted issues move from
/// `compatible` to `blocked`.
pub fn get_compatible_replacements(
    compiled: &Value,
    step_index: &Value,
    options: &Value,
    registry: &Registry,
) -> Result<Value, JsError> {
    let options = default_object(options);
    if !get_opt(compiled, "plans").is_truthy() {
        return Ok(failure("Invalid compiled program: missing plans".into()));
    }
    let search_order = search_order_of(&options, compiled)?;
    let Some(location) = find_step_by_index(compiled, step_index)? else {
        return Ok(failure(format!(
            "Step with index {} not found",
            to_string(step_index)?
        )));
    };
    let op = get(&location.step, "op")?;
    let current_is_starter = check_is_starter(registry, &op, &search_order)?;
    let is_starter_position =
        location.chain_index == 0 || (current_is_starter && has_no_predecessor(&location.step)?);
    let old_entry = effect_entry(registry, &op);
    let old_instance = instance_of(&old_entry);

    // Collect all registered ops
    let mut starters: Vec<Value> = Vec::new();
    let mut non_starters: Vec<Value> = Vec::new();
    let mut predictions = Object::new();
    for (op_name, spec) in registry.ops.iter() {
        let name = Value::from(op_name.as_str());
        let is_starter = check_is_starter(registry, &name, &search_order)?;
        let prediction = predict_replacement(
            &name,
            spec,
            &Value::Object(Object::new()),
            old_instance,
            &options,
            registry,
        )?;
        set_plain(&mut predictions, op_name, prediction);
        if is_starter {
            starters.push(name);
        } else {
            non_starters.push(name);
        }
    }

    if strict_equals(&get(&options, "preflight")?, &true.into()) {
        // Opt-in: move candidates whose prediction found hard issues out of the
        // compatible list, with the reason attached.
        let mut blocked = Vec::new();
        let mut filter_issues = |names: &[Value]| -> Result<Vec<Value>, JsError> {
            let mut ok = Vec::new();
            for name in names {
                let key = to_property_key(name)?;
                let issues = get(&object_member(&predictions, &key), "issues")?;
                if length_positive(&issues)? {
                    blocked.push(obj([("effect", name.clone()), ("issues", issues)]));
                } else {
                    ok.push(name.clone());
                }
            }
            Ok(ok)
        };
        let (compatible, incompatible) = if is_starter_position {
            (filter_issues(&starters)?, non_starters)
        } else {
            (filter_issues(&non_starters)?, starters)
        };
        return Ok(obj([
            ("success", true.into()),
            ("compatible", Value::Array(compatible)),
            ("incompatible", Value::Array(incompatible)),
            ("blocked", Value::Array(blocked)),
            ("predictions", Value::Object(predictions)),
        ]));
    }

    let (compatible, incompatible) = if is_starter_position {
        (starters, non_starters)
    } else {
        (non_starters, starters)
    };
    Ok(obj([
        ("success", true.into()),
        ("compatible", Value::Array(compatible)),
        ("incompatible", Value::Array(incompatible)),
        ("predictions", Value::Object(predictions)),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> Registry {
        let mut reg = Registry::new();
        reg.register_op(
            "synth.noise",
            Value::from_json(r#"{"name":"noise","args":[{"name":"scale","type":"float","default":10},{"name":"seed","type":"float","default":1}]}"#).unwrap(),
        );
        reg.register_op(
            "filter.kaleid",
            Value::from_json(
                r#"{"name":"kaleid","args":[{"name":"nSides","type":"float","default":4}]}"#,
            )
            .unwrap(),
        );
        reg.register_op(
            "filter.bloom",
            Value::from_json(
                r#"{"name":"bloom","args":[{"name":"intensity","type":"float","default":0.5}]}"#,
            )
            .unwrap(),
        );
        reg.register_starter_ops(&["synth.noise"]);
        reg
    }

    fn program() -> Value {
        Value::from_json(
            r#"{"plans":[{"chain":[
                {"op":"synth.noise","args":{"scale":10,"seed":1},"from":null,"temp":0},
                {"op":"filter.kaleid","args":{"nSides":6},"from":0,"temp":1},
                {"op":"_write","args":{"tex":{"kind":"output","name":"o0"}},"from":1,"temp":2,"builtin":true}
            ],"write":{"kind":"output","name":"o0"}}],"searchNamespaces":["synth","filter"]}"#,
        )
        .unwrap()
    }

    #[test]
    fn steps_and_replacement() {
        let reg = registry();
        let steps = list_steps(&program(), &Value::Undefined, &reg).unwrap();
        assert_eq!(steps.at(0).get("isStarterPosition"), &Value::Bool(true));
        assert_eq!(
            steps.at(1).get("canReplaceWithNonStarter"),
            &Value::Bool(true)
        );

        let ok = replace_effect(
            &program(),
            &Value::Number(1.0),
            &"bloom".into(),
            &Value::from_json(r#"{"intensity":0.12345}"#).unwrap(),
            &Value::Undefined,
            &reg,
        )
        .unwrap();
        assert_eq!(ok.get("success"), &Value::Bool(true));
        let step = ok.get("program").get("plans").at(0).get("chain").at(1);
        assert_eq!(step.get("op"), &Value::from("filter.bloom"));
        assert_eq!(step.get("args").get("intensity"), &Value::Number(0.123));

        let bad = replace_effect(
            &program(),
            &Value::Number(0.0),
            &"kaleid".into(),
            &Value::Undefined,
            &Value::Undefined,
            &reg,
        )
        .unwrap();
        assert!(bad.get("error").as_str().unwrap().contains("non-starter"));

        let missing = replace_effect(
            &program(),
            &Value::Number(2.0),
            &"bloom".into(),
            &Value::Undefined,
            &Value::Undefined,
            &reg,
        )
        .unwrap();
        assert_eq!(
            missing.get("error"),
            &Value::from("Step with index 2 not found")
        );
    }

    #[test]
    fn compatible_replacements_classify_ops() {
        let reg = registry();
        let result =
            get_compatible_replacements(&program(), &Value::Number(1.0), &Value::Undefined, &reg)
                .unwrap();
        assert_eq!(
            result.get("compatible"),
            &Value::from_json(r#"["filter.kaleid","filter.bloom"]"#).unwrap()
        );
        assert_eq!(
            result.get("incompatible"),
            &Value::from_json(r#"["synth.noise"]"#).unwrap()
        );
        // No effect instances are registered: availability is unknown.
        let prediction = result.get("predictions").get("filter.bloom");
        assert!(prediction.get("available").is_undefined());
        assert!(prediction.get("passes").is_undefined());
    }
}
