//! Port of `demo/shaders/lib/dsl-utils.js`: `extractEffectsFromDsl`, the chain
//! of effects (with their arguments and global step indices) that ProgramState
//! and the demo UI build their state and controls from.

use super::console::{self, ConsoleArg};
use crate::JsError;
use crate::error_formatter::{format_dsl_error, is_dsl_syntax_error};
use crate::registry::Registry;
use crate::unparser::jsv::{cannot_read, entries, get, member, not_a_function, spread_into};
use crate::value::{Object, Value};

/// `compile(src)` of `lang/index.js`: `validate(parse(lex(src)))`.
pub fn compile(src: &str, registry: &Registry) -> Result<Value, JsError> {
    let tokens = crate::lexer::lex(src)?;
    let ast = crate::parser::parse_with_registry(&tokens, registry)?;
    crate::validator::validate(&ast, registry)
}

/// One effect step of a program (`EffectInfo` of the reference).
#[derive(Debug, Clone, PartialEq)]
pub struct EffectInfo {
    /// Full effect identifier, the compiled step's `op` (e.g. `"synth.noise"`).
    pub effect_key: String,
    /// The step's resolved namespace (`step.namespace?.namespace ||
    /// step.namespace?.resolved || null`).
    pub namespace: Value,
    /// Short effect name: the last `.` segment of `op`.
    pub name: String,
    /// Full operation name (same as `effect_key`).
    pub full_name: String,
    /// Resolved argument values (a copy of the step's `args`, with automation
    /// configs from `rawKwargs` restored where validation normalized them).
    pub args: Object,
    /// The step's original keyword arguments (`step.rawKwargs || {}`).
    pub raw_kwargs: Value,
    /// Global step index over every plan's chain.
    pub step_index: usize,
    /// The step's temporary texture index (`step.temp`).
    pub temp: Value,
}

impl EffectInfo {
    /// The reference's object: `{ effectKey, namespace, name, fullName, args,
    /// rawKwargs, stepIndex, temp }`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("effectKey", Value::from(self.effect_key.as_str()));
        o.insert("namespace", self.namespace.clone());
        o.insert("name", Value::from(self.name.as_str()));
        o.insert("fullName", Value::from(self.full_name.as_str()));
        o.insert("args", Value::Object(self.args.clone()));
        o.insert("rawKwargs", self.raw_kwargs.clone());
        o.insert("stepIndex", Value::from(self.step_index));
        o.insert("temp", self.temp.clone());
        Value::Object(o)
    }
}

/// An array of effect infos as the reference's array of objects.
pub fn effects_to_value(effects: &[EffectInfo]) -> Value {
    Value::Array(effects.iter().map(EffectInfo::to_value).collect())
}

/// `x && typeof x === 'object' && (x.type === <t> || ... || x._ast?.type === <t>)`
/// over the automation types (oscillator, MIDI, audio).
pub(crate) fn is_automation_config(v: &Value) -> bool {
    if !matches!(v, Value::Object(_) | Value::Array(_)) {
        return false;
    }
    let is_auto = |t: &Value| matches!(t.as_str(), Some("Oscillator" | "Midi" | "Audio"));
    is_auto(&member(v, "type")) || is_auto(&member(&member(v, "_ast"), "type"))
}

/// `extractEffectsFromDsl(dsl)`: the program's effect steps in chain order, or
/// an empty array when the program does not compile (the error is reported to
/// the console: syntax errors formatted with their source line, anything else
/// as is). Like the reference, a failure part-way returns the effects found
/// before it.
pub fn extract_effects_from_dsl(dsl: &str, registry: &Registry) -> Vec<EffectInfo> {
    let mut effects = Vec::new();
    // `if (!dsl || typeof dsl !== 'string') return effects`
    if dsl.is_empty() {
        return effects;
    }
    if let Err(err) = collect_effects(dsl, registry, &mut effects) {
        if is_dsl_syntax_error(&err) {
            let formatted = match format_dsl_error(&Value::from(dsl), &err, &Value::Undefined) {
                Ok(text) => text,
                // formatDslError threw inside the catch block: the reference
                // would propagate that error out of extractEffectsFromDsl. The
                // formatter only throws for non-string sources, which `dsl` is
                // not.
                Err(e) => e.to_string(),
            };
            console::warn(&[ConsoleArg::from(format!("DSL Syntax Error:\n{formatted}"))]);
        } else {
            console::warn(&[
                "Failed to parse DSL for effect extraction:".into(),
                err.into(),
            ]);
        }
    }
    effects
}

fn collect_effects(
    dsl: &str,
    registry: &Registry,
    effects: &mut Vec<EffectInfo>,
) -> Result<(), JsError> {
    // Compile to get resolved args with rawKwargs preserved on each step
    let result = compile(dsl, registry)?;
    let plans = member(&result, "plans");
    if !result.is_truthy() || !plans.is_truthy() {
        return Ok(());
    }
    let plans = for_of(&plans, "result.plans")?;

    let mut global_step_index = 0usize;
    for plan in &plans {
        let chain = get(plan, "chain")?;
        if !chain.is_truthy() {
            continue;
        }
        for step in for_of(&chain, "plan.chain")? {
            let full_op_name = get(&step, "op")?;
            let ns = get(&step, "namespace")?;
            let namespace = {
                let a = member(&ns, "namespace");
                if a.is_truthy() {
                    a
                } else {
                    let b = member(&ns, "resolved");
                    if b.is_truthy() { b } else { Value::Null }
                }
            };

            let full = match &full_op_name {
                Value::String(s) => s.clone(),
                Value::Undefined | Value::Null => {
                    return Err(cannot_read(&full_op_name, "includes"));
                }
                _ => return Err(not_a_function("fullOpName.includes")),
            };
            let short_name = if full.contains('.') {
                full.rsplit('.').next().unwrap_or_default().to_owned()
            } else {
                full.clone()
            };

            // Use rawKwargs directly from the compiled step (set by validator)
            let raw_kwargs = get(&step, "rawKwargs")?;
            let raw_args = if raw_kwargs.is_truthy() {
                raw_kwargs
            } else {
                Value::Object(Object::new())
            };
            let step_args = get(&step, "args")?;
            let mut args = Object::new();
            if step_args.is_truthy() {
                spread_into(&mut args, &step_args);
            }

            // Preserve automation bindings even if validation normalized them to scalars
            for (param_name, raw_val) in entries(&raw_args) {
                let is_raw_automation = is_automation_config(&raw_val);
                let current_val = crate::unparser::jsv::object_member(&args, &param_name);
                let is_arg_automation = is_automation_config(&current_val);
                if is_raw_automation && !is_arg_automation {
                    crate::unparser::jsv::set_plain(&mut args, &param_name, raw_val);
                }
            }

            effects.push(EffectInfo {
                effect_key: full.clone(),
                namespace,
                name: short_name,
                full_name: full,
                args,
                raw_kwargs: raw_args,
                step_index: global_step_index,
                temp: get(&step, "temp")?,
            });
            global_step_index += 1;
        }
    }
    Ok(())
}

/// The values a `for (const x of v)` loop visits.
pub(crate) fn for_of(v: &Value, expr: &str) -> Result<Vec<Value>, JsError> {
    crate::unparser::jsv::iterate(v, expr)
}
