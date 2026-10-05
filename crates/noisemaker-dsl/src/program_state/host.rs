//! The renderer side of ProgramState.
//!
//! The reference ProgramState holds a `CanvasRenderer` and reaches through it
//! to the live pipeline: it reads `renderer.currentDsl` and `renderer.enums`,
//! converts parameter values with `renderer.convertParameterForUniform`, writes
//! `renderer.pipeline.graph.passes[i].uniforms` directly, and calls
//! `pipeline.broadcastChainScopedParam`, `pipeline.checkAsyncRegen`,
//! `pipeline.recreateTextures(pipeline.collectDefaultUniforms())` and
//! `pipeline.setUniform`. [`ProgramHost`] is exactly that surface. Several of
//! those members are optional in the reference (`if (pipeline.setUniform)`):
//! the `has_*` methods answer those presence checks.
//!
//! The data helpers ProgramState shares with the renderer live here too:
//! `convertParameterForUniform` and `resolveEnumValue` (renderer/canvas.js) and
//! `writeUniformAliases` (runtime/uniform-aliases.js), so that every host
//! converts and aliases parameters with one implementation.

use std::borrow::Cow;
use std::rc::Rc;

use crate::JsError;
use crate::js::math_round;
use crate::unparser::jsv::{
    cannot_read, has_property, member, not_a_function, set_plain, strict_equals, to_property_key,
    to_string,
};
use crate::value::{Object, Value};

/// The renderer (and its pipeline) a ProgramState drives.
///
/// Pass indices refer to `pipeline.graph.passes`.
pub trait ProgramHost {
    /// `renderer.currentDsl` (`""` when there is none).
    fn current_dsl(&self) -> String;

    /// `renderer.enums`: the enum tree `toDsl` formats enum names with and
    /// `convertParameterForUniform` resolves enum names in (`undefined` when
    /// the renderer has none).
    fn enums(&self) -> Rc<Value> {
        Rc::new(Value::Undefined)
    }

    /// Whether the renderer has `convertParameterForUniform`.
    fn has_convert_parameter_for_uniform(&self) -> bool {
        true
    }

    /// `renderer.convertParameterForUniform(value, spec)`.
    fn convert_parameter_for_uniform(&self, value: &Value, spec: &Value) -> Result<Value, JsError> {
        convert_parameter_for_uniform(value, spec, &self.enums())
    }

    /// `renderer.pipeline.graph.passes`, or `None` when the renderer has no
    /// pipeline, the pipeline no graph, or the graph no passes.
    fn graph_passes(&mut self) -> Option<&mut Vec<Value>>;

    /// Whether the pipeline has `broadcastChainScopedParam` (the reference calls
    /// it unconditionally, so a pipeline without it throws a `TypeError`).
    fn has_broadcast_chain_scoped_param(&self) -> bool {
        true
    }

    /// `pipeline.broadcastChainScopedParam(passes[pass_index], uniformName,
    /// scopedName)`. The names are the property keys ProgramState wrote
    /// (`pass.uniforms[uniformName]` and `pass.uniforms[scopedName]`).
    fn broadcast_chain_scoped_param(
        &mut self,
        pass_index: usize,
        uniform_name: &str,
        scoped_name: &str,
    ) -> Result<(), JsError>;

    /// Whether the pipeline has `checkAsyncRegen`.
    fn has_check_async_regen(&self) -> bool {
        true
    }

    /// `pipeline.checkAsyncRegen(nodeId, effectKey, stepValues)` (`node_id` and
    /// `effect_key` are the pass members, truthy).
    fn check_async_regen(
        &mut self,
        node_id: &Value,
        effect_key: &Value,
        step_values: &Object,
    ) -> Result<(), JsError>;

    /// Whether the pipeline has `recreateTextures`.
    fn has_recreate_textures(&self) -> bool {
        true
    }

    /// Whether the pipeline has `collectDefaultUniforms`.
    fn has_collect_default_uniforms(&self) -> bool {
        true
    }

    /// `pipeline.collectDefaultUniforms()`.
    fn collect_default_uniforms(&mut self) -> Result<Object, JsError>;

    /// `pipeline.recreateTextures(uniforms)`.
    fn recreate_textures(&mut self, uniforms: Object) -> Result<(), JsError>;

    /// Whether the pipeline has `setUniform`.
    fn has_set_uniform(&self) -> bool {
        true
    }

    /// `pipeline.setUniform(name, value)`.
    fn set_uniform(&mut self, name: &str, value: &Value) -> Result<(), JsError>;
}

impl<H: ProgramHost + ?Sized> ProgramHost for Box<H> {
    fn current_dsl(&self) -> String {
        (**self).current_dsl()
    }
    fn enums(&self) -> Rc<Value> {
        (**self).enums()
    }
    fn has_convert_parameter_for_uniform(&self) -> bool {
        (**self).has_convert_parameter_for_uniform()
    }
    fn convert_parameter_for_uniform(&self, value: &Value, spec: &Value) -> Result<Value, JsError> {
        (**self).convert_parameter_for_uniform(value, spec)
    }
    fn graph_passes(&mut self) -> Option<&mut Vec<Value>> {
        (**self).graph_passes()
    }
    fn has_broadcast_chain_scoped_param(&self) -> bool {
        (**self).has_broadcast_chain_scoped_param()
    }
    fn broadcast_chain_scoped_param(
        &mut self,
        pass_index: usize,
        uniform_name: &str,
        scoped_name: &str,
    ) -> Result<(), JsError> {
        (**self).broadcast_chain_scoped_param(pass_index, uniform_name, scoped_name)
    }
    fn has_check_async_regen(&self) -> bool {
        (**self).has_check_async_regen()
    }
    fn check_async_regen(
        &mut self,
        node_id: &Value,
        effect_key: &Value,
        step_values: &Object,
    ) -> Result<(), JsError> {
        (**self).check_async_regen(node_id, effect_key, step_values)
    }
    fn has_recreate_textures(&self) -> bool {
        (**self).has_recreate_textures()
    }
    fn has_collect_default_uniforms(&self) -> bool {
        (**self).has_collect_default_uniforms()
    }
    fn collect_default_uniforms(&mut self) -> Result<Object, JsError> {
        (**self).collect_default_uniforms()
    }
    fn recreate_textures(&mut self, uniforms: Object) -> Result<(), JsError> {
        (**self).recreate_textures(uniforms)
    }
    fn has_set_uniform(&self) -> bool {
        (**self).has_set_uniform()
    }
    fn set_uniform(&mut self, name: &str, value: &Value) -> Result<(), JsError> {
        (**self).set_uniform(name, value)
    }
}

/// `parseFloat(value)` (`ToString` first).
pub(crate) fn parse_float_value(value: &Value) -> Result<f64, JsError> {
    Ok(match value {
        Value::Number(n) => {
            // parseFloat(ToString(n)): the shortest decimal reads back exactly,
            // except the sign of zero (`parseFloat('0')` for -0) and the
            // non-finite spellings.
            if *n == 0.0 { 0.0 } else { *n }
        }
        other => crate::js::parse_float(&to_string(other)?),
    })
}

/// `parseInt(value, radix)` (`ToString` first).
pub(crate) fn parse_int_value(value: &Value, radix: u32) -> Result<f64, JsError> {
    Ok(crate::js::parse_int(&to_string(value)?, radix))
}

/// `resolveEnumValue(path)` of CanvasRenderer, over the renderer's enum tree
/// `enums` (`this._enums`): numbers and booleans resolve to themselves, a
/// dotted string walks the tree to a number, boolean or `{ value }` entry;
/// anything else is `null`.
pub fn resolve_enum_value(path: &Value, enums: &Value) -> Value {
    let path = match path {
        Value::Undefined | Value::Null => return Value::Null,
        Value::Number(_) | Value::Bool(_) => return path.clone(),
        Value::String(s) => s,
        _ => return Value::Null,
    };
    // Walk by reference; only prototype members and primitive bases (which a
    // real enum tree never routes through) are materialized.
    let mut node: Cow<'_, Value> = Cow::Borrowed(enums);
    for segment in path.split('.').filter(|s| !s.is_empty()) {
        if !node.is_truthy() {
            return Value::Null;
        }
        let next: Cow<'_, Value> = match node {
            Cow::Borrowed(Value::Object(o)) => match o.get(segment) {
                Some(v) => Cow::Borrowed(v),
                None => Cow::Owned(member(node.as_ref(), segment)),
            },
            Cow::Borrowed(v) => Cow::Owned(member(v, segment)),
            Cow::Owned(ref v) => Cow::Owned(member(v, segment)),
        };
        if next.is_undefined() {
            return Value::Null;
        }
        node = next;
    }
    match node.as_ref() {
        Value::Number(_) | Value::Bool(_) => node.into_owned(),
        v @ (Value::Object(_) | Value::Array(_)) => {
            let value = member(v, "value");
            if value.is_undefined() {
                Value::Null
            } else {
                value
            }
        }
        _ => Value::Null,
    }
}

/// `hex.slice(start, end)` by UTF-16 code units, then `parseInt(_, 16) / 255`.
fn hex_channel(units: &[u16], start: usize) -> f64 {
    let end = (start + 2).min(units.len());
    let start = start.min(units.len());
    let part = String::from_utf16_lossy(&units[start..end]);
    crate::js::parse_int(&part, 16) / 255.0
}

/// `[r, g, b]` of a `#rrggbb` string (`value.slice(1)` split in two-unit
/// channels, each `parseInt(_, 16) / 255`).
pub(crate) fn hex_to_rgb(value: &str) -> Value {
    let units: Vec<u16> = value.encode_utf16().skip(1).collect();
    Value::Array(vec![
        Value::Number(hex_channel(&units, 0)),
        Value::Number(hex_channel(&units, 2)),
        Value::Number(hex_channel(&units, 4)),
    ])
}

/// `convertParameterForUniform(value, spec)` of CanvasRenderer, with the
/// renderer's enum tree `enums`: enum names resolve to their values, booleans
/// and buttons become booleans, ints are rounded (booleans 0/1), floats
/// parsed, colors become `[r, g, b]` (hex strings decoded, arrays trimmed or
/// padded), vec3/vec4 components parsed.
pub fn convert_parameter_for_uniform(
    value: &Value,
    spec: &Value,
    enums: &Value,
) -> Result<Value, JsError> {
    if !spec.is_truthy() {
        return Ok(value.clone());
    }
    let spec_enum = member(spec, "enum");
    let spec_enum_path = member(spec, "enumPath");
    let spec_type = member(spec, "type");
    if (spec_enum.is_truthy()
        || spec_enum_path.is_truthy()
        || strict_equals(&spec_type, &Value::from("member")))
        && let Value::String(name) = value
    {
        let mut enum_value = resolve_enum_value(value, enums);
        if enum_value.is_nullish() && (spec_enum.is_truthy() || spec_enum_path.is_truthy()) {
            let base = if spec_enum.is_truthy() {
                &spec_enum
            } else {
                &spec_enum_path
            };
            let path = format!("{}.{}", to_string(base)?, name);
            enum_value = resolve_enum_value(&Value::from(path), enums);
        }
        if !enum_value.is_nullish() {
            return Ok(enum_value);
        }
    }

    match spec_type.as_str() {
        Some("boolean" | "button") => return Ok(Value::Bool(value.is_truthy())),
        Some("int") => {
            if let Value::Bool(b) = value {
                return Ok(Value::Number(if *b { 1.0 } else { 0.0 }));
            }
            return Ok(Value::Number(match value {
                Value::Number(n) => math_round(*n),
                other => parse_int_value(other, 10)?,
            }));
        }
        Some("float") => {
            return Ok(match value {
                Value::Number(_) => value.clone(),
                other => Value::Number(parse_float_value(other)?),
            });
        }
        Some("color") => {
            if let Value::Array(items) = value {
                let mut result = Vec::with_capacity(3);
                for component in items.iter().take(3) {
                    result.push(match component {
                        Value::Number(_) => component.clone(),
                        other => Value::Number(parse_float_value(other)?),
                    });
                }
                while result.len() < 3 {
                    result.push(Value::Number(0.0));
                }
                return Ok(Value::Array(result));
            }
            if let Value::String(s) = value
                && s.starts_with('#')
            {
                return Ok(hex_to_rgb(s));
            }
        }
        Some("vec3" | "vec4") => {
            if let Value::Array(items) = value {
                let mut result = Vec::with_capacity(items.len());
                for component in items {
                    result.push(match component {
                        Value::Number(_) => component.clone(),
                        other => Value::Number(parse_float_value(other)?),
                    });
                }
                return Ok(Value::Array(result));
            }
        }
        _ => {}
    }
    Ok(value.clone())
}

/// `obj[key] = value` where `obj` is a pass member the reference assigns
/// through (`pass.uniforms[name] = ...`): plain objects get the member; a
/// primitive throws the `TypeError` strict-mode code throws.
pub(crate) fn assign_member(target: &mut Value, key: &str, value: Value) -> Result<(), JsError> {
    match target {
        Value::Object(o) => {
            set_plain(o, key, value);
            Ok(())
        }
        Value::Array(items) => {
            // An array index assignment; other keys would become non-index
            // own properties, which an array value cannot hold.
            if crate::value::is_array_index(key)
                && let Ok(i) = key.parse::<usize>()
            {
                if i >= items.len() {
                    items.resize(i + 1, Value::Undefined);
                }
                items[i] = value;
            }
            Ok(())
        }
        Value::Function(_) => Ok(()),
        Value::Undefined | Value::Null => Err(JsError::type_error(format!(
            "Cannot set properties of {} (setting '{key}')",
            if target.is_null() {
                "null"
            } else {
                "undefined"
            }
        ))),
        primitive => Err(JsError::type_error(format!(
            "Cannot create property '{key}' on {} '{}'",
            primitive.type_of(),
            to_string(primitive)?
        ))),
    }
}

/// `key in obj` where `obj` must be an object (the `TypeError` V8 throws for a
/// primitive right operand otherwise).
pub(crate) fn in_object(key: &str, obj: &Value) -> Result<bool, JsError> {
    match obj {
        Value::Object(_) | Value::Array(_) | Value::Function(_) => Ok(has_property(obj, key)),
        _ => Err(JsError::type_error(format!(
            "Cannot use 'in' operator to search for '{key}' in {}",
            to_string(obj)?
        ))),
    }
}

/// `writeUniformAliases(pass, paramName, uniformName, value)` of
/// runtime/uniform-aliases.js: write `value` to every shader uniform the pass
/// feeds from this parameter (`pass.uniformAliases = { shaderUniform:
/// globalName }`). Returns whether any aliased uniform was written.
pub fn write_uniform_aliases(
    pass: &mut Value,
    param_name: &str,
    uniform_name: &Value,
    value: &Value,
) -> Result<bool, JsError> {
    // `pass?.uniformAliases` and `pass.uniforms` are own data members of a
    // graph pass.
    let aliases = pass.get("uniformAliases");
    if !aliases.is_truthy() || !pass.get("uniforms").is_truthy() {
        return Ok(false);
    }
    let mut wrote = false;
    for (shader_name, global_name) in crate::unparser::jsv::entries(&aliases.clone()) {
        if !strict_equals(&global_name, &Value::from(param_name))
            && !strict_equals(&global_name, uniform_name)
        {
            continue;
        }
        // Array.isArray(value) ? value.slice() : value
        let uniforms = pass
            .get_mut("uniforms")
            .ok_or_else(|| cannot_read(&Value::Undefined, &shader_name))?;
        assign_member(uniforms, &shader_name, value.clone())?;
        wrote = true;
    }
    Ok(wrote)
}

/// `pass.id.match(/^node_(\d+)_/)`: the digits, or `None` when the id does not
/// match. A non-string id throws as `pass.id.match` would.
pub(crate) fn pass_node_digits(id: &Value) -> Result<Option<String>, JsError> {
    let id = match id {
        Value::String(s) => s,
        Value::Undefined | Value::Null => return Err(cannot_read(id, "match")),
        _ => return Err(not_a_function("pass.id.match")),
    };
    let Some(rest) = id.strip_prefix("node_") else {
        return Ok(None);
    };
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    if digits.is_empty() || !rest[digits.len()..].starts_with('_') {
        return Ok(None);
    }
    Ok(Some(digits))
}

/// `parseInt(digits, 10)` of a `\d+` capture (correctly rounded, as V8 reads
/// long decimal strings).
pub(crate) fn parse_digits(digits: &str) -> f64 {
    digits.parse::<f64>().unwrap_or(f64::NAN)
}

/// `stepKey.match(/^step_(\d+)$/)`: the digits, or `None`.
pub(crate) fn step_key_digits(step_key: &str) -> Option<&str> {
    let digits = step_key.strip_prefix("step_")?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())).then_some(digits)
}

/// The property key of a uniform name value (`ToPropertyKey`).
pub(crate) fn key_of(v: &Value) -> Result<String, JsError> {
    to_property_key(v)
}
