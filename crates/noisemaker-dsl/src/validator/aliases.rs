//! Port of `lang/effectAliases.js` (`checkEffectAlias`) and `lang/paramAliases.js`
//! over the registry's effect-alias and param-alias tables.

use super::heap::V;
use crate::error::JsError;
use crate::registry::Registry;

/// `checkEffectAlias(opName)`: the deprecation warning for a hidden effect that
/// a newer effect replaces, or `None`.
pub fn check_effect_alias(registry: &Registry, op_name: &str) -> Option<String> {
    let new_name = registry
        .effect_aliases
        .get(op_name)
        .filter(|n| !n.is_empty())?;
    let old_name = op_name.rsplit('.').next().unwrap_or(op_name);
    Some(format!(
        "effect '{old_name}' is deprecated, use '{new_name}' instead."
    ))
}

/// `ALIAS_EOL_DATE` of `lang/paramAliases.js`.
pub const ALIAS_EOL_DATE: &str = "2027-06-01";

/// `resolveParamAliases(opName, kwargs)` on the validator's object values:
/// migrate every deprecated key the registry maps to its new name (the new
/// key's value wins when both are present), rewrite the object in place, and
/// return the deprecation warnings in registry order.
pub(crate) fn resolve_param_aliases(
    registry: &Registry,
    op_name: &str,
    kwargs: &V,
) -> Result<Vec<String>, JsError> {
    let Some(aliases) = registry.param_aliases.get(op_name) else {
        return Ok(Vec::new());
    };
    let mut warnings = Vec::new();
    let mut changed = false;
    let mut rewritten: Vec<(String, V)> = Vec::new();
    for (key, value) in object_entries_shallow(kwargs) {
        let mut mapped: Option<String> = None;
        for (old_name, new_name) in aliases {
            if key == *old_name {
                mapped = Some(new_name.clone());
                break;
            }
        }
        match mapped {
            Some(new_name) => {
                warnings.push(format!(
                    "param '{key}' is deprecated, use '{new_name}' instead. Aliases will be removed on {ALIAS_EOL_DATE}."
                ));
                if !rewritten.iter().any(|(k, _)| *k == new_name) {
                    rewritten.push((new_name, value));
                }
                changed = true;
            }
            None => rewritten.push((key, value)),
        }
    }
    if changed {
        let fresh = V::new_object();
        for (k, v) in rewritten {
            fresh.set(&k, v)?;
        }
        if let V::Obj(o) = kwargs {
            if let V::Obj(f) = &fresh {
                *o.borrow_mut() = f.borrow().clone();
            }
        }
    }
    Ok(warnings)
}

/// The enumerable own entries of an object value (empty for other types).
fn object_entries_shallow(value: &V) -> Vec<(String, V)> {
    match value {
        V::Obj(o) => o.borrow().entries(),
        _ => Vec::new(),
    }
}

/// `getParamAliases(opName)` as ordered `(oldName, newName)` pairs.
pub fn get_param_aliases(registry: &Registry, op_name: &str) -> Vec<(String, String)> {
    registry
        .param_aliases
        .get(op_name)
        .map(|m| m.iter().cloned().collect())
        .unwrap_or_default()
}
