//! Ports of `lang/paramAliases.js` (`resolveParamAliases`) and
//! `lang/effectAliases.js` (`checkEffectAlias`) over the registry's alias tables.

use crate::error::JsError;
use crate::registry::Registry;

use super::heap::V;

/// `ALIAS_EOL_DATE`.
pub const ALIAS_EOL_DATE: &str = "2026-09-01";

/// `Array.prototype`'s own property names (V8), which `key in array` finds.
const ARRAY_PROTOTYPE_NAMES: &[&str] = &[
    "length",
    "constructor",
    "at",
    "concat",
    "copyWithin",
    "fill",
    "find",
    "findIndex",
    "findLast",
    "findLastIndex",
    "lastIndexOf",
    "pop",
    "push",
    "reverse",
    "shift",
    "unshift",
    "slice",
    "sort",
    "splice",
    "includes",
    "indexOf",
    "join",
    "keys",
    "entries",
    "values",
    "forEach",
    "filter",
    "flat",
    "flatMap",
    "map",
    "every",
    "some",
    "reduce",
    "reduceRight",
    "toReversed",
    "toSorted",
    "toSpliced",
    "with",
    "toLocaleString",
    "toString",
];

/// `Function.prototype`'s own property names (V8).
const FUNCTION_PROTOTYPE_NAMES: &[&str] = &[
    "length",
    "name",
    "constructor",
    "apply",
    "bind",
    "call",
    "toString",
    "arguments",
    "caller",
];

/// `Object.prototype`'s own property names (V8).
const OBJECT_PROTOTYPE_NAMES: &[&str] = &[
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
];

/// `key in obj`: own properties, then the prototype chain's. A primitive right
/// operand throws V8's TypeError.
fn has_property(obj: &V, key: &str) -> Result<bool, JsError> {
    let inherited = match obj {
        V::Obj(_) => false,
        V::Arr(_) => ARRAY_PROTOTYPE_NAMES.contains(&key),
        V::Func(_) => FUNCTION_PROTOTYPE_NAMES.contains(&key),
        _ => {
            return Err(JsError::type_error(format!(
                "Cannot use 'in' operator to search for '{key}' in {}",
                obj.to_js_string()
            )));
        }
    };
    Ok(obj.has_own(key) || inherited || OBJECT_PROTOTYPE_NAMES.contains(&key))
}

/// `resolveParamAliases(opName, kwargs)`: renames deprecated keys of `kwargs` in
/// place (the new key wins when both are present) and returns one warning per
/// alias hit.
pub(crate) fn resolve_param_aliases(
    registry: &Registry,
    op_name: &str,
    kwargs: &V,
) -> Result<Vec<String>, JsError> {
    let mut warnings = Vec::new();
    let Some(aliases) = registry.param_aliases.get(op_name) else {
        return Ok(warnings);
    };
    for (old_name, new_name) in aliases {
        if !has_property(kwargs, old_name)? {
            continue;
        }
        if !has_property(kwargs, new_name)? {
            let value = kwargs.get(old_name)?;
            kwargs.set(new_name, value)?;
        }
        // `delete kwargs[oldName]` (own properties only).
        if let V::Obj(o) = kwargs {
            o.borrow_mut().remove(old_name);
        }
        warnings.push(format!(
            "param '{old_name}' is deprecated, use '{new_name}' instead. Aliases will be removed on {ALIAS_EOL_DATE}."
        ));
    }
    Ok(warnings)
}

/// `checkEffectAlias(opName)`: the deprecation warning for a hidden effect that
/// a newer effect replaces, or `None`.
pub(crate) fn check_effect_alias(registry: &Registry, op_name: &str) -> Option<String> {
    let new_name = registry
        .effect_aliases
        .get(op_name)
        .filter(|n| !n.is_empty())?;
    let old_name = op_name.rsplit('.').next().unwrap_or(op_name);
    Some(format!(
        "effect '{old_name}' is deprecated, use '{new_name}' instead. Aliases will be removed on {ALIAS_EOL_DATE}."
    ))
}
