//! Port of `lang/effectAliases.js` (`checkEffectAlias`) over the registry's
//! effect-alias table.

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
