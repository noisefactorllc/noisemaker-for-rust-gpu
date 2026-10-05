//! Port of `lang/enumPaths.js`: enum member paths (`palette.sherbet`,
//! `filter.channel.channel.r`) and the enum prefixes of parameters.

use crate::js::trim;
use crate::value::Value;

use super::heap::V;

/// `normalizeMemberPath(value)` for a registry value (a parameter's `enum`,
/// `enumPath` or `default`): the non-empty string segments, or `None`.
pub fn normalize_member_path(value: &Value) -> Option<Vec<String>> {
    if !value.is_truthy() {
        return None;
    }
    let parts: Vec<String> = match value {
        Value::Array(items) => items
            .iter()
            .filter_map(|seg| seg.as_str().filter(|s| !s.is_empty()).map(str::to_owned))
            .collect(),
        Value::String(s) => s
            .split('.')
            .map(trim)
            .filter(|seg| !seg.is_empty())
            .map(str::to_owned)
            .collect(),
        Value::Number(n) => vec![crate::js::number_to_string(*n)],
        _ => return None,
    };
    (!parts.is_empty()).then_some(parts)
}

/// `normalizeMemberPath(value)` for a value of the validator's object graph (an
/// AST node's `path`).
pub(crate) fn normalize_member_path_v(value: &V) -> Option<Vec<String>> {
    if !value.truthy() {
        return None;
    }
    let parts: Vec<String> = match value {
        V::Arr(items) => items
            .borrow()
            .iter()
            .filter_map(|seg| seg.as_str().filter(|s| !s.is_empty()).map(str::to_owned))
            .collect(),
        V::Str(s) => s
            .split('.')
            .map(trim)
            .filter(|seg| !seg.is_empty())
            .map(str::to_owned)
            .collect(),
        V::Num(n) => vec![crate::js::number_to_string(*n)],
        _ => return None,
    };
    (!parts.is_empty()).then_some(parts)
}

/// `pathStartsWith(path, prefix)`; an absent or empty prefix matches every path.
pub fn path_starts_with(path: &[String], prefix: Option<&[String]>) -> bool {
    let Some(prefix) = prefix.filter(|p| !p.is_empty()) else {
        return true;
    };
    path.len() >= prefix.len() && path.iter().zip(prefix).all(|(a, b)| a == b)
}

/// `applyEnumPrefix(path, prefix)`: `path` qualified by the part of `prefix` it
/// does not already start with.
pub fn apply_enum_prefix(path: &[String], prefix: Option<&[String]>) -> Vec<String> {
    if path.is_empty() {
        return path.to_vec();
    }
    let Some(prefix) = prefix.filter(|p| !p.is_empty()) else {
        return path.to_vec();
    };
    if path_starts_with(path, Some(prefix)) {
        return path.to_vec();
    }
    for i in 1..prefix.len() {
        let suffix = &prefix[i..];
        if path_starts_with(path, Some(suffix)) {
            return prefix[..i].iter().chain(path).cloned().collect();
        }
    }
    prefix.iter().chain(path).cloned().collect()
}

/// `stripEnumPrefix(path, prefix)`: `path` without the part of `prefix` it starts
/// with (the unparser's inverse of [`apply_enum_prefix`]).
pub fn strip_enum_prefix(path: &Value, prefix: &Value) -> Option<Vec<String>> {
    let normalized_path = normalize_member_path(path)?;
    let Some(normalized_prefix) = normalize_member_path(prefix) else {
        return Some(normalized_path);
    };
    if path_starts_with(&normalized_path, Some(&normalized_prefix)) {
        return Some(normalized_path[normalized_prefix.len()..].to_vec());
    }
    for i in (1..normalized_prefix.len()).rev() {
        let suffix = &normalized_prefix[i..];
        if path_starts_with(&normalized_path, Some(suffix)) {
            return Some(normalized_path[suffix.len()..].to_vec());
        }
    }
    Some(normalized_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn prefixes() {
        let prefix = p(&["filter", "channel", "channel"]);
        assert_eq!(
            apply_enum_prefix(&p(&["r"]), Some(&prefix)),
            p(&["filter", "channel", "channel", "r"])
        );
        assert_eq!(
            apply_enum_prefix(&p(&["channel", "r"]), Some(&prefix)),
            p(&["filter", "channel", "channel", "r"])
        );
        assert_eq!(
            apply_enum_prefix(&p(&["channel", "channel", "r"]), Some(&prefix)),
            p(&["filter", "channel", "channel", "r"])
        );
        assert_eq!(apply_enum_prefix(&p(&["x"]), None), p(&["x"]));
        assert_eq!(
            normalize_member_path(&Value::from(" a . b ..c ")),
            Some(p(&["a", "b", "c"]))
        );
        assert_eq!(normalize_member_path(&Value::from("")), None);
        assert_eq!(
            strip_enum_prefix(
                &Value::from("filter.channel.channel.r"),
                &Value::from("filter.channel.channel")
            ),
            Some(p(&["r"]))
        );
    }
}
