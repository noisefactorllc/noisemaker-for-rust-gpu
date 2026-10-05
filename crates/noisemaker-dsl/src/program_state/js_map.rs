//! A JavaScript `Map` over [`Value`] keys.
//!
//! ProgramState keeps its routing overrides and media/text metadata in `Map`s
//! keyed by whatever the caller passed: numbers from the setters, strings after
//! `deserialize` (`new Map(Object.entries(...))`). The two never match each other
//! (`map.get(0)` misses a `"0"` key), and that difference is observable, so the
//! port keeps the keys as JavaScript values.

use crate::JsError;
use crate::unparser::jsv::{same_value_zero, to_property_key};
use crate::value::{Object, Value};

/// An insertion-ordered map with JavaScript `Map` semantics: keys compare with
/// SameValueZero (`NaN` matches `NaN`, `-0` is stored as `+0`), setting an
/// existing key keeps its position, deleting a key keeps the order of the rest.
///
/// Object keys never match one another: a [`Value`] owns its members, so two
/// objects are never the same reference.
#[derive(Clone, Debug, PartialEq)]
pub struct JsMap<V> {
    entries: Vec<(Value, V)>,
}

impl<V> Default for JsMap<V> {
    fn default() -> Self {
        JsMap {
            entries: Vec::new(),
        }
    }
}

impl<V> JsMap<V> {
    /// `new Map()`.
    pub fn new() -> Self {
        Self::default()
    }

    fn position(&self, key: &Value) -> Option<usize> {
        self.entries
            .iter()
            .position(|(k, _)| same_value_zero(k, key))
    }

    /// `map.size`.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// `map.get(key)`.
    pub fn get(&self, key: &Value) -> Option<&V> {
        self.position(key).map(|i| &self.entries[i].1)
    }

    pub fn get_mut(&mut self, key: &Value) -> Option<&mut V> {
        self.position(key).map(|i| &mut self.entries[i].1)
    }

    /// `map.has(key)`.
    pub fn has(&self, key: &Value) -> bool {
        self.position(key).is_some()
    }

    /// `map.set(key, value)`.
    pub fn set(&mut self, key: Value, value: V) {
        match self.position(&key) {
            Some(i) => self.entries[i].1 = value,
            None => {
                // Map.prototype.set: "If key is -0𝔽, set key to +0𝔽."
                let mut key = key;
                if let Value::Number(n) = &mut key {
                    *n += 0.0;
                }
                self.entries.push((key, value));
            }
        }
    }

    /// `map.delete(key)`.
    pub fn delete(&mut self, key: &Value) -> bool {
        match self.position(key) {
            Some(i) => {
                self.entries.remove(i);
                true
            }
            None => false,
        }
    }

    /// `map.clear()`.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Entries in insertion order (`map.entries()`).
    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&Value, &V)> {
        self.entries.iter().map(|(k, v)| (k, v))
    }

    /// Keys in insertion order (`map.keys()`).
    pub fn keys(&self) -> impl ExactSizeIterator<Item = &Value> {
        self.entries.iter().map(|(k, _)| k)
    }
}

impl JsMap<Value> {
    /// `new Map(Object.entries(obj))`: string keys in the object's property order.
    pub fn from_object(obj: &Object) -> Self {
        let mut map = JsMap::new();
        for (k, v) in obj.iter() {
            map.set(Value::String(k.clone()), v.clone());
        }
        map
    }

    /// `new Map(Object.entries(value))` for any operand `Object.entries` accepts
    /// (strings enumerate their code units, arrays their indices, other
    /// primitives nothing).
    pub fn from_entries_of(value: &Value) -> Self {
        let mut map = JsMap::new();
        for (k, v) in crate::unparser::jsv::entries(value) {
            map.set(Value::String(k), v);
        }
        map
    }

    /// `Object.fromEntries(map)`: every key through `ToPropertyKey`, so `0` and
    /// `"0"` land on the same member (the later value wins, the first position
    /// stays), and integer-like keys order first as in any JavaScript object.
    pub fn to_object(&self) -> Result<Object, JsError> {
        let mut obj = Object::new();
        for (k, v) in &self.entries {
            crate::unparser::jsv::set_plain(&mut obj, &to_property_key(k)?, v.clone());
        }
        Ok(obj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_value_zero_keys() {
        let mut m = JsMap::new();
        m.set(Value::Number(-0.0), Value::from("a"));
        m.set(Value::Number(f64::NAN), Value::from("nan"));
        m.set(Value::from("0"), Value::from("s"));
        assert_eq!(m.get(&Value::Number(0.0)), Some(&Value::from("a")));
        assert_eq!(m.get(&Value::Number(f64::NAN)), Some(&Value::from("nan")));
        assert_eq!(m.get(&Value::from("0")), Some(&Value::from("s")));
        assert_eq!(m.len(), 3);
        // Setting an existing key keeps its position.
        m.set(Value::Number(0.0), Value::from("b"));
        assert_eq!(m.keys().next(), Some(&Value::Number(0.0)));
        assert!(
            m.keys()
                .next()
                .unwrap()
                .as_f64()
                .unwrap()
                .is_sign_positive()
        );
        // Object.fromEntries: 0 and "0" collide on the property key "0".
        let obj = m.to_object().unwrap();
        assert_eq!(
            Value::Object(obj).to_json().unwrap(),
            r#"{"0":"s","NaN":"nan"}"#
        );
        assert!(m.delete(&Value::Number(0.0)));
        assert!(!m.has(&Value::Number(0.0)));
    }
}
