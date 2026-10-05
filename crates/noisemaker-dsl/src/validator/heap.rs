//! JavaScript object semantics for the validator.
//!
//! `validate` mutates the objects it is given and shares them between its outputs:
//! `resolveParamAliases` renames the keys of a call's `kwargs` object, which the
//! step then publishes as `rawKwargs`; a member-typed argument rewrites its AST
//! node's `path`, which every other reference to that node (an outer step's
//! `rawKwargs`, a variable's stored call) then shows; automation descriptors keep
//! their AST node as `_ast`. The reference serializes the result only at the end,
//! so every output shows the final state of every shared object.
//!
//! [`V`] reproduces that object graph: objects and arrays are shared, mutable
//! references (`{...x}` copies one level, as in JavaScript), and [`V::to_value`]
//! serializes the graph at the end like `JSON.stringify`. The operations follow
//! the ECMAScript semantics the reference relies on (property reads with the
//! `TypeError`s V8 raises, `hasOwnProperty`, `ToString`, strict equality,
//! relational comparison, JSON cloning).

use std::cell::RefCell;
use std::rc::Rc;

use indexmap::IndexMap;

use crate::error::JsError;
use crate::js::number_to_string;
use crate::value::{Object, Value, is_array_index};

/// The own property names of `Object.prototype` (V8). A property read that misses
/// an object's own properties finds these.
const OBJECT_PROTOTYPE_METHODS: &[&str] = &[
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
    "toLocaleString",
];

/// A JavaScript value.
#[derive(Clone, Default)]
pub(crate) enum V {
    #[default]
    Undefined,
    Null,
    Bool(bool),
    Num(f64),
    Str(Rc<str>),
    Obj(Rc<RefCell<JsObject>>),
    Arr(Rc<RefCell<Vec<V>>>),
    /// A function value: a closure the validator creates, whose text is its
    /// source as `Function.prototype.toString` gives it, or an inherited method,
    /// whose text is its name. Functions never serialize.
    Func(Rc<str>),
}

/// An ordinary object: own properties in JavaScript property order, and the
/// non-enumerable ones (`Object.defineProperty(o, k, {enumerable: false})`, see
/// [`Object::define_hidden`]), which reads and `hasOwnProperty` see but key
/// enumeration, spreads, JSON cloning and serialization do not.
#[derive(Clone, Default)]
pub(crate) struct JsObject {
    props: IndexMap<String, V>,
    hidden: IndexMap<String, V>,
}

impl JsObject {
    pub(crate) fn get(&self, key: &str) -> Option<&V> {
        self.props.get(key).or_else(|| self.hidden.get(key))
    }

    /// `Object.prototype.hasOwnProperty.call(obj, key)`.
    pub(crate) fn contains(&self, key: &str) -> bool {
        self.props.contains_key(key) || self.hidden.contains_key(key)
    }

    /// `obj[key] = value`: an existing key keeps its position (and its
    /// enumerability), a new array-index key is placed among the index keys in
    /// ascending order, a new string key is appended.
    pub(crate) fn insert(&mut self, key: &str, value: V) {
        if let Some(slot) = self.props.get_mut(key) {
            *slot = value;
            return;
        }
        if let Some(slot) = self.hidden.get_mut(key) {
            *slot = value;
            return;
        }
        if is_array_index(key) {
            let n: u64 = key.parse().expect("array index");
            let pos = self
                .props
                .keys()
                .position(|k| !is_array_index(k) || k.parse::<u64>().expect("array index") > n)
                .unwrap_or(self.props.len());
            self.props.shift_insert(pos, key.to_owned(), value);
        } else {
            self.props.insert(key.to_owned(), value);
        }
    }

    /// `delete obj[key]`.
    pub(crate) fn remove(&mut self, key: &str) -> Option<V> {
        self.props
            .shift_remove(key)
            .or_else(|| self.hidden.shift_remove(key))
    }

    /// `Object.entries(obj)`.
    pub(crate) fn entries(&self) -> Vec<(String, V)> {
        self.props
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }

    /// `{...obj}`: the enumerable own properties only.
    fn enumerable_copy(&self) -> JsObject {
        JsObject {
            props: self.props.clone(),
            hidden: IndexMap::new(),
        }
    }
}

/// The number of UTF-16 code units of `s` (`s.length`).
pub(crate) fn utf16_len(s: &str) -> usize {
    s.chars().map(char::len_utf16).sum()
}

/// `s.slice(0, n)` in UTF-16 code units. A surrogate pair split by `n` leaves a
/// lone surrogate in JavaScript, which a Rust string cannot hold; the replacement
/// character stands for it.
pub(crate) fn utf16_prefix(s: &str, n: usize) -> String {
    let mut out = String::new();
    let mut units = 0;
    for c in s.chars() {
        let w = c.len_utf16();
        if units + w > n {
            if units < n {
                out.push('\u{FFFD}');
            }
            break;
        }
        out.push(c);
        units += w;
    }
    out
}

impl V {
    pub(crate) fn str(s: &str) -> V {
        V::Str(Rc::from(s))
    }

    pub(crate) fn num(n: f64) -> V {
        V::Num(n)
    }

    pub(crate) fn new_object() -> V {
        V::Obj(Rc::new(RefCell::new(JsObject::default())))
    }

    pub(crate) fn object_from(entries: Vec<(&str, V)>) -> V {
        let mut o = JsObject::default();
        for (k, v) in entries {
            o.insert(k, v);
        }
        V::Obj(Rc::new(RefCell::new(o)))
    }

    pub(crate) fn array_from(items: Vec<V>) -> V {
        V::Arr(Rc::new(RefCell::new(items)))
    }

    /// A fresh object graph holding `value`.
    pub(crate) fn from_value(value: &Value) -> V {
        match value {
            Value::Undefined => V::Undefined,
            Value::Null => V::Null,
            Value::Bool(b) => V::Bool(*b),
            Value::Number(n) => V::Num(*n),
            Value::String(s) => V::str(s),
            Value::Array(a) => V::array_from(a.iter().map(V::from_value).collect()),
            Value::Object(o) => {
                let mut obj = JsObject::default();
                for (k, v) in o.iter() {
                    obj.insert(k, V::from_value(v));
                }
                for (k, v) in o.hidden_members() {
                    obj.hidden.insert(k.clone(), V::from_value(v));
                }
                V::Obj(Rc::new(RefCell::new(obj)))
            }
            Value::Function(src) => V::Func(Rc::from(src.as_str())),
        }
    }

    /// The value as data (the object graph unshared). Serializing the result
    /// follows `JSON.stringify`.
    pub(crate) fn to_value(&self) -> Value {
        match self {
            V::Undefined => Value::Undefined,
            V::Null => Value::Null,
            V::Bool(b) => Value::Bool(*b),
            V::Num(n) => Value::Number(*n),
            V::Str(s) => Value::String(s.to_string()),
            V::Arr(a) => Value::Array(a.borrow().iter().map(V::to_value).collect()),
            V::Obj(o) => {
                let o = o.borrow();
                let mut out = Object::new();
                for (k, v) in o.props.iter() {
                    out.insert(k.clone(), v.to_value());
                }
                for (k, v) in o.hidden.iter() {
                    out.define_hidden(k.clone(), v.to_value());
                }
                Value::Object(out)
            }
            V::Func(src) => Value::Function(src.to_string()),
        }
    }

    pub(crate) fn is_undefined(&self) -> bool {
        matches!(self, V::Undefined)
    }

    /// `v === null`.
    pub(crate) fn is_null(&self) -> bool {
        matches!(self, V::Null)
    }

    /// `v === null || v === undefined`.
    pub(crate) fn is_nullish(&self) -> bool {
        matches!(self, V::Undefined | V::Null)
    }

    /// `!!v`.
    pub(crate) fn truthy(&self) -> bool {
        match self {
            V::Undefined | V::Null => false,
            V::Bool(b) => *b,
            V::Num(n) => *n != 0.0 && !n.is_nan(),
            V::Str(s) => !s.is_empty(),
            V::Obj(_) | V::Arr(_) | V::Func(_) => true,
        }
    }

    /// `typeof v`.
    pub(crate) fn type_of(&self) -> &'static str {
        match self {
            V::Undefined => "undefined",
            V::Null | V::Obj(_) | V::Arr(_) => "object",
            V::Bool(_) => "boolean",
            V::Num(_) => "number",
            V::Str(_) => "string",
            V::Func(_) => "function",
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            V::Str(s) => Some(s),
            _ => None,
        }
    }

    /// `v === "text"`.
    pub(crate) fn is_str(&self, text: &str) -> bool {
        matches!(self, V::Str(s) if &**s == text)
    }

    pub(crate) fn is_array(&self) -> bool {
        matches!(self, V::Arr(_))
    }

    /// The elements of an array (a snapshot), or `None`.
    pub(crate) fn elements(&self) -> Option<Vec<V>> {
        match self {
            V::Arr(a) => Some(a.borrow().clone()),
            _ => None,
        }
    }

    /// `v[key]`: own properties, then (for objects) `Object.prototype`'s. Reading
    /// a property of `undefined` or `null` throws V8's TypeError.
    pub(crate) fn get(&self, key: &str) -> Result<V, JsError> {
        Ok(match self {
            V::Undefined | V::Null => {
                return Err(JsError::type_error(format!(
                    "Cannot read properties of {} (reading '{key}')",
                    if self.is_undefined() {
                        "undefined"
                    } else {
                        "null"
                    }
                )));
            }
            V::Obj(o) => match o.borrow().get(key) {
                Some(v) => v.clone(),
                None if key == "__proto__" => V::new_object(),
                None if OBJECT_PROTOTYPE_METHODS.contains(&key) => V::Func(Rc::from(key)),
                None => V::Undefined,
            },
            V::Arr(a) => {
                let a = a.borrow();
                if key == "length" {
                    V::Num(a.len() as f64)
                } else if is_array_index(key) {
                    key.parse::<usize>()
                        .ok()
                        .and_then(|i| a.get(i).cloned())
                        .unwrap_or_default()
                } else {
                    V::Undefined
                }
            }
            V::Str(s) => {
                if key == "length" {
                    V::Num(utf16_len(s) as f64)
                } else if is_array_index(key) {
                    let units: Vec<u16> = s.encode_utf16().collect();
                    match key.parse::<usize>().ok().and_then(|i| units.get(i)) {
                        Some(&u) => V::Str(Rc::from(String::from_utf16_lossy(&[u]).as_str())),
                        None => V::Undefined,
                    }
                } else {
                    V::Undefined
                }
            }
            V::Bool(_) | V::Num(_) | V::Func(_) => V::Undefined,
        })
    }

    /// `v?.[key]`: `undefined` for a nullish base.
    pub(crate) fn get_opt(&self, key: &str) -> V {
        if self.is_nullish() {
            return V::Undefined;
        }
        self.get(key).unwrap_or_default()
    }

    /// `Object.prototype.hasOwnProperty.call(v, key)` for a non-nullish `v`.
    pub(crate) fn has_own(&self, key: &str) -> bool {
        match self {
            V::Obj(o) => o.borrow().contains(key),
            V::Arr(a) => {
                key == "length"
                    || (is_array_index(key)
                        && key.parse::<usize>().is_ok_and(|i| i < a.borrow().len()))
            }
            V::Str(s) => {
                key == "length"
                    || (is_array_index(key) && key.parse::<usize>().is_ok_and(|i| i < utf16_len(s)))
            }
            // Functions own `length` and `name` (the validator never asks).
            V::Func(_) => key == "length" || key == "name",
            V::Undefined | V::Null | V::Bool(_) | V::Num(_) => false,
        }
    }

    /// `v[key] = value` (strict mode, as in the reference's ES modules).
    pub(crate) fn set(&self, key: &str, value: V) -> Result<(), JsError> {
        match self {
            V::Obj(o) => {
                o.borrow_mut().insert(key, value);
                Ok(())
            }
            V::Arr(a) => {
                if is_array_index(key) {
                    let i: usize = key.parse().expect("array index");
                    let mut a = a.borrow_mut();
                    if i >= a.len() {
                        a.resize(i + 1, V::Undefined);
                    }
                    a[i] = value;
                }
                // Other array properties never serialize.
                Ok(())
            }
            V::Undefined | V::Null => Err(JsError::type_error(format!(
                "Cannot set properties of {} (setting '{key}')",
                if self.is_undefined() {
                    "undefined"
                } else {
                    "null"
                }
            ))),
            V::Func(_) => Ok(()),
            V::Bool(_) | V::Num(_) | V::Str(_) => Err(JsError::type_error(format!(
                "Cannot create property '{key}' on {} '{}'",
                self.type_of(),
                self.to_js_string()
            ))),
        }
    }

    /// `{...v}`: a new object with the own enumerable properties of `v`.
    pub(crate) fn spread(&self) -> V {
        let obj = match self {
            V::Obj(o) => o.borrow().enumerable_copy(),
            V::Arr(a) => {
                let mut o = JsObject::default();
                for (i, v) in a.borrow().iter().enumerate() {
                    o.insert(&i.to_string(), v.clone());
                }
                o
            }
            V::Str(s) => {
                let mut o = JsObject::default();
                for (i, u) in s.encode_utf16().enumerate() {
                    o.insert(&i.to_string(), V::str(&String::from_utf16_lossy(&[u])));
                }
                o
            }
            _ => JsObject::default(),
        };
        V::Obj(Rc::new(RefCell::new(obj)))
    }

    /// `JSON.parse(JSON.stringify(v))` for an object or array: a fresh graph in
    /// which `undefined` and function members are dropped, `undefined` and
    /// function array elements and non-finite numbers become `null`, and `-0`
    /// becomes `0`.
    pub(crate) fn json_clone(&self) -> V {
        fn element(v: &V) -> V {
            match v {
                V::Undefined | V::Func(_) => V::Null,
                other => member(other).unwrap_or(V::Null),
            }
        }
        fn member(v: &V) -> Option<V> {
            Some(match v {
                V::Undefined | V::Func(_) => return None,
                V::Null => V::Null,
                V::Bool(b) => V::Bool(*b),
                V::Num(n) => {
                    if n.is_finite() {
                        V::Num(if *n == 0.0 { 0.0 } else { *n })
                    } else {
                        V::Null
                    }
                }
                V::Str(s) => V::Str(s.clone()),
                V::Arr(a) => V::array_from(a.borrow().iter().map(element).collect()),
                V::Obj(o) => {
                    let mut out = JsObject::default();
                    for (k, v) in o.borrow().props.iter() {
                        if let Some(v) = member(v) {
                            out.insert(k, v);
                        }
                    }
                    V::Obj(Rc::new(RefCell::new(out)))
                }
            })
        }
        member(self).unwrap_or_default()
    }

    /// `String(v)` / a template-literal substitution.
    pub(crate) fn to_js_string(&self) -> String {
        match self {
            V::Undefined => "undefined".into(),
            V::Null => "null".into(),
            V::Bool(b) => b.to_string(),
            V::Num(n) => number_to_string(*n),
            V::Str(s) => s.to_string(),
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
                .join(","),
            V::Obj(_) => "[object Object]".into(),
            V::Func(src) => src.to_string(),
        }
    }

    /// `ToNumber(v)` (objects through `ToPrimitive`).
    pub(crate) fn to_number(&self) -> f64 {
        match self {
            V::Undefined => f64::NAN,
            V::Null => 0.0,
            V::Bool(b) => f64::from(u8::from(*b)),
            V::Num(n) => *n,
            V::Str(s) => crate::js::string_to_number(s),
            V::Arr(_) | V::Obj(_) | V::Func(_) => crate::js::string_to_number(&self.to_js_string()),
        }
    }

    /// `a === b` (objects by identity).
    pub(crate) fn strict_equals(&self, other: &V) -> bool {
        match (self, other) {
            (V::Undefined, V::Undefined) | (V::Null, V::Null) => true,
            (V::Bool(a), V::Bool(b)) => a == b,
            (V::Num(a), V::Num(b)) => a == b,
            (V::Str(a), V::Str(b)) => a == b,
            (V::Obj(a), V::Obj(b)) => Rc::ptr_eq(a, b),
            (V::Arr(a), V::Arr(b)) => Rc::ptr_eq(a, b),
            (V::Func(a), V::Func(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// `v.slice(0, end)`, or `v.slice()` for `end: None`: a shallow copy of an
    /// array's leading elements, a string's leading UTF-16 code units. `expr` is
    /// how V8 names the callee in the TypeError for any other receiver.
    pub(crate) fn slice(&self, expr: &str, end: Option<usize>) -> Result<V, JsError> {
        match self {
            V::Arr(a) => {
                let a = a.borrow();
                let n = end.map_or(a.len(), |e| e.min(a.len()));
                Ok(V::array_from(a[..n].to_vec()))
            }
            V::Str(s) => Ok(match end {
                Some(n) => V::str(&utf16_prefix(s, n)),
                None => V::Str(s.clone()),
            }),
            V::Undefined | V::Null => self.get("slice"),
            _ => Err(JsError::type_error(format!(
                "{expr}.slice is not a function"
            ))),
        }
    }
}

impl std::fmt::Debug for V {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_value())
    }
}

/// `a < b` for a number `b` (the abstract relational comparison).
pub(crate) fn less_than(a: &V, b: f64) -> bool {
    a.to_number() < b
}

/// `a > b` for a number `b`.
pub(crate) fn greater_than(a: &V, b: f64) -> bool {
    a.to_number() > b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_clone_semantics() {
        let v = V::from_value(&Value::from_json(r#"{"a":1,"b":[1,2]}"#).unwrap());
        v.set("n", V::Num(f64::NAN)).unwrap();
        v.set("u", V::Undefined).unwrap();
        v.set("z", V::Num(-0.0)).unwrap();
        let c = v.json_clone();
        assert_eq!(
            c.to_value().to_json().unwrap(),
            r#"{"a":1,"b":[1,2],"n":null,"z":0}"#
        );
        assert!(c.get("u").unwrap().is_undefined());
        assert!(!c.has_own("u"));
    }

    #[test]
    fn shared_mutation_is_visible() {
        let inner = V::new_object();
        let outer = V::object_from(vec![("x", inner.clone())]);
        inner.set("k", V::Num(1.0)).unwrap();
        assert_eq!(outer.to_value().to_json().unwrap(), r#"{"x":{"k":1}}"#);
    }

    #[test]
    fn property_reads() {
        let s = V::str("😀a");
        assert!(matches!(s.get("length").unwrap(), V::Num(n) if n == 3.0));
        assert!(s.has_own("length"));
        assert!(matches!(
            V::new_object().get("toString").unwrap(),
            V::Func(_)
        ));
        let err = V::Undefined.get("map").unwrap_err();
        assert_eq!(
            err.to_string(),
            "TypeError: Cannot read properties of undefined (reading 'map')"
        );
        assert_eq!(utf16_prefix("ab😀", 3), "ab\u{FFFD}");
    }
}
