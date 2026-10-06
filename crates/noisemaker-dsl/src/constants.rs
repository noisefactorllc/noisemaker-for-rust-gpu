//! Port of `lang/constants.js`: the block categories of the visual program
//! editor, which of them begin a chain, and the DSL block type of each.

use crate::unparser::jsv::object_member;
use crate::value::{Object, Value};

/// `STARTER_BLOCK_CATEGORIES`: the categories whose blocks begin a chain.
pub const STARTER_BLOCK_CATEGORIES: &[&str] = &["Synths", "Generators", "Surfaces"];

/// `DEFAULT_BLOCK_CATEGORY_TYPE`.
pub const DEFAULT_BLOCK_CATEGORY_TYPE: &str = "post";

/// `BLOCK_CATEGORY_TYPES`: category → block type, in the reference's order.
pub const BLOCK_CATEGORY_TYPES: &[(&str, &str)] = &[
    ("Synths", "synth"),
    ("Generators", "synth"),
    ("Surfaces", "synth"),
    ("Mixers", "mixer"),
    ("Post", "post"),
    ("Geometry", "post"),
    ("Color & FX", "post"),
    ("Modulation", "post"),
    ("Control", "post"),
    ("Utilities", "post"),
    ("Variables", "post"),
];

/// `BLOCK_CATEGORY_TYPES` as the reference's object.
pub fn block_category_types_value() -> Value {
    let mut o = Object::new();
    for (category, ty) in BLOCK_CATEGORY_TYPES {
        o.insert(*category, Value::from(*ty));
    }
    Value::Object(o)
}

/// `isStarterBlockCategory(category)`: a non-empty string naming a starter
/// category.
pub fn is_starter_block_category(category: &Value) -> bool {
    category
        .as_str()
        .is_some_and(|c| STARTER_BLOCK_CATEGORIES.contains(&c))
}

/// The block type of a known category.
pub fn block_category_type(category: &str) -> Option<&'static str> {
    BLOCK_CATEGORY_TYPES
        .iter()
        .find(|(c, _)| *c == category)
        .map(|(_, t)| *t)
}

/// `getBlockCategoryType(category)`: the category's block type, or
/// [`DEFAULT_BLOCK_CATEGORY_TYPE`] for anything else (non-strings, the empty
/// string, unknown categories). A property read, so the names of
/// `Object.prototype` members (`constructor`, `toString`, ...) return those
/// inherited members, as in the reference.
pub fn get_block_category_type(category: &Value) -> Value {
    let Some(category) = category.as_str().filter(|c| !c.is_empty()) else {
        return Value::from(DEFAULT_BLOCK_CATEGORY_TYPE);
    };
    let Value::Object(table) = block_category_types_value() else {
        unreachable!()
    };
    let found = object_member(&table, category);
    if found.is_truthy() {
        found
    } else {
        Value::from(DEFAULT_BLOCK_CATEGORY_TYPE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories() {
        assert!(is_starter_block_category(&Value::from("Synths")));
        assert!(!is_starter_block_category(&Value::from("Mixers")));
        assert!(!is_starter_block_category(&Value::Number(1.0)));
        assert_eq!(
            get_block_category_type(&Value::from("Mixers")).as_str(),
            Some("mixer")
        );
        assert_eq!(
            get_block_category_type(&Value::from("nope")).as_str(),
            Some("post")
        );
        assert_eq!(get_block_category_type(&Value::Null).as_str(), Some("post"));
        assert_eq!(block_category_type("Color & FX"), Some("post"));
    }
}
