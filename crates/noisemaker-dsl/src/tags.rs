//! Port of `runtime/tags.js`: effect tags and effect namespaces.
//!
//! Tags are curated labels for discovering and grouping effects; the tag table
//! is fixed ([`TAG_DEFINITIONS`]). Namespaces group effects for the DSL's
//! `search` directive: the built-in ones ([`BUILTIN_NAMESPACES`]) are always
//! registered, and hosts add their own with [`Registry::register_namespace`].
//! The reference keeps the namespace table in a module-level map behind a
//! read-only `NAMESPACE_DESCRIPTIONS` proxy and a live `VALID_NAMESPACES`
//! array; here it lives in the [`Registry`], readable through
//! [`Registry::namespace_descriptions`], [`Registry::valid_namespaces`] and
//! [`Registry::get_namespace_description`] and writable only through
//! [`Registry::register_namespace`] and [`Registry::unregister_namespace`]
//! (the proxy's "use registerNamespace()" rule, enforced by the type system).
//! The parser's `search` directive accepts exactly the registered namespaces.

use crate::error::JsError;
use crate::lexer::is_reserved_keyword;
use crate::registry::Registry;
use crate::unparser::jsv::object_member;
use crate::value::{Object, Value};

/// One entry of `TAG_DEFINITIONS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TagDefinition {
    pub id: &'static str,
    pub description: &'static str,
}

impl TagDefinition {
    /// `{id, description}`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("id", Value::from(self.id));
        o.insert("description", Value::from(self.description));
        Value::Object(o)
    }
}

const fn tag(id: &'static str, description: &'static str) -> TagDefinition {
    TagDefinition { id, description }
}

/// `TAG_DEFINITIONS`, in the reference's key order.
pub const TAG_DEFINITIONS: &[TagDefinition] = &[
    tag("color", "Color manipulation"),
    tag("distort", "Input distortion"),
    tag("edges", "Accentuate or isolate texture edges"),
    tag("geometric", "Shapes"),
    tag("lens", "Emulated camera lens effects"),
    tag("noise", "Very noisy"),
    tag("transform", "Moves stuff around"),
    tag("util", "Utility function"),
    tag("sim", "Simulations with temporal state"),
    tag("3d", "3D volumetric effects"),
    tag("audio", "Audio-reactive effects"),
    tag("agents", "Particle and agent-based systems"),
    tag("antialiasing", "Edge smoothing and antialiasing"),
    tag("artist", "Artistic media emulation"),
    tag("blend", "Compositing and blend modes"),
    tag("blur", "Blur and softening"),
    tag("fractal", "Fractal patterns"),
    tag("geometry", "3D mesh geometry"),
    tag("glitch", "Glitch and corruption effects"),
    tag("image", "Image-input effects"),
    tag("mesh", "3D mesh rendering"),
    tag("midi", "MIDI-reactive effects"),
    tag("palette", "Color palette mapping"),
    tag("pattern", "Repeating or structured patterns"),
    tag("pixel", "Pixelation and pixel-level effects"),
    tag("text", "Text rendering"),
    tag("tiling", "Seamless tiling"),
    tag("video", "Video-input effects"),
];

/// `VALID_TAGS`: the ids of [`TAG_DEFINITIONS`], in order.
pub const VALID_TAGS: &[&str] = &[
    "color",
    "distort",
    "edges",
    "geometric",
    "lens",
    "noise",
    "transform",
    "util",
    "sim",
    "3d",
    "audio",
    "agents",
    "antialiasing",
    "artist",
    "blend",
    "blur",
    "fractal",
    "geometry",
    "glitch",
    "image",
    "mesh",
    "midi",
    "palette",
    "pattern",
    "pixel",
    "text",
    "tiling",
    "video",
];

/// `BUILTIN_NAMESPACE`: the namespace that needs no `search` directive.
pub const BUILTIN_NAMESPACE: &str = "io";

/// `IO_FUNCTIONS`: the pipeline I/O functions of the built-in `io` namespace.
pub const IO_FUNCTIONS: &[&str] = &["read", "write", "read3d", "write3d", "render", "render3d"];

/// The built-in namespaces with their descriptions, in registration order.
pub const BUILTIN_NAMESPACES: &[(&str, &str)] = &[
    (
        "io",
        "Pipeline I/O functions (built-in, no search required)",
    ),
    (
        "classicNoisedeck",
        "Complex shaders ported from the original noisedeck.app pipeline",
    ),
    ("synth", "Generator effects"),
    ("mixer", "Blend two sources from A to B"),
    ("filter", "Apply special effects to 2D input"),
    ("render", "Rendering utilities and feedback loops"),
    ("points", "Particle and agent-based simulations"),
    ("synth3d", "3D volumetric generators"),
    ("filter3d", "3D volumetric processors"),
    ("user", "User-defined effects"),
];

/// Names reserved for functions or literals that are neither lexer keywords
/// nor [`IO_FUNCTIONS`] (`_RESERVED_FUNCTION_NAMES`): a namespace of one of
/// these names would shadow it in bare-name resolution.
pub const RESERVED_FUNCTION_NAMES: &[&str] = &["from", "osc", "midi", "audio", "null", "undefined"];

/// `_ID_PATTERN` as its message prints it.
pub const NAMESPACE_ID_PATTERN: &str = "/^[a-z][a-zA-Z0-9]*$/";

/// A registered namespace: `{id, description}` (frozen in the reference).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceDescriptor {
    pub id: String,
    pub description: String,
}

impl NamespaceDescriptor {
    /// `{id, description}`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("id", Value::from(self.id.as_str()));
        o.insert("description", Value::from(self.description.as_str()));
        Value::Object(o)
    }
}

/// `isValidTag(tagId)`.
pub fn is_valid_tag(tag_id: &str) -> bool {
    VALID_TAGS.contains(&tag_id)
}

/// The tag definition of `tag_id`, if it is a tag.
pub fn tag_definition(tag_id: &str) -> Option<&'static TagDefinition> {
    TAG_DEFINITIONS.iter().find(|t| t.id == tag_id)
}

/// `TAG_DEFINITIONS` as the reference's object (`{id: {id, description}}`).
pub fn tag_definitions_value() -> Value {
    let mut o = Object::new();
    for t in TAG_DEFINITIONS {
        o.insert(t.id, t.to_value());
    }
    Value::Object(o)
}

/// `getTagDefinition(tagId)`: `TAG_DEFINITIONS[tagId] || null`. A property
/// read, so the names of `Object.prototype` members (`constructor`,
/// `toString`, ...) return those inherited members, as in the reference.
pub fn get_tag_definition(tag_id: &str) -> Value {
    let Value::Object(table) = tag_definitions_value() else {
        unreachable!()
    };
    let found = object_member(&table, tag_id);
    if found.is_truthy() {
        found
    } else {
        Value::Null
    }
}

/// The result of `validateTags`.
#[derive(Debug, Clone, PartialEq)]
pub struct TagValidation {
    pub valid: bool,
    /// The entries that are not tags, as given.
    pub invalid_tags: Vec<Value>,
}

impl TagValidation {
    /// `{valid, invalidTags}`.
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("valid", Value::Bool(self.valid));
        o.insert("invalidTags", Value::Array(self.invalid_tags.clone()));
        Value::Object(o)
    }
}

/// `validateTags(tags)`: a non-array is invalid with no invalid entries.
pub fn validate_tags(tags: &Value) -> TagValidation {
    let Value::Array(tags) = tags else {
        return TagValidation {
            valid: false,
            invalid_tags: Vec::new(),
        };
    };
    let invalid_tags: Vec<Value> = tags
        .iter()
        .filter(|t| !t.as_str().is_some_and(is_valid_tag))
        .cloned()
        .collect();
    TagValidation {
        valid: invalid_tags.is_empty(),
        invalid_tags,
    }
}

/// `isIOFunction(funcName)`.
pub fn is_io_function(func_name: &str) -> bool {
    IO_FUNCTIONS.contains(&func_name)
}

/// `_ID_PATTERN.test(id)`.
fn matches_id_pattern(id: &str) -> bool {
    let mut chars = id.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_alphanumeric())
}

fn is_builtin_namespace(id: &str) -> bool {
    BUILTIN_NAMESPACES.iter().any(|(b, _)| *b == id)
}

impl Registry {
    /// `isValidNamespace(id)`: a registered namespace (built-in or added).
    pub fn is_valid_namespace(&self, id: &str) -> bool {
        self.namespaces.contains_key(id)
    }

    /// `getNamespaceDescription(id)`.
    pub fn get_namespace_description(&self, id: &str) -> Option<NamespaceDescriptor> {
        self.namespaces
            .get(id)
            .map(|description| NamespaceDescriptor {
                id: id.to_owned(),
                description: description.clone(),
            })
    }

    /// `VALID_NAMESPACES`: the registered namespace ids, built-ins first, then
    /// the added ones in registration order.
    pub fn valid_namespaces(&self) -> Vec<&str> {
        self.namespaces.keys().map(String::as_str).collect()
    }

    /// `NAMESPACE_DESCRIPTIONS`: every registered namespace, in
    /// [`Registry::valid_namespaces`] order.
    pub fn namespace_descriptions(&self) -> Vec<NamespaceDescriptor> {
        self.namespaces
            .iter()
            .map(|(id, description)| NamespaceDescriptor {
                id: id.clone(),
                description: description.clone(),
            })
            .collect()
    }

    /// `NAMESPACE_DESCRIPTIONS` as the reference's object view
    /// (`{id: {id, description}}`).
    pub fn namespace_descriptions_value(&self) -> Value {
        let mut o = Object::new();
        for d in self.namespace_descriptions() {
            o.insert(d.id.clone(), d.to_value());
        }
        Value::Object(o)
    }

    /// `registerNamespace(id, {description})`. Once registered, the id is
    /// accepted by the DSL's `search` directive and by
    /// [`Registry::is_valid_namespace`].
    ///
    /// The id must match `/^[a-z][a-zA-Z0-9]*$/` and must not be a DSL
    /// keyword, an I/O function, a reserved function name or a built-in
    /// namespace; the description must be non-empty. Registering an id again
    /// with the same description returns the existing descriptor; with a
    /// different description it fails.
    pub fn register_namespace(
        &mut self,
        id: &str,
        description: &str,
    ) -> Result<NamespaceDescriptor, JsError> {
        let mut descriptor = Object::new();
        descriptor.insert("description", Value::from(description));
        self.register_namespace_value(&Value::from(id), &Value::Object(descriptor))
    }

    /// [`Registry::register_namespace`] over JavaScript values, with the
    /// reference's checks of their types (`id` a non-empty string,
    /// `descriptor` an object whose `description` is a non-empty string) and
    /// its error messages.
    pub fn register_namespace_value(
        &mut self,
        id: &Value,
        descriptor: &Value,
    ) -> Result<NamespaceDescriptor, JsError> {
        let id = match id {
            Value::String(s) if !s.is_empty() => s.as_str(),
            _ => {
                return Err(JsError::error(
                    "Invalid namespace id: must be a non-empty string",
                ));
            }
        };
        if !matches_id_pattern(id) {
            return Err(JsError::error(format!(
                "Invalid namespace id '{id}': must match {NAMESPACE_ID_PATTERN}"
            )));
        }
        if is_reserved_keyword(id) {
            return Err(JsError::error(format!(
                "Cannot register namespace '{id}': reserved DSL keyword"
            )));
        }
        if is_io_function(id) {
            return Err(JsError::error(format!(
                "Cannot register namespace '{id}': reserved IO function name"
            )));
        }
        if RESERVED_FUNCTION_NAMES.contains(&id) {
            return Err(JsError::error(format!(
                "Cannot register namespace '{id}': reserved function name"
            )));
        }
        if is_builtin_namespace(id) {
            return Err(JsError::error(format!(
                "Cannot register namespace '{id}': built-in namespace"
            )));
        }
        // `descriptor === null || typeof descriptor !== 'object'`.
        if !matches!(descriptor, Value::Object(_) | Value::Array(_)) {
            return Err(JsError::error(format!(
                "Invalid descriptor for namespace '{id}': must be an object"
            )));
        }
        let description = match descriptor.get("description") {
            Value::String(s) if !s.is_empty() => s.clone(),
            _ => {
                return Err(JsError::error(format!(
                    "Invalid descriptor for namespace '{id}': 'description' must be a non-empty string"
                )));
            }
        };
        if let Some(existing) = self.namespaces.get(id) {
            if *existing != description {
                return Err(JsError::error(format!(
                    "Cannot re-register namespace '{id}' with a different description"
                )));
            }
            return Ok(NamespaceDescriptor {
                id: id.to_owned(),
                description,
            });
        }
        self.namespaces.insert(id.to_owned(), description.clone());
        Ok(NamespaceDescriptor {
            id: id.to_owned(),
            description,
        })
    }

    /// `unregisterNamespace(id)`: `true` when the namespace was registered.
    /// Built-in namespaces cannot be removed. Effects registered under the
    /// namespace stay in the registry but `search` no longer reaches them.
    pub fn unregister_namespace(&mut self, id: &str) -> Result<bool, JsError> {
        if is_builtin_namespace(id) {
            return Err(JsError::error(format!(
                "Cannot unregister namespace '{id}': built-in namespace"
            )));
        }
        Ok(self.namespaces.shift_remove(id).is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tag_table() {
        assert_eq!(TAG_DEFINITIONS.len(), VALID_TAGS.len());
        for (t, id) in TAG_DEFINITIONS.iter().zip(VALID_TAGS) {
            assert_eq!(t.id, *id);
        }
        assert!(is_valid_tag("3d"));
        assert!(!is_valid_tag("constructor"));
        assert_eq!(
            get_tag_definition("blur").get("description").as_str(),
            Some("Blur and softening")
        );
        assert!(get_tag_definition("nope").is_null());
        assert!(matches!(get_tag_definition("toString"), Value::Function(_)));
    }

    #[test]
    fn namespace_lifecycle() {
        let mut reg = Registry::new();
        assert_eq!(reg.valid_namespaces().len(), 10);
        let d = reg.register_namespace("myFoo", "Foo collection").unwrap();
        assert_eq!(d.description, "Foo collection");
        assert!(reg.is_valid_namespace("myFoo"));
        assert_eq!(reg.valid_namespaces().last(), Some(&"myFoo"));
        assert_eq!(
            reg.register_namespace("myFoo", "Foo collection").unwrap(),
            d
        );
        assert!(
            reg.register_namespace("myFoo", "other")
                .unwrap_err()
                .to_string()
                .contains("different description")
        );
        assert_eq!(reg.unregister_namespace("myFoo"), Ok(true));
        assert_eq!(reg.unregister_namespace("myFoo"), Ok(false));
        assert!(reg.unregister_namespace("synth").is_err());
        for (id, needle) in [
            ("", "non-empty string"),
            ("Foo", "must match /^[a-z][a-zA-Z0-9]*$/"),
            ("search", "reserved DSL keyword"),
            ("read3d", "reserved IO function name"),
            ("osc", "reserved function name"),
            ("user", "built-in namespace"),
        ] {
            let err = reg.register_namespace(id, "x").unwrap_err().to_string();
            assert!(err.contains(needle), "{id}: {err}");
        }
        let err = reg
            .register_namespace_value(&Value::from("okName"), &Value::Null)
            .unwrap_err();
        assert!(err.to_string().contains("must be an object"));
    }
}
