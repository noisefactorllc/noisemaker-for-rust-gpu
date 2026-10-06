//! Port of `runtime/effect-validator.js`: validate an effect definition against
//! the definition grammar the runtime consumes (metadata, globals, lifecycle
//! hooks, passes, bindings, textures, uniform layouts, compile-time defines,
//! conditions, counts, workgroups and the UI control layer).
//!
//! The result is the reference's error list, in its order, with its messages;
//! `[]` for a valid definition. Validation is structural and side-effect free.
//! Like the reference, it can still throw where it interpolates a value whose
//! string conversion throws (an object with a non-callable `toString`), so the
//! entry points return `Result`.
//!
//! JavaScript distinguishes three definition shapes the port models with
//! [`Definition`]: plain objects (and malformed containers), `Effect` instances
//! (whose prototype supplies the lifecycle hooks and whose extra runtime state
//! is not diagnosed), and `Effect` subclass constructors.

use std::sync::OnceLock;

use crate::error::JsError;
use crate::js::{is_finite_number, number_to_string, parse_float};
use crate::registry::Registry;
use crate::unparser::jsv::{entries, keys, member, same_value_zero, strict_equals, to_string};
use crate::value::{Object, Value};

const GLOBAL_TYPES: &[&str] = &[
    "float", "int", "boolean", "vec2", "vec3", "vec4", "mat3", "color", "surface", "volume",
    "geometry", "member", "palette", "button", "string",
];

const UI_CONTROLS: &[&str] = &[
    "slider", "checkbox", "dropdown", "color", "button", "vector3", "vec3",
];

const UI_KEYS: &[&str] = &[
    "label",
    "control",
    "category",
    "hidden",
    "hint",
    "format",
    "buttonLabel",
    "enabledBy",
    "multiline",
    "resetOnChange",
];

const ENABLED_BY_OPS: &[&str] = &["eq", "neq", "lt", "gt", "gte", "lte", "in", "notIn"];

const GLOBAL_SPEC_KEYS: &[&str] = &[
    "type",
    "default",
    "uniform",
    "define",
    "choices",
    "enum",
    "min",
    "max",
    "step",
    "zero",
    "randMin",
    "randMax",
    "randChance",
    "randChoices",
    "colorModeUniform",
    "ui",
];

const PASS_KEYS: &[&str] = &[
    "name",
    "program",
    "type",
    "entryPoint",
    "drawMode",
    "drawBuffers",
    "count",
    "countUniform",
    "repeat",
    "blend",
    "workgroups",
    "storageBuffers",
    "storageTextures",
    "viewport",
    "conditions",
    "defines",
    "uniforms",
    "inputs",
    "outputs",
];

const TEXTURE_SPEC_KEYS: &[&str] = &[
    "width",
    "height",
    "depth",
    "format",
    "is3D",
    "filter",
    "mipmaps",
    "persistent",
];

/// Filtering policies are authorable on 3D textures only.
const TEXTURE_FILTERS: &[&str] = &["nearest", "linear"];

const CONDITION_CONTAINER_KEYS: &[&str] = &["runIf", "skipIf"];

const DIM_KEYWORDS: &[&str] = &["screen", "auto", "input", "resolution"];

const FORMATS: &[&str] = &[
    "rgba16f",
    "rgba16float",
    "rgba8",
    "rgba8unorm",
    "rgba32f",
    "rgba32float",
];

const DRAW_MODES: &[&str] = &["points", "triangles", "billboards"];

const PASS_TYPES: &[&str] = &["render", "compute"];

const LAYOUT_ENTRY_KEYS: &[&str] = &["name", "slot", "components"];

const BYTE_LAYOUT_KEYS: &[&str] = &["name", "offset", "size", "type"];

const DIM_SPEC_KEYS: &[&str] = &[
    "param",
    "power",
    "multiply",
    "default",
    "paramDefault",
    "screenDivide",
    "scale",
    "clamp",
    "inputOverride",
];

const TOP_LEVEL_KEYS: &[&str] = &[
    "name",
    "namespace",
    "func",
    "description",
    "tags",
    "globals",
    "passes",
    "textures",
    "textures3d",
    "shaders",
    "uniformLayout",
    "uniformLayouts",
    "paramAliases",
    "openCategories",
    "defaultProgram",
    "hidden",
    "deprecatedBy",
    "externalTexture",
    "externalMesh",
    "builtinMeshes",
    "outputTex3d",
    "outputGeo",
    "state",
    "uniforms",
    "onInit",
    "onUpdate",
    "onDestroy",
    "asyncInit",
];

const PIPELINE_INPUTS: &[&str] = &[
    "inputTex",
    "inputTex3d",
    "inputGeo",
    "inputXyz",
    "inputVel",
    "inputRgba",
    "noise",
    "midiNoteGrid",
    "feedback",
    "selfTex",
    "outputTex",
    "none",
];

const PIPELINE_OUTPUTS: &[&str] = &[
    "outputTex",
    "outputTex3d",
    "outputXyz",
    "outputVel",
    "outputRgba",
];

/// `VALID_TAGS` (`runtime/tags.js`, `TAG_DEFINITIONS` key order).
pub use crate::tags::VALID_TAGS;

/// The lifecycle hooks `Effect.prototype` defines.
const EFFECT_PROTOTYPE_METHODS: &[&str] = &[
    "constructor",
    "onInit",
    "onUpdate",
    "onDestroy",
    "asyncInit",
];

/// The shape of the value handed to `validateEffectDefinition`.
#[derive(Debug, Clone, Copy)]
pub enum Definition<'a> {
    /// Any JavaScript value: a plain definition object (a JSON or Portable
    /// package), or a malformed container (`null`, an array, a primitive, a
    /// plain function).
    Plain(&'a Value),
    /// An `Effect` instance: its own properties, plus the methods of its
    /// subclass prototype (if any). `Effect.prototype` supplies `onInit`,
    /// `onUpdate`, `onDestroy` and `asyncInit`.
    Instance {
        own: &'a Value,
        prototype_methods: &'a [String],
    },
    /// An `Effect` subclass constructor: its prototype's own method names, and
    /// the class's own properties in `Object.getOwnPropertyNames` order
    /// (`length`, `name`, `prototype`, then static members).
    Subclass {
        prototype_methods: &'a [String],
        statics: &'a Object,
    },
}

/// `isObj(value)`: a non-null, non-array object.
fn is_obj(v: &Value) -> bool {
    matches!(v, Value::Object(_))
}

/// `typeof v === 'string' && v` (a non-empty string).
fn non_empty_string(v: &Value) -> bool {
    matches!(v, Value::String(s) if !s.is_empty())
}

/// `list.includes(v)` over a list of strings.
fn includes(list: &[&str], v: &Value) -> bool {
    matches!(v, Value::String(s) if list.contains(&s.as_str()))
}

/// `Number.isInteger(v)`.
fn is_integer(v: &Value) -> bool {
    v.is_integer()
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

/// `value[key]` for a member of a definition container.
fn at(v: &Value, key: &str) -> Value {
    member(v, key)
}

/// `stdEnums` (`lang/std_enums.js`) over the catalog's palette table.
fn std_enums() -> &'static Value {
    static STD: OnceLock<Value> = OnceLock::new();
    STD.get_or_init(|| {
        let mut reg = Registry::new();
        let palettes = noisemaker_effects::share_file("share/palettes.json")
            .and_then(|bytes| std::str::from_utf8(bytes).ok())
            .and_then(|text| Value::from_json(text).ok())
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default();
        reg.palettes = palettes;
        Value::Object(reg.std_enums())
    })
}

/// What `resolveStdEnum` found.
#[derive(Debug, PartialEq)]
enum StdEnum {
    Leaf,
    Table,
}

/// `resolveStdEnum(pathStr)`: a std enum table, a leaf entry, or nothing.
///
/// `node[part]` sees inherited members too (`__proto__` resolves to
/// `Object.prototype`, a table); own members are walked by reference.
fn resolve_std_enum(path: &Value) -> Option<StdEnum> {
    let Value::String(path) = path else {
        return None;
    };
    if path.is_empty() {
        return None;
    }
    let parts: Vec<&str> = path.split('.').collect();
    let walkable =
        |node: &Value| node.is_truthy() && matches!(node, Value::Object(_) | Value::Array(_));
    let mut node: &Value = std_enums();
    let mut i = 0;
    while i < parts.len() {
        if !walkable(node) {
            return None;
        }
        let own = match node {
            Value::Object(o) => o.get(parts[i]),
            Value::Array(a) if crate::value::is_array_index(parts[i]) => {
                parts[i].parse::<usize>().ok().and_then(|n| a.get(n))
            }
            _ => None,
        };
        match own {
            Some(next) => {
                node = next;
                i += 1;
            }
            None => break,
        }
    }
    if i == parts.len() {
        return classify_std_enum(node);
    }
    // An inherited member: continue on owned values.
    let mut owned = member(node, parts[i]);
    if owned.is_undefined() {
        return None;
    }
    for part in &parts[i + 1..] {
        if !walkable(&owned) {
            return None;
        }
        let next = member(&owned, part);
        if next.is_undefined() {
            return None;
        }
        owned = next;
    }
    classify_std_enum(&owned)
}

fn classify_std_enum(node: &Value) -> Option<StdEnum> {
    if node.is_truthy() && matches!(node, Value::Object(_) | Value::Array(_)) {
        if !member(node, "value").is_undefined() {
            return Some(StdEnum::Leaf);
        }
        return Some(StdEnum::Table);
    }
    None
}

/// The validation context: declared global keys and their uniform names.
struct Context {
    global_keys: Vec<String>,
    global_uniform_names: Vec<String>,
}

impl Context {
    fn has_global(&self, name: &Value) -> bool {
        matches!(name, Value::String(s) if self.global_keys.iter().any(|k| k == s))
    }

    /// `referencesGlobal(name, context)`.
    fn references_global(&self, name: &Value) -> bool {
        self.has_global(name)
            || matches!(name, Value::String(s) if self.global_uniform_names.iter().any(|k| k == s))
    }
}

type Errors = Vec<String>;

/// `validateDimSpec(spec, errors, label)`: a dimension expression consumed by
/// `pipeline.resolveDimension()`.
fn validate_dim_spec(spec: &Value, errors: &mut Errors, label: &str) {
    match spec {
        Value::Number(n) => {
            if !n.is_finite() || *n <= 0.0 {
                errors.push(format!(
                    "{label}: dimension must be a positive finite number, keyword, percentage, or dimension expression"
                ));
            }
        }
        Value::String(s) => {
            if DIM_KEYWORDS.contains(&s.as_str()) {
                return;
            }
            if let Some(body) = s.strip_suffix('%')
                && !body.is_empty()
                && body.chars().all(|c| c.is_ascii_digit() || c == '.')
            {
                let percent = parse_float(s);
                if !percent.is_finite() || percent <= 0.0 {
                    errors.push(format!("{label}: invalid percentage '{s}'"));
                }
                return;
            }
            errors.push(format!("{label}: invalid dimension '{s}'"));
        }
        Value::Object(_) => {
            for key in keys(spec) {
                if !DIM_SPEC_KEYS.contains(&key.as_str()) {
                    errors.push(format!("{label}: unknown dimension field '{key}'"));
                }
            }
            let finite_or_absent = |field: &str, errors: &mut Errors| {
                let v = at(spec, field);
                if !v.is_undefined() && !is_finite_number(&v) {
                    errors.push(format!("{label}: \"{field}\" must be a finite number"));
                }
            };
            if !at(spec, "param").is_undefined() {
                if !non_empty_string(&at(spec, "param")) {
                    errors.push(format!("{label}: \"param\" must be a non-empty string"));
                }
                finite_or_absent("power", errors);
                finite_or_absent("multiply", errors);
                finite_or_absent("default", errors);
                finite_or_absent("paramDefault", errors);
                let input_override = at(spec, "inputOverride");
                if !input_override.is_undefined() && !non_empty_string(&input_override) {
                    errors.push(format!(
                        "{label}: \"inputOverride\" must be a non-empty string"
                    ));
                }
                return;
            }
            if !at(spec, "screenDivide").is_undefined() {
                if !non_empty_string(&at(spec, "screenDivide")) {
                    errors.push(format!(
                        "{label}: \"screenDivide\" must be a non-empty string"
                    ));
                }
                finite_or_absent("default", errors);
                return;
            }
            if !at(spec, "scale").is_undefined() {
                if !is_finite_number(&at(spec, "scale")) {
                    errors.push(format!("{label}: \"scale\" must be a finite number"));
                }
                let clamp = at(spec, "clamp");
                if !clamp.is_undefined() {
                    if !is_obj(&clamp) {
                        errors.push(format!("{label}: \"clamp\" must be an object"));
                    } else {
                        for field in ["min", "max"] {
                            let v = at(&clamp, field);
                            if !v.is_undefined() && !is_finite_number(&v) {
                                errors.push(format!(
                                    "{label}: \"clamp.{field}\" must be a finite number"
                                ));
                            }
                        }
                        for key in keys(&clamp) {
                            if key != "min" && key != "max" {
                                errors.push(format!("{label}: unknown clamp field '{key}'"));
                            }
                        }
                    }
                }
                return;
            }
            errors.push(format!(
                "{label}: dimension object must reference \"param\", \"screenDivide\", or \"scale\""
            ));
        }
        _ => errors.push(format!("{label}: invalid dimension specification")),
    }
}

/// A layout entry candidate for conflict checks.
struct LayoutClaim {
    name: Value,
    slot: Value,
    components: Value,
}

/// `validateUniformLayout(layout, errors, label)`: named-key map, array of
/// entries, or the byte form; duplicate or overlapping claims are conflicts.
fn validate_uniform_layout(
    layout: &Value,
    errors: &mut Errors,
    label: &str,
) -> Result<(), JsError> {
    if let Value::Array(items) = layout {
        let mut claims = Vec::new();
        for (i, item) in items.iter().enumerate() {
            if is_obj(item) {
                claims.push(LayoutClaim {
                    name: at(item, "name"),
                    slot: at(item, "slot"),
                    components: at(item, "components"),
                });
            }
            validate_layout_entry(item, errors, &format!("{label}[{i}]"));
        }
        return check_layout_conflicts(&claims, errors, label);
    }
    if !is_obj(layout) {
        errors.push(format!("{label}: must be an object or array layout"));
        return Ok(());
    }
    if strict_equals(&at(layout, "type"), &"byte".into()) {
        let Value::Array(byte_layout) = at(layout, "layout") else {
            errors.push(format!("{label}: byte layout requires a \"layout\" array"));
            return Ok(());
        };
        for key in keys(layout) {
            if key != "type" && key != "layout" {
                errors.push(format!("{label}: unknown byte-layout field '{key}'"));
            }
        }
        let mut byte_entries: Vec<&Value> = Vec::new();
        for (i, entry) in byte_layout.iter().enumerate() {
            let entry_label = format!("{label}.layout[{i}]");
            if !is_obj(entry) {
                errors.push(format!("{entry_label}: entry must be an object"));
                continue;
            }
            for key in keys(entry) {
                if !BYTE_LAYOUT_KEYS.contains(&key.as_str()) && key != "components" {
                    errors.push(format!("{entry_label}: unknown field '{key}'"));
                }
            }
            let name = at(entry, "name");
            let offset = at(entry, "offset");
            let size = at(entry, "size");
            if !non_empty_string(&name) {
                errors.push(format!("{entry_label}: missing \"name\" string"));
            }
            if !is_integer(&offset) || num(&offset) < 0.0 {
                errors.push(format!(
                    "{entry_label}: \"offset\" must be a non-negative integer"
                ));
            }
            if !is_integer(&size) || num(&size) <= 0.0 {
                errors.push(format!(
                    "{entry_label}: \"size\" must be a positive integer"
                ));
            }
            if !non_empty_string(&at(entry, "type")) {
                errors.push(format!("{entry_label}: missing \"type\" string"));
            }
            if non_empty_string(&name)
                && is_integer(&offset)
                && num(&offset) >= 0.0
                && is_integer(&size)
                && num(&size) > 0.0
            {
                byte_entries.push(entry);
            }
        }
        check_byte_layout_conflicts(&byte_entries, errors, label);
        return Ok(());
    }
    let mut claims = Vec::new();
    for (name, spec) in entries(layout) {
        let entry_label = format!("{label}['{name}']");
        if !is_obj(&spec) {
            errors.push(format!("{entry_label}: layout entry must be an object"));
            continue;
        }
        // `{ name, ...spec }`: `name` keeps the first position; a spec `name`
        // member overrides its value.
        let mut entry = Object::new();
        entry.insert("name", Value::from(name.as_str()));
        crate::unparser::jsv::spread_into(&mut entry, &spec);
        validate_layout_entry(&Value::Object(entry), errors, &entry_label);
        let slot = at(&spec, "slot");
        if is_integer(&slot) {
            claims.push(LayoutClaim {
                name: Value::from(name.as_str()),
                slot,
                components: at(&spec, "components"),
            });
        }
    }
    check_layout_conflicts(&claims, errors, label)
}

/// Semantic component order of the backends' uniform packing.
fn component_order(c: char) -> u32 {
    match c {
        'x' => 0,
        'y' => 1,
        'z' => 2,
        _ => 3,
    }
}

/// `validateLayoutEntry(entry, errors, label)`.
fn validate_layout_entry(entry: &Value, errors: &mut Errors, label: &str) {
    if !is_obj(entry) {
        errors.push(format!("{label}: layout entry must be an object"));
        return;
    }
    for key in keys(entry) {
        if !LAYOUT_ENTRY_KEYS.contains(&key.as_str()) {
            errors.push(format!("{label}: unknown field '{key}'"));
        }
    }
    if !non_empty_string(&at(entry, "name")) {
        errors.push(format!("{label}: missing \"name\" string"));
    }
    let slot = at(entry, "slot");
    if !is_integer(&slot) || num(&slot) < 0.0 {
        errors.push(format!("{label}: \"slot\" must be a non-negative integer"));
    }
    let components = match at(entry, "components") {
        Value::String(c)
            if (1..=4).contains(&c.chars().count())
                && c.chars().all(|ch| matches!(ch, 'x' | 'y' | 'z' | 'w')) =>
        {
            c
        }
        _ => {
            errors.push(format!(
                "{label}: \"components\" must be 1-4 characters from xyzw"
            ));
            return;
        }
    };
    let chars: Vec<char> = components.chars().collect();
    for i in 1..chars.len() {
        if component_order(chars[i]) <= component_order(chars[i - 1]) {
            errors.push(format!(
                "{label}: \"components\" '{components}' must be in ascending xyzw order"
            ));
            break;
        }
    }
}

/// `checkByteLayoutConflicts(entries, errors, label)`.
fn check_byte_layout_conflicts(entries: &[&Value], errors: &mut Errors, label: &str) {
    let n = |v: &Value, k: &str| num(&at(v, k));
    let s = |v: &Value, k: &str| number_to_string(num(&at(v, k)));
    for (i, a) in entries.iter().enumerate() {
        for b in &entries[i + 1..] {
            let a_name = at(a, "name");
            let b_name = at(b, "name");
            let a_name_text = a_name.as_str().unwrap_or_default();
            if strict_equals(&a_name, &b_name) {
                errors.push(format!(
                    "{label}: duplicate byte-layout entries '{a_name_text}' (offsets {} and {})",
                    s(a, "offset"),
                    s(b, "offset")
                ));
                continue;
            }
            let a_start = n(a, "offset");
            let a_end = a_start + n(a, "size");
            let b_start = n(b, "offset");
            let b_end = b_start + n(b, "size");
            if a_start < b_end && b_start < a_end {
                errors.push(format!(
                    "{label}: byte layout conflict: '{a_name_text}' (offset {}, size {}) overlaps '{}' (offset {}, size {})",
                    s(a, "offset"),
                    s(a, "size"),
                    b_name.as_str().unwrap_or_default(),
                    s(b, "offset"),
                    s(b, "size")
                ));
            }
        }
    }
}

/// `checkLayoutConflicts(entries, errors, label)`.
fn check_layout_conflicts(
    claims: &[LayoutClaim],
    errors: &mut Errors,
    label: &str,
) -> Result<(), JsError> {
    let usable = |c: &LayoutClaim| is_integer(&c.slot) && matches!(c.components, Value::String(_));
    for (i, a) in claims.iter().enumerate() {
        if !usable(a) {
            continue;
        }
        for b in &claims[i + 1..] {
            if !usable(b) || !strict_equals(&a.slot, &b.slot) {
                continue;
            }
            let (Value::String(ac), Value::String(bc)) = (&a.components, &b.components) else {
                continue;
            };
            let overlap = ac.chars().any(|c| bc.contains(c));
            if ac == bc {
                errors.push(format!(
                    "{label}: duplicate layout entries '{}' and '{}' claim slot {} components '{ac}'",
                    to_string(&a.name)?,
                    to_string(&b.name)?,
                    number_to_string(num(&a.slot))
                ));
            } else if overlap {
                errors.push(format!(
                    "{label}: layout conflict at slot {}: '{}' ({ac}) overlaps '{}' ({bc})",
                    number_to_string(num(&a.slot)),
                    to_string(&a.name)?,
                    to_string(&b.name)?
                ));
            }
        }
    }
    Ok(())
}

/// `validateEnabledBy(cond, errors, label, context)`: a global name, a
/// `{param, op}` condition, or `{and: [...]}` / `{or: [...]}` groups.
fn validate_enabled_by(cond: &Value, errors: &mut Errors, label: &str, context: &Context) {
    if let Value::String(name) = cond {
        if !context.has_global(cond) {
            errors.push(format!(
                "{label}: enabledBy references unknown global '{name}'"
            ));
        }
        return;
    }
    if !is_obj(cond) {
        errors.push(format!(
            "{label}: \"enabledBy\" must be a global name or condition object"
        ));
        return;
    }
    if !at(cond, "and").is_undefined() || !at(cond, "or").is_undefined() {
        for key in keys(cond) {
            if key != "and" && key != "or" {
                errors.push(format!("{label}: unknown enabledBy field '{key}'"));
            }
        }
        for branch in ["and", "or"] {
            let list = at(cond, branch);
            if list.is_undefined() {
                continue;
            }
            match list {
                Value::Array(items) => {
                    for sub in &items {
                        validate_enabled_by(sub, errors, label, context);
                    }
                }
                _ => errors.push(format!("{label}: \"enabledBy.{branch}\" must be an array")),
            }
        }
        return;
    }
    for key in keys(cond) {
        if key != "param" && !ENABLED_BY_OPS.contains(&key.as_str()) {
            errors.push(format!("{label}: unknown enabledBy field '{key}'"));
        }
    }
    let param = at(cond, "param");
    let Value::String(param_name) = &param else {
        errors.push(format!(
            "{label}: \"enabledBy\" requires a \"param\" string"
        ));
        return;
    };
    if param_name.is_empty() {
        errors.push(format!(
            "{label}: \"enabledBy\" requires a \"param\" string"
        ));
        return;
    }
    if !context.has_global(&param) {
        errors.push(format!(
            "{label}: enabledBy references unknown global '{param_name}'"
        ));
    }
    if !ENABLED_BY_OPS.iter().any(|op| !at(cond, op).is_undefined()) {
        errors.push(format!(
            "{label}: \"enabledBy\" requires one of eq/neq/lt/gt/in/notIn"
        ));
    }
    for (field, msg) in [("in", "enabledBy.in"), ("notIn", "enabledBy.notIn")] {
        let v = at(cond, field);
        if !v.is_undefined() && !matches!(v, Value::Array(_)) {
            errors.push(format!("{label}: \"{msg}\" must be an array"));
        }
    }
}

/// `validateUi(ui, errors, label, context)`.
fn validate_ui(
    ui: &Value,
    errors: &mut Errors,
    label: &str,
    context: &Context,
) -> Result<(), JsError> {
    if !is_obj(ui) {
        errors.push(format!("{label}: must be an object"));
        return Ok(());
    }
    for key in keys(ui) {
        if !UI_KEYS.contains(&key.as_str()) {
            errors.push(format!("{label}: unknown field '{key}'"));
        }
    }
    let ui_label = at(ui, "label");
    if !ui_label.is_undefined() && !non_empty_string(&ui_label) {
        errors.push(format!("{label}: \"label\" must be a non-empty string"));
    }
    let control = at(ui, "control");
    if !control.is_undefined()
        && !strict_equals(&control, &false.into())
        && !includes(UI_CONTROLS, &control)
    {
        errors.push(format!(
            "{label}: unknown control '{}'",
            to_string(&control)?
        ));
    }
    let category = at(ui, "category");
    if !category.is_undefined() && !non_empty_string(&category) {
        errors.push(format!("{label}: \"category\" must be a non-empty string"));
    }
    for field in ["hidden", "multiline", "resetOnChange"] {
        let v = at(ui, field);
        if !v.is_undefined() && !matches!(v, Value::Bool(_)) {
            errors.push(format!("{label}: \"{field}\" must be a boolean"));
        }
    }
    for key in ["hint", "format", "buttonLabel"] {
        let v = at(ui, key);
        if !v.is_undefined() && !non_empty_string(&v) {
            errors.push(format!("{label}: \"{key}\" must be a non-empty string"));
        }
    }
    let enabled_by = at(ui, "enabledBy");
    if !enabled_by.is_undefined() {
        validate_enabled_by(&enabled_by, errors, label, context);
    }
    Ok(())
}

/// `typeof spec.type === 'string' ? spec.type : null`.
fn spec_type(spec: &Value) -> Option<String> {
    at(spec, "type").as_str().map(str::to_owned)
}

/// `/^#[0-9a-fA-F]{6}$/`.
fn is_hex_color(s: &str) -> bool {
    s.len() == 7 && s.starts_with('#') && s[1..].bytes().all(|b| b.is_ascii_hexdigit())
}

/// `validateDefault(spec, errors, label)`.
fn validate_default(spec: &Value, errors: &mut Errors, label: &str) {
    let ty = spec_type(spec);
    let value = at(spec, "default");
    match ty.as_deref() {
        Some("float" | "palette" | "button") => {
            if !is_finite_number(&value) {
                errors.push(format!("{label}: \"default\" must be a finite number"));
            }
        }
        Some("int") => {
            if !is_finite_number(&value) || !is_integer(&value) {
                errors.push(format!("{label}: \"default\" must be a finite integer"));
            }
        }
        Some("boolean") => {
            if !matches!(value, Value::Bool(_)) {
                errors.push(format!("{label}: \"default\" must be a boolean"));
            }
        }
        Some(t @ ("vec2" | "vec3" | "vec4" | "mat3")) => {
            let dims = match t {
                "vec2" => 2,
                "vec3" => 3,
                "vec4" => 4,
                _ => 9,
            };
            let ok = matches!(&value, Value::Array(a) if a.len() == dims && a.iter().all(is_finite_number));
            if !ok {
                errors.push(format!(
                    "{label}: \"default\" must be an array of {dims} finite numbers"
                ));
            }
        }
        Some("color") => match &value {
            Value::Array(a) => {
                if a.len() != 3 || !a.iter().all(is_finite_number) {
                    errors.push(format!(
                        "{label}: \"default\" must be a 3-component color array"
                    ));
                }
            }
            Value::String(s) if is_hex_color(s) => {}
            _ => errors.push(format!(
                "{label}: \"default\" must be a 3-component color array or '#rrggbb' string"
            )),
        },
        Some("string") => {
            if !matches!(value, Value::String(_)) {
                errors.push(format!("{label}: \"default\" must be a string"));
            }
        }
        Some(t @ ("surface" | "volume" | "geometry" | "member")) => match &value {
            Value::String(s) => {
                if t == "member" && resolve_std_enum(&value) != Some(StdEnum::Leaf) {
                    errors.push(format!(
                        "{label}: \"default\" '{s}' does not resolve to a std enum value"
                    ));
                }
            }
            _ => errors.push(format!("{label}: \"default\" must be a string")),
        },
        // Unknown type already reported separately.
        _ => {}
    }
}

/// `rangeDims(type)`: the component count of a min/max/default array.
fn range_dims(ty: Option<&str>) -> usize {
    match ty {
        Some("vec2") => 2,
        Some("vec3" | "color") => 3,
        Some("vec4") => 4,
        Some("mat3") => 9,
        _ => 1,
    }
}

/// `validateRangeBounds(spec, errors, label)`: bound shapes, ordering and default
/// containment.
fn validate_range_bounds(spec: &Value, errors: &mut Errors, label: &str) {
    let ty = spec_type(spec);
    let dims = range_dims(ty.as_deref());

    let mut min: Option<Vec<f64>> = None;
    let mut max: Option<Vec<f64>> = None;
    for field in ["min", "max"] {
        let value = at(spec, field);
        if value.is_undefined() {
            continue;
        }
        let bound = match &value {
            Value::Number(n) if n.is_finite() => Some(vec![*n]),
            Value::Array(a) if a.len() == dims && a.iter().all(is_finite_number) => {
                Some(a.iter().map(num).collect())
            }
            Value::Array(_) => {
                errors.push(format!(
                    "{label}: \"{field}\" must be an array of {dims} finite numbers for type '{}'",
                    ty.as_deref().unwrap_or("unknown")
                ));
                None
            }
            _ => {
                errors.push(format!(
                    "{label}: \"{field}\" must be a finite number or an array of {dims} finite numbers"
                ));
                None
            }
        };
        if bound.is_some() {
            if field == "min" {
                min = bound;
            } else {
                max = bound;
            }
        }
    }

    if let (Some(lo), Some(hi)) = (&min, &max) {
        let same_form = (lo.len() == 1) == (hi.len() == 1);
        if !same_form {
            errors.push(format!(
                "{label}: \"min\" and \"max\" must both be scalars or both be arrays"
            ));
        } else {
            // `min[i] > max[i]` (an index past `max` compares with undefined: false).
            if (0..lo.len()).any(|i| hi.get(i).is_some_and(|h| lo[i] > *h)) {
                errors.push(format!("{label}: \"min\" must not exceed \"max\""));
            }
        }
    }

    // Default containment: broadcast scalar bounds, compare componentwise.
    let dflt = at(spec, "default");
    match (&dflt, &min, &max) {
        (Value::Array(d), Some(lo), Some(hi)) if d.iter().all(is_finite_number) => {
            for (i, v) in d.iter().enumerate() {
                let v = num(v);
                let lo_i = if lo.len() == 1 { lo.first() } else { lo.get(i) };
                let hi_i = if hi.len() == 1 { hi.first() } else { hi.get(i) };
                let below = lo_i.is_some_and(|l| v < *l);
                let above = hi_i.is_some_and(|h| v > *h);
                if below || above {
                    let joined: Vec<String> = d.iter().map(|x| number_to_string(num(x))).collect();
                    errors.push(format!(
                        "{label}: default [{}] is outside the declared range",
                        joined.join(", ")
                    ));
                    break;
                }
            }
        }
        (Value::Number(d), Some(lo), Some(hi))
            if d.is_finite() && lo.len() == 1 && hi.len() == 1 && (*d < lo[0] || *d > hi[0]) =>
        {
            errors.push(format!(
                "{label}: default {} is outside the declared range [{}, {}]",
                number_to_string(*d),
                number_to_string(lo[0]),
                number_to_string(hi[0])
            ));
        }
        _ => {}
    }
}

/// `validateGlobals(globals, errors, context)`.
fn validate_globals(
    globals: &Value,
    errors: &mut Errors,
    context: &Context,
) -> Result<(), JsError> {
    if globals.is_nullish() {
        return Ok(());
    }
    if !is_obj(globals) {
        errors.push("\"globals\" must be an object".into());
        return Ok(());
    }

    let mut uniform_owners: Vec<(String, String)> = Vec::new();
    for (key, spec) in entries(globals) {
        let label = format!("Global '{key}'");
        if !is_obj(&spec) {
            errors.push(format!("{label}: must be an object"));
            continue;
        }

        for field in keys(&spec) {
            if !GLOBAL_SPEC_KEYS.contains(&field.as_str()) {
                errors.push(format!("{label}: unknown field '{field}'"));
            }
        }

        let ty = at(&spec, "type");
        if !ty.is_truthy() {
            errors.push(format!("{label}: Missing \"type\""));
        } else if !includes(GLOBAL_TYPES, &ty) {
            errors.push(format!("{label}: Unknown type '{}'", to_string(&ty)?));
        }

        if !at(&spec, "default").is_undefined() {
            validate_default(&spec, errors, &label);
        }

        if !at(&spec, "min").is_undefined() || !at(&spec, "max").is_undefined() {
            validate_range_bounds(&spec, errors, &label);
        }

        for field in ["step", "zero", "randMin", "randMax", "randChance"] {
            let v = at(&spec, field);
            if !v.is_undefined() && !is_finite_number(&v) {
                errors.push(format!("{label}: \"{field}\" must be a finite number"));
            }
        }
        let rand_choices = at(&spec, "randChoices");
        if !rand_choices.is_undefined()
            && !matches!(&rand_choices, Value::Array(a) if a.iter().all(is_finite_number))
        {
            errors.push(format!(
                "{label}: \"randChoices\" must be an array of finite numbers"
            ));
        }

        let uniform = at(&spec, "uniform");
        if !uniform.is_undefined() && !non_empty_string(&uniform) {
            errors.push(format!("{label}: \"uniform\" must be a non-empty string"));
        } else if let Value::String(u) = &uniform
            && !u.is_empty()
        {
            let owner = uniform_owners
                .iter()
                .find(|(name, _)| name == u)
                .map(|(_, owner)| owner.clone());
            match owner {
                Some(owner) if !owner.is_empty() => errors.push(format!(
                    "{label}: uniform '{u}' conflicts with global '{owner}'"
                )),
                _ => match uniform_owners.iter_mut().find(|(name, _)| name == u) {
                    Some(slot) => slot.1 = key.clone(),
                    None => uniform_owners.push((u.clone(), key.clone())),
                },
            }
        }

        let define = at(&spec, "define");
        if !define.is_undefined() && !non_empty_string(&define) {
            errors.push(format!("{label}: \"define\" must be a non-empty string"));
        }

        let color_mode_uniform = at(&spec, "colorModeUniform");
        if !color_mode_uniform.is_undefined() && !non_empty_string(&color_mode_uniform) {
            errors.push(format!(
                "{label}: \"colorModeUniform\" must be a non-empty string"
            ));
        }

        let choices = at(&spec, "choices");
        if !choices.is_undefined() {
            if !is_obj(&choices) {
                errors.push(format!(
                    "{label}: \"choices\" must be an object mapping names to values"
                ));
            } else {
                let mut numeric: Vec<Value> = Vec::new();
                let string_type = strict_equals(&ty, &"string".into());
                for (choice_name, value) in entries(&choices) {
                    if value.is_null() {
                        continue; // Section headers in dropdown menus.
                    }
                    if string_type {
                        // String-typed globals carry string-valued choices.
                        if !matches!(value, Value::String(_)) {
                            errors.push(format!(
                                "{label}: choices['{choice_name}'] must be a string for type 'string'"
                            ));
                        }
                        continue;
                    }
                    if !is_finite_number(&value) {
                        errors.push(format!(
                            "{label}: choices['{choice_name}'] must be a number or null"
                        ));
                    } else {
                        numeric.push(value);
                    }
                }
                let default = at(&spec, "default");
                if !numeric.is_empty()
                    && is_finite_number(&default)
                    && !numeric.iter().any(|n| same_value_zero(n, &default))
                {
                    errors.push(format!(
                        "{label}: default {} is not among the declared choice values",
                        number_to_string(num(&default))
                    ));
                }
            }
        }

        let enum_spec = at(&spec, "enum");
        if !enum_spec.is_undefined() {
            if !non_empty_string(&enum_spec) {
                errors.push(format!("{label}: \"enum\" must be a non-empty string"));
            } else if resolve_std_enum(&enum_spec) != Some(StdEnum::Table) {
                errors.push(format!(
                    "{label}: enum '{}' does not resolve to a std enum table",
                    enum_spec.as_str().unwrap_or_default()
                ));
            }
        }

        let ui = at(&spec, "ui");
        if !ui.is_undefined() {
            validate_ui(&ui, errors, &format!("{label}.ui"), context)?;
        }
    }
    Ok(())
}

/// `validateTextureMap(textures, errors, containerName)`.
fn validate_texture_map(
    textures: &Value,
    errors: &mut Errors,
    container_name: &str,
) -> Result<(), JsError> {
    if textures.is_undefined() {
        return Ok(());
    }
    if !is_obj(textures) {
        errors.push(format!("\"{container_name}\" must be an object"));
        return Ok(());
    }
    let is_3d = container_name == "textures3d";
    for (name, spec) in entries(textures) {
        let label = format!("Texture '{name}'");
        if !is_obj(&spec) {
            errors.push(format!("{label}: must be an object"));
            continue;
        }
        for key in keys(&spec) {
            if !TEXTURE_SPEC_KEYS.contains(&key.as_str()) {
                errors.push(format!("{label}: unknown field '{key}'"));
            }
        }
        for dim in ["width", "height"] {
            let v = at(&spec, dim);
            if !v.is_undefined() {
                validate_dim_spec(&v, errors, &format!("{label}.{dim}"));
            }
        }
        let depth = at(&spec, "depth");
        if !depth.is_undefined() && (!is_finite_number(&depth) || num(&depth) <= 0.0) {
            errors.push(format!(
                "{label}: \"depth\" must be a positive finite number"
            ));
        }
        let format = at(&spec, "format");
        if !format.is_undefined() && !includes(FORMATS, &format) {
            errors.push(format!("{label}: unknown format '{}'", to_string(&format)?));
        }
        let is3d = at(&spec, "is3D");
        if !is3d.is_undefined() && !matches!(is3d, Value::Bool(_)) {
            errors.push(format!("{label}: \"is3D\" must be a boolean"));
        }
        let filter = at(&spec, "filter");
        if !filter.is_undefined() {
            if !is_3d {
                errors.push(format!(
                    "{label}: \"filter\" is only supported on 3D texture specs (\"textures3d\")"
                ));
            } else if !includes(TEXTURE_FILTERS, &filter) {
                errors.push(format!(
                    "{label}: unknown filter '{}' (expected 'nearest' or 'linear')",
                    to_string(&filter)?
                ));
            }
        }
        for field in ["mipmaps", "persistent"] {
            let v = at(&spec, field);
            if v.is_undefined() {
                continue;
            }
            if is_3d {
                errors.push(format!(
                    "{label}: \"{field}\" is only supported on 2D texture specs (\"textures\")"
                ));
            } else if !matches!(v, Value::Bool(_)) {
                errors.push(format!("{label}: \"{field}\" must be a boolean"));
            }
        }
    }
    Ok(())
}

/// `/^o[0-7]$/`.
fn is_output_surface(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 2 && b[0] == b'o' && (b'0'..=b'7').contains(&b[1])
}

/// `validatePass(source, pass, index, errors, context)`.
fn validate_pass(
    source: &Source<'_>,
    pass: &Value,
    index: usize,
    errors: &mut Errors,
    context: &Context,
) -> Result<(), JsError> {
    let label = format!("Pass {index}");
    if !is_obj(pass) {
        errors.push(format!("{label}: must be an object"));
        return Ok(());
    }

    let program = at(pass, "program");
    if !program.is_truthy() || !matches!(program, Value::String(_)) {
        errors.push(format!("{label}: Missing \"program\" string"));
    }

    for key in keys(pass) {
        if !PASS_KEYS.contains(&key.as_str()) {
            errors.push(format!("{label}: unknown field '{key}'"));
        }
    }

    for field in ["name", "entryPoint"] {
        let v = at(pass, field);
        if !v.is_undefined() && !non_empty_string(&v) {
            errors.push(format!("{label}: \"{field}\" must be a non-empty string"));
        }
    }
    let pass_type = at(pass, "type");
    if !pass_type.is_undefined() && !includes(PASS_TYPES, &pass_type) {
        errors.push(format!(
            "{label}: unknown pass type '{}'",
            to_string(&pass_type)?
        ));
    }
    let draw_mode = at(pass, "drawMode");
    if !draw_mode.is_undefined() && !includes(DRAW_MODES, &draw_mode) {
        errors.push(format!(
            "{label}: unknown drawMode '{}'",
            to_string(&draw_mode)?
        ));
    }
    let draw_buffers = at(pass, "drawBuffers");
    if !draw_buffers.is_undefined() && (!is_integer(&draw_buffers) || num(&draw_buffers) < 1.0) {
        errors.push(format!(
            "{label}: \"drawBuffers\" must be a positive integer"
        ));
    }
    let count = at(pass, "count");
    if !count.is_undefined() {
        if let Value::String(c) = &count {
            if !["auto", "screen", "input"].contains(&c.as_str()) {
                errors.push(format!("{label}: unknown count '{c}'"));
            }
        } else if !is_integer(&count) || num(&count) < 1.0 {
            errors.push(format!(
                "{label}: \"count\" must be a positive integer, 'auto', 'screen', or 'input'"
            ));
        }
    }
    let count_uniform = at(pass, "countUniform");
    if !count_uniform.is_undefined() {
        if !non_empty_string(&count_uniform) {
            errors.push(format!(
                "{label}: \"countUniform\" must be a non-empty string"
            ));
        } else if !context.references_global(&count_uniform) {
            errors.push(format!(
                "{label}: countUniform '{}' does not reference a declared global",
                count_uniform.as_str().unwrap_or_default()
            ));
        }
    }
    let repeat = at(pass, "repeat");
    if !repeat.is_undefined() {
        if let Value::String(r) = &repeat {
            if r.is_empty() {
                errors.push(format!("{label}: \"repeat\" string must name a uniform"));
            }
        } else if !is_integer(&repeat) || num(&repeat) < 1.0 {
            errors.push(format!(
                "{label}: \"repeat\" must be a positive integer or a uniform name string"
            ));
        }
    }
    let blend = at(pass, "blend");
    if !blend.is_undefined() {
        let ok = match &blend {
            Value::Bool(_) => true,
            Value::Array(a) => a.len() == 2 && a.iter().all(non_empty_string),
            _ => false,
        };
        if !ok {
            errors.push(format!(
                "{label}: \"blend\" must be a boolean or [src, dst] factor strings"
            ));
        }
    }
    let workgroups = at(pass, "workgroups");
    if !workgroups.is_undefined() {
        let ok = matches!(&workgroups, Value::Array(a)
            if (1..=3).contains(&a.len()) && a.iter().all(|v| is_finite_number(v) || non_empty_string(v)));
        if !ok {
            errors.push(format!(
                "{label}: \"workgroups\" must be an array of 1-3 numbers or uniform names"
            ));
        }
    }
    for field in ["storageBuffers", "storageTextures"] {
        let v = at(pass, field);
        if !v.is_undefined() && !is_obj(&v) {
            errors.push(format!("{label}: \"{field}\" must be an object"));
        }
    }
    let viewport = at(pass, "viewport");
    if !viewport.is_undefined() {
        if !is_obj(&viewport) {
            errors.push(format!("{label}: \"viewport\" must be an object"));
        } else {
            for (key, v) in entries(&viewport) {
                if key == "width" || key == "height" {
                    validate_dim_spec(&v, errors, &format!("{label}.viewport.{key}"));
                } else if ["x", "y", "w", "h"].contains(&key.as_str()) {
                    if !is_finite_number(&v) {
                        errors.push(format!("{label}.viewport.{key} must be a finite number"));
                    }
                } else {
                    errors.push(format!("{label}.viewport: unknown field '{key}'"));
                }
            }
        }
    }
    let conditions = at(pass, "conditions");
    if !conditions.is_undefined() {
        if !is_obj(&conditions) {
            errors.push(format!("{label}: \"conditions\" must be an object"));
        } else {
            for key in keys(&conditions) {
                if !CONDITION_CONTAINER_KEYS.contains(&key.as_str()) {
                    errors.push(format!("{label}.conditions: unknown field '{key}'"));
                }
            }
            for list_key in CONDITION_CONTAINER_KEYS {
                let list = at(&conditions, list_key);
                if list.is_undefined() {
                    continue;
                }
                let Value::Array(list) = list else {
                    errors.push(format!("{label}.conditions.{list_key} must be an array"));
                    continue;
                };
                for condition in &list {
                    if !is_obj(condition) {
                        errors.push(format!(
                            "{label}.conditions.{list_key}: condition must be an object"
                        ));
                        continue;
                    }
                    for key in keys(condition) {
                        if key != "uniform" && key != "equals" {
                            errors.push(format!(
                                "{label}.conditions.{list_key}: unknown condition field '{key}'"
                            ));
                        }
                    }
                    let uniform = at(condition, "uniform");
                    if !non_empty_string(&uniform) {
                        errors.push(format!(
                            "{label}.conditions.{list_key}: \"uniform\" must be a non-empty string"
                        ));
                    } else if !context.references_global(&uniform) {
                        errors.push(format!(
                            "{label}.conditions.{list_key}: uniform '{}' does not reference a declared global",
                            uniform.as_str().unwrap_or_default()
                        ));
                    }
                    if at(condition, "equals").is_undefined() {
                        errors.push(format!(
                            "{label}.conditions.{list_key}: condition requires an \"equals\" value"
                        ));
                    }
                }
            }
        }
    }

    let uniforms = at(pass, "uniforms");
    if !uniforms.is_undefined() {
        if !is_obj(&uniforms) {
            errors.push(format!("{label}: \"uniforms\" must be an object"));
        } else {
            for (uniform_name, v) in entries(&uniforms) {
                // Numeric literals are preserved; strings name runtime uniforms.
                if !is_finite_number(&v) && !non_empty_string(&v) {
                    errors.push(format!(
                        "{label}: uniforms['{uniform_name}'] must be a finite number or a non-empty string"
                    ));
                }
            }
        }
    }

    let defines = at(pass, "defines");
    if !defines.is_undefined() {
        if !is_obj(&defines) {
            errors.push(format!("{label}: \"defines\" must be an object"));
        } else {
            for (key, v) in entries(&defines) {
                if !matches!(v, Value::String(_)) && !is_finite_number(&v) {
                    errors.push(format!(
                        "{label}: defines['{key}'] must be a string or finite number"
                    ));
                }
            }
        }
    }

    let or_empty = |v: Value| {
        if v.is_truthy() {
            v
        } else {
            Value::Object(Object::new())
        }
    };
    let mut declared_textures = keys(&or_empty(source.get("textures")));
    declared_textures.extend(keys(&or_empty(source.get("textures3d"))));
    let external_texture = source.get("externalTexture");

    let inputs = at(pass, "inputs");
    if !inputs.is_undefined() {
        if !is_obj(&inputs) {
            errors.push(format!("{label}: \"inputs\" must be an object"));
        } else {
            for (uniform_name, tex_ref) in entries(&inputs) {
                let Value::String(tex) = &tex_ref else {
                    errors.push(format!(
                        "{label}: inputs['{uniform_name}'] must be a non-empty texture reference string"
                    ));
                    continue;
                };
                if tex.is_empty() {
                    errors.push(format!(
                        "{label}: inputs['{uniform_name}'] must be a non-empty texture reference string"
                    ));
                    continue;
                }
                if PIPELINE_INPUTS.contains(&tex.as_str())
                    || is_output_surface(tex)
                    || tex.starts_with("global_")
                    || declared_textures.iter().any(|t| t == tex)
                    || context.has_global(&tex_ref)
                    || matches!(&external_texture, Value::String(e) if e == tex)
                {
                    continue;
                }
                errors.push(format!(
                    "{label}: inputs['{uniform_name}'] references unsupported texture '{tex}'"
                ));
            }
        }
    }

    let outputs = at(pass, "outputs");
    if !outputs.is_undefined() {
        if !is_obj(&outputs) {
            errors.push(format!("{label}: \"outputs\" must be an object"));
        } else {
            for (attachment, tex_ref) in entries(&outputs) {
                let tex = match &tex_ref {
                    Value::String(t) if !t.is_empty() => t,
                    _ => {
                        errors.push(format!(
                            "{label}: outputs['{attachment}'] must be a non-empty texture reference string"
                        ));
                        continue;
                    }
                };
                if PIPELINE_OUTPUTS.contains(&tex.as_str())
                    || tex.starts_with("global_")
                    || declared_textures.iter().any(|t| t == tex)
                {
                    continue;
                }
                errors.push(format!(
                    "{label}: outputs['{attachment}'] references unsupported output '{tex}'"
                ));
            }
        }
    }
    Ok(())
}

/// The object the reference validates (`source`): property reads follow the
/// definition's JavaScript shape.
enum Source<'a> {
    /// A plain value (own members; functions have an empty `name`).
    Plain(&'a Value),
    /// An `Effect` instance: own members, then prototype methods.
    Instance {
        own: &'a Value,
        prototype_methods: &'a [String],
    },
    /// The null-prototype merge of a subclass constructor.
    Merged(Object),
}

impl Source<'_> {
    fn get(&self, key: &str) -> Value {
        match self {
            Source::Plain(Value::Function(_)) => match key {
                "name" => Value::from(""),
                _ => Value::Undefined,
            },
            Source::Plain(v) => member(v, key),
            Source::Instance {
                own,
                prototype_methods,
            } => match own.as_object().and_then(|o| o.get(key)) {
                Some(v) => v.clone(),
                None if prototype_methods.iter().any(|m| m == key)
                    || EFFECT_PROTOTYPE_METHODS.contains(&key) =>
                {
                    Value::Function(format!("{key}() {{}}"))
                }
                None => Value::Undefined,
            },
            Source::Merged(o) => o.get(key).cloned().unwrap_or(Value::Undefined),
        }
    }
}

/// `validateEffectDefinition(def)` for a definition of any JavaScript shape.
pub fn validate_definition(definition: Definition<'_>) -> Result<Vec<String>, JsError> {
    let mut errors: Errors = Vec::new();

    // Effect instances and subclass constructors carry legitimate extra state,
    // so top-level unknown-field diagnosis runs on plain definition objects only.
    let (source, diagnose_top_level_unknowns, top_level_keys) = match definition {
        Definition::Plain(def) => {
            if !def.is_truthy() {
                return Ok(vec!["Effect definition is null or undefined".into()]);
            }
            if matches!(def, Value::Array(_)) {
                errors.push(
                    "Effect definition must be a plain object or Effect instance, not an array"
                        .into(),
                );
                return Ok(errors);
            }
            if !is_obj(def) && !matches!(def, Value::Function(_)) {
                errors.push(format!(
                    "Effect definition must be a plain object or Effect instance, not {}",
                    def.type_of()
                ));
                return Ok(errors);
            }
            (Source::Plain(def), true, keys(def))
        }
        Definition::Instance {
            own,
            prototype_methods,
        } => (
            Source::Instance {
                own,
                prototype_methods,
            },
            false,
            Vec::new(),
        ),
        Definition::Subclass {
            prototype_methods,
            statics,
        } => {
            let mut merged = Object::new();
            for key in prototype_methods {
                if key != "constructor" {
                    merged.insert(key.clone(), Value::Function(format!("{key}() {{}}")));
                }
            }
            for (key, value) in statics.iter() {
                merged.insert(key.clone(), value.clone());
            }
            (Source::Merged(merged), false, Vec::new())
        }
    };

    let globals = source.get("globals");
    let context_globals = Context {
        global_keys: if is_obj(&globals) {
            keys(&globals)
        } else {
            Vec::new()
        },
        global_uniform_names: Vec::new(),
    };
    let mut context = context_globals;
    if is_obj(&globals) {
        for spec in crate::unparser::jsv::values(&globals) {
            if is_obj(&spec)
                && let Value::String(u) = at(&spec, "uniform")
                && !u.is_empty()
                && !context.global_uniform_names.contains(&u)
            {
                context.global_uniform_names.push(u);
            }
        }
    }

    // --- name (existing message preserved) ---
    if !non_empty_string(&source.get("name")) {
        errors.push("Missing or invalid \"name\" property".into());
    }

    // --- simple typed metadata ---
    for field in ["namespace", "func"] {
        let v = source.get(field);
        if !v.is_undefined() && !non_empty_string(&v) {
            errors.push(format!("\"{field}\" must be a non-empty string"));
        }
    }
    let description = source.get("description");
    if !description.is_undefined() && !matches!(description, Value::String(_)) {
        errors.push("\"description\" must be a string".into());
    }
    let tags = source.get("tags");
    if !tags.is_undefined() {
        match &tags {
            Value::Array(list) => {
                for tag in list {
                    match tag {
                        Value::String(t) if !t.is_empty() => {
                            if !VALID_TAGS.contains(&t.as_str()) {
                                errors.push(format!("Unknown tag '{t}'"));
                            }
                        }
                        _ => errors.push("\"tags\" must contain non-empty strings".into()),
                    }
                }
            }
            _ => errors.push("\"tags\" must be an array of tag strings".into()),
        }
    }
    let open_categories = source.get("openCategories");
    if !open_categories.is_undefined()
        && !matches!(&open_categories, Value::Array(a) if a.iter().all(|c| matches!(c, Value::String(_))))
    {
        errors.push("\"openCategories\" must be an array of strings".into());
    }
    let default_program = source.get("defaultProgram");
    if !default_program.is_undefined() && !matches!(default_program, Value::String(_)) {
        errors.push("\"defaultProgram\" must be a string".into());
    }
    let hidden = source.get("hidden");
    if !hidden.is_undefined() && !matches!(hidden, Value::Bool(_)) {
        errors.push("\"hidden\" must be a boolean".into());
    }
    for field in ["deprecatedBy", "externalTexture", "externalMesh"] {
        let v = source.get(field);
        if !v.is_undefined() && !non_empty_string(&v) {
            errors.push(format!("\"{field}\" must be a non-empty string"));
        }
    }
    let builtin_meshes = source.get("builtinMeshes");
    if !builtin_meshes.is_undefined() {
        if !is_obj(&builtin_meshes) {
            errors.push("\"builtinMeshes\" must be an object".into());
        } else {
            for (key, v) in entries(&builtin_meshes) {
                if !non_empty_string(&v) {
                    errors.push(format!("builtinMeshes['{key}'] must be a non-empty string"));
                }
            }
        }
    }
    for field in ["outputTex3d", "outputGeo"] {
        let v = source.get(field);
        if !v.is_undefined() && !non_empty_string(&v) {
            errors.push(format!("\"{field}\" must be a non-empty string"));
        }
    }

    // --- lifecycle hooks must be functions ---
    for hook in ["onInit", "onUpdate", "onDestroy", "asyncInit"] {
        let v = source.get(hook);
        if !v.is_undefined() && !matches!(v, Value::Function(_)) {
            errors.push(format!("\"{hook}\" must be a function"));
        }
    }

    // --- globals ---
    validate_globals(&globals, &mut errors, &context)?;

    // --- passes ---
    match source.get("passes") {
        Value::Array(passes) if !passes.is_empty() => {
            for (index, pass) in passes.iter().enumerate() {
                validate_pass(&source, pass, index, &mut errors, &context)?;
            }
        }
        _ => errors.push("Missing or empty \"passes\" array".into()),
    }

    // --- textures ---
    validate_texture_map(&source.get("textures"), &mut errors, "textures")?;
    validate_texture_map(&source.get("textures3d"), &mut errors, "textures3d")?;

    // --- shaders ---
    let shaders = source.get("shaders");
    if !shaders.is_undefined() {
        if !is_obj(&shaders) {
            errors
                .push("\"shaders\" must be an object mapping program names to shader maps".into());
        } else {
            for (prog, program_shaders) in entries(&shaders) {
                if !is_obj(&program_shaders) {
                    errors.push(format!("shaders['{prog}'] must be an object"));
                }
            }
        }
    }

    // --- uniform layouts ---
    let uniform_layout = source.get("uniformLayout");
    if !uniform_layout.is_undefined() {
        validate_uniform_layout(&uniform_layout, &mut errors, "uniformLayout")?;
    }
    let uniform_layouts = source.get("uniformLayouts");
    if !uniform_layouts.is_undefined() {
        if !is_obj(&uniform_layouts) {
            errors.push(
                "\"uniformLayouts\" must be an object mapping program names to layouts".into(),
            );
        } else {
            for (prog, layout) in entries(&uniform_layouts) {
                validate_uniform_layout(
                    &layout,
                    &mut errors,
                    &format!("uniformLayouts['{prog}']"),
                )?;
            }
        }
    }

    // --- param aliases ---
    let param_aliases = source.get("paramAliases");
    if !param_aliases.is_undefined() {
        if !is_obj(&param_aliases) {
            errors
                .push("\"paramAliases\" must be an object mapping aliases to global names".into());
        } else {
            for (alias, target) in entries(&param_aliases) {
                match &target {
                    Value::String(t) if !t.is_empty() => {
                        if !context.has_global(&target) {
                            errors.push(format!(
                                "paramAliases['{alias}'] references unknown global '{t}'"
                            ));
                        }
                    }
                    _ => errors.push(format!(
                        "paramAliases['{alias}'] must be a non-empty string"
                    )),
                }
            }
        }
    }

    // --- top-level unknown-field diagnosis (plain definition objects only) ---
    if diagnose_top_level_unknowns {
        for key in top_level_keys {
            if !TOP_LEVEL_KEYS.contains(&key.as_str()) {
                errors.push(format!("Unknown definition field '{key}'"));
            }
        }
    }

    Ok(errors)
}

/// `validateEffectDefinition(def)` for a plain value (a JSON definition, a
/// Portable package, or a malformed container): the error list as a JavaScript
/// array of strings (`[]` when valid).
pub fn validate_effect_definition(def: &Value) -> Result<Value, JsError> {
    errors_value(validate_definition(Definition::Plain(def)))
}

/// `validateEffectDefinition(instance)` for an `Effect` instance given by its own
/// properties (the shape of a catalog `definition.json`).
pub fn validate_effect_instance(own: &Value) -> Result<Value, JsError> {
    errors_value(validate_definition(Definition::Instance {
        own,
        prototype_methods: &[],
    }))
}

fn errors_value(errors: Result<Vec<String>, JsError>) -> Result<Value, JsError> {
    Ok(Value::Array(errors?.into_iter().map(Value::from).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_definition() -> Value {
        Value::from_json(
            r##"{
            "name": "Validator Probe", "namespace": "synth", "func": "validatorProbe",
            "description": "Used by the definition-validator tests", "tags": ["noise", "util"],
            "openCategories": ["general"], "defaultProgram": "search synth\nvalidatorProbe().write(o0)",
            "hidden": false,
            "uniformLayout": {"resolution": {"slot": 0, "components": "xy"}, "time": {"slot": 0, "components": "z"}},
            "uniformLayouts": {"alt": {"amount": {"slot": 0, "components": "x"}}},
            "paramAliases": {"amt": "amount"},
            "textures": {
                "scratch": {"width": 64, "height": "100%", "format": "rgba16f"},
                "scaled": {"width": {"param": "volumeSize", "power": 2, "default": 1024}, "height": {"screenDivide": "zoom", "default": 8}}
            },
            "globals": {
                "amount": {"type": "float", "default": 0.5, "uniform": "amount", "min": 0, "max": 1, "step": 0.01, "zero": 0,
                           "ui": {"label": "amount", "control": "slider", "category": "effect"}},
                "mode": {"type": "int", "default": 1, "uniform": "mode", "define": "PROBE_MODE", "choices": {"off": 0, "on": 1, "Group:": null},
                         "ui": {"label": "mode", "control": "dropdown", "enabledBy": "amount"}},
                "flag": {"type": "boolean", "default": true, "uniform": "flag", "ui": {"label": "flag", "control": "checkbox", "enabledBy": {"param": "mode", "eq": 1}}},
                "tint": {"type": "color", "default": [1, 0, 0], "uniform": "tint", "ui": {"label": "tint", "control": "color"}},
                "point": {"type": "vec3", "default": [0, 0, 0], "uniform": "point", "min": [-1, -1, -1], "max": [1, 1, 1],
                          "ui": {"label": "point", "control": "vector3", "format": "x/y/z"}},
                "table": {"type": "int", "default": 7, "choices": {"a": 7, "b": 9}, "ui": {"label": "table", "control": "dropdown", "hidden": true}},
                "surfaceIn": {"type": "surface", "default": "none", "colorModeUniform": "surfaceActive", "ui": {"label": "surface", "control": false}}
            },
            "passes": [{
                "name": "render", "program": "probe", "type": "compute", "drawMode": "points", "count": "input",
                "countUniform": "mode", "repeat": 2, "blend": ["ONE", "ONE_MINUS_SRC_ALPHA"], "drawBuffers": 2,
                "workgroups": [8, 8, 1], "viewport": {"width": {"param": "volumeSize", "paramDefault": 64}, "height": 32},
                "conditions": {"runIf": [{"uniform": "mode", "equals": 1}], "skipIf": [{"uniform": "flag", "equals": false}]},
                "uniforms": {"amount": "amount", "literal": 3},
                "inputs": {"srcTex": "inputTex", "scratchTex": "scratch", "paramTex": "surfaceIn"},
                "outputs": {"fragColor": "outputTex"}
            }]
        }"##,
        )
        .unwrap()
    }

    fn errors(def: &Value) -> Vec<String> {
        validate_definition(Definition::Plain(def)).unwrap()
    }

    #[test]
    fn valid_definition_has_no_errors() {
        assert_eq!(errors(&valid_definition()), Vec::<String>::new());
    }

    #[test]
    fn containers_and_unknown_fields() {
        assert_eq!(
            errors(&Value::Null),
            vec!["Effect definition is null or undefined"]
        );
        assert_eq!(
            errors(&Value::from(42.0)),
            vec!["Effect definition must be a plain object or Effect instance, not number"]
        );
        let mut def = valid_definition();
        def.set("globalz", Value::object());
        assert_eq!(errors(&def), vec!["Unknown definition field 'globalz'"]);
        // Instances do not diagnose top-level extras.
        assert!(
            validate_definition(Definition::Instance {
                own: &def,
                prototype_methods: &[]
            })
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn member_enum_resolution() {
        assert_eq!(resolve_std_enum(&"oscType".into()), Some(StdEnum::Table));
        assert_eq!(
            resolve_std_enum(&"oscType.sine".into()),
            Some(StdEnum::Leaf)
        );
        assert_eq!(resolve_std_enum(&"palette".into()), Some(StdEnum::Table));
        assert_eq!(resolve_std_enum(&"noSuchTable".into()), None);
        assert_eq!(resolve_std_enum(&"__proto__".into()), Some(StdEnum::Table));
        assert_eq!(resolve_std_enum(&"constructor".into()), None);
    }

    #[test]
    fn layout_conflicts_and_ranges() {
        let mut def = valid_definition();
        def.set(
            "uniformLayout",
            Value::from_json(
                r#"{"a": {"slot": 1, "components": "x"}, "b": {"slot": 1, "components": "xy"}}"#,
            )
            .unwrap(),
        );
        assert_eq!(
            errors(&def),
            vec!["uniformLayout: layout conflict at slot 1: 'a' (x) overlaps 'b' (xy)"]
        );
        let mut def = valid_definition();
        def.get_mut("globals")
            .unwrap()
            .get_mut("amount")
            .unwrap()
            .set("default", Value::Number(2.0));
        assert_eq!(
            errors(&def),
            vec!["Global 'amount': default 2 is outside the declared range [0, 1]"]
        );
    }
}
