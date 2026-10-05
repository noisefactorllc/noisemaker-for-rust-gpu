//! WGSL source analysis of the reference WebGPU backend.
//!
//! The reference does not reflect shaders; it reads them with regular expressions:
//! binding declarations (`parseShaderBindings`), entry points
//! (`detectEntryPoints`), define injection (`injectDefines`), the declared uniform
//! struct size (`parseDeclaredUniformBufferSize`) and the uniform packing layout
//! (`parsePackedUniformLayout` and its four strategies). These choices decide what
//! bytes reach the GPU, so each function here is a line-for-line port with the
//! reference's exact patterns (see [`crate::jsre`]).

use indexmap::IndexMap;
use noisemaker_dsl::Value;
use noisemaker_dsl::js::{is_js_whitespace, number_to_string, parse_int, trim};

use crate::jsre::{JsRegex, group};
use crate::jsv::to_js_string;

/// `stripWGSLComments`: comments become spaces (newlines kept), nested block
/// comments included.
pub fn strip_wgsl_comments(source: &str) -> String {
    let chars: Vec<char> = source.chars().collect();
    let mut stripped = String::with_capacity(source.len());
    let mut block_depth = 0usize;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let next = chars.get(i + 1).copied();
        if block_depth > 0 {
            if c == '/' && next == Some('*') {
                stripped.push_str("  ");
                block_depth += 1;
                i += 2;
            } else if c == '*' && next == Some('/') {
                stripped.push_str("  ");
                block_depth -= 1;
                i += 2;
            } else {
                stripped.push(if c == '\n' { '\n' } else { ' ' });
                i += 1;
            }
        } else if c == '/' && next == Some('/') {
            stripped.push_str("  ");
            i += 2;
            while i < chars.len() && chars[i] != '\n' {
                stripped.push(' ');
                i += 1;
            }
        } else if c == '/' && next == Some('*') {
            stripped.push_str("  ");
            block_depth = 1;
            i += 2;
        } else {
            stripped.push(c);
            i += 1;
        }
    }
    stripped
}

/// `hasShaderBindings`.
pub fn has_shader_bindings(source: &str) -> bool {
    JsRegex::new(r"@binding\s*\(", "").test(&strip_wgsl_comments(source))
}

/// The binding classes of `parseShaderBindings`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BindingKind {
    StorageTexture,
    Texture,
    Sampler,
    Uniform,
    Storage,
    Unknown,
}

impl BindingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            BindingKind::StorageTexture => "storage_texture",
            BindingKind::Texture => "texture",
            BindingKind::Sampler => "sampler",
            BindingKind::Uniform => "uniform",
            BindingKind::Storage => "storage",
            BindingKind::Unknown => "unknown",
        }
    }
}

/// One parsed `@group(N) @binding(M) var<...> name: type` declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct ShaderBinding {
    pub group: u32,
    pub binding: u32,
    pub kind: BindingKind,
    pub name: String,
    /// The `var<...>` address space text (`uniform`, `storage, read_write`, or empty).
    pub storage: String,
    /// The declared type, trimmed.
    pub type_decl: String,
}

fn parse_index(digits: &str) -> u32 {
    let n = parse_int(digits, 10);
    if n > u32::MAX as f64 {
        u32::MAX
    } else {
        n as u32
    }
}

/// `parseShaderBindings`: every binding declaration of the comment-stripped source,
/// minus non-storage bindings whose name appears only once (the dead-binding
/// filter), sorted by group then binding.
pub fn parse_shader_bindings(source: &str) -> Vec<ShaderBinding> {
    let source_no_comments = strip_wgsl_comments(source);
    let binding_re = JsRegex::new(
        r"@group\s*\(\s*(\d+)\s*\)\s*@binding\s*\(\s*(\d+)\s*\)\s*var(?:<([^>]+)>)?\s+(\w+)\s*:\s*([^;]+)",
        "g",
    );
    let mut bindings = Vec::new();
    for caps in binding_re.exec_all(&source_no_comments) {
        let group_index = parse_index(group(&caps, 1).unwrap());
        let binding = parse_index(group(&caps, 2).unwrap());
        let storage = group(&caps, 3).unwrap_or("").to_owned();
        let name = group(&caps, 4).unwrap().to_owned();
        let type_decl = trim(group(&caps, 5).unwrap()).to_owned();
        let kind = if type_decl.contains("texture_storage_2d") {
            BindingKind::StorageTexture
        } else if type_decl.contains("texture_2d") || type_decl.contains("texture_3d") {
            BindingKind::Texture
        } else if type_decl == "sampler" {
            BindingKind::Sampler
        } else if storage.contains("uniform") {
            BindingKind::Uniform
        } else if storage.contains("storage") {
            BindingKind::Storage
        } else {
            BindingKind::Unknown
        };
        bindings.push(ShaderBinding {
            group: group_index,
            binding,
            kind,
            name,
            storage,
            type_decl,
        });
    }
    let mut filtered: Vec<ShaderBinding> = bindings
        .into_iter()
        .filter(|b| {
            if matches!(b.kind, BindingKind::StorageTexture | BindingKind::Storage) {
                return true;
            }
            let re = JsRegex::new(&format!(r"\b{}\b", b.name), "g");
            re.count(&source_no_comments) > 1
        })
        .collect();
    filtered.sort_by(|a, b| a.group.cmp(&b.group).then(a.binding.cmp(&b.binding)));
    filtered
}

/// The entry point names `detectEntryPoints` finds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EntryPoints {
    pub vertex: Option<String>,
    pub fragment: Option<String>,
    pub compute: Option<String>,
}

/// `detectEntryPoints`.
pub fn detect_entry_points(source: &str) -> EntryPoints {
    let first = |pattern: &str| {
        JsRegex::new(pattern, "")
            .exec(source)
            .and_then(|c| group(&c, 1).map(str::to_owned))
    };
    EntryPoints {
        vertex: first(r"@vertex\s*\n?\s*fn\s+(\w+)"),
        fragment: first(r"@fragment\s*\n?\s*fn\s+(\w+)"),
        compute: first(r"@compute[^f]*fn\s+(\w+)"),
    }
}

/// `parseEntryPointBindings`: for every entry point, the binding indices whose
/// names occur (as whole words) in its function body.
pub fn parse_entry_point_bindings(
    source: &str,
    bindings: &[ShaderBinding],
) -> IndexMap<String, Vec<u32>> {
    let mut map: IndexMap<String, Vec<u32>> = IndexMap::new();
    let re = JsRegex::new(
        r"@(?:compute|vertex|fragment)[^f]*fn\s+(\w+)\s*\([^)]*\)[^{]*\{",
        "g",
    );
    for caps in re.exec_all(source) {
        let entry_point = group(&caps, 1).unwrap().to_owned();
        let start = caps.get(0).unwrap().end();
        // Simple brace counting from just after the opening brace; the body ends
        // before the matching closing brace.
        let mut brace_count = 1i64;
        let mut end = start;
        for (i, c) in source[start..].char_indices() {
            if brace_count <= 0 {
                break;
            }
            if c == '{' {
                brace_count += 1;
            } else if c == '}' {
                brace_count -= 1;
            }
            end = start + i;
        }
        let body = &source[start..end];
        let mut used = Vec::new();
        for binding in bindings {
            if JsRegex::new(&format!(r"\b{}\b", binding.name), "").test(body)
                && !used.contains(&binding.binding)
            {
                used.push(binding.binding);
            }
        }
        map.insert(entry_point, used);
    }
    map
}

/// `injectDefines`: each define becomes a `const` declaration prepended to the
/// source (`bool`, `i32` for integral numbers, `f32` otherwise, untyped for
/// anything else).
pub fn inject_defines(source: &str, defines: &Value) -> String {
    let Some(obj) = defines.as_object() else {
        // `Object.keys` of a non-object define set: arrays enumerate their indices,
        // primitives have no keys.
        if let Value::Array(items) = defines {
            if items.is_empty() {
                return source.to_owned();
            }
            let mut injected = String::new();
            for (i, v) in items.iter().enumerate() {
                injected.push_str(&define_line(&i.to_string(), v));
            }
            return injected + source;
        }
        return source.to_owned();
    };
    if obj.is_empty() {
        return source.to_owned();
    }
    let mut injected = String::new();
    for (key, value) in obj.iter() {
        injected.push_str(&define_line(key, value));
    }
    injected + source
}

fn define_line(key: &str, value: &Value) -> String {
    match value {
        Value::Bool(b) => format!("const {key}: bool = {b};\n"),
        Value::Number(n) => {
            if value.is_integer() {
                format!("const {key}: i32 = {};\n", number_to_string(*n))
            } else {
                format!("const {key}: f32 = {};\n", number_to_string(*n))
            }
        }
        other => format!("const {key} = {};\n", to_js_string(other)),
    }
}

/// `computeWgslTypeSize`: `(size, align)`, `(0, 4)` for unrecognized types.
pub fn compute_wgsl_type_size(type_expr: &str) -> (u64, u64) {
    let array_re = JsRegex::new(r"^array\s*<\s*(.+?)\s*,\s*(\d+)\s*>$", "");
    if let Some(caps) = array_re.exec(type_expr) {
        let elem_type = group(&caps, 1).unwrap();
        let count = parse_int(group(&caps, 2).unwrap(), 10) as u64;
        let (elem_size, elem_align) = compute_wgsl_type_size(elem_type);
        let stride = elem_size.max(16);
        return (stride * count, elem_align.max(16));
    }
    let t: String = type_expr
        .chars()
        .filter(|c| !is_js_whitespace(*c))
        .collect();
    match t.as_str() {
        "f32" | "i32" | "u32" | "bool" => (4, 4),
        "f16" => (2, 2),
        "vec2<f32>" | "vec2f" | "vec2<i32>" | "vec2i" | "vec2<u32>" | "vec2u" => (8, 8),
        "vec3<f32>" | "vec3f" | "vec3<i32>" | "vec3i" | "vec3<u32>" | "vec3u" => (12, 16),
        "vec4<f32>" | "vec4f" | "vec4<i32>" | "vec4i" | "vec4<u32>" | "vec4u" => (16, 16),
        "mat3x3<f32>" | "mat3x3f" => (48, 16),
        "mat4x4<f32>" | "mat4x4f" => (64, 16),
        _ => (0, 4),
    }
}

/// `computeWgslStructSize`: std140-style size of a struct body, rounded to the
/// largest member alignment. Unknown member types are skipped.
pub fn compute_wgsl_struct_size(body: &str) -> u64 {
    let cleaned = JsRegex::new(r"//[^\n]*", "g").replace_all(body, "");
    let chars: Vec<char> = cleaned.chars().collect();
    let mut fields: Vec<String> = Vec::new();
    let mut depth = 0i64;
    let mut start = 0usize;
    for (i, &c) in chars.iter().enumerate() {
        if c == '<' {
            depth += 1;
        } else if c == '>' {
            depth -= 1;
        } else if (c == ',' || c == ';') && depth == 0 {
            let piece: String = chars[start..i].iter().collect();
            let piece = trim(&piece);
            if !piece.is_empty() {
                fields.push(piece.to_owned());
            }
            start = i + 1;
        }
    }
    let tail: String = chars[start.min(chars.len())..].iter().collect();
    let tail = trim(&tail);
    if !tail.is_empty() {
        fields.push(tail.to_owned());
    }

    let mut offset = 0u64;
    let mut max_align = 4u64;
    for field in &fields {
        let Some(colon) = field.find(':') else {
            continue;
        };
        let type_expr = trim(&field[colon + 1..]);
        let (size, align) = compute_wgsl_type_size(type_expr);
        if size == 0 {
            continue;
        }
        offset = offset.div_ceil(align) * align;
        offset += size;
        if align > max_align {
            max_align = align;
        }
    }
    offset.div_ceil(max_align) * max_align
}

/// `parseDeclaredUniformBufferSize`: the largest struct size among the structs
/// bound as `var<uniform>`, 0 when none is found.
pub fn parse_declared_uniform_buffer_size(source: &str) -> u64 {
    let binding_re = JsRegex::new(r"var<uniform>\s+\w+\s*:\s*(\w+)\s*;", "g");
    let mut largest = 0u64;
    for caps in binding_re.exec_all(source) {
        let struct_name = group(&caps, 1).unwrap();
        let struct_re = JsRegex::new(&format!(r"struct\s+{struct_name}\s*\{{([^}}]+)\}}"), "g");
        let Some(m) = struct_re.exec(source) else {
            continue;
        };
        let size = compute_wgsl_struct_size(group(&m, 1).unwrap());
        if size > largest {
            largest = size;
        }
    }
    largest
}

/// One entry of a slot-based uniform layout: a uniform packed into the
/// `components` of vec4 slot `slot`.
#[derive(Debug, Clone, PartialEq)]
pub struct SlotEntry {
    pub name: String,
    /// `entry.slot` (a number; non-numbers coerce to NaN as in `Math.max`).
    pub slot: f64,
    /// `entry.components`: normally a string such as `"xy"`.
    pub components: Value,
}

/// One entry of a byte-based uniform layout (`parseWgslStructByteLayout`).
#[derive(Debug, Clone, PartialEq)]
pub struct ByteEntry {
    pub name: String,
    pub offset: f64,
    pub size: f64,
    /// `'float' | 'int' | 'uint'`.
    pub ty: Value,
    /// Component count.
    pub components: Value,
}

/// A byte layout and the struct size the parser attached to it.
#[derive(Debug, Clone, PartialEq)]
pub struct ByteLayout {
    pub entries: Vec<ByteEntry>,
    /// `layout.structSize` (only parser-produced layouts carry it).
    pub struct_size: Option<f64>,
}

/// A program's `packedUniformLayout`.
#[derive(Debug, Clone, PartialEq)]
pub enum UniformLayout {
    /// `{ type: 'byte', layout: [...] }`.
    Byte(ByteLayout),
    /// The slot format (array entries, or the object format normalized to them).
    Slots(Vec<SlotEntry>),
}

fn component_rank(components: &Value) -> i64 {
    let first = match components {
        Value::String(s) => s.chars().next().map(String::from),
        Value::Array(a) => a.first().map(to_js_string),
        _ => None,
    };
    match first.as_deref() {
        Some("x") => 0,
        Some("y") => 1,
        Some("z") => 2,
        Some("w") => 3,
        _ => i64::MIN,
    }
}

/// The reference's `(a, b) => a.slot - b.slot || componentOrder[...] - ...` sort.
fn sort_slots(layout: &mut [SlotEntry]) {
    layout.sort_by(|a, b| {
        if a.slot != b.slot {
            return a
                .slot
                .partial_cmp(&b.slot)
                .unwrap_or(std::cmp::Ordering::Equal);
        }
        component_rank(&a.components).cmp(&component_rank(&b.components))
    });
}

/// `parsePackedUniformLayout`: the byte layout of a plain uniform struct, else
/// the comment-annotated named-struct layout, else the `params.field.c` access
/// layout, else the `uniforms.data[N].c` unpack layout.
pub fn parse_packed_uniform_layout(source: &str) -> Option<UniformLayout> {
    if let Some(byte) = parse_wgsl_struct_byte_layout(source)
        && !byte.entries.is_empty()
    {
        return Some(UniformLayout::Byte(byte));
    }
    if let Some(named) = parse_named_struct_layout(source)
        && !named.is_empty()
    {
        return Some(UniformLayout::Slots(named));
    }
    if let Some(access) = parse_params_access_layout(source)
        && !access.is_empty()
    {
        return Some(UniformLayout::Slots(access));
    }
    if !source.contains("uniforms.data[") {
        return None;
    }
    let unpack_re = JsRegex::new(
        r"(?:let\s+)?(\w+)(?:\s*:\s*[^\n=]+)?\s*=\s*(?:max\s*\([^,]+,\s*)?(?:i32\s*\(\s*)?uniforms\.data\[(\d+)\]\.([xyzw]+)",
        "g",
    );
    let mut layout = Vec::new();
    for caps in unpack_re.exec_all(source) {
        layout.push(SlotEntry {
            name: group(&caps, 1).unwrap().to_owned(),
            slot: parse_int(group(&caps, 2).unwrap(), 10),
            components: Value::from(group(&caps, 3).unwrap()),
        });
    }
    if layout.is_empty() {
        return None;
    }
    sort_slots(&mut layout);
    Some(UniformLayout::Slots(layout))
}

/// `getWgslTypeInfo`: `(size, align, baseType, components)`.
pub fn get_wgsl_type_info(ty: &str) -> (u64, u64, &'static str, u32) {
    match ty.to_lowercase().as_str() {
        "f32" => (4, 4, "float", 1),
        "i32" => (4, 4, "int", 1),
        "u32" => (4, 4, "uint", 1),
        "vec2f" | "vec2<f32>" => (8, 8, "float", 2),
        "vec3f" | "vec3<f32>" => (12, 16, "float", 3),
        "vec4f" | "vec4<f32>" => (16, 16, "float", 4),
        "vec2i" | "vec2<i32>" => (8, 8, "int", 2),
        "vec3i" | "vec3<i32>" => (12, 16, "int", 3),
        "vec4i" | "vec4<i32>" => (16, 16, "int", 4),
        "vec2u" | "vec2<u32>" => (8, 8, "uint", 2),
        "vec3u" | "vec3<u32>" => (12, 16, "uint", 3),
        "vec4u" | "vec4<u32>" => (16, 16, "uint", 4),
        _ => (4, 4, "float", 1),
    }
}

const PARAM_STRUCT_RE: &str = r"struct\s+(\w*(?:Params|Uniforms|Config|Settings))\s*\{([^}]+)\}";

/// `parseWgslStructByteLayout`: the byte layout of the first param-like struct
/// when it has no arrays and no `// (a, b, ...)` annotations, is bound as a
/// `var<uniform>`, and is used as `var.field`.
pub fn parse_wgsl_struct_byte_layout(source: &str) -> Option<ByteLayout> {
    let struct_re = JsRegex::new(PARAM_STRUCT_RE, "gi");
    let caps = struct_re.exec(source)?;
    let struct_name = group(&caps, 1).unwrap();
    let struct_body = group(&caps, 2).unwrap();

    if JsRegex::new(r"\barray\s*<", "").test(struct_body) {
        return None;
    }
    if JsRegex::new(r"\/\/\s*\(\s*\w+(?:\s*,\s*\w+)+\s*\)", "").test(struct_body) {
        return None;
    }
    let binding_re = JsRegex::new(
        &format!(r"var<uniform>\s+(\w+)\s*:\s*{struct_name}\s*;"),
        "",
    );
    let binding = binding_re.exec(source)?;
    let uniform_var_name = group(&binding, 1).unwrap();

    let field_re = JsRegex::new(
        r"(\w+)\s*:\s*(f32|i32|u32|vec2f|vec3f|vec4f|vec2<f32>|vec3<f32>|vec4<f32>|vec2i|vec3i|vec4i|vec2<i32>|vec3<i32>|vec4<i32>|vec2u|vec3u|vec4u|vec2<u32>|vec3<u32>|vec4<u32>)",
        "gi",
    );
    let mut entries = Vec::new();
    let mut offset = 0u64;
    let mut max_align = 4u64;
    for field in field_re.exec_all(struct_body) {
        let field_name = group(&field, 1).unwrap();
        let field_type = group(&field, 2).unwrap().to_lowercase();
        let (size, align, base_type, components) = get_wgsl_type_info(&field_type);
        max_align = max_align.max(align);
        offset = offset.div_ceil(align) * align;
        if field_name.starts_with('_') || field_name.to_lowercase().starts_with("pad") {
            offset += size;
            continue;
        }
        entries.push(ByteEntry {
            name: field_name.to_owned(),
            offset: offset as f64,
            size: size as f64,
            ty: Value::from(base_type),
            components: Value::from(components),
        });
        offset += size;
    }
    let struct_size = offset.div_ceil(max_align) * max_align;

    let usage_re = JsRegex::new(&format!(r"{uniform_var_name}\.(\w+)"), "");
    if !usage_re.test(source) {
        return None;
    }
    if entries.is_empty() {
        return None;
    }
    Some(ByteLayout {
        entries,
        struct_size: Some(struct_size as f64),
    })
}

/// `parseNamedStructLayout`: slot layouts from `// (name1, name2, ...)` field
/// annotations of every param-like struct (single-component fields without an
/// annotation use the field name).
pub fn parse_named_struct_layout(source: &str) -> Option<Vec<SlotEntry>> {
    const COMPONENT_NAMES: [&str; 4] = ["x", "y", "z", "w"];
    let mut layout = Vec::new();
    let struct_re = JsRegex::new(PARAM_STRUCT_RE, "gi");
    let field_re = JsRegex::new(
        r"(\w+)\s*:\s*(vec[234]<f32>|f32|i32|u32|array<[^>]+>)[^,;\n]*(?:,|;)?\s*(?:\/\/\s*\(([^)]+)\))?",
        "gi",
    );
    for structure in struct_re.exec_all(source) {
        let struct_body = group(&structure, 2).unwrap();
        // Each field takes one vec4 slot.
        for (slot, field) in field_re.exec_all(struct_body).enumerate() {
            let field_name = group(&field, 1).unwrap();
            let field_type = group(&field, 2).unwrap();
            let comment_names = group(&field, 3);
            let mut num_components = 4usize;
            if field_type == "f32" || field_type == "i32" || field_type == "u32" {
                num_components = 1;
            } else if field_type.starts_with("vec2") {
                num_components = 2;
            } else if field_type.starts_with("vec3") {
                num_components = 3;
            } else if field_type.starts_with("vec4") {
                num_components = 4;
            }
            if let Some(comment_names) = comment_names {
                let names: Vec<&str> = comment_names.split(',').map(trim).collect();
                for (i, name) in names.iter().enumerate().take(num_components) {
                    let lower = name.to_lowercase();
                    if !name.is_empty()
                        && *name != "_"
                        && !lower.starts_with("pad")
                        && !lower.starts_with("unused")
                        && !name.starts_with('_')
                    {
                        layout.push(SlotEntry {
                            name: (*name).to_owned(),
                            slot: slot as f64,
                            components: Value::from(COMPONENT_NAMES[i]),
                        });
                    }
                }
            } else if num_components == 1 {
                layout.push(SlotEntry {
                    name: field_name.to_owned(),
                    slot: slot as f64,
                    components: Value::from("x"),
                });
            }
        }
    }
    if layout.is_empty() {
        None
    } else {
        Some(layout)
    }
}

/// `parseParamsAccessLayout`: slot layouts from `let v = params.field.c;`
/// statements against the field order of the first param-like struct.
pub fn parse_params_access_layout(source: &str) -> Option<Vec<SlotEntry>> {
    let struct_re = JsRegex::new(
        r"struct\s+\w*(?:Params|Uniforms|Config|Settings)\s*\{([^}]+)\}",
        "gi",
    );
    let structure = struct_re.exec(source)?;
    let struct_body = group(&structure, 1).unwrap();
    let field_order_re = JsRegex::new(
        r"(\w+)\s*:\s*(?:vec[234]<f32>|f32|i32|u32|array<[^>]+>)",
        "gi",
    );
    let mut field_slots: IndexMap<String, u32> = IndexMap::new();
    for (slot, field) in field_order_re.exec_all(struct_body).enumerate() {
        field_slots.insert(group(&field, 1).unwrap().to_owned(), slot as u32);
    }
    let access_re = JsRegex::new(
        r"(?:let\s+)?(\w+)(?:\s*:\s*[^=\n]+)?\s*=\s*(?:i32\s*\(\s*)?params\.(\w+)\.([xyzw]+)",
        "g",
    );
    let mut layout = Vec::new();
    for access in access_re.exec_all(source) {
        let var_name = group(&access, 1).unwrap();
        let field_name = group(&access, 2).unwrap();
        let components = group(&access, 3).unwrap();
        if let Some(&slot) = field_slots.get(field_name) {
            layout.push(SlotEntry {
                name: var_name.to_owned(),
                slot: slot as f64,
                components: Value::from(components),
            });
        }
    }
    sort_slots(&mut layout);
    if layout.is_empty() {
        None
    } else {
        Some(layout)
    }
}

/// Normalize a definition-provided `spec.uniformLayout` value the way
/// `packUniformsWithLayout` reads it: `{type: 'byte', layout: [...]}`, an array of
/// `{name, slot, components}`, or an object `{name: {slot, components}}`.
pub fn normalize_spec_layout(layout: &Value) -> UniformLayout {
    if let Value::Object(o) = layout
        && o.get("type").and_then(Value::as_str) == Some("byte")
        && let Some(Value::Array(entries)) = o.get("layout")
    {
        let entries = entries
            .iter()
            .map(|e| ByteEntry {
                name: to_js_string(e.get("name")),
                offset: crate::jsv::to_number(e.get("offset")),
                size: crate::jsv::to_number(e.get("size")),
                ty: e.get("type").clone(),
                components: e.get("components").clone(),
            })
            .collect();
        return UniformLayout::Byte(ByteLayout {
            entries,
            struct_size: None,
        });
    }
    match layout {
        Value::Array(entries) => UniformLayout::Slots(
            entries
                .iter()
                .map(|e| SlotEntry {
                    name: to_js_string(e.get("name")),
                    slot: crate::jsv::to_number(e.get("slot")),
                    components: e.get("components").clone(),
                })
                .collect(),
        ),
        Value::Object(o) => UniformLayout::Slots(
            o.iter()
                .map(|(name, spec)| SlotEntry {
                    name: name.clone(),
                    slot: crate::jsv::to_number(spec.get("slot")),
                    components: spec.get("components").clone(),
                })
                .collect(),
        ),
        _ => UniformLayout::Slots(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_nested_comments_keeping_lines() {
        let s = "a /* b /* c */ d */ e // f\ng";
        assert_eq!(strip_wgsl_comments(s), "a                   e     \ng");
    }

    #[test]
    fn bindings_drop_dead_names_and_sort() {
        let src = "@group(0) @binding(2) var<uniform> u: U;\n\
                   @group(0) @binding(0) var tex: texture_2d<f32>;\n\
                   @group(0) @binding(1) var samp: sampler;\n\
                   @group(0) @binding(3) var dead: texture_2d<f32>; // dead dead\n\
                   fn main() { let c = textureSample(tex, samp, vec2f(0.0)) * u.x; }";
        let b = parse_shader_bindings(src);
        let names: Vec<_> = b.iter().map(|b| b.name.as_str()).collect();
        assert_eq!(names, ["tex", "samp", "u"]);
        assert_eq!(b[2].kind, BindingKind::Uniform);
        assert_eq!(b[2].type_decl, "U");
    }

    #[test]
    fn defines_follow_js_number_formatting() {
        let defines = Value::from_json(r#"{"A":3,"B":0.5,"C":true,"D":"vec2f(1.0)"}"#).unwrap();
        assert_eq!(
            inject_defines("X", &defines),
            "const A: i32 = 3;\nconst B: f32 = 0.5;\nconst C: bool = true;\nconst D = vec2f(1.0);\nX"
        );
        assert_eq!(inject_defines("X", &Value::object()), "X");
    }

    #[test]
    fn struct_sizes() {
        assert_eq!(compute_wgsl_struct_size("a: f32, b: vec3<f32>, c: f32"), 32);
        assert_eq!(
            compute_wgsl_struct_size("data: array<vec4<f32>, 2>, // two slots\n"),
            32
        );
        let src = "struct Uniforms { data: array<vec4<f32>, 3>, };\n\
                   @group(0) @binding(0) var<uniform> uniforms: Uniforms;";
        assert_eq!(parse_declared_uniform_buffer_size(src), 48);
    }

    #[test]
    fn byte_layout_of_plain_struct() {
        let src = "struct Params { resolution: vec2f, time: f32, _pad: f32, color: vec3<f32>, count: i32, }\n\
                   @group(0) @binding(0) var<uniform> u: Params;\n\
                   fn f() { let x = u.time; }";
        let Some(UniformLayout::Byte(layout)) = parse_packed_uniform_layout(src) else {
            panic!("expected a byte layout");
        };
        let fields: Vec<_> = layout
            .entries
            .iter()
            .map(|e| (e.name.as_str(), e.offset, e.size))
            .collect();
        assert_eq!(
            fields,
            [
                ("resolution", 0.0, 8.0),
                ("time", 8.0, 4.0),
                ("color", 16.0, 12.0),
                ("count", 28.0, 4.0)
            ]
        );
        assert_eq!(layout.struct_size, Some(32.0));
    }

    #[test]
    fn named_struct_layout_from_annotations() {
        let src = "struct FooParams {\n\
                   dims_freq : vec4<f32>,  // (width, height, channels, frequency)\n\
                   settings : vec4<f32>,   // (octaves, _, pad0, seed)\n\
                   };\n";
        let layout = parse_named_struct_layout(src).unwrap();
        let entries: Vec<_> = layout
            .iter()
            .map(|e| (e.name.as_str(), e.slot, e.components.as_str().unwrap()))
            .collect();
        assert_eq!(
            entries,
            [
                ("width", 0.0, "x"),
                ("height", 0.0, "y"),
                ("channels", 0.0, "z"),
                ("frequency", 0.0, "w"),
                ("octaves", 1.0, "x"),
                ("seed", 1.0, "w")
            ]
        );
    }

    #[test]
    fn data_array_unpack_layout() {
        let src = "struct Uniforms { data: array<vec4<f32>, 2> };\n\
                   @group(0) @binding(0) var<uniform> uniforms: Uniforms;\n\
                   fn f() {\n let res = uniforms.data[0].xy;\n let t: f32 = uniforms.data[0].z;\n\
                   let n = i32(uniforms.data[1].x);\n let s = max(1, i32(uniforms.data[0].w));\n }";
        let Some(UniformLayout::Slots(layout)) = parse_packed_uniform_layout(src) else {
            panic!("expected a slot layout");
        };
        let entries: Vec<_> = layout
            .iter()
            .map(|e| (e.name.as_str(), e.slot, e.components.as_str().unwrap()))
            .collect();
        assert_eq!(
            entries,
            [
                ("res", 0.0, "xy"),
                ("t", 0.0, "z"),
                ("s", 0.0, "w"),
                ("n", 1.0, "x")
            ]
        );
    }

    #[test]
    fn entry_point_detection() {
        let src = "@vertex\nfn vs(x: u32) -> V { }\n@fragment fn fs_main(in: V) -> @location(0) vec4f { }\n\
                   @compute @workgroup_size(8, 8, 1)\nfn cs(@builtin(global_invocation_id) id: vec3u) { }";
        let ep = detect_entry_points(src);
        assert_eq!(ep.vertex.as_deref(), Some("vs"));
        assert_eq!(ep.fragment.as_deref(), Some("fs_main"));
        assert_eq!(ep.compute.as_deref(), Some("cs"));
    }
}
