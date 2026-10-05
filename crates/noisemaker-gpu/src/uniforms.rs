//! Uniform packing of the reference WebGPU backend.
//!
//! `packUniforms` (std140-like packing of every merged uniform in key order),
//! `packUniformsWithLayout` (vec4 slot layouts), `packUniformsWithByteLayout`
//! (byte-offset layouts), `_resolveUniformAlias` and the single-value buffers of
//! `createSingleUniformBuffer`. Values are [`Value`]s and every store applies the
//! coercion of the `DataView`/typed-array setter the reference uses, including its
//! `RangeError` on out-of-bounds writes.

use noisemaker_dsl::js::math_round;
use noisemaker_dsl::{Object, Value};

use crate::jsre::{JsRegex, group};
use crate::jsv::{to_f32, to_number};
use crate::wgsl::{ByteLayout, SlotEntry, UniformLayout};

/// A `RangeError` raised by a `DataView` access or typed-array view.
pub fn range_error(what: &str) -> String {
    format!("RangeError: {what}")
}

/// ECMAScript `ToIndex` for a `DataView` byte offset.
fn to_index(offset: f64) -> Result<usize, String> {
    let integer = if offset.is_nan() { 0.0 } else { offset.trunc() };
    if !(0.0..=9_007_199_254_740_991.0).contains(&integer) {
        return Err(range_error("Offset is outside the bounds of the DataView"));
    }
    Ok(integer as usize)
}

/// A little-endian `DataView` over a byte buffer.
struct DataView<'a> {
    bytes: &'a mut [u8],
}

impl DataView<'_> {
    fn slot(&mut self, offset: f64) -> Result<&mut [u8], String> {
        let index = to_index(offset)?;
        if index
            .checked_add(4)
            .is_none_or(|end| end > self.bytes.len())
        {
            return Err(range_error("Offset is outside the bounds of the DataView"));
        }
        Ok(&mut self.bytes[index..index + 4])
    }

    fn set_f32(&mut self, offset: f64, value: f32) -> Result<(), String> {
        self.slot(offset)?.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn set_i32(&mut self, offset: f64, value: i32) -> Result<(), String> {
        self.slot(offset)?.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }

    fn set_u32(&mut self, offset: f64, value: u32) -> Result<(), String> {
        self.slot(offset)?.copy_from_slice(&value.to_le_bytes());
        Ok(())
    }
}

/// `new ArrayBuffer(length)` (`ToIndex` of the requested length).
fn array_buffer(length: f64) -> Result<Vec<u8>, String> {
    let integer = if length.is_nan() { 0.0 } else { length.trunc() };
    if !(0.0..=9_007_199_254_740_991.0).contains(&integer) {
        return Err(range_error("Invalid array buffer length"));
    }
    Ok(vec![0u8; integer as usize])
}

/// `_resolveUniformAlias`: a merged uniform, or the `width`/`height` of
/// `resolution`, or `channels` (4), or `channelCount` (4) for byte layouts.
pub fn resolve_uniform_alias(name: &str, uniforms: &Object, include_channel_count: bool) -> Value {
    let direct = uniforms.get_or_undefined(name);
    if !direct.is_undefined() {
        return direct.clone();
    }
    let resolution = uniforms.get_or_undefined("resolution");
    if name == "width" && resolution.is_truthy() {
        return resolution.get("0").clone();
    }
    if name == "height" && resolution.is_truthy() {
        return resolution.get("1").clone();
    }
    if name == "channels" {
        return Value::Number(4.0);
    }
    if include_channel_count && name == "channelCount" {
        return Value::Number(4.0);
    }
    Value::Undefined
}

/// `packUniformsWithLayout`.
pub fn pack_uniforms_with_layout(
    uniforms: &Object,
    layout: &UniformLayout,
) -> Result<Vec<u8>, String> {
    match layout {
        UniformLayout::Byte(byte) => pack_uniforms_with_byte_layout(uniforms, byte),
        UniformLayout::Slots(entries) => pack_uniforms_with_slots(uniforms, entries),
    }
}

/// `componentOffset[key]` with `key` converted to a property key.
fn component_offset(key: &Value) -> f64 {
    match crate::jsv::to_js_string(key).as_str() {
        "x" => 0.0,
        "y" => 4.0,
        "z" => 8.0,
        "w" => 12.0,
        _ => f64::NAN,
    }
}

/// `entry.components.length` (strings and arrays have one; `undefined` and `null`
/// throw).
fn components_length(components: &Value) -> Result<Option<usize>, String> {
    match components {
        Value::String(s) => Ok(Some(s.encode_utf16().count())),
        Value::Array(a) => Ok(Some(a.len())),
        Value::Undefined | Value::Null => Err(format!(
            "TypeError: Cannot read properties of {} (reading 'length')",
            if components.is_undefined() {
                "undefined"
            } else {
                "null"
            }
        )),
        _ => Ok(None),
    }
}

/// `entry.components[0]`.
fn first_component(components: &Value) -> Value {
    match components {
        Value::String(s) => s
            .chars()
            .next()
            .map(|c| Value::from(c.to_string()))
            .unwrap_or(Value::Undefined),
        Value::Array(a) => a.first().cloned().unwrap_or(Value::Undefined),
        _ => Value::Undefined,
    }
}

fn pack_uniforms_with_slots(uniforms: &Object, layout: &[SlotEntry]) -> Result<Vec<u8>, String> {
    // `Math.max(maxSlot, entry.slot)`: NaN is sticky.
    let mut max_slot = 0.0f64;
    for entry in layout {
        if max_slot.is_nan() || entry.slot.is_nan() {
            max_slot = f64::NAN;
        } else if entry.slot > max_slot {
            max_slot = entry.slot;
        }
    }
    let mut buffer = array_buffer((max_slot + 1.0) * 16.0)?;
    let mut view = DataView { bytes: &mut buffer };

    for entry in layout {
        let value = resolve_uniform_alias(&entry.name, uniforms, false);
        if value.is_nullish() {
            continue;
        }
        let slot_offset = entry.slot * 16.0;
        let Some(len) = components_length(&entry.components)? else {
            continue;
        };
        match len {
            1 => {
                let offset = slot_offset + component_offset(&entry.components);
                match &value {
                    Value::Bool(b) => view.set_f32(offset, if *b { 1.0 } else { 0.0 })?,
                    Value::Number(n) => view.set_f32(offset, *n as f32)?,
                    _ => {}
                }
            }
            2..=4 => {
                let offset = if len == 4 {
                    slot_offset
                } else {
                    slot_offset + component_offset(&first_component(&entry.components))
                };
                match &value {
                    Value::Array(items) => {
                        for (i, item) in items.iter().enumerate().take(len) {
                            view.set_f32(offset + (i * 4) as f64, to_f32(item))?;
                        }
                    }
                    Value::Number(n) => view.set_f32(offset, *n as f32)?,
                    _ => {}
                }
            }
            _ => {}
        }
    }
    Ok(buffer)
}

/// `packUniformsWithByteLayout`.
pub fn pack_uniforms_with_byte_layout(
    uniforms: &Object,
    layout: &ByteLayout,
) -> Result<Vec<u8>, String> {
    let mut total_size = layout.struct_size.unwrap_or(0.0);
    if total_size == 0.0 || total_size.is_nan() {
        total_size = 0.0;
        for entry in &layout.entries {
            let end = entry.offset + entry.size;
            total_size = if total_size.is_nan() || end.is_nan() {
                f64::NAN
            } else {
                total_size.max(end)
            };
        }
    }
    let buffer_size = (total_size / 16.0).ceil() * 16.0;
    let length = if buffer_size.is_nan() {
        f64::NAN
    } else {
        buffer_size.max(16.0)
    };
    let mut buffer = array_buffer(length)?;
    let mut view = DataView { bytes: &mut buffer };

    for entry in &layout.entries {
        let value = resolve_uniform_alias(&entry.name, uniforms, true);
        if value.is_nullish() {
            continue;
        }
        let ty = entry.ty.as_str().unwrap_or("");
        let offset = entry.offset;
        let store = |view: &mut DataView, at: f64, v: &Value| -> Result<(), String> {
            match ty {
                "int" => view.set_i32(at, noisemaker_dsl::js::to_int32(math_round(to_number(v)))),
                "uint" => view.set_u32(at, noisemaker_dsl::js::to_uint32(math_round(to_number(v)))),
                _ => view.set_f32(at, to_f32(v)),
            }
        };
        if matches!(entry.components, Value::Number(c) if c == 1.0) {
            match &value {
                Value::Bool(b) => {
                    if ty == "int" || ty == "uint" {
                        view.set_i32(offset, *b as i32)?;
                    } else {
                        view.set_f32(offset, if *b { 1.0 } else { 0.0 })?;
                    }
                }
                Value::Number(_) => store(&mut view, offset, &value)?,
                _ => {}
            }
        } else if let Value::Array(items) = &value {
            let limit = (items.len() as f64).min(to_number(&entry.components));
            let mut i = 0usize;
            while (i as f64) < limit {
                store(&mut view, offset + (i * 4) as f64, &items[i])?;
                i += 1;
            }
        } else if let Value::Number(_) = &value {
            store(&mut view, offset, &value)?;
        }
    }
    Ok(buffer)
}

/// The reusable `packUniforms` scratch buffer (`_uniformBufferData`). The reference
/// never clears it, so bytes a pack does not write (alignment padding, vec3 and
/// mat3 column tails, the tail up to 256 bytes) keep the values of earlier packs.
#[derive(Debug, Clone)]
pub struct PackScratch {
    buffer: Vec<u8>,
}

impl Default for PackScratch {
    fn default() -> Self {
        PackScratch {
            buffer: vec![0u8; 512],
        }
    }
}

fn align_to(offset: f64, alignment: f64) -> f64 {
    (offset / alignment).ceil() * alignment
}

impl PackScratch {
    /// `packUniforms`: std140-style packing of every defined uniform in key order;
    /// integral numbers become `i32` except `time`, `deltaTime` and `aspect`.
    /// Returns the used bytes (at least 256).
    pub fn pack_uniforms(&mut self, uniforms: &Object) -> Result<Vec<u8>, String> {
        let mut estimated = 0.0f64;
        for value in uniforms.values() {
            match value {
                Value::Undefined => continue,
                Value::Number(_) | Value::Bool(_) => estimated += 4.0,
                Value::Array(a) => estimated += a.len() as f64 * 4.0 + 12.0,
                _ => {}
            }
        }
        let buffer_size = 256f64.max(((estimated + 64.0) / 16.0).ceil() * 16.0);
        if buffer_size > self.buffer.len() as f64 {
            self.buffer = vec![0u8; buffer_size as usize];
        }
        let mut view = DataView {
            bytes: &mut self.buffer,
        };
        let mut offset = 0.0f64;
        for (name, value) in uniforms.iter() {
            match value {
                Value::Undefined | Value::Null => continue,
                Value::Bool(b) => {
                    offset = align_to(offset, 4.0);
                    view.set_i32(offset, *b as i32)?;
                    offset += 4.0;
                }
                Value::Number(n) => {
                    offset = align_to(offset, 4.0);
                    if value.is_integer()
                        && name != "time"
                        && name != "deltaTime"
                        && name != "aspect"
                    {
                        view.set_i32(offset, noisemaker_dsl::js::to_int32(*n))?;
                    } else {
                        view.set_f32(offset, *n as f32)?;
                    }
                    offset += 4.0;
                }
                Value::Array(items) => match items.len() {
                    2 => {
                        offset = align_to(offset, 8.0);
                        view.set_f32(offset, to_f32(&items[0]))?;
                        view.set_f32(offset + 4.0, to_f32(&items[1]))?;
                        offset += 8.0;
                    }
                    3 => {
                        offset = align_to(offset, 16.0);
                        for (i, item) in items.iter().enumerate() {
                            view.set_f32(offset + (i * 4) as f64, to_f32(item))?;
                        }
                        offset += 16.0;
                    }
                    4 => {
                        offset = align_to(offset, 16.0);
                        for (i, item) in items.iter().enumerate() {
                            view.set_f32(offset + (i * 4) as f64, to_f32(item))?;
                        }
                        offset += 16.0;
                    }
                    9 => {
                        offset = align_to(offset, 16.0);
                        for col in 0..3 {
                            for row in 0..3 {
                                view.set_f32(
                                    offset + (row * 4) as f64,
                                    to_f32(&items[col * 3 + row]),
                                )?;
                            }
                            offset += 16.0;
                        }
                    }
                    16 => {
                        offset = align_to(offset, 16.0);
                        for (i, item) in items.iter().enumerate() {
                            view.set_f32(offset + (i * 4) as f64, to_f32(item))?;
                        }
                        offset += 64.0;
                    }
                    _ => {
                        for item in items {
                            offset = align_to(offset, 4.0);
                            view.set_f32(offset, to_f32(item))?;
                            offset += 4.0;
                        }
                    }
                },
                _ => {}
            }
        }
        let used = 256f64.max(align_to(offset, 16.0)) as usize;
        if used > self.buffer.len() {
            return Err(range_error("Invalid typed array length"));
        }
        Ok(self.buffer[..used].to_vec())
    }
}

/// `createSingleUniformBuffer`'s data: the bytes written for one value, or `None`
/// when the value has no buffer form.
pub fn single_uniform_data(value: &Value, type_decl: &str) -> Option<Vec<u8>> {
    let f32s = |v: &[f32]| v.iter().flat_map(|f| f.to_le_bytes()).collect::<Vec<u8>>();
    match value {
        Value::Bool(b) => Some((*b as i32).to_le_bytes().to_vec()),
        Value::Number(n) => {
            if type_decl == "i32" || type_decl == "u32" {
                Some(
                    noisemaker_dsl::js::to_int32(math_round(*n))
                        .to_le_bytes()
                        .to_vec(),
                )
            } else {
                Some((*n as f32).to_le_bytes().to_vec())
            }
        }
        Value::Array(items) => {
            let get = |i: usize| to_f32(&items[i]);
            match items.len() {
                2 => Some(f32s(&[get(0), get(1)])),
                3 => Some(f32s(&[get(0), get(1), get(2), 0.0])),
                4 => Some(f32s(&[get(0), get(1), get(2), get(3)])),
                9 => {
                    let mut m = [0f32; 12];
                    for col in 0..3 {
                        for row in 0..3 {
                            m[col * 4 + row] = get(col * 3 + row);
                        }
                    }
                    Some(f32s(&m))
                }
                len => {
                    if JsRegex::new(r"^array<\s*vec4", "").test(type_decl) {
                        let count = JsRegex::new(r"^array<[^,>]+(?:<[^>]+>)?\s*,\s*(\d+)\s*>", "")
                            .exec(type_decl)
                            .map(|m| {
                                noisemaker_dsl::js::parse_int(group(&m, 1).unwrap(), 10) as usize
                            })
                            .unwrap_or_else(|| len.div_ceil(4));
                        let mut flat = vec![0f32; count * 4];
                        for (i, slot) in flat.iter_mut().enumerate().take(len) {
                            *slot = get(i);
                        }
                        Some(f32s(&flat))
                    } else {
                        Some(f32s(&(0..len).map(get).collect::<Vec<_>>()))
                    }
                }
            }
        }
        _ => None,
    }
}

/// The default value `createBindGroup` substitutes for a missing or non-numeric
/// individual uniform.
pub fn default_single_uniform(type_decl: &str) -> Value {
    if type_decl == "i32" || type_decl == "u32" {
        return Value::Number(0.0);
    }
    let zeros = |n: usize| Value::Array(vec![Value::Number(0.0); n]);
    if type_decl.starts_with("vec2") {
        return zeros(2);
    }
    if type_decl.starts_with("vec3") {
        return zeros(3);
    }
    if type_decl.starts_with("vec4") {
        return zeros(4);
    }
    if type_decl.starts_with("array<") {
        let re = JsRegex::new(r"^array<\s*([^,>]+(?:<[^>]+>)?)\s*,\s*(\d+)\s*>", "");
        return match re.exec(type_decl) {
            Some(m) => {
                let elem_type = noisemaker_dsl::js::trim(group(&m, 1).unwrap());
                let count = noisemaker_dsl::js::parse_int(group(&m, 2).unwrap(), 10);
                let stride = if elem_type == "f32" || elem_type == "i32" || elem_type == "u32" {
                    4.0
                } else if elem_type.starts_with("vec2") {
                    8.0
                } else {
                    16.0
                };
                zeros(((count * stride) / 4.0) as usize)
            }
            None => Value::Number(0.0),
        };
    }
    if type_decl.starts_with("mat3") {
        return Value::Array(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                .into_iter()
                .map(Value::Number)
                .collect(),
        );
    }
    Value::Number(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgsl::normalize_spec_layout;

    fn obj(json: &str) -> Object {
        Value::from_json(json).unwrap().as_object().unwrap().clone()
    }

    fn f32_at(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }

    #[test]
    fn slot_layout_packs_components() {
        let layout = normalize_spec_layout(
            &Value::from_json(
                r#"{"resolution":{"slot":0,"components":"xy"},"time":{"slot":0,"components":"z"},
                    "seed":{"slot":1,"components":"z"},"wrap":{"slot":1,"components":"w"},
                    "color":{"slot":2,"components":"xyz"}}"#,
            )
            .unwrap(),
        );
        let uniforms =
            obj(r#"{"resolution":[256,128],"time":0.25,"seed":7,"wrap":true,"color":[0.5,1,2,9]}"#);
        let bytes = pack_uniforms_with_layout(&uniforms, &layout).unwrap();
        assert_eq!(bytes.len(), 48);
        assert_eq!(f32_at(&bytes, 0), 256.0);
        assert_eq!(f32_at(&bytes, 4), 128.0);
        assert_eq!(f32_at(&bytes, 8), 0.25);
        assert_eq!(f32_at(&bytes, 24), 7.0);
        assert_eq!(f32_at(&bytes, 28), 1.0);
        assert_eq!(f32_at(&bytes, 32), 0.5);
        assert_eq!(f32_at(&bytes, 40), 2.0);
        assert_eq!(f32_at(&bytes, 44), 0.0);
    }

    #[test]
    fn slot_layout_aliases_width_height_channels() {
        let layout = normalize_spec_layout(
            &Value::from_json(
                r#"[{"name":"width","slot":0,"components":"x"},{"name":"height","slot":0,"components":"y"},
                    {"name":"channels","slot":0,"components":"z"}]"#,
            )
            .unwrap(),
        );
        let bytes = pack_uniforms_with_layout(&obj(r#"{"resolution":[64,32]}"#), &layout).unwrap();
        assert_eq!(
            (f32_at(&bytes, 0), f32_at(&bytes, 4), f32_at(&bytes, 8)),
            (64.0, 32.0, 4.0)
        );
    }

    #[test]
    fn byte_layout_rounds_ints() {
        let layout = ByteLayout {
            entries: vec![
                crate::wgsl::ByteEntry {
                    name: "count".into(),
                    offset: 0.0,
                    size: 4.0,
                    ty: "int".into(),
                    components: 1u32.into(),
                },
                crate::wgsl::ByteEntry {
                    name: "dir".into(),
                    offset: 8.0,
                    size: 8.0,
                    ty: "float".into(),
                    components: 2u32.into(),
                },
            ],
            struct_size: Some(16.0),
        };
        let bytes = pack_uniforms_with_byte_layout(&obj(r#"{"count":2.5,"dir":[1,2,3]}"#), &layout)
            .unwrap();
        assert_eq!(bytes.len(), 16);
        assert_eq!(i32::from_le_bytes(bytes[0..4].try_into().unwrap()), 3);
        assert_eq!((f32_at(&bytes, 8), f32_at(&bytes, 12)), (1.0, 2.0));
    }

    #[test]
    fn pack_uniforms_is_std140_and_keeps_stale_padding() {
        let mut scratch = PackScratch::default();
        let first = scratch
            .pack_uniforms(&obj(r#"{"a":[1,2,3,4],"time":1}"#))
            .unwrap();
        assert_eq!(first.len(), 256);
        assert_eq!(f32_at(&first, 12), 4.0);
        assert_eq!(f32_at(&first, 16), 1.0); // `time` stays float
        // A vec3 leaves the fourth float untouched: the stale 4.0 survives.
        let second = scratch
            .pack_uniforms(&obj(r#"{"v":[5,6,7],"n":3}"#))
            .unwrap();
        assert_eq!(f32_at(&second, 8), 7.0);
        assert_eq!(f32_at(&second, 12), 4.0);
        assert_eq!(i32::from_le_bytes(second[16..20].try_into().unwrap()), 3);
    }

    #[test]
    fn single_uniform_buffers() {
        assert_eq!(
            single_uniform_data(&Value::Number(2.6), "i32").unwrap(),
            3i32.to_le_bytes()
        );
        assert_eq!(
            single_uniform_data(&Value::from_json("[1,2,3]").unwrap(), "vec3<f32>")
                .unwrap()
                .len(),
            16
        );
        let arr = default_single_uniform("array<vec4<f32>, 4>");
        assert_eq!(arr.as_array().unwrap().len(), 16);
        let bytes = single_uniform_data(&arr, "array<vec4<f32>, 4>").unwrap();
        assert_eq!(bytes.len(), 64);
        assert_eq!(
            default_single_uniform("mat3x3<f32>")
                .as_array()
                .unwrap()
                .len(),
            9
        );
    }
}
