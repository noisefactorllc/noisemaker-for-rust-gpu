//! Port of `runtime/resources.js`: liveness analysis and texture pooling
//! (linear-scan register allocation of physical textures to the virtual textures
//! the expanded passes read and write).

use indexmap::IndexMap;

use crate::error::JsError;
use crate::expander::{js_values, member, starts_with};
use crate::value::{Object, Value};

/// The pass interval during which a virtual texture is live:
/// `{ start, end }` of `analyzeLiveness`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub start: usize,
    pub end: usize,
}

/// `touch(texId, index)` of `analyzeLiveness`.
fn touch(
    lifetime: &mut IndexMap<String, Lifetime>,
    tex_id: &Value,
    index: usize,
) -> Result<(), JsError> {
    if !tex_id.is_truthy() {
        return Ok(());
    }
    // Ignore globals for liveness analysis (they are infinite).
    if starts_with(tex_id, "texId", "global_")? {
        return Ok(());
    }
    let Value::String(id) = tex_id else {
        unreachable!("starts_with accepts strings only")
    };
    match lifetime.get_mut(id) {
        None => {
            lifetime.insert(
                id.clone(),
                Lifetime {
                    start: index,
                    end: index,
                },
            );
        }
        Some(l) => {
            l.start = l.start.min(index);
            l.end = l.end.max(index);
        }
    }
    Ok(())
}

/// `analyzeLiveness(passes)`: `Map<virtualId, {start, end}>` in first-use order.
/// Inputs are read and outputs written at the pass index; `global_` textures
/// (and empty bindings) are not tracked.
pub fn analyze_liveness(passes: &[Value]) -> Result<IndexMap<String, Lifetime>, JsError> {
    let mut lifetime = IndexMap::new();
    for (index, pass) in passes.iter().enumerate() {
        let inputs = member(pass, "inputs")?;
        if inputs.is_truthy() {
            for tex in js_values(inputs) {
                touch(&mut lifetime, &tex, index)?;
            }
        }
        let outputs = member(pass, "outputs")?;
        if outputs.is_truthy() {
            for tex in js_values(outputs) {
                touch(&mut lifetime, &tex, index)?;
            }
        }
    }
    Ok(lifetime)
}

/// A released physical slot: `{ id, availableAfter }`.
struct FreeSlot {
    id: String,
    available_after: usize,
}

/// `allocateResources(passes)`: `Map<virtualId, physicalId>` (as an ordered
/// object, in allocation order). Each pass first allocates its outputs (reusing
/// a slot released by a strictly earlier pass, else a new `phys_N`), then
/// releases the inputs whose lifetime ends at this pass.
pub fn allocate_resources(passes: &[Value]) -> Result<Object, JsError> {
    let lifetime = analyze_liveness(passes)?;
    let mut allocations = Object::new();
    let mut free_list: Vec<FreeSlot> = Vec::new();
    let mut physical_count = 0usize;

    for (i, pass) in passes.iter().enumerate() {
        // 1. Allocate outputs (definitions).
        let outputs = member(pass, "outputs")?;
        if outputs.is_truthy() {
            for tex_id in js_values(outputs) {
                if starts_with(&tex_id, "texId", "global_")? {
                    continue; // Globals are pre-allocated.
                }
                let Value::String(tex_id) = tex_id else {
                    unreachable!("starts_with accepts strings only")
                };
                if allocations.contains_key(&tex_id) {
                    continue;
                }
                // A slot is free if it was released in a strictly previous pass.
                match free_list.iter().position(|item| item.available_after < i) {
                    Some(free_idx) => {
                        let item = free_list.remove(free_idx);
                        allocations.insert(tex_id, Value::String(item.id));
                    }
                    None => {
                        let id = format!("phys_{physical_count}");
                        physical_count += 1;
                        allocations.insert(tex_id, Value::String(id));
                    }
                }
            }
        }

        // 2. Release inputs (last uses).
        let inputs = member(pass, "inputs")?;
        if inputs.is_truthy() {
            for tex_id in js_values(inputs) {
                if starts_with(&tex_id, "texId", "global_")? {
                    continue;
                }
                let Value::String(tex_id) = tex_id else {
                    unreachable!("starts_with accepts strings only")
                };
                if let Some(l) = lifetime.get(&tex_id)
                    && l.end == i
                    && let Some(Value::String(phys_id)) = allocations.get(&tex_id)
                    && !phys_id.is_empty()
                {
                    // It becomes available after this pass is done.
                    free_list.push(FreeSlot {
                        id: phys_id.clone(),
                        available_after: i,
                    });
                }
            }
        }
    }
    Ok(allocations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::js;

    fn pass(inputs: Value, outputs: Value) -> Value {
        js!({"inputs": inputs, "outputs": outputs})
    }

    #[test]
    fn pooling_reuses_released_slots() {
        // a -> b -> c -> d: a is released after pass 1 and reused by pass 2's output.
        let passes = vec![
            pass(js!({}), js!({"color": "a"})),
            pass(js!({"src": "a"}), js!({"color": "b"})),
            pass(js!({"src": "b"}), js!({"color": "c"})),
            pass(js!({"src": "c"}), js!({"color": "global_o0"})),
        ];
        let live = analyze_liveness(&passes).unwrap();
        assert_eq!(live["a"], Lifetime { start: 0, end: 1 });
        assert_eq!(live["c"], Lifetime { start: 2, end: 3 });
        assert!(!live.contains_key("global_o0"));
        let alloc = allocate_resources(&passes).unwrap();
        assert_eq!(
            Value::Object(alloc).to_json().unwrap(),
            r#"{"a":"phys_0","b":"phys_1","c":"phys_0"}"#
        );
    }

    #[test]
    fn unbound_inputs_throw_like_the_reference() {
        // An undefined input is skipped by the liveness analysis but read with
        // `texId.startsWith` by the allocator.
        let mut inputs = Object::new();
        inputs.insert("tex", Value::Undefined);
        let passes = vec![pass(Value::Object(inputs), js!({"color": "x"}))];
        assert_eq!(
            allocate_resources(&passes),
            Err(JsError::type_error(
                "Cannot read properties of undefined (reading 'startsWith')"
            ))
        );
        let passes = vec![pass(js!({"tex": 3}), js!({"color": "x"}))];
        assert_eq!(
            analyze_liveness(&passes),
            Err(JsError::type_error("texId.startsWith is not a function"))
        );
    }
}
