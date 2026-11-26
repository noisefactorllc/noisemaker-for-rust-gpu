//! Port of `runtime/resources.js`: liveness analysis and texture pooling
//! (linear-scan register allocation of physical textures to the virtual textures
//! the expanded passes read and write).

use indexmap::IndexMap;
use std::collections::HashSet;

use crate::error::JsError;
use crate::expander::{js_values, member, starts_with};
use crate::js::same_value_zero;
use crate::unparser::jsv::to_string;
use crate::value::{Object, Value};

/// The pass interval during which a virtual texture is live:
/// `{ start, end }` of `analyzeLiveness`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lifetime {
    pub start: usize,
    pub end: usize,
}

fn build_input_aliases(passes: &[Value]) -> Result<IndexMap<String, String>, JsError> {
    let mut written = HashSet::new();
    for pass in passes {
        let outputs = member(pass, "outputs")?;
        if outputs.is_truthy() {
            for texture in js_values(outputs) {
                if let Value::String(id) = texture {
                    written.insert(id);
                }
            }
        }
    }
    let mut aliases = IndexMap::new();
    for (index, pass) in passes.iter().enumerate() {
        let inputs = member(pass, "inputs")?;
        let node = member(pass, "nodeId")?;
        if !inputs.is_truthy() || !node.is_truthy() {
            continue;
        }
        let node = to_string(node)?;
        let own_output = format!("{node}_out");
        for texture in js_values(inputs) {
            let Value::String(id) = texture else { continue };
            if written.contains(&id)
                || aliases.contains_key(&id)
                || !["inputTex", "inputColor"]
                    .iter()
                    .any(|name| id == format!("{node}_{name}"))
            {
                continue;
            }
            let mut upstream = None;
            for previous in passes[..index].iter().rev() {
                let outputs = member(previous, "outputs")?;
                if !outputs.is_truthy() {
                    continue;
                }
                for output in js_values(outputs) {
                    let Value::String(output) = output else {
                        continue;
                    };
                    if output == own_output {
                        continue;
                    }
                    if output
                        .strip_prefix("node_")
                        .and_then(|s| s.strip_suffix("_out"))
                        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    {
                        upstream = Some(output);
                        break;
                    }
                }
                if upstream.is_some() {
                    break;
                }
            }
            if let Some(upstream) = upstream {
                aliases.insert(id, upstream);
            }
        }
    }
    Ok(aliases)
}

fn resolve_input(aliases: &IndexMap<String, String>, input: Value) -> Value {
    match input.as_str().and_then(|id| aliases.get(id)) {
        Some(upstream) => upstream.clone().into(),
        None => input,
    }
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
    let aliases = build_input_aliases(passes)?;
    for (index, pass) in passes.iter().enumerate() {
        let inputs = member(pass, "inputs")?;
        if inputs.is_truthy() {
            for tex in js_values(inputs) {
                touch(&mut lifetime, &resolve_input(&aliases, tex), index)?;
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
    let aliases = build_input_aliases(passes)?;
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

        // 2. Release inputs (last uses). A pass may read one texture under
        // several names (lighting's inputTex and heightMap); release it once
        // (`new Set(Object.values(pass.inputs))`), or its slot is handed to two
        // textures that are live together.
        let inputs = member(pass, "inputs")?;
        if inputs.is_truthy() {
            let mut unique: Vec<Value> = Vec::new();
            for tex_id in js_values(inputs) {
                let tex_id = resolve_input(&aliases, tex_id);
                if !unique.iter().any(|seen| same_value_zero(seen, &tex_id)) {
                    unique.push(tex_id);
                }
            }
            for tex_id in unique {
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
    fn unwritten_pipeline_input_alias_keeps_its_upstream_output_live() {
        for input in ["inputTex", "inputColor"] {
            let alias = format!("node_1_{input}");
            let passes = vec![
                js!({"nodeId": "node_0", "outputs": {"color": "node_0_out"}}),
                js!({"nodeId": "node_1", "inputs": {"input": (alias.clone())}, "outputs": {"color": "node_1_out"}}),
                js!({"nodeId": "node_2", "inputs": {"input": "node_1_out"}, "outputs": {"color": "node_2_out"}}),
            ];
            let live = analyze_liveness(&passes).unwrap();
            assert_eq!(live["node_0_out"], Lifetime { start: 0, end: 1 });
            assert!(!live.contains_key(&alias));
            let allocated = allocate_resources(&passes).unwrap();
            assert_eq!(allocated.get("node_0_out"), allocated.get("node_2_out"));
            assert_ne!(allocated.get("node_0_out"), allocated.get("node_1_out"));
        }
    }

    #[test]
    fn written_node_input_is_a_real_texture() {
        let passes = vec![
            js!({"nodeId": "node_0", "outputs": {"color": "node_0_out"}}),
            js!({"nodeId": "node_1", "outputs": {"color": "node_1_inputColor"}}),
            js!({"nodeId": "node_1", "inputs": {"inputColor": "node_1_inputColor"}, "outputs": {"color": "node_1_out"}}),
        ];
        let live = analyze_liveness(&passes).unwrap();
        assert_eq!(live["node_0_out"], Lifetime { start: 0, end: 0 });
        assert_eq!(live["node_1_inputColor"], Lifetime { start: 1, end: 2 });
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
    fn a_texture_read_under_two_names_in_one_pass_is_released_once() {
        // Upstream 8e583593: lighting reads the same texture as inputTex and
        // heightMap. Releasing it once per name put its slot on the free list
        // twice, so C and D, live together until the last pass, shared it.
        let passes = vec![
            pass(js!({}), js!({"out": "A"})),
            pass(js!({"inputTex": "A", "heightMap": "A"}), js!({"out": "B"})),
            pass(js!({"inputTex": "B"}), js!({"out": "C"})),
            pass(js!({"inputTex": "B"}), js!({"out": "D"})),
            pass(js!({"a": "C", "b": "D"}), js!({"out": "E"})),
        ];
        let alloc = allocate_resources(&passes).unwrap();
        assert_eq!(
            alloc.get("C"),
            alloc.get("A"),
            "C reuses the slot A released"
        );
        assert_ne!(alloc.get("D"), alloc.get("C"), "C and D are live together");
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
