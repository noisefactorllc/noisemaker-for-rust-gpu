//! The compiled render graph the pipeline executes.
//!
//! `compileGraph` (reference `runtime/compiler.js`) produces
//! `{id, source, passes, programs, allocations, textures, renderSurface, mediaSteps}`
//! with `allocations` and `textures` as `Map`s. Passes and texture specs stay
//! JavaScript objects ([`Object`]): the reference reads and writes them as
//! open-ended records (the oscillator proxy copies a fixed field list, uniform
//! objects keep `undefined`-valued keys, dimension specs are numbers, strings or
//! objects), and every field it passes through must stay reachable.

use indexmap::IndexMap;
use noisemaker_dsl::{Object, Value};

/// A render graph in the reference shape.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    /// `graph.id` (the source hash).
    pub id: Value,
    /// `graph.source`.
    pub source: Value,
    /// `graph.passes`, each the reference pass object.
    pub passes: Vec<Object>,
    /// `graph.programs`: program id → program spec (`{wgsl, uniformLayout, defines,
    /// fragmentEntryPoint, vertexWGSL, ...}`).
    pub programs: IndexMap<String, Value>,
    /// `graph.allocations` (a `Map` virtual texture id → physical id), when present.
    pub allocations: Option<IndexMap<String, Value>>,
    /// `graph.textures` (a `Map` texture id → spec), when present.
    pub textures: Option<IndexMap<String, Object>>,
    /// `graph.renderSurface`.
    pub render_surface: Value,
    /// `graph.mediaSteps`.
    pub media_steps: Value,
    /// Every other graph member.
    pub extra: Object,
}

fn ordered_map(v: &Value) -> Option<IndexMap<String, Value>> {
    match v {
        Value::Object(o) => Some(o.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
        // A Map serialized as entry pairs.
        Value::Array(entries) => Some(
            entries
                .iter()
                .filter_map(|e| match e {
                    Value::Array(pair) if pair.len() == 2 => {
                        Some((crate::jsv::to_js_string(&pair[0]), pair[1].clone()))
                    }
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

impl Graph {
    /// Build a graph from the reference's JSON shape (`Map`s serialized as objects,
    /// as `parity/batch-golden.mjs` writes `<name>.graph.json`).
    pub fn from_value(v: &Value) -> Result<Graph, String> {
        let obj = v
            .as_object()
            .ok_or_else(|| "a graph must be an object".to_owned())?;
        let mut graph = Graph::default();
        for (key, value) in obj.iter() {
            match key.as_str() {
                "id" => graph.id = value.clone(),
                "source" => graph.source = value.clone(),
                "passes" => {
                    let passes = value
                        .as_array()
                        .ok_or_else(|| "graph.passes must be an array".to_owned())?;
                    graph.passes = passes
                        .iter()
                        .map(|p| {
                            p.as_object()
                                .cloned()
                                .ok_or_else(|| "every pass must be an object".to_owned())
                        })
                        .collect::<Result<_, _>>()?;
                }
                "programs" => {
                    graph.programs = ordered_map(value).unwrap_or_default();
                }
                "allocations" => graph.allocations = ordered_map(value),
                "textures" => {
                    graph.textures = ordered_map(value).map(|m| {
                        m.into_iter()
                            .filter_map(|(k, v)| v.as_object().cloned().map(|o| (k, o)))
                            .collect()
                    });
                }
                "renderSurface" => graph.render_surface = value.clone(),
                "mediaSteps" => graph.media_steps = value.clone(),
                _ => {
                    graph.extra.insert(key.clone(), value.clone());
                }
            }
        }
        Ok(graph)
    }

    /// Parse a graph from JSON text.
    pub fn from_json(text: &str) -> Result<Graph, String> {
        let v = Value::from_json(text).map_err(|e| e.to_string())?;
        Graph::from_value(&v)
    }

    /// Parse a graph the reference page serialized with `JSON.stringify`
    /// (`parity/batch-golden.mjs`'s `<name>.graph.json`). `JSON.stringify` writes
    /// NaN as `null`, and the only `null` uniform values a rendered graph holds are
    /// NaNs: the host converts function-valued parameters with
    /// `parseFloat`/`parseInt` (`convertParameterForUniform`). Uniform values (and
    /// their array elements) that are `null` therefore decode as NaN.
    pub fn from_reference_json(text: &str) -> Result<Graph, String> {
        let mut graph = Graph::from_json(text)?;
        for p in &mut graph.passes {
            if let Some(uniforms) = pass::uniforms_mut(p) {
                for value in uniforms.values_mut() {
                    match value {
                        Value::Null => *value = Value::Number(f64::NAN),
                        Value::Array(items) => {
                            for item in items.iter_mut() {
                                if item.is_null() {
                                    *item = Value::Number(f64::NAN);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(graph)
    }

    /// The graph back in its JSON shape (`Map`s as objects).
    pub fn to_value(&self) -> Value {
        let mut o = Object::new();
        o.insert("id", self.id.clone());
        o.insert("source", self.source.clone());
        o.insert(
            "passes",
            Value::Array(self.passes.iter().cloned().map(Value::Object).collect()),
        );
        o.insert(
            "programs",
            Value::Object(
                self.programs
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            ),
        );
        if let Some(a) = &self.allocations {
            o.insert(
                "allocations",
                Value::Object(a.iter().map(|(k, v)| (k.clone(), v.clone())).collect()),
            );
        }
        if let Some(t) = &self.textures {
            o.insert(
                "textures",
                Value::Object(
                    t.iter()
                        .map(|(k, v)| (k.clone(), Value::Object(v.clone())))
                        .collect(),
                ),
            );
        }
        o.insert("renderSurface", self.render_surface.clone());
        o.insert("mediaSteps", self.media_steps.clone());
        for (k, v) in self.extra.iter() {
            o.insert(k.clone(), v.clone());
        }
        Value::Object(o)
    }

    /// `graph.renderSurface` when it is a non-empty string.
    pub fn render_surface_name(&self) -> Option<&str> {
        self.render_surface.as_str().filter(|s| !s.is_empty())
    }
}

/// Typed reads of the pass fields the runtime uses (JavaScript member semantics:
/// absent members read as `undefined`).
pub mod pass {
    use noisemaker_dsl::{Object, Value};

    /// `pass[key]`.
    pub fn get<'a>(pass: &'a Object, key: &str) -> &'a Value {
        pass.get_or_undefined(key)
    }

    /// `pass.id` as text (for messages).
    pub fn id(pass: &Object) -> String {
        crate::jsv::interpolate(get(pass, "id"))
    }

    /// `pass.program` as a property key (`programs.get(pass.program)`).
    pub fn program(pass: &Object) -> String {
        crate::jsv::to_js_string(get(pass, "program"))
    }

    /// `pass.inputs` when it is an object.
    pub fn inputs(pass: &Object) -> Option<&Object> {
        get(pass, "inputs").as_object()
    }

    /// `pass.outputs` when it is an object.
    pub fn outputs(pass: &Object) -> Option<&Object> {
        get(pass, "outputs").as_object()
    }

    /// `pass.uniforms` when it is an object.
    pub fn uniforms(pass: &Object) -> Option<&Object> {
        get(pass, "uniforms").as_object()
    }

    /// `pass.uniforms` (mutable) when it is an object.
    pub fn uniforms_mut(pass: &mut Object) -> Option<&mut Object> {
        pass.get_mut("uniforms").and_then(Value::as_object_mut)
    }

    /// `pass.drawMode` as a string.
    pub fn draw_mode(pass: &Object) -> Option<&str> {
        get(pass, "drawMode").as_str()
    }
}
