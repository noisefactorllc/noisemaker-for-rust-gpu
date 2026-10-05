//! A pure-data [`ProgramHost`]: the mock renderers of the reference tests
//! (`Object.create(CanvasRenderer.prototype)` with a `_pipeline = { graph,
//! broadcastChainScopedParam() {} }`, or `{ pipeline, currentDsl,
//! convertParameterForUniform(v) { return v } }`) as data.
//!
//! The pipeline holds a compiled graph (`compileGraph` output) whose pass
//! uniforms ProgramState writes, and records every pipeline call ProgramState
//! makes, with its arguments, in call order. The calls do nothing else: the
//! side effects of a real pipeline (chain broadcasts, texture reallocation,
//! asyncInit regeneration) belong to the GPU host.

use std::rc::Rc;

use super::host::{ProgramHost, convert_parameter_for_uniform};
use crate::JsError;
use crate::unparser::jsv::entries;
use crate::value::{Object, Value};

/// How the mock renderer provides `convertParameterForUniform`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockConvert {
    /// CanvasRenderer's conversion over the mock's enum tree.
    Canvas,
    /// `convertParameterForUniform(value) { return value }`.
    Passthrough,
    /// No `convertParameterForUniform` member.
    Absent,
}

/// The mock renderer.
#[derive(Debug, Clone)]
pub struct MockHost {
    /// `renderer.currentDsl`.
    pub current_dsl: String,
    /// `renderer.enums` (and the tree CanvasRenderer's conversion resolves in).
    pub enums: Rc<Value>,
    /// The renderer's `convertParameterForUniform`.
    pub convert: MockConvert,
    /// `renderer.pipeline` (`None`: no pipeline).
    pub pipeline: Option<MockPipeline>,
}

/// The mock pipeline: a graph and the calls made on it.
#[derive(Debug, Clone)]
pub struct MockPipeline {
    /// `pipeline.graph` (`{ passes: [...], ... }`).
    pub graph: Value,
    /// Which optional pipeline members exist.
    pub methods: MockMethods,
    /// Every call, in order: `{ fn, ...arguments }`.
    pub calls: Vec<Value>,
}

/// Presence of the pipeline members ProgramState calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockMethods {
    pub broadcast_chain_scoped_param: bool,
    pub check_async_regen: bool,
    pub recreate_textures: bool,
    pub collect_default_uniforms: bool,
    pub set_uniform: bool,
}

impl MockMethods {
    /// Every member present (a `Pipeline`).
    pub const ALL: MockMethods = MockMethods {
        broadcast_chain_scoped_param: true,
        check_async_regen: true,
        recreate_textures: true,
        collect_default_uniforms: true,
        set_uniform: true,
    };

    /// No member present (`{ graph }`).
    pub const NONE: MockMethods = MockMethods {
        broadcast_chain_scoped_param: false,
        check_async_regen: false,
        recreate_textures: false,
        collect_default_uniforms: false,
        set_uniform: false,
    };

    /// The members named in `names` (`broadcastChainScopedParam`,
    /// `checkAsyncRegen`, `recreateTextures`, `collectDefaultUniforms`,
    /// `setUniform`).
    pub fn from_names<S: AsRef<str>>(names: &[S]) -> MockMethods {
        let has = |n: &str| names.iter().any(|s| s.as_ref() == n);
        MockMethods {
            broadcast_chain_scoped_param: has("broadcastChainScopedParam"),
            check_async_regen: has("checkAsyncRegen"),
            recreate_textures: has("recreateTextures"),
            collect_default_uniforms: has("collectDefaultUniforms"),
            set_uniform: has("setUniform"),
        }
    }
}

impl MockPipeline {
    pub fn new(graph: Value, methods: MockMethods) -> Self {
        MockPipeline {
            graph,
            methods,
            calls: Vec::new(),
        }
    }

    fn record(&mut self, name: &str, args: Vec<(&str, Value)>) {
        let mut call = Object::new();
        call.insert("fn", Value::from(name));
        for (k, v) in args {
            call.insert(k, v);
        }
        self.calls.push(Value::Object(call));
    }
}

impl MockHost {
    /// A renderer without pipeline, DSL or enums, converting like CanvasRenderer.
    pub fn new() -> Self {
        MockHost {
            current_dsl: String::new(),
            enums: Rc::new(Value::Undefined),
            convert: MockConvert::Canvas,
            pipeline: None,
        }
    }

    /// Take the recorded pipeline calls.
    pub fn take_calls(&mut self) -> Vec<Value> {
        self.pipeline
            .as_mut()
            .map(|p| std::mem::take(&mut p.calls))
            .unwrap_or_default()
    }

    fn pipeline_mut(&mut self, member: &str) -> Result<&mut MockPipeline, JsError> {
        self.pipeline.as_mut().ok_or_else(|| {
            JsError::type_error(format!(
                "Cannot read properties of null (reading '{member}')"
            ))
        })
    }
}

impl Default for MockHost {
    fn default() -> Self {
        Self::new()
    }
}

/// `collectDefaultUniforms()` of the reference Pipeline: every pass's uniforms
/// merged in pass order (`Object.assign`).
pub fn collect_default_uniforms(graph: &Value) -> Object {
    let mut uniforms = Object::new();
    if let Value::Array(passes) = graph.get("passes") {
        for pass in passes {
            let u = pass.get("uniforms");
            if u.is_truthy() {
                for (k, v) in entries(u) {
                    uniforms.insert(k, v);
                }
            }
        }
    }
    uniforms
}

impl ProgramHost for MockHost {
    fn current_dsl(&self) -> String {
        self.current_dsl.clone()
    }

    fn enums(&self) -> Rc<Value> {
        self.enums.clone()
    }

    fn has_convert_parameter_for_uniform(&self) -> bool {
        self.convert != MockConvert::Absent
    }

    fn convert_parameter_for_uniform(&self, value: &Value, spec: &Value) -> Result<Value, JsError> {
        match self.convert {
            MockConvert::Passthrough => Ok(value.clone()),
            _ => convert_parameter_for_uniform(value, spec, &self.enums),
        }
    }

    fn graph_passes(&mut self) -> Option<&mut Vec<Value>> {
        let pipeline = self.pipeline.as_mut()?;
        match pipeline.graph.get_mut("passes") {
            Some(Value::Array(passes)) => Some(passes),
            _ => None,
        }
    }

    fn has_broadcast_chain_scoped_param(&self) -> bool {
        self.pipeline
            .as_ref()
            .is_some_and(|p| p.methods.broadcast_chain_scoped_param)
    }

    fn broadcast_chain_scoped_param(
        &mut self,
        pass_index: usize,
        uniform_name: &str,
        scoped_name: &str,
    ) -> Result<(), JsError> {
        let p = self.pipeline_mut("broadcastChainScopedParam")?;
        p.record(
            "broadcastChainScopedParam",
            vec![
                ("pass", Value::from(pass_index)),
                ("uniformName", Value::from(uniform_name)),
                ("scopedName", Value::from(scoped_name)),
            ],
        );
        Ok(())
    }

    fn has_check_async_regen(&self) -> bool {
        self.pipeline
            .as_ref()
            .is_some_and(|p| p.methods.check_async_regen)
    }

    fn check_async_regen(
        &mut self,
        node_id: &Value,
        effect_key: &Value,
        step_values: &Object,
    ) -> Result<(), JsError> {
        let p = self.pipeline_mut("checkAsyncRegen")?;
        p.record(
            "checkAsyncRegen",
            vec![
                ("nodeId", node_id.clone()),
                ("effectKey", effect_key.clone()),
                ("stepValues", Value::Object(step_values.clone())),
            ],
        );
        Ok(())
    }

    fn has_recreate_textures(&self) -> bool {
        self.pipeline
            .as_ref()
            .is_some_and(|p| p.methods.recreate_textures)
    }

    fn has_collect_default_uniforms(&self) -> bool {
        self.pipeline
            .as_ref()
            .is_some_and(|p| p.methods.collect_default_uniforms)
    }

    fn collect_default_uniforms(&mut self) -> Result<Object, JsError> {
        let p = self.pipeline_mut("collectDefaultUniforms")?;
        p.record("collectDefaultUniforms", vec![]);
        Ok(collect_default_uniforms(&p.graph))
    }

    fn recreate_textures(&mut self, uniforms: Object) -> Result<(), JsError> {
        let p = self.pipeline_mut("recreateTextures")?;
        p.record(
            "recreateTextures",
            vec![("uniforms", Value::Object(uniforms))],
        );
        Ok(())
    }

    fn has_set_uniform(&self) -> bool {
        self.pipeline
            .as_ref()
            .is_some_and(|p| p.methods.set_uniform)
    }

    fn set_uniform(&mut self, name: &str, value: &Value) -> Result<(), JsError> {
        let p = self.pipeline_mut("setUniform")?;
        p.record(
            "setUniform",
            vec![("name", Value::from(name)), ("value", value.clone())],
        );
        Ok(())
    }
}
