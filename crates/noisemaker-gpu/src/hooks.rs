//! Extension points for host-side effect behavior.
//!
//! The reference pipeline asks the effect registry (`getEffect(effectKey)`) for
//! three kinds of native behavior: the production lifecycle hooks
//! (`onInit`/`onUpdate`/`onDestroy`, GAP-026; e.g. `synth/media`), CPU-generated
//! overlay textures (`asyncInit`; the fibers/scratches/strayHair tracers), and the
//! effect's parameter table (`globals`, consulted by `checkAsyncRegen`). Native
//! ports of those effects register here; an empty registry is exactly the
//! reference running effects that define none of them.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use noisemaker_dsl::{Object, Value};

/// The `onUpdate` context (`{time, delta, uniforms}`).
pub struct UpdateContext<'a> {
    pub time: f64,
    pub delta: f64,
    /// The pipeline's global uniforms.
    pub uniforms: &'a Object,
}

/// An effect's lifecycle hooks. `has_*` report whether the effect really defines
/// a hook (the reference ignores base-class no-ops).
pub trait EffectLifecycle {
    fn has_on_init(&self) -> bool {
        false
    }
    fn on_init(&mut self) {}
    fn has_on_update(&self) -> bool {
        false
    }
    /// Per-frame uniforms bound for keys the pass does not resolve itself.
    fn on_update(&mut self, _context: &UpdateContext<'_>) -> Option<Object> {
        None
    }
    fn has_on_destroy(&self) -> bool {
        false
    }
    fn on_destroy(&mut self) -> Result<(), String> {
        Ok(())
    }
}

/// What an `asyncInit` effect may do (`context` of `_startAsyncInit`).
pub trait AsyncInitContext {
    /// `updateTexture(texName, canvas)`: upload an RGBA8 image (top row first) to
    /// the node-scoped texture `<nodeId>_<texName>` (with `flipY: true`).
    fn update_texture(&mut self, tex_name: &str, width: u32, height: u32, rgba: &[u8]);
    fn width(&self) -> f64;
    fn height(&self) -> f64;
    /// The parameters the overlay is generated from.
    fn params(&self) -> &Object;
    fn is_cancelled(&self) -> bool;
}

/// An effect that generates textures on the CPU (`asyncInit`).
pub trait AsyncInitEffect {
    fn async_init(&self, context: &mut dyn AsyncInitContext) -> Result<(), String>;
}

/// A registered effect: its native hooks and parameter table.
#[derive(Clone, Default)]
pub struct EffectHooks {
    pub lifecycle: Option<Rc<RefCell<dyn EffectLifecycle>>>,
    pub async_init: Option<Rc<dyn AsyncInitEffect>>,
    /// `effectDef.globals` (parameter name → spec).
    pub globals: Object,
    /// `effectDef.tags`.
    pub tags: Vec<String>,
}

/// The effects with native behavior, by effect key (`synth.media`, ...).
#[derive(Clone, Default)]
pub struct EffectRegistry {
    effects: HashMap<String, EffectHooks>,
}

impl EffectRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the hooks of `effect_key` (`namespace.func`).
    pub fn register(&mut self, effect_key: impl Into<String>, hooks: EffectHooks) {
        self.effects.insert(effect_key.into(), hooks);
    }

    /// `getEffect(effectKey)`.
    pub fn get(&self, effect_key: &str) -> Option<&EffectHooks> {
        self.effects.get(effect_key)
    }

    /// `effectDef.tags?.includes(tag)` from the embedded catalog definition.
    pub fn catalog_tags(effect_key: &str) -> Vec<String> {
        let Some((ns, func)) = effect_key.split_once('.') else {
            return Vec::new();
        };
        let Some(effect) = noisemaker_effects::effect(ns, func) else {
            return Vec::new();
        };
        let Ok(def) = Value::from_json(effect.definition_json) else {
            return Vec::new();
        };
        def.get("tags")
            .as_array()
            .map(|t| {
                t.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default()
    }
}
