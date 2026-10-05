//! Extension points for host-side effect behavior.
//!
//! The reference pipeline asks the effect registry (`getEffect(effectKey)`) for
//! three kinds of native behavior: the production lifecycle hooks
//! (`onInit`/`onUpdate`/`onDestroy`, GAP-026; e.g. `synth/media`), CPU-generated
//! overlay textures (`asyncInit`; the fibers/scratches/strayHair tracers), and the
//! effect's parameter table (`globals`, consulted by `checkAsyncRegen`). Native
//! ports of those effects register here ([`crate::effects::native_effects`]
//! registers the catalog's); an empty registry is exactly the reference running
//! effects that define none of them.
//!
//! `asyncInit` is asynchronous in the reference: it uploads its canvas as it
//! draws, yields between worms, and stops when `isCancelled()` turns true. The
//! pipeline runs each [`AsyncInitEffect`] on a worker thread and uploads the
//! images it sends from the render thread ([`crate::Pipeline::poll_async_effects`]).

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

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
    /// The hook object itself, for effect-specific host calls (synth/media's
    /// `setMediaDimensions`).
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// An image an asyncInit uploads: RGBA8 rows, row 0 first.
///
/// `flip_y` is the `flipY` of `updateTextureFromSource(texId, source, {flipY})`:
/// `true` reverses the rows on upload (a canvas, top row first, uploaded the
/// way the reference uploads it), `false` stores the rows as given (an image
/// already in texture row order, such as [`noisemaker_host::Rgba8Image`]).
#[derive(Clone, PartialEq, Eq)]
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub flip_y: bool,
}

impl std::fmt::Debug for TextureImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextureImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.data.len())
            .field("flip_y", &self.flip_y)
            .finish()
    }
}

/// What an `asyncInit` effect may do (`context` of `_startAsyncInit`).
pub trait AsyncInitContext {
    /// `updateTexture(texName, canvas)`: upload an image to the node-scoped
    /// texture `<nodeId>_<texName>` (unless the run was cancelled).
    fn update_texture(&mut self, tex_name: &str, image: TextureImage);
    /// `context.width` (the pipeline width when the run started).
    fn width(&self) -> f64;
    /// `context.height`.
    fn height(&self) -> f64;
    /// `context.params`: a copy of the step values (a debounced regeneration) or
    /// of the global uniforms (`initAsyncEffects`).
    fn params(&self) -> &Object;
    /// `context.isCancelled()`.
    fn is_cancelled(&self) -> bool;
}

/// An effect that generates textures on the CPU (`asyncInit`). It runs on a
/// worker thread; returning an error is the reference's rejected promise
/// (logged, never thrown into the render loop).
pub trait AsyncInitEffect: Send + Sync {
    fn async_init(&self, context: &mut dyn AsyncInitContext) -> Result<(), String>;
}

/// A registered effect: its native hooks and parameter table.
#[derive(Clone, Default)]
pub struct EffectHooks {
    pub lifecycle: Option<Rc<RefCell<dyn EffectLifecycle>>>,
    pub async_init: Option<Arc<dyn AsyncInitEffect>>,
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

    /// The registered effect keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.effects.keys().map(String::as_str)
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
