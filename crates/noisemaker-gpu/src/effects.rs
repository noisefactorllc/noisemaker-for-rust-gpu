//! Native hooks of the catalog effects that define them.
//!
//! - `synth/media` (shaders/effects/synth/media/definition.js): `onInit`
//!   sets its image dimensions to 1x1, `onUpdate` returns
//!   `{imageSize: [imageWidth || 1, imageHeight || 1]}` (bound only where a
//!   pass does not resolve `imageSize`), and `setMediaDimensions(w, h)`
//!   updates them ([`MediaLifecycle`]).
//! - `filter/fibers`, `filter/scratches`, `filter/strayHair`: `asyncInit`
//!   traces worms on a canvas of the render size and uploads it as
//!   `<nodeId>_overlayTex`, through [`noisemaker_host::overlay`]
//!   ([`OverlayAsyncInit`]).
//!
//! [`native_effects`] registers them with their catalog `globals` and `tags`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use noisemaker_dsl::{Object, Value};
use noisemaker_host::canvas::StrokeCanvas;
use noisemaker_host::overlay::{OverlayEffect, OverlayParams, run_async_init};
use noisemaker_host::worm::TraceOutcome;

use crate::hooks::{
    AsyncInitContext, AsyncInitEffect, EffectHooks, EffectLifecycle, EffectRegistry, TextureImage,
    UpdateContext,
};

/// The effect key of synth/media.
pub const MEDIA_EFFECT_KEY: &str = "synth.media";

/// The catalog definition of `namespace/name` as a value.
pub fn catalog_definition(namespace: &str, name: &str) -> Option<Value> {
    let effect = noisemaker_effects::effect(namespace, name)?;
    Value::from_json(effect.definition_json).ok()
}

fn hooks_for(namespace: &str, name: &str) -> EffectHooks {
    let def = catalog_definition(namespace, name).unwrap_or_default();
    EffectHooks {
        lifecycle: None,
        async_init: None,
        globals: def.get("globals").as_object().cloned().unwrap_or_default(),
        tags: def
            .get("tags")
            .as_array()
            .map(|t| {
                t.iter()
                    .filter_map(|v| v.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// The registry of every catalog effect with native behavior: synth/media's
/// lifecycle and the three overlay asyncInits. `media` is the synth/media
/// hook object the registry shares (the reference's registry singleton), for
/// [`MediaLifecycle::set_media_dimensions`].
pub fn native_effects_with(media: Rc<RefCell<MediaLifecycle>>) -> EffectRegistry {
    let mut registry = EffectRegistry::new();
    let mut hooks = hooks_for("synth", "media");
    hooks.lifecycle = Some(media);
    registry.register(MEDIA_EFFECT_KEY, hooks);
    for effect in OverlayEffect::ALL {
        let mut hooks = hooks_for("filter", effect.func());
        hooks.async_init = Some(Arc::new(OverlayAsyncInit { effect }));
        registry.register(format!("filter.{}", effect.func()), hooks);
    }
    registry
}

/// [`native_effects_with`] with a new synth/media hook object.
pub fn native_effects() -> EffectRegistry {
    native_effects_with(Rc::new(RefCell::new(MediaLifecycle::default())))
}

/// synth/media's lifecycle: `this.state.imageWidth` / `imageHeight`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MediaLifecycle {
    /// `state.imageWidth` (`None` until `onInit` or `setMediaDimensions`).
    pub image_width: Option<f64>,
    /// `state.imageHeight`.
    pub image_height: Option<f64>,
}

impl MediaLifecycle {
    /// `setMediaDimensions(width, height)`: called when the media source
    /// changes.
    pub fn set_media_dimensions(&mut self, width: f64, height: f64) {
        self.image_width = Some(width);
        self.image_height = Some(height);
    }
}

/// `value || 1` for a Number state member.
fn or_one(value: Option<f64>) -> f64 {
    match value {
        Some(v) if v != 0.0 && !v.is_nan() => v,
        _ => 1.0,
    }
}

impl EffectLifecycle for MediaLifecycle {
    fn has_on_init(&self) -> bool {
        true
    }

    fn on_init(&mut self) {
        self.image_width = Some(1.0);
        self.image_height = Some(1.0);
    }

    fn has_on_update(&self) -> bool {
        true
    }

    fn on_update(&mut self, _context: &UpdateContext<'_>) -> Option<Object> {
        let mut uniforms = Object::new();
        uniforms.insert(
            "imageSize",
            Value::Array(vec![
                Value::Number(or_one(self.image_width)),
                Value::Number(or_one(self.image_height)),
            ]),
        );
        Some(uniforms)
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A parameter of an asyncInit's `params` as the overlay reads it.
fn overlay_param(value: &Value) -> noisemaker_host::js::JsValue {
    use noisemaker_host::js::JsValue;
    match value {
        Value::Undefined => JsValue::Undefined,
        Value::Null => JsValue::Null,
        Value::Bool(b) => JsValue::Bool(*b),
        Value::Number(n) => JsValue::Number(*n),
        Value::String(s) => JsValue::String(s.clone()),
        Value::Array(_) | Value::Object(_) | Value::Function(_) => JsValue::Object,
    }
}

/// The asyncInit of filter/fibers, filter/scratches or filter/strayHair: a
/// canvas of `context.width x context.height`, cleared and uploaded, then
/// traced and uploaded as the reference uploads it (`flipY: true` of the
/// canvas, i.e. [`StrokeCanvas::upload_image`]'s texture rows).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayAsyncInit {
    pub effect: OverlayEffect,
}

impl AsyncInitEffect for OverlayAsyncInit {
    fn async_init(&self, context: &mut dyn AsyncInitContext) -> Result<(), String> {
        // canvas.width = width; canvas.height = height
        let width = noisemaker_host::js::to_uint32(context.width());
        let height = noisemaker_host::js::to_uint32(context.height());
        let params = OverlayParams {
            seed: overlay_param(context.params().get_or_undefined("seed")),
            density: overlay_param(context.params().get_or_undefined("density")),
        };
        let context = RefCell::new(context);
        let mut canvas = StrokeCanvas::new(width, height);
        let outcome = run_async_init(
            self.effect,
            &mut canvas,
            &params,
            &mut || context.borrow().is_cancelled(),
            &mut |name, c: &mut StrokeCanvas| {
                let image = c.upload_image();
                context.borrow_mut().update_texture(
                    name,
                    TextureImage {
                        width: image.width,
                        height: image.height,
                        data: image.data,
                        flip_y: false,
                    },
                );
            },
        );
        match outcome {
            TraceOutcome::Failed => Err(format!(
                "RangeError: {} overlay of {width}x{height} cannot be traced",
                self.effect.func()
            )),
            TraceOutcome::Completed | TraceOutcome::Cancelled => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn media_lifecycle_follows_the_definition() {
        let mut media = MediaLifecycle::default();
        let globals = Object::new();
        let context = UpdateContext {
            time: 0.0,
            delta: 0.0,
            uniforms: &globals,
        };
        let size = |m: &mut MediaLifecycle| {
            m.on_update(&context)
                .unwrap()
                .get_or_undefined("imageSize")
                .clone()
        };
        assert_eq!(size(&mut media), Value::from_json("[1,1]").unwrap());
        media.set_media_dimensions(640.0, 0.0);
        assert_eq!(size(&mut media), Value::from_json("[640,1]").unwrap());
        media.on_init();
        assert_eq!(size(&mut media), Value::from_json("[1,1]").unwrap());
    }

    #[test]
    fn registry_has_the_catalog_hooks() {
        let registry = native_effects();
        let media = registry.get("synth.media").unwrap();
        assert!(media.lifecycle.is_some() && media.async_init.is_none());
        assert!(media.globals.contains_key("imageSize"));
        for key in ["filter.fibers", "filter.scratches", "filter.strayHair"] {
            let hooks = registry.get(key).unwrap();
            assert!(hooks.async_init.is_some(), "{key}");
            assert!(hooks.globals.contains_key("density"), "{key}");
            assert!(hooks.globals.contains_key("seed"), "{key}");
        }
    }
}
