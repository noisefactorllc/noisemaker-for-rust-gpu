//! The pipeline's external input state and its automation evaluation.
//!
//! `osc(...)`, `midi(...)` and `audio(...)` reach pass uniforms as
//! configuration objects. The reference pipeline (`runtime/pipeline.js`)
//! evaluates them every frame with `evaluateAutomation` against
//! `externalState = { midi: MidiState | null, audio: AudioState | null }`;
//! `updateGlobalUniforms` copies the audio waveform and spectrum into the
//! global uniforms and uploads the MIDI note grid.
//!
//! Both halves are [`noisemaker_input`]'s exact ports (`evaluateAutomation`
//! with V8's `Math.sin`/`Math.cos`, [`MidiState`], [`AudioState`],
//! [`noisemaker_input::InputGlobals`]); this module only adapts the
//! pipeline's [`Value`] uniforms to the evaluator's [`JsValue`] descriptors
//! and holds the states the host shares with the pipeline.

use std::cell::RefCell;
use std::rc::Rc;

use noisemaker_dsl::{Object, Value};
use noisemaker_input::automation as eval;
pub use noisemaker_input::automation::{AudioInputRequirements, AudioRequirement};
pub use noisemaker_input::{AudioState, AutomationContext, JsValue, MidiState};

/// A MIDI state shared between the host (which feeds it messages) and the
/// pipeline (which reads it every frame), as the reference shares one
/// `MidiState` object between `MidiInputManager` and the renderer.
pub type SharedMidiState = Rc<RefCell<MidiState>>;

/// An audio state shared between the host and the pipeline.
pub type SharedAudioState = Rc<RefCell<AudioState>>;

/// `pipeline.externalState`: the MIDI and audio states, either of which may be
/// absent (`null`).
#[derive(Clone, Default)]
pub struct ExternalState {
    /// `externalState.midi`.
    pub midi: Option<SharedMidiState>,
    /// `externalState.audio`.
    pub audio: Option<SharedAudioState>,
}

impl std::fmt::Debug for ExternalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalState")
            .field("midi", &self.midi.is_some())
            .field("audio", &self.audio.is_some())
            .finish()
    }
}

/// A pipeline value as the evaluator's JavaScript value. Objects keep their
/// enumerable members in order (automation descriptors carry no
/// non-enumerable members); every converted object is a distinct identity,
/// as the descriptors of a graph are a tree.
pub fn to_js_value(value: &Value) -> JsValue {
    match value {
        Value::Undefined => JsValue::Undefined,
        Value::Null => JsValue::Null,
        Value::Bool(b) => JsValue::Bool(*b),
        Value::Number(n) => JsValue::Number(*n),
        Value::String(s) => JsValue::string(s),
        Value::Array(items) => JsValue::array(items.iter().map(to_js_value)),
        Value::Object(o) => JsValue::object(o.iter().map(|(k, v)| (k.clone(), to_js_value(v)))),
        Value::Function(source) => JsValue::Function(source.as_str().into()),
    }
}

/// `isAutomationValue(value)`.
pub fn is_automation_value(value: &Value) -> bool {
    matches!(value, Value::Object(_)) && eval::is_automation_value(&to_js_value(value))
}

/// `pipeline.resolveUniformValue(value, time, paramSpec)`: the evaluated
/// value of an automation descriptor (rounded for `int` specs), or `None`
/// when `value` is not one (the reference returns it unchanged).
pub fn resolve_uniform_value(
    value: &Value,
    time: f64,
    spec: &Value,
    external: &ExternalState,
    context: AutomationContext,
) -> Option<Value> {
    if !matches!(value, Value::Object(_)) {
        return None;
    }
    let config = to_js_value(value);
    if !eval::is_automation_value(&config) {
        return None;
    }
    let midi = external.midi.as_ref().map(|m| m.borrow());
    let audio = external.audio.as_ref().map(|a| a.borrow());
    let state = eval::ExternalState::new(midi.as_deref(), audio.as_deref());
    eval::resolve_uniform_value(&config, time, &to_js_value(spec), state, context)
        .map(Value::Number)
}

/// `pipeline.getAudioInputRequirements()` over a graph's passes: `tagged`
/// says whether a pass's effect is tagged `audio`.
pub fn audio_input_requirements(
    passes: &[Object],
    tagged: impl Fn(&Object) -> bool,
) -> AudioInputRequirements {
    let converted: Vec<(JsValue, bool)> = passes
        .iter()
        .map(|p| (to_js_value(p.get_or_undefined("uniforms")), tagged(p)))
        .collect();
    eval::audio_input_requirements(converted.iter().map(|(uniforms, audio_tagged)| {
        eval::PassAudioInfo {
            uniforms,
            audio_tagged: *audio_tagged,
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(json: &str) -> Value {
        Value::from_json(json).unwrap()
    }

    fn resolve(json: &str, t: f64, spec: &str, external: &ExternalState) -> Option<f64> {
        resolve_uniform_value(
            &value(json),
            t,
            &value(spec),
            external,
            AutomationContext::at(0.0),
        )
        .map(|v| v.as_f64().unwrap())
    }

    #[test]
    fn descriptors_are_recognized() {
        assert!(is_automation_value(&value(
            r#"{"type":"Oscillator","oscType":0}"#
        )));
        assert!(is_automation_value(&value(r#"{"_ast":{"type":"Midi"}}"#)));
        assert!(!is_automation_value(&value(r#"{"type":"Number"}"#)));
        assert!(!is_automation_value(&value("[1,2]")));
        assert!(!is_automation_value(&Value::Number(1.0)));
    }

    #[test]
    fn oscillators_evaluate_with_spec_ranges() {
        let none = ExternalState::default();
        let sine =
            r#"{"type":"Oscillator","oscType":0,"min":0,"max":1,"speed":1,"offset":0,"seed":1}"#;
        assert_eq!(resolve(sine, 0.5, "{}", &none), Some(1.0));
        assert_eq!(
            resolve(sine, 0.5, r#"{"min":10,"max":20}"#, &none),
            Some(20.0)
        );
        let saw =
            r#"{"type":"Oscillator","oscType":2,"min":0,"max":1,"speed":1,"offset":0,"seed":1}"#;
        assert_eq!(
            resolve(saw, 0.26, r#"{"min":0,"max":10,"type":"int"}"#, &none),
            Some(3.0)
        );
        assert_eq!(resolve("1.5", 0.0, "{}", &none), None);
    }

    #[test]
    fn external_states_feed_midi_and_audio() {
        let midi = r#"{"type":"Midi","channel":1,"mode":2,"min":0,"max":1,"sensitivity":1}"#;
        let audio = r#"{"type":"Audio","band":0,"min":0,"max":1}"#;
        let none = ExternalState::default();
        assert_eq!(resolve(midi, 0.0, "{}", &none), Some(0.0));
        assert_eq!(resolve(audio, 0.0, "{}", &none), Some(0.0));

        let mut midi_state = MidiState::new();
        midi_state.handle_message(&[0x90, 60, 127], None);
        let mut audio_state = AudioState::new();
        audio_state.set_bands(0.5, 0.25, 0.125);
        let external = ExternalState {
            midi: Some(Rc::new(RefCell::new(midi_state))),
            audio: Some(Rc::new(RefCell::new(audio_state))),
        };
        assert_eq!(resolve(midi, 0.0, "{}", &external), Some(1.0));
        assert_eq!(resolve(audio, 0.0, "{}", &external), Some(0.5));
    }

    #[test]
    fn audio_requirements_visit_pass_uniforms() {
        let passes: Vec<Object> = vec![
            value(r#"{"uniforms":{"a":{"type":"Audio","band":4,"min":0,"max":1}}}"#)
                .as_object()
                .cloned()
                .unwrap(),
            value(r#"{"uniforms":{"b":1}}"#)
                .as_object()
                .cloned()
                .unwrap(),
        ];
        let req = audio_input_requirements(&passes, |_| false);
        assert!(req.needs_legacy && req.needs_legacy_raw);
        assert!(req.selected.is_empty());
        let req = audio_input_requirements(&passes[1..], |_| true);
        assert!(req.needs_legacy && !req.needs_legacy_raw);
    }
}
