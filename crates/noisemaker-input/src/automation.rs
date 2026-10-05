//! Automation evaluation: a port of the top of the reference's
//! `shaders/src/runtime/pipeline.js` — `evaluateAutomation`,
//! `evaluateOscillator`, `integrateAutomation`, `evaluateMidi`,
//! `evaluateAudio`, `hasAudioSelectorIntent`, `isValidAudioSelector`,
//! `Pipeline.resolveUniformValue` and `Pipeline.getAudioInputRequirements`.
//!
//! Descriptors are [`JsValue`]s and every check keeps its JavaScript meaning.
//! The external state is generic: anything implementing [`MidiSource`] and
//! [`AudioSource`] (this crate's [`MidiState`] and [`AudioState`] do) can be
//! read, and an absent source evaluates like the reference's `null` state.
//! `Math.sin`/`Math.cos` are V8's ([`crate::jsmath`]), so values match the
//! reference bit for bit.

use crate::audio::{AudioState, DeviceSelector};
use crate::clock::system_now_ms;
use crate::js::JsValue;
use crate::jsmath::{js_clamp, js_cos, js_min, js_round, js_sin};
use crate::midi::{MidiChannelState, MidiState, PortSelector, SelectorKey};

/// `Math.PI * 2`.
pub const TAU: f64 = std::f64::consts::PI * 2.0;
/// `MAX_AUTOMATION_DEPTH`.
pub const MAX_AUTOMATION_DEPTH: usize = 8;

/// A `{min, max}` range of `AUTOMATION_FIELD_RANGES`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldRange {
    /// Lower bound.
    pub min: f64,
    /// Upper bound.
    pub max: f64,
}

/// `AUTOMATION_FIELD_RANGES.unit`.
pub const UNIT_RANGE: FieldRange = FieldRange { min: 0.0, max: 1.0 };
/// `AUTOMATION_FIELD_RANGES.oscillatorSpeed`.
pub const OSCILLATOR_SPEED_RANGE: FieldRange = FieldRange {
    min: -20.0,
    max: 20.0,
};
/// `AUTOMATION_FIELD_RANGES.oscillatorOffset`.
pub const OSCILLATOR_OFFSET_RANGE: FieldRange = FieldRange {
    min: -1.0,
    max: 1.0,
};
/// `AUTOMATION_FIELD_RANGES.oscillatorSeed`.
pub const OSCILLATOR_SEED_RANGE: FieldRange = FieldRange {
    min: 1.0,
    max: 9999.0,
};
/// `AUTOMATION_FIELD_RANGES.midiSensitivity`.
pub const MIDI_SENSITIVITY_RANGE: FieldRange = FieldRange {
    min: 0.0,
    max: 10.0,
};

/// The `range` argument of the evaluators: none (`null`), a fixed field
/// range, or a consumer's parameter spec (any JavaScript value; only an
/// object with finite `min` and `max` scales).
#[derive(Clone, Copy, Debug)]
pub enum Range<'a> {
    /// `null`.
    None,
    /// One of the `AUTOMATION_FIELD_RANGES`.
    Field(FieldRange),
    /// A parameter spec.
    Spec(&'a JsValue),
}

impl Range<'_> {
    fn bounds(&self) -> Option<(f64, f64)> {
        match self {
            Range::None => None,
            Range::Field(range) => Some((range.min, range.max)),
            Range::Spec(spec) => {
                if !spec.truthy() {
                    return None;
                }
                let min = spec.get("min");
                let max = spec.get("max");
                if !min.is_finite_number() || !max.is_finite_number() {
                    return None;
                }
                Some((min.as_number()?, max.as_number()?))
            }
        }
    }
}

/// The `context` of an evaluation: `Date.now()` sampled once per top-level
/// call.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AutomationContext {
    /// Wall-clock milliseconds for MIDI trigger falloff.
    pub wall_time: f64,
}

impl AutomationContext {
    /// A context reading `Date.now()` from the system clock.
    pub fn now() -> Self {
        AutomationContext {
            wall_time: system_now_ms(),
        }
    }

    /// A context at `wall_time` milliseconds.
    pub fn at(wall_time: f64) -> Self {
        AutomationContext { wall_time }
    }
}

/// The note `evaluateMidi` reads: a channel, or a zone voice (gate 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoteView {
    /// `note.key`.
    pub key: f64,
    /// `note.velocity`.
    pub velocity: f64,
    /// `note.gate`.
    pub gate: f64,
    /// `note.time`.
    pub time: f64,
}

/// The channel fields `evaluateMidi` reads. `None` is `undefined`.
pub trait MidiChannelSource {
    /// `key`, `velocity`, `gate` and `time` of the channel.
    fn note(&self) -> NoteView;
    /// `cc[index]`.
    fn cc(&self, index: usize) -> Option<f64>;
    /// `cc14[index]`.
    fn cc14(&self, index: usize) -> Option<f64>;
    /// `nrpn.get(parameter)`.
    fn nrpn(&self, parameter: f64) -> Option<f64>;
    /// `pitchBend`.
    fn pitch_bend(&self) -> Option<f64>;
    /// `pressure`.
    fn pressure(&self) -> Option<f64>;
    /// `polyPressure[key]`.
    fn poly_pressure(&self, key: f64) -> Option<f64>;
}

/// A zone voice (`getZoneVoice` result): the held note and its channel.
#[derive(Clone, Copy, Debug)]
pub struct VoiceView<'a, C: ?Sized> {
    /// `voice.key`.
    pub key: f64,
    /// `voice.velocity`.
    pub velocity: f64,
    /// `voice.time`.
    pub time: f64,
    /// `voice.channel`.
    pub channel: &'a C,
}

/// What `evaluateAutomation`/`evaluateMidi` call on `externalState.midi`.
pub trait MidiSource {
    /// The channel type.
    type Channel: MidiChannelSource + ?Sized;
    /// Whether the source has `getPortState`; without it the reference reads
    /// the source itself for every descriptor.
    fn has_port_state(&self) -> bool {
        true
    }
    /// `getPortState(config)`.
    fn port_state(&self, config: &JsValue) -> Option<&Self>;
    /// `getZoneVoice?.(config)`; `None` also when the method is absent.
    fn zone_voice(&self, config: &JsValue) -> Option<VoiceView<'_, Self::Channel>>;
    /// `getChannel(n)`.
    fn channel(&self, n: &JsValue) -> &Self::Channel;
}

/// The band fields `evaluateAudio` reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AudioLevels {
    /// `low`.
    pub low: f64,
    /// `mid`.
    pub mid: f64,
    /// `high`.
    pub high: f64,
    /// `vol`.
    pub vol: f64,
    /// `raw`.
    pub raw: f64,
    /// `rawReady === true`.
    pub raw_ready: bool,
}

/// What `evaluateAudio` calls on `externalState.audio`.
pub trait AudioSource {
    /// `getDeviceChannelState?.(config)`; `None` also when the method is
    /// absent.
    fn device_channel_state(&self, config: &JsValue) -> Option<&Self>;
    /// The band fields.
    fn levels(&self) -> AudioLevels;
}

/// `externalState`: the MIDI and audio states, either of which may be absent.
#[derive(Debug)]
pub struct ExternalState<'a, M: ?Sized = MidiState, A: ?Sized = AudioState> {
    /// `externalState.midi`.
    pub midi: Option<&'a M>,
    /// `externalState.audio`.
    pub audio: Option<&'a A>,
}

impl<M: ?Sized, A: ?Sized> Clone for ExternalState<'_, M, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M: ?Sized, A: ?Sized> Copy for ExternalState<'_, M, A> {}

impl<'a, M: ?Sized, A: ?Sized> ExternalState<'a, M, A> {
    /// States for evaluation.
    pub fn new(midi: Option<&'a M>, audio: Option<&'a A>) -> Self {
        ExternalState { midi, audio }
    }
}

impl ExternalState<'static, MidiState, AudioState> {
    /// No MIDI or audio state.
    pub fn none() -> Self {
        ExternalState {
            midi: None,
            audio: None,
        }
    }
}

// ---------------------------------------------------------------- MidiState

fn selector_key(value: &JsValue) -> SelectorKey<'_> {
    match value {
        JsValue::String(s) if !s.is_empty() => SelectorKey::Str(s),
        other if other.truthy() => SelectorKey::OtherTruthy,
        _ => SelectorKey::Falsy,
    }
}

impl MidiChannelSource for MidiChannelState {
    fn note(&self) -> NoteView {
        NoteView {
            key: f64::from(self.key),
            velocity: f64::from(self.velocity),
            gate: f64::from(self.gate),
            time: self.time,
        }
    }

    fn cc(&self, index: usize) -> Option<f64> {
        self.cc.get(index).map(|&v| f64::from(v))
    }

    fn cc14(&self, index: usize) -> Option<f64> {
        self.cc14.get(index).map(|&v| f64::from(v))
    }

    fn nrpn(&self, parameter: f64) -> Option<f64> {
        if !(0.0..=65535.0).contains(&parameter) || parameter.trunc() != parameter {
            return None;
        }
        self.nrpn.get(&(parameter as u16)).map(|&v| f64::from(v))
    }

    fn pitch_bend(&self) -> Option<f64> {
        Some(f64::from(self.pitch_bend))
    }

    fn pressure(&self) -> Option<f64> {
        Some(f64::from(self.pressure))
    }

    fn poly_pressure(&self, key: f64) -> Option<f64> {
        if !(0.0..128.0).contains(&key) || key.trunc() != key {
            return None;
        }
        Some(f64::from(self.poly_pressure[key as usize]))
    }
}

impl MidiSource for MidiState {
    type Channel = MidiChannelState;

    fn port_state(&self, config: &JsValue) -> Option<&Self> {
        self.get_port_state(&PortSelector {
            name: selector_key(config.get("name")),
            id: selector_key(config.get("id")),
        })
    }

    fn zone_voice(&self, config: &JsValue) -> Option<VoiceView<'_, MidiChannelState>> {
        let zone = config.get("zone");
        let zone = if zone.strict_eq_number(0.0) {
            0
        } else if zone.strict_eq_number(1.0) {
            1
        } else {
            return None;
        };
        let members = config.get("members");
        let members = if members.is_undefined() {
            None
        } else if members.is_integer() && (1.0..=15.0).contains(&members.as_number()?) {
            Some(members.as_number()? as u8)
        } else {
            return None;
        };
        self.get_zone_voice(zone, members).map(|voice| VoiceView {
            key: f64::from(voice.note.key),
            velocity: f64::from(voice.note.velocity),
            time: voice.note.time,
            channel: voice.channel,
        })
    }

    fn channel(&self, n: &JsValue) -> &MidiChannelState {
        self.get_channel(n.channel_key().map_or(1, i64::from))
    }
}

// ---------------------------------------------------------------- AudioState

impl AudioSource for AudioState {
    fn device_channel_state(&self, config: &JsValue) -> Option<&Self> {
        let channel = match config.get("channel") {
            JsValue::Undefined => None,
            JsValue::Number(n) => Some(*n),
            _ => Some(f64::NAN),
        };
        self.get_device_channel_state(&DeviceSelector {
            name: selector_key(config.get("name")),
            id: selector_key(config.get("id")),
            channel,
        })
    }

    fn levels(&self) -> AudioLevels {
        AudioLevels {
            low: self.low,
            mid: self.mid,
            high: self.high,
            vol: self.vol,
            raw: self.raw,
            raw_ready: self.raw_ready,
        }
    }
}

// ---------------------------------------------------------------- oscillators

fn osc_sine(t: f64) -> f64 {
    // Smooth continuous sine: 0->1->0 over t=0..1, no discontinuity at wrap.
    (1.0 - js_cos(t * TAU)) * 0.5
}

fn osc_tri(t: f64) -> f64 {
    let tf = t - t.floor();
    1.0 - (tf * 2.0 - 1.0).abs()
}

fn osc_saw(t: f64) -> f64 {
    t - t.floor()
}

fn osc_saw_inv(t: f64) -> f64 {
    1.0 - (t - t.floor())
}

fn osc_square(t: f64) -> f64 {
    if (t - t.floor()) >= 0.5 { 1.0 } else { 0.0 }
}

/// `hash21(px, py, s)`.
fn hash21(px: f64, py: f64, s: f64) -> f64 {
    let mut x = (px * 234.34 + s) % 1.0;
    let mut y = (py * 435.345 + s) % 1.0;
    if x < 0.0 {
        x += 1.0;
    }
    if y < 0.0 {
        y += 1.0;
    }
    let p = x + y + (x + y) * 34.23;
    (x * y * p) % 1.0
}

/// `noise2D(px, py, s)`: value noise.
fn noise_2d(px: f64, py: f64, s: f64) -> f64 {
    let ix = px.floor();
    let iy = py.floor();
    let mut fx = px - ix;
    let mut fy = py - iy;
    fx = fx * fx * (3.0 - 2.0 * fx);
    fy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash21(ix, iy, s);
    let b = hash21(ix + 1.0, iy, s);
    let c = hash21(ix, iy + 1.0, s);
    let d = hash21(ix + 1.0, iy + 1.0, s);
    a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
}

/// `oscNoise(t, seed)`: looping noise sampled on a circle.
fn osc_noise(t: f64, seed: f64) -> f64 {
    let temporal = t % 1.0;
    let angle = temporal * TAU;
    let radius = 2.0;
    let loop_x = js_cos(angle) * radius;
    let loop_y = js_sin(angle) * radius;
    let n1 = noise_2d(loop_x + seed, loop_y + seed, seed);
    let n2 = noise_2d(loop_x + seed * 2.0, loop_y + seed * 2.0, seed);
    (n1 + n2) / 2.0
}

/// `oscNoise2d(time, speed, seed)`: two-stage periodic noise (osc2d).
fn osc_noise_2d(time: f64, speed: f64, seed: f64) -> f64 {
    let periodic_value = |x: f64, v: f64| (js_sin((x - v) * TAU) + 1.0) * 0.5;
    let px = ((seed % 16.0).abs() + 0.5) / 16.0;
    let py = (((seed / 16.0).floor() % 16.0).abs() + 0.5) / 16.0;
    let time_noise = noise_2d(px, py, seed + 12345.0);
    let value_noise = noise_2d(px, py, seed);
    let scaled_time = periodic_value(time, time_noise) * speed;
    periodic_value(scaled_time, value_noise)
}

/// 16-point Gauss-Legendre nodes on [-1, 1].
const INTEGRATION_NODES: [f64; 16] = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.755404408355003,
    -0.6178762444026438,
    -0.4580167776572274,
    -0.2816035507792589,
    -0.0950125098376374,
    0.0950125098376374,
    0.2816035507792589,
    0.4580167776572274,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const INTEGRATION_WEIGHTS: [f64; 16] = [
    0.0271524594117541,
    0.0622535239386479,
    0.0951585116824928,
    0.1246289712555339,
    0.1495959888165767,
    0.1691565193950025,
    0.1826034150449236,
    0.1894506104550685,
    0.1894506104550685,
    0.1826034150449236,
    0.1691565193950025,
    0.1495959888165767,
    0.1246289712555339,
    0.0951585116824928,
    0.0622535239386479,
    0.0271524594117541,
];
const INTEGRATION_RULES: [(&[f64], &[f64]); 4] = [
    (&INTEGRATION_NODES, &INTEGRATION_WEIGHTS),
    (
        &[
            -0.9602898564975363,
            -0.7966664774136267,
            -0.525532409916329,
            -0.1834346424956498,
            0.1834346424956498,
            0.525532409916329,
            0.7966664774136267,
            0.9602898564975363,
        ],
        &[
            0.1012285362903763,
            0.2223810344533745,
            0.3137066458778873,
            0.362683783378362,
            0.362683783378362,
            0.3137066458778873,
            0.2223810344533745,
            0.1012285362903763,
        ],
    ),
    (
        &[
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ],
        &[
            0.3478548451374538,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374538,
        ],
    ),
    (&[-0.5773502691896257, 0.5773502691896257], &[1.0, 1.0]),
];

// ---------------------------------------------------------------- predicates

fn type_is(value: &JsValue, kind: &str) -> bool {
    value.get("type").strict_eq_str(kind) || value.get("_ast").get("type").strict_eq_str(kind)
}

/// `isAutomationValue(value)`: an object whose `type` (or `_ast.type`) is
/// `Oscillator`, `Midi` or `Audio`.
pub fn is_automation_value(value: &JsValue) -> bool {
    value.truthy()
        && matches!(value, JsValue::Object(_))
        && (type_is(value, "Oscillator") || type_is(value, "Midi") || type_is(value, "Audio"))
}

/// `scaleAutomationValue(value, range)`.
fn scale_automation_value(value: f64, range: Range<'_>) -> f64 {
    match range.bounds() {
        Some((min, max)) => min + value * (max - min),
        None => value,
    }
}

/// `hasDynamicAutomationFields(config)`.
fn has_dynamic_automation_fields(config: &JsValue) -> bool {
    let fields: &[&str] = if type_is(config, "Midi") {
        &["min", "max", "sensitivity"]
    } else {
        &["min", "max"]
    };
    fields
        .iter()
        .any(|field| is_automation_value(config.get(field)))
}

/// `oscPrimitive(type, x)`: the antiderivative of the simple shapes; `None`
/// (`null`) for other kinds.
fn osc_primitive(kind: &JsValue, x: f64) -> Option<f64> {
    let whole = x.floor();
    let fraction = x - whole;
    let kind = kind.as_number()?;
    if kind == 0.0 {
        Some(x * 0.5 - js_sin(x * TAU) / (2.0 * TAU))
    } else if kind == 1.0 {
        let partial = if fraction < 0.5 {
            fraction * fraction
        } else {
            2.0 * fraction - fraction * fraction - 0.5
        };
        Some(whole * 0.5 + partial)
    } else if kind == 2.0 {
        Some(whole * 0.5 + fraction * fraction * 0.5)
    } else if kind == 3.0 {
        Some(x - (whole * 0.5 + fraction * fraction * 0.5))
    } else if kind == 4.0 {
        Some(whole * 0.5 + crate::jsmath::js_max(0.0, fraction - 0.5))
    } else {
        None
    }
}

/// `canIntegrateOscillatorExactly(config)`.
fn can_integrate_oscillator_exactly(config: &JsValue) -> bool {
    let kind = config.get("oscType").to_number();
    (0.0..=4.0).contains(&kind)
        && ["min", "max", "speed", "offset", "seed"]
            .iter()
            .all(|field| config.get(field).is_finite_number())
}

/// `integrateSimpleOscillator(config, normalizedTime)`.
fn integrate_simple_oscillator(config: &JsValue, normalized_time: f64) -> f64 {
    let kind = config.get("oscType");
    let num = |field: &str| config.get(field).as_number().unwrap_or(f64::NAN);
    let (min, max, speed, offset) = (num("min"), num("max"), num("speed"), num("offset"));
    if speed == 0.0 {
        let none = ExternalState::<MidiState, AudioState>::none();
        let value = evaluate_oscillator(
            config,
            0.0,
            none,
            0,
            &mut Vec::new(),
            AutomationContext::at(f64::NAN),
        );
        return value * normalized_time;
    }
    // `null - null` is 0 in JavaScript.
    let start = osc_primitive(kind, offset).unwrap_or(0.0);
    let end = osc_primitive(kind, offset + speed * normalized_time).unwrap_or(0.0);
    let raw_integral = (end - start) / speed;
    min * normalized_time + (max - min) * raw_integral
}

// ---------------------------------------------------------------- evaluators

/// `resolveAutomationField(value, ..., fallback, context)`.
#[allow(clippy::too_many_arguments)]
fn resolve_automation_field<M, A>(
    value: &JsValue,
    normalized_time: f64,
    range: FieldRange,
    external: ExternalState<'_, M, A>,
    depth: usize,
    stack: &mut Vec<usize>,
    fallback: f64,
    context: AutomationContext,
) -> f64
where
    M: MidiSource + ?Sized,
    A: AudioSource + ?Sized,
{
    if is_automation_value(value) {
        return evaluate_automation_at(
            value,
            normalized_time,
            Range::Field(range),
            external,
            depth + 1,
            stack,
            context,
        );
    }
    match value.as_number() {
        Some(n) if n.is_finite() => n,
        _ => fallback,
    }
}

/// `integrateAutomation(config, normalizedTime, range, ...)`: the integral of
/// an automated rate over `[0, normalizedTime]`.
fn integrate_automation<M, A>(
    config: &JsValue,
    normalized_time: f64,
    range: Range<'_>,
    external: ExternalState<'_, M, A>,
    depth: usize,
    stack: &mut Vec<usize>,
    context: AutomationContext,
) -> f64
where
    M: MidiSource + ?Sized,
    A: AudioSource + ?Sized,
{
    let integral = if type_is(config, "Oscillator") && can_integrate_oscillator_exactly(config) {
        integrate_simple_oscillator(config, normalized_time)
    } else if (type_is(config, "Midi") || type_is(config, "Audio"))
        && !has_dynamic_automation_fields(config)
    {
        // External inputs expose the current snapshot rather than a history.
        evaluate_automation_at(
            config,
            normalized_time,
            Range::None,
            external,
            depth + 1,
            stack,
            context,
        ) * normalized_time
    } else {
        // Decrease the quadrature order as rate modulators nest.
        let (nodes, weights) = INTEGRATION_RULES[depth.min(INTEGRATION_RULES.len() - 1)];
        let midpoint = normalized_time * 0.5;
        let half_width = normalized_time * 0.5;
        let mut sum = 0.0;
        for (node, weight) in nodes.iter().zip(weights) {
            let sample_time = midpoint + half_width * node;
            sum += weight
                * evaluate_automation_at(
                    config,
                    sample_time,
                    Range::None,
                    external,
                    depth + 1,
                    stack,
                    context,
                );
        }
        half_width * sum
    };
    match range.bounds() {
        Some((min, max)) => min * normalized_time + integral * (max - min),
        None => integral,
    }
}

/// `evaluateAutomation(config, normalizedTime, range, externalState)`: the
/// value of an `osc()`, `midi()` or `audio()` descriptor at a normalized
/// time, scaled into `range`. Non-descriptors scale 0.
pub fn evaluate_automation<M, A>(
    config: &JsValue,
    normalized_time: f64,
    range: Range<'_>,
    external: ExternalState<'_, M, A>,
    context: AutomationContext,
) -> f64
where
    M: MidiSource + ?Sized,
    A: AudioSource + ?Sized,
{
    evaluate_automation_at(
        config,
        normalized_time,
        range,
        external,
        0,
        &mut Vec::new(),
        context,
    )
}

fn evaluate_automation_at<M, A>(
    config: &JsValue,
    normalized_time: f64,
    range: Range<'_>,
    external: ExternalState<'_, M, A>,
    depth: usize,
    stack: &mut Vec<usize>,
    context: AutomationContext,
) -> f64
where
    M: MidiSource + ?Sized,
    A: AudioSource + ?Sized,
{
    let identity = config.identity();
    if !is_automation_value(config)
        || depth > MAX_AUTOMATION_DEPTH
        || identity.is_some_and(|id| stack.contains(&id))
    {
        return scale_automation_value(0.0, range);
    }
    let id = identity.expect("automation values are objects");
    stack.push(id);
    let value = if type_is(config, "Oscillator") {
        evaluate_oscillator(config, normalized_time, external, depth, stack, context)
    } else if type_is(config, "Midi") {
        let midi_state = match external.midi {
            Some(midi) if midi.has_port_state() => midi.port_state(config),
            other => other,
        };
        let min = resolve_automation_field(
            config.get("min"),
            normalized_time,
            UNIT_RANGE,
            external,
            depth,
            stack,
            0.0,
            context,
        );
        let max = resolve_automation_field(
            config.get("max"),
            normalized_time,
            UNIT_RANGE,
            external,
            depth,
            stack,
            1.0,
            context,
        );
        let sensitivity = resolve_automation_field(
            config.get("sensitivity"),
            normalized_time,
            MIDI_SENSITIVITY_RANGE,
            external,
            depth,
            stack,
            1.0,
            context,
        );
        evaluate_midi(config, midi_state, context.wall_time, min, max, sensitivity)
    } else if config.get("_invalid").truthy() {
        match config.get("min").as_number() {
            Some(n) if n.is_finite() => n,
            _ => 0.0,
        }
    } else {
        let min = resolve_automation_field(
            config.get("min"),
            normalized_time,
            UNIT_RANGE,
            external,
            depth,
            stack,
            0.0,
            context,
        );
        let max = resolve_automation_field(
            config.get("max"),
            normalized_time,
            UNIT_RANGE,
            external,
            depth,
            stack,
            1.0,
            context,
        );
        evaluate_audio(config, external.audio, min, max)
    };
    if let Some(position) = stack.iter().rposition(|&entry| entry == id) {
        stack.remove(position);
    }
    scale_automation_value(value, range)
}

/// `evaluateOscillator(osc, normalizedTime, externalState, ...)`.
fn evaluate_oscillator<M, A>(
    osc: &JsValue,
    normalized_time: f64,
    external: ExternalState<'_, M, A>,
    depth: usize,
    stack: &mut Vec<usize>,
    context: AutomationContext,
) -> f64
where
    M: MidiSource + ?Sized,
    A: AudioSource + ?Sized,
{
    let kind = osc.get("oscType");
    let min = resolve_automation_field(
        osc.get("min"),
        normalized_time,
        UNIT_RANGE,
        external,
        depth,
        stack,
        0.0,
        context,
    );
    let max = resolve_automation_field(
        osc.get("max"),
        normalized_time,
        UNIT_RANGE,
        external,
        depth,
        stack,
        1.0,
        context,
    );
    let offset = resolve_automation_field(
        osc.get("offset"),
        normalized_time,
        OSCILLATOR_OFFSET_RANGE,
        external,
        depth,
        stack,
        0.0,
        context,
    );
    let seed = resolve_automation_field(
        osc.get("seed"),
        normalized_time,
        OSCILLATOR_SEED_RANGE,
        external,
        depth,
        stack,
        1.0,
        context,
    );
    let speed_value = osc.get("speed");
    let kind_number = kind.as_number();
    // Kind 6 samples its speed directly; the reference also integrates it
    // first, but discards the phase (the evaluation has no side effects).
    let value = if kind_number == Some(6.0) {
        let speed = resolve_automation_field(
            speed_value,
            normalized_time,
            OSCILLATOR_SPEED_RANGE,
            external,
            depth,
            stack,
            1.0,
            context,
        );
        osc_noise_2d(
            normalized_time + offset,
            if speed.is_finite() { speed } else { 1.0 },
            seed,
        )
    } else {
        // A modulated rate is frequency modulation: phase is the integral of
        // rate. Literal rates keep the closed form exactly.
        let phase = if is_automation_value(speed_value) {
            integrate_automation(
                speed_value,
                normalized_time,
                Range::Field(OSCILLATOR_SPEED_RANGE),
                external,
                depth,
                stack,
                context,
            )
        } else {
            normalized_time
                * match speed_value.as_number() {
                    Some(n) if n.is_finite() => n,
                    _ => 1.0,
                }
        };
        let t = phase + offset;
        match kind_number {
            Some(0.0) => osc_sine(t),
            Some(1.0) => osc_tri(t),
            Some(2.0) => osc_saw(t),
            Some(3.0) => osc_saw_inv(t),
            Some(4.0) => osc_square(t),
            Some(5.0) => osc_noise(t, seed),
            _ => 0.0,
        }
    };
    min + value * (max - min)
}

/// `evaluateMidi(config, midiState, currentTime, min, max, sensitivity)`.
pub fn evaluate_midi<M>(
    config: &JsValue,
    midi_state: Option<&M>,
    current_time: f64,
    min: f64,
    max: f64,
    sensitivity: f64,
) -> f64
where
    M: MidiSource + ?Sized,
{
    let Some(midi_state) = midi_state.filter(|_| !config.get("_invalid").truthy()) else {
        return min;
    };
    let mode = config.get("mode");
    let zone = config.get("zone");
    let has_zone = !zone.is_undefined();
    if has_zone
        && (!config.get("channel").is_undefined()
            || (!zone.strict_eq_number(0.0) && !zone.strict_eq_number(1.0)))
    {
        return min;
    }
    let members = config.get("members");
    if !members.is_undefined() {
        let valid = has_zone
            && members.is_integer()
            && members
                .as_number()
                .is_some_and(|m| (1.0..=15.0).contains(&m));
        if !valid {
            return min;
        }
    }
    let channel_value = config.get("channel");
    if !has_zone
        && mode.to_number() >= 5.0
        && !(channel_value.is_integer()
            && channel_value
                .as_number()
                .is_some_and(|c| (1.0..=16.0).contains(&c)))
    {
        return min;
    }
    let voice = if has_zone {
        match midi_state.zone_voice(config) {
            Some(voice) => Some(voice),
            None => return min,
        }
    } else {
        None
    };
    let channel: &M::Channel = match &voice {
        Some(voice) => voice.channel,
        None => midi_state.channel(channel_value),
    };
    let note = match &voice {
        Some(voice) => NoteView {
            key: voice.key,
            velocity: voice.velocity,
            gate: 1.0,
            time: voice.time,
        },
        None => channel.note(),
    };
    let mode_number = mode.as_number();
    let mut raw_value = 0.0;
    match mode_number {
        Some(0.0) => raw_value = note.key,
        Some(1.0) => {
            if note.gate == 1.0 {
                raw_value = note.key;
            }
        }
        Some(2.0) => {
            if note.gate == 1.0 {
                raw_value = note.velocity;
            }
        }
        Some(3.0) => {
            if note.gate == 1.0 {
                raw_value = note.key;
                let elapsed = current_time - note.time;
                let decay = js_min(1.0, elapsed * sensitivity * 0.001);
                raw_value *= 1.0 - decay;
            }
        }
        Some(m) if m == 5.0 || m == 6.0 => {
            let cc = config.get("cc");
            let cc = if cc.is_nullish() {
                &JsValue::Number(1.0)
            } else {
                cc
            };
            let limit = if m == 6.0 { 31.0 } else { 127.0 };
            let Some(index) = cc
                .as_number()
                .filter(|&c| crate::jsmath::is_integer(c) && c >= 0.0 && c <= limit)
            else {
                return min;
            };
            let normalized = if m == 6.0 {
                channel.cc14(index as usize).unwrap_or(0.0) / 16383.0
            } else {
                channel.cc(index as usize).unwrap_or(0.0) / 127.0
            };
            return min + normalized * (max - min);
        }
        Some(7.0) => {
            let nrpn = config.get("nrpn");
            let Some(parameter) = nrpn
                .as_number()
                .filter(|&p| crate::jsmath::is_integer(p) && (0.0..=16382.0).contains(&p))
            else {
                return min;
            };
            return min + channel.nrpn(parameter).unwrap_or(0.0) / 16383.0 * (max - min);
        }
        Some(8.0) => return min + channel.pitch_bend().unwrap_or(8192.0) / 16383.0 * (max - min),
        Some(9.0) => return min + channel.pressure().unwrap_or(0.0) / 127.0 * (max - min),
        Some(10.0) => {
            return min + channel.poly_pressure(note.key).unwrap_or(0.0) / 127.0 * (max - min);
        }
        _ => {
            // 4: velocity (default) - velocity with falloff
            if note.gate == 1.0 {
                raw_value = note.velocity;
                let elapsed = current_time - note.time;
                let decay = js_min(1.0, elapsed * sensitivity * 0.001);
                raw_value *= 1.0 - decay;
            }
        }
    }
    let normalized = raw_value / 127.0;
    min + normalized * (max - min)
}

/// `hasAudioSelectorIntent(config)`.
pub fn has_audio_selector_intent(config: &JsValue) -> bool {
    let ast = config.get("_ast");
    let source = if ast.get("type").strict_eq_str("Audio") {
        ast
    } else {
        config
    };
    !config.get("name").is_undefined()
        || !config.get("id").is_undefined()
        || !config.get("channel").is_undefined()
        || !source.get("name").is_undefined()
        || !source.get("id").is_undefined()
        || !source.get("channel").is_undefined()
}

/// `isValidAudioSelector(config)`.
pub fn is_valid_audio_selector(config: &JsValue) -> bool {
    let ast = config.get("_ast");
    let source = if ast.get("type").strict_eq_str("Audio") {
        ast
    } else {
        config
    };
    let (name, id, channel) = (config.get("name"), config.get("id"), config.get("channel"));
    if (!source.get("name").is_undefined() && name.is_undefined())
        || (!source.get("id").is_undefined() && id.is_undefined())
        || (!source.get("channel").is_undefined() && channel.is_undefined())
    {
        return false;
    }
    if !name.is_undefined() && !matches!(name, JsValue::String(s) if !s.is_empty()) {
        return false;
    }
    if !id.is_undefined() && (!matches!(id, JsValue::String(s) if !s.is_empty()) || !name.truthy())
    {
        return false;
    }
    channel.is_integer()
        && channel
            .as_number()
            .is_some_and(|c| (1.0..=32.0).contains(&c))
}

/// `evaluateAudio(config, audioState, min, max)`.
pub fn evaluate_audio<A>(config: &JsValue, audio_state: Option<&A>, min: f64, max: f64) -> f64
where
    A: AudioSource + ?Sized,
{
    if config.get("_invalid").truthy() {
        return min;
    }
    let Some(audio_state) = audio_state else {
        return min;
    };
    let band = config.get("band");
    let has_selector = has_audio_selector_intent(config);
    if has_selector && !is_valid_audio_selector(config) {
        return min;
    }
    let selected = if has_selector {
        audio_state.device_channel_state(config)
    } else {
        Some(audio_state)
    };
    let Some(selected) = selected else {
        return min;
    };
    let levels = selected.levels();
    let raw_value = match band.as_number() {
        Some(0.0) => levels.low,
        Some(1.0) => levels.mid,
        Some(2.0) => levels.high,
        Some(4.0) => {
            if !levels.raw_ready {
                return min;
            }
            // `raw || 0`: NaN and -0 read as 0.
            let raw = if levels.raw == 0.0 || levels.raw.is_nan() {
                0.0
            } else {
                levels.raw
            };
            (js_clamp(raw, -1.0, 1.0) + 1.0) * 0.5
        }
        Some(3.0) => levels.vol,
        _ => 0.0,
    };
    let raw_value = js_clamp(raw_value, 0.0, 1.0);
    min + raw_value * (max - min)
}

/// `Pipeline.resolveUniformValue(value, time, paramSpec)`: `None` when the
/// value is not an automation descriptor (the reference returns it
/// unchanged); otherwise the evaluation scaled into the spec's range, rounded
/// with `Math.round` for `type: 'int'` specs.
pub fn resolve_uniform_value<M, A>(
    value: &JsValue,
    time: f64,
    param_spec: &JsValue,
    external: ExternalState<'_, M, A>,
    context: AutomationContext,
) -> Option<f64>
where
    M: MidiSource + ?Sized,
    A: AudioSource + ?Sized,
{
    if !is_automation_value(value) {
        return None;
    }
    let resolved = evaluate_automation(value, time, Range::Spec(param_spec), external, context);
    Some(if param_spec.get("type").strict_eq_str("int") {
        js_round(resolved)
    } else {
        resolved
    })
}

// ---------------------------------------------------------------- requirements

/// One selected-device capture a compiled graph needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioRequirement {
    /// Exact device id, when the descriptor names one.
    pub id: Option<String>,
    /// Readable device name, when the descriptor names one.
    pub name: Option<String>,
    /// One-based channel.
    pub channel: u32,
    /// Whether the raw signal is read.
    pub needs_raw: bool,
}

/// `getAudioInputRequirements()` result.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AudioInputRequirements {
    /// The legacy aggregate (default input) is read.
    pub needs_legacy: bool,
    /// The aggregate raw signal is read.
    pub needs_legacy_raw: bool,
    /// Selected device channels, deduplicated by (id, name, channel).
    pub selected: Vec<AudioRequirement>,
}

/// One render pass as `getAudioInputRequirements` reads it.
#[derive(Clone, Copy, Debug)]
pub struct PassAudioInfo<'a> {
    /// `pass.uniforms`.
    pub uniforms: &'a JsValue,
    /// Whether the pass's effect is tagged `audio`.
    pub audio_tagged: bool,
}

/// `Pipeline.getAudioInputRequirements()` over a graph's passes.
pub fn audio_input_requirements<'a>(
    passes: impl IntoIterator<Item = PassAudioInfo<'a>>,
) -> AudioInputRequirements {
    let mut requirements = AudioInputRequirements::default();
    let mut visited: Vec<usize> = Vec::new();
    for pass in passes {
        if pass.audio_tagged {
            requirements.needs_legacy = true;
        }
        visit_requirements(pass.uniforms, &mut requirements, &mut visited);
    }
    requirements
}

fn visit_requirements(value: &JsValue, out: &mut AudioInputRequirements, visited: &mut Vec<usize>) {
    if !value.is_object_like() {
        return;
    }
    let id = value.identity().expect("objects have identity");
    if visited.contains(&id) {
        return;
    }
    visited.push(id);
    match value {
        JsValue::Object(object) => {
            if type_is(value, "Audio") {
                let band = value.get("band");
                let has_selector_intent = has_audio_selector_intent(value);
                let has_valid_band = !matches!(value.get("_invalid"), JsValue::Bool(true))
                    && band.is_integer()
                    && band.as_number().is_some_and(|b| (0.0..=4.0).contains(&b));
                if !has_valid_band {
                    return;
                }
                visit_requirements(value.get("min"), out, visited);
                visit_requirements(value.get("max"), out, visited);
                if has_selector_intent && is_valid_audio_selector(value) {
                    let requirement = AudioRequirement {
                        id: value
                            .get("id")
                            .as_str()
                            .filter(|s| !s.is_empty())
                            .map(str::to_string),
                        name: value.get("name").as_str().map(str::to_string),
                        channel: value.get("channel").as_number().map_or(1, |c| c as u32),
                        needs_raw: band.strict_eq_number(4.0),
                    };
                    match out.selected.iter_mut().find(|existing| {
                        existing.id == requirement.id
                            && existing.name == requirement.name
                            && existing.channel == requirement.channel
                    }) {
                        Some(existing) => existing.needs_raw |= requirement.needs_raw,
                        None => out.selected.push(requirement),
                    }
                } else if !has_selector_intent {
                    out.needs_legacy = true;
                    if band.strict_eq_number(4.0) {
                        out.needs_legacy_raw = true;
                    }
                }
                return;
            }
            for item in object.values_in_property_order() {
                visit_requirements(item, out, visited);
            }
        }
        JsValue::Array(array) => {
            for item in array.items() {
                visit_requirements(item, out, visited);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
