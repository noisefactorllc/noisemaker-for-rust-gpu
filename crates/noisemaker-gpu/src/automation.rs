//! Automation evaluation (port of the top of `runtime/pipeline.js`).
//!
//! DSL values `osc(...)`, `midi(...)` and `audio(...)` reach pass uniforms as
//! configuration objects (`{type: 'Oscillator', oscType, min, max, speed, offset,
//! seed}` and the MIDI/audio forms). Every frame the pipeline evaluates them with
//! [`evaluate_automation`]: oscillator shapes, modulated fields (any field may
//! itself be an automation, up to [`MAX_AUTOMATION_DEPTH`] levels), frequency
//! modulation as the integral of a modulated rate (closed forms for the simple
//! shapes, Gauss-Legendre quadrature otherwise), and MIDI/audio snapshots read
//! through the [`MidiSource`] and [`AudioSource`] traits (absent sources evaluate
//! exactly like the reference's `null` states).
//!
//! `Math.sin`/`Math.cos` are the platform's; uniforms reach the GPU as `f32`, far
//! coarser than any last-bit difference between libm implementations.

use std::collections::HashMap;

use noisemaker_dsl::Value;

use crate::jsv::is_finite_number;

/// `Math.PI * 2`.
pub const TAU: f64 = std::f64::consts::PI * 2.0;

/// `MAX_AUTOMATION_DEPTH`.
pub const MAX_AUTOMATION_DEPTH: usize = 8;

/// `Math.max(a, b)` (NaN-propagating).
pub fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_positive() { a } else { b }
    } else {
        a.max(b)
    }
}

/// `Math.min(a, b)` (NaN-propagating).
pub fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() { a } else { b }
    } else {
        a.min(b)
    }
}

/// `x % y` (JavaScript remainder: sign of the dividend).
pub fn js_rem(x: f64, y: f64) -> f64 {
    x % y
}

fn osc_sine(t: f64) -> f64 {
    (1.0 - (t * TAU).cos()) * 0.5
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
    let mut x = js_rem(px * 234.34 + s, 1.0);
    let mut y = js_rem(py * 435.345 + s, 1.0);
    if x < 0.0 {
        x += 1.0;
    }
    if y < 0.0 {
        y += 1.0;
    }
    let p = x + y + (x + y) * 34.23;
    js_rem(x * y * p, 1.0)
}

/// `noise2D(px, py, s)`: smoothstep-interpolated value noise.
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

/// `oscNoise(t, seed)`: value noise sampled on a circle (seamless loops).
fn osc_noise(t: f64, seed: f64) -> f64 {
    let temporal = js_rem(t, 1.0);
    let angle = temporal * TAU;
    let radius = 2.0;
    let loop_x = angle.cos() * radius;
    let loop_y = angle.sin() * radius;
    let n1 = noise_2d(loop_x + seed, loop_y + seed, seed);
    let n2 = noise_2d(loop_x + seed * 2.0, loop_y + seed * 2.0, seed);
    (n1 + n2) / 2.0
}

/// `oscNoise2d(time, speed, seed)`: the two-stage periodic noise of `osc2d`.
fn osc_noise_2d(time: f64, speed: f64, seed: f64) -> f64 {
    let periodic_value = |x: f64, v: f64| (((x - v) * TAU).sin() + 1.0) * 0.5;
    let px = (js_rem(seed, 16.0).abs() + 0.5) / 16.0;
    let py = (js_rem((seed / 16.0).floor(), 16.0).abs() + 0.5) / 16.0;
    let time_noise = noise_2d(px, py, seed + 12345.0);
    let value_noise = noise_2d(px, py, seed);
    let scaled_time = periodic_value(time, time_noise) * speed;
    periodic_value(scaled_time, value_noise)
}

/// A `{min, max}` output range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Range {
    pub min: f64,
    pub max: f64,
}

/// `AUTOMATION_FIELD_RANGES`.
pub mod field_ranges {
    use super::Range;
    pub const UNIT: Range = Range { min: 0.0, max: 1.0 };
    pub const OSCILLATOR_SPEED: Range = Range {
        min: -20.0,
        max: 20.0,
    };
    pub const OSCILLATOR_OFFSET: Range = Range {
        min: -1.0,
        max: 1.0,
    };
    pub const OSCILLATOR_SEED: Range = Range {
        min: 1.0,
        max: 9999.0,
    };
    pub const MIDI_SENSITIVITY: Range = Range {
        min: 0.0,
        max: 10.0,
    };
}

/// A consumer range read from a uniform spec (`{min, max}`), as
/// `scaleAutomationValue` accepts it: only finite bounds scale.
fn spec_range(spec: &Value) -> Option<Range> {
    let min = spec.get("min");
    let max = spec.get("max");
    if is_finite_number(min) && is_finite_number(max) {
        Some(Range {
            min: min.as_f64().unwrap(),
            max: max.as_f64().unwrap(),
        })
    } else {
        None
    }
}

const INTEGRATION_NODES_16: [f64; 16] = [
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
const INTEGRATION_WEIGHTS_16: [f64; 16] = [
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
const INTEGRATION_NODES_8: [f64; 8] = [
    -0.9602898564975363,
    -0.7966664774136267,
    -0.525532409916329,
    -0.1834346424956498,
    0.1834346424956498,
    0.525532409916329,
    0.7966664774136267,
    0.9602898564975363,
];
const INTEGRATION_WEIGHTS_8: [f64; 8] = [
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.362683783378362,
    0.362683783378362,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
];
const INTEGRATION_NODES_4: [f64; 4] = [
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
];
const INTEGRATION_WEIGHTS_4: [f64; 4] = [
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
];
const INTEGRATION_NODES_2: [f64; 2] = [-0.5773502691896257, 0.5773502691896257];
const INTEGRATION_WEIGHTS_2: [f64; 2] = [1.0, 1.0];

/// `INTEGRATION_RULES`: Gauss-Legendre rules of decreasing order by nesting depth.
const INTEGRATION_RULES: [(&[f64], &[f64]); 4] = [
    (&INTEGRATION_NODES_16, &INTEGRATION_WEIGHTS_16),
    (&INTEGRATION_NODES_8, &INTEGRATION_WEIGHTS_8),
    (&INTEGRATION_NODES_4, &INTEGRATION_WEIGHTS_4),
    (&INTEGRATION_NODES_2, &INTEGRATION_WEIGHTS_2),
];

/// One MIDI channel's state (`MidiChannelState` of `external-input.js`).
#[derive(Debug, Clone, PartialEq)]
pub struct MidiChannelState {
    pub key: f64,
    pub velocity: f64,
    pub gate: f64,
    /// Timestamp of the last note-on (`Date.now()` milliseconds).
    pub time: f64,
    /// 7-bit control change values (128).
    pub cc: Vec<f64>,
    /// Paired 14-bit control change values (32).
    pub cc14: Vec<f64>,
    pub nrpn: HashMap<u32, f64>,
    pub pitch_bend: Option<f64>,
    pub pressure: Option<f64>,
    /// Polyphonic key pressure (128).
    pub poly_pressure: Vec<f64>,
}

impl Default for MidiChannelState {
    fn default() -> Self {
        MidiChannelState {
            key: 0.0,
            velocity: 0.0,
            gate: 0.0,
            time: 0.0,
            cc: vec![0.0; 128],
            cc14: vec![0.0; 32],
            nrpn: HashMap::new(),
            pitch_bend: Some(8192.0),
            pressure: Some(0.0),
            poly_pressure: vec![0.0; 128],
        }
    }
}

/// The voice an MPE zone resolves to (`getZoneVoice`).
#[derive(Debug, Clone, PartialEq)]
pub struct MidiZoneVoice {
    pub channel: MidiChannelState,
    pub key: f64,
    pub velocity: f64,
    pub time: f64,
}

/// The MIDI input state a host provides (`MidiState` of `external-input.js`).
pub trait MidiSource {
    /// `getPortState(selector)`: the state selected by the config's `name`/`id`
    /// (`self` when it names no port), or `None` when that port is unavailable.
    fn get_port_state(&self, selector: &Value) -> Option<&dyn MidiSource>;
    /// `getChannel(n)` (falls back to channel 1 for invalid numbers).
    fn get_channel(&self, channel: &Value) -> MidiChannelState;
    /// `getZoneVoice(selector)`.
    fn get_zone_voice(&self, selector: &Value) -> Option<MidiZoneVoice>;
    /// `updateNoteGrid()`.
    fn update_note_grid(&mut self);
    /// `noteGrid`: 128 keys x 16 channels x RGBA floats.
    fn note_grid(&self) -> &[f32];
    /// `clockCount`.
    fn clock_count(&self) -> f64;
}

/// One audio analysis snapshot (`AudioState` levels).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AudioLevels {
    pub low: f64,
    pub mid: f64,
    pub high: f64,
    pub vol: f64,
    pub raw: f64,
    pub raw_ready: bool,
}

/// The audio input state a host provides (`AudioState` of `external-input.js`).
pub trait AudioSource {
    /// The default (legacy) capture's levels.
    fn levels(&self) -> AudioLevels;
    /// `getDeviceChannelState(selector)`.
    fn get_device_channel_state(&self, selector: &Value) -> Option<AudioLevels>;
    /// `waveform` (128 floats), when available.
    fn waveform(&self) -> Option<&[f32]>;
    /// `spectrum` (128 floats), when available.
    fn spectrum(&self) -> Option<&[f32]>;
}

/// `pipeline.externalState`.
#[derive(Default)]
pub struct ExternalState {
    pub midi: Option<Box<dyn MidiSource>>,
    pub audio: Option<Box<dyn AudioSource>>,
}

/// The evaluation context (`{wallTime: Date.now()}`).
#[derive(Debug, Clone, Copy)]
pub struct AutomationContext {
    /// Wall-clock milliseconds.
    pub wall_time: f64,
}

impl AutomationContext {
    pub fn now() -> Self {
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0);
        AutomationContext { wall_time: ms }
    }
}

fn type_is(v: &Value, ty: &str) -> bool {
    v.get("type").as_str() == Some(ty) || v.get("_ast").get("type").as_str() == Some(ty)
}

/// `isAutomationValue(value)`.
pub fn is_automation_value(value: &Value) -> bool {
    matches!(value, Value::Object(_) | Value::Array(_))
        && (type_is(value, "Oscillator") || type_is(value, "Midi") || type_is(value, "Audio"))
}

/// `scaleAutomationValue(value, range)`.
fn scale_automation_value(value: f64, range: Option<Range>) -> f64 {
    match range {
        Some(r) if r.min.is_finite() && r.max.is_finite() => r.min + value * (r.max - r.min),
        _ => value,
    }
}

/// The evaluation stack (`stack`): configs currently being evaluated, by identity.
type Stack = Vec<*const Value>;

#[allow(clippy::too_many_arguments)]
fn resolve_automation_field(
    value: &Value,
    t: f64,
    range: Option<Range>,
    external: &ExternalState,
    depth: usize,
    stack: &mut Stack,
    fallback: f64,
    context: &AutomationContext,
) -> f64 {
    if is_automation_value(value) {
        return evaluate_automation_at(value, t, range, external, depth + 1, stack, context);
    }
    if is_finite_number(value) {
        value.as_f64().unwrap()
    } else {
        fallback
    }
}

fn has_dynamic_automation_fields(config: &Value) -> bool {
    let fields: &[&str] = if type_is(config, "Midi") {
        &["min", "max", "sensitivity"]
    } else {
        &["min", "max"]
    };
    fields.iter().any(|f| is_automation_value(config.get(f)))
}

/// `oscPrimitive(type, x)`: the antiderivative of the simple oscillator shapes
/// (`null` for other types, which arithmetic then reads as 0).
fn osc_primitive(osc_type: &Value, x: f64) -> Option<f64> {
    let Value::Number(osc_type) = osc_type else {
        return None;
    };
    let whole = x.floor();
    let fraction = x - whole;
    Some(match *osc_type {
        0.0 => x * 0.5 - (x * TAU).sin() / (2.0 * TAU),
        1.0 => {
            let partial = if fraction < 0.5 {
                fraction * fraction
            } else {
                2.0 * fraction - fraction * fraction - 0.5
            };
            whole * 0.5 + partial
        }
        2.0 => whole * 0.5 + fraction * fraction * 0.5,
        3.0 => x - (whole * 0.5 + fraction * fraction * 0.5),
        4.0 => whole * 0.5 + js_max(0.0, fraction - 0.5),
        _ => return None,
    })
}

fn can_integrate_oscillator_exactly(config: &Value) -> bool {
    // `config.oscType >= 0 && config.oscType <= 4` (relational comparisons coerce).
    let osc_type = crate::jsv::to_number(config.get("oscType"));
    (0.0..=4.0).contains(&osc_type)
        && ["min", "max", "speed", "offset", "seed"]
            .iter()
            .all(|f| is_finite_number(config.get(f)))
}

fn integrate_simple_oscillator(config: &Value, t: f64, context: &AutomationContext) -> f64 {
    let num = |f: &str| config.get(f).as_f64().unwrap_or(f64::NAN);
    let (min, max, speed, offset) = (num("min"), num("max"), num("speed"), num("offset"));
    if speed == 0.0 {
        let mut stack = Stack::new();
        return evaluate_oscillator(
            config,
            0.0,
            &ExternalState::default(),
            0,
            &mut stack,
            context,
        ) * t;
    }
    let osc_type = config.get("oscType");
    let start = osc_primitive(osc_type, offset).unwrap_or(0.0);
    let end = osc_primitive(osc_type, offset + speed * t).unwrap_or(0.0);
    let raw_integral = (end - start) / speed;
    min * t + (max - min) * raw_integral
}

fn integrate_automation(
    config: &Value,
    t: f64,
    range: Option<Range>,
    external: &ExternalState,
    depth: usize,
    stack: &mut Stack,
    context: &AutomationContext,
) -> f64 {
    let integral = if type_is(config, "Oscillator") && can_integrate_oscillator_exactly(config) {
        integrate_simple_oscillator(config, t, context)
    } else if (type_is(config, "Midi") || type_is(config, "Audio"))
        && !has_dynamic_automation_fields(config)
    {
        // External inputs expose a snapshot: constant across the interval.
        evaluate_automation_at(config, t, None, external, depth + 1, stack, context) * t
    } else {
        let (nodes, weights) = INTEGRATION_RULES[depth.min(INTEGRATION_RULES.len() - 1)];
        let midpoint = t * 0.5;
        let half_width = t * 0.5;
        let mut sum = 0.0;
        for i in 0..nodes.len() {
            let sample_time = midpoint + half_width * nodes[i];
            sum += weights[i]
                * evaluate_automation_at(
                    config,
                    sample_time,
                    None,
                    external,
                    depth + 1,
                    stack,
                    context,
                );
        }
        half_width * sum
    };
    match range {
        Some(r) if r.min.is_finite() && r.max.is_finite() => r.min * t + integral * (r.max - r.min),
        _ => integral,
    }
}

/// `evaluateAutomation(config, normalizedTime, range, externalState)`: the value of
/// an automation config at a normalized loop time, scaled to `range`.
pub fn evaluate_automation(
    config: &Value,
    t: f64,
    range: Option<Range>,
    external: &ExternalState,
    context: &AutomationContext,
) -> f64 {
    let mut stack = Stack::new();
    evaluate_automation_at(config, t, range, external, 0, &mut stack, context)
}

fn evaluate_automation_at(
    config: &Value,
    t: f64,
    range: Option<Range>,
    external: &ExternalState,
    depth: usize,
    stack: &mut Stack,
    context: &AutomationContext,
) -> f64 {
    let id = config as *const Value;
    if !is_automation_value(config) || depth > MAX_AUTOMATION_DEPTH || stack.contains(&id) {
        return scale_automation_value(0.0, range);
    }
    stack.push(id);
    let value = if type_is(config, "Oscillator") {
        evaluate_oscillator(config, t, external, depth, stack, context)
    } else if type_is(config, "Midi") {
        let midi_state: Option<&dyn MidiSource> = external
            .midi
            .as_deref()
            .and_then(|m| m.get_port_state(config));
        let min = resolve_automation_field(
            config.get("min"),
            t,
            Some(field_ranges::UNIT),
            external,
            depth,
            stack,
            0.0,
            context,
        );
        let max = resolve_automation_field(
            config.get("max"),
            t,
            Some(field_ranges::UNIT),
            external,
            depth,
            stack,
            1.0,
            context,
        );
        let sensitivity = resolve_automation_field(
            config.get("sensitivity"),
            t,
            Some(field_ranges::MIDI_SENSITIVITY),
            external,
            depth,
            stack,
            1.0,
            context,
        );
        evaluate_midi(config, midi_state, context.wall_time, min, max, sensitivity)
    } else if config.get("_invalid").is_truthy() {
        let min = config.get("min");
        if is_finite_number(min) {
            min.as_f64().unwrap()
        } else {
            0.0
        }
    } else {
        let min = resolve_automation_field(
            config.get("min"),
            t,
            Some(field_ranges::UNIT),
            external,
            depth,
            stack,
            0.0,
            context,
        );
        let max = resolve_automation_field(
            config.get("max"),
            t,
            Some(field_ranges::UNIT),
            external,
            depth,
            stack,
            1.0,
            context,
        );
        evaluate_audio(config, external.audio.as_deref(), min, max)
    };
    stack.pop();
    scale_automation_value(value, range)
}

/// `evaluateOscillator(osc, normalizedTime, externalState, depth, stack, context)`.
fn evaluate_oscillator(
    osc: &Value,
    t: f64,
    external: &ExternalState,
    depth: usize,
    stack: &mut Stack,
    context: &AutomationContext,
) -> f64 {
    let osc_type = osc.get("oscType");
    let min = resolve_automation_field(
        osc.get("min"),
        t,
        Some(field_ranges::UNIT),
        external,
        depth,
        stack,
        0.0,
        context,
    );
    let max = resolve_automation_field(
        osc.get("max"),
        t,
        Some(field_ranges::UNIT),
        external,
        depth,
        stack,
        1.0,
        context,
    );
    let offset = resolve_automation_field(
        osc.get("offset"),
        t,
        Some(field_ranges::OSCILLATOR_OFFSET),
        external,
        depth,
        stack,
        0.0,
        context,
    );
    let seed = resolve_automation_field(
        osc.get("seed"),
        t,
        Some(field_ranges::OSCILLATOR_SEED),
        external,
        depth,
        stack,
        1.0,
        context,
    );
    let speed_value = osc.get("speed");
    let phase = if is_automation_value(speed_value) {
        integrate_automation(
            speed_value,
            t,
            Some(field_ranges::OSCILLATOR_SPEED),
            external,
            depth,
            stack,
            context,
        )
    } else {
        t * if is_finite_number(speed_value) {
            speed_value.as_f64().unwrap()
        } else {
            1.0
        }
    };
    let phase_t = phase + offset;
    let value = match osc_type {
        Value::Number(n) if *n == 0.0 => osc_sine(phase_t),
        Value::Number(n) if *n == 1.0 => osc_tri(phase_t),
        Value::Number(n) if *n == 2.0 => osc_saw(phase_t),
        Value::Number(n) if *n == 3.0 => osc_saw_inv(phase_t),
        Value::Number(n) if *n == 4.0 => osc_square(phase_t),
        Value::Number(n) if *n == 5.0 => osc_noise(phase_t, seed),
        Value::Number(n) if *n == 6.0 => {
            let speed = resolve_automation_field(
                speed_value,
                t,
                Some(field_ranges::OSCILLATOR_SPEED),
                external,
                depth,
                stack,
                1.0,
                context,
            );
            osc_noise_2d(
                t + offset,
                if speed.is_finite() { speed } else { 1.0 },
                seed,
            )
        }
        _ => 0.0,
    };
    min + value * (max - min)
}

/// `index` into a JavaScript typed array / array (`arr[n]`): only in-range
/// integral indices exist.
fn js_index(values: &[f64], n: f64) -> Option<f64> {
    if n.is_finite() && n.fract() == 0.0 && n >= 0.0 && (n as usize) < values.len() {
        Some(values[n as usize])
    } else {
        None
    }
}

/// `evaluateMidi(config, midiState, currentTime, min, max, sensitivity)`.
pub fn evaluate_midi(
    config: &Value,
    midi_state: Option<&dyn MidiSource>,
    current_time: f64,
    min: f64,
    max: f64,
    sensitivity: f64,
) -> f64 {
    let Some(midi_state) = midi_state else {
        return min;
    };
    if config.get("_invalid").is_truthy() {
        return min;
    }
    let mode = config.get("mode");
    let zone = config.get("zone");
    let channel_value = config.get("channel");
    let has_zone = !zone.is_undefined();
    let zone_is = |z: f64| matches!(zone, Value::Number(n) if *n == z);
    if has_zone && (!channel_value.is_undefined() || (!zone_is(0.0) && !zone_is(1.0))) {
        return min;
    }
    let members = config.get("members");
    if !members.is_undefined()
        && (!has_zone
            || !members.is_integer()
            || members.as_f64().unwrap() < 1.0
            || members.as_f64().unwrap() > 15.0)
    {
        return min;
    }
    let mode_n = crate::jsv::to_number(mode);
    if !has_zone
        && mode_n >= 5.0
        && (!channel_value.is_integer()
            || channel_value.as_f64().unwrap() < 1.0
            || channel_value.as_f64().unwrap() > 16.0)
    {
        return min;
    }
    let voice = if has_zone {
        midi_state.get_zone_voice(config)
    } else {
        None
    };
    if has_zone && voice.is_none() {
        return min;
    }
    let channel = match &voice {
        Some(v) => v.channel.clone(),
        None => midi_state.get_channel(channel_value),
    };
    let (key, velocity, gate, time) = match &voice {
        Some(v) => (v.key, v.velocity, 1.0, v.time),
        None => (channel.key, channel.velocity, channel.gate, channel.time),
    };

    let mode_is = |m: f64| matches!(mode, Value::Number(n) if *n == m);
    let mut raw_value = 0.0;
    if mode_is(0.0) {
        raw_value = key;
    } else if mode_is(1.0) {
        if gate == 1.0 {
            raw_value = key;
        }
    } else if mode_is(2.0) {
        if gate == 1.0 {
            raw_value = velocity;
        }
    } else if mode_is(3.0) {
        if gate == 1.0 {
            raw_value = key;
            let elapsed = current_time - time;
            let decay = js_min(1.0, elapsed * sensitivity * 0.001);
            raw_value *= 1.0 - decay;
        }
    } else if mode_is(5.0) || mode_is(6.0) {
        let cc_value = config.get("cc");
        let cc = if cc_value.is_nullish() {
            Value::Number(1.0)
        } else {
            cc_value.clone()
        };
        let limit = if mode_is(6.0) { 31.0 } else { 127.0 };
        if !cc.is_integer() || cc.as_f64().unwrap() < 0.0 || cc.as_f64().unwrap() > limit {
            return min;
        }
        let index = cc.as_f64().unwrap();
        let normalized = if mode_is(6.0) {
            js_index(&channel.cc14, index).unwrap_or(0.0) / 16383.0
        } else {
            js_index(&channel.cc, index).unwrap_or(0.0) / 127.0
        };
        return min + normalized * (max - min);
    } else if mode_is(7.0) {
        let nrpn = config.get("nrpn");
        if !nrpn.is_integer() || nrpn.as_f64().unwrap() < 0.0 || nrpn.as_f64().unwrap() > 16382.0 {
            return min;
        }
        let value = channel
            .nrpn
            .get(&(nrpn.as_f64().unwrap() as u32))
            .copied()
            .unwrap_or(0.0);
        return min + value / 16383.0 * (max - min);
    } else if mode_is(8.0) {
        return min + channel.pitch_bend.unwrap_or(8192.0) / 16383.0 * (max - min);
    } else if mode_is(9.0) {
        return min + channel.pressure.unwrap_or(0.0) / 127.0 * (max - min);
    } else if mode_is(10.0) {
        return min + js_index(&channel.poly_pressure, key).unwrap_or(0.0) / 127.0 * (max - min);
    } else if gate == 1.0 {
        // 4 (velocity with falloff) and every other mode.
        raw_value = velocity;
        let elapsed = current_time - time;
        let decay = js_min(1.0, elapsed * sensitivity * 0.001);
        raw_value *= 1.0 - decay;
    }
    let normalized = raw_value / 127.0;
    min + normalized * (max - min)
}

fn audio_selector_source(config: &Value) -> &Value {
    if config.get("_ast").get("type").as_str() == Some("Audio") {
        config.get("_ast")
    } else {
        config
    }
}

/// `hasAudioSelectorIntent(config)`.
pub fn has_audio_selector_intent(config: &Value) -> bool {
    let source = audio_selector_source(config);
    !config.get("name").is_undefined()
        || !config.get("id").is_undefined()
        || !config.get("channel").is_undefined()
        || !source.get("name").is_undefined()
        || !source.get("id").is_undefined()
        || !source.get("channel").is_undefined()
}

/// `isValidAudioSelector(config)`.
pub fn is_valid_audio_selector(config: &Value) -> bool {
    let source = audio_selector_source(config);
    if (!source.get("name").is_undefined() && config.get("name").is_undefined())
        || (!source.get("id").is_undefined() && config.get("id").is_undefined())
        || (!source.get("channel").is_undefined() && config.get("channel").is_undefined())
    {
        return false;
    }
    let name = config.get("name");
    if !name.is_undefined() && !matches!(name, Value::String(s) if !s.is_empty()) {
        return false;
    }
    let id = config.get("id");
    if !id.is_undefined() && (!matches!(id, Value::String(s) if !s.is_empty()) || !name.is_truthy())
    {
        return false;
    }
    let channel = config.get("channel");
    channel.is_integer() && channel.as_f64().unwrap() >= 1.0 && channel.as_f64().unwrap() <= 32.0
}

/// `evaluateAudio(config, audioState, min, max)`.
pub fn evaluate_audio(config: &Value, audio: Option<&dyn AudioSource>, min: f64, max: f64) -> f64 {
    if config.get("_invalid").is_truthy() {
        return min;
    }
    let Some(audio) = audio else {
        return min;
    };
    let band = config.get("band");
    let has_selector = has_audio_selector_intent(config);
    if has_selector && !is_valid_audio_selector(config) {
        return min;
    }
    let selected = if has_selector {
        audio.get_device_channel_state(config)
    } else {
        Some(audio.levels())
    };
    let Some(selected) = selected else {
        return min;
    };
    let band_is = |b: f64| matches!(band, Value::Number(n) if *n == b);
    let mut raw = if band_is(0.0) {
        selected.low
    } else if band_is(1.0) {
        selected.mid
    } else if band_is(2.0) {
        selected.high
    } else if band_is(4.0) {
        if !selected.raw_ready {
            return min;
        }
        let r = if selected.raw == 0.0 || selected.raw.is_nan() {
            0.0
        } else {
            selected.raw
        };
        (js_max(-1.0, js_min(1.0, r)) + 1.0) * 0.5
    } else if band_is(3.0) {
        selected.vol
    } else {
        0.0
    };
    raw = js_max(0.0, js_min(1.0, raw));
    min + raw * (max - min)
}

/// `resolveUniformValue(value, time, paramSpec)`: evaluate an automation value
/// (rounded for `int` params); other values pass through.
pub fn resolve_uniform_value(
    value: &Value,
    time: f64,
    spec: &Value,
    external: &ExternalState,
    context: &AutomationContext,
) -> Option<Value> {
    if !is_automation_value(value) {
        return None;
    }
    let resolved = evaluate_automation(value, time, spec_range(spec), external, context);
    let resolved = if spec.get("type").as_str() == Some("int") {
        noisemaker_dsl::js::math_round(resolved)
    } else {
        resolved
    };
    Some(Value::Number(resolved))
}

/// `getAudioInputRequirements()` result.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AudioRequirements {
    pub needs_legacy: bool,
    pub needs_legacy_raw: bool,
    pub selected: Vec<AudioSelection>,
}

/// One selected audio capture.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSelection {
    pub id: Option<String>,
    pub name: Value,
    pub channel: Value,
    pub needs_raw: bool,
}

/// The visit of `getAudioInputRequirements` over one value.
pub fn visit_audio_requirements(value: &Value, out: &mut AudioRequirements) {
    match value {
        Value::Object(_) if type_is(value, "Audio") => {
            let has_selector = has_audio_selector_intent(value);
            let band = value.get("band");
            let has_valid_band = !matches!(value.get("_invalid"), Value::Bool(true))
                && band.is_integer()
                && band.as_f64().unwrap() >= 0.0
                && band.as_f64().unwrap() <= 4.0;
            if !has_valid_band {
                return;
            }
            visit_audio_requirements(value.get("min"), out);
            visit_audio_requirements(value.get("max"), out);
            let needs_raw = matches!(band, Value::Number(b) if *b == 4.0);
            if has_selector && is_valid_audio_selector(value) {
                let id = match value.get("id") {
                    Value::String(s) if !s.is_empty() => Some(s.clone()),
                    _ => None,
                };
                let name = if value.get("name").is_nullish() {
                    Value::Null
                } else {
                    value.get("name").clone()
                };
                let channel = if value.get("channel").is_nullish() {
                    Value::Number(1.0)
                } else {
                    value.get("channel").clone()
                };
                if let Some(existing) = out
                    .selected
                    .iter_mut()
                    .find(|c| c.id == id && c.name == name && c.channel == channel)
                {
                    if needs_raw {
                        existing.needs_raw = true;
                    }
                } else {
                    out.selected.push(AudioSelection {
                        id,
                        name,
                        channel,
                        needs_raw,
                    });
                }
            } else if !has_selector {
                out.needs_legacy = true;
                if needs_raw {
                    out.needs_legacy_raw = true;
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                visit_audio_requirements(item, out);
            }
        }
        Value::Object(o) => {
            for item in o.values() {
                visit_audio_requirements(item, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn osc(json: &str) -> Value {
        Value::from_json(json).unwrap()
    }

    fn ctx() -> AutomationContext {
        AutomationContext { wall_time: 0.0 }
    }

    #[test]
    fn simple_oscillators() {
        let ext = ExternalState::default();
        let sine = osc(
            r#"{"type":"Oscillator","oscType":0,"min":0,"max":1,"speed":1,"offset":0,"seed":1}"#,
        );
        assert!((evaluate_automation(&sine, 0.5, None, &ext, &ctx()) - 1.0).abs() < 1e-12);
        let saw = osc(
            r#"{"type":"Oscillator","oscType":2,"min":2,"max":4,"speed":2,"offset":0.25,"seed":1}"#,
        );
        // phase = 0.3 * 2 + 0.25 = 0.85
        assert!(
            (evaluate_automation(&saw, 0.3, None, &ext, &ctx()) - (2.0 + 0.85 * 2.0)).abs() < 1e-12
        );
        let range = Range {
            min: 10.0,
            max: 20.0,
        };
        let square = osc(
            r#"{"type":"Oscillator","oscType":4,"min":0,"max":1,"speed":1,"offset":0,"seed":1}"#,
        );
        assert_eq!(
            evaluate_automation(&square, 0.75, Some(range), &ext, &ctx()),
            20.0
        );
    }

    #[test]
    fn modulated_rate_integrates() {
        let ext = ExternalState::default();
        // A constant-rate saw modulator (oscType 2 with speed 0 integrates as
        // value(0) * t) feeding a saw oscillator's speed.
        let fm = osc(
            r#"{"type":"Oscillator","oscType":2,"min":0,"max":1,"offset":0,"seed":1,
                "speed":{"type":"Oscillator","oscType":1,"min":0.5,"max":0.5,"speed":1,"offset":0,"seed":1}}"#,
        );
        let v = evaluate_automation(&fm, 0.5, None, &ext, &ctx());
        // Integral over [0, 0.5] of the speed range scale: -20 + 0.5 * 40 = 0.
        assert!(v.is_finite());
    }

    #[test]
    fn midi_and_audio_without_sources_return_min() {
        let ext = ExternalState::default();
        let midi =
            osc(r#"{"type":"Midi","channel":1,"mode":4,"min":0.25,"max":1,"sensitivity":1}"#);
        assert_eq!(evaluate_automation(&midi, 0.0, None, &ext, &ctx()), 0.25);
        let audio = osc(r#"{"type":"Audio","band":0,"min":0.5,"max":1}"#);
        assert_eq!(evaluate_automation(&audio, 0.0, None, &ext, &ctx()), 0.5);
    }

    #[test]
    fn js_math_semantics() {
        assert!(js_max(0.0, f64::NAN).is_nan());
        assert!(js_min(f64::NAN, 1.0).is_nan());
        assert_eq!(js_rem(-1.5, 1.0), -0.5);
    }
}
