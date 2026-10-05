//! Unit tests ported from the runtime assertions of the reference's
//! `test_midi.js`, `test_audio.js`, `test_oscillators.js`,
//! `test_nested_automation.js` and `test_midi_audio_integration.js`. Exact
//! parity over many generated descriptors is gated by
//! parity/check_automation.mjs.

use std::sync::Arc;

use super::*;
use crate::audio::ChannelValues;
use crate::clock::ManualClock;
use crate::midi::MidiPortRef;

const NOW: f64 = 1_700_000_000_000.0;

fn obj(props: &[(&str, JsValue)]) -> JsValue {
    JsValue::object(props.iter().map(|(k, v)| (k.to_string(), v.clone())))
}

fn n(x: f64) -> JsValue {
    JsValue::Number(x)
}

fn s(x: &str) -> JsValue {
    JsValue::string(x)
}

fn midi_config(extra: &[(&str, JsValue)]) -> JsValue {
    let mut props = vec![
        ("type", s("Midi")),
        ("min", n(0.0)),
        ("max", n(1.0)),
        ("sensitivity", n(1.0)),
    ];
    props.extend(extra.iter().cloned());
    obj(&props)
}

fn audio_config(extra: &[(&str, JsValue)]) -> JsValue {
    let mut props = vec![("type", s("Audio")), ("min", n(0.0)), ("max", n(1.0))];
    props.extend(extra.iter().cloned());
    obj(&props)
}

fn resolve(
    value: &JsValue,
    time: f64,
    midi: Option<&MidiState>,
    audio: Option<&AudioState>,
) -> f64 {
    resolve_spec(value, time, &JsValue::Undefined, midi, audio)
}

fn resolve_spec(
    value: &JsValue,
    time: f64,
    spec: &JsValue,
    midi: Option<&MidiState>,
    audio: Option<&AudioState>,
) -> f64 {
    resolve_uniform_value(
        value,
        time,
        spec,
        ExternalState::new(midi, audio),
        AutomationContext::at(NOW),
    )
    .expect("automation value")
}

fn approx(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected}"
    );
}

fn midi_state() -> MidiState {
    MidiState::with_clock(Arc::new(ManualClock::new(NOW)), true)
}

#[test]
fn non_automation_values_pass_through() {
    let none = ExternalState::none();
    for value in [
        n(42.0),
        s("test"),
        JsValue::Null,
        obj(&[("type", s("Other"))]),
    ] {
        assert!(
            resolve_uniform_value(
                &value,
                0.0,
                &JsValue::Undefined,
                none,
                AutomationContext::at(NOW)
            )
            .is_none()
        );
    }
}

#[test]
fn midi_note_modes() {
    let mut midi = midi_state();
    midi.channels[0].key = 60;
    midi.channels[0].gate = 0;
    approx(
        resolve(
            &midi_config(&[("channel", n(1.0)), ("mode", n(0.0))]),
            0.0,
            Some(&midi),
            None,
        ),
        60.0 / 127.0,
        1e-15,
    );
    midi.channels[0].key = 127;
    let ranged = midi_config(&[("channel", n(1.0)), ("mode", n(0.0)), ("max", n(10.0))]);
    approx(resolve(&ranged, 0.0, Some(&midi), None), 10.0, 1e-12);
    let gate_note = midi_config(&[
        ("channel", n(1.0)),
        ("mode", n(1.0)),
        ("min", n(5.0)),
        ("max", n(10.0)),
    ]);
    assert_eq!(resolve(&gate_note, 0.0, Some(&midi), None), 5.0);
    midi.channels[0].note_on_at(64, 100, NOW);
    approx(
        resolve(
            &midi_config(&[("channel", n(1.0)), ("mode", n(1.0))]),
            0.0,
            Some(&midi),
            None,
        ),
        64.0 / 127.0,
        1e-15,
    );
    approx(
        resolve(
            &midi_config(&[("channel", n(1.0)), ("mode", n(2.0))]),
            0.0,
            Some(&midi),
            None,
        ),
        100.0 / 127.0,
        1e-15,
    );
}

#[test]
fn midi_trigger_modes_decay_with_wall_time() {
    let mut midi = midi_state();
    midi.channels[0].key = 127;
    midi.channels[0].velocity = 127;
    midi.channels[0].gate = 1;
    midi.channels[0].time = NOW - 500.0;
    let trigger = midi_config(&[("channel", n(1.0)), ("mode", n(3.0))]);
    approx(resolve(&trigger, 0.0, Some(&midi), None), 0.5, 1e-12);
    let fast = midi_config(&[
        ("channel", n(1.0)),
        ("mode", n(3.0)),
        ("sensitivity", n(4.0)),
    ]);
    assert_eq!(resolve(&fast, 0.0, Some(&midi), None), 0.0);
    midi.channels[0].time = NOW - 1000.0;
    assert_eq!(
        resolve(
            &midi_config(&[("channel", n(1.0)), ("mode", n(4.0))]),
            0.0,
            Some(&midi),
            None
        ),
        0.0
    );
    midi.channels[0].time = NOW;
    assert_eq!(
        resolve(
            &midi_config(&[("channel", n(1.0)), ("mode", n(4.0))]),
            0.0,
            Some(&midi),
            None
        ),
        1.0
    );
}

#[test]
fn midi_without_state_returns_min() {
    let config = midi_config(&[
        ("channel", n(1.0)),
        ("mode", n(4.0)),
        ("min", n(5.0)),
        ("max", n(10.0)),
    ]);
    assert_eq!(resolve(&config, 0.0, None, None), 5.0);
}

#[test]
fn cc_holds_independently_of_notes_channels_and_ports() {
    let mut midi = midi_state();
    let left = Some(MidiPortRef::new("left", "Controller"));
    let right = Some(MidiPortRef::new("right", "Controller"));
    let config = midi_config(&[
        ("channel", n(2.0)),
        ("mode", n(5.0)),
        ("cc", n(74.0)),
        ("min", n(0.2)),
        ("max", n(0.8)),
        ("id", s("left")),
        ("name", s("Controller")),
    ]);
    midi.handle_message(&[0xb1, 74, 127], left);
    midi.handle_message(&[0xb0, 74, 12], left);
    midi.handle_message(&[0xb1, 74, 8], right);
    midi.handle_message(&[0x91, 60, 30], left);
    midi.handle_message(&[0x81, 60, 0], left);
    approx(resolve(&config, 0.0, Some(&midi), None), 0.8, 1e-12);
    midi.handle_message(&[0xb1, 74, 64], left);
    // Node: 0.2 + 64 / 127 * (0.8 - 0.2)
    assert_eq!(resolve(&config, 0.0, Some(&midi), None), 0.5023622047244095);
    midi.handle_message(&[0xb1, 74, 0], left);
    approx(resolve(&config, 0.0, Some(&midi), None), 0.2, 1e-15);
}

#[test]
fn cc_defaults_to_controller_one_and_invalid_selectors_are_inert() {
    let mut midi = midi_state();
    midi.handle_message(&[0xb0, 1, 127], None);
    midi.handle_message(&[0x90, 60, 127], None);
    assert_eq!(
        resolve(
            &midi_config(&[("channel", n(1.0)), ("mode", n(5.0))]),
            0.0,
            Some(&midi),
            None
        ),
        1.0
    );
    for (mode, cc) in [
        (5.0, n(128.0)),
        (6.0, n(32.0)),
        (6.0, n(-1.0)),
        (5.0, n(0.5)),
    ] {
        let config = midi_config(&[
            ("channel", n(1.0)),
            ("mode", n(mode)),
            ("cc", cc),
            ("min", n(0.2)),
        ]);
        assert_eq!(resolve(&config, 0.0, Some(&midi), None), 0.2);
    }
    for mode in [5.0, 6.0] {
        for channel in [
            n(0.0),
            n(17.0),
            n(1.5),
            JsValue::Bool(true),
            s("1"),
            JsValue::Undefined,
        ] {
            let config = midi_config(&[("channel", channel), ("mode", n(mode)), ("min", n(0.2))]);
            assert_eq!(resolve(&config, 0.0, Some(&midi), None), 0.2);
        }
    }
}

#[test]
fn expression_modes_decode_wire_values() {
    let mut midi = midi_state();
    let port = Some(MidiPortRef::new("expressive", "Expressive"));
    midi.register_port("expressive", "Expressive");
    let read = |midi: &MidiState, mode: f64| {
        let config = midi_config(&[
            ("channel", n(2.0)),
            ("mode", n(mode)),
            ("id", s("expressive")),
            ("name", s("Expressive")),
        ]);
        resolve(&config, 0.0, Some(midi), None)
    };
    approx(read(&midi, 8.0), 8192.0 / 16383.0, 1e-15);
    midi.handle_message(&[0xe1, 127, 127], port);
    midi.handle_message(&[0xd1, 91], port);
    midi.handle_message(&[0x91, 60, 100], port);
    midi.handle_message(&[0xa1, 60, 63], port);
    assert_eq!(read(&midi, 8.0), 1.0);
    approx(read(&midi, 9.0), 91.0 / 127.0, 1e-15);
    approx(read(&midi, 10.0), 63.0 / 127.0, 1e-15);
    midi.handle_message(&[0x81, 60, 0], port);
    assert_eq!(read(&midi, 10.0), 0.0);
}

#[test]
fn invalid_expression_selectors_stay_inert() {
    let mut midi = midi_state();
    midi.handle_message(&[0x91, 60, 100], None);
    midi.handle_message(&[0xd1, 127], None);
    let cases: Vec<Vec<(&str, JsValue)>> = vec![
        vec![("channel", n(17.0)), ("mode", n(8.0))],
        vec![("channel", n(0.0)), ("mode", n(9.0))],
        vec![("channel", n(1.5)), ("mode", n(10.0))],
        vec![("channel", n(2.0)), ("mode", n(7.0)), ("nrpn", n(16383.0))],
        vec![("zone", n(0.0)), ("channel", n(2.0)), ("mode", n(9.0))],
        vec![("zone", n(2.0)), ("mode", n(9.0))],
        vec![("zone", n(0.0)), ("members", n(16.0)), ("mode", n(9.0))],
        vec![("channel", n(2.0)), ("members", n(2.0)), ("mode", n(9.0))],
    ];
    for fields in cases {
        let mut props = vec![("type", s("Midi")), ("min", n(0.2)), ("max", n(1.0))];
        props.extend(fields);
        assert_eq!(resolve(&obj(&props), 0.0, Some(&midi), None), 0.2);
    }
}

#[test]
fn mpe_zones_select_the_newest_voice_across_ports() {
    let clock = Arc::new(ManualClock::new(NOW));
    let mut midi = MidiState::with_clock(clock.clone(), true);
    let a = Some(MidiPortRef::new("mpe-a", "MPE"));
    let b = Some(MidiPortRef::new("mpe-b", "MPE"));
    let read = |midi: &MidiState, mode: f64, port: Option<(&str, &str)>| {
        let mut props = vec![
            ("type", s("Midi")),
            ("zone", n(0.0)),
            ("mode", n(mode)),
            ("cc", n(74.0)),
            ("min", n(0.0)),
            ("max", n(1.0)),
        ];
        if let Some((id, name)) = port {
            props.push(("id", s(id)));
            props.push(("name", s(name)));
        }
        resolve(&obj(&props), 0.0, Some(midi), None)
    };
    let pa = Some(("mpe-a", "MPE"));
    midi.handle_message(&[0xe1, 0, 96], a);
    midi.handle_message(&[0x91, 60, 80], a);
    midi.handle_message(&[0xa1, 60, 40], a);
    approx(read(&midi, 8.0, pa), 12288.0 / 16383.0, 1e-15);
    midi.handle_message(&[0x91, 62, 100], a);
    midi.handle_message(&[0xa1, 62, 70], a);
    approx(read(&midi, 0.0, pa), 62.0 / 127.0, 1e-15);
    midi.handle_message(&[0x81, 62, 0], a);
    approx(read(&midi, 0.0, pa), 60.0 / 127.0, 1e-15);
    approx(read(&midi, 10.0, pa), 40.0 / 127.0, 1e-15);
    midi.handle_message(&[0x92, 67, 110], a);
    midi.handle_message(&[0xb2, 74, 100], a);
    midi.handle_message(&[0x92, 70, 120], b);
    midi.handle_message(&[0xb2, 74, 10], b);
    approx(read(&midi, 5.0, None), 10.0 / 127.0, 1e-15);
    approx(read(&midi, 5.0, pa), 100.0 / 127.0, 1e-15);
    midi.handle_message(&[0x82, 70, 0], b);
    approx(read(&midi, 0.0, None), 67.0 / 127.0, 1e-15);
    midi.handle_message(&[0xb2, 64, 127], a);
    midi.handle_message(&[0x92, 67, 0], a);
    approx(read(&midi, 0.0, pa), 60.0 / 127.0, 1e-15);
    midi.handle_message(&[0xb1, 123, 0], a);
    assert_eq!(read(&midi, 8.0, pa), 0.0);
    midi.handle_message(&[0x91, 64, 100], a);
    midi.handle_message(&[0xb1, 120, 0], a);
    assert_eq!(read(&midi, 0.0, pa), 0.0);
}

#[test]
fn audio_bands_scale_and_clamp() {
    let mut audio = AudioState::new();
    audio.low = 0.7;
    approx(
        resolve(&audio_config(&[("band", n(0.0))]), 0.0, None, Some(&audio)),
        0.7,
        1e-15,
    );
    audio.low = 0.5;
    let ranged = audio_config(&[("band", n(0.0)), ("min", n(0.2)), ("max", n(0.8))]);
    approx(resolve(&ranged, 0.0, None, Some(&audio)), 0.5, 1e-15);
    audio.high = 1.5;
    assert_eq!(
        resolve(&audio_config(&[("band", n(2.0))]), 0.0, None, Some(&audio)),
        1.0
    );
    let spec = obj(&[("min", n(10.0)), ("max", n(50.0))]);
    approx(
        resolve_spec(
            &audio_config(&[("band", n(0.0))]),
            0.0,
            &spec,
            None,
            Some(&audio),
        ),
        30.0,
        1e-12,
    );
    let int_spec = obj(&[("min", n(0.0)), ("max", n(5.0)), ("type", s("int"))]);
    assert_eq!(
        resolve_spec(
            &audio_config(&[("band", n(0.0))]),
            0.0,
            &int_spec,
            None,
            Some(&audio)
        ),
        3.0
    );
    assert_eq!(
        resolve(&audio_config(&[("band", n(0.0))]), 0.0, None, None),
        0.0
    );
    let invalid = audio_config(&[
        ("band", n(0.0)),
        ("min", n(0.2)),
        ("max", n(0.9)),
        ("_invalid", JsValue::Bool(true)),
    ]);
    approx(resolve(&invalid, 0.0, None, Some(&audio)), 0.2, 1e-15);
}

#[test]
fn raw_audio_needs_a_real_sample() {
    let mut audio = AudioState::new();
    let config = audio_config(&[("band", n(4.0)), ("min", n(0.2)), ("max", n(0.8))]);
    approx(resolve(&config, 0.0, None, Some(&audio)), 0.2, 1e-15);
    audio.set_raw(0.0);
    approx(resolve(&config, 0.0, None, Some(&audio)), 0.5, 1e-15);
    audio.set_raw(1.0);
    approx(resolve(&config, 0.0, None, Some(&audio)), 0.8, 1e-15);
    audio.set_raw_unavailable();
    approx(resolve(&config, 0.0, None, Some(&audio)), 0.2, 1e-15);
}

#[test]
fn selected_audio_channels_are_exact() {
    let mut audio = AudioState::new();
    audio.low = 0.1;
    audio.register_device("a", "Interface", Some(2));
    audio.register_device("b", "Interface", Some(2));
    audio.set_channel_values(
        "a",
        2,
        &ChannelValues {
            low: Some(0.35),
            raw: Some(-0.5),
            ..Default::default()
        },
    );
    audio.set_channel_values(
        "b",
        2,
        &ChannelValues {
            low: Some(0.85),
            raw: Some(0.5),
            ..Default::default()
        },
    );
    let selected = audio_config(&[
        ("band", n(0.0)),
        ("channel", n(2.0)),
        ("name", s("Interface")),
        ("id", s("b")),
    ]);
    approx(resolve(&selected, 0.0, None, Some(&audio)), 0.85, 1e-15);
    let raw = audio_config(&[
        ("band", n(4.0)),
        ("channel", n(2.0)),
        ("name", s("Interface")),
        ("id", s("b")),
    ]);
    approx(resolve(&raw, 0.0, None, Some(&audio)), 0.75, 1e-15);
    approx(
        resolve(&audio_config(&[("band", n(0.0))]), 0.0, None, Some(&audio)),
        0.1,
        1e-15,
    );
    let ambiguous = audio_config(&[
        ("band", n(0.0)),
        ("channel", n(1.0)),
        ("name", s("Interface")),
        ("min", n(0.25)),
    ]);
    approx(resolve(&ambiguous, 0.0, None, Some(&audio)), 0.25, 1e-15);
}

#[test]
fn audio_channel_32_is_distinct() {
    let mut audio = AudioState::new();
    audio.register_default_channels(32);
    audio
        .get_default_channel_state_mut(31.0)
        .unwrap()
        .set_raw(-0.6);
    audio
        .get_default_channel_state_mut(32.0)
        .unwrap()
        .set_raw(0.8);
    let config = audio_config(&[("band", n(4.0)), ("channel", n(32.0))]);
    approx(resolve(&config, 0.0, None, Some(&audio)), 0.9, 1e-15);
    let config = audio_config(&[("band", n(4.0)), ("channel", n(31.0))]);
    approx(resolve(&config, 0.0, None, Some(&audio)), 0.2, 1e-15);
}

// Pinned values of the reference's `test_oscillators.js`.
#[test]
fn oscillator_kinds_keep_their_pinned_outputs() {
    let pins: [(f64, f64, [f64; 4]); 7] = [
        (7.0, 0.0, [0.0, 0.5, 1.0, 0.5]),
        (7.0, 1.0, [0.0, 0.5, 1.0, 0.5]),
        (7.0, 2.0, [0.0, 0.25, 0.5, 0.75]),
        (7.0, 3.0, [1.0, 0.75, 0.5, 0.25]),
        (7.0, 4.0, [0.0, 0.0, 1.0, 1.0]),
        (
            7.0,
            5.0,
            [
                0.378248872499931,
                0.7515301125166395,
                0.269999372498686,
                0.549302812511977,
            ],
        ),
        (
            42.0,
            5.0,
            [
                0.06935776014230743,
                0.5376382001337624,
                0.40049026022115086,
                0.6929954000587877,
            ],
        ),
    ];
    for (seed, kind, values) in pins {
        let config = obj(&[
            ("type", s("Oscillator")),
            ("oscType", n(kind)),
            ("min", n(0.0)),
            ("max", n(1.0)),
            ("speed", n(1.0)),
            ("offset", n(0.0)),
            ("seed", n(seed)),
        ]);
        for (time, expected) in [0.0, 0.25, 0.5, 0.75].into_iter().zip(values) {
            approx(resolve(&config, time, None, None), expected, 1e-9);
        }
    }
}

#[test]
fn noise2d_loops_at_whole_speeds() {
    for speed in [1.0, 2.0, 3.0] {
        for offset in [0.0, 0.25] {
            for seed in [7.0, 42.0] {
                let config = obj(&[
                    ("type", s("Oscillator")),
                    ("oscType", n(6.0)),
                    ("min", n(0.0)),
                    ("max", n(1.0)),
                    ("speed", n(speed)),
                    ("offset", n(offset)),
                    ("seed", n(seed)),
                ]);
                approx(
                    resolve(&config, 1.0, None, None),
                    resolve(&config, 0.0, None, None),
                    1e-9,
                );
            }
        }
    }
}

#[test]
fn rate_modulation_integrates_and_is_seekable() {
    let rate = obj(&[
        ("type", s("Oscillator")),
        ("oscType", n(0.0)),
        ("min", n(0.0)),
        ("max", n(1.0)),
        ("speed", n(1.0)),
        ("offset", n(0.0)),
        ("seed", n(1.0)),
    ]);
    let carrier = obj(&[
        ("type", s("Oscillator")),
        ("oscType", n(2.0)),
        ("min", n(0.0)),
        ("max", n(1.0)),
        ("speed", rate),
        ("offset", n(0.0)),
        ("seed", n(1.0)),
    ]);
    let quarter = resolve(&carrier, 0.25, None, None);
    let expected = ((-20.0 / TAU) % 1.0 + 1.0) % 1.0;
    approx(quarter, expected, 1e-9);
    assert_eq!(
        resolve(&carrier, 0.75, None, None),
        resolve(&carrier, 0.75, None, None)
    );
}

#[test]
fn external_snapshots_drive_oscillator_rate() {
    let midi_rate = midi_config(&[("channel", n(1.0)), ("mode", n(2.0))]);
    let audio_rate = audio_config(&[("band", n(4.0))]);
    let carrier = |rate: JsValue| {
        obj(&[
            ("type", s("Oscillator")),
            ("oscType", n(2.0)),
            ("min", n(0.0)),
            ("max", n(1.0)),
            ("speed", rate),
            ("offset", n(0.0)),
            ("seed", n(1.0)),
        ])
    };
    let mut midi = midi_state();
    let mut audio = AudioState::new();
    midi.channels[0].gate = 1;
    midi.channels[0].velocity = 127;
    audio.set_raw(1.0);
    approx(
        resolve(
            &carrier(midi_rate.clone()),
            0.0125,
            Some(&midi),
            Some(&audio),
        ),
        0.25,
        1e-9,
    );
    approx(
        resolve(
            &carrier(audio_rate.clone()),
            0.0125,
            Some(&midi),
            Some(&audio),
        ),
        0.25,
        1e-9,
    );
    midi.channels[0].velocity = 0;
    audio.set_raw(-1.0);
    approx(
        resolve(&carrier(midi_rate), 0.0125, Some(&midi), Some(&audio)),
        0.75,
        1e-9,
    );
    approx(
        resolve(&carrier(audio_rate), 0.0125, Some(&midi), Some(&audio)),
        0.75,
        1e-9,
    );
}

#[test]
fn invalid_outer_audio_never_evaluates_nested_inputs() {
    let inner = audio_config(&[
        ("band", n(0.0)),
        ("channel", n(1.0)),
        ("name", s("Inner Interface")),
        ("id", s("inner-id")),
    ]);
    let outer = audio_config(&[
        ("band", n(0.0)),
        ("min", inner),
        ("_invalid", JsValue::Bool(true)),
    ]);
    let mut audio = AudioState::new();
    audio.register_device("inner-id", "Inner Interface", Some(1));
    audio.set_channel_values(
        "inner-id",
        1,
        &ChannelValues {
            low: Some(0.8),
            ..Default::default()
        },
    );
    assert_eq!(resolve(&outer, 0.5, None, Some(&audio)), 0.0);
    let requirements = audio_input_requirements([PassAudioInfo {
        uniforms: &obj(&[("amount", outer)]),
        audio_tagged: false,
    }]);
    assert!(requirements.selected.is_empty());
}

#[test]
fn cycles_evaluate_to_zero_instead_of_recursing() {
    let json = serde_json::json!({
        "$id": "first", "type": "Oscillator", "oscType": 0, "min": 0, "max": 1, "offset": 0, "seed": 1,
        "speed": {"type": "Oscillator", "oscType": 1, "min": 0, "max": 1, "offset": 0, "seed": 1, "speed": {"$ref": "first"}}
    });
    let value = JsValue::from_json(&json).unwrap();
    let result = resolve(&value, 0.3, None, None);
    assert!(result.is_finite());
}

#[test]
fn requirements_deduplicate_and_keep_identity() {
    let aggregate = audio_config(&[("band", n(0.0))]);
    let selected = audio_config(&[
        ("band", n(4.0)),
        ("channel", n(7.0)),
        ("name", s("Interface")),
        ("id", s("interface-b")),
    ]);
    let selected_fft = audio_config(&[
        ("band", n(0.0)),
        ("channel", n(7.0)),
        ("name", s("Interface")),
        ("id", s("interface-b")),
    ]);
    let name_only = audio_config(&[
        ("band", n(1.0)),
        ("channel", n(2.0)),
        ("name", s("Unique Interface")),
    ]);
    let pass1 = obj(&[("scale", aggregate), ("rotation", selected_fft)]);
    let pass2 = obj(&[
        ("repeated", selected),
        ("nested", JsValue::array([name_only])),
    ]);
    let requirements = audio_input_requirements([
        PassAudioInfo {
            uniforms: &pass1,
            audio_tagged: false,
        },
        PassAudioInfo {
            uniforms: &pass2,
            audio_tagged: false,
        },
    ]);
    assert!(requirements.needs_legacy);
    assert!(!requirements.needs_legacy_raw);
    assert_eq!(
        requirements.selected,
        vec![
            AudioRequirement {
                id: Some("interface-b".into()),
                name: Some("Interface".into()),
                channel: 7,
                needs_raw: true
            },
            AudioRequirement {
                id: None,
                name: Some("Unique Interface".into()),
                channel: 2,
                needs_raw: false
            },
        ]
    );
    let tagged = audio_input_requirements([PassAudioInfo {
        uniforms: &obj(&[]),
        audio_tagged: true,
    }]);
    assert!(tagged.needs_legacy && !tagged.needs_legacy_raw);
}

#[test]
fn malformed_selectors_never_request_legacy_capture() {
    let ast = obj(&[
        ("type", s("Audio")),
        ("channel", obj(&[("type", s("Number")), ("value", n(0.0))])),
        (
            "name",
            obj(&[("type", s("String")), ("value", s("Interface"))]),
        ),
    ]);
    let invalid = audio_config(&[("band", n(4.0)), ("name", s("Interface")), ("_ast", ast)]);
    let requirements = audio_input_requirements([PassAudioInfo {
        uniforms: &obj(&[("scale", invalid)]),
        audio_tagged: false,
    }]);
    assert_eq!(requirements, AudioInputRequirements::default());
}
