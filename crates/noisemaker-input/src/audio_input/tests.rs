//! AudioInputManager tests ported from the reference's
//! `test_external_input.js`: the fake backend reproduces its fixtures
//! (FIXTURE_DEVICES, two-channel tracks, and analysers whose frequency and
//! time-domain data depend on their creation order like MockAudioContext's).

use std::sync::{Arc, Mutex};

use super::*;

const FREQ_LEVELS: [u8; 4] = [64, 200, 10, 128];
const TIME_LEVELS: [u8; 4] = [255, 128, 160, 96];

struct FixtureAnalyser {
    freq: u8,
    time: u8,
}

impl CaptureAnalyser for FixtureAnalyser {
    fn write(&mut self, _interleaved: &[f32], _channels: usize) {}
    fn set_smoothing_time_constant(&mut self, _value: f64) {}
    fn get_byte_frequency_data(&mut self, array: &mut [u8]) {
        array.fill(self.freq);
    }
    fn get_byte_time_domain_data(&mut self, array: &mut [u8]) {
        array.fill(self.time);
    }
}

#[derive(Default)]
struct FixtureBackend {
    analysers: usize,
    opened: Arc<Mutex<Vec<String>>>,
    deviceless: bool,
    no_inventory: bool,
}

const FIXTURE_DEVICES: [(&str, &str); 4] = [
    ("fixture-device", "Fixture Microphone"),
    ("other-device", "Other Interface"),
    ("stack-a", "Stacked Input"),
    ("stack-b", "Stacked Input"),
];

impl AudioBackend for FixtureBackend {
    type Stream = ();

    fn open_default(&mut self, _sink: CaptureSink) -> Result<((), CaptureInfo), String> {
        self.opened.lock().unwrap().push("default".into());
        if self.deviceless {
            return Ok((
                (),
                CaptureInfo {
                    device_id: None,
                    label: "Fixture Microphone".into(),
                    channel_count: None,
                },
            ));
        }
        Ok((
            (),
            CaptureInfo {
                device_id: Some("fixture-device".into()),
                label: "Fixture Microphone".into(),
                channel_count: Some(2),
            },
        ))
    }

    fn open_device(&mut self, id: &str, _sink: CaptureSink) -> Result<((), CaptureInfo), String> {
        let Some((id, label)) = FIXTURE_DEVICES.iter().find(|(d, _)| *d == id) else {
            return Err("device not found".into());
        };
        self.opened.lock().unwrap().push(id.to_string());
        Ok((
            (),
            CaptureInfo {
                device_id: Some(id.to_string()),
                label: label.to_string(),
                channel_count: Some(2),
            },
        ))
    }

    fn enumerate(&mut self) -> Vec<AudioInputDevice> {
        if self.no_inventory {
            return Vec::new();
        }
        FIXTURE_DEVICES
            .iter()
            .map(|(id, name)| AudioInputDevice {
                id: id.to_string(),
                name: name.to_string(),
            })
            .collect()
    }

    fn create_analyser(&mut self, _smoothing: f64) -> Box<dyn CaptureAnalyser> {
        let index = self.analysers;
        self.analysers += 1;
        Box::new(FixtureAnalyser {
            freq: FREQ_LEVELS[index % 4],
            time: TIME_LEVELS[index % 4],
        })
    }
}

fn approx(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected}"
    );
}

fn requirement(
    id: Option<&str>,
    name: Option<&str>,
    channel: u32,
    needs_raw: bool,
) -> AudioRequirement {
    AudioRequirement {
        id: id.map(str::to_string),
        name: name.map(str::to_string),
        channel,
        needs_raw,
    }
}

fn with_requirements(
    selected: Vec<AudioRequirement>,
) -> impl FnMut() -> Option<AudioInputRequirements> + Send {
    move || {
        Some(AudioInputRequirements {
            needs_legacy: true,
            needs_legacy_raw: false,
            selected: selected.clone(),
        })
    }
}

fn warnings(manager: &mut AudioInputManager<FixtureBackend>) -> Arc<Mutex<Vec<String>>> {
    let warnings = Arc::new(Mutex::new(Vec::new()));
    let sink = warnings.clone();
    manager.on_warning(move |message| sink.lock().unwrap().push(message.to_string()));
    warnings
}

#[test]
fn populates_the_captured_device_default_channels_and_raw_readiness() {
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(FixtureBackend::default());
    assert!(manager.enable(&mut audio));
    assert!(manager.enabled());
    manager.update(&mut audio);

    let one = audio.get_default_channel_state(1.0).unwrap();
    let two = audio.get_default_channel_state(2.0).unwrap();
    approx(one.low, 200.0 / 255.0, 1e-12);
    approx(two.low, 10.0 / 255.0, 1e-12);
    assert!(one.raw_ready);
    assert_eq!(one.raw, 0.0);
    approx(two.raw, (160.0 - 128.0) / 127.5, 1e-12);

    let device = audio
        .get_device_channel_state(&DeviceSelector::new(None, Some("fixture-device"), Some(2)))
        .unwrap();
    approx(device.high, 10.0 / 255.0, 1e-12);
    assert!(device.raw_ready);
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(
                Some("Fixture Microphone"),
                None,
                Some(1)
            ))
            .is_some()
    );
    assert!(audio.raw_ready);
    approx(audio.raw, (255.0 - 128.0) / 127.5, 1e-12);
    approx(audio.low, 64.0 / 255.0, 1e-12);
}

#[test]
fn opens_and_populates_every_selected_device() {
    let mut audio = AudioState::new();
    let backend = FixtureBackend::default();
    let opened = backend.opened.clone();
    let mut manager = AudioInputManager::new(backend);
    let warnings = warnings(&mut manager);
    manager.set_requirements_provider(with_requirements(vec![
        requirement(None, None, 1, false),
        requirement(Some("other-device"), Some("Other Interface"), 2, true),
        requirement(None, Some("Fixture Microphone"), 1, false),
    ]));
    assert!(manager.enable(&mut audio));
    assert_eq!(opened.lock().unwrap().len(), 2, "default + selected");
    manager.update(&mut audio);
    let selected = audio
        .get_device_channel_state(&DeviceSelector::new(None, Some("other-device"), Some(2)))
        .unwrap();
    approx(selected.low, 64.0 / 255.0, 1e-12);
    assert!(selected.raw_ready);
    approx(selected.raw, (255.0 - 128.0) / 127.5, 1e-12);
    assert!(warnings.lock().unwrap().is_empty());
}

#[test]
fn resyncs_captures_when_the_graph_changes() {
    let mut audio = AudioState::new();
    let backend = FixtureBackend::default();
    let opened = backend.opened.clone();
    let mut manager = AudioInputManager::new(backend);
    let selected = Arc::new(Mutex::new(Vec::new()));
    let provider = selected.clone();
    manager.set_requirements_provider(move || {
        Some(AudioInputRequirements {
            needs_legacy: true,
            needs_legacy_raw: false,
            selected: provider.lock().unwrap().clone(),
        })
    });
    assert!(manager.enable(&mut audio));
    assert_eq!(opened.lock().unwrap().len(), 1);
    *selected.lock().unwrap() = vec![requirement(
        Some("other-device"),
        Some("Other Interface"),
        1,
        false,
    )];
    for _ in 0..60 {
        manager.update(&mut audio);
    }
    assert_eq!(opened.lock().unwrap().len(), 2);
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(None, Some("other-device"), Some(1)))
            .is_some()
    );
    selected.lock().unwrap().clear();
    for _ in 0..60 {
        manager.update(&mut audio);
    }
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(None, Some("other-device"), Some(1)))
            .is_none()
    );
    assert_eq!(
        manager.capture_ids(),
        vec![Some("fixture-device".to_string())]
    );
}

#[test]
fn warns_about_uncapturable_requirements() {
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(FixtureBackend::default());
    let warnings = warnings(&mut manager);
    manager.set_requirements_provider(with_requirements(vec![
        requirement(Some("missing-device"), Some("Ghost Interface"), 1, true),
        requirement(None, Some("Stacked Input"), 1, true),
    ]));
    assert!(manager.enable(&mut audio));
    let warnings = warnings.lock().unwrap();
    assert!(
        warnings
            .iter()
            .any(|m| m.contains("missing-device (failed to open)"))
    );
    assert!(
        warnings
            .iter()
            .any(|m| m.contains("Stacked Input") && m.contains("multiple devices"))
    );
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(None, Some("missing-device"), Some(1)))
            .is_none()
    );
}

#[test]
fn warns_when_a_captured_device_lacks_the_requested_channel() {
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(FixtureBackend::default());
    let warnings = warnings(&mut manager);
    manager.set_requirements_provider(with_requirements(vec![
        requirement(None, None, 3, false),
        requirement(Some("other-device"), Some("Other Interface"), 3, true),
        requirement(Some("other-device"), Some("Other Interface"), 2, true),
        requirement(None, Some("Fixture Microphone"), 3, false),
        requirement(None, Some("Fixture Microphone"), 2, false),
    ]));
    assert!(manager.enable(&mut audio));
    let text = warnings.lock().unwrap().join("\n");
    assert!(text.contains("default input channel 3") && text.contains("only exposes 2"));
    assert!(text.contains("Other Interface channel 3"));
    assert!(text.contains("Fixture Microphone channel 3"));
    assert_eq!(text.matches("only exposes 2").count(), 3);
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(None, Some("other-device"), Some(2)))
            .is_some()
    );
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(None, Some("other-device"), Some(3)))
            .is_none()
    );
}

#[test]
fn name_only_binding_without_a_device_id_warns() {
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(FixtureBackend {
        deviceless: true,
        no_inventory: true,
        ..Default::default()
    });
    let warnings = warnings(&mut manager);
    manager.set_requirements_provider(with_requirements(vec![requirement(
        None,
        Some("Fixture Microphone"),
        1,
        true,
    )]));
    assert!(manager.enable(&mut audio));
    assert!(
        warnings
            .lock()
            .unwrap()
            .iter()
            .any(|m| m.contains("Fixture Microphone"))
    );
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(
                Some("Fixture Microphone"),
                None,
                Some(1)
            ))
            .is_none()
    );
}

#[test]
fn deviceless_capture_warns_for_missing_default_channels() {
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(FixtureBackend {
        deviceless: true,
        ..Default::default()
    });
    let warnings = warnings(&mut manager);
    manager.set_requirements_provider(with_requirements(vec![
        requirement(None, None, 2, false),
        requirement(None, None, 1, false),
    ]));
    assert!(manager.enable(&mut audio));
    let text = warnings.lock().unwrap().join("\n");
    assert!(text.contains("default input channel 2 (captured device only exposes 1 channel(s))"));
    assert!(!text.contains("channel 1 (captured device"));
    assert!(audio.get_default_channel_state(1.0).is_some());
    assert!(audio.get_default_channel_state(2.0).is_none());
}

#[test]
fn disable_clears_default_channels_devices_and_raw_readiness() {
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(FixtureBackend::default());
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let sink = statuses.clone();
    manager.on_status_change(move |m| sink.lock().unwrap().push(m.to_string()));
    assert!(manager.enable(&mut audio));
    manager.update(&mut audio);
    assert!(audio.get_default_channel_state(1.0).unwrap().raw_ready);
    manager.disable(&mut audio);
    assert!(audio.get_default_channel_state(1.0).is_none());
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(None, Some("fixture-device"), Some(1)))
            .is_none()
    );
    assert!(!audio.raw_ready);
    assert!(!manager.enabled());
    assert_eq!(
        *statuses.lock().unwrap(),
        vec![
            "Audio input enabled".to_string(),
            "Audio input disabled".to_string()
        ]
    );
}

#[test]
fn real_analysers_route_channels_through_the_splitter() {
    // The default backend analysers are real AnalyserNodes: silence on
    // channel 1 and a DC offset on channel 2 reach their own states.
    struct Sampled;
    impl AudioBackend for Sampled {
        type Stream = ();
        fn open_default(&mut self, sink: CaptureSink) -> Result<((), CaptureInfo), String> {
            let frames: Vec<f32> = (0..512).flat_map(|_| [0.0f32, 0.25]).collect();
            sink.push_interleaved(&frames, 2);
            Ok((
                (),
                CaptureInfo {
                    device_id: Some("dev".into()),
                    label: "Dev".into(),
                    channel_count: Some(2),
                },
            ))
        }
        fn open_device(
            &mut self,
            _id: &str,
            _sink: CaptureSink,
        ) -> Result<((), CaptureInfo), String> {
            Err("none".into())
        }
        fn enumerate(&mut self) -> Vec<AudioInputDevice> {
            Vec::new()
        }
    }
    let mut audio = AudioState::new();
    let mut manager = AudioInputManager::new(Sampled);
    assert!(manager.enable(&mut audio));
    let one = audio.get_default_channel_state(1.0).unwrap();
    let two = audio.get_default_channel_state(2.0).unwrap();
    assert_eq!(one.raw, 0.0);
    assert!(one.raw_ready);
    // 128 * (0.25 + 1) = 160 for every sample.
    assert_eq!(two.raw, (160.0 - 128.0) / 127.5);
    // The aggregate analyser down-mixes the stereo pair: 0.125 -> byte 144.
    assert_eq!(audio.raw, (144.0 - 128.0) / 127.5);
}
