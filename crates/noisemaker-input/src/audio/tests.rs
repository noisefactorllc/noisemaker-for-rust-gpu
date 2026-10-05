//! Unit tests ported from the AudioState assertions of the reference's
//! `test_external_input.js` and `test_audio.js`.

use super::*;

/// A fake analyser that fills every bin with one value, like the reference's
/// test analysers.
struct Constant {
    bins: usize,
    value: u8,
    calls: usize,
}

impl FrequencyDataSource for Constant {
    fn frequency_bin_count(&self) -> usize {
        self.bins
    }

    fn get_byte_frequency_data(&mut self, array: &mut [u8]) {
        self.calls += 1;
        array.fill(self.value);
    }
}

fn approx(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected}"
    );
}

#[test]
fn reuses_the_frequency_buffer() {
    let mut audio = AudioState::without_device_registry();
    let mut analyser = Constant {
        bins: 128,
        value: 0,
        calls: 0,
    };
    audio.update_from_analyser(Some(&mut analyser), 5.0);
    let first = audio.frequency_data().unwrap().as_ptr();
    audio.update_from_analyser(Some(&mut analyser), 5.0);
    assert_eq!(analyser.calls, 2);
    assert_eq!(audio.frequency_data().unwrap().as_ptr(), first);
}

#[test]
fn initializes_at_rest() {
    let audio = AudioState::new();
    assert_eq!(
        (audio.low, audio.mid, audio.high, audio.vol),
        (0.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(audio.fft.len(), 16);
    assert!(audio.waveform.iter().all(|&w| w == 0.5));
}

#[test]
fn set_bands_clamps_and_averages() {
    let mut audio = AudioState::new();
    audio.set_bands(0.5, 0.3, 0.8);
    assert_eq!((audio.low, audio.mid, audio.high), (0.5, 0.3, 0.8));
    approx(audio.vol, (0.5 + 0.3 + 0.8) / 3.0, 1e-15);
    audio.set_bands(-0.5, 1.5, 0.5);
    assert_eq!((audio.low, audio.mid, audio.high), (0.0, 1.0, 0.5));
    audio.set_bands(f64::NAN, 0.0, 0.0);
    assert!(
        audio.low.is_nan() && audio.vol.is_nan(),
        "Math.min/max propagate NaN"
    );
}

#[test]
fn reset_clears_values_and_waveform() {
    let mut audio = AudioState::new();
    audio.set_bands(0.5, 0.5, 0.5);
    audio.set_waveform(&[255; 128]);
    audio.reset();
    assert_eq!(
        (audio.low, audio.mid, audio.high, audio.vol, audio.raw),
        (0.0, 0.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(audio.waveform[0], 0.5);
}

#[test]
fn reset_aggregate_keeps_selected_channels() {
    let mut audio = AudioState::new();
    audio.register_device("selected", "Selected", Some(1));
    audio.register_default_channels(1);
    let selected = DeviceSelector::new(None, Some("selected"), Some(1));
    let mut analyser = Constant {
        bins: 128,
        value: 128,
        calls: 0,
    };
    audio.update_from_analyser(Some(&mut analyser), 5.0);
    audio
        .get_device_channel_state_mut(&selected)
        .unwrap()
        .update_from_analyser(Some(&mut analyser), 5.0);
    audio.set_raw(0.8);
    audio
        .get_device_channel_state_mut(&selected)
        .unwrap()
        .set_raw(-0.25);
    audio
        .get_default_channel_state_mut(1.0)
        .unwrap()
        .set_raw(0.5);
    audio.set_spectrum(&[128; 128]);
    audio.set_waveform(&[255; 128]);
    audio.reset_aggregate();
    assert_eq!(
        (audio.low, audio.mid, audio.high, audio.vol, audio.raw),
        (0.0, 0.0, 0.0, 0.0, 0.0)
    );
    assert!(!audio.raw_ready);
    assert!(audio.fft.iter().all(|&v| v == 0.0));
    assert!(audio.spectrum.iter().all(|&v| v == 0.0));
    assert!(audio.waveform.iter().all(|&v| v == 0.5));
    assert_eq!(
        audio.get_device_channel_state(&selected).unwrap().raw,
        -0.25
    );
    assert_eq!(audio.get_default_channel_state(1.0).unwrap().raw, 0.5);
    analyser.value = 32;
    audio.update_from_analyser(Some(&mut analyser), 5.0);
    audio
        .get_device_channel_state_mut(&selected)
        .unwrap()
        .update_from_analyser(Some(&mut analyser), 5.0);
    approx(audio.low, 32.0 / 255.0, 1e-12);
    approx(
        audio.get_device_channel_state(&selected).unwrap().low,
        80.0 / 255.0,
        1e-12,
    );
    audio.reset();
    assert!(!audio.get_device_channel_state(&selected).unwrap().raw_ready);
    assert!(!audio.get_default_channel_state(1.0).unwrap().raw_ready);
}

#[test]
fn raw_signal_is_bipolar_and_clamped() {
    let mut audio = AudioState::new();
    audio.set_raw(-0.75);
    assert_eq!(audio.raw, -0.75);
    audio.set_raw(2.0);
    assert_eq!(audio.raw, 1.0);
    audio.set_raw(-2.0);
    assert_eq!(audio.raw, -1.0);
    audio.set_raw(f64::NAN);
    assert_eq!(audio.raw, 0.0);
    assert!(audio.raw_ready);
}

#[test]
fn device_raw_invalidation_clears_readiness() {
    let mut audio = AudioState::new();
    audio.register_device("interface-a", "Interface", Some(2));
    audio.set_channel_values(
        "interface-a",
        1,
        &ChannelValues {
            raw: Some(0.0),
            ..Default::default()
        },
    );
    audio.set_channel_values(
        "interface-a",
        2,
        &ChannelValues {
            raw: Some(-0.5),
            ..Default::default()
        },
    );
    let one = DeviceSelector::new(None, Some("interface-a"), Some(1));
    let two = DeviceSelector::new(None, Some("interface-a"), Some(2));
    assert!(audio.get_device_channel_state(&one).unwrap().raw_ready);
    audio.set_device_raw_unavailable("interface-a");
    assert!(!audio.get_device_channel_state(&one).unwrap().raw_ready);
    assert!(!audio.get_device_channel_state(&two).unwrap().raw_ready);
}

#[test]
fn devices_are_isolated_by_id_and_channel() {
    let mut audio = AudioState::new();
    audio.register_device("left", "Interface", Some(2));
    audio.register_device("right", "Interface", Some(4));
    audio.set_channel_values(
        "left",
        2,
        &ChannelValues {
            low: Some(0.2),
            mid: Some(0.3),
            high: Some(0.4),
            vol: Some(0.5),
            raw: Some(-0.6),
        },
    );
    audio.set_channel_values(
        "right",
        2,
        &ChannelValues {
            low: Some(0.8),
            mid: Some(0.7),
            high: Some(0.6),
            vol: Some(0.5),
            raw: Some(0.4),
        },
    );
    let selected = audio
        .get_device_channel_state(&DeviceSelector::new(
            Some("Interface"),
            Some("right"),
            Some(2),
        ))
        .unwrap();
    assert_eq!((selected.low, selected.raw), (0.8, 0.4));
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(Some("Interface"), None, Some(2)))
            .is_none()
    );
}

#[test]
fn unique_names_resolve_and_disconnect_is_inert() {
    let mut audio = AudioState::new();
    audio.register_device("solo", "Unique Interface", Some(2));
    audio.set_channel_values(
        "solo",
        1,
        &ChannelValues {
            low: Some(0.9),
            raw: Some(0.25),
            ..Default::default()
        },
    );
    let by_name = DeviceSelector::new(Some("Unique Interface"), None, Some(1));
    assert_eq!(audio.get_device_channel_state(&by_name).unwrap().low, 0.9);
    audio.disconnect_device("solo");
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(
                Some("Unique Interface"),
                Some("solo"),
                Some(1)
            ))
            .is_none()
    );
    assert!(!audio.get_devices()[0].connected);
    assert!(
        audio
            .get_device_channel_state(&DeviceSelector::new(
                Some("Stereo"),
                Some("stereo"),
                Some(3)
            ))
            .is_none()
    );
}

#[test]
fn smoothing_is_a_rolling_mean() {
    let mut audio = AudioState::new();
    audio.set_max_buffer_length(3.0);
    audio.smoothing_buffers_mut().low = vec![0.9];
    approx(audio.smooth(Band::Low, 0.3), 0.6, 1e-15);
    audio.smooth(Band::Low, 0.3);
    audio.smooth(Band::Low, 0.3);
    assert_eq!(audio.smoothing_buffers().low.len(), 3);
}

#[test]
fn waveform_and_spectrum_scale_bytes() {
    let mut audio = AudioState::new();
    let mut raw = [0u8; 128];
    raw[..64].fill(255);
    audio.set_waveform(&raw);
    assert_eq!(
        (audio.waveform[0], audio.waveform[63], audio.waveform[64]),
        (1.0, 1.0, 0.0)
    );
    audio.set_spectrum(&[51; 4]);
    assert_eq!(audio.spectrum[3], 0.2);
    assert_eq!(audio.spectrum[4], 0.0);
}

#[test]
fn default_channels_register_and_disconnect() {
    let mut audio = AudioState::new();
    assert!(!audio.register_default_channels(0));
    assert!(!audio.register_default_channels(33));
    assert!(audio.register_default_channels(2));
    audio.get_default_channel_state_mut(2.0).unwrap().low = 0.75;
    audio.disconnect_default_input();
    assert!(audio.get_default_channel_state(2.0).is_none());
    audio.register_default_channels(1);
    assert!(audio.get_default_channel_state(2.0).is_none());
    assert_eq!(audio.get_default_channel_state(1.0).unwrap().low, 0.0);
    let channel_only = DeviceSelector::new(None, None, Some(1));
    assert!(audio.get_device_channel_state(&channel_only).is_some());
}

#[test]
fn inventory_blocks_ambiguous_names() {
    let mut audio = AudioState::new();
    audio.register_device("left", "Interface", Some(2));
    let name_only = DeviceSelector::new(Some("Interface"), None, Some(1));
    let device = |id: &str, connected| AudioDeviceInfo {
        id: id.into(),
        name: "Interface".into(),
        connected,
        channel_count: 1,
    };
    audio.set_device_inventory(&[device("left", true), device("right", true)]);
    assert!(audio.get_device_channel_state(&name_only).is_none());
    audio.set_device_inventory(&[device("left", true), device("right", false)]);
    assert!(audio.get_device_channel_state(&name_only).is_some());
    audio.set_device_inventory(&[device("right", true)]);
    assert!(audio.get_device_channel_state(&name_only).is_none());
}

#[test]
fn avg_bins_handles_short_buffers() {
    assert_eq!(avg_bins(&[], 1, 2), 0.0);
    assert_eq!(avg_bins(&[255, 255], 1, 2), 1.0);
    approx(avg_bins(&[0, 51, 102], 1, 47), 76.5 / 255.0, 1e-15);
}
