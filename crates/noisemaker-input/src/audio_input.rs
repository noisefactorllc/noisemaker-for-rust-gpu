//! Audio capture: a port of the reference's `AudioInputManager`
//! (`external-input.js`) over a pluggable [`AudioBackend`] instead of
//! `getUserMedia` and Web Audio.
//!
//! As in the reference: the default (browser-selected) input feeds an
//! aggregate analyser (`fftSize` 256) that fills the legacy bands, spectrum,
//! waveform and raw signal of the [`AudioState`], and a channel splitter feeds
//! one analyser per channel that fills the default-input channels and the
//! captured device's channels. The compiled graph's selected-device
//! requirements ([`crate::automation::audio_input_requirements`]) open and
//! close additional device captures, re-checked every 60 updates, and bindings
//! that cannot be captured are reported once per distinct warning.
//!
//! Hosts call [`AudioInputManager::update`] once per frame (the reference's
//! `requestAnimationFrame` loop): it feeds the samples the backend captured
//! since the last update into the analysers, then reads them exactly like
//! `_updateLoop`. With the `cpal` feature, `host::cpal::CpalBackend`
//! captures real devices.

use std::collections::VecDeque;
use std::fmt;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;

use crate::analyser::AudioAnalyzer;
use crate::audio::{AudioState, DeviceSelector};
use crate::automation::{AudioInputRequirements, AudioRequirement};
use noisemaker_dsl::js::math_clamp;

/// `fftSize` of every analyser the reference manager creates.
pub const INPUT_FFT_SIZE: u32 = 256;
/// Default `smoothingTimeConstant` of the reference manager.
pub const DEFAULT_INPUT_SMOOTHING: f64 = 0.8;
/// Highest channel count the manager captures.
pub const MAX_CAPTURE_CHANNELS: u32 = 32;
/// Updates between requirement re-checks.
pub const REQUIREMENTS_CHECK_INTERVAL: u32 = 60;
/// Frames a sink keeps when the host stops calling `update`.
const SINK_CAPACITY_FRAMES: usize = 1 << 20;

/// An input device as the backend enumerates it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioInputDevice {
    /// Device id (`deviceId`).
    pub id: String,
    /// Readable name (`label`).
    pub name: String,
}

/// What an opened capture reports (the track's `getSettings()` and `label`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureInfo {
    /// `settings.deviceId`, when known.
    pub device_id: Option<String>,
    /// `track.label`.
    pub label: String,
    /// `settings.channelCount`, when known.
    pub channel_count: Option<u32>,
}

/// Queued `(channels, interleaved samples)` chunks.
type Chunks = Arc<Mutex<VecDeque<(usize, Vec<f32>)>>>;

/// The samples a backend captured for one stream, waiting for `update`.
#[derive(Clone, Default)]
pub struct CaptureSink {
    chunks: Chunks,
}

impl fmt::Debug for CaptureSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CaptureSink").finish_non_exhaustive()
    }
}

impl CaptureSink {
    /// Queues interleaved float frames of `channels` channels.
    pub fn push_interleaved(&self, samples: &[f32], channels: usize) {
        if channels == 0 || samples.is_empty() {
            return;
        }
        if let Ok(mut chunks) = self.chunks.lock() {
            chunks.push_back((channels, samples.to_vec()));
            let mut frames: usize = chunks.iter().map(|(c, s)| s.len() / c).sum();
            while frames > SINK_CAPACITY_FRAMES {
                let Some((c, s)) = chunks.pop_front() else {
                    break;
                };
                frames -= s.len() / c;
            }
        }
    }

    fn drain(&self) -> Vec<(usize, Vec<f32>)> {
        match self.chunks.lock() {
            Ok(mut chunks) => chunks.drain(..).collect(),
            Err(_) => Vec::new(),
        }
    }
}

/// The analyser calls the manager makes (an `AnalyserNode`).
pub trait CaptureAnalyser {
    /// Feeds interleaved frames (the node's input).
    fn write(&mut self, interleaved: &[f32], channels: usize);
    /// `smoothingTimeConstant = value`; invalid values are rejected.
    fn set_smoothing_time_constant(&mut self, value: f64);
    /// `getByteFrequencyData`.
    fn get_byte_frequency_data(&mut self, array: &mut [u8]);
    /// `getByteTimeDomainData`.
    fn get_byte_time_domain_data(&mut self, array: &mut [u8]);
}

impl CaptureAnalyser for AudioAnalyzer {
    fn write(&mut self, interleaved: &[f32], channels: usize) {
        AudioAnalyzer::write(self, interleaved, channels);
    }

    fn set_smoothing_time_constant(&mut self, value: f64) {
        let _ = AudioAnalyzer::set_smoothing_time_constant(self, value);
    }

    fn get_byte_frequency_data(&mut self, array: &mut [u8]) {
        AudioAnalyzer::get_byte_frequency_data(self, array);
    }

    fn get_byte_time_domain_data(&mut self, array: &mut [u8]) {
        AudioAnalyzer::get_byte_time_domain_data(self, array);
    }
}

/// A source of audio input (`navigator.mediaDevices` and `AudioContext`).
pub trait AudioBackend {
    /// An open capture; dropping it stops the capture.
    type Stream;
    /// Whether capture is supported at all.
    fn is_supported(&self) -> bool {
        true
    }
    /// `getUserMedia({audio})`: the default input, delivering to `sink`.
    fn open_default(&mut self, sink: CaptureSink) -> Result<(Self::Stream, CaptureInfo), String>;
    /// `getUserMedia({audio: {deviceId: {exact: id}}})`.
    fn open_device(
        &mut self,
        id: &str,
        sink: CaptureSink,
    ) -> Result<(Self::Stream, CaptureInfo), String>;
    /// `enumerateDevices()` audio inputs.
    fn enumerate(&mut self) -> Vec<AudioInputDevice>;
    /// `createAnalyser()` with `fftSize` 256 and the given smoothing.
    fn create_analyser(&mut self, smoothing: f64) -> Box<dyn CaptureAnalyser> {
        Box::new(AudioAnalyzer::for_input_manager(smoothing))
    }
}

struct CaptureChannel {
    analyser: Box<dyn CaptureAnalyser>,
    fft_data: Vec<u8>,
    time_data: Vec<u8>,
    default_channel: Option<u32>,
    device_channel: Option<(String, u32)>,
}

struct Capture<S> {
    device_name: String,
    sink: CaptureSink,
    /// Keeps the capture running; dropping it stops the stream.
    _stream: Option<S>,
    channels: Vec<CaptureChannel>,
    registered_device: bool,
    feeds_main: bool,
}

type RequirementsProvider = Box<dyn FnMut() -> Option<AudioInputRequirements> + Send>;
type MessageCallback = Box<dyn FnMut(&str) + Send>;

/// `AudioInputManager`.
pub struct AudioInputManager<B: AudioBackend> {
    backend: B,
    enabled: bool,
    smoothing: f64,
    main: Option<Box<dyn CaptureAnalyser>>,
    fft_data: Vec<u8>,
    time_data: Vec<u8>,
    device_id: Option<String>,
    device_name: String,
    channel_count: u32,
    captures: IndexMap<Option<String>, Capture<B::Stream>>,
    requirements_tick: u32,
    last_unmet_warning: String,
    requirements: Option<RequirementsProvider>,
    on_status: Option<MessageCallback>,
    on_warning: Option<MessageCallback>,
}

impl<B: AudioBackend> fmt::Debug for AudioInputManager<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioInputManager")
            .field("enabled", &self.enabled)
            .field("smoothing", &self.smoothing)
            .field("device_id", &self.device_id)
            .field("device_name", &self.device_name)
            .field("channel_count", &self.channel_count)
            .finish_non_exhaustive()
    }
}

/// `(sum / 4) / 255` of four bins.
fn band(fft: &[u8], bins: [usize; 4]) -> f64 {
    let sum: u32 = bins
        .iter()
        .map(|&i| u32::from(fft.get(i).copied().unwrap_or(0)))
        .sum();
    f64::from(sum) / 4.0 / 255.0
}

/// The band levels and raw sample `_updateLoop` derives from one analyser's
/// byte data: low = bins 0-3, mid = bins 4/6/8/10, high = bins 16/20/24/28,
/// raw = (time-domain mean - 128) / 127.5.
pub fn levels_from_bytes(fft: &[u8], time: &[u8]) -> (f64, f64, f64, f64) {
    let low = band(fft, [0, 1, 2, 3]);
    let mid = band(fft, [4, 6, 8, 10]);
    let high = band(fft, [16, 20, 24, 28]);
    let sum: u64 = time.iter().map(|&b| u64::from(b)).sum();
    let raw = (sum as f64 / time.len() as f64 - 128.0) / 127.5;
    (low, mid, high, raw)
}

impl<B: AudioBackend> AudioInputManager<B> {
    /// A disabled manager over `backend`.
    pub fn new(backend: B) -> Self {
        AudioInputManager {
            backend,
            enabled: false,
            smoothing: DEFAULT_INPUT_SMOOTHING,
            main: None,
            fft_data: Vec::new(),
            time_data: Vec::new(),
            device_id: None,
            device_name: String::new(),
            channel_count: 1,
            captures: IndexMap::new(),
            requirements_tick: 0,
            last_unmet_warning: String::new(),
            requirements: None,
            on_status: None,
            on_warning: None,
        }
    }

    /// The backend.
    pub fn backend(&mut self) -> &mut B {
        &mut self.backend
    }

    /// `enabled`.
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// `smoothing = value`: clamped to 0..1, applied to the aggregate
    /// analyser (per-channel analysers keep the value they were created with).
    pub fn set_smoothing(&mut self, value: f64) {
        self.smoothing = math_clamp(value, 0.0, 1.0);
        if let Some(main) = self.main.as_mut() {
            main.set_smoothing_time_constant(self.smoothing);
        }
    }

    /// The smoothing given to new analysers.
    pub fn smoothing(&self) -> f64 {
        self.smoothing
    }

    /// The id of the default capture's device (`_deviceId`).
    pub fn device_id(&self) -> Option<&str> {
        self.device_id.as_deref()
    }

    /// The name of the default capture's device (`_deviceName`).
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// The default capture's channel count (`_channelCount`).
    pub fn channel_count(&self) -> u32 {
        self.channel_count
    }

    /// Ids of the open captures, the default one first (`null` for a
    /// deviceless default capture).
    pub fn capture_ids(&self) -> Vec<Option<String>> {
        self.captures.keys().cloned().collect()
    }

    /// The last unmet-requirements warning (`_lastUnmetWarning`).
    pub fn last_unmet_warning(&self) -> &str {
        &self.last_unmet_warning
    }

    /// Where requirements come from (`renderer.pipeline.getAudioInputRequirements`).
    pub fn set_requirements_provider(
        &mut self,
        provider: impl FnMut() -> Option<AudioInputRequirements> + Send + 'static,
    ) {
        self.requirements = Some(Box::new(provider));
    }

    /// `onStatusChange(callback)`.
    pub fn on_status_change(&mut self, callback: impl FnMut(&str) + Send + 'static) {
        self.on_status = Some(Box::new(callback));
    }

    /// Receives the warnings the reference writes with `console.warn`.
    pub fn on_warning(&mut self, callback: impl FnMut(&str) + Send + 'static) {
        self.on_warning = Some(Box::new(callback));
    }

    fn notify_status(&mut self, message: &str) {
        if let Some(callback) = self.on_status.as_mut() {
            callback(message);
        }
    }

    fn warn(&mut self, message: &str) {
        if let Some(callback) = self.on_warning.as_mut() {
            callback(message);
        }
    }

    /// `enable()`: opens the default input, registers its default channels
    /// and device, syncs the selected-device captures and runs the first
    /// update.
    pub fn enable(&mut self, audio: &mut AudioState) -> bool {
        if self.enabled {
            return true;
        }
        if !self.backend.is_supported() {
            self.warn("Web Audio API or getUserMedia not supported");
            self.notify_status("Audio input not supported");
            return false;
        }
        let sink = CaptureSink::default();
        let (stream, info) = match self.backend.open_default(sink.clone()) {
            Ok(opened) => opened,
            Err(error) => {
                self.warn(&format!("Audio access denied: {error}"));
                self.notify_status("Audio access denied");
                return false;
            }
        };
        let mut main = self.backend.create_analyser(self.smoothing);
        main.set_smoothing_time_constant(self.smoothing);
        self.main = Some(main);
        self.fft_data = vec![0; INPUT_FFT_SIZE as usize / 2];
        self.time_data = vec![0; INPUT_FFT_SIZE as usize];
        self.channel_count = capture_channel_count(info.channel_count);
        self.device_id = info.device_id.filter(|id| !id.is_empty());
        self.device_name = info.label;
        let (device_id, device_name, channel_count) = (
            self.device_id.clone(),
            self.device_name.clone(),
            self.channel_count,
        );
        self.register_capture(
            audio,
            device_id,
            device_name,
            channel_count,
            sink,
            Some(stream),
            true,
        );
        self.enabled = true;
        self.requirements_tick = 0;
        self.last_unmet_warning.clear();
        self.sync_captures(audio);
        self.update(audio);
        self.notify_status("Audio input enabled");
        true
    }

    /// `disable()`: stops every capture and makes the default channels and
    /// captured devices unavailable.
    pub fn disable(&mut self, audio: &mut AudioState) {
        if !self.enabled {
            return;
        }
        let captured: Vec<String> = self
            .captures
            .iter()
            .filter(|(_, c)| c.registered_device)
            .filter_map(|(id, _)| id.clone())
            .collect();
        self.captures.clear();
        self.main = None;
        self.fft_data.clear();
        self.time_data.clear();
        self.enabled = false;
        audio.reset_aggregate();
        audio.disconnect_default_input();
        for id in captured {
            audio.disconnect_device(&id);
        }
        self.device_id = None;
        self.device_name.clear();
        self.channel_count = 1;
        self.notify_status("Audio input disabled");
    }

    /// `toggle()`.
    pub fn toggle(&mut self, audio: &mut AudioState) -> bool {
        if self.enabled {
            self.disable(audio);
            false
        } else {
            self.enable(audio)
        }
    }

    /// `_registerCapture`: per-channel analysers and their state wiring.
    #[allow(clippy::too_many_arguments)]
    fn register_capture(
        &mut self,
        audio: &mut AudioState,
        device_id: Option<String>,
        device_name: String,
        channel_count: u32,
        sink: CaptureSink,
        stream: Option<B::Stream>,
        default_channels: bool,
    ) {
        if default_channels {
            audio.register_default_channels(channel_count);
        }
        let registered_device = match &device_id {
            Some(id) => audio.register_device(id, &device_name, Some(channel_count)),
            None => false,
        };
        let mut channels = Vec::new();
        for channel in 1..=channel_count {
            let analyser = self.backend.create_analyser(self.smoothing);
            let default_channel = (default_channels
                && audio
                    .get_default_channel_state(f64::from(channel))
                    .is_some())
            .then_some(channel);
            let device_channel = device_id.as_ref().and_then(|id| {
                audio
                    .get_device_channel_state(&DeviceSelector::new(None, Some(id), Some(channel)))
                    .map(|_| (id.clone(), channel))
            });
            channels.push(CaptureChannel {
                analyser,
                fft_data: vec![0; INPUT_FFT_SIZE as usize / 2],
                time_data: vec![0; INPUT_FFT_SIZE as usize],
                default_channel,
                device_channel,
            });
        }
        self.captures.insert(
            device_id,
            Capture {
                device_name,
                sink,
                _stream: stream,
                channels,
                registered_device,
                feeds_main: default_channels,
            },
        );
    }

    /// `_syncCaptures()`: opens, keeps and closes the selected-device
    /// captures the requirements need; returns the unmet requirements.
    pub fn sync_captures(&mut self, audio: &mut AudioState) -> Vec<String> {
        let requirements = self.requirements.as_mut().and_then(|provider| provider());
        let selected: Vec<AudioRequirement> = requirements.map(|r| r.selected).unwrap_or_default();
        let inventory: Vec<AudioInputDevice> = self
            .backend
            .enumerate()
            .into_iter()
            .filter(|d| !d.id.is_empty() && d.id != "default" && d.id != "communications")
            .collect();
        let mut unmet = Vec::new();
        let mut wanted: IndexMap<String, String> = IndexMap::new();
        for requirement in &selected {
            if requirement.id.is_none() && requirement.name.is_none() {
                continue;
            }
            let device_id;
            let device_name;
            if let Some(id) = &requirement.id {
                if self.device_id.as_ref() == Some(id)
                    || self.captures.contains_key(&Some(id.clone()))
                {
                    continue;
                }
                device_id = id.clone();
                device_name = inventory
                    .iter()
                    .find(|d| &d.id == id)
                    .map(|d| d.name.clone())
                    .unwrap_or_default();
            } else if self.device_id.is_some()
                && !self.device_name.is_empty()
                && requirement.name.as_deref() == Some(self.device_name.as_str())
                && self.captures.contains_key(&self.device_id)
            {
                // The named device is the browser-selected capture itself.
                continue;
            } else {
                let name = requirement.name.clone().unwrap_or_default();
                let matches: Vec<&AudioInputDevice> =
                    inventory.iter().filter(|d| d.name == name).collect();
                if matches.len() == 1 {
                    device_id = matches[0].id.clone();
                    device_name = name;
                    if self.device_id.as_ref() == Some(&device_id)
                        || self.captures.contains_key(&Some(device_id.clone()))
                    {
                        continue;
                    }
                } else {
                    let captured = self
                        .captures
                        .values()
                        .filter(|c| c.device_name == name)
                        .count();
                    let channel = requirement.channel;
                    if matches.len() > 1 {
                        unmet.push(format!(
                            "{name} channel {channel} (name matches multiple devices)"
                        ));
                    } else if captured == 1 {
                        unmet.push(format!("{name} channel {channel} (no deviceId available)"));
                    } else {
                        unmet.push(format!(
                            "{name} channel {channel} (not found among input devices)"
                        ));
                    }
                    continue;
                }
            }
            wanted.insert(device_id, device_name);
        }

        let stale: Vec<Option<String>> = self
            .captures
            .keys()
            .filter(|id| {
                **id != self.device_id && !id.as_ref().is_some_and(|id| wanted.contains_key(id))
            })
            .cloned()
            .collect();
        for id in stale {
            if let Some(capture) = self.captures.shift_remove(&id)
                && capture.registered_device
                && let Some(id) = &id
            {
                audio.disconnect_device(id);
            }
        }

        for (device_id, device_name) in wanted {
            let sink = CaptureSink::default();
            match self.backend.open_device(&device_id, sink.clone()) {
                Ok((stream, info)) => {
                    let channel_count = capture_channel_count(info.channel_count);
                    let name = if info.label.is_empty() {
                        device_name
                    } else {
                        info.label
                    };
                    self.register_capture(
                        audio,
                        Some(device_id),
                        name,
                        channel_count,
                        sink,
                        Some(stream),
                        false,
                    );
                }
                Err(error) => {
                    let label = if device_name.is_empty() {
                        device_id.clone()
                    } else {
                        device_name.clone()
                    };
                    self.warn(&format!(
                        "[Noisemaker] failed to open audio input device {label}: {error}"
                    ));
                    unmet.push(format!("{label} (failed to open)"));
                }
            }
        }

        // Post-open channel validation.
        for requirement in &selected {
            let capture = if requirement.id.is_none() && requirement.name.is_none() {
                self.captures.get(&self.device_id)
            } else if let Some(id) = &requirement.id {
                self.captures.get(&Some(id.clone()))
            } else {
                let name = requirement.name.clone().unwrap_or_default();
                let matches: Vec<&AudioInputDevice> =
                    inventory.iter().filter(|d| d.name == name).collect();
                if matches.len() == 1 {
                    self.captures.get(&Some(matches[0].id.clone()))
                } else {
                    self.captures.values().find(|c| c.device_name == name)
                }
            };
            if let Some(capture) = capture {
                let available = capture.channels.len() as u32;
                if requirement.channel > available {
                    let label = requirement
                        .name
                        .clone()
                        .filter(|n| !n.is_empty())
                        .or_else(|| requirement.id.clone().filter(|i| !i.is_empty()))
                        .unwrap_or_else(|| "default input".to_string());
                    unmet.push(format!(
                        "{label} channel {} (captured device only exposes {available} channel(s))",
                        requirement.channel
                    ));
                }
            }
        }

        if unmet.is_empty() {
            self.last_unmet_warning.clear();
        } else {
            let message = format!(
                "[Noisemaker] {} selected-device audio binding(s) could not be captured ({}); they evaluate to min.",
                unmet.len(),
                unmet.join(", ")
            );
            if message != self.last_unmet_warning {
                self.last_unmet_warning = message.clone();
                self.warn(&message);
            }
        }
        unmet
    }

    /// One frame of `_updateLoop`: feeds the captured samples, fills the
    /// aggregate and every captured channel, and re-checks the requirements
    /// every 60 updates.
    pub fn update(&mut self, audio: &mut AudioState) {
        if !self.enabled {
            return;
        }
        // Feed what the backends captured since the last update.
        for capture in self.captures.values_mut() {
            for (channels, samples) in capture.sink.drain() {
                if capture.feeds_main
                    && let Some(main) = self.main.as_mut()
                {
                    main.write(&samples, channels);
                }
                let frames = samples.len() / channels;
                let mut mono = vec![0.0f32; frames];
                for (index, channel) in capture.channels.iter_mut().enumerate() {
                    if index < channels {
                        for (f, value) in mono.iter_mut().enumerate() {
                            *value = samples[f * channels + index];
                        }
                    } else {
                        mono.fill(0.0);
                    }
                    channel.analyser.write(&mono, 1);
                }
            }
        }

        let Some(main) = self.main.as_mut() else {
            return;
        };
        main.get_byte_frequency_data(&mut self.fft_data);
        audio.set_spectrum(&self.fft_data);
        main.get_byte_time_domain_data(&mut self.time_data);
        audio.set_waveform(&self.time_data);
        let (low, mid, high, raw) = levels_from_bytes(&self.fft_data, &self.time_data);
        audio.low = low;
        audio.mid = mid;
        audio.high = high;
        audio.vol = (low + mid + high) / 3.0;
        audio.set_raw(raw);

        for capture in self.captures.values_mut() {
            for channel in &mut capture.channels {
                channel
                    .analyser
                    .get_byte_frequency_data(&mut channel.fft_data);
                channel
                    .analyser
                    .get_byte_time_domain_data(&mut channel.time_data);
                let (low, mid, high, raw) =
                    levels_from_bytes(&channel.fft_data, &channel.time_data);
                if let Some(n) = channel.default_channel
                    && let Some(state) = audio.default_channels_mut().and_then(|c| c.get_mut(&n))
                {
                    state.set_bands(low, mid, high);
                    state.set_raw(raw);
                }
                if let Some((id, n)) = &channel.device_channel
                    && let Some(state) = audio
                        .device_entry_mut(id)
                        .and_then(|e| e.channels.get_mut(n))
                {
                    state.set_bands(low, mid, high);
                    state.set_raw(raw);
                }
            }
        }

        self.requirements_tick += 1;
        if self.requirements_tick >= REQUIREMENTS_CHECK_INTERVAL {
            self.requirements_tick = 0;
            self.sync_captures(audio);
        }
    }
}

/// `Number.isInteger(channelCount) && channelCount >= 1 ? Math.min(32, n) : 1`.
fn capture_channel_count(count: Option<u32>) -> u32 {
    match count {
        Some(n) if n >= 1 => n.min(MAX_CAPTURE_CHANNELS),
        _ => 1,
    }
}

#[cfg(test)]
mod tests;
