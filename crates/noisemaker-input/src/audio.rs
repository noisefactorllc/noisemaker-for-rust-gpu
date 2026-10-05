//! Audio input state: a port of `AudioState` (and `_avgBins`) from the
//! reference's `shaders/src/runtime/external-input.js`.
//!
//! The Web Audio plumbing becomes plain methods the host calls: band levels and
//! the raw signal ([`AudioState::set_bands`], [`AudioState::set_raw`]), spectrum
//! and waveform bytes, analysis from any [`FrequencyDataSource`] (such as
//! [`crate::analyser::AudioAnalyzer`]), and the registry of capture devices and
//! their one-based channels, including the default input's channels.
//!
//! JavaScript `Map`s are [`IndexMap`]s with the same insertion-order behaviour,
//! and `Math.min`/`Math.max` keep their NaN and signed-zero semantics.

use std::fmt;

use indexmap::IndexMap;

use crate::midi::SelectorKey;
use noisemaker_dsl::js::{math_clamp, math_max, math_min};

/// Number of coarse FFT bins (`fft`).
pub const FFT_BINS: usize = 16;
/// Number of spectrum values (`spectrum`).
pub const SPECTRUM_LEN: usize = 128;
/// Number of waveform samples (`waveform`).
pub const WAVEFORM_LEN: usize = 128;
/// Highest one-based channel number a selector may address.
pub const MAX_SELECTED_CHANNEL: u32 = 32;

/// Something that fills unsigned-byte frequency data like
/// `AnalyserNode.getByteFrequencyData`.
pub trait FrequencyDataSource {
    /// `frequencyBinCount`.
    fn frequency_bin_count(&self) -> usize;
    /// `getByteFrequencyData(array)`: writes `min(bins, array.len())` values.
    fn get_byte_frequency_data(&mut self, array: &mut [u8]);
}

/// `_avgBins(buf, from, to)`: the mean of `buf[from..min(to, len)]`, scaled to
/// 0-1; 0 for an empty range.
pub fn avg_bins(buf: &[u8], from: usize, to: usize) -> f64 {
    let end = to.min(buf.len());
    if end <= from {
        return 0.0;
    }
    let mut sum = 0.0;
    for &value in &buf[from..end] {
        sum += f64::from(value);
    }
    sum / (end - from) as f64 / 255.0
}

/// One of the smoothed bands of [`AudioState::update_from_analyser`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    /// The low band.
    Low,
    /// The mid band.
    Mid,
    /// The high band.
    High,
}

/// The rolling smoothing buffers (`_smoothingBuffers`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmoothingBuffers {
    /// Recent raw low band values.
    pub low: Vec<f64>,
    /// Recent raw mid band values.
    pub mid: Vec<f64>,
    /// Recent raw high band values.
    pub high: Vec<f64>,
}

/// Analysed values for one device channel (`setChannelValues` input). Absent
/// and non-finite values are ignored.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChannelValues {
    /// Low band level.
    pub low: Option<f64>,
    /// Mid band level.
    pub mid: Option<f64>,
    /// High band level.
    pub high: Option<f64>,
    /// Volume.
    pub vol: Option<f64>,
    /// Bipolar raw sample.
    pub raw: Option<f64>,
}

/// Device identity and connection state (`getDevices()` entries and
/// `setDeviceInventory` input).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioDeviceInfo {
    /// Device id.
    pub id: String,
    /// Readable device name.
    pub name: String,
    /// Whether the device is connected.
    pub connected: bool,
    /// Number of captured channels (1 in inventory entries, which carry none).
    pub channel_count: u32,
}

/// A registered capture device (`_devices` entries).
#[derive(Clone, Debug)]
pub struct AudioDeviceEntry {
    /// Device id.
    pub id: String,
    /// Readable name, updated on re-registration.
    pub name: String,
    /// Whether the device is available.
    pub connected: bool,
    /// Number of captured channels.
    pub channel_count: u32,
    /// Independently analysed channels by one-based number.
    pub channels: IndexMap<u32, AudioState>,
}

/// The selector of `getDeviceChannelState`: a compiled `audio()` descriptor's
/// `name`, `id` and one-based `channel`. `channel` is `None` when undefined;
/// a non-number channel is `Some(NaN)`, which no lookup accepts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceSelector<'a> {
    /// Readable device name.
    pub name: SelectorKey<'a>,
    /// Exact device id.
    pub id: SelectorKey<'a>,
    /// One-based channel number.
    pub channel: Option<f64>,
}

impl<'a> DeviceSelector<'a> {
    /// A selector from optional strings and channel.
    pub fn new(name: Option<&'a str>, id: Option<&'a str>, channel: Option<u32>) -> Self {
        DeviceSelector {
            name: name.into(),
            id: id.into(),
            channel: channel.map(f64::from),
        }
    }
}

/// Audio analysis state (`AudioState`).
#[derive(Clone)]
pub struct AudioState {
    /// Low frequency band level (0-1).
    pub low: f64,
    /// Mid frequency band level (0-1).
    pub mid: f64,
    /// High frequency band level (0-1).
    pub high: f64,
    /// Overall volume level (0-1).
    pub vol: f64,
    /// Bipolar time-domain signal (-1 to 1).
    pub raw: f64,
    /// True only after the raw capture path has supplied a real sample.
    pub raw_ready: bool,
    /// Coarse FFT bins (0-1).
    pub fft: [f32; FFT_BINS],
    /// Full-resolution spectrum (0-1).
    pub spectrum: [f32; SPECTRUM_LEN],
    /// Time-domain waveform (0-1, 0.5 = silence).
    pub waveform: [f32; WAVEFORM_LEN],
    smoothing: SmoothingBuffers,
    frequency_data: Option<Vec<u8>>,
    max_buffer_length: f64,
    devices: Option<IndexMap<String, AudioDeviceEntry>>,
    devices_by_name: Option<IndexMap<String, Option<String>>>,
    device_inventory: Option<IndexMap<String, Option<String>>>,
    default_channels: Option<IndexMap<u32, AudioState>>,
    default_connected: bool,
}

impl fmt::Debug for AudioState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AudioState")
            .field("low", &self.low)
            .field("mid", &self.mid)
            .field("high", &self.high)
            .field("vol", &self.vol)
            .field("raw", &self.raw)
            .field("raw_ready", &self.raw_ready)
            .finish_non_exhaustive()
    }
}

impl Default for AudioState {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioState {
    /// `new AudioState()`: with a device registry.
    pub fn new() -> Self {
        Self::with_registry(true)
    }

    /// `new AudioState({ deviceRegistry: false })`.
    pub fn without_device_registry() -> Self {
        Self::with_registry(false)
    }

    fn with_registry(registry: bool) -> Self {
        AudioState {
            low: 0.0,
            mid: 0.0,
            high: 0.0,
            vol: 0.0,
            raw: 0.0,
            raw_ready: false,
            fft: [0.0; FFT_BINS],
            spectrum: [0.0; SPECTRUM_LEN],
            waveform: [0.5; WAVEFORM_LEN],
            smoothing: SmoothingBuffers::default(),
            frequency_data: None,
            max_buffer_length: 5.0,
            devices: registry.then(IndexMap::new),
            devices_by_name: registry.then(IndexMap::new),
            device_inventory: None,
            default_channels: registry.then(IndexMap::new),
            default_connected: false,
        }
    }

    /// Whether this state keeps a device registry (the root state).
    pub fn has_device_registry(&self) -> bool {
        self.devices.is_some()
    }

    /// `setDeviceInventory(devices)`: complete device discovery, independent of
    /// the devices being captured. Disconnected and unnamed entries are
    /// skipped; a name seen with two ids becomes ambiguous.
    pub fn set_device_inventory(&mut self, devices: &[AudioDeviceInfo]) {
        let mut names: IndexMap<String, Option<String>> = IndexMap::new();
        for device in devices {
            if !device.connected || device.id.is_empty() || device.name.is_empty() {
                continue;
            }
            let value = match names.get(&device.name) {
                None => Some(device.id.clone()),
                Some(Some(previous)) if *previous == device.id => Some(device.id.clone()),
                Some(_) => None,
            };
            names.insert(device.name.clone(), value);
        }
        self.device_inventory = Some(names);
    }

    /// The physical inventory (`_deviceInventory`): name to id, `None` when
    /// ambiguous.
    pub fn device_inventory(&self) -> Option<&IndexMap<String, Option<String>>> {
        self.device_inventory.as_ref()
    }

    /// Connected-name index (`_devicesByName`): name to device id, `None` when
    /// ambiguous.
    pub fn devices_by_name(&self) -> Option<&IndexMap<String, Option<String>>> {
        self.devices_by_name.as_ref()
    }

    /// `registerDefaultChannels(channelCount)`: registers the independently
    /// analysed channels of the default input (1-32). Returns `false` for an
    /// invalid count or without a registry.
    pub fn register_default_channels(&mut self, channel_count: u32) -> bool {
        let Some(channels) = self.default_channels.as_mut() else {
            return false;
        };
        if !(1..=MAX_SELECTED_CHANNEL).contains(&channel_count) {
            return false;
        }
        self.default_connected = true;
        for channel in 1..=channel_count {
            channels
                .entry(channel)
                .or_insert_with(AudioState::without_device_registry);
        }
        let stale: Vec<u32> = channels
            .keys()
            .copied()
            .filter(|&channel| channel > channel_count)
            .collect();
        for channel in stale {
            if let Some(mut state) = channels.shift_remove(&channel) {
                state.reset();
            }
        }
        true
    }

    /// `getDefaultChannelState(channel)`: a default-input channel, while the
    /// default input is connected.
    pub fn get_default_channel_state(&self, channel: f64) -> Option<&AudioState> {
        if !self.default_connected || !valid_channel(channel) {
            return None;
        }
        self.default_channels.as_ref()?.get(&(channel as u32))
    }

    /// Mutable `getDefaultChannelState(channel)`.
    pub fn get_default_channel_state_mut(&mut self, channel: f64) -> Option<&mut AudioState> {
        if !self.default_connected || !valid_channel(channel) {
            return None;
        }
        self.default_channels.as_mut()?.get_mut(&(channel as u32))
    }

    /// The default input's channels (`_defaultChannels`).
    pub fn default_channels(&self) -> Option<&IndexMap<u32, AudioState>> {
        self.default_channels.as_ref()
    }

    /// Mutable access to the default input's channels, connected or not.
    pub fn default_channels_mut(&mut self) -> Option<&mut IndexMap<u32, AudioState>> {
        self.default_channels.as_mut()
    }

    /// Whether the default input is connected (`_defaultConnected`).
    pub fn default_connected(&self) -> bool {
        self.default_connected
    }

    /// `disconnectDefaultInput()`: makes the default channels unavailable and
    /// clears their samples.
    pub fn disconnect_default_input(&mut self) {
        self.default_connected = false;
        for state in self
            .default_channels
            .iter_mut()
            .flat_map(|c| c.values_mut())
        {
            state.reset();
        }
    }

    /// `updateFromAnalyser(analyser, smoothing)`: extracts the low, mid and
    /// high bands (rolling average over `smoothing` frames, clamped to 1-10),
    /// the 16 coarse FFT bins and the volume. Does nothing without an analyser.
    pub fn update_from_analyser(
        &mut self,
        analyser: Option<&mut dyn FrequencyDataSource>,
        smoothing: f64,
    ) {
        let Some(analyser) = analyser else {
            return;
        };
        self.max_buffer_length = math_max(1.0, math_min(10.0, smoothing));
        let bins = analyser.frequency_bin_count();
        if self
            .frequency_data
            .as_ref()
            .is_none_or(|data| data.len() != bins)
        {
            self.frequency_data = Some(vec![0; bins]);
        }
        let mut buf = self.frequency_data.take().expect("frequency buffer");
        analyser.get_byte_frequency_data(&mut buf);

        // With fftSize=256 at 44.1kHz: 128 bins, each ~172Hz wide.
        let raw_low = avg_bins(&buf, 1, 2);
        let raw_mid = avg_bins(&buf, 2, 12);
        let raw_high = avg_bins(&buf, 12, 47);

        self.low = self.smooth(Band::Low, raw_low);
        self.mid = self.smooth(Band::Mid, raw_mid);
        self.high = self.smooth(Band::High, raw_high);

        let step = (buf.len() / 16).max(1);
        let mut sum = 0.0;
        for i in 0..FFT_BINS {
            // An index past the buffer reads `undefined`: NaN.
            let v = buf
                .get(i * step)
                .map_or(f64::NAN, |&b| f64::from(b) / 255.0);
            self.fft[i] = v as f32;
            sum += v;
        }
        self.vol = sum / 16.0;
        self.frequency_data = Some(buf);
    }

    /// The analyser byte buffer reused between frames (`_frequencyData`).
    pub fn frequency_data(&self) -> Option<&[u8]> {
        self.frequency_data.as_deref()
    }

    /// `setBands(low, mid, high)`: sets the bands directly, clamped to 0-1, and
    /// the volume to their mean.
    pub fn set_bands(&mut self, low: f64, mid: f64, high: f64) {
        self.low = math_clamp(low, 0.0, 1.0);
        self.mid = math_clamp(mid, 0.0, 1.0);
        self.high = math_clamp(high, 0.0, 1.0);
        self.vol = (self.low + self.mid + self.high) / 3.0;
    }

    /// `setRaw(value)`: one signed time-domain sample, clamped to -1..1
    /// (non-finite values store 0), and marks the raw signal ready.
    pub fn set_raw(&mut self, value: f64) {
        self.raw = if value.is_finite() {
            math_clamp(value, -1.0, 1.0)
        } else {
            0.0
        };
        self.raw_ready = true;
    }

    /// `setRawUnavailable()`: raw input unavailable, without representing it
    /// as a real zero sample.
    pub fn set_raw_unavailable(&mut self) {
        self.raw = 0.0;
        self.raw_ready = false;
    }

    /// `registerDevice(device)`: registers or reconnects one capture device and
    /// its one-based channels. A missing or zero channel count is 1. Returns
    /// `false` for an empty id or without a registry.
    pub fn register_device(&mut self, id: &str, name: &str, channel_count: Option<u32>) -> bool {
        let Some(devices) = self.devices.as_mut() else {
            return false;
        };
        if id.is_empty() {
            return false;
        }
        let channel_count = match channel_count {
            Some(count) if count >= 1 => count,
            _ => 1,
        };
        let topology_changed = match devices.get_mut(id) {
            None => {
                devices.insert(
                    id.to_string(),
                    AudioDeviceEntry {
                        id: id.to_string(),
                        name: name.to_string(),
                        connected: true,
                        channel_count,
                        channels: IndexMap::new(),
                    },
                );
                true
            }
            Some(entry) => {
                let changed =
                    entry.name != name || !entry.connected || entry.channel_count != channel_count;
                entry.name = name.to_string();
                entry.connected = true;
                entry.channel_count = channel_count;
                changed
            }
        };
        let entry = devices.get_mut(id).expect("registered device");
        for channel in 1..=channel_count {
            entry
                .channels
                .entry(channel)
                .or_insert_with(AudioState::without_device_registry);
        }
        let stale: Vec<u32> = entry
            .channels
            .keys()
            .copied()
            .filter(|&channel| channel > channel_count)
            .collect();
        for channel in stale {
            if let Some(mut state) = entry.channels.shift_remove(&channel) {
                state.reset();
            }
        }
        if topology_changed {
            self.rebuild_device_name_index();
        }
        true
    }

    /// `setChannelValues(id, channel, values)`: updates one connected device
    /// channel; levels clamp to 0-1 and `raw` goes through `setRaw`.
    pub fn set_channel_values(&mut self, id: &str, channel: u32, values: &ChannelValues) -> bool {
        let Some(state) = self
            .devices
            .as_mut()
            .and_then(|devices| devices.get_mut(id))
            .filter(|entry| entry.connected)
            .and_then(|entry| entry.channels.get_mut(&channel))
        else {
            return false;
        };
        for (value, field) in [
            (values.low, &mut state.low),
            (values.mid, &mut state.mid),
            (values.high, &mut state.high),
            (values.vol, &mut state.vol),
        ] {
            if let Some(value) = value.filter(|v| v.is_finite()) {
                *field = math_clamp(value, 0.0, 1.0);
            }
        }
        if let Some(raw) = values.raw.filter(|v| v.is_finite()) {
            state.set_raw(raw);
        }
        true
    }

    /// `setDeviceRawUnavailable(id)`: marks raw samples unavailable on every
    /// channel of a device.
    pub fn set_device_raw_unavailable(&mut self, id: &str) {
        let Some(entry) = self
            .devices
            .as_mut()
            .and_then(|devices| devices.get_mut(id))
        else {
            return;
        };
        for state in entry.channels.values_mut() {
            state.set_raw_unavailable();
        }
    }

    fn device_for(&self, selector: &DeviceSelector<'_>) -> Option<&AudioDeviceEntry> {
        let devices = self.devices.as_ref()?;
        if let SelectorKey::Str(id) = selector.id {
            return devices.get(id);
        }
        if selector.id != SelectorKey::Falsy {
            return None;
        }
        let SelectorKey::Str(name) = selector.name else {
            return None;
        };
        let id = match &self.device_inventory {
            Some(inventory) => inventory.get(name).cloned().flatten(),
            None => self.devices_by_name.as_ref()?.get(name).cloned().flatten(),
        }?;
        devices.get(&id)
    }

    /// `getDeviceChannelState(selector)`: the state a compiled `audio()`
    /// descriptor reads. An exact id is authoritative; a name-only selector
    /// must match one connected device; a channel-only selector reads the
    /// default input; an empty selector is the aggregate (`self`).
    pub fn get_device_channel_state(&self, selector: &DeviceSelector<'_>) -> Option<&AudioState> {
        let has_name = selector.name != SelectorKey::Falsy;
        let has_id = selector.id != SelectorKey::Falsy;
        if !has_name && !has_id && selector.channel.is_none() {
            return Some(self);
        }
        let channel = selector.channel.filter(|&c| valid_channel(c))?;
        if !has_name && !has_id {
            return self.get_default_channel_state(channel);
        }
        let entry = self.device_for(selector).filter(|entry| entry.connected)?;
        entry.channels.get(&(channel as u32))
    }

    /// Mutable `getDeviceChannelState(selector)`.
    pub fn get_device_channel_state_mut(
        &mut self,
        selector: &DeviceSelector<'_>,
    ) -> Option<&mut AudioState> {
        let has_name = selector.name != SelectorKey::Falsy;
        let has_id = selector.id != SelectorKey::Falsy;
        if !has_name && !has_id && selector.channel.is_none() {
            return Some(self);
        }
        let channel = selector.channel.filter(|&c| valid_channel(c))?;
        if !has_name && !has_id {
            return self.get_default_channel_state_mut(channel);
        }
        let id = self
            .device_for(selector)
            .filter(|entry| entry.connected)?
            .id
            .clone();
        self.devices
            .as_mut()?
            .get_mut(&id)?
            .channels
            .get_mut(&(channel as u32))
    }

    /// `disconnectDevice(id)`: marks a device unavailable, keeping its
    /// identity, and clears its channels.
    pub fn disconnect_device(&mut self, id: &str) {
        let Some(entry) = self
            .devices
            .as_mut()
            .and_then(|devices| devices.get_mut(id))
        else {
            return;
        };
        entry.connected = false;
        for state in entry.channels.values_mut() {
            state.reset();
        }
        self.rebuild_device_name_index();
    }

    fn rebuild_device_name_index(&mut self) {
        let (Some(devices), Some(by_name)) = (&self.devices, &mut self.devices_by_name) else {
            return;
        };
        by_name.clear();
        for entry in devices.values() {
            if !entry.connected || entry.name.is_empty() {
                continue;
            }
            if by_name.contains_key(&entry.name) {
                by_name.insert(entry.name.clone(), None);
            } else {
                by_name.insert(entry.name.clone(), Some(entry.id.clone()));
            }
        }
    }

    /// `getDevices()`: identity, connection state and channel count of every
    /// registered device.
    pub fn get_devices(&self) -> Vec<AudioDeviceInfo> {
        self.device_entries()
            .map(|entry| AudioDeviceInfo {
                id: entry.id.clone(),
                name: entry.name.clone(),
                connected: entry.connected,
                channel_count: entry.channel_count,
            })
            .collect()
    }

    /// The registered device entries, in registration order.
    pub fn device_entries(&self) -> impl Iterator<Item = &AudioDeviceEntry> {
        self.devices.iter().flat_map(|devices| devices.values())
    }

    /// The registered device entry with `id`.
    pub fn device_entry(&self, id: &str) -> Option<&AudioDeviceEntry> {
        self.devices.as_ref()?.get(id)
    }

    /// Mutable access to the registered device entry with `id`, connected or
    /// not.
    pub fn device_entry_mut(&mut self, id: &str) -> Option<&mut AudioDeviceEntry> {
        self.devices.as_mut()?.get_mut(id)
    }

    /// `setSpectrum(frequencyData)`: the first 128 bytes, scaled to 0-1.
    pub fn set_spectrum(&mut self, frequency_data: &[u8]) {
        for (dst, &src) in self.spectrum.iter_mut().zip(frequency_data) {
            *dst = (f64::from(src) / 255.0) as f32;
        }
    }

    /// `setWaveform(timeDomainData)`: the first 128 bytes, scaled to 0-1.
    pub fn set_waveform(&mut self, time_domain_data: &[u8]) {
        for (dst, &src) in self.waveform.iter_mut().zip(time_domain_data) {
            *dst = (f64::from(src) / 255.0) as f32;
        }
    }

    /// `_smooth(band, value)`: pushes a value into the band's rolling buffer
    /// (dropping one old value when over the limit) and returns the mean.
    pub fn smooth(&mut self, band: Band, value: f64) -> f64 {
        let max = self.max_buffer_length;
        let buffer = match band {
            Band::Low => &mut self.smoothing.low,
            Band::Mid => &mut self.smoothing.mid,
            Band::High => &mut self.smoothing.high,
        };
        buffer.push(value);
        if buffer.len() as f64 > max {
            buffer.remove(0);
        }
        buffer.iter().fold(0.0, |a, b| a + b) / buffer.len() as f64
    }

    /// The rolling smoothing buffers (`_smoothingBuffers`).
    pub fn smoothing_buffers(&self) -> &SmoothingBuffers {
        &self.smoothing
    }

    /// Mutable rolling smoothing buffers.
    pub fn smoothing_buffers_mut(&mut self) -> &mut SmoothingBuffers {
        &mut self.smoothing
    }

    /// The smoothing frame count (`_maxBufferLength`).
    pub fn max_buffer_length(&self) -> f64 {
        self.max_buffer_length
    }

    /// Sets the smoothing frame count directly (`_maxBufferLength = n`).
    pub fn set_max_buffer_length(&mut self, frames: f64) {
        self.max_buffer_length = frames;
    }

    /// `resetAggregate()`: clears the legacy aggregate (bands, raw, fft,
    /// spectrum, waveform, smoothing) without disturbing channel states.
    pub fn reset_aggregate(&mut self) {
        self.low = 0.0;
        self.mid = 0.0;
        self.high = 0.0;
        self.vol = 0.0;
        self.raw = 0.0;
        self.raw_ready = false;
        self.fft = [0.0; FFT_BINS];
        self.spectrum = [0.0; SPECTRUM_LEN];
        self.waveform = [0.5; WAVEFORM_LEN];
        self.smoothing = SmoothingBuffers::default();
    }

    /// `reset()`: the aggregate and every default and device channel.
    pub fn reset(&mut self) {
        self.reset_aggregate();
        for state in self
            .default_channels
            .iter_mut()
            .flat_map(|c| c.values_mut())
        {
            state.reset();
        }
        for entry in self.devices.iter_mut().flat_map(|d| d.values_mut()) {
            for state in entry.channels.values_mut() {
                state.reset();
            }
        }
    }
}

/// `Number.isInteger(channel) && channel >= 1 && channel <= 32`.
fn valid_channel(channel: f64) -> bool {
    channel.is_finite()
        && channel.trunc() == channel
        && (1.0..=f64::from(MAX_SELECTED_CHANNEL)).contains(&channel)
}

#[cfg(test)]
mod tests;
