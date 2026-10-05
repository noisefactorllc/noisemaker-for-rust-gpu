//! An [`AudioBackend`] over cpal.
//!
//! Devices are identified by cpal's stable [`cpal::DeviceId`] text and named
//! by their description, the counterparts of `deviceId` and `label`. Each
//! capture uses the device's default input configuration; samples of any
//! format are converted to float and queued for the manager's next update.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

use crate::audio_input::{AudioBackend, AudioInputDevice, CaptureInfo, CaptureSink};

/// Audio capture through cpal.
pub struct CpalBackend {
    host: cpal::Host,
}

impl std::fmt::Debug for CpalBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpalBackend")
            .field("host", &self.host.id())
            .finish()
    }
}

impl Default for CpalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpalBackend {
    /// The platform's default host.
    pub fn new() -> Self {
        Self::with_host(cpal::default_host())
    }

    /// A specific cpal host.
    pub fn with_host(host: cpal::Host) -> Self {
        CpalBackend { host }
    }
}

fn device_id(device: &cpal::Device) -> Option<String> {
    device.id().ok().map(|id| id.to_string())
}

fn device_name(device: &cpal::Device) -> String {
    device
        .description()
        .map(|description| description.name().to_string())
        .unwrap_or_default()
}

fn build<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    sink: CaptureSink,
    channels: usize,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                let samples: Vec<f32> = data.iter().map(|&s| s.to_sample::<f32>()).collect();
                sink.push_interleaved(&samples, channels);
            },
            |_error| {},
            None,
        )
        .map_err(|error| error.to_string())
}

fn open(device: &cpal::Device, sink: CaptureSink) -> Result<(cpal::Stream, CaptureInfo), String> {
    let supported = device
        .default_input_config()
        .map_err(|error| error.to_string())?;
    let channels = usize::from(supported.channels());
    let config = supported.config();
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build::<f32>(device, config, sink, channels),
        SampleFormat::F64 => build::<f64>(device, config, sink, channels),
        SampleFormat::I8 => build::<i8>(device, config, sink, channels),
        SampleFormat::I16 => build::<i16>(device, config, sink, channels),
        SampleFormat::I32 => build::<i32>(device, config, sink, channels),
        SampleFormat::I64 => build::<i64>(device, config, sink, channels),
        SampleFormat::U8 => build::<u8>(device, config, sink, channels),
        SampleFormat::U16 => build::<u16>(device, config, sink, channels),
        SampleFormat::U32 => build::<u32>(device, config, sink, channels),
        SampleFormat::U64 => build::<u64>(device, config, sink, channels),
        other => Err(format!("unsupported sample format {other}")),
    }?;
    stream.play().map_err(|error| error.to_string())?;
    let info = CaptureInfo {
        device_id: device_id(device),
        label: device_name(device),
        channel_count: Some(channels as u32),
    };
    Ok((stream, info))
}

impl AudioBackend for CpalBackend {
    type Stream = cpal::Stream;

    fn open_default(&mut self, sink: CaptureSink) -> Result<(Self::Stream, CaptureInfo), String> {
        let device = self
            .host
            .default_input_device()
            .ok_or_else(|| "no default input device".to_string())?;
        open(&device, sink)
    }

    fn open_device(
        &mut self,
        id: &str,
        sink: CaptureSink,
    ) -> Result<(Self::Stream, CaptureInfo), String> {
        let device = self
            .host
            .input_devices()
            .map_err(|error| error.to_string())?
            .find(|device| device_id(device).as_deref() == Some(id))
            .ok_or_else(|| format!("audio input device {id} not found"))?;
        open(&device, sink)
    }

    fn enumerate(&mut self) -> Vec<AudioInputDevice> {
        let Ok(devices) = self.host.input_devices() else {
            return Vec::new();
        };
        devices
            .filter_map(|device| {
                Some(AudioInputDevice {
                    id: device_id(&device)?,
                    name: device_name(&device),
                })
            })
            .collect()
    }
}
