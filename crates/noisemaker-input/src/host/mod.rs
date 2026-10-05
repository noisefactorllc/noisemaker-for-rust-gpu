//! Host device integrations, each behind a cargo feature (off by default).
//!
//! - `midir`: `midir::MidirBackend`, a [`crate::midi_input::MidiBackend`]
//!   over the midir crate (CoreMIDI, ALSA, WinMM/WinRT, Web MIDI).
//! - `cpal`: `cpal::CpalBackend`, a [`crate::audio_input::AudioBackend`]
//!   over the cpal crate (CoreAudio, ALSA/JACK/PipeWire, WASAPI).

#[cfg(feature = "cpal")]
pub mod cpal;
#[cfg(feature = "midir")]
pub mod midir;
