//! `ExternalInputManager` (`external-input.js`): the MIDI and audio input
//! managers of one host, with one status callback for both.

use std::sync::{Arc, Mutex};

use crate::audio_input::{AudioBackend, AudioInputManager};
use crate::midi_input::{MidiBackend, MidiInputManager};

/// `ExternalInputManager`: a [`MidiInputManager`] and an
/// [`AudioInputManager`] side by side.
pub struct ExternalInputManager<M: MidiBackend, A: AudioBackend> {
    /// `this.midi`.
    pub midi: MidiInputManager<M>,
    /// `this.audio`.
    pub audio: AudioInputManager<A>,
}

impl<M: MidiBackend, A: AudioBackend> ExternalInputManager<M, A> {
    /// `new ExternalInputManager(renderer)`: disabled managers over the given
    /// backends.
    pub fn new(midi: M, audio: A) -> Self {
        ExternalInputManager {
            midi: MidiInputManager::new(midi),
            audio: AudioInputManager::new(audio),
        }
    }

    /// `onStatusChange(callback)`: `callback` receives the status messages of
    /// both managers.
    pub fn on_status_change(&mut self, callback: impl FnMut(&str) + Send + 'static) {
        let shared = Arc::new(Mutex::new(callback));
        let midi = shared.clone();
        self.midi.on_status_change(move |message, _status| {
            if let Ok(mut cb) = midi.lock() {
                cb(message);
            }
        });
        self.audio.on_status_change(move |message| {
            if let Ok(mut cb) = shared.lock() {
                cb(message);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioState;
    use crate::audio_input::{AudioInputDevice, CaptureInfo, CaptureSink};
    use crate::midi::MidiState;
    use crate::midi_input::{MidiAccessError, MidiInputInfo, MidiSink};

    struct NoMidi;

    impl MidiBackend for NoMidi {
        type Connection = ();
        fn request_access(&mut self) -> Result<(), MidiAccessError> {
            Err(MidiAccessError::Unsupported)
        }
        fn inputs(&mut self) -> Vec<MidiInputInfo> {
            Vec::new()
        }
        fn connect(&mut self, _: &MidiInputInfo, _: MidiSink) -> Result<(), String> {
            Err("no MIDI".into())
        }
    }

    struct NoAudio;

    impl AudioBackend for NoAudio {
        type Stream = ();
        fn is_supported(&self) -> bool {
            false
        }
        fn open_default(&mut self, _: CaptureSink) -> Result<((), CaptureInfo), String> {
            Err("no audio".into())
        }
        fn open_device(&mut self, _: &str, _: CaptureSink) -> Result<((), CaptureInfo), String> {
            Err("no audio".into())
        }
        fn enumerate(&mut self) -> Vec<AudioInputDevice> {
            Vec::new()
        }
    }

    #[test]
    fn one_callback_for_both_managers() {
        let mut manager = ExternalInputManager::new(NoMidi, NoAudio);
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = seen.clone();
        manager.on_status_change(move |m| sink.lock().unwrap().push(m.to_owned()));
        assert!(!manager.midi.enable(&mut MidiState::new()));
        assert!(!manager.audio.enable(&mut AudioState::new()));
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "{seen:?}");
        assert_eq!(seen[1], "Audio input not supported");
    }
}
