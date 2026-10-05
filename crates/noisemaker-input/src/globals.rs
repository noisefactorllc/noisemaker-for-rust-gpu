//! The external-input part of the reference pipeline's
//! `updateGlobalUniforms` (`shaders/src/runtime/pipeline.js`): the
//! `audioWaveform` and `audioSpectrum` uniform arrays read by synth/scope and
//! synth/spectrum, the `midiNoteGrid` texture read by synth/roll, and
//! `midiClockCount`.

use crate::audio::{AudioState, SPECTRUM_LEN, WAVEFORM_LEN};
use crate::midi::{MidiState, NOTE_GRID_HEIGHT, NOTE_GRID_LEN, NOTE_GRID_WIDTH};

/// Width of the `midiNoteGrid` texture.
pub const MIDI_NOTE_GRID_WIDTH: u32 = NOTE_GRID_WIDTH as u32;
/// Height of the `midiNoteGrid` texture.
pub const MIDI_NOTE_GRID_HEIGHT: u32 = NOTE_GRID_HEIGHT as u32;

/// The external-input global uniforms. They persist between frames like the
/// pipeline's `globalUniforms` object: a value is only replaced when its
/// source is present.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputGlobals {
    /// `audioWaveform`: 128 values, 0.5 = silence; `None` until an audio
    /// state has been set.
    pub audio_waveform: Option<[f32; WAVEFORM_LEN]>,
    /// `audioSpectrum`: 128 values; `None` until an audio state has been set.
    pub audio_spectrum: Option<[f32; SPECTRUM_LEN]>,
    /// `midiClockCount`.
    pub midi_clock_count: f64,
}

/// What to upload as the `midiNoteGrid` texture this frame
/// (`uploadDataTexture('midiNoteGrid', data, 128, 16)`, RGBA float).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NoteGridUpload<'a> {
    /// The MIDI state's grid, just refreshed by `updateNoteGrid()`.
    State(&'a [f32; NOTE_GRID_LEN]),
    /// No MIDI state but a pass samples `midiNoteGrid`: an all-zero grid.
    Empty,
    /// Nothing to upload.
    None,
}

/// All-zero note grid for [`NoteGridUpload::Empty`].
pub static EMPTY_NOTE_GRID: [f32; NOTE_GRID_LEN] = [0.0; NOTE_GRID_LEN];

impl InputGlobals {
    /// The per-frame update of `updateGlobalUniforms`: copies the audio
    /// state's waveform and spectrum, refreshes the MIDI note grid and clock
    /// count, and says what to upload as `midiNoteGrid`. `needs_note_grid` is
    /// the pipeline's `_needsMidiNoteGrid` (a pass input names `midiNoteGrid`).
    pub fn update<'a>(
        &mut self,
        midi: Option<&'a mut MidiState>,
        audio: Option<&AudioState>,
        needs_note_grid: bool,
    ) -> NoteGridUpload<'a> {
        if let Some(audio) = audio {
            self.audio_waveform = Some(audio.waveform);
            self.audio_spectrum = Some(audio.spectrum);
        }
        let upload = match midi {
            Some(midi) => {
                midi.update_note_grid();
                self.midi_clock_count = midi.clock_count as f64;
                NoteGridUpload::State(&midi.note_grid)
            }
            None if needs_note_grid => NoteGridUpload::Empty,
            None => NoteGridUpload::None,
        };
        // `g.midiClockCount = g.midiClockCount || 0`: NaN reads as 0.
        if self.midi_clock_count.is_nan() {
            self.midi_clock_count = 0.0;
        }
        upload
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globals_follow_the_present_sources() {
        let mut globals = InputGlobals::default();
        assert_eq!(globals.update(None, None, false), NoteGridUpload::None);
        assert_eq!(globals.update(None, None, true), NoteGridUpload::Empty);
        assert!(globals.audio_waveform.is_none());

        let mut audio = AudioState::new();
        audio.set_waveform(&[255; 128]);
        let mut midi = MidiState::new();
        midi.handle_message(&[0x90, 60, 127], None);
        midi.handle_message(&[0xf8], None);
        match globals.update(Some(&mut midi), Some(&audio), false) {
            NoteGridUpload::State(grid) => assert_eq!((grid[240], grid[241]), (1.0, 1.0)),
            other => panic!("{other:?}"),
        }
        assert_eq!(globals.midi_clock_count, 1.0);
        assert_eq!(globals.audio_waveform.unwrap()[0], 1.0);

        // Removed sources keep their last values, as the reference's
        // persistent globalUniforms object does.
        globals.update(None, None, false);
        assert_eq!(globals.midi_clock_count, 1.0);
        assert_eq!(globals.audio_waveform.unwrap()[0], 1.0);
    }
}
