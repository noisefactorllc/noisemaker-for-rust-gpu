//! External input state of the Noisemaker engine: a port of the reference's
//! `shaders/src/runtime/external-input.js` and of the parts of
//! `shaders/src/runtime/pipeline.js` that read it.
//!
//! - [`midi`]: `MidiChannelState` and `MidiState` — per-port isolated states
//!   behind a legacy aggregate, CC/CC14 ownership by origin, NRPN/RPN, RP-015
//!   reset, MPE zones and voices, the 128x16 note grid, the clock count.
//! - [`audio`]: `AudioState` — bands, smoothing, raw signal readiness,
//!   spectrum and waveform, capture devices and their channels, the default
//!   input's channels.
//! - [`analyser`]: [`AudioAnalyzer`], Chromium's `AnalyserNode` (the reference
//!   host fills its audio state from it), bit-exact with Chromium 153.
//! - [`automation`]: `evaluateAutomation` and its oscillator, MIDI and audio
//!   evaluators, `resolveUniformValue` and `getAudioInputRequirements`, over
//!   [`js::JsValue`] descriptors, generic over [`automation::MidiSource`] and
//!   [`automation::AudioSource`].
//! - [`globals`]: the `audioWaveform`, `audioSpectrum`, `midiNoteGrid` and
//!   `midiClockCount` globals of `updateGlobalUniforms`.
//! - [`midi_input`] and [`audio_input`]: `MidiInputManager` and
//!   `AudioInputManager` over pluggable backends; [`host`] has the `midir` and
//!   `cpal` backends behind features of the same names (off by default).
//!
//! `Date.now()` is an injectable [`clock::Clock`]. JavaScript number semantics
//! are the workspace's ([`noisemaker_dsl::js`](mod@noisemaker_dsl::js)), except V8's `Math.sin` and
//! `Math.cos`, which [`jsmath`] ports as Node computes them.

pub mod analyser;
pub mod audio;
pub mod audio_input;
pub mod automation;
pub mod clock;
pub mod globals;
pub mod host;
pub mod js;
pub mod jsmath;
pub mod midi;
pub mod midi_input;

pub use analyser::AudioAnalyzer;
pub use audio::AudioState;
pub use audio_input::{AudioBackend, AudioInputManager};
pub use automation::{
    AudioInputRequirements, AutomationContext, ExternalState, evaluate_automation,
    resolve_uniform_value,
};
pub use clock::{Clock, ManualClock, SystemClock};
pub use globals::{InputGlobals, NoteGridUpload};
pub use js::JsValue;
pub use midi::{MidiChannelState, MidiState};
pub use midi_input::{MidiBackend, MidiInputManager};
