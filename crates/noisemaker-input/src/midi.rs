//! MIDI input state: a port of `MidiChannelState` and `MidiState` from the
//! reference's `shaders/src/runtime/external-input.js`.
//!
//! The Web MIDI plumbing of the reference (`MIDIAccess`, `MIDIInput.open`,
//! `onmidimessage`) becomes plain methods the host calls: [`MidiState::register_port`],
//! [`MidiState::disconnect_port`], [`MidiState::set_port_inventory`] and
//! [`MidiState::handle_message`]. Everything else follows the reference statement
//! by statement: per-port isolated states behind a legacy aggregate, CC and paired
//! 14-bit CC ownership by origin, NRPN/RPN transactions, RP-015 controller reset,
//! MPE zone configuration (RPN 6) and zone voice selection, the 128x16 note grid,
//! and the 24 PPQ clock count.
//!
//! JavaScript `Map`s are [`IndexMap`]s with the same insertion-order behaviour
//! (`set` on an existing key keeps its position, `delete` keeps the order of the
//! rest), so every observable iteration order matches the reference.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;

use crate::clock::{Clock, SystemClock};

/// Number of MIDI channels.
pub const MIDI_CHANNELS: usize = 16;
/// Width of the note grid texture (one texel per key).
pub const NOTE_GRID_WIDTH: usize = 128;
/// Height of the note grid texture (one row per channel).
pub const NOTE_GRID_HEIGHT: usize = 16;
/// Number of `f32` values in the RGBA note grid.
pub const NOTE_GRID_LEN: usize = NOTE_GRID_WIDTH * NOTE_GRID_HEIGHT * 4;
/// Pitch bend centre value.
pub const PITCH_BEND_CENTER: u16 = 8192;

/// `midiNoteOrder`: the reference's module-level note-on counter. Every
/// note-on that is not a copy of a port's note takes the next value, across
/// all states of the process, exactly like `++midiNoteOrder`.
static MIDI_NOTE_ORDER: AtomicU64 = AtomicU64::new(0);

fn next_note_order() -> u64 {
    MIDI_NOTE_ORDER.fetch_add(1, Ordering::SeqCst) + 1
}

/// The current value of the process-wide note-on counter (the order given to
/// the most recent note-on). Parity tools use it as a baseline.
pub fn note_order_counter() -> u64 {
    MIDI_NOTE_ORDER.load(Ordering::SeqCst)
}

/// `retainedResetControllers`: controllers RP-015 Reset All Controllers keeps.
pub fn is_retained_reset_controller(cc: u8) -> bool {
    matches!(cc, 0 | 7 | 10 | 32 | 70..=79 | 91..=95)
}

/// Where a value came from: a Web MIDI port id, or messages without a port
/// (the reference's `unscopedMidiOrigin` symbol). Values written through the
/// channel API directly have no origin (`null`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MidiOrigin {
    /// Messages handled without a port.
    Unscoped,
    /// Messages from the port with this id.
    Port(Arc<str>),
}

impl MidiOrigin {
    /// The origin of messages from the port `id`.
    pub fn port(id: &str) -> Self {
        MidiOrigin::Port(Arc::from(id))
    }

    fn is_port(&self, id: &str) -> bool {
        matches!(self, MidiOrigin::Port(port) if &**port == id)
    }
}

/// One held note of a channel (`heldNotes` entries).
#[derive(Clone, Debug, PartialEq)]
pub struct HeldNote {
    /// Note number.
    pub key: u8,
    /// Note-on velocity.
    pub velocity: u8,
    /// Note-on time in milliseconds (`Date.now()` or the source port's time).
    pub time: f64,
    /// Note-on order (`midiNoteOrder`).
    pub order: u64,
    /// Where the note came from.
    pub origin: Option<MidiOrigin>,
}

/// The two parameter-number families of the Data Entry controllers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ParameterFamily {
    /// Non-registered parameter numbers (CC 99/98).
    Nrpn,
    /// Registered parameter numbers (CC 101/100).
    Rpn,
}

impl ParameterFamily {
    /// `'nrpn'` or `'rpn'`.
    pub fn as_str(self) -> &'static str {
        match self {
            ParameterFamily::Nrpn => "nrpn",
            ParameterFamily::Rpn => "rpn",
        }
    }
}

/// The parameter-number selector bytes of each family (`_selectors`):
/// `[msb, lsb]`, `None` until received.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParameterSelectors {
    /// NRPN `[CC99, CC98]`.
    pub nrpn: [Option<u8>; 2],
    /// RPN `[CC101, CC100]`.
    pub rpn: [Option<u8>; 2],
}

impl ParameterSelectors {
    fn family(&self, family: ParameterFamily) -> [Option<u8>; 2] {
        match family {
            ParameterFamily::Nrpn => self.nrpn,
            ParameterFamily::Rpn => self.rpn,
        }
    }

    fn family_mut(&mut self, family: ParameterFamily) -> &mut [Option<u8>; 2] {
        match family {
            ParameterFamily::Nrpn => &mut self.nrpn,
            ParameterFamily::Rpn => &mut self.rpn,
        }
    }
}

/// A completed Data Entry update (`{family, parameter, value}`), with the
/// channels reset by an MPE configuration message (`resetChannels`) when the
/// update was RPN 6 Data Entry MSB.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterChange {
    /// Parameter family.
    pub family: ParameterFamily,
    /// 14-bit parameter number.
    pub parameter: u16,
    /// New 14-bit value.
    pub value: u16,
    /// Channels (1-16) whose zone ownership changed; present only for MPE
    /// configuration messages.
    pub reset_channels: Option<Vec<u8>>,
}

/// MPE zone member counts (`mpeZones`); `None` until configured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MpeZones {
    /// Lower zone member count (manager channel 1).
    pub lower: Option<u8>,
    /// Upper zone member count (manager channel 16).
    pub upper: Option<u8>,
}

/// State of one MIDI channel (`MidiChannelState`).
#[derive(Clone)]
pub struct MidiChannelState {
    /// Last note number.
    pub key: u8,
    /// Last note-on velocity.
    pub velocity: u8,
    /// Gate: 1 while the last note is on, 0 otherwise.
    pub gate: u8,
    /// Time of the last note-on in milliseconds.
    pub time: f64,
    /// Per-key velocity of held notes (0 = released).
    pub keys: [u8; 128],
    /// Last value of every 7-bit controller.
    pub cc: [u8; 128],
    /// Paired 14-bit values of controllers 0-31 (MSB) with 32-63 (LSB).
    pub cc14: [u16; 32],
    cc_ports: Box<[Option<MidiOrigin>; 128]>,
    cc14_ports: Box<[Option<MidiOrigin>; 32]>,
    /// 14-bit pitch bend, 8192 at rest.
    pub pitch_bend: u16,
    /// Channel pressure.
    pub pressure: u8,
    /// Polyphonic key pressure.
    pub poly_pressure: [u8; 128],
    /// NRPN values by parameter number.
    pub nrpn: IndexMap<u16, u16>,
    /// RPN values by parameter number.
    pub rpn: IndexMap<u16, u16>,
    /// Held notes by key, in note-on order of first press.
    pub held_notes: IndexMap<u8, HeldNote>,
    selectors: ParameterSelectors,
    parameter_family: Option<ParameterFamily>,
    nrpn_ports: IndexMap<u16, MidiOrigin>,
    rpn_ports: IndexMap<u16, MidiOrigin>,
    pitch_bend_port: Option<MidiOrigin>,
    pressure_port: Option<MidiOrigin>,
    poly_pressure_ports: Box<[Option<MidiOrigin>; 128]>,
}

impl Default for MidiChannelState {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MidiChannelState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MidiChannelState")
            .field("key", &self.key)
            .field("velocity", &self.velocity)
            .field("gate", &self.gate)
            .field("time", &self.time)
            .field("held_notes", &self.held_notes)
            .field("pitch_bend", &self.pitch_bend)
            .field("pressure", &self.pressure)
            .finish_non_exhaustive()
    }
}

impl MidiChannelState {
    /// A channel at rest.
    pub fn new() -> Self {
        MidiChannelState {
            key: 0,
            velocity: 0,
            gate: 0,
            time: 0.0,
            keys: [0; 128],
            cc: [0; 128],
            cc14: [0; 32],
            cc_ports: Box::new(std::array::from_fn(|_| None)),
            cc14_ports: Box::new(std::array::from_fn(|_| None)),
            pitch_bend: PITCH_BEND_CENTER,
            pressure: 0,
            poly_pressure: [0; 128],
            nrpn: IndexMap::new(),
            rpn: IndexMap::new(),
            held_notes: IndexMap::new(),
            selectors: ParameterSelectors::default(),
            parameter_family: None,
            nrpn_ports: IndexMap::new(),
            rpn_ports: IndexMap::new(),
            pitch_bend_port: None,
            pressure_port: None,
            poly_pressure_ports: Box::new(std::array::from_fn(|_| None)),
        }
    }

    /// `noteOn(key, velocity)`: a note-on stamped with `Date.now()`.
    pub fn note_on(&mut self, key: u8, velocity: u8) {
        self.note_on_with(key, velocity, None, None, &mut || SystemClock.now_ms());
    }

    /// `noteOn(key, velocity)` with `Date.now()` returning `time_ms`.
    pub fn note_on_at(&mut self, key: u8, velocity: u8, time_ms: f64) {
        self.note_on_with(key, velocity, None, None, &mut || time_ms);
    }

    /// `noteOn(key, velocity, sourceNote, origin)`: the time and order come from
    /// `source_note` when given, otherwise from `now` and the note-on counter.
    pub fn note_on_from(
        &mut self,
        key: u8,
        velocity: u8,
        source_note: Option<&HeldNote>,
        origin: Option<MidiOrigin>,
        time_ms: f64,
    ) {
        self.note_on_with(key, velocity, source_note, origin, &mut || time_ms);
    }

    fn note_on_with(
        &mut self,
        key: u8,
        velocity: u8,
        source_note: Option<&HeldNote>,
        origin: Option<MidiOrigin>,
        now: &mut dyn FnMut() -> f64,
    ) {
        self.key = key;
        self.velocity = velocity;
        self.gate = 1;
        self.time = match source_note {
            Some(note) => note.time,
            None => now(),
        };
        // Uint8Array store: an out-of-range index is ignored.
        if let Some(slot) = self.keys.get_mut(key as usize) {
            *slot = velocity;
        }
        let order = match source_note {
            Some(note) => note.order,
            None => next_note_order(),
        };
        // Repeated NoteOn for one port/channel/key retriggers that identity.
        self.held_notes.insert(
            key,
            HeldNote {
                key,
                velocity,
                time: self.time,
                order,
                origin,
            },
        );
    }

    /// `controlChange(controller, value)`: stores a 7-bit controller, updates
    /// its 14-bit pair from this channel's bytes, and runs the channel-mode and
    /// parameter-number controllers. Returns the completed Data Entry update.
    pub fn control_change(&mut self, controller: u8, value: u8) -> Option<ParameterChange> {
        if controller > 127 || value > 127 {
            return None;
        }
        self.cc[controller as usize] = value;
        if controller < 64 {
            let msb = (controller & 31) as usize;
            self.cc14[msb] = (u16::from(self.cc[msb]) << 7) | u16::from(self.cc[msb + 32]);
        }
        if controller == 121 {
            self.reset_controllers();
        } else if controller == 120 || controller == 123 {
            self.clear_notes();
        } else if (98..=101).contains(&controller) {
            let family = if controller < 100 {
                ParameterFamily::Nrpn
            } else {
                ParameterFamily::Rpn
            };
            self.parameter_family = Some(family);
            let index = if controller == 99 || controller == 101 {
                0
            } else {
                1
            };
            self.selectors.family_mut(family)[index] = Some(value);
        } else if matches!(controller, 6 | 38 | 96 | 97) {
            let family = self.parameter_family?;
            let [msb, lsb] = self.selectors.family(family);
            let (Some(msb), Some(lsb)) = (msb, lsb) else {
                return None;
            };
            if msb == 127 && lsb == 127 {
                return None;
            }
            let parameter = (u16::from(msb) << 7) | u16::from(lsb);
            let values = match family {
                ParameterFamily::Nrpn => &mut self.nrpn,
                ParameterFamily::Rpn => &mut self.rpn,
            };
            let previous = values.get(&parameter).copied().unwrap_or(0);
            let next = match controller {
                6 => u16::from(value) << 7,
                38 => (previous & 0x3f80) | u16::from(value),
                _ => {
                    let step: i32 = if controller == 96 { 1 } else { -1 };
                    (i32::from(previous) + step).clamp(0, 16383) as u16
                }
            };
            values.insert(parameter, next);
            return Some(ParameterChange {
                family,
                parameter,
                value: next,
                reset_channels: None,
            });
        }
        None
    }

    /// `resetControllers()`: RP-015 resets controllers without erasing notes or
    /// stored parameters.
    pub fn reset_controllers(&mut self) {
        for cc in 0..128u8 {
            if !is_retained_reset_controller(cc) {
                self.cc[cc as usize] = if cc == 11 || (98..=101).contains(&cc) {
                    127
                } else {
                    0
                };
            }
        }
        for cc in 0..32 {
            self.cc14[cc] = (u16::from(self.cc[cc]) << 7) | u16::from(self.cc[cc + 32]);
        }
        self.pitch_bend = PITCH_BEND_CENTER;
        self.pressure = 0;
        self.poly_pressure = [0; 128];
        self.parameter_family = None;
        self.selectors = ParameterSelectors::default();
    }

    /// `clearNotes()`: releases every note and its key pressure.
    pub fn clear_notes(&mut self) {
        self.gate = 0;
        self.keys = [0; 128];
        self.held_notes.clear();
        self.poly_pressure = [0; 128];
    }

    /// `noteOff(key)`: closes the gate and releases `key`, or every note when
    /// `key` is `None`. The last key and velocity are kept.
    pub fn note_off(&mut self, key: Option<u8>) {
        self.gate = 0;
        let Some(key) = key else {
            self.clear_notes();
            return;
        };
        if let Some(slot) = self.keys.get_mut(key as usize) {
            *slot = 0;
        }
        self.held_notes.shift_remove(&key);
        if let Some(slot) = self.poly_pressure.get_mut(key as usize) {
            *slot = 0;
        }
    }

    /// `reset()`: back to the initial state.
    pub fn reset(&mut self) {
        *self = MidiChannelState::new();
    }

    /// Origin of each 7-bit controller value (`_ccPorts`).
    pub fn cc_origins(&self) -> &[Option<MidiOrigin>; 128] {
        &self.cc_ports
    }

    /// Origin of each paired 14-bit controller value (`_cc14Ports`).
    pub fn cc14_origins(&self) -> &[Option<MidiOrigin>; 32] {
        &self.cc14_ports
    }

    /// Origin of the pitch bend value (`_pitchBendPort`).
    pub fn pitch_bend_origin(&self) -> Option<&MidiOrigin> {
        self.pitch_bend_port.as_ref()
    }

    /// Origin of the channel pressure value (`_pressurePort`).
    pub fn pressure_origin(&self) -> Option<&MidiOrigin> {
        self.pressure_port.as_ref()
    }

    /// Origin of each key pressure value (`_polyPressurePorts`).
    pub fn poly_pressure_origins(&self) -> &[Option<MidiOrigin>; 128] {
        &self.poly_pressure_ports
    }

    /// Origin of each NRPN value (`_nrpnPorts`).
    pub fn nrpn_origins(&self) -> &IndexMap<u16, MidiOrigin> {
        &self.nrpn_ports
    }

    /// Origin of each RPN value (`_rpnPorts`).
    pub fn rpn_origins(&self) -> &IndexMap<u16, MidiOrigin> {
        &self.rpn_ports
    }

    /// The parameter-number selector bytes (`_selectors`).
    pub fn parameter_selectors(&self) -> &ParameterSelectors {
        &self.selectors
    }

    /// The family the Data Entry controllers address (`_parameterFamily`).
    pub fn parameter_family(&self) -> Option<ParameterFamily> {
        self.parameter_family
    }

    fn parameter_values_mut(&mut self, family: ParameterFamily) -> &mut IndexMap<u16, u16> {
        match family {
            ParameterFamily::Nrpn => &mut self.nrpn,
            ParameterFamily::Rpn => &mut self.rpn,
        }
    }

    fn parameter_origins_mut(&mut self, family: ParameterFamily) -> &mut IndexMap<u16, MidiOrigin> {
        match family {
            ParameterFamily::Nrpn => &mut self.nrpn_ports,
            ParameterFamily::Rpn => &mut self.rpn_ports,
        }
    }
}

/// Identity of a Web MIDI input port as the host reports it (`{id, name}`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MidiPortRef<'a> {
    /// Unique port id.
    pub id: &'a str,
    /// Readable port name.
    pub name: &'a str,
}

impl<'a> MidiPortRef<'a> {
    /// A port reference.
    pub fn new(id: &'a str, name: &'a str) -> Self {
        MidiPortRef { id, name }
    }
}

/// Structured port identity and connection state (`getPorts()` entries and
/// `setPortInventory` input).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidiPortInfo {
    /// Unique port id.
    pub id: String,
    /// Readable port name.
    pub name: String,
    /// Whether the port is connected.
    pub connected: bool,
}

/// A registered port (`_ports` entries).
#[derive(Clone, Debug)]
pub struct MidiPortEntry {
    id: Arc<str>,
    /// Readable name, updated on reconnect.
    pub name: String,
    /// Whether the port is available.
    pub connected: bool,
    /// The port's isolated state.
    pub state: MidiState,
}

impl MidiPortEntry {
    /// Port id.
    pub fn id(&self) -> &str {
        &self.id
    }
}

/// One selector field of a compiled `midi()` descriptor (`name` or `id`), as
/// `getPortState` observes a JavaScript value: falsy, a non-empty string, or
/// another truthy value (which never matches a port).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectorKey<'a> {
    /// `undefined`, `null`, `''`, `0`, `false`, `NaN`.
    Falsy,
    /// A non-empty string.
    Str(&'a str),
    /// Any other truthy value (numbers, booleans, objects).
    OtherTruthy,
}

impl<'a> SelectorKey<'a> {
    fn truthy(self) -> bool {
        !matches!(self, SelectorKey::Falsy)
    }

    fn as_str(self) -> Option<&'a str> {
        match self {
            SelectorKey::Str(s) => Some(s),
            _ => None,
        }
    }
}

impl<'a> From<Option<&'a str>> for SelectorKey<'a> {
    fn from(value: Option<&'a str>) -> Self {
        match value {
            Some(s) if !s.is_empty() => SelectorKey::Str(s),
            _ => SelectorKey::Falsy,
        }
    }
}

/// The selector of `getPortState`: a compiled descriptor's `name` and `id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortSelector<'a> {
    /// Readable port name.
    pub name: SelectorKey<'a>,
    /// Exact port id.
    pub id: SelectorKey<'a>,
}

impl<'a> PortSelector<'a> {
    /// A selector from optional name and id strings.
    pub fn new(name: Option<&'a str>, id: Option<&'a str>) -> Self {
        PortSelector {
            name: name.into(),
            id: id.into(),
        }
    }

    /// The empty selector, which resolves to the aggregate.
    pub fn any() -> Self {
        PortSelector {
            name: SelectorKey::Falsy,
            id: SelectorKey::Falsy,
        }
    }
}

/// The newest held note of an MPE zone (`getZoneVoice` result): the note and
/// the channel it is held on.
#[derive(Clone, Copy, Debug)]
pub struct ZoneVoice<'a> {
    /// The held note.
    pub note: &'a HeldNote,
    /// The channel state the note is held on.
    pub channel: &'a MidiChannelState,
    /// The channel number (1-16).
    pub channel_number: u8,
}

/// Complete MIDI state for the 16 channels (`MidiState`).
#[derive(Clone)]
pub struct MidiState {
    /// Per-channel state; index 0 is channel 1.
    pub channels: Box<[MidiChannelState; MIDI_CHANNELS]>,
    /// MIDI clock pulses (24 PPQ) received.
    pub clock_count: u64,
    /// Note grid texture data: 128 keys x 16 channels, RGBA.
    pub note_grid: Box<[f32; NOTE_GRID_LEN]>,
    ports: Option<IndexMap<Arc<str>, MidiPortEntry>>,
    ports_by_name: Option<IndexMap<String, Option<Arc<str>>>>,
    port_inventory: Option<IndexMap<String, Option<String>>>,
    unscoped: Option<Box<MidiState>>,
    /// MPE zone member counts.
    pub mpe_zones: MpeZones,
    clock: Arc<dyn Clock>,
}

impl fmt::Debug for MidiState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MidiState")
            .field("clock_count", &self.clock_count)
            .field("mpe_zones", &self.mpe_zones)
            .field("ports", &self.get_ports())
            .finish_non_exhaustive()
    }
}

impl Default for MidiState {
    fn default() -> Self {
        Self::new()
    }
}

/// Which isolated state a message updates before the aggregate.
enum Source {
    Port(Arc<str>),
    Unscoped,
}

impl MidiState {
    /// `new MidiState()`: a state with a port registry, reading `Date.now()`
    /// from the system clock.
    pub fn new() -> Self {
        Self::with_clock(Arc::new(SystemClock), true)
    }

    /// `new MidiState({ portRegistry: false })`.
    pub fn without_port_registry() -> Self {
        Self::with_clock(Arc::new(SystemClock), false)
    }

    /// A state reading `Date.now()` from `clock`; `port_registry` as the
    /// reference's constructor option. Port states share the clock.
    pub fn with_clock(clock: Arc<dyn Clock>, port_registry: bool) -> Self {
        MidiState {
            channels: Box::new(std::array::from_fn(|_| MidiChannelState::new())),
            clock_count: 0,
            note_grid: Box::new([0.0; NOTE_GRID_LEN]),
            ports: port_registry.then(IndexMap::new),
            ports_by_name: port_registry.then(IndexMap::new),
            port_inventory: None,
            unscoped: port_registry.then(|| Box::new(MidiState::with_clock(clock.clone(), false))),
            mpe_zones: MpeZones::default(),
            clock,
        }
    }

    /// The clock note-ons read.
    pub fn clock(&self) -> &Arc<dyn Clock> {
        &self.clock
    }

    /// Whether this state keeps a port registry (the root state).
    pub fn has_port_registry(&self) -> bool {
        self.ports.is_some()
    }

    /// `registerPort(port)`: registers or reconnects one input port and
    /// returns its isolated state. `None` without a registry or for an empty id.
    pub fn register_port(&mut self, id: &str, name: &str) -> Option<&mut MidiState> {
        let key = self.register_port_key(id, name)?;
        self.ports
            .as_mut()
            .and_then(|ports| ports.get_mut(&*key))
            .map(|entry| &mut entry.state)
    }

    fn register_port_key(&mut self, id: &str, name: &str) -> Option<Arc<str>> {
        let ports = self.ports.as_mut()?;
        if id.is_empty() {
            return None;
        }
        let topology_changed;
        let key = match ports.get_mut(id) {
            None => {
                let key: Arc<str> = Arc::from(id);
                ports.insert(
                    key.clone(),
                    MidiPortEntry {
                        id: key.clone(),
                        name: name.to_string(),
                        connected: true,
                        state: MidiState::with_clock(self.clock.clone(), false),
                    },
                );
                topology_changed = true;
                key
            }
            Some(entry) => {
                topology_changed = entry.name != name || !entry.connected;
                entry.name = name.to_string();
                entry.connected = true;
                entry.id.clone()
            }
        };
        if topology_changed {
            self.rebuild_port_name_index();
        }
        Some(key)
    }

    /// `disconnectPort(id)`: marks a port unavailable, keeping its identity
    /// record, and clears every aggregate value and note that came from it.
    pub fn disconnect_port(&mut self, id: &str) {
        let Some(entry) = self.ports.as_mut().and_then(|ports| ports.get_mut(id)) else {
            return;
        };
        entry.connected = false;
        entry.state.reset();
        for channel in self.channels.iter_mut() {
            for cc in 0..128 {
                if channel.cc_ports[cc].as_ref().is_some_and(|o| o.is_port(id)) {
                    channel.cc[cc] = 0;
                    channel.cc_ports[cc] = None;
                }
            }
            for cc in 0..32 {
                if channel.cc14_ports[cc]
                    .as_ref()
                    .is_some_and(|o| o.is_port(id))
                {
                    channel.cc14[cc] = 0;
                    channel.cc14_ports[cc] = None;
                }
            }
        }
        let origin = MidiOrigin::port(id);
        for channel in self.channels.iter_mut() {
            clear_note_origin(channel, &origin);
            for family in [ParameterFamily::Nrpn, ParameterFamily::Rpn] {
                let stale: Vec<u16> = channel
                    .parameter_origins_mut(family)
                    .iter()
                    .filter(|(_, o)| o.is_port(id))
                    .map(|(parameter, _)| *parameter)
                    .collect();
                for parameter in stale {
                    channel
                        .parameter_values_mut(family)
                        .shift_remove(&parameter);
                    channel
                        .parameter_origins_mut(family)
                        .shift_remove(&parameter);
                }
            }
            if channel
                .pitch_bend_port
                .as_ref()
                .is_some_and(|o| o.is_port(id))
            {
                channel.pitch_bend = PITCH_BEND_CENTER;
                channel.pitch_bend_port = None;
            }
            if channel
                .pressure_port
                .as_ref()
                .is_some_and(|o| o.is_port(id))
            {
                channel.pressure = 0;
                channel.pressure_port = None;
            }
            for key in 0..128 {
                if channel.poly_pressure_ports[key]
                    .as_ref()
                    .is_some_and(|o| o.is_port(id))
                {
                    channel.poly_pressure[key] = 0;
                    channel.poly_pressure_ports[key] = None;
                }
            }
        }
        self.rebuild_port_name_index();
    }

    /// `getPortState(selector)`: the state a compiled `midi()` descriptor
    /// reads. An id is authoritative; a name-only selector must match exactly
    /// one port. An empty selector resolves to the aggregate (`self`).
    pub fn get_port_state(&self, selector: &PortSelector<'_>) -> Option<&MidiState> {
        if !selector.name.truthy() && !selector.id.truthy() {
            return Some(self);
        }
        if selector.id.truthy() {
            let entry = selector
                .id
                .as_str()
                .and_then(|id| self.ports.as_ref()?.get(id));
            return entry.filter(|e| e.connected).map(|e| &e.state);
        }
        if let Some(inventory) = &self.port_inventory {
            let id = selector
                .name
                .as_str()
                .and_then(|name| inventory.get(name).cloned().flatten());
            let entry = id.and_then(|id| self.ports.as_ref()?.get(id.as_str()));
            return entry.filter(|e| e.connected).map(|e| &e.state);
        }
        let id = selector
            .name
            .as_str()
            .and_then(|name| self.ports_by_name.as_ref()?.get(name).cloned().flatten())?;
        self.ports.as_ref()?.get(&*id).map(|e| &e.state)
    }

    /// `getPortState({id})`.
    pub fn port_state_by_id(&self, id: &str) -> Option<&MidiState> {
        self.get_port_state(&PortSelector::new(None, Some(id)))
    }

    /// `setPortInventory(ports)`: physical discovery, independent of which
    /// inputs could be opened. Disconnected and unnamed entries are skipped; a
    /// name seen with two ids becomes ambiguous.
    pub fn set_port_inventory(&mut self, ports: &[MidiPortInfo]) {
        let mut names: IndexMap<String, Option<String>> = IndexMap::new();
        for port in ports {
            if !port.connected || port.id.is_empty() || port.name.is_empty() {
                continue;
            }
            let value = match names.get(&port.name) {
                None => Some(port.id.clone()),
                Some(Some(previous)) if *previous == port.id => Some(port.id.clone()),
                Some(_) => None,
            };
            names.insert(port.name.clone(), value);
        }
        self.port_inventory = Some(names);
    }

    /// The physical inventory (`_portInventory`): name to id, `None` when
    /// ambiguous; `None` until [`MidiState::set_port_inventory`] is called.
    pub fn port_inventory(&self) -> Option<&IndexMap<String, Option<String>>> {
        self.port_inventory.as_ref()
    }

    /// Connected-name index (`_portsByName`): name to port id, `None` when
    /// ambiguous.
    pub fn ports_by_name(&self) -> Option<&IndexMap<String, Option<Arc<str>>>> {
        self.ports_by_name.as_ref()
    }

    fn rebuild_port_name_index(&mut self) {
        let (Some(ports), Some(by_name)) = (&self.ports, &mut self.ports_by_name) else {
            return;
        };
        by_name.clear();
        for entry in ports.values() {
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

    /// `getPorts()`: identity and connection state of every registered port.
    pub fn get_ports(&self) -> Vec<MidiPortInfo> {
        self.ports
            .iter()
            .flat_map(|ports| ports.values())
            .map(|entry| MidiPortInfo {
                id: entry.id.to_string(),
                name: entry.name.clone(),
                connected: entry.connected,
            })
            .collect()
    }

    /// The registered port entries, in registration order.
    pub fn port_entries(&self) -> impl Iterator<Item = &MidiPortEntry> {
        self.ports.iter().flat_map(|ports| ports.values())
    }

    /// The registered port entry with `id`.
    pub fn port_entry(&self, id: &str) -> Option<&MidiPortEntry> {
        self.ports.as_ref()?.get(id)
    }

    /// Mutable access to the registered port entry with `id`.
    pub fn port_entry_mut(&mut self, id: &str) -> Option<&mut MidiPortEntry> {
        self.ports.as_mut()?.get_mut(id)
    }

    /// The isolated state of messages without a port (`_unscopedState`).
    pub fn unscoped_state(&self) -> Option<&MidiState> {
        self.unscoped.as_deref()
    }

    /// Mutable access to the isolated state of messages without a port.
    pub fn unscoped_state_mut(&mut self) -> Option<&mut MidiState> {
        self.unscoped.as_deref_mut()
    }

    /// `getChannel(n)`: channel `n` (1-16); any other number falls back to
    /// channel 1.
    pub fn get_channel(&self, n: i64) -> &MidiChannelState {
        match usize::try_from(n) {
            Ok(n @ 1..=MIDI_CHANNELS) => &self.channels[n - 1],
            _ => &self.channels[0],
        }
    }

    /// Mutable `getChannel(n)`.
    pub fn get_channel_mut(&mut self, n: i64) -> &mut MidiChannelState {
        match usize::try_from(n) {
            Ok(n @ 1..=MIDI_CHANNELS) => &mut self.channels[n - 1],
            _ => &mut self.channels[0],
        }
    }

    /// `channel(n).noteOn(key, velocity)` with this state's clock.
    pub fn channel_note_on(&mut self, n: i64, key: u8, velocity: u8) {
        let clock = self.clock.clone();
        self.get_channel_mut(n)
            .note_on_with(key, velocity, None, None, &mut || clock.now_ms());
    }

    /// `getZoneVoice({zone, members})`: the newest physically held note of an
    /// MPE zone (0 = lower, 1 = upper). Each port resolves its own zone first;
    /// the aggregate itself is never consulted when a registry exists.
    pub fn get_zone_voice(&self, zone: u8, members: Option<u8>) -> Option<ZoneVoice<'_>> {
        if zone > 1 || members.is_some_and(|m| !(1..=15).contains(&m)) {
            return None;
        }
        let mut newest: Option<ZoneVoice<'_>> = None;
        if let Some(ports) = &self.ports {
            let scopes = self.unscoped.as_deref().into_iter().chain(
                ports
                    .values()
                    .filter(|port| port.connected)
                    .map(|port| &port.state),
            );
            for scope in scopes {
                let voice = scope.get_zone_voice(zone, members);
                if let Some(voice) = voice
                    && newest.is_none_or(|n| voice.note.order > n.note.order)
                {
                    newest = Some(voice);
                }
            }
            return newest;
        }
        let configured = if zone == 0 {
            self.mpe_zones.lower
        } else {
            self.mpe_zones.upper
        };
        let count = members.or(configured).unwrap_or(15);
        let (first, last) = if zone == 0 {
            (2u8, 1 + count)
        } else {
            (16 - count, 15u8)
        };
        for index in first..=last {
            let channel = &self.channels[index as usize - 1];
            for note in channel.held_notes.values() {
                if newest.is_none_or(|n| note.order > n.note.order) {
                    newest = Some(ZoneVoice {
                        note,
                        channel,
                        channel_number: index,
                    });
                }
            }
        }
        newest
    }

    /// `_configureMpeZone(master, count)`: applies an MPE configuration
    /// message; returns the channels whose zone ownership changed.
    fn configure_mpe_zone(&mut self, master: u8, count: u8) -> Vec<u8> {
        if (master != 1 && master != 16) || count > 15 {
            return Vec::new();
        }
        let previous = self.mpe_zones;
        let lower = master == 1;
        if lower {
            self.mpe_zones.lower = Some(count);
            self.mpe_zones.upper.get_or_insert(0);
        } else {
            self.mpe_zones.upper = Some(count);
            self.mpe_zones.lower.get_or_insert(0);
        }
        let other = if lower {
            &mut self.mpe_zones.upper
        } else {
            &mut self.mpe_zones.lower
        };
        let other_count = other.unwrap_or(0);
        if count > 0 && count + other_count > 14 {
            *other = Some(14u8.saturating_sub(count));
        }
        let zones = self.mpe_zones;
        let mut changed = Vec::new();
        for channel in 1..=16u8 {
            if zone_owner(previous, channel) == zone_owner(zones, channel) {
                continue;
            }
            changed.push(channel);
            let state = &mut self.channels[channel as usize - 1];
            let selectors = state.selectors;
            let family = state.parameter_family;
            let mut selector_bytes = [0u8; 4];
            selector_bytes.copy_from_slice(&state.cc[98..102]);
            state.clear_notes();
            state.reset_controllers();
            // MCM is not CC121: keep parameter selection, including the active
            // configuration transaction, so further Data Entry remains valid.
            state.selectors = selectors;
            state.parameter_family = family;
            state.cc[98..102].copy_from_slice(&selector_bytes);
            // Neutral timbre is an adapter default, not a mandated CC74 reset.
            state.cc[74] = 64;
        }
        changed
    }

    /// `handleMessage(data, port)`: processes one raw MIDI message; note-ons
    /// read this state's clock.
    pub fn handle_message(
        &mut self,
        data: &[u8],
        port: Option<MidiPortRef<'_>>,
    ) -> Option<ParameterChange> {
        let clock = self.clock.clone();
        self.handle_message_with(data, port, &mut || clock.now_ms())
    }

    /// `handleMessage(data, port)` with `Date.now()` returning `time_ms`; for
    /// hosts that queue messages with their arrival time.
    pub fn handle_message_at(
        &mut self,
        data: &[u8],
        port: Option<MidiPortRef<'_>>,
        time_ms: f64,
    ) -> Option<ParameterChange> {
        self.handle_message_with(data, port, &mut || time_ms)
    }

    fn handle_message_with(
        &mut self,
        data: &[u8],
        port: Option<MidiPortRef<'_>>,
        now: &mut dyn FnMut() -> f64,
    ) -> Option<ParameterChange> {
        if data.is_empty() {
            return None;
        }
        let source = if self.ports.is_some() {
            match port {
                Some(port) => Some(Source::Port(self.register_port_key(port.id, port.name)?)),
                None => Some(Source::Unscoped),
            }
        } else {
            None
        };
        let parameter_change = match &source {
            Some(source) => self
                .source_state_mut(source)
                .handle_message_with(data, None, now),
            None => None,
        };
        let status = data[0];
        if status == 0xf8 {
            self.clock_count += 1;
            return None;
        }
        let key = *data.get(1)?;
        let velocity = data.get(2).copied();
        let channel = (status & 0x0f) + 1;
        let message_type = status & 0xf0;
        if key > 127 {
            return None;
        }
        if message_type != 0xd0 && !velocity.is_some_and(|v| v <= 127) {
            return None;
        }
        let velocity = velocity.unwrap_or(0);
        let ch = channel as usize - 1;
        let origin = match (&source, port) {
            (Some(Source::Port(id)), _) => MidiOrigin::Port(id.clone()),
            (_, Some(port)) => MidiOrigin::port(port.id),
            _ => MidiOrigin::Unscoped,
        };
        let MidiState {
            channels,
            ports,
            unscoped,
            ..
        } = self;
        let source_state: Option<&MidiState> = match &source {
            Some(Source::Port(id)) => ports.as_ref().and_then(|p| p.get(id)).map(|e| &e.state),
            Some(Source::Unscoped) => unscoped.as_deref(),
            None => None,
        };
        let source_channel = source_state.map(|s| &s.channels[ch]);
        match message_type {
            0xe0 => {
                let state = &mut channels[ch];
                state.pitch_bend = u16::from(key) | (u16::from(velocity) << 7);
                state.pitch_bend_port = Some(origin);
                return None;
            }
            0xd0 => {
                let state = &mut channels[ch];
                state.pressure = key;
                state.pressure_port = Some(origin);
                return None;
            }
            0xa0 => {
                let state = &mut channels[ch];
                state.poly_pressure[key as usize] = velocity;
                state.poly_pressure_ports[key as usize] = Some(origin);
                return None;
            }
            0xb0 => {
                let (Some(source_state), Some(source_channel)) = (source_state, source_channel)
                else {
                    let mut change = channels[ch].control_change(key, velocity);
                    if let Some(change) = change.as_mut()
                        && change.family == ParameterFamily::Rpn
                        && change.parameter == 6
                        && key == 6
                    {
                        change.reset_channels = Some(self.configure_mpe_zone(channel, velocity));
                    }
                    return change;
                };
                let k = key as usize;
                {
                    let state = &mut channels[ch];
                    state.cc[k] = source_channel.cc[k];
                    state.cc_ports[k] = Some(origin.clone());
                    if key < 64 {
                        let msb = (key & 31) as usize;
                        state.cc14[msb] = source_channel.cc14[msb];
                        state.cc14_ports[msb] = Some(origin.clone());
                    }
                }
                if let Some(change) = &parameter_change {
                    let state = &mut channels[ch];
                    state
                        .parameter_values_mut(change.family)
                        .insert(change.parameter, change.value);
                    state
                        .parameter_origins_mut(change.family)
                        .insert(change.parameter, origin.clone());
                    for &index in change.reset_channels.iter().flatten() {
                        let i = index as usize - 1;
                        clear_note_origin(&mut channels[i], &origin);
                        copy_controller_reset(
                            &mut channels[i],
                            &source_state.channels[i],
                            &origin,
                            true,
                        );
                    }
                }
                if key == 120 || key == 123 {
                    let state = &mut channels[ch];
                    clear_note_origin(state, &origin);
                    for note in 0..128 {
                        if state.poly_pressure_ports[note].as_ref() == Some(&origin) {
                            state.poly_pressure[note] = 0;
                        }
                    }
                }
                if key == 121 {
                    copy_controller_reset(&mut channels[ch], source_channel, &origin, false);
                }
                return parameter_change;
            }
            _ => {}
        }
        if message_type == 0x90 && velocity > 0 {
            let source_note = source_channel.and_then(|c| c.held_notes.get(&key));
            channels[ch].note_on_with(key, velocity, source_note, Some(origin), now);
        } else if message_type == 0x80 || (message_type == 0x90 && velocity == 0) {
            let state = &mut channels[ch];
            if source_channel.is_none() {
                state.note_off(Some(key));
            } else {
                // Preserve legacy aggregate gate behavior without erasing a
                // same-key held note or pressure supplied by another port.
                state.gate = 0;
                if state
                    .held_notes
                    .get(&key)
                    .is_some_and(|note| note.origin.as_ref() == Some(&origin))
                {
                    state.keys[key as usize] = 0;
                    state.held_notes.shift_remove(&key);
                }
                if state.poly_pressure_ports[key as usize].as_ref() == Some(&origin) {
                    state.poly_pressure[key as usize] = 0;
                }
            }
        }
        None
    }

    fn source_state_mut(&mut self, source: &Source) -> &mut MidiState {
        match source {
            Source::Port(id) => {
                &mut self
                    .ports
                    .as_mut()
                    .and_then(|ports| ports.get_mut(&**id))
                    .expect("registered port")
                    .state
            }
            Source::Unscoped => self.unscoped.as_deref_mut().expect("unscoped state"),
        }
    }

    /// `updateNoteGrid()`: packs the 16 channels' keys into the 128x16 RGBA
    /// grid: R = velocity / 127, G = gate (0 or 1), B = A = 0.
    pub fn update_note_grid(&mut self) {
        for ch in 0..MIDI_CHANNELS {
            let keys = &self.channels[ch].keys;
            let row = ch * NOTE_GRID_WIDTH * 4;
            for (k, &v) in keys.iter().enumerate() {
                let offset = row + k * 4;
                self.note_grid[offset] = if v > 0 {
                    (f64::from(v) / 127.0) as f32
                } else {
                    0.0
                };
                self.note_grid[offset + 1] = if v > 0 { 1.0 } else { 0.0 };
            }
        }
    }

    /// `reset()`: resets every channel, the clock count, the grid, the MPE
    /// zones, and every isolated state. Port registrations are kept.
    pub fn reset(&mut self) {
        for channel in self.channels.iter_mut() {
            channel.reset();
        }
        self.clock_count = 0;
        self.note_grid.fill(0.0);
        self.mpe_zones = MpeZones::default();
        if let Some(unscoped) = self.unscoped.as_deref_mut() {
            unscoped.reset();
        }
        if let Some(ports) = self.ports.as_mut() {
            for entry in ports.values_mut() {
                entry.state.reset();
            }
        }
    }
}

/// The owner of a channel under a zone layout (`owner` in `_configureMpeZone`).
fn zone_owner(zones: MpeZones, channel: u8) -> Option<&'static str> {
    let lower = zones.lower.unwrap_or(0);
    let upper = zones.upper.unwrap_or(0);
    if lower > 0 && channel == 1 {
        return Some("lowerManager");
    }
    if upper > 0 && channel == 16 {
        return Some("upperManager");
    }
    if lower > 0 && channel >= 2 && channel <= lower + 1 {
        return Some("lower");
    }
    if upper > 0 && channel >= 16 - upper && channel <= 15 {
        return Some("upper");
    }
    None
}

/// `_clearNoteOrigin(channel, origin)`: releases the notes `origin` holds.
fn clear_note_origin(channel: &mut MidiChannelState, origin: &MidiOrigin) {
    let keys = &mut channel.keys;
    channel.held_notes.retain(|key, note| {
        if note.origin.as_ref() != Some(origin) {
            return true;
        }
        if let Some(slot) = keys.get_mut(*key as usize) {
            *slot = 0;
        }
        false
    });
    if !channel.held_notes.contains_key(&channel.key) {
        channel.gate = 0;
    }
}

/// `_copyControllerReset(channel, source, origin, resetTimbre)`: applies a
/// port's controller reset to the aggregate without touching values other
/// origins own.
fn copy_controller_reset(
    channel: &mut MidiChannelState,
    source: &MidiChannelState,
    origin: &MidiOrigin,
    reset_timbre: bool,
) {
    for cc in 0..128u8 {
        let c = cc as usize;
        if (is_retained_reset_controller(cc) && !(reset_timbre && cc == 74))
            || (channel.cc_ports[c].is_some() && channel.cc_ports[c].as_ref() != Some(origin))
        {
            continue;
        }
        channel.cc[c] = source.cc[c];
        channel.cc_ports[c] = Some(origin.clone());
    }
    for cc in 0..32 {
        if channel.cc14_ports[cc].is_some() && channel.cc14_ports[cc].as_ref() != Some(origin) {
            continue;
        }
        channel.cc14[cc] = source.cc14[cc];
        channel.cc14_ports[cc] = Some(origin.clone());
    }
    if channel.pitch_bend_port.is_none() || channel.pitch_bend_port.as_ref() == Some(origin) {
        channel.pitch_bend = PITCH_BEND_CENTER;
    }
    if channel.pressure_port.is_none() || channel.pressure_port.as_ref() == Some(origin) {
        channel.pressure = 0;
    }
    for key in 0..128 {
        if channel.poly_pressure_ports[key].as_ref() == Some(origin) {
            channel.poly_pressure[key] = 0;
        }
    }
}

#[cfg(test)]
mod tests;
