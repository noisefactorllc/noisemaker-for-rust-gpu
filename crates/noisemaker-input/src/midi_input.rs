//! MIDI input devices: a port of the reference's `MidiInputManager`
//! (`external-input.js`) over a pluggable [`MidiBackend`] instead of Web MIDI.
//!
//! The manager keeps the reference's observable behaviour: a structured
//! status (`idle`, `unsupported`, `enabled`, `connected`, `disconnected`,
//! `denied`, `error`, `disabled`) with its messages and device counts; a
//! physical port inventory that keeps disconnected ports visible and is mirrored
//! into [`MidiState::set_port_inventory`]; per-port isolation through
//! [`MidiState::register_port`] / [`MidiState::disconnect_port`]; failed opens
//! reported without failing the whole enable; and messages from a port that has
//! been disconnected, superseded or disabled never reaching the state.
//!
//! Web MIDI delivers messages and hot-plug events asynchronously; here the
//! backend queues messages with their arrival time (the reference stamps
//! note-ons with `Date.now()` on arrival) and the host calls
//! [`MidiInputManager::update`] (apply queued messages) and
//! [`MidiInputManager::refresh`] (poll hot-plug, the `statechange` handler).
//! With the `midir` feature, [`crate::host::midir::MidirBackend`] connects real
//! devices.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use indexmap::IndexMap;

use crate::clock::{Clock, SystemClock};
use crate::midi::{MidiPortInfo, MidiPortRef, MidiState};

/// A MIDI input as the backend reports it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MidiInputInfo {
    /// Stable port id.
    pub id: String,
    /// Readable port name.
    pub name: String,
}

/// Why MIDI access failed (`requestMIDIAccess` rejection).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MidiAccessError {
    /// No MIDI support on this host (`MIDI not supported`).
    Unsupported,
    /// Permission denied (`NotAllowedError`/`SecurityError`).
    Denied(String),
    /// Any other failure.
    Failed(String),
}

/// The structured status states of the reference manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MidiStatusState {
    /// Not enabled yet.
    Idle,
    /// No MIDI support.
    Unsupported,
    /// Access granted and every input opened.
    Enabled,
    /// A port was hot-plugged and opened.
    Connected,
    /// A port went away.
    Disconnected,
    /// Access denied.
    Denied,
    /// Access or a port open failed.
    Error,
    /// Disabled by the host.
    Disabled,
}

impl MidiStatusState {
    /// The reference's state string.
    pub fn as_str(self) -> &'static str {
        match self {
            MidiStatusState::Idle => "idle",
            MidiStatusState::Unsupported => "unsupported",
            MidiStatusState::Enabled => "enabled",
            MidiStatusState::Connected => "connected",
            MidiStatusState::Disconnected => "disconnected",
            MidiStatusState::Denied => "denied",
            MidiStatusState::Error => "error",
            MidiStatusState::Disabled => "disabled",
        }
    }
}

/// `getStatus()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MidiStatus {
    /// Status state.
    pub state: MidiStatusState,
    /// Human-readable message.
    pub message: String,
    /// Connected ports in the inventory.
    pub device_count: usize,
    /// The port the status is about.
    pub port: Option<MidiInputInfo>,
    /// The error text of a failed open.
    pub error: Option<String>,
}

/// One message queued by a backend.
#[derive(Clone, Debug)]
struct QueuedMessage {
    port: MidiInputInfo,
    token: u64,
    data: Vec<u8>,
    time_ms: f64,
}

/// Where a backend delivers the messages of one opened input.
#[derive(Clone)]
pub struct MidiSink {
    queue: Arc<Mutex<VecDeque<QueuedMessage>>>,
    port: MidiInputInfo,
    token: u64,
    clock: Arc<dyn Clock>,
}

impl fmt::Debug for MidiSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MidiSink")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

impl MidiSink {
    /// Queues one complete MIDI message, stamped with its arrival time.
    pub fn deliver(&self, data: &[u8]) {
        let message = QueuedMessage {
            port: self.port.clone(),
            token: self.token,
            data: data.to_vec(),
            time_ms: self.clock.now_ms(),
        };
        if let Ok(mut queue) = self.queue.lock() {
            queue.push_back(message);
        }
    }

    /// The input this sink belongs to.
    pub fn port(&self) -> &MidiInputInfo {
        &self.port
    }
}

/// A source of MIDI inputs (Web MIDI's `MIDIAccess`).
pub trait MidiBackend {
    /// An open input; dropping it closes the input.
    type Connection;
    /// `requestMIDIAccess()`.
    fn request_access(&mut self) -> Result<(), MidiAccessError>;
    /// The inputs present now (all connected).
    fn inputs(&mut self) -> Vec<MidiInputInfo>;
    /// `input.open()` with `onmidimessage` delivering to `sink`.
    fn connect(
        &mut self,
        input: &MidiInputInfo,
        sink: MidiSink,
    ) -> Result<Self::Connection, String>;
}

type StatusCallback = Box<dyn FnMut(&str, &MidiStatus) + Send>;
type PortsCallback = Box<dyn FnMut(&[MidiPortInfo]) + Send>;

/// `MidiInputManager`.
pub struct MidiInputManager<B: MidiBackend> {
    backend: B,
    clock: Arc<dyn Clock>,
    enabled: bool,
    status: MidiStatus,
    known: IndexMap<String, MidiInputInfo>,
    inventory: IndexMap<String, MidiPortInfo>,
    active: HashMap<String, (B::Connection, u64)>,
    tokens: HashMap<String, u64>,
    queue: Arc<Mutex<VecDeque<QueuedMessage>>>,
    on_status: Option<StatusCallback>,
    on_ports: Option<PortsCallback>,
}

impl<B: MidiBackend> fmt::Debug for MidiInputManager<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MidiInputManager")
            .field("enabled", &self.enabled)
            .field("status", &self.status)
            .field("ports", &self.get_ports())
            .finish_non_exhaustive()
    }
}

impl<B: MidiBackend> MidiInputManager<B> {
    /// A disabled manager over `backend`; messages are stamped with the
    /// system clock.
    pub fn new(backend: B) -> Self {
        Self::with_clock(backend, Arc::new(SystemClock))
    }

    /// A disabled manager stamping message arrival with `clock`.
    pub fn with_clock(backend: B, clock: Arc<dyn Clock>) -> Self {
        MidiInputManager {
            backend,
            clock,
            enabled: false,
            status: MidiStatus {
                state: MidiStatusState::Idle,
                message: String::new(),
                device_count: 0,
                port: None,
                error: None,
            },
            known: IndexMap::new(),
            inventory: IndexMap::new(),
            active: HashMap::new(),
            tokens: HashMap::new(),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            on_status: None,
            on_ports: None,
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

    /// `getStatus()`.
    pub fn get_status(&self) -> MidiStatus {
        self.status.clone()
    }

    /// `getPorts()`: every port encountered, disconnected ones included.
    pub fn get_ports(&self) -> Vec<MidiPortInfo> {
        self.inventory.values().cloned().collect()
    }

    /// `onStatusChange(callback)`: called with the message and the status.
    pub fn on_status_change(&mut self, callback: impl FnMut(&str, &MidiStatus) + Send + 'static) {
        self.on_status = Some(Box::new(callback));
    }

    /// `onPortsChange(callback)`: called with the inventory.
    pub fn on_ports_change(&mut self, callback: impl FnMut(&[MidiPortInfo]) + Send + 'static) {
        self.on_ports = Some(Box::new(callback));
    }

    fn connected_count(&self) -> usize {
        self.inventory.values().filter(|p| p.connected).count()
    }

    fn notify_status(
        &mut self,
        state: MidiStatusState,
        message: String,
        port: Option<MidiInputInfo>,
        error: Option<String>,
    ) {
        self.status = MidiStatus {
            state,
            message,
            device_count: match state {
                MidiStatusState::Unsupported
                | MidiStatusState::Denied
                | MidiStatusState::Disabled => 0,
                MidiStatusState::Error if port.is_none() => 0,
                _ => self.connected_count(),
            },
            port,
            error,
        };
        if let Some(callback) = self.on_status.as_mut() {
            callback(&self.status.message, &self.status);
        }
    }

    fn notify_ports(&mut self) {
        let ports = self.get_ports();
        if let Some(callback) = self.on_ports.as_mut() {
            callback(&ports);
        }
    }

    /// `enable()`: requests access, registers every present input and opens
    /// it. Returns whether MIDI access is enabled; a port that fails to open
    /// leaves an `error` status but does not fail the enable.
    pub fn enable(&mut self, midi: &mut MidiState) -> bool {
        if self.enabled {
            return true;
        }
        if let Err(error) = self.backend.request_access() {
            match error {
                MidiAccessError::Unsupported => self.notify_status(
                    MidiStatusState::Unsupported,
                    "MIDI not supported".into(),
                    None,
                    None,
                ),
                MidiAccessError::Denied(_) => self.notify_status(
                    MidiStatusState::Denied,
                    "MIDI access denied".into(),
                    None,
                    None,
                ),
                MidiAccessError::Failed(_) => self.notify_status(
                    MidiStatusState::Error,
                    "MIDI access failed".into(),
                    None,
                    None,
                ),
            }
            return false;
        }
        self.known.clear();
        self.inventory.clear();
        self.active.clear();
        self.queue.lock().map(|mut q| q.clear()).ok();
        let present = self.sync_inventory(midi);
        self.notify_ports();
        let mut open_failures = 0;
        for input in present {
            if !self.connect_input(&input, midi) {
                open_failures += 1;
            }
        }
        self.sync_inventory(midi);
        self.notify_ports();
        self.enabled = true;
        let count = self.connected_count();
        if open_failures == 0 {
            let plural = if count != 1 { "s" } else { "" };
            self.notify_status(
                MidiStatusState::Enabled,
                format!("MIDI enabled ({count} device{plural})"),
                None,
                None,
            );
        }
        true
    }

    /// `disable()`: closes every input and makes their states unavailable.
    pub fn disable(&mut self, midi: &mut MidiState) {
        let ids: Vec<String> = self.known.keys().cloned().collect();
        for id in &ids {
            self.invalidate(id);
            midi.disconnect_port(id);
            let name = self.known[id].name.clone();
            self.inventory.insert(
                id.clone(),
                MidiPortInfo {
                    id: id.clone(),
                    name,
                    connected: false,
                },
            );
        }
        if !ids.is_empty() || self.enabled {
            midi.set_port_inventory(&self.get_ports());
        }
        self.queue.lock().map(|mut q| q.clear()).ok();
        self.enabled = false;
        self.notify_ports();
        self.notify_status(
            MidiStatusState::Disabled,
            "MIDI disabled".into(),
            None,
            None,
        );
    }

    /// `toggle()`: returns the new enabled state.
    pub fn toggle(&mut self, midi: &mut MidiState) -> bool {
        if self.enabled {
            self.disable(midi);
            false
        } else {
            self.enable(midi)
        }
    }

    /// Ends an input's current operation and connection (`_invalidatePortOperation`).
    fn invalidate(&mut self, id: &str) {
        *self.tokens.entry(id.to_string()).or_insert(0) += 1;
        self.active.remove(id);
    }

    /// `_syncPortInventory()`: reconciles the inventory with the inputs
    /// present now; returns them.
    fn sync_inventory(&mut self, midi: &mut MidiState) -> Vec<MidiInputInfo> {
        let present = self.backend.inputs();
        let gone: Vec<(String, String)> = self
            .known
            .iter()
            .filter(|(id, input)| !present.iter().any(|p| p.id == **id && p == *input))
            .map(|(id, input)| (id.clone(), input.name.clone()))
            .collect();
        for (id, name) in gone {
            self.invalidate(&id);
            midi.disconnect_port(&id);
            self.inventory.insert(
                id.clone(),
                MidiPortInfo {
                    id,
                    name,
                    connected: false,
                },
            );
        }
        for input in &present {
            self.known.insert(input.id.clone(), input.clone());
            self.inventory.insert(
                input.id.clone(),
                MidiPortInfo {
                    id: input.id.clone(),
                    name: input.name.clone(),
                    connected: true,
                },
            );
        }
        midi.set_port_inventory(&self.get_ports());
        present
    }

    /// `_connectInput`/`_openInput`: (re)opens one input and registers its
    /// isolated state.
    fn connect_input(&mut self, input: &MidiInputInfo, midi: &mut MidiState) -> bool {
        self.active.remove(&input.id);
        midi.disconnect_port(&input.id);
        let token = {
            let token = self.tokens.entry(input.id.clone()).or_insert(0);
            *token += 1;
            *token
        };
        let sink = MidiSink {
            queue: self.queue.clone(),
            port: input.clone(),
            token,
            clock: self.clock.clone(),
        };
        match self.backend.connect(input, sink) {
            Ok(connection) => {
                self.sync_inventory(midi);
                if self.tokens.get(&input.id) != Some(&token) || !self.known.contains_key(&input.id)
                {
                    return false;
                }
                midi.register_port(&input.id, &input.name);
                self.active.insert(input.id.clone(), (connection, token));
                true
            }
            Err(error) => {
                self.sync_inventory(midi);
                midi.disconnect_port(&input.id);
                let message = format!("MIDI device failed to open: {}", input.name);
                self.notify_status(
                    MidiStatusState::Error,
                    message,
                    Some(input.clone()),
                    Some(error),
                );
                false
            }
        }
    }

    /// The `statechange` handler: polls the backend and connects new inputs
    /// and disconnects vanished ones, reporting each change.
    pub fn refresh(&mut self, midi: &mut MidiState) {
        if !self.enabled {
            return;
        }
        let before: Vec<MidiInputInfo> = self
            .known
            .values()
            .filter(|input| self.inventory.get(&input.id).is_some_and(|p| p.connected))
            .cloned()
            .collect();
        let present = self.sync_inventory(midi);
        for input in &before {
            if !present.contains(input) {
                let message = format!("MIDI disconnected: {}", input.name);
                self.notify_status(
                    MidiStatusState::Disconnected,
                    message,
                    Some(input.clone()),
                    None,
                );
                self.notify_ports();
            }
        }
        for input in &present {
            // Only changes raise statechange events: an input that was already
            // present (open, or failed to open) is left alone.
            if before.contains(input) {
                continue;
            }
            if self.connect_input(input, midi) {
                let message = format!("MIDI connected: {}", input.name);
                self.notify_status(
                    MidiStatusState::Connected,
                    message,
                    Some(input.clone()),
                    None,
                );
            }
            self.notify_ports();
        }
    }

    /// Applies the queued messages of currently open inputs to `midi`, each
    /// with its arrival time; messages of closed or superseded inputs are
    /// dropped (the reference's `isCurrent()` check).
    pub fn update(&mut self, midi: &mut MidiState) {
        let messages: Vec<QueuedMessage> = match self.queue.lock() {
            Ok(mut queue) => queue.drain(..).collect(),
            Err(_) => return,
        };
        if !self.enabled {
            return;
        }
        for message in messages {
            let current = self
                .active
                .get(&message.port.id)
                .is_some_and(|(_, token)| *token == message.token);
            if !current {
                continue;
            }
            midi.handle_message_at(
                &message.data,
                Some(MidiPortRef::new(&message.port.id, &message.port.name)),
                message.time_ms,
            );
        }
    }
}

#[cfg(test)]
mod tests;
