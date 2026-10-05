//! MidiInputManager tests ported from the reference's `test_external_input.js`
//! with a fake backend in place of Web MIDI.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use super::*;
use crate::clock::ManualClock;

#[derive(Default)]
struct FakeBackend {
    access: Option<MidiAccessError>,
    inputs: Vec<MidiInputInfo>,
    fail_open: HashSet<String>,
    sinks: Arc<Mutex<HashMap<String, MidiSink>>>,
}

impl MidiBackend for FakeBackend {
    type Connection = ();

    fn request_access(&mut self) -> Result<(), MidiAccessError> {
        match &self.access {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    fn inputs(&mut self) -> Vec<MidiInputInfo> {
        self.inputs.clone()
    }

    fn connect(&mut self, input: &MidiInputInfo, sink: MidiSink) -> Result<(), String> {
        if self.fail_open.contains(&input.id) {
            return Err("open failed".into());
        }
        self.sinks.lock().unwrap().insert(input.id.clone(), sink);
        Ok(())
    }
}

fn input(id: &str, name: &str) -> MidiInputInfo {
    MidiInputInfo {
        id: id.into(),
        name: name.into(),
    }
}

fn send(sinks: &Arc<Mutex<HashMap<String, MidiSink>>>, id: &str, data: &[u8]) {
    sinks.lock().unwrap()[id].clone().deliver(data);
}

fn manager(
    backend: FakeBackend,
) -> (
    MidiInputManager<FakeBackend>,
    Arc<Mutex<HashMap<String, MidiSink>>>,
) {
    let sinks = backend.sinks.clone();
    let clock = Arc::new(ManualClock::new(1_000.0));
    (MidiInputManager::with_clock(backend, clock), sinks)
}

#[test]
fn routes_messages_with_port_identity_and_reports_hot_plug() {
    let (mut manager, sinks) = manager(FakeBackend {
        inputs: vec![input("port-id", "Launch Control XL")],
        ..Default::default()
    });
    let statuses = Arc::new(Mutex::new(Vec::new()));
    let inventories = Arc::new(Mutex::new(Vec::new()));
    {
        let statuses = statuses.clone();
        manager.on_status_change(move |_, status| statuses.lock().unwrap().push(status.state));
        let inventories = inventories.clone();
        manager.on_ports_change(move |ports| inventories.lock().unwrap().push(ports.to_vec()));
    }
    let mut midi = MidiState::new();
    assert!(manager.enable(&mut midi));
    let status = manager.get_status();
    assert_eq!(status.state, MidiStatusState::Enabled);
    assert_eq!(status.message, "MIDI enabled (1 device)");
    assert_eq!(status.device_count, 1);
    assert!(statuses.lock().unwrap().contains(&MidiStatusState::Enabled));
    assert_eq!(
        manager.get_ports(),
        vec![MidiPortInfo {
            id: "port-id".into(),
            name: "Launch Control XL".into(),
            connected: true
        }]
    );

    send(&sinks, "port-id", &[0x90, 60, 100]);
    manager.update(&mut midi);
    assert_eq!(
        midi.port_state_by_id("port-id").unwrap().get_channel(1).key,
        60
    );
    assert_eq!(midi.get_channel(1).time, 1_000.0);

    manager.backend().inputs.clear();
    manager.refresh(&mut midi);
    let status = manager.get_status();
    assert_eq!(status.state, MidiStatusState::Disconnected);
    assert_eq!(status.port.unwrap().id, "port-id");
    assert!(midi.port_state_by_id("port-id").is_none());
    assert!(
        inventories
            .lock()
            .unwrap()
            .iter()
            .any(|ports| ports.first().is_some_and(|p| !p.connected))
    );
}

#[test]
fn access_failures_are_classified() {
    for (error, state) in [
        (
            MidiAccessError::Failed("adapter unavailable".into()),
            MidiStatusState::Error,
        ),
        (
            MidiAccessError::Denied("NotAllowedError".into()),
            MidiStatusState::Denied,
        ),
        (MidiAccessError::Unsupported, MidiStatusState::Unsupported),
    ] {
        let (mut manager, _) = manager(FakeBackend {
            access: Some(error),
            ..Default::default()
        });
        let mut midi = MidiState::new();
        assert!(!manager.enable(&mut midi));
        assert_eq!(manager.get_status().state, state);
        assert!(!manager.enabled());
    }
}

#[test]
fn a_rejected_port_open_is_an_error_without_failing_enable() {
    let (mut manager, _) = manager(FakeBackend {
        inputs: vec![input("broken-port", "Broken Controller")],
        fail_open: HashSet::from(["broken-port".to_string()]),
        ..Default::default()
    });
    let mut midi = MidiState::new();
    assert!(manager.enable(&mut midi));
    let status = manager.get_status();
    assert_eq!(status.state, MidiStatusState::Error);
    assert_eq!(
        status.message,
        "MIDI device failed to open: Broken Controller"
    );
    assert_eq!(status.error.as_deref(), Some("open failed"));
    assert!(midi.port_state_by_id("broken-port").is_none());
}

#[test]
fn failed_open_duplicates_stay_ambiguous() {
    let (mut manager, sinks) = manager(FakeBackend {
        inputs: vec![input("twin-a", "Twin"), input("twin-b", "Twin")],
        fail_open: HashSet::from(["twin-b".to_string()]),
        ..Default::default()
    });
    let mut midi = MidiState::new();
    assert!(manager.enable(&mut midi));
    send(&sinks, "twin-a", &[0xb0, 74, 127]);
    manager.update(&mut midi);
    assert!(
        midi.get_port_state(&crate::midi::PortSelector::new(Some("Twin"), None))
            .is_none()
    );
    assert_eq!(
        midi.port_state_by_id("twin-a").unwrap().get_channel(1).cc[74],
        127
    );
    assert!(midi.port_state_by_id("twin-b").is_none());
    assert_eq!(
        manager.get_ports().iter().filter(|p| p.connected).count(),
        2
    );
}

#[test]
fn disable_detaches_inputs_and_drops_stale_messages() {
    let (mut manager, sinks) = manager(FakeBackend {
        inputs: vec![input("added", "Added")],
        ..Default::default()
    });
    let mut midi = MidiState::new();
    assert!(manager.enable(&mut midi));
    send(&sinks, "added", &[0xd0, 127]);
    manager.disable(&mut midi);
    manager.update(&mut midi);
    assert!(midi.port_state_by_id("added").is_none());
    assert_eq!(midi.get_channel(1).pressure, 0);
    assert_eq!(manager.get_status().state, MidiStatusState::Disabled);
    assert_eq!(manager.get_status().device_count, 0);
    assert!(!manager.get_ports()[0].connected);
}

#[test]
fn a_reconnect_supersedes_the_previous_connection() {
    let (mut manager, sinks) = manager(FakeBackend::default());
    let mut midi = MidiState::new();
    assert!(manager.enable(&mut midi));
    manager.backend().inputs.push(input("reused", "Reused"));
    manager.refresh(&mut midi);
    assert_eq!(manager.get_status().state, MidiStatusState::Connected);
    let stale = sinks.lock().unwrap()["reused"].clone();
    manager.backend().inputs.clear();
    manager.refresh(&mut midi);
    manager.backend().inputs.push(input("reused", "Reused"));
    manager.refresh(&mut midi);
    send(&sinks, "reused", &[0xb0, 74, 90]);
    stale.deliver(&[0xb0, 74, 1]);
    manager.update(&mut midi);
    assert_eq!(
        midi.port_state_by_id("reused").unwrap().get_channel(1).cc[74],
        90
    );
}
