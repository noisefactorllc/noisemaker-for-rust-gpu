//! `nm-input-dump midi`: replays MidiState scenarios (format in
//! `tools/reference-input.mjs`) and prints, after every step, the operation's
//! return value, the full state snapshot and the sparse note grid.

use std::collections::HashMap;
use std::sync::Arc;

use noisemaker_input::clock::{Clock, ManualClock};
use noisemaker_input::midi::{
    MidiChannelState, MidiOrigin, MidiPortInfo, MidiPortRef, MidiState, ParameterChange,
    PortSelector, note_order_counter,
};
use serde_json::{Value, json};

use super::{array_field, bytes, field, num, num_field, read_num, str_field};

pub fn run(scenarios: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    for scenario in scenarios
        .as_array()
        .ok_or("expected an array of scenarios")?
    {
        run_one(scenario, emit)?;
    }
    Ok(())
}

/// The scenario's MIDI state, its clock, and the note-order baseline.
pub struct MidiRun {
    pub state: MidiState,
    pub clock: Arc<ManualClock>,
    pub baseline: u64,
}

impl MidiRun {
    pub fn new(scenario: &Value) -> Self {
        let clock = Arc::new(ManualClock::new(
            scenario.get("time").and_then(read_num).unwrap_or(0.0),
        ));
        let registry = scenario
            .get("registry")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        MidiRun {
            state: MidiState::with_clock(clock.clone(), registry),
            clock,
            baseline: note_order_counter(),
        }
    }

    /// Applies one step and returns its result.
    pub fn step(&mut self, step: &Value) -> Result<Value, String> {
        if let Some(time) = step.get("time").and_then(read_num) {
            self.clock.set(time);
        }
        let root = &mut self.state;
        Ok(match str_field(step, "op")? {
            "message" => {
                let data = bytes(field(step, "data")?)?;
                let port = step.get("port").filter(|p| !p.is_null());
                let change = match port {
                    Some(port) => {
                        let id = str_field(port, "id")?;
                        let name = str_field(port, "name")?;
                        root.handle_message(&data, Some(MidiPortRef::new(id, name)))
                    }
                    None => root.handle_message(&data, None),
                };
                parameter_change(change.as_ref(), self.baseline)
            }
            "register" => {
                let id = str_field(step, "id")?;
                let name = str_field(step, "name")?;
                match root.register_port(id, name) {
                    Some(_) => json!(format!("port:{id}")),
                    None => Value::Null,
                }
            }
            "disconnect" => {
                root.disconnect_port(str_field(step, "id")?);
                Value::Null
            }
            "inventory" => {
                let ports = array_field(step, "ports")?
                    .iter()
                    .map(|p| {
                        Ok(MidiPortInfo {
                            id: str_field(p, "id")?.to_string(),
                            name: str_field(p, "name")?.to_string(),
                            connected: p.get("connected").and_then(Value::as_bool).unwrap_or(true),
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                root.set_port_inventory(&ports);
                Value::Null
            }
            "reset" => {
                root.reset();
                Value::Null
            }
            "ports" => json!(
                root.get_ports()
                    .iter()
                    .map(|p| json!({"id": p.id, "name": p.name, "connected": p.connected}))
                    .collect::<Vec<_>>()
            ),
            "portState" => {
                let name = step.get("name").and_then(Value::as_str);
                let id = step.get("id").and_then(Value::as_str);
                let found = root.get_port_state(&PortSelector::new(name, id));
                state_name(root, found)
            }
            "zoneVoice" => {
                let target = target(root, step.get("target"))?;
                let zone = num_field(step, "zone")? as u8;
                let members = step.get("members").and_then(Value::as_u64).map(|m| m as u8);
                match target.get_zone_voice(zone, members) {
                    None => Value::Null,
                    Some(voice) => {
                        let owner = owner_of(root, voice.channel);
                        json!({
                            "key": voice.note.key,
                            "velocity": voice.note.velocity,
                            "time": num(voice.note.time),
                            "order": voice.note.order - self.baseline,
                            "origin": origin_name(voice.note.origin.as_ref()),
                            "channelNumber": voice.channel_number,
                            "channelOwner": owner,
                        })
                    }
                }
            }
            op => {
                let now = self.clock.now_ms();
                let channel_number = num_field(step, "channel")? as i64;
                let target = target_mut(root, step.get("target"))?;
                let channel = target.get_channel_mut(channel_number);
                match op {
                    "noteOn" => {
                        channel.note_on_at(
                            num_field(step, "key")? as u8,
                            num_field(step, "velocity")? as u8,
                            now,
                        );
                        Value::Null
                    }
                    "noteOff" => {
                        channel.note_off(step.get("key").and_then(Value::as_u64).map(|k| k as u8));
                        Value::Null
                    }
                    "controlChange" => {
                        let change = channel.control_change(
                            num_field(step, "controller")? as u8,
                            num_field(step, "value")? as u8,
                        );
                        parameter_change(change.as_ref(), self.baseline)
                    }
                    "resetControllers" => {
                        channel.reset_controllers();
                        Value::Null
                    }
                    "clearNotes" => {
                        channel.clear_notes();
                        Value::Null
                    }
                    "channelReset" => {
                        channel.reset();
                        Value::Null
                    }
                    "setField" => {
                        let value = num_field(step, "value")?;
                        match str_field(step, "field")? {
                            "key" => channel.key = value as u8,
                            "velocity" => channel.velocity = value as u8,
                            "gate" => channel.gate = value as u8,
                            "time" => channel.time = value,
                            other => return Err(format!("unknown channel field {other}")),
                        }
                        Value::Null
                    }
                    other => return Err(format!("unknown MIDI op {other}")),
                }
            }
        })
    }
}

fn run_one(scenario: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    let name = str_field(scenario, "name")?;
    let mut run = MidiRun::new(scenario);
    let mut cache = SnapshotCache::new();
    for (index, step) in array_field(scenario, "steps")?.iter().enumerate() {
        let ret = run
            .step(step)
            .map_err(|e| format!("{name} step {index}: {e}"))?;
        run.state.update_note_grid();
        emit(json!({
            "scenario": name,
            "step": index,
            "ret": ret,
            "snapshot": snapshot(&run.state, run.baseline, "root", &mut cache),
            "grid": sparse_grid(&run.state.note_grid[..]),
        }));
    }
    Ok(())
}

fn target<'a>(root: &'a MidiState, target: Option<&Value>) -> Result<&'a MidiState, String> {
    match target {
        None => Ok(root),
        Some(Value::String(s)) if s == "root" => Ok(root),
        Some(Value::String(s)) if s == "unscoped" => {
            root.unscoped_state().ok_or("no unscoped state".into())
        }
        Some(t) => {
            let id = str_field(t, "port")?;
            root.port_entry(id)
                .map(|e| &e.state)
                .ok_or(format!("no port {id}"))
        }
    }
}

fn target_mut<'a>(
    root: &'a mut MidiState,
    target: Option<&Value>,
) -> Result<&'a mut MidiState, String> {
    match target {
        None => Ok(root),
        Some(Value::String(s)) if s == "root" => Ok(root),
        Some(Value::String(s)) if s == "unscoped" => {
            root.unscoped_state_mut().ok_or("no unscoped state".into())
        }
        Some(t) => {
            let id = str_field(t, "port")?;
            root.port_entry_mut(id)
                .map(|e| &mut e.state)
                .ok_or(format!("no port {id}"))
        }
    }
}

fn state_name(root: &MidiState, state: Option<&MidiState>) -> Value {
    let Some(state) = state else {
        return Value::Null;
    };
    if std::ptr::eq(state, root) {
        return json!("root");
    }
    if root
        .unscoped_state()
        .is_some_and(|u| std::ptr::eq(u, state))
    {
        return json!("unscoped");
    }
    for entry in root.port_entries() {
        if std::ptr::eq(&entry.state, state) {
            return json!(format!("port:{}", entry.id()));
        }
    }
    json!("<unknown>")
}

fn owner_of(root: &MidiState, channel: &MidiChannelState) -> Value {
    let holds = |state: &MidiState| state.channels.iter().any(|c| std::ptr::eq(c, channel));
    if holds(root) {
        return json!("root");
    }
    if root.unscoped_state().is_some_and(holds) {
        return json!("unscoped");
    }
    for entry in root.port_entries() {
        if holds(&entry.state) {
            return json!(format!("port:{}", entry.id()));
        }
    }
    Value::Null
}

pub fn origin_name(origin: Option<&MidiOrigin>) -> Value {
    match origin {
        None => Value::Null,
        Some(MidiOrigin::Unscoped) => json!("<unscoped>"),
        Some(MidiOrigin::Port(id)) => json!(format!("port:{id}")),
    }
}

fn parameter_change(change: Option<&ParameterChange>, _baseline: u64) -> Value {
    let Some(change) = change else {
        return Value::Null;
    };
    let mut out = json!({
        "family": change.family.as_str(),
        "parameter": change.parameter,
        "value": change.value,
    });
    if let Some(channels) = &change.reset_channels {
        out["resetChannels"] = json!(channels);
    }
    out
}

fn sparse<T: Copy + PartialEq + Default + Into<Value>>(values: &[T]) -> Value {
    Value::Array(
        values
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != T::default())
            .map(|(i, v)| Value::Array(vec![json!(i), (*v).into()]))
            .collect(),
    )
}

fn sparse_origins(origins: &[Option<MidiOrigin>]) -> Value {
    Value::Array(
        origins
            .iter()
            .enumerate()
            .filter(|(_, o)| o.is_some())
            .map(|(i, o)| json!([i, origin_name(o.as_ref())]))
            .collect(),
    )
}

fn sparse_grid(grid: &[f32]) -> Value {
    Value::Array(
        grid.iter()
            .enumerate()
            .filter(|(_, v)| v.to_bits() != 0)
            .map(|(i, v)| json!([i, v.to_bits()]))
            .collect(),
    )
}

fn channel_snapshot(ch: &MidiChannelState, baseline: u64) -> Value {
    let selectors = ch.parameter_selectors();
    json!({
        "key": ch.key,
        "velocity": ch.velocity,
        "gate": ch.gate,
        "time": num(ch.time),
        "keys": sparse(&ch.keys),
        "cc": sparse(&ch.cc),
        "cc14": sparse(&ch.cc14),
        "ccPorts": sparse_origins(ch.cc_origins()),
        "cc14Ports": sparse_origins(ch.cc14_origins()),
        "pitchBend": ch.pitch_bend,
        "pressure": ch.pressure,
        "polyPressure": sparse(&ch.poly_pressure),
        "polyPressurePorts": sparse_origins(ch.poly_pressure_origins()),
        "pitchBendPort": origin_name(ch.pitch_bend_origin()),
        "pressurePort": origin_name(ch.pressure_origin()),
        "nrpn": ch.nrpn.iter().map(|(p, v)| json!([p, v])).collect::<Vec<_>>(),
        "rpn": ch.rpn.iter().map(|(p, v)| json!([p, v])).collect::<Vec<_>>(),
        "nrpnPorts": ch.nrpn_origins().iter().map(|(p, o)| json!([p, origin_name(Some(o))])).collect::<Vec<_>>(),
        "rpnPorts": ch.rpn_origins().iter().map(|(p, o)| json!([p, origin_name(Some(o))])).collect::<Vec<_>>(),
        "heldNotes": ch.held_notes.iter().map(|(k, n)| json!([k, {
            "key": n.key,
            "velocity": n.velocity,
            "time": num(n.time),
            "order": n.order - baseline,
            "origin": origin_name(n.origin.as_ref()),
        }])).collect::<Vec<_>>(),
        "selectors": {"nrpn": selectors.nrpn, "rpn": selectors.rpn},
        "parameterFamily": ch.parameter_family().map(|f| f.as_str()),
    })
}

/// A channel exactly in its initial state is written as "default" (both
/// sides decide by comparing with a fresh channel's snapshot).
fn channel_entry(channel: &MidiChannelState, baseline: u64) -> Value {
    thread_local! {
        static DEFAULT: Value = channel_snapshot(&MidiChannelState::new(), 0);
    }
    let snapshot = channel_snapshot(channel, baseline);
    if DEFAULT.with(|default| *default == snapshot) {
        json!("default")
    } else {
        snapshot
    }
}

/// Each state's channel snapshots from the previous step, keyed `root`,
/// `unscoped` and `port:<id>`.
pub type SnapshotCache = HashMap<String, Vec<Value>>;

/// The state with the channels changed since the previous step (`cache`), as
/// `[channel, snapshot]` pairs; see `tools/reference-input.mjs`.
pub fn snapshot(state: &MidiState, baseline: u64, key: &str, cache: &mut SnapshotCache) -> Value {
    let current: Vec<Value> = state
        .channels
        .iter()
        .map(|c| channel_entry(c, baseline))
        .collect();
    let previous = cache.remove(key).unwrap_or_default();
    let changed: Vec<Value> = current
        .iter()
        .enumerate()
        .filter(|(i, entry)| previous.get(*i) != Some(*entry))
        .map(|(i, entry)| json!([i + 1, entry]))
        .collect();
    cache.insert(key.to_string(), current);
    let mut out = json!({
        "clockCount": state.clock_count,
        "mpeZones": {"lower": state.mpe_zones.lower, "upper": state.mpe_zones.upper},
        "changed": changed,
    });
    if !state.has_port_registry() {
        return out;
    }
    let ports: Vec<Value> = state
        .port_entries()
        .map(|e| {
            json!({
                "id": e.id(),
                "name": e.name,
                "connected": e.connected,
                "state": snapshot(&e.state, baseline, &format!("port:{}", e.id()), cache),
            })
        })
        .collect();
    out["ports"] = json!(ports);
    out["unscoped"] = snapshot(
        state.unscoped_state().expect("unscoped state"),
        baseline,
        "unscoped",
        cache,
    );
    out["portInventory"] = match state.port_inventory() {
        Some(inventory) => json!(
            inventory
                .iter()
                .map(|(n, id)| json!([n, id]))
                .collect::<Vec<_>>()
        ),
        None => Value::Null,
    };
    out["portsByName"] = json!(
        state
            .ports_by_name()
            .expect("name index")
            .iter()
            .map(|(n, id)| json!([n, id.as_deref()]))
            .collect::<Vec<_>>()
    );
    out
}
