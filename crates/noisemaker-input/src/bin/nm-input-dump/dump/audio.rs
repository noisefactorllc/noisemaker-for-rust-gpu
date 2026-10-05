//! `nm-input-dump audio`: replays AudioState scenarios (format in
//! `tools/reference-input.mjs`) and prints, after every step, the operation's
//! return value and the full state snapshot.

use std::collections::HashMap;

use noisemaker_input::audio::{
    AudioDeviceInfo, AudioState, Band, ChannelValues, DeviceSelector, FrequencyDataSource,
};
use noisemaker_input::midi::SelectorKey;
use serde_json::{Value, json};

use super::{array_field, bytes, f32_bits, field, num, num_field, read_num, str_field};

pub fn run(scenarios: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    for scenario in scenarios
        .as_array()
        .ok_or("expected an array of scenarios")?
    {
        run_one(scenario, emit)?;
    }
    Ok(())
}

/// The reference scenarios' fake analyser: `frequencyBinCount` bins, filled
/// from `data` up to the shorter length.
struct FakeAnalyser {
    bins: usize,
    data: Vec<u8>,
}

impl FrequencyDataSource for FakeAnalyser {
    fn frequency_bin_count(&self) -> usize {
        self.bins
    }

    fn get_byte_frequency_data(&mut self, array: &mut [u8]) {
        let n = array.len().min(self.data.len());
        array[..n].copy_from_slice(&self.data[..n]);
    }
}

pub fn new_state(scenario: &Value) -> AudioState {
    if scenario
        .get("registry")
        .and_then(Value::as_bool)
        .unwrap_or(true)
    {
        AudioState::new()
    } else {
        AudioState::without_device_registry()
    }
}

fn target<'a>(
    root: &'a mut AudioState,
    target: Option<&Value>,
) -> Result<&'a mut AudioState, String> {
    let Some(target) = target.filter(|t| t.as_str() != Some("root")) else {
        return Ok(root);
    };
    let found = if let Some(channel) = target.get("default") {
        let channel = read_num(channel).ok_or("bad default channel")?;
        default_channel_mut(root, channel)
    } else {
        let id = str_field(target, "device")?;
        let channel = num_field(target, "channel")? as u32;
        device_channel_mut(root, id, channel)
    };
    found.ok_or_else(|| format!("no audio target {target}"))
}

// Channel states regardless of connection, as the reference's scenario
// targets address them (`_defaultChannels.get(n)`, `_devices.get(id).channels`).
fn default_channel_mut(root: &mut AudioState, channel: f64) -> Option<&mut AudioState> {
    root.default_channels_mut()?.get_mut(&(channel as u32))
}

fn device_channel_mut<'a>(
    root: &'a mut AudioState,
    id: &str,
    channel: u32,
) -> Option<&'a mut AudioState> {
    root.device_entry_mut(id)?.channels.get_mut(&channel)
}

fn state_name(root: &AudioState, state: Option<&AudioState>) -> Value {
    let Some(state) = state else {
        return Value::Null;
    };
    if std::ptr::eq(state, root) {
        return json!("root");
    }
    for (n, s) in root.default_channels().into_iter().flatten() {
        if std::ptr::eq(s, state) {
            return json!(format!("default:{n}"));
        }
    }
    for entry in root.device_entries() {
        for (n, s) in &entry.channels {
            if std::ptr::eq(s, state) {
                return json!(format!("device:{}:{n}", entry.id));
            }
        }
    }
    json!("<unknown>")
}

fn selector_key(value: Option<&Value>) -> SelectorKey<'_> {
    match value {
        Some(Value::String(s)) if !s.is_empty() => SelectorKey::Str(s),
        _ => SelectorKey::Falsy,
    }
}

pub fn step(root: &mut AudioState, step: &Value) -> Result<Value, String> {
    Ok(match str_field(step, "op")? {
        "registerDevice" => {
            let count = step.get("channelCount").and_then(Value::as_u64).map(|c| c as u32);
            json!(root.register_device(str_field(step, "id")?, str_field(step, "name")?, count))
        }
        "setChannelValues" => {
            let values = field(step, "values")?;
            let get = |key: &str| values.get(key).and_then(read_num);
            let values = ChannelValues {
                low: get("low"),
                mid: get("mid"),
                high: get("high"),
                vol: get("vol"),
                raw: get("raw"),
            };
            json!(root.set_channel_values(str_field(step, "id")?, num_field(step, "channel")? as u32, &values))
        }
        "setDeviceRawUnavailable" => {
            root.set_device_raw_unavailable(str_field(step, "id")?);
            Value::Null
        }
        "disconnectDevice" => {
            root.disconnect_device(str_field(step, "id")?);
            Value::Null
        }
        "setDeviceInventory" => {
            let devices = array_field(step, "devices")?
                .iter()
                .map(|d| {
                    Ok(AudioDeviceInfo {
                        id: str_field(d, "id")?.to_string(),
                        name: str_field(d, "name")?.to_string(),
                        connected: d.get("connected").and_then(Value::as_bool).unwrap_or(true),
                        channel_count: 1,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            root.set_device_inventory(&devices);
            Value::Null
        }
        "registerDefaultChannels" => json!(root.register_default_channels(num_field(step, "count")? as u32)),
        "disconnectDefaultInput" => {
            root.disconnect_default_input();
            Value::Null
        }
        "getDefaultChannelState" => {
            let found = root.get_default_channel_state(num_field(step, "channel")?);
            state_name(root, found)
        }
        "getDeviceChannelState" => {
            let selector = field(step, "selector")?;
            let channel = selector.get("channel").map(|c| read_num(c).unwrap_or(f64::NAN));
            let found = root.get_device_channel_state(&DeviceSelector {
                name: selector_key(selector.get("name")),
                id: selector_key(selector.get("id")),
                channel,
            });
            state_name(root, found)
        }
        "devices" => json!(
            root.get_devices()
                .iter()
                .map(|d| json!({"id": d.id, "name": d.name, "connected": d.connected, "channelCount": d.channel_count}))
                .collect::<Vec<_>>()
        ),
        op => {
            let state = target(root, step.get("target"))?;
            match op {
                "setBands" => {
                    state.set_bands(num_field(step, "low")?, num_field(step, "mid")?, num_field(step, "high")?);
                    Value::Null
                }
                "setRaw" => {
                    state.set_raw(num_field(step, "value")?);
                    Value::Null
                }
                "setRawUnavailable" => {
                    state.set_raw_unavailable();
                    Value::Null
                }
                "setField" => {
                    let value = field(step, "value")?;
                    match str_field(step, "field")? {
                        "rawReady" => state.raw_ready = value.as_bool().ok_or("rawReady is not a boolean")?,
                        name => {
                            let x = read_num(value).ok_or("field value is not a number")?;
                            match name {
                                "low" => state.low = x,
                                "mid" => state.mid = x,
                                "high" => state.high = x,
                                "vol" => state.vol = x,
                                "raw" => state.raw = x,
                                other => return Err(format!("unknown audio field {other}")),
                            }
                        }
                    }
                    Value::Null
                }
                "updateFromAnalyser" => {
                    let smoothing = step.get("smoothing").and_then(read_num).unwrap_or(5.0);
                    match step.get("analyser").filter(|a| !a.is_null()) {
                        None => state.update_from_analyser(None, smoothing),
                        Some(analyser) => {
                            let mut fake = FakeAnalyser {
                                bins: num_field(analyser, "bins")? as usize,
                                data: bytes(field(analyser, "data")?)?,
                            };
                            state.update_from_analyser(Some(&mut fake), smoothing);
                        }
                    }
                    Value::Null
                }
                "setSpectrum" => {
                    state.set_spectrum(&bytes(field(step, "data")?)?);
                    Value::Null
                }
                "setWaveform" => {
                    state.set_waveform(&bytes(field(step, "data")?)?);
                    Value::Null
                }
                "smooth" => {
                    let band = match str_field(step, "band")? {
                        "low" => Band::Low,
                        "mid" => Band::Mid,
                        "high" => Band::High,
                        other => return Err(format!("unknown band {other}")),
                    };
                    num(state.smooth(band, num_field(step, "value")?))
                }
                "setMaxBufferLength" => {
                    state.set_max_buffer_length(num_field(step, "value")?);
                    Value::Null
                }
                "resetAggregate" => {
                    state.reset_aggregate();
                    Value::Null
                }
                "reset" => {
                    state.reset();
                    Value::Null
                }
                other => return Err(format!("unknown audio op {other}")),
            }
        }
    })
}

fn run_one(scenario: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    let name = str_field(scenario, "name")?;
    let mut root = new_state(scenario);
    let mut cache = SnapshotCache::new();
    for (index, s) in array_field(scenario, "steps")?.iter().enumerate() {
        let ret = step(&mut root, s).map_err(|e| format!("{name} step {index}: {e}"))?;
        emit(
            json!({"scenario": name, "step": index, "ret": ret, "snapshot": snapshot(&root, &mut cache)}),
        );
    }
    Ok(())
}

/// Fields of one AudioState itself (not its registries).
fn own_snapshot(state: &AudioState) -> Value {
    let smoothing = state.smoothing_buffers();
    json!({
        "low": num(state.low),
        "mid": num(state.mid),
        "high": num(state.high),
        "vol": num(state.vol),
        "raw": num(state.raw),
        "rawReady": state.raw_ready,
        "fft": f32_bits(&state.fft),
        "spectrum": f32_bits(&state.spectrum),
        "waveform": f32_bits(&state.waveform),
        "smoothing": {
            "low": smoothing.low.iter().map(|&x| num(x)).collect::<Vec<_>>(),
            "mid": smoothing.mid.iter().map(|&x| num(x)).collect::<Vec<_>>(),
            "high": smoothing.high.iter().map(|&x| num(x)).collect::<Vec<_>>(),
        },
        "frequencyData": state.frequency_data(),
        "maxBufferLength": num(state.max_buffer_length()),
    })
}

/// Each state's own snapshot from the previous step, keyed `root`,
/// `default:<n>` and `device:<id>:<n>`.
pub type SnapshotCache = HashMap<String, Value>;

/// A state's own fields, or "unchanged" when they equal the previous step's.
fn own_entry(state: &AudioState, key: String, cache: &mut SnapshotCache) -> Value {
    let own = own_snapshot(state);
    let unchanged = cache.get(&key) == Some(&own);
    cache.insert(key, own.clone());
    if unchanged { json!("unchanged") } else { own }
}

/// The registries in full and every state's own fields (see
/// `tools/reference-input.mjs`).
pub fn snapshot(state: &AudioState, cache: &mut SnapshotCache) -> Value {
    let mut out = json!({ "root": own_entry(state, "root".into(), cache) });
    if !state.has_device_registry() {
        return out;
    }
    let devices: Vec<Value> = state
        .device_entries()
        .map(|e| {
            json!({
                "id": e.id,
                "name": e.name,
                "connected": e.connected,
                "channelCount": e.channel_count,
                "channels": e.channels.iter()
                    .map(|(n, s)| json!([n, own_entry(s, format!("device:{}:{n}", e.id), cache)]))
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    out["devices"] = json!(devices);
    out["devicesByName"] = json!(
        state
            .devices_by_name()
            .expect("name index")
            .iter()
            .map(|(n, id)| json!([n, id]))
            .collect::<Vec<_>>()
    );
    out["deviceInventory"] = match state.device_inventory() {
        Some(inventory) => json!(
            inventory
                .iter()
                .map(|(n, id)| json!([n, id]))
                .collect::<Vec<_>>()
        ),
        None => Value::Null,
    };
    let defaults: Vec<Value> = state
        .default_channels()
        .expect("default channels")
        .iter()
        .map(|(n, s)| json!([n, own_entry(s, format!("default:{n}"), cache)]))
        .collect();
    out["defaultChannels"] = json!(defaults);
    out["defaultConnected"] = json!(state.default_connected());
    out
}
