//! `nm-input-dump analyser`: replays AnalyserNode scenarios.
//!
//! Scenario: `{name, channels, sampleRate, samples: [interleaved float32],
//! options: {fftSize, smoothingTimeConstant, minDecibels, maxDecibels},
//! chunkFrames?, events: [{quantum, ops: [...]}]}`. An op is
//! `{op: "set", prop, value}` (an `AnalyserNode` attribute assignment; the
//! result records whether it threw) or `{op: "read", getters: [...]}` (getter
//! names in call order). Samples are written `chunkFrames` frames at a time
//! (default one render quantum), so partial quanta are exercised while every
//! event still happens at a render quantum boundary, as with the oracle's
//! `OfflineAudioContext.suspend()`.

use noisemaker_input::analyser::{AudioAnalyzer, RENDER_QUANTUM_FRAMES};
use serde_json::{Value, json};

use super::{array_field, f32_bits, field, num_field, read_num, str_field};

pub fn run(scenarios: &Value) -> Result<Value, String> {
    let scenarios = scenarios
        .as_array()
        .ok_or("expected an array of scenarios")?;
    scenarios
        .iter()
        .map(run_one)
        .collect::<Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn apply_option(analyser: &mut AudioAnalyzer, prop: &str, value: f64) -> Result<bool, String> {
    let result = match prop {
        "fftSize" => analyser.set_fft_size(value as u32).is_ok(),
        "smoothingTimeConstant" => analyser.set_smoothing_time_constant(value).is_ok(),
        "minDecibels" => analyser.set_min_decibels(value).is_ok(),
        "maxDecibels" => analyser.set_max_decibels(value).is_ok(),
        other => return Err(format!("unknown analyser attribute {other}")),
    };
    Ok(result)
}

fn run_one(scenario: &Value) -> Result<Value, String> {
    let name = str_field(scenario, "name")?;
    let channels = num_field(scenario, "channels")? as usize;
    let samples: Vec<f32> = array_field(scenario, "samples")?
        .iter()
        .map(|v| {
            read_num(v)
                .map(|x| x as f32)
                .ok_or("sample is not a number")
        })
        .collect::<Result<_, _>>()?;
    let chunk_frames = scenario
        .get("chunkFrames")
        .and_then(Value::as_u64)
        .map_or(RENDER_QUANTUM_FRAMES, |n| n as usize)
        .max(1);
    let options = field(scenario, "options")?;
    let mut analyser = AudioAnalyzer::new();
    // AnalyserNode constructor: fftSize, then SetMinMaxDecibels, then smoothing.
    analyser
        .set_fft_size(num_field(options, "fftSize")? as u32)
        .map_err(|e| e.to_string())?;
    analyser
        .set_decibel_range(
            num_field(options, "minDecibels")?,
            num_field(options, "maxDecibels")?,
        )
        .map_err(|e| e.to_string())?;
    analyser
        .set_smoothing_time_constant(num_field(options, "smoothingTimeConstant")?)
        .map_err(|e| e.to_string())?;

    let total_frames = samples.len() / channels;
    let mut written_frames = 0usize;
    let mut out = Vec::new();
    for event in array_field(scenario, "events")? {
        let quantum = num_field(event, "quantum")? as usize;
        let target = quantum * RENDER_QUANTUM_FRAMES;
        if target > total_frames {
            return Err(format!(
                "{name}: event at quantum {quantum} past the signal"
            ));
        }
        while written_frames < target {
            let frames = chunk_frames.min(target - written_frames);
            let start = written_frames * channels;
            analyser.write(&samples[start..start + frames * channels], channels);
            written_frames += frames;
        }
        let mut results = Vec::new();
        for op in array_field(event, "ops")? {
            match str_field(op, "op")? {
                "set" => {
                    let prop = str_field(op, "prop")?;
                    let ok = apply_option(&mut analyser, prop, num_field(op, "value")?)?;
                    results.push(json!({"op": "set", "prop": prop, "ok": ok}));
                }
                "read" => {
                    let mut read = serde_json::Map::new();
                    for getter in array_field(op, "getters")? {
                        let getter = getter.as_str().ok_or("getter is not a string")?;
                        let bins = analyser.frequency_bin_count();
                        let size = analyser.fft_size() as usize;
                        let value = match getter {
                            "byteFrequency" => {
                                let mut data = vec![0u8; bins];
                                analyser.get_byte_frequency_data(&mut data);
                                json!(data)
                            }
                            "floatFrequency" => {
                                let mut data = vec![0f32; bins];
                                analyser.get_float_frequency_data(&mut data);
                                f32_bits(&data)
                            }
                            "byteTimeDomain" => {
                                let mut data = vec![0u8; size];
                                analyser.get_byte_time_domain_data(&mut data);
                                json!(data)
                            }
                            "floatTimeDomain" => {
                                let mut data = vec![0f32; size];
                                analyser.get_float_time_domain_data(&mut data);
                                f32_bits(&data)
                            }
                            other => return Err(format!("unknown getter {other}")),
                        };
                        read.insert(getter.to_string(), value);
                    }
                    results.push(json!({"op": "read", "data": Value::Object(read)}));
                }
                other => return Err(format!("unknown analyser op {other}")),
            }
        }
        out.push(json!({"quantum": quantum, "frame": target, "results": results}));
    }
    Ok(json!({"name": name, "events": out}))
}
