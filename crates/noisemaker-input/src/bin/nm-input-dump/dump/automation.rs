//! `nm-input-dump automation`: builds the scenario's MIDI and audio states,
//! then evaluates every descriptor with `resolve_uniform_value` and computes
//! the audio input requirements of every pass set (format in
//! `tools/reference-input.mjs`).

use noisemaker_input::automation::{
    AutomationContext, ExternalState, PassAudioInfo, audio_input_requirements,
    resolve_uniform_value,
};
use noisemaker_input::js::JsValue;
use serde_json::{Value, json};

use super::{array_field, field, num, num_field, read_num, str_field};

pub fn run(scenarios: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    for scenario in scenarios
        .as_array()
        .ok_or("expected an array of scenarios")?
    {
        run_one(scenario, emit)?;
    }
    Ok(())
}

fn run_one(scenario: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    let name = str_field(scenario, "name")?;
    let midi = match scenario.get("midi").filter(|m| !m.is_null()) {
        None => None,
        Some(midi) => {
            let mut run = super::midi::MidiRun::new(midi);
            for step in array_field(midi, "steps")? {
                run.step(step).map_err(|e| format!("{name}: {e}"))?;
            }
            Some(run.state)
        }
    };
    let audio = match scenario.get("audio").filter(|a| !a.is_null()) {
        None => None,
        Some(audio) => {
            let mut state = super::audio::new_state(audio);
            for step in array_field(audio, "steps")? {
                super::audio::step(&mut state, step).map_err(|e| format!("{name}: {e}"))?;
            }
            Some(state)
        }
    };
    let external = ExternalState::new(midi.as_ref(), audio.as_ref());
    for (index, evaluation) in array_field(scenario, "evaluations")?.iter().enumerate() {
        let value = JsValue::from_json(field(evaluation, "value")?)?;
        let spec = match evaluation.get("spec") {
            None => JsValue::Undefined,
            Some(spec) => JsValue::from_json(spec)?,
        };
        let time = read_num(field(evaluation, "time")?).ok_or("time is not a number")?;
        let context = AutomationContext::at(num_field(evaluation, "wallTime")?);
        let result = match resolve_uniform_value(&value, time, &spec, external, context) {
            None => json!({"passthrough": true}),
            Some(resolved) => json!({"value": num(resolved)}),
        };
        emit(json!({"scenario": name, "evaluation": index, "result": result}));
    }
    let sets = scenario.get("requirements").and_then(Value::as_array);
    for (index, set) in sets.into_iter().flatten().enumerate() {
        let passes = array_field(set, "passes")?
            .iter()
            .map(|pass| {
                Ok((
                    JsValue::from_json(field(pass, "uniforms")?)?,
                    pass.get("audioTagged")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let result =
            audio_input_requirements(passes.iter().map(|(uniforms, audio_tagged)| PassAudioInfo {
                uniforms,
                audio_tagged: *audio_tagged,
            }));
        let result = json!({
            "needsLegacy": result.needs_legacy,
            "needsLegacyRaw": result.needs_legacy_raw,
            "selected": result.selected.iter().map(|r| json!({
                "id": r.id,
                "name": r.name,
                "channel": r.channel,
                "needsRaw": r.needs_raw,
            })).collect::<Vec<_>>(),
        });
        emit(json!({"scenario": name, "requirements": index, "result": result}));
    }
    Ok(())
}
