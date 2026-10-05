//! `nm-input-dump math`: `Math.sin`, `Math.cos` and `Math.round` of every
//! input, as `noisemaker_input::jsmath` computes them.

use noisemaker_input::jsmath::{js_cos, js_round, js_sin};
use serde_json::{Value, json};

use super::{array_field, num, read_num, str_field};

pub fn run(scenarios: &Value, emit: &mut dyn FnMut(Value)) -> Result<(), String> {
    let scenarios = scenarios
        .as_array()
        .ok_or("expected an array of scenarios")?;
    for scenario in scenarios {
        let inputs = array_field(scenario, "inputs")?
            .iter()
            .map(|v| read_num(v).ok_or_else(|| format!("not a number: {v}")))
            .collect::<Result<Vec<f64>, String>>()?;
        emit(json!({
            "scenario": str_field(scenario, "name")?,
            "sin": inputs.iter().map(|&x| num(js_sin(x))).collect::<Vec<_>>(),
            "cos": inputs.iter().map(|&x| num(js_cos(x))).collect::<Vec<_>>(),
            "round": inputs.iter().map(|&x| num(js_round(x))).collect::<Vec<_>>(),
        }));
    }
    Ok(())
}
