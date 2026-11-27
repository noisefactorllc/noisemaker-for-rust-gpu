//! Graph loading in the reference shape (`compileGraph` output as
//! `parity/batch-golden.mjs` serializes it).

use noisemaker_gpu::graph::pass;
use noisemaker_gpu::{Graph, Value};

#[test]
fn stable_sync_blend_color_aliases_preserve_zero_in_both_slots() {
    use noisemaker_gpu::backend::WebGpuBackend;
    use wgpu::BlendFactor as B;
    for (name, expected) in [
        ("SRC_COLOR", B::Src),
        ("DST_COLOR", B::Dst),
        ("ONE_MINUS_SRC_COLOR", B::OneMinusSrc),
        ("ONE_MINUS_DST_COLOR", B::OneMinusDst),
        ("ZERO", B::Zero),
    ] {
        for spelling in [name.to_owned(), name.to_lowercase().replace('_', "-")] {
            for slot in 0..2 {
                let mut pair = vec![Value::from("ZERO"), Value::from("ZERO")];
                pair[slot] = Value::from(spelling.as_str());
                let blend = WebGpuBackend::resolve_blend_state(&Value::Array(pair))
                    .unwrap()
                    .unwrap();
                let factors = [blend.color.src_factor, blend.color.dst_factor];
                assert_eq!(factors[slot], expected);
                assert_eq!(factors[1 - slot], B::Zero);
                assert_eq!(blend.alpha, blend.color);
            }
        }
    }
}

const GRAPH: &str = r#"{
  "id": "-ey86km",
  "source": "noise().write(o0)\nrender(o0)\n",
  "passes": [
    {"id": "node_0_pass_0", "program": "noise", "inputs": {}, "outputs": {"fragColor": "global_o0"},
     "uniforms": {"seed": 1, "speed": null}, "drawMode": "points", "customField": [1, 2]}
  ],
  "programs": {"noise": {"wgsl": "@fragment fn main() {}", "defines": {"TYPE": 10}}},
  "allocations": [["node_0_out", "phys_0"]],
  "textures": {"node_0_out": {"width": "screen", "height": 64, "format": "rgba16f"}},
  "renderSurface": "o0",
  "mediaSteps": [],
  "futureMember": {"kept": true}
}"#;

#[test]
fn reference_shape_loads_with_unknown_fields_kept() {
    let graph = Graph::from_json(GRAPH).unwrap();
    assert_eq!(graph.id, Value::from("-ey86km"));
    assert_eq!(graph.render_surface_name(), Some("o0"));
    assert_eq!(graph.passes.len(), 1);
    let p = &graph.passes[0];
    assert_eq!(pass::id(p), "node_0_pass_0");
    assert_eq!(pass::program(p), "noise");
    assert_eq!(pass::draw_mode(p), Some("points"));
    // Pass members the runtime does not name stay reachable.
    assert_eq!(pass::get(p, "customField").at(1), &Value::Number(2.0));
    assert_eq!(pass::get(p, "absent"), &Value::Undefined);
    // A `Map` serialized as entry pairs loads like an object.
    let allocations = graph.allocations.as_ref().unwrap();
    assert_eq!(allocations.get("node_0_out"), Some(&Value::from("phys_0")));
    let spec = &graph.textures.as_ref().unwrap()["node_0_out"];
    assert_eq!(spec.get_or_undefined("width"), &Value::from("screen"));
    assert_eq!(
        graph.programs["noise"].get("defines").get("TYPE"),
        &Value::Number(10.0)
    );
    // Unknown graph members are kept and written back.
    assert_eq!(
        graph.extra.get_or_undefined("futureMember").get("kept"),
        &Value::Bool(true)
    );
    let back = graph.to_value();
    assert_eq!(back.get("futureMember").get("kept"), &Value::Bool(true));
    assert_eq!(
        back.get("allocations").get("node_0_out"),
        &Value::from("phys_0")
    );
}

#[test]
fn plain_json_keeps_null_but_reference_json_restores_nan() {
    // `JSON.stringify` wrote the page's NaN uniforms as null.
    let plain = Graph::from_json(GRAPH).unwrap();
    assert_eq!(
        pass::uniforms(&plain.passes[0])
            .unwrap()
            .get_or_undefined("speed"),
        &Value::Null
    );
    let reference = Graph::from_reference_json(GRAPH).unwrap();
    let speed = pass::uniforms(&reference.passes[0])
        .unwrap()
        .get_or_undefined("speed");
    assert!(speed.as_f64().unwrap().is_nan());
}
