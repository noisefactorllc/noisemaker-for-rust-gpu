//! GPU integration tests: small graphs rendered on the device, checking the
//! reference behaviors end to end (uniform packing, mip regeneration order,
//! persistent textures, compute output buffers, readback conversion).

use noisemaker_gpu::{GpuDevice, Graph, Renderer, RendererOptions, Value};

fn device() -> GpuDevice {
    GpuDevice::create(&Default::default()).expect("a GPU adapter")
}

/// A one-program graph that writes `global_o0`.
fn graph(passes: &str, programs: &str, textures: &str) -> Graph {
    let json = format!(
        r#"{{"id":"t","source":"","passes":{passes},"programs":{programs},"allocations":{{}},
            "textures":{textures},"renderSurface":"o0","mediaSteps":[]}}"#
    );
    Graph::from_json(&json).expect("valid test graph")
}

fn wgsl(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

fn render(graph: Graph, size: u32, frames: u32) -> (Renderer, Vec<u8>) {
    let device = device();
    let mut renderer =
        Renderer::new(&device, graph, size, size, RendererOptions::default()).unwrap();
    for _ in 0..frames {
        renderer.render(0.25).unwrap();
    }
    let pixels = renderer.read_output().unwrap();
    assert_eq!(
        renderer.device_errors(),
        0,
        "device errors: {:#?}",
        renderer.pipeline().backend.device_error_log
    );
    (renderer, pixels.data)
}

#[test]
fn slot_layout_uniforms_reach_the_shader() {
    let src = r#"
        struct Uniforms { data: array<vec4<f32>, 2> };
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(uniforms.data[0].x, uniforms.data[0].y, uniforms.data[1].z, uniforms.data[1].w);
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"prog","inputs":{},"outputs":{"color":"global_o0"},
             "uniforms":{"a":0.25,"b":0.5,"c":[0.75,1.0]}}]"#,
        &format!(
            r#"{{"prog":{{"wgsl":{},"uniformLayout":{{"a":{{"slot":0,"components":"x"}},
                "b":{{"slot":0,"components":"y"}},"c":{{"slot":1,"components":"zw"}}}}}}}}"#,
            wgsl(src)
        ),
        "{}",
    );
    let (_, px) = render(g, 4, 1);
    assert_eq!(&px[..4], &[64, 128, 191, 255]);
}

#[test]
fn byte_layout_rounds_integers_and_skips_padding() {
    let src = r#"
        struct Params { gain: f32, count: i32, _pad: f32, tint: vec3<f32>, }
        @group(0) @binding(0) var<uniform> u: Params;
        @fragment fn main() -> @location(0) vec4<f32> {
            return vec4<f32>(u.tint * u.gain, f32(u.count) / 4.0);
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"prog","inputs":{},"outputs":{"color":"global_o0"},
             "uniforms":{"gain":0.5,"count":2.6,"tint":[1.0,0.5,0.25]}}]"#,
        &format!(r#"{{"prog":{{"wgsl":{}}}}}"#, wgsl(src)),
        "{}",
    );
    let (_, px) = render(g, 4, 1);
    assert_eq!(&px[..4], &[128, 64, 32, 191]);
}

#[test]
fn single_value_bindings_and_defaults() {
    // `level` comes from the pass, `aspect` from the globals, `missing` defaults to zeros.
    let src = r#"
        @group(0) @binding(0) var<uniform> level: f32;
        @group(0) @binding(1) var<uniform> aspect: f32;
        @group(0) @binding(2) var<uniform> missing: vec4<f32>;
        @fragment fn main() -> @location(0) vec4<f32> {
            return vec4<f32>(level, aspect * 0.5, missing.x, 1.0 + missing.w);
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"prog","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{"level":0.25}}]"#,
        &format!(r#"{{"prog":{{"wgsl":{}}}}}"#, wgsl(src)),
        "{}",
    );
    let (_, px) = render(g, 4, 1);
    assert_eq!(&px[..4], &[64, 128, 0, 255]);
}

/// `true` when `errors` is the reference's RESAMPLE_WGSL cascade: the invalid
/// resample module first, and nothing from the effect programs themselves.
fn only_resample_errors(errors: &[String], programs: &[&str]) -> bool {
    !errors.is_empty()
        && errors[0].contains("label = 'resample'")
        && !errors
            .iter()
            .any(|e| programs.iter().any(|p| e.contains(&format!("'{p}'"))))
}

/// Render without asserting on device errors; returns the renderer, the output
/// and the device errors observed.
fn render_with_errors(graph: Graph, size: u32, frames: u32) -> (Renderer, Vec<u8>, Vec<String>) {
    let device = device();
    let mut renderer =
        Renderer::new(&device, graph, size, size, RendererOptions::default()).unwrap();
    for _ in 0..frames {
        renderer.render(0.25).unwrap();
    }
    let pixels = renderer.read_output().unwrap();
    renderer.device_errors();
    let errors = renderer.pipeline().backend.device_error_log.clone();
    (renderer, pixels.data, errors)
}

#[test]
fn mip_chains_follow_the_reference_resample_shader() {
    // Pass A writes a checkerboard into a mipmapped texture; pass B samples its
    // level 1. The reference's RESAMPLE_WGSL constructs `vec2u(f32, f32)`, which
    // WGSL (Tint and naga alike) rejects, so its mip pipelines are invalid: every
    // generateMipmaps() submission fails validation and level 1 stays zero.
    let checker = r#"
        @group(0) @binding(0) var<uniform> gain: f32;
        @fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            let c = f32((u32(pos.x) + u32(pos.y)) % 2u) * gain;
            return vec4<f32>(c, c, c, 1.0);
        }"#;
    let sample = r#"
        @group(0) @binding(0) var tex: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main() -> @location(0) vec4<f32> {
            return textureSampleLevel(tex, samp, vec2<f32>(0.5, 0.5), 1.0);
        }"#;
    let g = graph(
        r#"[{"id":"a","program":"checker","inputs":{},"outputs":{"color":"node_mip"},"uniforms":{"gain":1}},
            {"id":"b","program":"sample","inputs":{"tex":"node_mip"},"outputs":{"color":"global_o0"},"uniforms":{}}]"#,
        &format!(
            r#"{{"checker":{{"wgsl":{}}},"sample":{{"wgsl":{}}}}}"#,
            wgsl(checker),
            wgsl(sample)
        ),
        r#"{"node_mip":{"width":4,"height":4,"format":"rgba16f","mipmaps":true,
            "usage":["render","sample","copySrc","copyDst"]}}"#,
    );
    let (renderer, px, errors) = render_with_errors(g, 4, 2);
    assert_eq!(&px[..4], &[0, 0, 0, 0]);
    assert!(
        only_resample_errors(&errors, &["checker", "sample"]),
        "{errors:#?}"
    );
    let level0 = renderer
        .pipeline()
        .backend
        .textures
        .get("node_mip")
        .unwrap();
    assert_eq!(level0.mip_levels, Some(3));
}

#[test]
fn persistent_textures_and_the_reference_copy_paths() {
    let fill = r#"
        @group(0) @binding(0) var<uniform> color: vec4<f32>;
        @fragment fn main() -> @location(0) vec4<f32> { return color; }"#;
    let g = graph(
        r#"[{"id":"a","program":"fill","inputs":{},"outputs":{"color":"node_keep"},"uniforms":{"color":[0.25,0.5,0.75,1.0]}},
            {"id":"b","program":"fill","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{"color":[0.25,0.5,0.75,1.0]}}]"#,
        &format!(r#"{{"fill":{{"wgsl":{}}}}}"#, wgsl(fill)),
        r#"{"node_keep":{"width":"screen","height":"screen","format":"rgba16f","persistent":true,
            "usage":["render","sample","copySrc","copyDst"]}}"#,
    );
    let (mut renderer, px, errors) = render_with_errors(g, 4, 1);
    assert!(errors.is_empty(), "{errors:#?}");
    assert_eq!(&px[..4], &[64, 128, 191, 255]);
    {
        // An equal-size copy is a texture-to-texture copy.
        let backend = &mut renderer.pipeline_mut().backend;
        let mut spec = noisemaker_gpu::Object::new();
        spec.insert("width", Value::Number(4.0));
        spec.insert("height", Value::Number(4.0));
        spec.insert("format", Value::from("rgba16f"));
        backend.create_texture("copy", &spec).unwrap();
        backend.copy_texture("node_keep", "copy");
        let copy = backend.read_pixels("copy").unwrap();
        assert!(copy.data.chunks(4).all(|p| p == [64, 128, 191, 255]));
    }
    // A resize recreates the persistent texture at the new size; the resampling
    // copy runs the reference's invalid RESAMPLE_WGSL, so the contents are lost.
    renderer.resize(8, 8).unwrap();
    let kept = renderer.read_pixels("node_keep").unwrap();
    assert_eq!((kept.width, kept.height), (8, 8));
    assert!(kept.data.iter().all(|&b| b == 0));
    renderer.device_errors();
    let errors = &renderer.pipeline().backend.device_error_log;
    assert!(
        only_resample_errors(errors, &["fill", "copy"]),
        "{errors:#?}"
    );
}

#[test]
fn shaders_without_bindings_take_the_legacy_bind_group() {
    // A source with no `@binding` declarations at all uses createLegacyBindGroup,
    // which binds the merged uniforms against the empty auto layout: the
    // reference's own validation failure, reproduced.
    let fill = r#"@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"#;
    let g = graph(
        r#"[{"id":"a","program":"fill","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{}}]"#,
        &format!(r#"{{"fill":{{"wgsl":{}}}}}"#, wgsl(fill)),
        "{}",
    );
    let (_, px, errors) = render_with_errors(g, 4, 1);
    assert_eq!(&px[..4], &[0, 0, 0, 0]);
    assert!(
        errors.iter().any(|e| e.contains("Number of bindings")),
        "{errors:#?}"
    );
}

#[test]
fn a_missing_mrt_target_is_recorded_once() {
    // color1 names a texture the graph never declares: the pass draws the
    // targets it has and records ERR_MISSING_RENDER_TARGET once, not per frame.
    let src = r#"
        @group(0) @binding(0) var<uniform> level: f32;
        struct Out { @location(0) a: vec4<f32>, @location(1) b: vec4<f32>, }
        @fragment fn main() -> Out {
            var o: Out;
            o.a = vec4<f32>(level, 0.0, 0.0, 1.0);
            o.b = vec4<f32>(0.0, 1.0, 0.0, 1.0);
            return o;
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"mrt","inputs":{},"outputs":{"color":"global_o0","color1":"absent"},"uniforms":{"level":1.0}}]"#,
        &format!(r#"{{"mrt":{{"wgsl":{}}}}}"#, wgsl(src)),
        "{}",
    );
    let (renderer, px) = render(g, 4, 3);
    assert_eq!(&px[..4], &[255, 0, 0, 255]);
    let missing: Vec<String> = renderer
        .pipeline()
        .backend
        .diagnostics
        .records
        .iter()
        .filter(|r| r.get("code").as_str() == Some("ERR_MISSING_RENDER_TARGET"))
        .map(|r| r.to_json().unwrap())
        .collect();
    assert_eq!(
        missing,
        [
            r#"{"code":"ERR_MISSING_RENDER_TARGET","backend":"webgpu","stage":"render","kind":"mrt","pass":"p","output":"absent"}"#
        ]
    );
}

#[test]
fn compute_output_buffer_is_copied_to_the_output() {
    let src = r#"
        struct Params { width: f32, value: f32, _a: f32, _b: f32, }
        @group(0) @binding(0) var<storage, read_write> output_buffer: array<f32>;
        @group(0) @binding(1) var<uniform> params: Params;
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let w = u32(params.width);
            let i = (id.y * w + id.x) * 4u;
            output_buffer[i] = params.value;
            output_buffer[i + 1u] = 0.0;
            output_buffer[i + 2u] = f32(id.x) / 8.0;
            output_buffer[i + 3u] = 1.0;
        }"#;
    let g = graph(
        r#"[{"id":"c","program":"comp","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{"value":0.5}}]"#,
        &format!(r#"{{"comp":{{"wgsl":{}}}}}"#, wgsl(src)),
        "{}",
    );
    let (_, px) = render(g, 8, 1);
    assert_eq!(&px[..4], &[128, 0, 0, 255]);
    // x = 4 -> 0.5 in the blue channel.
    assert_eq!(&px[16..20], &[128, 0, 128, 255]);
}

#[test]
fn reference_json_nulls_in_uniforms_decode_as_nan() {
    let g = Graph::from_reference_json(
        r#"{"passes":[{"id":"p","program":"x","uniforms":{"t":null,"v":[1,null]}}],"programs":{}}"#,
    )
    .unwrap();
    let u = g.passes[0].get("uniforms").unwrap();
    assert!(u.get("t").as_f64().unwrap().is_nan());
    assert!(u.get("v").at(1).as_f64().unwrap().is_nan());
    assert_eq!(u.get("v").at(0), &Value::Number(1.0));
}

#[test]
fn texture_pooling_shares_storage_between_disjoint_lifetimes() {
    // A and C share physical slot phys_0 (disjoint lifetimes), B is alone. With
    // `texturePooling`, C aliases A's backend texture; without it every virtual
    // texture owns its record.
    let fill = r#"
        @group(0) @binding(0) var<uniform> color: vec4<f32>;
        @fragment fn main() -> @location(0) vec4<f32> { return color; }"#;
    let copy = r#"
        @group(0) @binding(0) var src: texture_2d<f32>;
        @group(0) @binding(1) var samp: sampler;
        @fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            return textureSample(src, samp, pos.xy / vec2<f32>(textureDimensions(src)));
        }"#;
    let json = format!(
        r#"{{"id":"t","source":"","renderSurface":"o0","mediaSteps":[],
            "passes":[
              {{"id":"a","program":"fill","inputs":{{}},"outputs":{{"color":"node_a"}},"uniforms":{{"color":[0.5,0.25,0,1]}}}},
              {{"id":"b","program":"copy","inputs":{{"src":"node_a"}},"outputs":{{"color":"node_b"}},"uniforms":{{}}}},
              {{"id":"c","program":"copy","inputs":{{"src":"node_b"}},"outputs":{{"color":"node_c"}},"uniforms":{{}}}},
              {{"id":"d","program":"copy","inputs":{{"src":"node_c"}},"outputs":{{"color":"global_o0"}},"uniforms":{{}}}}],
            "programs":{{"fill":{{"wgsl":{}}},"copy":{{"wgsl":{}}}}},
            "allocations":{{"node_a":"phys_0","node_b":"phys_1","node_c":"phys_0"}},
            "textures":{{
              "node_a":{{"width":"screen","height":"screen","format":"rgba16f","usage":["render","sample","copySrc","copyDst"]}},
              "node_b":{{"width":"screen","height":"screen","format":"rgba16f","usage":["render","sample","copySrc","copyDst"]}},
              "node_c":{{"width":"screen","height":"screen","format":"rgba16f","usage":["render","sample","copySrc","copyDst"]}}}}}}"#,
        wgsl(fill),
        wgsl(copy)
    );
    let device = device();
    for pooling in [false, true] {
        let graph = Graph::from_json(&json).unwrap();
        let mut renderer = Renderer::new(
            &device,
            graph,
            4,
            4,
            RendererOptions {
                texture_pooling: pooling,
                ..Default::default()
            },
        )
        .unwrap();
        renderer.render(0.25).unwrap();
        let out = renderer.read_output().unwrap();
        assert_eq!(&out.data[..4], &[128, 64, 0, 255]);
        assert_eq!(renderer.device_errors(), 0);
        let plan = renderer.pipeline().get_resource_plan();
        assert_eq!(plan.pooling, pooling);
        if pooling {
            assert_eq!(
                plan.shared_textures,
                vec![vec!["node_a".to_owned(), "node_c".to_owned()]]
            );
        } else {
            assert!(plan.shared_textures.is_empty());
            assert_eq!(plan.textures.len(), 3);
        }
    }
}

#[test]
fn render_cubemap_renders_each_face_basis() {
    // The forward column of `cubeBasis` (a mat3x3 single-value uniform, packed as
    // three vec4 columns) mapped from [-1, 1] to [0, 1].
    let src = r#"
        @group(0) @binding(0) var<uniform> cubeBasis: mat3x3<f32>;
        @fragment fn main() -> @location(0) vec4<f32> {
            return vec4<f32>(cubeBasis[2] * 0.5 + vec3<f32>(0.5), 1.0);
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"face","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{}}]"#,
        &format!(r#"{{"face":{{"wgsl":{}}}}}"#, wgsl(src)),
        "{}",
    );
    let device = device();
    let mut renderer = Renderer::new(&device, g, 4, 4, RendererOptions::default()).unwrap();
    let faces = renderer
        .pipeline_mut()
        .render_cubemap(2.0, "o0", 0.25)
        .unwrap();
    let firsts: Vec<[u8; 4]> = faces
        .iter()
        .map(|f| {
            assert_eq!((f.width, f.height), (2, 2));
            f.data[..4].try_into().unwrap()
        })
        .collect();
    assert_eq!(
        firsts,
        vec![
            [255, 128, 128, 255],
            [0, 128, 128, 255],
            [128, 255, 128, 255],
            [128, 0, 128, 255],
            [128, 128, 255, 255],
            [128, 128, 0, 255],
        ]
    );
    // The pipeline returns to its previous size.
    let out = renderer.read_output().unwrap();
    assert_eq!((out.width, out.height), (4, 4));
    assert_eq!(renderer.device_errors(), 0);
}
