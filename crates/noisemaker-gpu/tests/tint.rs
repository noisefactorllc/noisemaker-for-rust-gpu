//! GPU tests of the Tint shader compiler (`backend/shaders.rs`): on Metal the
//! default device compiles with Tint, pipelines on Tint's MSL keep the WGSL
//! semantics, read storage-buffer sizes from their immediate block, and
//! never fall back to naga for valid shaders.

use noisemaker_gpu::{DeviceOptions, GpuDevice, Graph, Renderer, RendererOptions, ShaderCompiler};

/// A Metal device compiling with Tint, or `None` off Metal.
fn tint_device() -> Option<GpuDevice> {
    let options = DeviceOptions {
        shader_compiler: Some(ShaderCompiler::Tint),
        ..Default::default()
    };
    match GpuDevice::create(&options) {
        Ok(device) => Some(device),
        Err(e) => {
            let fallback = GpuDevice::create(&Default::default()).expect("a GPU adapter");
            assert!(!fallback.is_metal(), "Tint is unavailable on Metal: {e}");
            None
        }
    }
}

fn graph(passes: &str, programs: &str) -> Graph {
    let json = format!(
        r#"{{"id":"t","source":"","passes":{passes},"programs":{programs},"allocations":{{}},
            "textures":{{}},"renderSurface":"o0","mediaSteps":[]}}"#
    );
    Graph::from_json(&json).expect("valid test graph")
}

fn wgsl(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

/// Render one frame of `graph` and read the output; no device error, no
/// fallback to naga.
fn render(device: &GpuDevice, graph: Graph, size: u32) -> Vec<u8> {
    let mut renderer =
        Renderer::new(device, graph, size, size, RendererOptions::default()).unwrap();
    renderer.render(0.25).unwrap();
    let pixels = renderer.read_output().unwrap();
    let backend = &renderer.pipeline().backend;
    assert_eq!(backend.shader_compiler(), ShaderCompiler::Tint);
    assert!(
        backend.tint_fallback_log().is_empty(),
        "{:#?}",
        backend.tint_fallback_log()
    );
    assert_eq!(
        renderer.device_errors(),
        0,
        "device errors: {:#?}",
        renderer.pipeline().backend.device_error_log
    );
    pixels.data
}

#[test]
fn metal_devices_compile_with_tint_unless_opted_out() {
    let device = GpuDevice::create(&Default::default()).expect("a GPU adapter");
    let expected = match ShaderCompiler::from_env().unwrap() {
        Some(choice) => choice,
        None if device.is_metal() => ShaderCompiler::Tint,
        None => ShaderCompiler::Naga,
    };
    assert_eq!(device.shader_compiler, expected);
    assert_eq!(device.backend().shader_compiler(), expected);
    let naga = GpuDevice::create(&DeviceOptions {
        shader_compiler: Some(ShaderCompiler::Naga),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(naga.backend().shader_compiler(), ShaderCompiler::Naga);
}

#[test]
fn loops_and_uniforms_keep_their_semantics() {
    let Some(device) = tint_device() else { return };
    let src = r#"
        @group(0) @binding(0) var<uniform> scale: f32;
        fn sums(k: i32) -> vec4<i32> {
            var a = 0;
            for (var i = 0; i < 9; i++) {
                if (i == 3) { continue; }
                if (i == 7) { break; }
                a += i;
            }
            var b = 0;
            var w = k + 5;
            while (w > 0) { w -= 2; b += w; }
            return vec4<i32>(a, b, k, 255);
        }
        @fragment
        fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(sums(i32(pos.x * scale))) / 255.0;
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"prog","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{"scale":1.0}}]"#,
        &format!(r#"{{"prog":{{"wgsl":{}}}}}"#, wgsl(src)),
    );
    let px = render(&device, g, 4);
    for k in 0..4i32 {
        let a = (0..7).filter(|&i| i != 3).sum::<i32>();
        let (mut b, mut w) = (0, k + 5);
        while w > 0 {
            w -= 2;
            b += w;
        }
        let got: Vec<i32> = px[4 * k as usize..4 * k as usize + 4]
            .iter()
            .map(|&v| i32::from(v))
            .collect();
        assert_eq!(got, vec![a, b, k, 255], "column {k}");
    }
}

#[test]
fn runtime_sized_storage_reads_the_bound_buffer_size() {
    let Some(device) = tint_device() else { return };
    // An 8x8 canvas sizes output_buffer at 8 * 8 * 16 = 1024 bytes: 256
    // floats. arrayLength and the robustness clamps read that size from the
    // immediate block, in the compute pass and in the buffer-to-texture copy.
    let src = r#"
        struct Params { width: f32, value: f32, _a: f32, _b: f32, }
        @group(0) @binding(0) var<storage, read_write> output_buffer: array<f32>;
        @group(0) @binding(1) var<uniform> params: Params;
        @compute @workgroup_size(8, 8, 1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
            let w = u32(params.width);
            let i = (id.y * w + id.x) * 4u;
            output_buffer[i] = f32(arrayLength(&output_buffer)) / 1024.0;
            output_buffer[i + 1u] = params.value;
            output_buffer[i + 2u] = f32(id.x) / 8.0;
            output_buffer[i + 3u] = 1.0;
        }"#;
    let g = graph(
        r#"[{"id":"c","program":"comp","inputs":{},"outputs":{"color":"global_o0"},"uniforms":{"value":0.5}}]"#,
        &format!(r#"{{"comp":{{"wgsl":{}}}}}"#, wgsl(src)),
    );
    let px = render(&device, g, 8);
    assert_eq!(&px[..4], &[64, 128, 0, 255]);
    assert_eq!(&px[16..20], &[64, 128, 128, 255]);
}

#[test]
fn point_lists_render_through_tint() {
    let Some(device) = tint_device() else { return };
    // A point-list vertex stage gets Tint's [[point_size]] output.
    let src = r#"
        @group(0) @binding(0) var<uniform> color: vec4<f32>;
        struct VOut { @builtin(position) pos: vec4<f32>, @location(0) c: vec4<f32>, }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
            var o: VOut;
            let x = (f32(i % 4u) + 0.5) / 2.0 - 1.0;
            let y = (f32(i / 4u) + 0.5) / 2.0 - 1.0;
            o.pos = vec4<f32>(x, y, 0.0, 1.0);
            o.c = vec4<f32>(1.0, 0.0, 0.0, 1.0);
            return o;
        }
        @fragment fn main(v: VOut) -> @location(0) vec4<f32> { return v.c * color; }"#;
    let g = graph(
        r#"[{"id":"p","program":"pts","inputs":{},"outputs":{"color":"global_o0"},
            "uniforms":{"color":[1.0,1.0,1.0,1.0]},"drawMode":"points","count":16,"clear":true}]"#,
        &format!(
            r#"{{"pts":{{"wgsl":{},"vertexEntryPoint":"vs","topology":"point-list"}}}}"#,
            wgsl(src)
        ),
    );
    let px = render(&device, g, 4);
    let red = px
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|p| **p == [255, 0, 0, 255])
        .count();
    assert!(red > 0, "no point was drawn: {px:?}");
}

#[test]
fn each_stage_numbers_only_its_visible_bindings() {
    let Some(device) = tint_device() else { return };
    // The vertex stage sees only `offset` (its buffer 0); the fragment stage
    // sees `tex`, `samp` and `tint` (its texture 0, sampler 0 and buffer 0),
    // the indices wgpu-hal's Metal layout gives each stage.
    let fill = r#"
        @group(0) @binding(0) var<uniform> gain: f32;
        @fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
            return vec4<f32>(pos.x / 4.0 * gain, 0.0, 0.0, 1.0);
        }"#;
    let mix = r#"
        @group(0) @binding(0) var<uniform> offset: vec4<f32>;
        @group(0) @binding(1) var tex: texture_2d<f32>;
        @group(0) @binding(2) var samp: sampler;
        @group(0) @binding(3) var<uniform> tint: vec4<f32>;
        struct VOut { @builtin(position) pos: vec4<f32>, }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
            var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
            var o: VOut;
            o.pos = vec4<f32>(p[i] + offset.xy, 0.0, 1.0);
            return o;
        }
        @fragment fn main(v: VOut) -> @location(0) vec4<f32> {
            let t = textureSampleLevel(tex, samp, v.pos.xy / 4.0, 0.0);
            return vec4<f32>(t.r, tint.g, tint.b, 1.0);
        }"#;
    let json = format!(
        r#"{{"id":"t","source":"","passes":[
            {{"id":"a","program":"fill","inputs":{{}},"outputs":{{"color":"node_t"}},"uniforms":{{"gain":1}}}},
            {{"id":"b","program":"mix","inputs":{{"tex":"node_t"}},"outputs":{{"color":"global_o0"}},
              "uniforms":{{"offset":[0,0,0,0],"tint":[0,0.5,0.25,0]}}}}],
            "programs":{{"fill":{{"wgsl":{}}},"mix":{{"wgsl":{},"vertexEntryPoint":"vs"}}}},
            "allocations":{{}},"textures":{{"node_t":{{"width":4,"height":4,"format":"rgba16f",
              "usage":["render","sample","copySrc","copyDst"]}}}},
            "renderSurface":"o0","mediaSteps":[]}}"#,
        wgsl(fill),
        wgsl(mix)
    );
    let g = Graph::from_json(&json).expect("valid test graph");
    let px = render(&device, g, 4);
    for x in 0..4usize {
        let red = ((x as f32 + 0.5) / 4.0 * 255.0).round() as u8;
        assert_eq!(&px[4 * x..4 * x + 4], &[red, 128, 64, 255], "column {x}");
    }
}

#[test]
fn written_frag_depth_reads_its_range_from_the_immediate_block() {
    let Some(device) = tint_device() else { return };
    // A depth-tested ("triangles") pass whose fragment stage writes
    // frag_depth: Dawn clamps the depth to the viewport's range, read from
    // the immediate block the pipeline then has ([0, 1]). 0.5 passes the
    // depth test against the cleared 1.0, so the triangle lands.
    let src = r#"
        @group(0) @binding(0) var<uniform> color: vec4<f32>;
        struct VOut { @builtin(position) pos: vec4<f32>, }
        @vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
            var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(-1.0, 3.0), vec2(3.0, -1.0));
            var o: VOut;
            o.pos = vec4<f32>(p[i], 0.9, 1.0);
            return o;
        }
        struct Out { @location(0) c: vec4<f32>, @builtin(frag_depth) d: f32, }
        @fragment fn main(v: VOut) -> Out {
            var o: Out;
            o.c = color;
            o.d = 0.5;
            return o;
        }"#;
    let g = graph(
        r#"[{"id":"p","program":"depth","inputs":{},"outputs":{"color":"global_o0"},
            "uniforms":{"color":[0.0,1.0,0.0,1.0]},"drawMode":"triangles","count":3,"clear":true}]"#,
        &format!(
            r#"{{"depth":{{"wgsl":{},"vertexEntryPoint":"vs"}}}}"#,
            wgsl(src)
        ),
    );
    let mut renderer = Renderer::new(&device, g, 4, 4, RendererOptions::default()).unwrap();
    renderer.render(0.25).unwrap();
    let px = renderer.read_output().unwrap().data;
    assert_eq!(&px[..4], &[0, 255, 0, 255]);
    // The program's initial pipeline (compiled without a depth attachment) is
    // invalid in WebGPU, as in the reference: it is the only fallback, and the
    // depth-tested pipeline that drew is Tint's.
    let fallbacks = renderer.pipeline().backend.tint_fallback_log();
    assert_eq!(fallbacks.len(), 1, "{fallbacks:#?}");
    assert!(
        fallbacks[0].contains("does not have a depth attachment"),
        "{fallbacks:#?}"
    );
}
