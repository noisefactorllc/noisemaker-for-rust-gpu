//! GPU tests of the Metal shader lowering (`noisemaker_gpu::lowering`) of the
//! naga compiler path: the rewritten loops keep their WGSL semantics on the
//! device, and on Metal a constant-trip-count loop compiles to the arithmetic
//! of its unrolled form, the shape Metal's compiler gives the loop Tint
//! writes.

use noisemaker_gpu::lowering::lower_for_tint_msl;
use noisemaker_gpu::{DeviceOptions, GpuDevice, Graph, Renderer, RendererOptions, ShaderCompiler};

/// A device compiling with naga (the path the lowering applies to).
fn naga_device() -> GpuDevice {
    GpuDevice::create(&DeviceOptions {
        shader_compiler: Some(ShaderCompiler::Naga),
        ..Default::default()
    })
    .expect("a GPU adapter")
}

/// Render a one-pass graph running `src` (with its `scale` uniform at 1)
/// and read the RGBA8 output.
fn render_wgsl(device: &GpuDevice, src: &str, size: u32) -> Vec<u8> {
    let json = format!(
        r#"{{"id":"t","source":"","passes":[{{"id":"p","program":"prog","inputs":{{}},
            "outputs":{{"color":"global_o0"}},"uniforms":{{"scale":1.0}}}}],
            "programs":{{"prog":{{"wgsl":{}}}}},"allocations":{{}},"textures":{{}},
            "renderSurface":"o0","mediaSteps":[]}}"#,
        serde_json::to_string(src).unwrap()
    );
    let graph = Graph::from_json(&json).expect("valid test graph");
    let mut renderer =
        Renderer::new(device, graph, size, size, RendererOptions::default()).unwrap();
    renderer.render(0.0).unwrap();
    let pixels = renderer.read_output().unwrap();
    assert_eq!(
        renderer.device_errors(),
        0,
        "device errors: {:#?}",
        renderer.pipeline().backend.device_error_log
    );
    pixels.data
}

const LOOPS: &str = r#"
@group(0) @binding(0) var<uniform> scale: f32;

fn sums(k: i32) -> vec4<i32> {
    var a = 0;
    for (var i = 0; i < 9; i++) {
        if (i == 3) { continue; }
        if (i == 7) { break; }
        a += i;
    }
    var b = 0;
    for (var i: i32 = k; i >= 0; i = i - 1) {
        for (var j = 0; j <= k; j++) {
            if (j == 1) { continue; }
            b += j + 1;
        }
    }
    var c = 0;
    var w = k + 5;
    while (w > 0) {
        w -= 2;
        if (w == 1) { continue; }
        c += w;
    }
    var d = 0;
    loop {
        d += 3;
        if (d > 10 + k) { break; }
        continue;
    }
    return vec4<i32>(a, b, c, d);
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return vec4<f32>(sums(i32(pos.x * scale))) / 255.0;
}
"#;

/// `sums(k)` of [`LOOPS`], evaluated on the CPU.
fn sums(k: i32) -> [i32; 4] {
    let a = (0..7).filter(|&i| i != 3).sum();
    let b = (0..=k)
        .map(|_| (0..=k).filter(|&j| j != 1).map(|j| j + 1).sum::<i32>())
        .sum();
    let (mut c, mut w) = (0, k + 5);
    while w > 0 {
        w -= 2;
        if w != 1 {
            c += w;
        }
    }
    let mut d = 0;
    loop {
        d += 3;
        if d > 10 + k {
            break;
        }
    }
    [a, b, c, d]
}

#[test]
fn rewritten_loops_keep_their_semantics() {
    // The lowering rewrites every loop here: the constant `for` loop rotates,
    // the others get Tint's loop counter.
    let lowered = lower_for_tint_msl(LOOPS).expect("loops to rewrite");
    assert!(lowered.self_bounded);
    assert!(lowered.source.contains("nm_loop_idx_"));

    let device = naga_device();
    let px = render_wgsl(&device, LOOPS, 4);
    for k in 0..4 {
        let expected = sums(k);
        let got = &px[4 * k as usize..4 * k as usize + 4];
        let got: Vec<i32> = got.iter().map(|&v| i32::from(v)).collect();
        assert_eq!(got, expected, "column {k}");
    }
}

/// A float encoded into four bytes of the RGBA8 output.
const ENCODE: &str = r#"
@group(0) @binding(0) var<uniform> scale: f32;

fn encode(v: f32) -> vec4<f32> {
    let b = bitcast<u32>(v);
    let bytes = vec4<f32>(f32(b & 255u), f32((b >> 8u) & 255u), f32((b >> 16u) & 255u), f32(b >> 24u));
    return bytes * scale / 255.0;
}
fn h(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
"#;

#[test]
fn constant_trip_count_loops_compile_to_their_unrolled_arithmetic() {
    let device = naga_device();
    if !device.is_metal() {
        // The lowering mirrors Metal's compiler; other backends keep naga's shape.
        return;
    }
    let looped = format!(
        "{ENCODE}
        @fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
            var p = pos.xy * 0.37;
            var v = 0.0;
            var amp = 0.5;
            for (var i = 0; i < 5; i++) {{
                v += amp * h(p);
                p *= 2.03;
                amp *= 0.5;
            }}
            return encode(v);
        }}"
    );
    let step = "v += amp * h(p); p *= 2.03; amp *= 0.5;\n";
    let unrolled = format!(
        "{ENCODE}
        @fragment fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
            var p = pos.xy * 0.37;
            var v = 0.0;
            var amp = 0.5;
            {}
            return encode(v);
        }}",
        step.repeat(5)
    );
    let a = render_wgsl(&device, &looped, 16);
    let b = render_wgsl(&device, &unrolled, 16);
    assert_eq!(a, b);
}
