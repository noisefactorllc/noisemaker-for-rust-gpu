//! The linked Tint generates the MSL Chromium generates.
//!
//! The expected texts below are Chromium 153.0.8010.12's own MSL (Dawn
//! 50c9f7b4, Apple M4), dumped with Dawn's `dump_shaders` toggle while the
//! reference engine built the flock fixture's pipelines: the reference's
//! default vertex shader and a texture passthrough fragment shader. The
//! options are `dawn::options` with Dawn's Metal argument-table indices and
//! the entry point name Chromium derives from the page's origin.
//!
//! Tint is built for Apple targets only, so this test runs there only.
#![cfg(target_vendor = "apple")]

use noisemaker_tint::{Binding, MslOptions, ResourceClass, Stage, dawn, wgsl_to_msl};

const APPLE_M4: dawn::MetalGpu = dawn::MetalGpu {
    vendor: dawn::Vendor::Apple,
    apple9: true,
};

/// The isolated entry point name of a page served from http://127.0.0.1.
const ORIGIN_ENTRY_POINT: &str =
    "dawn_entry_point_687474703a2f2f3132372e302e302e3120687474703a2f2f3132372e302e302e31";

const DEFAULT_VERTEX_WGSL: &str = r#"
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    let pos = positions[vertexIndex];

    var out: VertexOutput;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
    return out;
}
"#;

const DEFAULT_VERTEX_CHROMIUM_MSL: &str = r#"#ifdef __clang__
#pragma clang diagnostic ignored "-Wall"
#endif

#pragma METAL fp math_mode(relaxed)
#include <metal_stdlib>
using namespace metal;

struct tint_struct {
  float4 tint_member;
  float2 tint_member_1;
};

template<typename T, size_t N>
struct tint_array {
  const constant T& operator[](size_t i) const constant { return elements[i]; }
  device T& operator[](size_t i) device { return elements[i]; }
  const device T& operator[](size_t i) const device { return elements[i]; }
  thread T& operator[](size_t i) thread { return elements[i]; }
  const thread T& operator[](size_t i) const thread { return elements[i]; }
  threadgroup T& operator[](size_t i) threadgroup { return elements[i]; }
  const threadgroup T& operator[](size_t i) const threadgroup { return elements[i]; }
  T elements[N];
};

struct tint_struct_1 {
  float4 tint_member_2 [[position]];
  float2 tint_member_3 [[user(locn0)]];
};

tint_struct v(uint v_1) {
  tint_array<float2, 3> const v_2 = tint_array<float2, 3>{float2(-1.0f), float2(3.0f, -1.0f), float2(-1.0f, 3.0f)};
  float2 const v_3 = v_2[min(v_1, 2u)];
  tint_struct v_4 = {};
  v_4.tint_member = float4(v_3, 0.0f, 1.0f);
  v_4.tint_member_1 = ((v_3 * 0.5f) + float2(0.5f));
  return v_4;
}

vertex tint_struct_1 dawn_entry_point_687474703a2f2f3132372e302e302e3120687474703a2f2f3132372e302e302e31(uint v_6 [[vertex_id]]) {
  tint_struct const v_7 = v(v_6);
  tint_struct_1 v_8 = {};
  v_8.tint_member_2 = v_7.tint_member;
  v_8.tint_member_3 = v_7.tint_member_1;
  return v_8;
}
"#;

const PASSTHROUGH_WGSL: &str = r#"@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var inputTexSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@fragment
fn main(in: VertexOutput) -> @location(0) vec4f {
    return textureSample(inputTex, inputTexSampler, in.uv);
}
"#;

const PASSTHROUGH_CHROMIUM_MSL: &str = r#"#ifdef __clang__
#pragma clang diagnostic ignored "-Wall"
#endif

#pragma METAL fp math_mode(relaxed)
#include <metal_stdlib>
using namespace metal;

struct tint_struct {
  float4 tint_member;
  float2 tint_member_1;
};

struct tint_struct_1 {
  texture2d<float, access::sample> tint_member_2;
  sampler tint_member_3;
};

struct tint_struct_2 {
  float4 tint_member_4 [[color(0)]];
};

struct tint_struct_3 {
  float2 tint_member_5 [[user(locn0)]];
};

float4 v(tint_struct v_1, tint_struct_1 v_2) {
  return v_2.tint_member_2.sample(v_2.tint_member_3, v_1.tint_member_1);
}

fragment tint_struct_2 dawn_entry_point_687474703a2f2f3132372e302e302e3120687474703a2f2f3132372e302e302e31(float4 v_4 [[position]], tint_struct_3 v_5 [[stage_in]], texture2d<float, access::sample> v_6 [[texture(0)]], sampler v_7 [[sampler(0)]]) {
  tint_struct_1 const v_8 = tint_struct_1{.tint_member_2=v_6, .tint_member_3=v_7};
  tint_struct_2 v_9 = {};
  v_9.tint_member_4 = v(tint_struct{.tint_member=v_4, .tint_member_1=v_5.tint_member_5}, v_8);
  return v_9;
}
"#;

/// Chromium's text as this process's Dawn would emit it (Dawn writes the
/// math-mode pragma only where Metal supports it).
fn expected(chromium: &str) -> String {
    if noisemaker_tint::math_mode_pragma_available() {
        chromium.to_owned()
    } else {
        chromium.replace("\n#pragma METAL fp math_mode(relaxed)\n", "")
    }
}

fn options(stage: Stage, entry_point: &str) -> MslOptions {
    let mut o = dawn::options(stage, entry_point, &APPLE_M4, false, 0xFFFF_FFFF);
    o.remapped_entry_point = ORIGIN_ENTRY_POINT.to_owned();
    o
}

#[test]
fn linked_revision_is_the_pin() {
    assert_eq!(
        noisemaker_tint::linked_dawn_revision(),
        noisemaker_tint::DAWN_COMMIT
    );
    assert_eq!(
        noisemaker_tint::DAWN_COMMIT,
        "50c9f7b4ee3fef0bdc9166056098271ea85ef9fc"
    );
}

#[test]
fn default_vertex_shader_matches_chromium() {
    let msl = wgsl_to_msl(DEFAULT_VERTEX_WGSL, &options(Stage::Vertex, "vs_main")).unwrap();
    assert_eq!(msl.source, expected(DEFAULT_VERTEX_CHROMIUM_MSL));
    assert!(!msl.has_invariant_attribute);
}

#[test]
fn passthrough_fragment_shader_matches_chromium() {
    let mut o = options(Stage::Fragment, "main");
    o.bindings = vec![
        Binding {
            group: 0,
            binding: 0,
            class: ResourceClass::Texture,
            slot: 0,
        },
        Binding {
            group: 0,
            binding: 1,
            class: ResourceClass::Sampler,
            slot: 0,
        },
    ];
    let msl = wgsl_to_msl(PASSTHROUGH_WGSL, &o).unwrap();
    assert_eq!(msl.source, expected(PASSTHROUGH_CHROMIUM_MSL));
}

#[test]
fn bindings_set_the_metal_indices() {
    let mut o = options(Stage::Fragment, "main");
    o.bindings = vec![
        Binding {
            group: 0,
            binding: 0,
            class: ResourceClass::Texture,
            slot: 3,
        },
        Binding {
            group: 0,
            binding: 1,
            class: ResourceClass::Sampler,
            slot: 5,
        },
    ];
    let msl = wgsl_to_msl(PASSTHROUGH_WGSL, &o).unwrap().source;
    assert!(
        msl.contains("[[texture(3)]]") && msl.contains("[[sampler(5)]]"),
        "{msl}"
    );
}

#[test]
fn point_list_vertex_stage_writes_the_point_size() {
    let o = dawn::options(Stage::Vertex, "vs_main", &APPLE_M4, true, 0xFFFF_FFFF);
    let msl = wgsl_to_msl(DEFAULT_VERTEX_WGSL, &o).unwrap().source;
    assert!(msl.contains("[[point_size]]"), "{msl}");
    assert!(
        msl.contains("vertex tint_struct_1 dawn_entry_point("),
        "{msl}"
    );
}

#[test]
fn range_analysis_elides_provably_in_bounds_clamps() {
    // Chromium 153 runs the robustness transform with integer range analysis.
    let wgsl = r#"
@fragment fn main() -> @location(0) vec4f {
    var a = array<f32, 4>(1.0, 2.0, 3.0, 4.0);
    var s = 0.0;
    for (var i = 0; i < 4; i++) { s += a[i]; }
    return vec4f(s);
}
"#;
    let on = wgsl_to_msl(wgsl, &options(Stage::Fragment, "main"))
        .unwrap()
        .source;
    assert!(!on.contains("min("), "{on}");
    let mut o = options(Stage::Fragment, "main");
    o.disable_integer_range_analysis = true;
    let off = wgsl_to_msl(wgsl, &o).unwrap().source;
    assert!(off.contains("min("), "{off}");
}

#[test]
fn compute_stage_reports_its_workgroup_size() {
    let wgsl = "@compute @workgroup_size(8, 4, 1) fn main() {}";
    let msl = wgsl_to_msl(wgsl, &options(Stage::Compute, "main")).unwrap();
    assert_eq!(msl.workgroup_size, [8, 4, 1]);
    assert!(msl.source.contains("kernel void"), "{}", msl.source);
}

#[test]
fn runtime_sized_storage_reads_its_size_from_the_immediate_block() {
    let wgsl = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<f32>;
@compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {
    buf[id.x] = f32(arrayLength(&buf));
}
"#;
    let mut o = options(Stage::Compute, "main");
    o.bindings = vec![Binding {
        group: 0,
        binding: 0,
        class: ResourceClass::Storage,
        slot: 1,
    }];
    o.buffer_sizes = vec![noisemaker_tint::BufferSize {
        group: 0,
        binding: 0,
        index: 0,
    }];
    o.buffer_sizes_offset = Some(0);
    o.immediate_slot = Some(0);
    o.strip_all_names = false;
    let msl = wgsl_to_msl(wgsl, &o).unwrap();
    assert!(
        msl.source.contains("tint_storage_buffer_sizes"),
        "{}",
        msl.source
    );
    assert!(msl.source.contains("[[buffer(0)]]") && msl.source.contains("[[buffer(1)]]"));
}

#[test]
fn invalid_wgsl_is_an_error() {
    let err = wgsl_to_msl("fn main( {", &options(Stage::Fragment, "main")).unwrap_err();
    assert!(err.0.starts_with("Error while parsing WGSL"), "{err}");
}
