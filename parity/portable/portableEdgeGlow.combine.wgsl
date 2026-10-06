// Edge Glow, pass 2: the input posterized to `levels` steps per channel,
// mixed toward the glow color by the edge strength.

struct Uniforms {
    levels: i32,
    glow: vec3<f32>,
}

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var edgeTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniforms: Uniforms;

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = vec2<i32>(pos.xy);
    let base = textureLoad(inputTex, c, 0);
    let edge = clamp(textureLoad(edgeTex, c, 0).r, 0.0, 1.0);
    let steps = f32(max(uniforms.levels, 2) - 1);
    let poster = floor(base.rgb * steps + vec3<f32>(0.5, 0.5, 0.5)) / steps;
    return vec4<f32>(mix(poster, uniforms.glow, edge), base.a);
}
