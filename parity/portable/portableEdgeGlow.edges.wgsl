// Edge Glow, pass 1: Sobel gradient magnitude of the input's luminance,
// scaled by strength (texel loads, edges clamped).

struct Uniforms {
    strength: f32,
}

@group(0) @binding(0) var inputTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn tap(p: vec2<i32>, size: vec2<i32>) -> f32 {
    let q = clamp(p, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
    return luma(textureLoad(inputTex, q, 0).rgb);
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<i32>(textureDimensions(inputTex));
    let c = vec2<i32>(pos.xy);
    let tl = tap(c + vec2<i32>(-1, -1), size);
    let t = tap(c + vec2<i32>(0, -1), size);
    let tr = tap(c + vec2<i32>(1, -1), size);
    let l = tap(c + vec2<i32>(-1, 0), size);
    let r = tap(c + vec2<i32>(1, 0), size);
    let bl = tap(c + vec2<i32>(-1, 1), size);
    let b = tap(c + vec2<i32>(0, 1), size);
    let br = tap(c + vec2<i32>(1, 1), size);
    let gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    let gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    let g = sqrt(gx * gx + gy * gy) * uniforms.strength;
    return vec4<f32>(g, g, g, 1.0);
}
