// Rings: concentric bands around the center (round, square or diamond),
// mixed from a hue ramp into the tint, drifting outward with time.

@group(0) @binding(0) var<uniform> resolution: vec2<f32>;
@group(0) @binding(1) var<uniform> time: f32;
@group(0) @binding(2) var<uniform> freq: f32;
@group(0) @binding(3) var<uniform> shape: i32;
@group(0) @binding(4) var<uniform> tint: vec3<f32>;
@group(0) @binding(5) var<uniform> speed: f32;

const TAU: f32 = 6.283185307179586;

fn distanceFromCenter(p: vec2<f32>) -> f32 {
    if (shape == 1) {
        return max(abs(p.x), abs(p.y));
    }
    if (shape == 2) {
        return (abs(p.x) + abs(p.y)) * 0.7071068;
    }
    return length(p);
}

@fragment
fn main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = position.xy / resolution;
    let d = distanceFromCenter(uv - vec2<f32>(0.5, 0.5));
    let band = 0.5 + 0.5 * cos(TAU * (d * freq - time * speed));
    let hue = vec3<f32>(0.5, 0.5, 0.5) + 0.5 * cos(TAU * (vec3<f32>(0.0, 0.33, 0.67) + d * 2.0));
    let color = mix(hue * 0.3, tint, band);
    return vec4<f32>(color, 1.0);
}
