// Rings: concentric bands around the center (round, square or diamond),
// mixed from a hue ramp into the tint, drifting outward with time.

#ifdef GL_ES
precision highp float;
precision highp int;
#endif

uniform vec2 resolution;
uniform float time;
uniform float freq;
uniform int shape;
uniform vec3 tint;
uniform float speed;

out vec4 fragColor;

const float TAU = 6.283185307179586;

float distanceFromCenter(vec2 p) {
    if (shape == 1) {
        return max(abs(p.x), abs(p.y));
    }
    if (shape == 2) {
        return (abs(p.x) + abs(p.y)) * 0.7071068;
    }
    return length(p);
}

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    float d = distanceFromCenter(uv - vec2(0.5, 0.5));
    float band = 0.5 + 0.5 * cos(TAU * (d * freq - time * speed));
    vec3 hue = vec3(0.5, 0.5, 0.5) + 0.5 * cos(TAU * (vec3(0.0, 0.33, 0.67) + d * 2.0));
    vec3 color = mix(hue * 0.3, tint, band);
    fragColor = vec4(color, 1.0);
}
