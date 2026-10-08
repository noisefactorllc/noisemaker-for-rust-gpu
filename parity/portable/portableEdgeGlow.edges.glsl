// Edge Glow, pass 1: Sobel gradient magnitude of the input's luminance,
// scaled by strength (texel loads, edges clamped).

#ifdef GL_ES
precision highp float;
precision highp int;
#endif

uniform sampler2D inputTex;
uniform float strength;

out vec4 fragColor;

float luma(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

float tap(ivec2 p, ivec2 size) {
    ivec2 q = clamp(p, ivec2(0, 0), size - ivec2(1, 1));
    return luma(texelFetch(inputTex, q, 0).rgb);
}

void main() {
    ivec2 size = textureSize(inputTex, 0);
    ivec2 c = ivec2(gl_FragCoord.xy);
    float tl = tap(c + ivec2(-1, -1), size);
    float t = tap(c + ivec2(0, -1), size);
    float tr = tap(c + ivec2(1, -1), size);
    float l = tap(c + ivec2(-1, 0), size);
    float r = tap(c + ivec2(1, 0), size);
    float bl = tap(c + ivec2(-1, 1), size);
    float b = tap(c + ivec2(0, 1), size);
    float br = tap(c + ivec2(1, 1), size);
    float gx = (tr + 2.0 * r + br) - (tl + 2.0 * l + bl);
    float gy = (bl + 2.0 * b + br) - (tl + 2.0 * t + tr);
    float g = sqrt(gx * gx + gy * gy) * strength;
    fragColor = vec4(g, g, g, 1.0);
}
