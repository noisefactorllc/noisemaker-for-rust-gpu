// Edge Glow, pass 2: the input posterized to `levels` steps per channel,
// mixed toward the glow color by the edge strength.

#ifdef GL_ES
precision highp float;
precision highp int;
#endif

uniform sampler2D inputTex;
uniform sampler2D edgeTex;
uniform int levels;
uniform vec3 glow;

out vec4 fragColor;

void main() {
    ivec2 c = ivec2(gl_FragCoord.xy);
    vec4 base = texelFetch(inputTex, c, 0);
    float edge = clamp(texelFetch(edgeTex, c, 0).r, 0.0, 1.0);
    float steps = float(max(levels, 2) - 1);
    vec3 poster = floor(base.rgb * steps + vec3(0.5, 0.5, 0.5)) / steps;
    fragColor = vec4(mix(poster, glow, edge), base.a);
}
