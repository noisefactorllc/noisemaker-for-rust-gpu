// Expander: inline chains as surface arguments ({kind: 'temp'}) resolve to
// the inline chain's output; read() arguments resolve to global surfaces.
search synth, mixer, filter
noise().blendMode(tex: noise(seed: 2).blur()).write(o0)
noise(seed: 3).blendMode(tex: solid()).blendMode(tex: read(o0)).write(o1)
noise(seed: 4).alphaMask(tex: perlin(), baseTex: noise().invert()).write(o2)
render(o2)
