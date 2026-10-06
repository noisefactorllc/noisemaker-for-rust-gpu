search filter, synth
noise(seed: 1, ridges: true).write(o1)
gradient().lighting(heightMap: read(o1), normalStrength: 2).write(o0)
render(o0)
