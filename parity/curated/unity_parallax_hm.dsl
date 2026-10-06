search filter, synth
noise(seed: 1, ridges: true).write(o1)
gradient().parallax(heightMap: read(o1), pivot: 0.5).write(o0)
render(o0)
