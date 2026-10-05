search synth, filter
noise(seed: 7, scaleX: 12, scaleY: 12).write(o0)
noise(seed: 1, scaleX: 50, scaleY: 50).parallax(direction: vec3(0.8, 0.3, 1), pivot: 0.3, heightMap: read(o0)).write(o1)
render(o1)
