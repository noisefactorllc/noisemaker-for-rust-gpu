search synth, mixer
gradient(seed: 1).write(o0)
noise(seed: 1, scaleX: 50, scaleY: 50).write(o1)
solid(color: #ff8000).write(o2)
mashup(source: read(o0), layers: 3, smoothness: 0.2, layer0_tex: read(o1), layer2_tex: read(o2)).write(o3)
render(o3)
