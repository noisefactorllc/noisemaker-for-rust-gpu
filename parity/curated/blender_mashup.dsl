search synth, mixer
noise(seed: 1, scaleX: 40, scaleY: 40).write(o0)
noise(seed: 7, scaleX: 12, scaleY: 12).write(o1)
mashup(source: read(o0), layer0_tex: read(o0), layer1_tex: read(o1), layers: 3).write(o2)
render(o2)
