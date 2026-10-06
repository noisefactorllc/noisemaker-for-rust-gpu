search synth, filter, mixer
noise(scaleX: 30, scaleY: 30, seed: 1).write(o0)
noise(scaleX: 50, scaleY: 50, seed: 2).blendMode(tex: read(o0), mode: hardLight, mix: 0.5).write(o1)
render(o1)
