search synth, filter
noise(seed: 1, scaleX: 50, scaleY: 50).morphology(mode: dilate, shape: round, radius: 9).write(o0)
render(o0)
