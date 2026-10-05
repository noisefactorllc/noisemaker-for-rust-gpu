search synth, filter
noise(seed: 1, scaleX: 50, scaleY: 50).morphology(mode: erode, shape: square, radius: 9).write(o0)
render(o0)
