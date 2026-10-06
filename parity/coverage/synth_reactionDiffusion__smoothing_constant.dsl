search synth
noise(seed: 3, scaleX: 30, scaleY: 30).write(o1)
reactionDiffusion(tex: read(o1), smoothing: constant).write(o0)
render(o0)
