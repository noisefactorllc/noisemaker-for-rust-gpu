search synth, filter, classicNoisedeck
noise(seed: 1, scaleX: 50, scaleY: 50).write(o0)
gradient(seed: 1).coalesce(tex: o0, blendMode: phoenix).write(o1)
render(o1)
