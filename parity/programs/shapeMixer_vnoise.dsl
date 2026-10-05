search synth, filter, classicNoisedeck
noise(seed: 1, scaleX: 50, scaleY: 50).write(o0)
gradient(seed: 1).shapeMixer(tex: o0, loopOffset: 300, seed: 3).write(o1)
render(o1)
