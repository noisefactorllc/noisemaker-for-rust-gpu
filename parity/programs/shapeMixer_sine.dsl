search synth, filter, classicNoisedeck
noise(seed: 1, scaleX: 50, scaleY: 50).write(o0)
gradient(seed: 1).shapeMixer(tex: o0, loopOffset: 380, seed: 5).write(o1)
render(o1)
