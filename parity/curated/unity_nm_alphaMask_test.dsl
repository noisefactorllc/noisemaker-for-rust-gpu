search synth, mixer
noise(seed: 1, colorMode: 1).write(o0)
gradient(seed: 1).alphaMask(tex: o0, mix: 60).write(o1)
render(o1)
