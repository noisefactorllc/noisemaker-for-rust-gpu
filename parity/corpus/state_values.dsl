search synth, filter

noise(octaves: frame, ridges: time, colorMode: deltaTime, seed: resolution)
  .write(o0)

osc2d(oscType: time, freq: frame)
  .write(o1)

render(o0)
