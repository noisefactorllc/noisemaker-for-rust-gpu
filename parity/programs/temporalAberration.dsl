search synth, filter

noise(seed: 1, scaleX: 50, scaleY: 50)
  .temporalAberration(redDelay: 0, greenDelay: 4, blueDelay: 8)
  .write(o0)

render(o0)
