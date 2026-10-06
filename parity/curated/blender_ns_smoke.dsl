search synth, filter, render, mixer

noise(type: hermite, ridges: true, speed: 30, colorMode: mono)
  .write(o0)

navierStokes(
  tex: read(o0),
  iterations: 40,
  speed: 55.14,
  dyeDecay: 99.1,
  velocityDecay: 100,
  inputForce: 0.39,
  inputDye: 1,
  inputIntensity: 9.55
)
  .write(o1)

render(o1)
