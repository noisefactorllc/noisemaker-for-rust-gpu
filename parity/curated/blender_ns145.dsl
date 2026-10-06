search synth, filter, render, mixer

noise(type: hermite, ridges: true, speed: 30, colorMode: mono)
  .write(o0)

navierStokes(tex: read(o0), zoom: x4, iterations: 40, smoothing: bSpline4x4, speed: 145, dyeDecay: 97.52, velocityDecay: 100, inputForce: 1, inputDye: 1, inputIntensity: 6.01)
  .write(o1)

render(o1)
