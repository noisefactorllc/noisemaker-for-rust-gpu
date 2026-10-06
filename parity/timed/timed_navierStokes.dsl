search synth
noise(seed: 1, scaleX: 50, scaleY: 50).write(o0)
navierStokes(tex: read(o0), zoom: x4, iterations: 20, speed: 100, inputForce: 1, inputDye: 1, inputIntensity: 6).write(o1)
render(o1)
