search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .scatter(mode: anisotropic, radius: 17, smoothness: 38, seed: 1)
  .write(o0)

render(o0)
