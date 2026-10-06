search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .morphology(mode: dilate, radius: 6, shape: square)
  .write(o0)

render(o0)
