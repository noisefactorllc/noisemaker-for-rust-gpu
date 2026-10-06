search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .morphology(mode: erode, radius: 5.5, shape: round)
  .write(o0)

render(o0)
