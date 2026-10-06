search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .stipple(mode: reticulation, cellSize: 13, grainSize: 2.5, density: 57, seed: 1)
  .write(o0)

render(o0)
