search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .stipple(mode: mezzoStrokes, cellSize: 11, grainSize: 3.5, density: 63, seed: 1)
  .write(o0)

render(o0)
