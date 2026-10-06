search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .scatter(mode: lightenOnly, radius: 15, smoothness: 31, seed: 1)
  .write(o0)

render(o0)
