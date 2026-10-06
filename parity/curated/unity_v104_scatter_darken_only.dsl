search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .scatter(mode: darkenOnly, radius: 13, smoothness: 27, seed: 1)
  .write(o0)

render(o0)
