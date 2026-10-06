search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .scatter(mode: clumped, radius: 19, smoothness: 43, seed: 1)
  .write(o0)

render(o0)
