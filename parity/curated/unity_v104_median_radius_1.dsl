search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .median(radius: 1, threshold: 18)
  .write(o0)

render(o0)
