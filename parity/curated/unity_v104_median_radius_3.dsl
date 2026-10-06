search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .median(radius: 3, threshold: 71)
  .write(o0)

render(o0)
