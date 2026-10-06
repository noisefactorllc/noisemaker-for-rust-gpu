search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .median(radius: 2, threshold: 37)
  .write(o0)

render(o0)
