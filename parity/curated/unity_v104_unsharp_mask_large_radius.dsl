search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .unsharpMask(amount: 275, radius: 40, threshold: 17)
  .write(o0)

render(o0)
