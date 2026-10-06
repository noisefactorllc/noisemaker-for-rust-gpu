search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .directionalBlur(angle: 37, distance: 75)
  .write(o0)

render(o0)
