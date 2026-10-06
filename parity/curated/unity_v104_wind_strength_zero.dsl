search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .wind(method: blast, direction: fromRight, strength: 0, threshold: 21)
  .write(o0)

render(o0)
