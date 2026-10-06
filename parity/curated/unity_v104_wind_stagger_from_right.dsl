search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .wind(method: stagger, direction: fromRight, strength: 83, threshold: 17)
  .write(o0)

render(o0)
