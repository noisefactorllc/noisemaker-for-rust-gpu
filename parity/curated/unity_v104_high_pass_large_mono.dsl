search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .highPass(radius: 48, mono: true)
  .write(o0)

render(o0)
