search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .hatch(mode: coloredPencil, direction: leftDiag, strokeLength: 63, balance: 41, pressure: 72)
  .write(o0)

render(o0)
