search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .spinBlur(amount: 37, centerX: 0.35, centerY: 0.3)
  .write(o0)

render(o0)
