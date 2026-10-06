search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .edge(kernel: contour, level: 61, contourSide: upper, size: kernel7x7, channel: color, amount: 148, invert: on, threshold: 9, blend: screen, mix: 81)
  .write(o0)

render(o0)
