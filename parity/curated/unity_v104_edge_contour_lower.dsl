search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .edge(kernel: contour, level: 42, contourSide: lower, size: kernel5x5, channel: luminance, amount: 135, invert: off, threshold: 11, blend: overlay, mix: 76)
  .write(o0)

render(o0)
