search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .halftone(mode: color, pattern: dot, frequency: 27, sharpness: 74)
  .write(o0)

render(o0)
