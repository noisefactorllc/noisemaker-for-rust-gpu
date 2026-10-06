search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .halftone(mode: mono, pattern: line, frequency: 29, monoAngle: 63, sharpness: 77)
  .write(o0)

render(o0)
