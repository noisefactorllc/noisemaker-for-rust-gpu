search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .halftone(mode: mono, pattern: dot, frequency: 31, monoAngle: 37, sharpness: 69)
  .write(o0)

render(o0)
