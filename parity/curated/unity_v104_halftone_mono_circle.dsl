search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .halftone(mode: mono, pattern: circle, frequency: 33, monoAngle: 21, sharpness: 72)
  .write(o0)

render(o0)
