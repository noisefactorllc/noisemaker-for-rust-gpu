search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .dither(type: errorDiffusion, matrixScale: 3, threshold: 0.08, palette: pico8, levels: 6, mix: 0.82)
  .write(o0)

render(o0)
