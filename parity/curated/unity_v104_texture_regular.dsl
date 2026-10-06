search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .texture(mode: regular, alpha: 0.82, scale: 2.75, intensity: 67, contrast: 32, mono: false)
  .write(o0)

render(o0)
