search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .lowPoly(scale: 37, seed: 1, mode: edges, edgeStrength: 0.34, borderWidth: 43, lightIntensity: 67, alpha: 0.92)
  .write(o0)

render(o0)
