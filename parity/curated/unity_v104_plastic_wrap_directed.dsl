search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .plasticWrap(highlight: 72, detail: 56, smoothness: 34, lightDirection: vec3(0.2, -0.4, 0.8))
  .write(o0)

render(o0)
