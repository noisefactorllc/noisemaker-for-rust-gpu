search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .plasticWrap(highlight: 72, detail: 56, smoothness: 34, lightDirection: vec3(0, 0, 0))
  .write(o0)

render(o0)
