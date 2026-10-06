search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .parallax(heightMap: testPattern(pattern: dotGrid, gridSize: 9), direction: vec3(0.73, -0.41, 1), pivot: 0.43)
  .write(o0)

render(o0)
