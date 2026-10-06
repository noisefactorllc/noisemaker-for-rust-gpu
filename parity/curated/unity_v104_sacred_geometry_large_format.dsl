search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .parallax(heightMap: sacredGeometry(geometry: starPolygon, scale: 12, starPoints: 7, rotation: -31, thickness: 0.27, smoothness: 0.04), direction: vec3(0.62, 0.39, 1), pivot: 0.52)
  .write(o0)

render(o0)
