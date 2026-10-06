search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .parallax(heightMap: mandala(scale: 13, rotation: 27, symmetry: 11, layers: 7, layerSpacing: 1.7, twist: 19, shapeGrowth: 0.35), direction: vec3(-0.58, 0.44, 1), pivot: 0.47)
  .write(o0)

render(o0)
