search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .extrude(type: pyramids, size: 19, depth: 64, depthSource: random, solidFront: false)
  .write(o0)

render(o0)
