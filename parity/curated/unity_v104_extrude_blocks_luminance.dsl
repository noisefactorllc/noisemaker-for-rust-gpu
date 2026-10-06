search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .extrude(type: blocks, size: 23, depth: 58, depthSource: luminance, solidFront: true)
  .write(o0)

render(o0)
