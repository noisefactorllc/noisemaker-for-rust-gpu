search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .mosaicTiles(mode: shifted, tileSize: 21, maxOffset: 29, gapFill: unaltered, seed: 1)
  .write(o0)

render(o0)
