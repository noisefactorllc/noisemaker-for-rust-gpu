search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .mosaicTiles(mode: shifted, tileSize: 17, maxOffset: 34, gapFill: inverse, seed: 1)
  .write(o0)

render(o0)
