search synth, filter

testPattern(pattern: colorGrid, gridSize: 7)
  .mosaicTiles(mode: mosaic, tileSize: 19, groutWidth: 9, relief: 82, seed: 1)
  .write(o0)

render(o0)
