search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .pondRipples(style: pondRipples, amount: 67, ridges: 9, wrap: repeat, antialias: false)
  .write(o0)

render(o0)
