search synth, filter

testPattern(pattern: uvMap, gridSize: 7)
  .pondRipples(style: pondRipples, amount: 71, ridges: 11, wrap: clamp, antialias: true)
  .write(o0)

render(o0)
