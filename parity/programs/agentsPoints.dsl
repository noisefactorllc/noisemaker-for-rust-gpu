search synth, filter, render, points, mixer

perlin(
  scale: 100,
  octaves: 2,
  dimensions: 3,
  seed: 48
)
  .subchain(name: "flow field particles", id: "lkjw") {
    .pointsEmit(stateSize: x1024)
    .flow(
      behavior: chaotic,
      stride: 51,
      strideDeviation: 0.5,
      kink: 5.4
    )
    .pointsRender(
      density: 100,
      intensity: 74.59,
      inputIntensity: 21.46
    )
  }
  .write(o0)

render(o0)
