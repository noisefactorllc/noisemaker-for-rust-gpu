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
    .pointsBillboardRender(
      shapeMode: soft,
      depositOpacity: 100,
      pointSize: 33.19,
      sizeVariation: 100,
      seed: 0,
      density: 0.78,
      intensity: 44.72,
      inputIntensity: 18.23
    )
  }
  .blur()
  .write(o0)

navierStokes(
  tex: read(o0),
  zoom: x4,
  iterations: 40,
  smoothing: bSpline4x4,
  speed: 145,
  dyeDecay: 97.52,
  velocityDecay: 100,
  inputForce: 1,
  inputDye: 1,
  inputIntensity: 6.01
)
  .write(o1)

render(o1)
