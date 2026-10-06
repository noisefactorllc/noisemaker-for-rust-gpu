search synth, filter, render, points, mixer

perlin(scale: 100, octaves: 2, dimensions: 3, seed: 48)
  .subchain(name: "flow field particles", id: "lkjw") {
    .pointsEmit(stateSize: x1024)
    .flow(behavior: chaotic, stride: 51, strideDeviation: 0.5, kink: 5.4)
    .pointsBillboardRender(shapeMode: soft, depositOpacity: 100, pointSize: 33.19, sizeVariation: 100, seed: 0, density: 0.78, intensity: 44.72, inputIntensity: 18.23)
  }
  .write(o0)

render(o0)
