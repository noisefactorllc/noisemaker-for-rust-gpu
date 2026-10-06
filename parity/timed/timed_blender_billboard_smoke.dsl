search synth, filter, render, points, mixer

noise(type: hermite, ridges: true, speed: 30, colorMode: mono)
  .subchain(name: "flow field particles", id: "lkjw") {
    .pointsEmit()
    .flow(behavior: unruly, stride: 25)
    .pointsBillboardRender(density: 36.63, intensity: 96.07)
  }
  .write(o0)

render(o0)
