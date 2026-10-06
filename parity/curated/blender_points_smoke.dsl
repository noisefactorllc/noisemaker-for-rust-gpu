search synth, filter, render, points, mixer

noise(
  type: hermite,
  ridges: true,
  speed: 30,
  colorMode: mono
)
  .subchain(name: "flow field particles", id: "lkjw") {
    .pointsEmit()
    .flow(behavior: unruly, stride: 25)
    .pointsRender(
      density: 36.63,
      intensity: 96.07,
      inputIntensity: 0
    )
  }
  .write(o0)

render(o0)
