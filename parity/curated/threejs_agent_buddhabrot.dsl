search synth, filter, render, points, mixer
noise(scaleX: 80, scaleY: 80, seed: 5, speed: 30)
  .subchain(name: "agents", id: "agbuddhabrot") {
    .pointsEmit()
    .buddhabrot()
    .pointsRender(density: 50, intensity: 80)
  }
  .write(o0)
render(o0)
