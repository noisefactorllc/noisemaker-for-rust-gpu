search synth, filter, render, points, mixer

perlin(octaves: 6, ridges: true)
  .subchain(name: "flow field particles", id: "lkjw") {
    .pointsEmit()
    .flow(behavior: randomMix, stride: 39, kink: 5.9)
    .flock(separation: 2.8, alignment: 1.4, cohesion: 2.9, perceptionRadius: 35, separationRadius: 41, maxSpeed: 6.8)
    .pointsRender()
  }
  .write(o0)

render(o0)
