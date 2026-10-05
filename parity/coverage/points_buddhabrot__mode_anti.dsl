search points, synth, render

perlin()
  .pointsEmit(stateSize: 512)
  .buddhabrot(mode: anti)
  .pointsRender(intensity: 99)
  .write(o0)

render(o0)
