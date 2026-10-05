search synth, render

noise(seed: 1)
  .meshRender(rotateX: 35, rotateY: -30, viewScale: 1.6, meshColor: #7ab0d8)
  .write(o0)

render(o0)
