search synth, filter

noise(octaves: () => Math.sin(time) * 4 + 5, seed: () => seed + 1, speed: () => time)
  .blur(radiusX: () => time % 10, radiusY: () => 3)
  .write(o0)

render(o0)
