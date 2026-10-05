search synth, filter

noise(seed: 1,).blur(radiusX: 2, radiusY: 3,).write(o0)
noise(1,).write(o1)
noise(2, 3,).blur().write(o2)
noise().blur(
  radiusX: 4,
  radiusY: 5,
).write(o3)
let m = midi(1, channel: 2, min: 0,)
render(o0)
