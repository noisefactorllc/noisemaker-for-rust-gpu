search synth, filter

let base = noise(2, 4)
base(6).write(o0)
base().write(o1)

let kw = noise(octaves: 3, seed: 1)
kw(seed: 5, scaleX: 10).write(o2)

let nsv = from(synth, noise(seed: 9))
nsv(octaves: 2).write(o3)

let alias = blur
noise().alias(radiusX: 3).write(o4)

let chained = noise(octaves: 5).blur()
chained().write(o5)

let twice = kw
twice(scaleY: 7).write(o6)

render(o0)
