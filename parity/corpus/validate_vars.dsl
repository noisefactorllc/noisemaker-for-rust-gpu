search synth, filter

let a = null
let b = undefined
let c = nowhere
let d = time
let e = noise
let f = 5
let g = f
let h = noise(octaves: 3)
let i = noise().blur()
let j = oscKind.saw
let k = synth.noise
let l = foo.bar
let n = #ff0000
let o = "str"
let p = blur
let q = d
let r = (2 + 3) * 4

e(octaves: 4).write(o0)
h(5).write(o1)
h(octaves: 6, seed: 2).write(o2)
noise(octaves: f, seed: g).write(o3)
noise(octaves: j, seed: r).write(o4)
noise(octaves: l).write(o5)
solid(color: n).write(o6)
noise(octaves: d, scaleX: q).write(o7)
noise(type: k).write(o0)
noise().p(radiusX: g).write(o1)
noise(octaves: o).write(o2)

render(o0)
