search synth, filter

let a = from(synth, noise())
let b = from(filter, blur(radiusX: 2, radiusY: 3))
let c = from(synth.sub, noise(1))
let d = from(filter, from(synth, noise()))
let e = from(classicNoisedeck, noise(seed: 1))
from(synth, noise(seed: 2)).blur().write(o0)
noise().from(filter, blur()).write(o1)
render(o0)
