search synth

let bare = osc()
let kind = osc(oscKind.tri)
let positional = osc(oscKind.saw, 0.25, 0.75, 2, 0.5, 3)
let partial = osc(oscKind.square, 0.1)
let typed = osc(type: oscKind.noise, min: 0.2, max: 0.8, speed: 4, offset: 0.1, seed: 9)
let onlyKw = osc(speed: 2, seed: 5)
let typeOnly = osc(type: 1)
let fallthrough = osc(1, 2)
let otherKw = osc(foo: 1, speed: 2)
let notKind = osc(kind.sine)
noise(octaves: osc(oscKind.sine, 1, 4), seed: osc(min: 1, max: 10)).write(o0)
render(o0)
