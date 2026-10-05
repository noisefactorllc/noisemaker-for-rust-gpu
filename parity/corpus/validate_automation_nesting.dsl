search synth

let first = osc(type: oscKind.sine, speed: second)
let second = osc(type: oscKind.tri, speed: first)
noise(octaves: first).write(o0)
noise(octaves: second).write(o1)

let self = osc(speed: self)
noise(octaves: self).write(o2)

let rate9 = osc(type: oscKind.sine)
let rate8 = osc(type: oscKind.sine, speed: rate9)
let rate7 = osc(type: oscKind.sine, speed: rate8)
let rate6 = osc(type: oscKind.sine, speed: rate7)
let rate5 = osc(type: oscKind.sine, speed: rate6)
let rate4 = osc(type: oscKind.sine, speed: rate5)
let rate3 = osc(type: oscKind.sine, speed: rate4)
let rate2 = osc(type: oscKind.sine, speed: rate3)
let rate1 = osc(type: oscKind.sine, speed: rate2)
let rate0 = osc(type: oscKind.saw, speed: rate1)
noise(octaves: rate1).write(o3)
noise(octaves: rate0).write(o4)

let movement = midi(channel: 1)
let gate = audio(band: audioBand.vol, min: movement)
noise(octaves: gate).write(o5)
let badGate = audio(band: movement)
noise(octaves: badGate).write(o6)
let deepMidi = midi(1, min: osc(speed: osc(speed: osc())))
noise(octaves: deepMidi).write(o7)

render(o0)
