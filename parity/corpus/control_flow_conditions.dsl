search synth, filter

let depth = 4
let fast = osc(speed: 2)
let shape = oscKind.sine
let lost = nowhere.path

noise().write(o0)

if (1) {
  noise(octaves: depth).write(o1)
} elif (depth) {
  solid().write(o1)
} elif (time) {
  noise().blur().write(o2)
} elif (bogus) {
  solid(alpha: 0.5).write(o1)
} else {
  noise(seed: 7).write(o3)
}

if (0) { noise().write(o1) } elif (2 - 2) { noise().write(o2) } elif (-1) { noise().write(o3) } elif (0.5) { noise().write(o4) }
if (0 / 0) { noise().write(o1) } elif (1 / 0) { noise().write(o2) } elif (true) { noise().write(o3) } elif (false) { noise().write(o4) }
if (frame) { noise().write(o1) } elif (fast) { noise().write(o2) } elif (shape) { noise().write(o3) } elif (lost) { noise().write(o4) }
if (oscKind.sine) { noise().write(o1) } elif (oscKind.tri) { noise().write(o2) } elif (foo.bar) { noise().write(o3) } elif (oscKind) { noise().write(o4) }
if ("text") { noise().write(o1) } elif (#ff0000) { noise().write(o2) } elif (o1) { noise().write(o3) } elif (noise()) { noise().write(o4) }
if (noise().blur()) { noise().write(o1) } elif (from(synth, noise()).blur()) { noise().write(o2) } elif (osc(speed: 1)) { noise().write(o3) }

render(o0)
