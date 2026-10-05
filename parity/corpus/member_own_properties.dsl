search synth, filter

let n = noise(3)
let c = #ff8800
let label = n.name

noise(octaves: c.value.length, seed: label.length, scaleX: n.args.length)
  .write(o0)

render(o0)
