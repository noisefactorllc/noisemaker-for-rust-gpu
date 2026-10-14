search synth, filter, mixer, classicNoisedeck

// Canonical parameter names compile; rawKwargs keeps the call site's keys.
noise(type: 3, scaleX: 20, scaleY: 30, speed: 5).write(o0)

// Arguments land on the compiled step in their source key order.
noise(octaves: 4).write(o1)

// A filter call, followed by unknown arguments in their final key order.
noise().palette(index: 3, offset: 0.5, bogus: 1).write(o2)

// A variable's call: the merged kwargs object carries the call site's keys.
let n = noise(scaleX: 10)
n(scaleY: 20).write(o3)

// Inside a surface argument: the nested step carries its own kwargs copy.
noise().blendMode(tex: noise(type: 2).palette(rotation: 1), mix: 0.5).write(o4)

render(o0)
