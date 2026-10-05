search synth, filter

noise(wrap: true, ridges: false).write(o0)
noise(wrap: 0, ridges: 3).write(o1)
noise(wrap: () => time > 1).write(o2)
noise(wrap: () => time >).write(o3)
noise(wrap: time, ridges: b1).write(o4)
noise(wrap: bogus).write(o5)
noise(wrap: "yes").write(o6)
noise(wrap: #fff).write(o7)
noise(wrap: osc(type: oscKind.sine)).write(o0)
noise(ridges: oscKind.saw).write(o1)

render(o0)
