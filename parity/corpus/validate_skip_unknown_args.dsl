search synth, filter

noise(_skip: true).write(o0)
noise(_skip: false).write(o1)
noise(_skip: 1).write(o2)
noise(octaves: 2, bogus: 1, other: "x").write(o3)
noise().blur(_skip: true, radius: 2).write(o4)
noise().blur(_skip: time).write(o5)
noise(octaves: 2, octaves: 3).write(o6)

render(o0)
