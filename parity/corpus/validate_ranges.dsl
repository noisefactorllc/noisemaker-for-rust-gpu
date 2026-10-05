search synth, filter

noise(octaves: 100, scaleX: -5, seed: 0.5, speed: 1000).write(o0)
noise(octaves: true, wrap: 0, ridges: 2).write(o1)
noise(octaves: false, scaleY: 1000).write(o2)
solid(alpha: 1.5).write(o3)
noise(octaves: 3 * 2 + 1, scaleX: 100 / 3).write(o4)
noise(octaves: Math.PI, seed: -Math.PI).write(o5)
noise(speed: -100, scaleX: 1).write(o6)
noise(octaves: (((4)))).write(o7)

render(o0)
