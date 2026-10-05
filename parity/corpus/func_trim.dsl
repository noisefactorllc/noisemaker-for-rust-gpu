search synth

noise(octaves: () => ﻿time * 4, seed: () =>  frame , speed: () => 　time ).write(o0)
noise(octaves: () => time).write(o1)
noise(octaves: () =>  time , ridges: () =>  frame ).write(o2)

render(o0)
