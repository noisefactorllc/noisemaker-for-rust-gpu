search mixer, synth

noise(seed: 1, ridges: true)
.write(o0)

perlin()
.applyMode(tex: read(o0))
.write(o1)
