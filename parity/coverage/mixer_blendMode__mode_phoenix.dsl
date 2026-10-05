search mixer, synth

noise(ridges: true, colorMode: mono)
.write(o0)

perlin()
.blendMode(tex: read(o0), mode: phoenix)
.write(o1)
