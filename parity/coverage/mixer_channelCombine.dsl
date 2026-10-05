search mixer, synth, filter

noise(ridges: true, colorMode: mono)
.write(o0)

perlin(colorMode: mono)
.write(o1)

gradient(type: linear)
.write(o2)

channelCombine(rTex: read(o0), gTex: read(o1), bTex: read(o2))
.write(o3)
