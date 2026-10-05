search mixer, synth

noise(ridges: true, colorMode: mono)
.write(o0)

noise(ridges: true)
.centerMask(tex: read(o0), mix: -75)
.write(o1)
