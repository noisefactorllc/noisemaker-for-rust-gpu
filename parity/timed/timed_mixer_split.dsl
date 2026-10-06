search mixer, synth

noise(ridges: true, colorMode: mono)
.write(o0)

noise(seed: 2, ridges: true)
.split(tex: read(o0), softness: 1)
.write(o1)
