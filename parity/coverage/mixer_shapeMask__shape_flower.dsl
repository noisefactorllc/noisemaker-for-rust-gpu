search mixer, synth

noise(ridges: true, colorMode: mono)
.write(o0)

noise(ridges: true)
.shapeMask(tex: read(o0), shape: flower)
.write(o1)
