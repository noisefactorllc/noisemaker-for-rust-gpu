search mixer, synth

noise(ridges: true, colorMode: mono)
.write(o0)

noise(ridges: true)
.patternMix(tex: read(o0), type: radialLines)
.write(o1)
