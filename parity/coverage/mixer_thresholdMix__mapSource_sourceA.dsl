search mixer, synth

noise()
.write(o0)

solid(color: #000000)
.thresholdMix(tex: read(o0), mapSource: sourceA)
.write(o1)
