search mixer, synth

cell()
.write(o0)

noise(ridges: true)
.distortion(tex: read(o0), mapSource: sourceA)
.write(o1)
