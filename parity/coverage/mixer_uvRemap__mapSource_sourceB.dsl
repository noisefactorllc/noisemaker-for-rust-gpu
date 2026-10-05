search mixer, synth

pattern()
.write(o0)

noise(ridges: true)
.uvRemap(tex: read(o0), scale: 25, mapSource: sourceB)
.write(o1)
