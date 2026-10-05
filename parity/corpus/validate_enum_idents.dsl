search synth, filter

noise(type: simplex).write(o0)
noise(type: bogus).write(o1)
noise(type: synth.noise.type.simplex).write(o2)
noise(type: oscKind.saw).write(o3)
noise(colorMode: hsv, loopOffset: circle).write(o4)
noise(octaves: oscKind.noise2d).write(o5)
noise(octaves: palette.sherbet).write(o6)
noise(seed: time, speed: mouse).write(o7)
sacredGeometry(geometry: seed).write(o0)
sacredGeometry(geometry: flower).write(o1)
sacredGeometry(geometry: synth.sacredGeometry.geometry.metatron).write(o2)
noise(type: synth.noise.type.bogus).write(o3)
noise(octaves: synth.noise.type).write(o4)
noise(octaves: midiZone.upper, seed: audioBand.raw).write(o5)

render(o0)
