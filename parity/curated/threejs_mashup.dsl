search synth, filter, mixer
noise(ridges: true, scaleX: 40, scaleY: 40, seed: 3).write(o0)
solid(color: #ee3322).write(o1)
solid(color: #2266cc).write(o2)
gradient().write(o3)
mashup(source: read(o3), layer0_tex: read(o0), layer1_tex: read(o1), layer2_tex: read(o2), layers: 3, smoothness: 0.3).write(o4)
render(o4)
