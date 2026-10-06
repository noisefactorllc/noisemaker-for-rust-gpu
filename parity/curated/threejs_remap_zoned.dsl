search synth
noise(scaleX: 30, scaleY: 30, seed: 3).write(o1)
remap(zoneCount: 1, zone0_count: 4, zone0_v0: [0.1, 0.1, 0.9, 0.1], zone0_v1: [0.9, 0.9, 0.1, 0.9], zone0_alpha: 100, zone0_tex: read(o1), bgColor: #102030, bgAlpha: 1).write(o0)
render(o0)
