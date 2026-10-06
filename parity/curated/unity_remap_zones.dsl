search synth, filter
noise(seed: 1, ridges: true).write(o1)
remap(bgColor: #336699, bgAlpha: 1, zoneCount: 1, zone0_tex: read(o1), zone0_count: 3, zone0_alpha: 1, zone0_v0: [0.1, 0.1, 0.9, 0.1], zone0_v1: [0.5, 0.9, 0, 0]).write(o0)
render(o0)
