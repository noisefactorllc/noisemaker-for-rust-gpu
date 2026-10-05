search synth, filter
noise(seed: 1, scaleX: 50, scaleY: 50).edge(kernel: contour, level: 40, channel: luminance).write(o0)
render(o0)
