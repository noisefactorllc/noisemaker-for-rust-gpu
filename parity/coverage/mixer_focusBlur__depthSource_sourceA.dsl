search synth, mixer
noise(seed: 1, scaleX: 50, scaleY: 50).write(o0)
gradient(seed: 1).focusBlur(tex: o0, depthSource: sourceA).write(o1)
render(o1)
