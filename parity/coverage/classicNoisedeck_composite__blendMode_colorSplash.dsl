search synth, filter, classicNoisedeck
noise(seed:1,scaleX:50,scaleY:50).write(o0)
gradient(seed:1).composite(tex:o0, blendMode: colorSplash).write(o1)
render(o1)
