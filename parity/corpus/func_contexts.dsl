search synth, filter, mixer, classicNoisedeck

noise().write(o0)

solid(color: () => time).write(o1)
cellNoise(paletteOffset: () => time, palette: () => time).write(o2)
cellNoise(paletteOffset: [() => time, 0.5, 0.5]).write(o3)
noise().channel(channel: () => time).write(o4)
noise().text(text: () => time, font: () => time).write(o5)
media(imageSize: () => time).write(o6)
noise().blendMode(tex: () => time, mix: () => time).write(o7)
noise(octaves: osc(min: () => time, max: () => 1)).write(o0)
noise(octaves: midi(channel: () => 1)).write(o1)
noise(octaves: audio(band: () => 1, min: () => 0)).write(o2)
noise(octaves: () => time +).write(o3)

render(o0)
