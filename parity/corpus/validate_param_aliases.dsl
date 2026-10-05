search synth, filter, mixer, classicNoisedeck

// Deprecated parameter names are renamed in place (S007); rawKwargs shows the rename.
noise(noiseType: 3, xScale: 20, yScale: 30, loopAmp: 5).write(o0)

// Old and new name together: the new name wins and the old key is dropped.
noise(type: 4, noiseType: 2).write(o1)

// Aliases on a filter, followed by unknown arguments in their final key order.
noise().palette(paletteIndex: 3, paletteOffset: 0.5, bogus: 1).write(o2)

// A variable's call: the merged kwargs object is renamed, the call site's is not.
let n = noise(xScale: 10)
n(yScale: 20).write(o3)

// Inside a surface argument: the nested step renames its own kwargs copy.
noise().blendMode(tex: noise(noiseType: 2).palette(paletteRotation: 1), mixAmt: 0.5).write(o4)

render(o0)
