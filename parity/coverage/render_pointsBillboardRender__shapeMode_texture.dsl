search synth, render
shape(seed: 1).write(o1)
solid().pointsEmit(stateSize: 128).pointsBillboardRender(shapeMode: texture, tex: read(o1)).write(o0)
render(o0)
