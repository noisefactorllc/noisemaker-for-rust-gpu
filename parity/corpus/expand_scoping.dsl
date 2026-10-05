// Expander: per-chain and per-particle-pipeline scoping of global textures and
// their sizing params (stateSize, zoom, volumeSize) across several chains,
// including a chain that starts a second particle pipeline.
search points, synth, synth3d, render, filter
solid().pointsEmit(stateSize: 64).physarum().pointsRender().write(o0)
noise().pointsEmit(stateSize: 128).flow().pointsEmit(stateSize: 256).pointsRender().write(o1)
cellularAutomata(zoom: x8).write(o2)
cellularAutomata(zoom: x4).blur().write(o3)
reactionDiffusion(zoom: x2).write(o6)
noise3d(volumeSize: x32).render3d().write(o4)
cellularAutomata3d(volumeSize: x16).render3d().write(o5)
render(o0)
