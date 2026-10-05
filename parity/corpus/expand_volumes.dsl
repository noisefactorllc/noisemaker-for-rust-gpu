// Expander: read3d()/write3d() volume and geometry lanes. A reader that runs
// before its writer (previous frame), a re-export of a volume, a reader of the
// re-export, volume/geometry surface params, and volumeSize handoffs between
// chains (resolveVolume/resolveExport).
search synth3d, filter3d, render
read3d(vol1, geo1).render3d().write(o1)
noise3d(volumeSize: x32).write3d(vol0, geo0)
read3d(vol0, geo0).write3d(vol1, geo1)
read3d(vol1, geo1).flow3d().render3d().write(o0)
cellularAutomata3d(source: vol0, geoSource: geo0).write3d(vol2, geo2)
read3d(vol2, geo2).palette3d().write3d(vol3, geo3)
read3d(vol3, geo3).renderLit3d().write(o2)
render(o0)
