search synth3d, filter3d, render

cellularAutomata3d(source: vol1, geoSource: geo2).write3d(vol0, geo0)
cellularAutomata3d(source: read3d(vol3), geoSource: read3d(geo4)).write3d(vol1, geo1)
cellularAutomata3d(source: none, geoSource: none).write3d(vol2, geo2)
cellularAutomata3d(source: foo, geoSource: bar).write3d(vol3, geo3)
cellularAutomata3d(source: read3d(vol9), geoSource: read3d(foo)).write3d(vol4, geo4)
cellularAutomata3d(source: "vol1", geoSource: "geo1").write3d(vol5, geo5)
cellularAutomata3d(source: 5, geoSource: o1).write3d(vol6, geo6)
cellularAutomata3d(source: vol12, geoSource: geo9).write3d(vol7, geo7)
read3d(vol0, geo0).write3d(vol1, geo1)
read3d(volA, geoB).write3d(vol2, geo2)
cellularAutomata3d().write3d(state, x)

render(o0)
