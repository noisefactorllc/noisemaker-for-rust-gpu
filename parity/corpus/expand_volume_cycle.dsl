// Expander: volumes exported in a cycle (each chain reads the other's
// export), direct re-exports that form a cycle (resolveExport's visited
// guard), and write3d targets of "none".
search synth3d, filter3d, render
read3d(vol4, geo4).flow3d().write3d(vol5, geo5)
read3d(vol5, geo5).palette3d().write3d(vol4, geo4)
read3d(vol7, geo7).write3d(vol8, geo8)
read3d(vol8, geo8).write3d(vol7, geo7)
noise3d(volumeSize: x16).write3d(none, geo6)
noise3d(volumeSize: x64).write3d(vol6, none)
read3d(vol6, geo6).render3d().write(o0)
render(o0)
