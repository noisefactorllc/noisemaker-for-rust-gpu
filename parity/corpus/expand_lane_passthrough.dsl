// Expander: skipped steps and subchain markers pass the 2D, 3D and geometry
// lanes through; read3d of a plain name; 3D consumers without a 3D input
// fall back to their pipeline names.
search synth, synth3d, filter3d, render
noise3d().palette3d(_skip: true).render3d().write(o0)
noise3d().subchain(name: "vol") { .palette3d() }.render3d().write(o1)
noise3d(volumeSize: x16).subchain(name: "outer") { .palette3d(_skip: true) }.render3d().write(o2)
read3d(foo, geo0).render3d().write(o3)
noise().render3d().write(o4)
render(o0)
