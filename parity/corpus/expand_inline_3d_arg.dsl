// Expander: an inline 3D generator as a 2D surface argument registers no 2D
// output, so the input binds undefined; compileGraph's allocator then throws.
search synth, mixer, synth3d
noise().blendMode(tex: noise3d()).write(o1)
render(o1)
