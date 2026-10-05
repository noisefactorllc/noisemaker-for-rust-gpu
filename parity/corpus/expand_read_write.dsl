// Expander: read() starters, chainable write(), final blits after inline
// writes to non-output surfaces, write(none), read-then-write no-ops, and
// pipeline-surface arguments without a 2D input.
search synth, filter
noise().write(xyz0)
noise(seed: 2).write(mesh0)
read(o0).write(o0)
read(xyz0).blur().write(o1)
noise(seed: 3).write(o2).blur().write(o3)
noise(seed: 4).write(none)
read(o3).write(none).write(o4)
read(xyz0).lighting().write(o5)
render(o3)
