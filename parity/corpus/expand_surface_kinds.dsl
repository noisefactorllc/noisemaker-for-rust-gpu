// Expander: surface arguments of every kind - agent surfaces bind global
// textures, while mesh and state references are not texture arguments (no
// input binding; the argument object reaches the uniforms).
search synth, mixer
noise().blendMode(tex: xyz0).write(o2)
noise().blendMode(tex: vel1).write(o3)
noise().blendMode(tex: rgba2).write(o4)
noise().blendMode(tex: mesh0).write(o5)
noise().blendMode(tex: time).write(o6)
noise().blendMode(tex: none).write(o7)
render(o2)
