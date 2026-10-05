// Expander: compile-time defines from step args - a let variable (an
// {_varRef, value} argument), an automation object (stringified into the
// program variant name), and defaults - each a distinct program variant.
search synth, filter
let t = 4
noise(type: t, loopOffset: 120).write(o0)
noise(type: osc(type: oscKind.sine)).write(o1)
noise(type: t).blur().write(o2)
render(o0)
