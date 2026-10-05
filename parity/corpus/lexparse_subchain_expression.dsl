search synth, filter

noise().write(o0)
let grouped = noise().subchain(name: "in expression") { .blur() }
let named = noise().subchain(foo: "x") { .blur() }.blur()
render(o0)
