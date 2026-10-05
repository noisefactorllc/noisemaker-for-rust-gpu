search synth, filter, render

noise()
  .subchain("positional name") {
    .blur()
  }
  .subchain(name: "named", id: "sc1") {
    .blur()
    .blur(radiusX: 3)
  }
  .subchain() {
    .blur()
  }
  .subchain(id: "only-id") { .blur() }
  .subchain(name: "", id: "") { .blur() }
  .subchain(name: "trailing comma",) { .blur() }
  .write(o0)
noise()
  .subchain(nme: "typo", name: "ok") { .blur() }
  .subchain(name: "first", name: "second") { .blur() }
  .subchain(name: "a" id: "b") { .blur() }
  .subchain(foo: "x" name: "a" name: "b" id: "s", bar: "y") { .blur() }
  .write(o1)
render(o0)
