search synth, filter

noise()
  .subchain(name: "group", id: "g1") {
    .blur(radiusX: 2)
    .bogus()
    .noise()
    .read(o1)
  }
  .write(o0)

noise()
  // before the subchain
  .subchain("named") {
    // inside the subchain
    .blur(noiseType: 1)
  }
  // after the subchain
  .write(o1)

noise()
  .subchain(id: "only-id") { .blur() }
  .subchain(name: "", id: "") { .blur() }
  .subchain() {
    .blur()
    .blur(radiusY: 3)
  }
  .write(o2)

render(o0)
