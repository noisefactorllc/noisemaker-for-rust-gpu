// dropped: comments before the search directive
/* also dropped */
search synth, filter

// leading comment on a let
let depth = 4
/* block comment
   spanning lines */ noise(octaves: depth)
  // before the dot
  .blur()
  . /* after the dot */ blur(radiusX: 2)
  // before write
  .write(o0)
// comment; then a semicolon
;
noise() /* inline */ .write(o1)
noise()
  // before a subchain
  .subchain(name: "sc") {
    // inside the body
    .blur()
    . // after a body dot
    blur()
    // dropped before the closing brace
  }
  .write(o2)
/*/ not closed by its own slash */
/**/
// before render
render(o0) // trailing after render
/* trailing block */
// last line comment
