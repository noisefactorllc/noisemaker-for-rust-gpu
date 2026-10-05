// before the search directive
search synth, filter
// leading comment of the first plan
noise()
  // comment before a step
  .blur()
  /* block comment before a write */
  .write(o0)

// leading comment of a plan whose head is a read
read(o0)
  // comment before a step of it
  .blur()
  .write(o1)

if (1) {
  noise().write(o2)
}

// leading comment of the render directive
render(o0)
// trailing comment one
/* trailing comment two */
