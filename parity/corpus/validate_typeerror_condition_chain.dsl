search synth, filter

noise().write(o0)
if (osc(type: oscKind.sine).blur()) {
  noise().write(o1)
}

render(o0)
