search synth, filter, mixer

// A member argument rewrites its AST node's path; the outer step's rawKwargs
// (the original AST) shows the rewritten path.
noise().blendMode(tex: noise().channel(channel: foo.bar), mix: 0.5).write(o0)

// The nested step renames its own kwargs copy; the outer rawKwargs keeps the AST.
noise().blendMode(tex: noise(noiseType: 2), mixAmt: 0.3).write(o1)

// A variable's stored call shares its argument nodes with every call of it: the
// rewritten path persists in the variable and shows in a later return value.
let ch = channel(channel: foo.bar)
noise().ch().write(o2)
return ch

// rawKwargs of a direct member argument.
noise().channel(channel: g.x).write(o3)

render(o0)
