search synth
// é 😀 comment with UTF-16 surrogate pairs
let s = "😀😀"; let t = 'é'; let u = """😀
😀"""; noise(seed: 1).write(o0)
noise(seed: 2) /* 😀 */ .write(o1)
render(o0)
