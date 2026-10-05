search synth

let empty = []
let one = [1]
let three = [1, 2, 3]
let nested = [[1], [2, 3], []]
let exprs = [-1, 2 + 0.5, Math.PI, (4)]
let mixed = [o0, "a", #fff, true, false, foo.bar, () => time, 0]
noise(pos: [0.05, 0.1, 0.45, 0.95]).write(o0)
render(o0)
