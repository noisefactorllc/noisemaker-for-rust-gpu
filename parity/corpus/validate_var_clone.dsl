search synth, filter

let nan = 0 / 0
let inf = 1 / 0
let ninf = -1 / 0
let negzero = -0
let big = 1000000 * 1000000 * 1000000 * 1000000

noise(octaves: nan, seed: inf, speed: ninf).write(o0)
noise(octaves: negzero, scaleX: big).write(o1)
if (nan) { noise().write(o2) } elif (inf) { noise().write(o3) } elif (negzero) { noise().write(o4) }
return nan
return inf
return negzero
return big

render(o0)
