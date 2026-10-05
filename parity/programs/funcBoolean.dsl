search synth

let flicker = () => frame % 2 > 0

noise(ridges: flicker, octaves: 3).write(o0)

render(o0)
