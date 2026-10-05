search synth, filter, classicNoisedeck

let wobble = () => Math.sin(time) * 4
let flicker = () => frame % 2 > 0

noise(octaves: wobble, ridges: flicker, seed: () => seed + 1, speed: () => time)
  .blur(radiusX: () => time % 10, radiusY: () => mouse.y * 5)
  .write(o0)

solid(alpha: () => (time % 1)).write(o1)

cellNoise(scale: () => 50 + Math.cos(time) * 25, shape: () => frame % 3, speed: () => 2)
  .write(o2)

render(o0)
