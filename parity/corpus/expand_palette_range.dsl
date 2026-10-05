// Expander: classicNoisedeck palette expansion at the table bounds and out of
// range (0, negative, past the last preset: no expansion).
search classicNoisedeck
fractal(palette: 0).write(o0)
fractal(palette: -2).write(o1)
fractal(palette: 56).write(o2)
fractal(palette: 55).write(o3)
noise(palette: 1).write(o4)
cellNoise(palette: 12).shapeMixer(palette: 40).write(o5)
render(o5)
