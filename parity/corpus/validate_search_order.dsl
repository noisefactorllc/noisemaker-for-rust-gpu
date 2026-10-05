search classicNoisedeck, synth, filter

noise().write(o0)
noise(loopAmp: 2).write(o1)
from(synth, noise(noiseType: 1)).write(o2)
perlin().blur().write(o3)
from(filter, noise()).write(o4)
cellNoise().noise().write(o5)

render(o0)
