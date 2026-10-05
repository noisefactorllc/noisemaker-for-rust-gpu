search filter, synth, classicNoisedeck

noise().write(o0)
from(classicNoisedeck, noise()).write(o1)
from(synth, noise(seed: 3)).from(filter, blur(radiusX: 2)).write(o2)
from(bogus, noise()).write(o3)
noise().from(synth, blur()).write(o4)
bogus().write(o5)
from(classicNoisedeck, noise(loopAmp: 3)).write(o6)
noise().blendMode(tex: from(synth, solid()), mode: 2).write(o7)

render(o0)
