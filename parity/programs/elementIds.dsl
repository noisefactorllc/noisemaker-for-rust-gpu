search synth, filter
noise(seed: 7, _id: "Noise_1").blur(radiusX: 2, _id: "Blur_1").write(o0, _id: "Output_1")
read(o0, _id: "Read_1").invert(_id: "Invert_1").write(o1, _id: "Output_2")
render(o1)
