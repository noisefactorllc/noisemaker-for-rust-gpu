// Expander: the midiNoteGrid input, an effect's own outputTex as input
// (feedback copies), and selfTex in a chain without a 2D write target (a
// write3d chain), which reads the current input or global_inputTex.
search synth, synth3d, filter
roll().write(o4)
noise3d().convolutionFeedback().write3d(vol0, geo0)
noise().convolutionFeedback().write3d(vol1, geo1)
noise().motionBlur().write(o5)
render(o4)
