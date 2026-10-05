search synth, filter

// double, single, empty and escaped strings
let a = "double"
let b = 'single'
let c = ""
let d = ''
let e = "say \"hi\" and \\ done"
let f = 'it\'s'
let g = "tab\tand é escapes stay raw"
let h = "escaped newline \
continues"; let afterDrift = 1
let i = "é → 😀 utf-16"
let j = """triple single line"""
let k = """
multi
line "quoted" text
"""
let l = """"""
let m = """a "" b"""
noise().text(text: "😀 emoji then col", font: 'mono').write(o0)
noise().text(text: """two
lines""").write(o1)
render(o0)
