search synth, filter

noise().text(text: "Hello", font: serif, justify: left, style: "Bold").write(o0)
noise().text(text: 5).write(o1)
noise().text(font: bogus).write(o2)
noise().text(justify: right).write(o3)
noise().text(justify: left, font: monospace).write(o4)
noise().text(font: "Comic Sans", justify: "right").write(o5)
noise().text(text: hello).write(o6)
noise().text(size: "big", color: "white").write(o7)
noise().text(text: """multi
line""").write(o0)
noise().text(text: 'single \'quoted\'', style: "").write(o1)

render(o0)
