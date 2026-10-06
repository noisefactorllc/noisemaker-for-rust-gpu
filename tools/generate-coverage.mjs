#!/usr/bin/env node
// generate-coverage.mjs — generate the effect × mode coverage corpus.
//
// For every effect of the catalog this writes parity/coverage/<ns>_<func>.dsl, a
// program that runs the effect with its defaults, and one variant per discrete
// branch of the effect: every value of every choice-valued parameter (compile-time
// defines included) and the flipped value of every boolean parameter:
//
//   parity/coverage/<ns>_<func>__<param>_<value>.dsl
//
// The base program is the effect's own `defaultProgram` when the definition has
// one; otherwise a program built for the effect's role in a chain (generator,
// 2D filter, mixer, particle stage, volume generator/processor/renderer, mesh,
// loop). Every generated program is compiled by the unmodified reference
// compileGraph, and a variant is kept only when the reference graph carries the
// requested value on one of the effect's passes (as a uniform or as a define), so
// each kept fixture provably exercises its branch. Dropped candidates are listed
// with the reason on stderr.
//
// Informative fixtures: a golden is parity evidence only when it shows
// structure (parity/sweep.py: not uniform, at most 99% one colour). Where an
// effect's generated program renders a degenerate golden (a mixer whose second
// input defaults to black, a branch whose shapes never meet, a camera that sees
// nothing, particles that saturate the canvas), the effect takes an
// author-curated program as its base (CURATED_BASES: the shared program pool,
// the sibling ports' curated programs, or another effect's defaultProgram) or
// the variant takes the inputs and arguments that make its branch visible
// (VARIANT_FIXES), each checked against the reference's render.
//
// The timed tier: every effect that evolves across frames also gets
// parity/timed/timed_<ns>_<func>.dsl, its base program, which the sweep renders
// with the timed protocol (fresh state, render(((frame + 1) / 600) % 1) per
// frame, samples after the frame counts of the manifest's protocol). An effect
// evolves when the catalog shows any of:
//   state     -- a texture the effect reads at or before the pass that writes
//                it (ping-pong state and feedback surfaces: the read sees the
//                previous frame);
//   particles -- a points-namespace effect or a points* render stage;
//   sim       -- the definition's `sim` tag;
//   time      -- a shader reads the time, deltaTime or frame uniform (a member
//                access such as u.time, a bare uniform of that name, a packed
//                uniformLayout slot, or a packed uniforms.data[] unpack into a
//                variable of that name, as the reference's packed-layout parser
//                reads it);
//   onUpdate  -- a native per-frame lifecycle hook.
// Oscillator automation gets parity/timed/timed_osc_<kind>.dsl, one program per
// oscillator kind modulating an otherwise static noise. Every shared or curated
// program (parity/programs, parity/curated: among them the flagship hero, the
// sibling ports' hero/north_star) whose reference graph runs an effect with
// state, particles, a simulation or a per-frame hook gets
// parity/timed/timed_<name>.dsl (its .obj sidecar copied beside it). parity/timed/
// manifest.json lists every timed case with its effect, its reasons and the
// shared sample protocol.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node tools/generate-coverage.mjs [--check]
//
// --check regenerates in memory and exits 1 when parity/coverage or
// parity/timed differs.

import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const CATALOG = join(ROOT, 'crates', 'noisemaker-effects', 'catalog')
const OUT = join(ROOT, 'parity', 'coverage')
const TIMED_OUT = join(ROOT, 'parity', 'timed')

// The timed tier's sample protocol: samples every second for five seconds and
// after the early frame counts.
const TIMED_PROTOCOL = { runSeconds: 5, sampleEvery: 1, sampleFrames: [1, 2, 4, 10, 30] }
const OSC_KINDS = ['sine', 'tri', 'saw', 'sawInv', 'square', 'noise']

// Effect -> its base program: a curated program file, or another effect's
// defaultProgram. Variants place their argument into the effect's own call.
const CURATED_BASES = {
  // The mixers' second input defaults to none (black): multiply, darken,
  // burn, subtract and similar modes rendered uniform black.
  'classicNoisedeck.coalesce': { file: 'parity/programs/coalesce.dsl' },
  'classicNoisedeck.composite': { file: 'parity/programs/composite.dsl' },
  'classicNoisedeck.shapeMixer': { file: 'parity/programs/shapeMixer.dsl' },
  'mixer.focusBlur': { file: 'parity/programs/focusBlur.dsl' },
  'mixer.thresholdMix': { file: 'parity/programs/thresholdMix.dsl' },
  // The defaultProgram routes no zone: a uniform background.
  'synth.remap': { file: 'parity/curated/babylonjs_remap_zones.dsl' },
  // A solid colour is legitimately flat; its evidence is the texture
  // effect's defaultProgram, which renders a texture over solid's colour.
  'synth.solid': { defaultProgramOf: 'filter.texture' },
  // The source* = brightness/darkness/... branches read the input texture,
  // which the generated program left empty.
  'synth.reactionDiffusion': { text: 'search synth\nnoise(seed: 3, scaleX: 30, scaleY: 30).write(o1)\nreactionDiffusion(tex: read(o1)).write(o0)\nrender(o0)\n' },
  // The piano roll scrolls by speed * deltaTime: at the default speed the
  // shift stays under a texel and the held notes never leave the left edge.
  'synth.roll': { text: 'search synth\nroll(speed: 2.5).write(o0)\nrender(o0)\n' },
  // 128 x 128 particles leave the attractors at fewer than 1% of the pixels.
  'points.attractor': { text: 'search synth, points, render\nsolid().pointsEmit(stateSize: 512).attractor().pointsRender().write(o0)\nrender(o0)\n' }
}

// Variant fixture -> what makes its branch visible: `program` replaces the
// generated program (with the branch's argument already in place), `args`
// adds arguments to named calls of the generated program.
const PERSPECTIVE_PARTICLES = (render) =>
  `search synth, points, render\nsolid().pointsEmit(stateSize: 512).attractor().${render}(viewMode: perspective).write(o0)\nrender(o0)\n`
const VARIANT_FIXES = {
  // shapeB (scale 27) sits inside shapeA's hollow: the shapes never meet.
  'classicNoisedeck_shapes3d__blendMode_intersect.dsl': { args: { shapes3d: [['shapeBScale', '70']] } },
  'classicNoisedeck_shapes3d__blendMode_smoothAMinusB.dsl': { args: { shapes3d: [['shapeBScale', '70']] } },
  'classicNoisedeck_shapes3d__blendMode_smoothIntersect.dsl': { args: { shapes3d: [['shapeBScale', '70']] } },
  // The palette is near black over the default iteration band.
  'classicNoisedeck_fractal__palette_summoning.dsl': { args: { fractal: [['repeatPalette', '4']] } },
  // Seed 1 puts the camera inside the sine field.
  'classicNoisedeck_noise3d__type_sine.dsl': { args: { noise3d: [['seed', '3']] } },
  // Speckle over a flat solid is flat.
  'filter_texture__mode_speckle.dsl': { program: 'search filter, synth\nnoise(seed: 1, scaleX: 50, scaleY: 50).texture(alpha: 0.75, mode: speckle).write(o0)\nrender(o0)\n' },
  // A full-stride clamped wormhole smears the edge colour over the image.
  'filter_wormhole__wrap_clamp.dsl': { args: { wormhole: [['stride', '0.25']] } },
  // Texture sprites need a texture.
  'render_pointsBillboardRender__shapeMode_texture.dsl': { program: 'search synth, render\nshape(seed: 1).write(o1)\nsolid().pointsEmit(stateSize: 128).pointsBillboardRender(shapeMode: texture, tex: read(o1)).write(o0)\nrender(o0)\n' },
  // A flat particle layout seen in perspective collapses to a point.
  'render_pointsBillboardRender__viewMode_perspective.dsl': { program: PERSPECTIVE_PARTICLES('pointsBillboardRender') },
  'render_pointsRender__viewMode_perspective.dsl': { program: PERSPECTIVE_PARTICLES('pointsRender') },
  // A million or more particles saturate a 256 x 256 trail.
  'render_pointsEmit__stateSize_x1024.dsl': { args: { pointsRender: [['intensity', '5'], ['density', '5']] } },
  'render_pointsEmit__stateSize_x2048.dsl': { args: { pointsEmit: [['layout', 'spiral']], pointsRender: [['intensity', '5'], ['density', '5']] } },
  // These fractal types leave nothing above the default iso threshold.
  'synth3d_flythrough3d__type_mandelbox.dsl': { args: { flythrough3d: [['power', '2'], ['voiSize', '2']], render3d: [['threshold', '0.2']] } },
  'synth3d_fractal3d__type_juliaCube.dsl': { args: { fractal3d: [['juliaX', '-80'], ['juliaY', '60'], ['juliaZ', '40'], ['power', '10']], render3d: [['threshold', '0.1']] } },
  'synth3d_fractal3d__type_mandelcube.dsl': { args: { fractal3d: [['power', '2'], ['bailout', '8'], ['iterations', '12']], render3d: [['threshold', '0.1']] } },
  // Resetting every frame leaves the small seed cube at the default size.
  'synth3d_reactionDiffusion3d__resetState_true.dsl': { args: { reactionDiffusion3d: [['volumeSize', 'x16'], ['colorMode', 'gradient']], render3d: [['threshold', '0.1']] } }
}

// MIDI input for the effects that visualize it (a sidecar next to every
// fixture of the effect, delivered by the minter and nm-render after the
// program loads): notes held on every channel across the piano-roll range.
function heldNotes () {
  const messages = []
  for (let ch = 0; ch < 16; ch++) {
    for (const key of [40, 50, 60, 70, 80]) messages.push([0x90 | ch, key + (ch % 5), 40 + ch * 5])
  }
  return JSON.stringify({ messages }) + '\n'
}
const MIDI_SIDECARS = { 'synth.roll': heldNotes }

function applyVariantFix (fname, src) {
  const fix = VARIANT_FIXES[fname]
  if (!fix) return src
  if (fix.program) return fix.program
  let out = src
  for (const [func, args] of Object.entries(fix.args || {})) {
    out = withCallArgs(out, func, args)
    if (out === null) throw new Error(`${fname}: cannot place ${func} arguments`)
  }
  return out
}
const TIME_UNIFORMS = ['time', 'deltaTime', 'frame']

console.log = (...args) => process.stderr.write(args.map(String).join(' ') + '\n')
console.warn = () => {}
console.info = () => {}

function sortedDirs (dir) {
  return readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory() && d.name !== 'share').map(d => d.name).sort()
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/

function isStarterEffect (def) {
  const passes = def.passes || []
  if (passes.length === 0) return true
  const pipelineInputs = new Set(['inputTex', 'inputTex3d', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7'])
  return !passes.some(p => p.inputs && Object.values(p.inputs).some(v => pipelineInputs.has(v)))
}

function passInputs (def) {
  return new Set((def.passes || []).flatMap(p => Object.values(p.inputs || {})))
}

// The chain the effect runs in when its definition has no defaultProgram.
function baseProgram (ns, func, def) {
  const inputs = passInputs(def)
  const globals = def.globals || {}
  const surfaceParams = Object.entries(globals).filter(([, g]) => g.type === 'surface')
  const needsSurface = surfaceParams.filter(([, g]) => g.default !== 'inputTex' && g.default !== 'none')
  if (ns === 'synth3d') {
    return { search: ['synth3d', 'render'], text: `${func}(@ARGS@).render3d().write(o0)` }
  }
  if (ns === 'filter3d') {
    return { search: ['synth3d', 'filter3d', 'render'], text: `noise3d().${func}(@ARGS@).render3d().write(o0)` }
  }
  if (ns === 'render' && inputs.has('inputTex3d')) {
    return { search: ['synth3d', 'render'], text: `noise3d().${func}(@ARGS@).write(o0)` }
  }
  if (ns === 'render' && (func === 'pointsRender' || func === 'pointsBillboardRender')) {
    return { search: ['synth', 'render'], text: `solid().pointsEmit(stateSize: 128).${func}(@ARGS@).write(o0)` }
  }
  if (ns === 'render' && func === 'pointsEmit') {
    return { search: ['synth', 'render'], text: `solid().pointsEmit(@ARGS@).pointsRender().write(o0)` }
  }
  if (ns === 'render' && (func === 'meshRender')) {
    return { search: ['render'], text: `meshLoader().meshRender(@ARGS@).write(o0)` }
  }
  if (ns === 'render' && func === 'loopBegin') {
    return { search: ['synth', 'filter', 'render'], text: `noise(seed: 1, scaleX: 50, scaleY: 50).loopBegin(@ARGS@).blur().loopEnd().write(o0)` }
  }
  if (ns === 'render' && func === 'loopEnd') {
    return { search: ['synth', 'filter', 'render'], text: `noise(seed: 1, scaleX: 50, scaleY: 50).loopBegin().blur().loopEnd(@ARGS@).write(o0)` }
  }
  if (ns === 'points') {
    return { search: ['synth', 'points', 'render'], text: `solid().pointsEmit(stateSize: 128).${func}(@ARGS@).pointsRender().write(o0)` }
  }
  if (needsSurface.length > 0) {
    const surfaceArgs = needsSurface.map(([k]) => `${k}: o0`).join(', ')
    const starter = isStarterEffect(def)
    const head = starter ? `${func}(${surfaceArgs}@SEP@@ARGS@)` : `gradient(seed: 1).${func}(${surfaceArgs}@SEP@@ARGS@)`
    return { search: ['synth', ns], text: `noise(seed: 1, colorMode: 1).write(o0)\n${head}.write(o1)`, render: 'o1' }
  }
  if (isStarterEffect(def)) {
    return { search: [ns], text: `${func}(@ARGS@).write(o0)` }
  }
  return { search: ['synth', ns], text: `noise(seed: 1, scaleX: 50, scaleY: 50).${func}(@ARGS@).write(o0)` }
}

function templateSource (tpl, args) {
  const argText = args.map(([k, v]) => `${k}: ${v}`).join(', ')
  const search = [...new Set(tpl.search)].join(', ')
  const body = tpl.text
    .replace('@SEP@', argText ? ', ' : '')
    .replace('@ARGS@', argText)
  return `search ${search}\n${body}\nrender(${tpl.render || 'o0'})\n`
}

// Insert or replace keyword arguments in the first call of `func` in `src`.
function withCallArgs (src, func, args) {
  if (!args.length) return src
  const re = new RegExp(`(^|[^A-Za-z0-9_])${func}\\s*\\(`, 'g')
  const m = re.exec(src)
  if (!m) return null
  const open = m.index + m[0].length - 1
  let depth = 0
  let close = -1
  let quote = null
  for (let i = open; i < src.length; i++) {
    const c = src[i]
    if (quote) {
      if (c === '\\') { i++; continue }
      if (c === quote) quote = null
      continue
    }
    if (c === '"' || c === "'") { quote = c; continue }
    if (c === '(') depth++
    else if (c === ')') { depth--; if (depth === 0) { close = i; break } }
  }
  if (close < 0) return null
  const inner = src.slice(open + 1, close)
  // Split the existing argument list at depth-0 commas.
  const items = []
  let cur = ''
  depth = 0
  quote = null
  for (let i = 0; i < inner.length; i++) {
    const c = inner[i]
    if (quote) {
      cur += c
      if (c === '\\') { cur += inner[++i]; continue }
      if (c === quote) quote = null
      continue
    }
    if (c === '"' || c === "'") { quote = c; cur += c; continue }
    if (c === '(' || c === '[' || c === '{') depth++
    if (c === ')' || c === ']' || c === '}') depth--
    if (c === ',' && depth === 0) { items.push(cur); cur = ''; continue }
    cur += c
  }
  if (cur.trim()) items.push(cur)
  const kept = items.map(s => s.trim()).filter(Boolean)
    .filter(item => !args.some(([k]) => new RegExp(`^${k}\\s*:`).test(item)))
  if (kept.some(item => !/^[A-Za-z_][A-Za-z0-9_]*\s*:/.test(item))) return null // positional args: skip
  const merged = [...kept, ...args.map(([k, v]) => `${k}: ${v}`)]
  return src.slice(0, open + 1) + merged.join(', ') + src.slice(close)
}

function choiceToken (name, value) {
  if (IDENT.test(name)) return name
  const sanitized = name.replace(/\s+(.)/g, (_, c) => c.toUpperCase()).replace(/\s+/g, '').replace(/[^a-zA-Z0-9_]/g, '')
  if (IDENT.test(sanitized)) return sanitized
  return String(value)
}

function fileToken (s) {
  return String(s).replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'x'
}

function sameValue (a, b) {
  if (typeof a === 'boolean' || typeof b === 'boolean') return (a === true || a === 1) === (b === true || b === 1)
  return Number(a) === Number(b)
}


function stripWgslComments (src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '')
}

// A shader reads time, deltaTime or frame (see the header).
function shaderReadsTime (src) {
  const code = stripWgslComments(src)
  if (/\b[A-Za-z_][A-Za-z0-9_]*\s*\.\s*(time|deltaTime|frame)\b/.test(code)) return true
  for (const name of TIME_UNIFORMS) {
    const uses = code.match(new RegExp(`\\b${name}\\b`, 'g')) || []
    if (new RegExp(`var<uniform>\\s+${name}\\b`).test(code) && uses.length > 1) return true
    if (new RegExp(`\\b${name}(?:\\s*:\\s*[^\\n=]+)?\\s*=\\s*(?:max\\s*\\([^,]+,\\s*)?(?:i32\\s*\\(\\s*)?uniforms\\.data\\[\\d+\\]`).test(code)) return true
  }
  return false
}

// Textures the effect reads at or before the pass that first writes them.
function stateTextures (def) {
  const firstRead = new Map()
  const firstWrite = new Map()
  ;(def.passes || []).forEach((p, i) => {
    for (const v of Object.values(p.inputs || {})) if (typeof v === 'string' && !firstRead.has(v)) firstRead.set(v, i)
    for (const v of Object.values(p.outputs || {})) if (typeof v === 'string' && !firstWrite.has(v)) firstWrite.set(v, i)
  })
  const chain = new Set(['inputTex', 'outputTex', 'inputTex3d', 'outputTex3d'])
  return [...firstWrite.keys()].filter(t => !chain.has(t) && firstRead.has(t) && firstRead.get(t) <= firstWrite.get(t)).sort()
}

// Why an effect evolves across frames (empty: it does not).
function timedReasons (ns, name, def) {
  const reasons = []
  if (stateTextures(def).length) reasons.push('state')
  if (ns === 'points' || /^points/.test(def.func || name)) reasons.push('particles')
  if ((def.tags || []).includes('sim')) reasons.push('sim')
  const dir = join(CATALOG, ns, name, 'wgsl')
  const sources = existsSync(dir) ? readdirSync(dir).filter(f => f.endsWith('.wgsl')).sort().map(f => readFileSync(join(dir, f), 'utf8')) : []
  const layouts = [def.uniformLayout, ...Object.values(def.uniformLayouts || {})].filter(l => l && typeof l === 'object')
  if (sources.some(shaderReadsTime) || layouts.some(l => TIME_UNIFORMS.some(t => t in l))) reasons.push('time')
  if (Array.isArray(def.jsHooks) && def.jsHooks.includes('onUpdate')) reasons.push('onUpdate')
  return reasons
}

function oscProgram (kind) {
  return `search synth\nnoise(seed: 1, speed: 0, scaleX: osc(type: oscKind.${kind}, min: 20, max: 80), scaleY: 50).write(o0)\nrender(o0)\n`
}

async function main () {
  if (!process.env.NM_REFERENCE_ROOT) {
    process.stderr.write('NM_REFERENCE_ROOT is not set\n')
    process.exit(2)
  }
  const check = process.argv.includes('--check')
  const { bootstrapReference } = await import(pathToFileURL(join(ROOT, 'tools', 'reference-oracle.mjs')).href)
  const ref = await bootstrapReference()

  const files = new Map()
  const timedFiles = new Map()
  const timedCases = {}
  const effectReasons = {}
  const dropped = []
  const uncovered = []
  for (const ns of sortedDirs(CATALOG)) {
    for (const name of sortedDirs(join(CATALOG, ns))) {
      const def = JSON.parse(readFileSync(join(CATALOG, ns, name, 'definition.json'), 'utf8'))
      const func = def.func || name
      const effectKey = `${ns}.${func}`
      const curated = CURATED_BASES[effectKey]
      let baseText = def.defaultProgram ? def.defaultProgram.trim() + '\n' : null
      if (curated?.file) baseText = readFileSync(join(ROOT, curated.file), 'utf8')
      if (curated?.text) baseText = curated.text
      if (curated?.defaultProgramOf) {
        const [cns, cfunc] = curated.defaultProgramOf.split('.')
        const other = JSON.parse(readFileSync(join(CATALOG, cns, cfunc, 'definition.json'), 'utf8'))
        baseText = other.defaultProgram.trim() + '\n'
      }
      const tpl = baseText ? null : baseProgram(ns, func, def)
      const build = (args) => {
        if (tpl) return templateSource(tpl, args)
        return withCallArgs(baseText, func, args)
      }
      // The effect's output must reach the presented surface: some pass at or after
      // the effect's last pass writes graph.renderSurface.
      const reachesOutput = (graph) => {
        const last = graph.passes.map((p, i) => p.effectKey === effectKey ? i : -1).reduce((a, b) => Math.max(a, b), -1)
        if (last < 0 || !graph.renderSurface) return false
        const target = `global_${graph.renderSurface}`
        return graph.passes.some((p, i) => i >= last && Object.values(p.outputs || {}).includes(target))
      }
      const compiles = (src) => {
        try {
          const graph = ref.compileGraph(src)
          if (!reachesOutput(graph)) return { error: `the effect does not reach the rendered surface ${graph.renderSurface}` }
          return { graph }
        } catch (err) {
          const msg = err?.message || (err?.diagnostics || err?.errors || []).map(d => d.message).join('; ') || JSON.stringify(err)
          return { error: msg }
        }
      }
      const base = build([])
      const baseResult = base && compiles(base)
      if (!base || baseResult.error) {
        uncovered.push(`${effectKey}: base program does not compile: ${base ? baseResult.error : 'no call site in defaultProgram'}`)
        continue
      }
      const stem = `${ns}_${func}`
      files.set(`${stem}.dsl`, base)
      const midi = MIDI_SIDECARS[effectKey]?.()
      if (midi) files.set(`${stem}.midi.json`, midi)
      const reasons = timedReasons(ns, name, def)
      effectReasons[effectKey] = reasons
      if (reasons.length) {
        timedFiles.set(`timed_${stem}.dsl`, base)
        if (midi) timedFiles.set(`timed_${stem}.midi.json`, midi)
        timedCases[`timed_${stem}`] = { effect: effectKey, reasons }
      }

      const effectPasses = (graph) => graph.passes.filter(p => p.effectKey === effectKey)
      for (const [param, spec] of Object.entries(def.globals || {})) {
        const variants = []
        if (spec.choices && typeof spec.choices === 'object') {
          for (const [choiceName, value] of Object.entries(spec.choices)) {
            if (choiceName.endsWith(':')) continue
            if (value === spec.default || (typeof value === 'number' && Number(spec.default) === value && spec.type !== 'member')) continue
            variants.push({ label: choiceName, token: choiceToken(choiceName, value), value })
          }
        } else if (spec.type === 'boolean') {
          const flipped = !(spec.default === true)
          variants.push({ label: String(flipped), token: String(flipped), value: flipped })
        }
        for (const v of variants) {
          const fname = `${stem}__${fileToken(param)}_${fileToken(v.label)}.dsl`
          const built = build([[param, v.token]])
          const src = built && applyVariantFix(fname, built)
          if (!src) { dropped.push(`${fname}: cannot place the argument in defaultProgram`); continue }
          const result = compiles(src)
          if (result.error) { dropped.push(`${fname}: reference compile failed: ${result.error.slice(0, 200)}`); continue }
          const uniform = spec.uniform || param
          const hit = effectPasses(result.graph).some(p => {
            if (spec.define) return p.program.includes(`__${spec.define}_${v.value}`) || (p.uniforms && sameValue(p.uniforms[uniform], v.value))
            return p.uniforms && uniform in p.uniforms && sameValue(p.uniforms[uniform], v.value)
          })
          if (!hit) { dropped.push(`${fname}: the reference graph does not carry ${param}=${JSON.stringify(v.value)}`); continue }
          files.set(fname, src)
          if (midi) files.set(fname.replace(/\.dsl$/, '.midi.json'), midi)
        }
      }
    }
  }

  for (const kind of OSC_KINDS) {
    const src = oscProgram(kind)
    const graph = ref.compileGraph(src)
    const automated = graph.passes.some(p => p.effectKey === 'synth.noise' && p.uniforms &&
      typeof p.uniforms.scaleX === 'object' && p.uniforms.scaleX !== null)
    if (!automated) throw new Error(`timed_osc_${kind}: the reference graph carries no oscillator on scaleX`)
    timedFiles.set(`timed_osc_${kind}.dsl`, src)
    timedCases[`timed_osc_${kind}`] = { effect: 'synth.noise', reasons: ['oscillator'] }
  }
  const evolvingKeys = new Set(Object.entries(effectReasons)
    .filter(([, r]) => r.some(x => x !== 'time')).map(([k]) => k))
  const programDirs = ['programs', 'curated']
  const programNames = new Set()
  for (const dir of programDirs) {
    const full = join(ROOT, 'parity', dir)
    for (const f of readdirSync(full).filter(f => f.endsWith('.dsl')).sort()) {
      const name = f.slice(0, -4)
      if (programNames.has(name)) throw new Error(`program ${name} exists in two directories`)
      programNames.add(name)
      if (existsSync(join(full, `${name}.portable.json`)) || existsSync(join(full, `${name}.midi.json`))) continue
      const src = readFileSync(join(full, f), 'utf8')
      let graph
      try { graph = ref.compileGraph(src) } catch { continue }
      const keys = [...new Set(graph.passes.map(p => p.effectKey).filter(k => evolvingKeys.has(k)))].sort()
      if (!keys.length) continue
      timedFiles.set(`timed_${name}.dsl`, src)
      if (existsSync(join(full, `${name}.obj`))) timedFiles.set(`timed_${name}.obj`, readFileSync(join(full, `${name}.obj`), 'utf8'))
      timedCases[`timed_${name}`] = { effect: null, program: `parity/${dir}/${f}`, reasons: ['program'], evolving: keys }
    }
  }
  const dslCount = (map) => [...map.keys()].filter(f => f.endsWith('.dsl')).length
  const timedNames = [...timedFiles.keys()].filter(f => f.endsWith('.dsl')).sort()
  const manifest = {
    protocol: TIMED_PROTOCOL,
    cases: Object.fromEntries(Object.keys(timedCases).sort().map(k => [k, timedCases[k]]))
  }
  timedFiles.set('manifest.json', JSON.stringify(manifest, null, 2) + '\n')

  const owned = f => f.endsWith('.dsl') || f.endsWith('.midi.json') || f.endsWith('.obj')
  const outputs = [[OUT, files, owned], [TIMED_OUT, timedFiles, f => owned(f) || f === 'manifest.json']]
  if (check) {
    const stale = []
    for (const [dir, map, owned] of outputs) {
      const names = [...map.keys()].sort()
      const existing = existsSync(dir) ? readdirSync(dir).filter(owned).sort() : []
      if (existing.join('\n') !== names.join('\n')) stale.push(`${dir}: file set differs`)
      for (const n of names) {
        const p = join(dir, n)
        if (!existsSync(p) || readFileSync(p, 'utf8') !== map.get(n)) stale.push(n)
      }
    }
    if (stale.length) {
      process.stderr.write(`[generate-coverage] parity/coverage or parity/timed is stale (${stale.length}): ${stale.slice(0, 10).join(', ')}\n`)
      process.exit(1)
    }
    process.stderr.write(`[generate-coverage] parity/coverage is current (${dslCount(files)} programs), parity/timed is current (${timedNames.length} programs)\n`)
    return
  }
  for (const [dir, map] of outputs) {
    rmSync(dir, { recursive: true, force: true })
    mkdirSync(dir, { recursive: true })
    for (const [n, text] of map) writeFileSync(join(dir, n), text)
  }
  for (const d of dropped) process.stderr.write(`[generate-coverage] dropped ${d}\n`)
  for (const u of uncovered) process.stderr.write(`[generate-coverage] UNCOVERED ${u}\n`)
  process.stderr.write(`[generate-coverage] ${dslCount(files)} programs, ${dropped.length} dropped variants, ${uncovered.length} uncovered effects; ` +
    `${timedNames.length} timed programs\n`)
}

main().catch(err => {
  process.stderr.write(`[generate-coverage] FAILED: ${err?.stack || err}\n`)
  process.exit(1)
})
