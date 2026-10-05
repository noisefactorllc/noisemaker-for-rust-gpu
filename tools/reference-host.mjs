#!/usr/bin/env node
// reference-host.mjs — run the reference engine's CPU-side host inputs and write
// their exact outputs, for the differential tests of crates/noisemaker-host.
//
// Subcommands:
//
//   obj --out DIR [FILE.obj ...]
//       Runs the reference parseOBJ + packMeshDataForTextures(..., 256, 256)
//       (shaders/src/runtime/obj-parser.js) on each file and on built-in
//       edge-case texts. Without files: every OBJ of the embedded catalog
//       (crates/noisemaker-effects/catalog/share/meshes) and of parity/programs.
//       Each case's text is decoded as the reference's fetch().text() does
//       (TextDecoder: UTF-8, BOM dropped, invalid sequences -> U+FFFD). Writes
//       DIR/manifest.json ([{name, obj, vertexCount, packedVertexCount}]),
//       DIR/<name>.obj (the exact bytes the case parsed) and
//       DIR/<name>.{positions,normals,uvs,positionData,normalData,uvData}.f32
//       (little-endian Float32Array bytes).
//
//   worm --effect fibers|scratches|strayHair --width W --height H
//        [--params JSON] [--cancel-after N] [--math chromium|node] [--out FILE]
//       Runs the effect definition's asyncInit (shaders/effects/filter/<effect>/
//       definition.js) on a mock canvas and writes every 2D-context operation, one
//       per line (the format of noisemaker_host::canvas::CallRecorder): numbers as
//       the 16 hex digits of their IEEE double bits, strings verbatim, and
//       "update <name>" where the asyncInit calls updateTexture(name, canvas).
//       --params is the asyncInit's params object (default {}); --cancel-after N
//       makes isCancelled() return true from its (N+1)th call on.
//
//       --math chromium (the default) evaluates Math.sin, Math.cos and Math.log
//       correctly rounded, as Chromium's V8 does (Chromium 153 matched correctly
//       rounded results on 3 million arguments of the tracer's ranges); Node's V8
//       returns other values for about 3 % of sin/cos and 7 % of log arguments,
//       which changes the traced coordinates. `math-check` verifies the
//       substitute against the browser. --math node keeps Node's functions.
//
//   math-check [--count N]
//       Evaluates Math.sin/cos/log in headless Chromium (playwright from
//       $NM_REFERENCE_ROOT) on N random arguments of the tracer's ranges and
//       compares the --math chromium substitute (and Node's own functions) with
//       it. The browser closes as soon as the comparison finishes. Exit 1 when the
//       substitute differs from Chromium.
//
//   text --out DIR [--cases FILE.json]
//       Captures the filter/text canvases the reference demo host uploads. In the
//       reference demo page (/demo/shaders/, WebGPU backend, the vendored
//       shade-mcp harness, as parity/batch-golden.mjs drives it) each case runs the
//       demo's own UIController.prototype._renderTextToCanvas on a text state
//       holding the case's values (textContent, font, size, posX, posY, color,
//       rotation, justify) and a renderer of the case's width, after the page
//       loaded the case's font. The canvas is uploaded exactly as the demo's
//       _updateTextTexture does, copyExternalImageToTexture({ source: canvas,
//       flipY: true }) into rgba8unorm, in a texture that also has COPY_SRC so
//       it can be read back; the bytes leave the page as a binary upload (never
//       as text). Writes DIR/<name>.png (the uploaded texture, row 0 first),
//       DIR/<name>.premul.png (the same copy with premultipliedAlpha: true, i.e.
//       the canvas backing store) and DIR/manifest.json (the cases with their
//       values). Without --cases, a built-in set: fonts (the bundled Nunito, the
//       CSS generic families, named system families, a missing family), sizes
//       in each of the canvas's glyph regimes (masks below 162 px, distance
//       fields to 255 px, paths from 256 px), multi-line text, alignment,
//       rotation, colours and canvas sizes. The browser closes when the
//       captures finish.
//
// Environment: NM_REFERENCE_ROOT, a checkout of the reference engine.

import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { deflateSync } from 'node:zlib'
import { createRequire } from 'node:module'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const SHADERS_DIR = join(REFERENCE_ROOT, 'shaders')

function parseArgs (argv) {
  const opts = { _: [] }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a.startsWith('--')) opts[a.slice(2)] = argv[++i]
    else opts._.push(a)
  }
  return opts
}

// ---- obj -------------------------------------------------------------------------

// Edge cases: [name, bytes]. Texts exercise the reference's parsing rules: faces with
// and without uvs/normals, relative (negative) and out-of-range indices, polygons,
// comments, malformed numbers and lines, JavaScript whitespace, and encodings.
function objEdgeCases () {
  const t = (s) => Buffer.from(s, 'utf8')
  return [
    ['empty', t('')],
    ['whitespace-only', t(' \n\t\r\n  \n')],
    ['comments-only', t('# a comment\n#v 1 2 3\n   # indented comment\n')],
    ['triangle-positions', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')],
    ['quad-positions', t('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n')],
    ['bowtie-quad', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nf 1 2 3 4\n')],
    ['pentagon-uv-normal', t('v 0 0 0\nv 2 0 0\nv 3 1 0\nv 1 2 0\nv -1 1 0\n' +
      'vt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 1\nvt 0 0.5\nvn 0 0 1\nvn 0 0.6 0.8\n' +
      'f 1/1/1 2/2/1 3/3/2 4/4/2 5/5/1\n')],
    ['v-vt-faces', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.25 0.75\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n')],
    ['v-vn-faces', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 -1\nf 1//1 2//1 3//1\n')],
    ['mixed-normal-refs', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 1\nvn 0 1 0\n' +
      'f 1//1 2 3\nf 2 4 3\n')],
    ['negative-indices', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvn 0 0 1\nf -3/-1/-1 -2/-1/-1 -1/-1/-1\n')],
    ['out-of-range-indices', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\nf 0 1 2\nf 1/7/9 2/8/9 3/9/9\n')],
    ['malformed-numbers', t('v 1.5abc .5 -.25e1\nv 1e400 -1e-400 -0\nv nan Infinity -Infinityx\n' +
      'v 0x10 1_000 +.e1\nv 1. 1e 1e+\nv\nvt\nvn 1 2\nf 1 2 3\nf 4 5 6\n')],
    ['malformed-indices', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\n' +
      'f 1abc 2.9 3e1\nf /1/1 2/ 3//\nf 1/x/y 2 3 4 # trailing comment\nf 1 2\nf\nf 1 1 1\n')],
    ['inline-comment-vertex', t('v 1 2 3 # comment\nv 4 5 6#x\nv 7 8 9\nf 1 2 3\n')],
    ['unknown-commands', t('o object\ng group\nusemtl m\ns 1\nmtllib x.mtl\nl 1 2\nV 9 9 9\n' +
      'v 0 0 0\nv 0 0 1\nv 0 1 0\nF 1 2 3\nf 1 2 3\n')],
    ['crlf-and-tabs', t('v\t0 0 0\r\nv 1\t0\t0\r\n  v 0 1 0  \r\nf 1\t2  3\r\n')],
    ['js-whitespace', t('﻿v 0 0 0\nv 1 0　0\nv 0 1 0\nf 1 2 3\n')],
    ['utf8-bom', Buffer.concat([Buffer.from([0xef, 0xbb, 0xbf]), t('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')])],
    ['invalid-utf8', Buffer.concat([t('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1'), Buffer.from([0xed, 0xa0, 0x80]),
      t(' 2 3\nv 4 '), Buffer.from([0xff]), t('5 6\nf 1 2 3\nf 4 5 6\n')])],
    ['shared-positions', t('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nv 0.00004 0 0\nv -0.00001 1 0\n' +
      'f 1 2 3\nf 1 3 4\nf 1 4 2\nf 5 3 2\nf 6 1 2\n')],
    ['degenerate-triangles', t('v 0 0 0\nv 0 0 0\nv 0 0 0\nv 1 1 1\nv 2 2 2\nf 1 2 3\nf 1 4 5\n')],
    ['huge-coordinates', t('v 1e30 0 0\nv 0 1e30 0\nv 0 0 1e30\nv 3.4028235677973366e38 0 0\n' +
      'v 1e300 1e300 1e300\nf 1 2 3\nf 4 2 3\nf 5 2 3\n')],
    ['rounding-keys', t('v 0.00005 0 0\nv 0.000149999 0 0\nv -0.00005 0 0\nv 1 1 0\nv 0 1 1\n' +
      'v 0.123456789 0.987654321 0.5\nf 1 4 5\nf 2 4 5\nf 3 4 5\nf 6 4 5\n')],
    ['long-digits', t('v 0.1000000000000000055511151231257827021181583404541015625 ' +
      '123456789012345678901234567890 1.00000000000000011102230246251565404236316680908203125\n' +
      'v 9007199254740993 0.30000000000000004 2.2250738585072014e-308\nv 0 0 1\n' +
      'f 1 2 3\nf 99999999999999999999 1 2\n')],
    ['vertex-w-and-uv-w', t('v 1 2 3 0.5\nv 4 5 6 1\nv 7 8 9 2\nvt 0.1 0.2 0.3\nvt 0.4 0.5\nvt 0.6\n' +
      'f 1/1 2/2 3/3\n')],
    ['many-vertices-truncated', t(Array.from({ length: 22000 }, (_, i) =>
      `v ${(i % 97) * 0.01} ${Math.floor(i / 97) * 0.01} ${(i % 13) * 0.001}`).join('\n') + '\n' +
      Array.from({ length: 21998 }, (_, i) => `f ${i + 1} ${i + 2} ${i + 3}`).join('\n') + '\n')]
  ]
}

function defaultObjFiles () {
  const files = []
  const meshes = join(ROOT, 'crates', 'noisemaker-effects', 'catalog', 'share', 'meshes')
  for (const f of readdirSync(meshes).filter(f => f.endsWith('.obj')).sort()) files.push(join(meshes, f))
  const programs = join(ROOT, 'parity', 'programs')
  for (const f of readdirSync(programs).filter(f => f.endsWith('.obj')).sort()) files.push(join(programs, f))
  return files
}

async function cmdObj (opts) {
  if (!opts.out) throw new Error('obj: --out DIR is required')
  const { parseOBJ, packMeshDataForTextures } = await import(pathToFileURL(join(SHADERS_DIR, 'src', 'runtime', 'obj-parser.js')).href)
  mkdirSync(opts.out, { recursive: true })
  const files = opts._.length ? opts._.map(f => resolve(f)) : defaultObjFiles()
  const cases = [
    ...files.map(f => [`${basename(dirname(f))}-${basename(f, '.obj')}`, readFileSync(f)]),
    ...(opts._.length ? [] : objEdgeCases().map(([name, bytes]) => [`edge-${name}`, bytes]))
  ]
  const manifest = []
  const decoder = new TextDecoder('utf-8')
  const warn = console.warn
  for (const [name, bytes] of cases) {
    writeFileSync(join(opts.out, `${name}.obj`), bytes)
    const mesh = parseOBJ(decoder.decode(bytes))
    console.warn = () => {} // the truncation warning of the large case
    const packed = packMeshDataForTextures(mesh.positions, mesh.normals, mesh.uvs, 256, 256)
    console.warn = warn
    const arrays = {
      positions: mesh.positions,
      normals: mesh.normals,
      uvs: mesh.uvs,
      positionData: packed.positionData,
      normalData: packed.normalData,
      uvData: packed.uvData
    }
    for (const [key, array] of Object.entries(arrays)) {
      writeFileSync(join(opts.out, `${name}.${key}.f32`), Buffer.from(array.buffer, array.byteOffset, array.byteLength))
    }
    manifest.push({ name, obj: `${name}.obj`, vertexCount: mesh.vertexCount, packedVertexCount: packed.vertexCount })
  }
  writeFileSync(join(opts.out, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
  console.error(`[reference-host] obj: ${manifest.length} cases -> ${opts.out}`)
}

// ---- correctly rounded Math.sin / Math.cos / Math.log -----------------------------
// Double-double arithmetic (about 104 significant bits), rounded once at the end:
// the same evaluation as noisemaker_host::js, with Dekker's exact product in place
// of a fused multiply-add.

function twoSum (a, b) { const s = a + b; const bb = s - a; return [s, (a - (s - bb)) + (b - bb)] }
function quickTwoSum (a, b) { const s = a + b; return [s, b - (s - a)] }
function split (a) { const c = 134217729 * a; const hi = c - (c - a); return [hi, a - hi] }
function twoProd (a, b) {
  const p = a * b
  const [ah, al] = split(a)
  const [bh, bl] = split(b)
  return [p, ((ah * bh - p) + ah * bl + al * bh) + al * bl]
}
function ddAdd (a, b) {
  let s = twoSum(a[0], b[0])
  const t = twoSum(a[1], b[1])
  s = quickTwoSum(s[0], s[1] + t[0])
  return quickTwoSum(s[0], s[1] + t[1])
}
const ddNeg = (a) => [-a[0], -a[1]]
function ddMul (a, b) { const p = twoProd(a[0], b[0]); return quickTwoSum(p[0], p[1] + (a[0] * b[1] + a[1] * b[0])) }
function ddMulD (a, b) { const p = twoProd(a[0], b); return quickTwoSum(p[0], p[1] + a[1] * b) }
function ddDivD (a, b) {
  const q1 = a[0] / b
  const p = twoProd(q1, b)
  return quickTwoSum(q1, ((a[0] - p[0]) - p[1] + a[1]) / b)
}
function ddDiv (a, b) {
  const q1 = a[0] / b[0]
  let r = ddAdd(a, ddNeg(ddMulD(b, q1)))
  const q2 = r[0] / b[0]
  r = ddAdd(r, ddNeg(ddMulD(b, q2)))
  const q3 = r[0] / b[0]
  return ddAdd(quickTwoSum(q1, q2), [q3, 0])
}
function sinSeries (r) {
  const r2 = ddMul(r, r)
  let term = r; let sum = r
  for (let n = 3; n <= 33; n += 2) { term = ddDivD(ddMul(term, r2), -((n - 1) * n)); sum = ddAdd(sum, term) }
  return sum
}
function cosSeries (r) {
  const r2 = ddMul(r, r)
  let term = [1, 0]; let sum = [1, 0]
  for (let n = 2; n <= 32; n += 2) { term = ddDivD(ddMul(term, r2), -((n - 1) * n)); sum = ddAdd(sum, term) }
  return sum
}
function roundHalfEven (x) {
  const f = Math.floor(x); const d = x - f
  if (d > 0.5) return f + 1
  if (d < 0.5) return f
  return f % 2 === 0 ? f : f + 1
}
function reduceQuadrant (x) {
  if (!(Math.abs(x) < 1048576)) return null
  const k = roundHalfEven(x * 6.36619772367581382433e-01)
  let acc = [x, 0]
  acc = ddAdd(acc, [-k * 1.57079632673412561417e+00, 0])
  acc = ddAdd(acc, [-k * 6.07710050630396597660e-11, 0])
  acc = ddAdd(acc, [-k * 2.02226624871116645580e-21, 0])
  acc = ddAdd(acc, ddNeg(twoProd(k, 8.47842766036889956997e-32)))
  return [acc, ((k % 4) + 4) % 4]
}
const nodeSin = Math.sin
const nodeCos = Math.cos
const nodeLog = Math.log
function crSin (x) {
  if (x === 0) return x
  if (!Number.isFinite(x)) return NaN
  const red = reduceQuadrant(x)
  if (!red) return nodeSin(x)
  const [r, q] = red
  return q === 0 ? sinSeries(r)[0] : q === 1 ? cosSeries(r)[0] : q === 2 ? -sinSeries(r)[0] : -cosSeries(r)[0]
}
function crCos (x) {
  if (!Number.isFinite(x)) return NaN
  if (x === 0) return 1
  const red = reduceQuadrant(x)
  if (!red) return nodeCos(x)
  const [r, q] = red
  return q === 0 ? cosSeries(r)[0] : q === 1 ? -sinSeries(r)[0] : q === 2 ? -cosSeries(r)[0] : sinSeries(r)[0]
}
const f64 = new Float64Array(1)
const u32 = new Uint32Array(f64.buffer)
function frexp (x) {
  f64[0] = x
  const exp = (u32[1] >>> 20) & 0x7ff
  if (exp === 0) { const [m, e] = frexp(x * 2 ** 54); return [m, e - 54] }
  u32[1] = (u32[1] & 0x800fffff) | (1022 << 20)
  return [f64[0], exp - 1022]
}
function crLog (x) {
  if (Number.isNaN(x) || x < 0) return NaN
  if (x === 0) return -Infinity
  if (x === Infinity) return x
  if (x === 1) return 0
  let [m, e] = frexp(x)
  if (m < Math.SQRT1_2) { m *= 2; e -= 1 }
  const s = ddDiv([m - 1, 0], twoSum(m, 1))
  const s2 = ddMul(s, s)
  let power = s; let sum = s
  for (let n = 3; n <= 61; n += 2) { power = ddMul(power, s2); sum = ddAdd(sum, ddDivD(power, n)) }
  let result = ddMulD(sum, 2)
  if (e !== 0) result = ddAdd(ddMulD([Math.LN2, 2.3190468138462996e-17], e), result)
  return result[0]
}

function installMath (mode) {
  if (mode === 'node') return
  if (mode !== 'chromium') throw new Error(`--math must be chromium or node, not ${mode}`)
  Math.sin = crSin
  Math.cos = crCos
  Math.log = crLog
}

// ---- worm ------------------------------------------------------------------------

const bitsView = new DataView(new ArrayBuffer(8))
function bits (x) {
  bitsView.setFloat64(0, x)
  return bitsView.getBigUint64(0).toString(16).padStart(16, '0')
}

// A CanvasRenderingContext2D stand-in that logs what the reference does with it and
// rejects anything else, so an unexpected API use cannot pass silently.
function recordingCanvas (log) {
  const canvas = { width: 300, height: 150 }
  const state = {}
  const ctx = {
    get canvas () { return canvas },
    clearRect (x, y, w, h) { log.push(`clearRect ${bits(x)} ${bits(y)} ${bits(w)} ${bits(h)}`) },
    beginPath () { log.push('beginPath') },
    moveTo (x, y) { log.push(`moveTo ${bits(x)} ${bits(y)}`) },
    lineTo (x, y) { log.push(`lineTo ${bits(x)} ${bits(y)}`) },
    stroke () { log.push('stroke') }
  }
  for (const prop of ['lineCap', 'lineJoin', 'strokeStyle']) {
    Object.defineProperty(ctx, prop, {
      get () { return state[prop] },
      set (v) { state[prop] = v; log.push(`${prop} ${v}`) }
    })
  }
  Object.defineProperty(ctx, 'lineWidth', {
    get () { return state.lineWidth },
    set (v) { state.lineWidth = v; log.push(`lineWidth ${bits(v)}`) }
  })
  const guarded = new Proxy(ctx, {
    get (target, prop) {
      if (!(prop in target)) throw new Error(`reference-host: unexpected 2D context member ${String(prop)}`)
      return target[prop]
    },
    set (target, prop, value) {
      if (!(prop in target)) throw new Error(`reference-host: unexpected 2D context property ${String(prop)}`)
      target[prop] = value
      return true
    }
  })
  canvas.getContext = (kind) => {
    if (kind !== '2d') throw new Error(`reference-host: unexpected getContext(${kind})`)
    return guarded
  }
  return canvas
}

async function cmdWorm (opts) {
  const effect = opts.effect
  if (!['fibers', 'scratches', 'strayHair'].includes(effect)) throw new Error('worm: --effect fibers|scratches|strayHair')
  const width = Number(opts.width ?? 256)
  const height = Number(opts.height ?? 256)
  const params = JSON.parse(opts.params ?? '{}')
  const cancelAfter = opts['cancel-after'] === undefined ? Infinity : Number(opts['cancel-after'])
  installMath(opts.math ?? 'chromium')
  const log = []
  globalThis.document = {
    createElement (tag) {
      if (tag !== 'canvas') throw new Error(`reference-host: unexpected createElement(${tag})`)
      return recordingCanvas(log)
    }
  }
  const mod = await import(pathToFileURL(join(SHADERS_DIR, 'effects', 'filter', effect, 'definition.js')).href)
  const def = typeof mod.default === 'function' ? new mod.default() : mod.default
  let polls = 0
  await def.asyncInit({
    updateTexture: (name, canvas) => {
      if (canvas.width !== width || canvas.height !== height) throw new Error('reference-host: canvas size changed')
      log.push(`update ${name}`)
    },
    width,
    height,
    params,
    isCancelled: () => polls++ >= cancelAfter
  })
  const text = log.join('\n') + '\n'
  if (opts.out) writeFileSync(opts.out, text)
  else process.stdout.write(text)
}

// ---- binary uploads from a page ---------------------------------------------------

// Pixels and other binary results leave a page as binary uploads, never as text:
// the page POSTs the bytes to /__nm_capture/<token>, and the route installed here
// keeps the request body as a Buffer under that token.
async function installCaptureRoute (page) {
  const uploads = new Map()
  await page.route('**/__nm_capture/*', async (route) => {
    const token = route.request().url().split('/__nm_capture/')[1]
    uploads.set(token, route.request().postDataBuffer())
    await route.fulfill({ status: 204 })
  })
  return uploads
}

// ---- math-check ------------------------------------------------------------------

async function cmdMathCheck (opts) {
  const count = Number(opts.count ?? 1000000)
  // Arguments of the tracer's ranges: angles up to a few hundred radians, TAU * u,
  // log arguments in (0, 1].
  let seed = 12345
  const rnd = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296 }
  const args = new Float64Array(count * 2)
  for (let i = 0; i < count; i++) {
    const k = i % 3
    args[2 * i] = k === 0 ? rnd() * 400 + rnd() / 4294967296 : k === 1 ? (rnd() - 0.5) * 2000 : rnd() * Math.PI * 2
    args[2 * i + 1] = Math.max(rnd() + rnd() / 4294967296, 1e-10)
  }
  const require = createRequire(join(REFERENCE_ROOT, 'package.json'))
  const { chromium } = require('playwright')
  const browser = await chromium.launch({ headless: true })
  let browserResults
  let version
  try {
    version = browser.version()
    const page = await browser.newPage()
    // a same-origin blank page for the uploads (about:blank has no origin to POST to)
    const origin = 'http://nm-reference-host.test'
    await page.route(`${origin}/`, (route) => route.fulfill({ status: 200, contentType: 'text/html', body: '<!doctype html><title>math-check</title>' }))
    const uploads = await installCaptureRoute(page)
    await page.goto(`${origin}/`)
    browserResults = new Float64Array(count * 3)
    const chunk = 200000
    for (let s = 0; s < count; s += chunk) {
      const m = Math.min(chunk, count - s)
      const token = `math-${s}`
      await page.evaluate(async ({ list, token }) => {
        const r = new Float64Array(list.length / 2 * 3)
        for (let i = 0; i < list.length / 2; i++) {
          r[3 * i] = Math.sin(list[2 * i]); r[3 * i + 1] = Math.cos(list[2 * i]); r[3 * i + 2] = Math.log(list[2 * i + 1])
        }
        const upload = await fetch(`/__nm_capture/${token}`, { method: 'POST', body: new Blob([r.buffer]) })
        if (!upload.ok) throw new Error(`upload failed: ${upload.status}`)
      }, { list: Array.from(args.subarray(2 * s, 2 * (s + m))), token })
      const bytes = uploads.get(token)
      uploads.delete(token)
      if (!bytes || bytes.length !== m * 3 * 8) throw new Error(`math-check: upload ${token} has ${bytes?.length} bytes`)
      // copy: the Buffer's offset need not be 8-byte aligned
      browserResults.set(new Float64Array(new Uint8Array(bytes).buffer), 3 * s)
    }
  } finally {
    await browser.close()
  }
  const mismatch = { sub: [0, 0, 0], node: [0, 0, 0] }
  for (let i = 0; i < count; i++) {
    const a = args[2 * i]; const u = args[2 * i + 1]
    const ref = [browserResults[3 * i], browserResults[3 * i + 1], browserResults[3 * i + 2]]
    const sub = [crSin(a), crCos(a), crLog(u)]
    const node = [nodeSin(a), nodeCos(a), nodeLog(u)]
    for (let k = 0; k < 3; k++) {
      if (!Object.is(sub[k], ref[k])) mismatch.sub[k]++
      if (!Object.is(node[k], ref[k])) mismatch.node[k]++
    }
  }
  console.log(`MATH-CHECK chromium=${version} arguments=${count} ` +
    `substitute-mismatches sin=${mismatch.sub[0]} cos=${mismatch.sub[1]} log=${mismatch.sub[2]} ` +
    `node-mismatches sin=${mismatch.node[0]} cos=${mismatch.node[1]} log=${mismatch.node[2]}`)
  if (mismatch.sub.some(n => n > 0)) process.exit(1)
}

// ---- text ------------------------------------------------------------------------

// The demo's text state for one case: the filter/text globals (definition defaults
// for the values a case leaves out) and the canvas width the demo takes from the
// renderer (canvas = width x width).
const TEXT_DEFAULTS = {
  text: 'Hello World', font: 'Nunito', size: 0.1, posX: 0.5, posY: 0.5, rotation: 0, color: '#ffffff', justify: 'center'
}

function textCases () {
  const c = (name, values = {}, width = 256) => ({ name, width, ...TEXT_DEFAULTS, ...values })
  return [
    c('default'),
    c('default-512', {}, 512),
    c('default-128', {}, 128),
    c('hello-large', { text: 'Hello', size: 0.25 }),
    c('big-word', { text: 'Big', size: 0.5 }),
    c('huge-glyphs', { text: 'Ag', size: 0.9 }, 512),
    c('huge-rotated', { text: 'R', size: 1.0, rotation: 15 }, 384),
    c('field-162', { text: 'Ag', size: 162 / 512 }, 512),
    c('field-205', { text: 'Sdf', size: 0.4 }, 512),
    c('field-rotated', { text: 'Rot', size: 0.38, rotation: 20 }, 512),
    c('field-orange', { text: 'Ok', size: 0.45, color: '#ff8000' }, 512),
    c('field-multiline', { text: 'Two\nLines', size: 0.35 }, 512),
    c('small', { text: 'small text 123', size: 0.05 }),
    c('multiline', { text: 'Line one\nLine two\nThird line', size: 0.08 }),
    c('multiline-empty-line', { text: 'top\n\nbottom', size: 0.09 }),
    c('left', { text: 'Left aligned', justify: 'left', posX: 0.05 }),
    c('right', { text: 'Right aligned', justify: 'right', posX: 0.95 }),
    c('start', { text: 'start', justify: 'start', posX: 0.2 }),
    c('end', { text: 'end', justify: 'end', posX: 0.8 }),
    c('invalid-justify', { text: 'justify?', justify: 'middle', posX: 0.3 }),
    c('rotate-30', { text: 'Rotated', rotation: 30, size: 0.12 }),
    c('rotate-90', { text: 'Vertical', rotation: 90 }),
    c('rotate-180', { text: 'Upside', rotation: 180 }),
    c('rotate--45', { text: 'Slanted\nTwo', rotation: -45, size: 0.09 }),
    c('position', { text: 'corner', posX: 0.15, posY: 0.1 }),
    c('offcanvas', { text: 'Clipped text', posX: 0.95, posY: 0.97, size: 0.15 }),
    c('orange', { text: 'Orange', color: '#ff8000', size: 0.15 }),
    c('array-color', { text: 'Array', color: [0.2, 0.6, 1.0], size: 0.15 }),
    c('dark-color', { text: 'Dark', color: '#203040', size: 0.2 }),
    c('kerning', { text: 'AVATAR To WAVE Ty', size: 0.09 }),
    c('digits', { text: '0123456789', size: 0.1 }),
    c('punctuation', { text: '"Hi!" (a/b) {c};', size: 0.08 }),
    c('latin-accents', { text: 'Ünïcødé façade', size: 0.09 }),
    c('symbols', { text: '€ ✓ → ★ π', size: 0.1 }),
    c('cjk', { text: '漢字かな', size: 0.12 }),
    c('empty', { text: '' }),
    c('zero-size', { text: 'nothing', size: 0 }),
    c('font-serif', { font: 'serif', text: 'Serif Text' }),
    c('font-sans-serif', { font: 'sans-serif', text: 'Sans Serif' }),
    c('font-monospace', { font: 'monospace', text: 'Mono 0O1l' }),
    c('font-cursive', { font: 'cursive', text: 'Cursive' }),
    c('font-fantasy', { font: 'fantasy', text: 'Fantasy' }),
    c('font-system-ui', { font: 'system-ui', text: 'System UI' }),
    c('font-helvetica', { font: 'Helvetica', text: 'Helvetica' }),
    c('font-arial', { font: 'Arial', text: 'Arial Text' }),
    c('font-times-new-roman', { font: 'Times New Roman', text: 'Times New' }),
    c('font-georgia', { font: 'Georgia', text: 'Georgia' }),
    c('font-courier-new', { font: 'Courier New', text: 'Courier' }),
    c('font-quoted-list', { font: '"No Such Font", Georgia, serif', text: 'Fallback' }),
    c('font-missing', { font: 'NoSuchFontFamily', text: 'Missing' }),
    c('font-empty', { font: '', text: 'Empty font' }),
    c('nunito-ttf', { font: 'NunitoBundledTTF', text: 'Hello World' }),
    c('nunito-ttf-kerning', { font: 'NunitoBundledTTF', text: 'AVATAR To WAVE Ty', size: 0.09 })
  ]
}

function writePng (path, width, height, rgba) {
  // minimal RGBA8 PNG encoder (zlib from node)
  const crcTable = new Int32Array(256).map((_, n) => {
    let c = n
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1)
    return c
  })
  const crc32 = (buf) => {
    let c = -1
    for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8)
    return (c ^ -1) >>> 0
  }
  const chunk = (type, data) => {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0)
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
    const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0)
    return Buffer.concat([len, body, crc])
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8; ihdr[9] = 6
  const raw = Buffer.alloc(height * (1 + width * 4))
  for (let y = 0; y < height; y++) {
    raw.set(rgba.subarray(y * width * 4, (y + 1) * width * 4), y * (1 + width * 4) + 1)
  }
  writeFileSync(path, Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))
  ]))
}

async function cmdText (opts) {
  if (!opts.out) throw new Error('text: --out DIR is required')
  mkdirSync(opts.out, { recursive: true })
  const cases = opts.cases ? JSON.parse(readFileSync(opts.cases, 'utf8')) : textCases()
  process.env.SHADE_VIEWER_ROOT = REFERENCE_ROOT
  process.env.SHADE_VIEWER_PATH = '/demo/shaders/'
  process.env.SHADE_EFFECTS_DIR = join(SHADERS_DIR, 'effects')
  process.env.SHADE_GLOBALS_PREFIX = '__noisemaker'
  process.env.SHADE_HEADLESS = process.env.SHADE_HEADLESS ?? '1'
  const { BrowserSession } = await import(pathToFileURL(join(REFERENCE_ROOT, 'vendor', 'shade-mcp', 'harness', 'index.js')).href)
  const session = new BrowserSession({ backend: 'webgpu' })
  await session.setup()
  const manifest = []
  try {
    const page = session.page
    const uploads = await installCaptureRoute(page)
    await session.setBackend('webgpu')
    await page.waitForFunction(() => !!window.__noisemakerCanvasRenderer && !!document.getElementById('dsl-editor'),
      null, { timeout: 120000 })
    // The bundled TTF under its own family name, to compare it with the page's
    // "Nunito" web font (the demo styles declare Nunito from fonts.noisefactor.io).
    const fontInfo = await page.evaluate(async () => {
      const face = new FontFace('NunitoBundledTTF', 'url(/demo/font/Nunito/Nunito-VariableFont_wght.ttf)',
        { weight: '100 1000' })
      document.fonts.add(await face.load())
      await document.fonts.ready
      const faces = []
      document.fonts.forEach((f) => faces.push(`${f.family} ${f.weight} ${f.style} ${f.status}`))
      return { faces, userAgent: navigator.userAgent }
    })
    console.error(`[reference-host] text: ${fontInfo.userAgent}`)
    for (const [caseIndex, tc] of cases.entries()) {
      const token = `text-${caseIndex}`
      const result = await page.evaluate(async ({ tc, token }) => {
        const { UIController } = await import('/demo/shaders/lib/demo-ui.js')
        const fontSize = Math.round(tc.size * tc.width)
        if (tc.font && Number.isFinite(fontSize)) {
          try { await document.fonts.load(`${fontSize}px ${tc.font}`, tc.text || 'x') } catch { /* not a font list */ }
        }
        const canvas = document.createElement('canvas')
        canvas.style.display = 'none'
        document.body.appendChild(canvas)
        const textState = {
          canvas, textureId: 'textTex_step_0', effectKey: 'step_0', textContent: tc.text, font: tc.font,
          size: tc.size, posX: tc.posX, posY: tc.posY, color: tc.color, rotation: tc.rotation, justify: tc.justify
        }
        const host = {
          _textInputs: new Map([[0, textState]]),
          _renderer: { _width: tc.width, _pipeline: {} },
          _hexToRgb: UIController.prototype._hexToRgb,
          _updateTextTexture () {}
        }
        UIController.prototype._renderTextToCanvas.call(host, 0)
        const ctxFont = canvas.getContext('2d').font
        // The page's WebGPU device when a pipeline exists, else one of its own
        // (the same Dawn copy path either way).
        if (!window.__nmTextDevice) {
          window.__nmTextDevice = window.__noisemakerCanvasRenderer?._pipeline?.backend?.device ||
            await (await navigator.gpu.requestAdapter()).requestDevice()
        }
        const device = window.__nmTextDevice
        // Reads the canvas as the texture receives it and uploads the bytes.
        const read = async (premultipliedAlpha, part) => {
          const texture = device.createTexture({
            size: { width: canvas.width, height: canvas.height },
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC |
              GPUTextureUsage.RENDER_ATTACHMENT
          })
          device.queue.copyExternalImageToTexture({ source: canvas, flipY: true },
            { texture, premultipliedAlpha }, { width: canvas.width, height: canvas.height })
          const bytesPerRow = Math.ceil(canvas.width * 4 / 256) * 256
          const buffer = device.createBuffer({ size: bytesPerRow * canvas.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
          const encoder = device.createCommandEncoder()
          encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width: canvas.width, height: canvas.height })
          device.queue.submit([encoder.finish()])
          await buffer.mapAsync(GPUMapMode.READ)
          const src = new Uint8Array(buffer.getMappedRange())
          const out = new Uint8Array(canvas.width * canvas.height * 4)
          for (let y = 0; y < canvas.height; y++) out.set(src.subarray(y * bytesPerRow, y * bytesPerRow + canvas.width * 4), y * canvas.width * 4)
          buffer.unmap(); buffer.destroy(); texture.destroy()
          const upload = await fetch(`/__nm_capture/${token}-${part}`, { method: 'POST', body: new Blob([out]) })
          if (!upload.ok) throw new Error(`upload failed: ${upload.status}`)
        }
        await read(false, 'straight')
        await read(true, 'premultiplied')
        canvas.remove()
        return { width: canvas.width, height: canvas.height, ctxFont }
      }, { tc, token })
      const take = (part) => {
        const bytes = uploads.get(`${token}-${part}`)
        uploads.delete(`${token}-${part}`)
        if (!bytes || bytes.length !== result.width * result.height * 4) {
          throw new Error(`text ${tc.name}: ${part} upload has ${bytes?.length} bytes`)
        }
        return bytes
      }
      writePng(join(opts.out, `${tc.name}.png`), result.width, result.height, take('straight'))
      writePng(join(opts.out, `${tc.name}.premul.png`), result.width, result.height, take('premultiplied'))
      manifest.push({ ...tc, canvasWidth: result.width, canvasHeight: result.height, ctxFont: result.ctxFont })
      console.error(`[reference-host] text ${tc.name}: ${result.width}x${result.height} font "${result.ctxFont}"`)
    }
    writeFileSync(join(opts.out, 'manifest.json'), JSON.stringify({ fonts: fontInfo.faces, cases: manifest }, null, 2) + '\n')
  } finally {
    await session.teardown()
  }
  console.error(`[reference-host] text: ${manifest.length} cases -> ${opts.out}`)
}

// ---- main ------------------------------------------------------------------------

const [command, ...rest] = process.argv.slice(2)
const commands = { obj: cmdObj, worm: cmdWorm, 'math-check': cmdMathCheck, text: cmdText }
if (!commands[command]) {
  console.error('usage: node tools/reference-host.mjs obj|worm|math-check [options]  (see the header)')
  process.exit(2)
}
commands[command](parseArgs(rest)).catch((err) => {
  console.error(`[reference-host] ${command} FAILED: ${err?.stack || err}`)
  process.exit(1)
})
