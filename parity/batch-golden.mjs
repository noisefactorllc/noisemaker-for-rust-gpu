#!/usr/bin/env node
// batch-golden.mjs — mint reference goldens on the reference engine's WebGPU backend.
//
// The golden of a fixture is the reference engine's own render of its DSL, in the
// reference demo page (/demo/shaders/), on the WebGPU backend, read back from the
// presented surface. One Chromium session (the vendored shade-mcp harness) serves
// a chunk of fixtures; each fixture runs this protocol:
//
//   1. pause the demo's render loop, pin the canvas and pipeline to SIZE x SIZE;
//   2. load the DSL through the demo's editor and run button, wait until the
//      pipeline runs exactly that source (graph.source, pass count from the
//      reference compileGraph, compilation finished, WebGPU backend);
//   3. load the meshes a mesh fixture needs (built-in defaults of each
//      externalMesh step, then the fixture's .obj sidecar into mesh0, as the
//      reference demo host does);
//   4. wait until host-supplied textures (media, text) exist and every asyncInit
//      overlay trace (fibers, scratches, strayHair) has finished; save each of
//      those textures as <name>.<textureId>.png so a candidate can bind the same
//      inputs;
//   5. reset to a fresh pipeline state: every pipeline-written texture cleared to
//      zero (host inputs, overlays, mesh data and the MIDI note grid keep their
//      contents), double-buffered surfaces back to their read/write orientation,
//      frame index and clock zeroed, global uniforms emptied;
//   6. render FRAMES frames at the pinned normalized TIME with pipeline.render();
//   7. read back the presented surface with the backend's own readPixels()
//      (rgba16float -> round(v * 255) clamped, the reference conversion; WebGPU
//      textures are top-down, so no flip) and write <name>.golden.png;
//   8. write <name>.graph.json: the graph the page rendered (Maps as objects,
//      program specs keep their WGSL source and drop GLSL), which a candidate can
//      render directly.
//
// Timed fixtures (--run-seconds N --sample-every S) skip the fresh-state reset's
// pinned clock: they step pipeline.render(t) with t = frame / 600 and save
// <name>.golden.t<sec>.png every S seconds.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/batch-golden.mjs <outDir> \
//       [--size 256] [--time 0.25] [--frames 8] [--chunk-size 60] \
//       [--run-seconds N --sample-every S] [--list names.txt] [--] prog.dsl...
//
// Exit 0 when every fixture minted, 1 otherwise. The last stdout lines are
// "BATCH-GOLDEN: minted=N failed=M total=T" and "DONE".

import { existsSync, mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { deflateSync } from 'node:zlib'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const HARNESS = join(REFERENCE_ROOT, 'vendor', 'shade-mcp', 'harness', 'index.js')
const EFFECTS_DIR = join(REFERENCE_ROOT, 'shaders', 'effects')
const VIEWER_PATH = '/demo/shaders/'
const GLOBALS_PREFIX = '__noisemaker'
const STATUS_TIMEOUT = 120000
// Host-supplied texture ids (media/text steps) and asyncInit overlay ids.
const EXTERNAL_TEXTURE_ID = /^[A-Za-z][A-Za-z0-9]*_step_\d+$/
const ASYNC_OVERLAY_ID = /^node_\d+_[A-Za-z][A-Za-z0-9]*$/
const MESH_TEXTURE_INPUT = /^global_mesh\d+_(positions|normals|uvs)/

// ---- PNG -------------------------------------------------------------------
function crc32 (buf) {
  let c = 0xffffffff
  for (let i = 0; i < buf.length; i++) {
    c ^= buf[i]
    for (let k = 0; k < 8; k++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1)
  }
  return (c ^ 0xffffffff) >>> 0
}
function pngChunk (type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length, 0)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body), 0)
  return Buffer.concat([len, body, crc])
}
function encodePng (width, height, rgba) {
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(width, 0); ihdr.writeUInt32BE(height, 4)
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0
  const raw = Buffer.alloc(height * (1 + width * 4))
  for (let y = 0; y < height; y++) {
    raw[y * (1 + width * 4)] = 0
    rgba.copy(raw, y * (1 + width * 4) + 1, y * width * 4, (y + 1) * width * 4)
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    pngChunk('IHDR', ihdr), pngChunk('IDAT', deflateSync(raw)), pngChunk('IEND', Buffer.alloc(0))
  ])
}

// ---- page helpers ----------------------------------------------------------

// Read a backend texture with the reference WebGPU backend's readPixels(). A
// texture the backend created without COPY_SRC usage (updateTextureFromSource:
// host media and text) is first copied exactly into a COPY_SRC texture of the
// same format by a textureLoad render pass. Every capture runs inside a WebGPU
// validation error scope, so a failed copy fails the capture instead of
// reading back zeros.
async function capture (page, textureId = null) {
  const result = await page.evaluate(async ({ textureId }) => {
    const p = window.__noisemakerRenderingPipeline
    const backend = p?.backend
    if (!backend) return { error: 'no pipeline backend' }
    let id = textureId
    if (!id) {
      const name = p.graph?.renderSurface || 'o0'
      const surface = p.surfaces?.get(name)
      if (!surface) return { error: `no render surface ${name}` }
      id = surface.read
    }
    const tex = backend.textures?.get(id)
    if (!tex) return { error: `no texture ${id}` }
    const device = backend.device
    device.pushErrorScope('validation')
    let px
    let temp = null
    try {
      const canCopy = typeof tex.usage === 'number' && (tex.usage & GPUTextureUsage.COPY_SRC) !== 0
      if (canCopy) {
        px = await backend.readPixels(id)
      } else {
        const format = tex.gpuFormat || 'rgba8unorm'
        temp = device.createTexture({
          size: { width: tex.width, height: tex.height, depthOrArrayLayers: 1 },
          format,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
        })
        const module = device.createShaderModule({
          code: `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], 0.0, 1.0);
}
@group(0) @binding(0) var src: texture_2d<f32>;
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  return textureLoad(src, vec2u(pos.xy), 0);
}`
        })
        const pipe = device.createRenderPipeline({
          layout: 'auto',
          vertex: { module, entryPoint: 'vs' },
          fragment: { module, entryPoint: 'fs', targets: [{ format }] },
          primitive: { topology: 'triangle-list' }
        })
        const bindGroup = device.createBindGroup({
          layout: pipe.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: tex.view }]
        })
        const encoder = device.createCommandEncoder()
        const pass = encoder.beginRenderPass({
          colorAttachments: [{ view: temp.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }]
        })
        pass.setPipeline(pipe)
        pass.setBindGroup(0, bindGroup)
        pass.draw(3)
        pass.end()
        device.queue.submit([encoder.finish()])
        backend.textures.set('__nm_capture', { handle: temp, width: tex.width, height: tex.height, gpuFormat: format })
        try {
          px = await backend.readPixels('__nm_capture')
        } finally {
          backend.textures.delete('__nm_capture')
        }
      }
    } finally {
      const err = await device.popErrorScope()
      if (temp) temp.destroy()
      if (err) return { error: `validation error while capturing ${id}: ${err.message}` }
    }
    let binary = ''
    const chunk = 0x8000
    for (let i = 0; i < px.data.length; i += chunk) {
      binary += String.fromCharCode.apply(null, px.data.subarray(i, i + chunk))
    }
    return { width: px.width, height: px.height, b64: btoa(binary) }
  }, { textureId })
  if (result.error) throw new Error(`readback failed: ${result.error}`)
  return encodePng(result.width, result.height, Buffer.from(result.b64, 'base64'))
}

// The graph the page rendered: Maps as objects, functions dropped, program
// specs without GLSL sources (WGSL kept).
async function pageGraph (page) {
  return page.evaluate(() => {
    const g = window.__noisemakerRenderingPipeline?.graph
    if (!g) return null
    const replacer = (_k, v) => {
      if (v instanceof Map) return Object.fromEntries(v)
      if (v instanceof Set) return [...v]
      return v
    }
    const plain = JSON.parse(JSON.stringify(g, replacer))
    for (const spec of Object.values(plain.programs || {})) {
      delete spec.glsl; delete spec.fragment; delete spec.vertex
    }
    delete plain.compiledAt
    return plain
  })
}

async function asyncOverlayIds (page) {
  const ids = await page.evaluate(() => {
    const p = window.__noisemakerRenderingPipeline
    const nodes = window.__nmAsyncInitNodes || new Set()
    const passes = p?.graph?.passes || []
    const written = new Set(passes.flatMap((pass) => Object.values(pass.outputs || {})))
    const found = new Set()
    for (const pass of passes) {
      for (const id of Object.values(pass.inputs || {})) {
        if (written.has(id) || !p.backend?.textures?.get(id)) continue
        for (const node of nodes) if (id.startsWith(`${node}_`)) found.add(id)
      }
    }
    return [...found]
  })
  return ids.filter((id) => ASYNC_OVERLAY_ID.test(id))
}

async function meshPlan (graph, dslPath) {
  const sidecar = dslPath.replace(/\.dsl$/, '.obj')
  const objText = existsSync(sidecar) ? readFileSync(sidecar, 'utf8') : null
  const readsMesh = (graph.passes || []).some(pass =>
    Object.values(pass.inputs || {}).some(id => typeof id === 'string' && MESH_TEXTURE_INPUT.test(id)))
  if (!readsMesh && objText === null) return null
  const builtins = []
  const seen = new Set()
  for (const pass of graph.passes || []) {
    const key = pass.effectKey
    if (!key || seen.has(`${pass.stepIndex}|${key}`)) continue
    seen.add(`${pass.stepIndex}|${key}`)
    const [ns, func] = key.split('.')
    const defPath = join(EFFECTS_DIR, ns, func, 'definition.js')
    if (!existsSync(defPath)) continue
    const mod = await import(pathToFileURL(defPath).href)
    const def = typeof mod.default === 'function' ? new mod.default() : mod.default
    if (!def?.externalMesh || !def.builtinMeshes) continue
    const first = Object.values(def.builtinMeshes)[0]
    if (first) builtins.push({ meshId: def.externalMesh, path: first })
  }
  return { objText, builtins }
}

async function applyMeshPlan (page, plan) {
  await page.waitForFunction(() => {
    if ((document.getElementById('status')?.textContent || '') !== 'compiled successfully') return false
    return [...document.querySelectorAll('.mesh-status')].every(el => el.textContent !== 'loading...')
  }, null, { timeout: STATUS_TIMEOUT })
  const results = await page.evaluate(async ({ objText, builtins }) => {
    const r = window.__noisemakerCanvasRenderer
    const out = []
    for (const b of builtins) out.push(await r.loadOBJFromURL(`${r._basePath}/${b.path}`, b.meshId))
    if (objText !== null) out.push(await r.loadOBJFromString(objText, 'mesh0'))
    else if (builtins.length === 0) out.push(await r.loadOBJFromString('', 'mesh0'))
    return out
  }, plan)
  const failed = results.filter(res => !res?.success)
  if (failed.length) throw new Error(`mesh load failed: ${JSON.stringify(failed)}`)
}

async function sizePage (page, size) {
  await page.evaluate(() => { if (window.__noisemakerSetPaused) window.__noisemakerSetPaused(true) })
  await page.evaluate((size) => {
    const r = window.__noisemakerCanvasRenderer
    const p = window.__noisemakerRenderingPipeline
    const canvas = r && r.canvas
    if (canvas) {
      const wd = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'width')
      const hd = Object.getOwnPropertyDescriptor(HTMLCanvasElement.prototype, 'height')
      if (wd && wd.set) wd.set.call(canvas, size)
      if (hd && hd.set) hd.set.call(canvas, size)
      for (const prop of ['width', 'height']) {
        Object.defineProperty(canvas, prop, { configurable: true, enumerable: true, get () { return size }, set () {} })
      }
      if (canvas.style) { canvas.style.width = size + 'px'; canvas.style.height = size + 'px' }
    }
    if (r && typeof r.resize === 'function') r.resize(size, size)
    else if (p && typeof p.resize === 'function') p.resize(size, size)
  }, size)
}

async function installAsyncInitTracker (page) {
  await page.evaluate(() => {
    const proto = Object.getPrototypeOf(window.__noisemakerRenderingPipeline)
    if (proto.__nmTracksAsyncInit) return
    if (typeof proto._startAsyncInit !== 'function') {
      throw new Error('reference pipeline has no _startAsyncInit; the asyncInit quiescence wait needs updating')
    }
    window.__nmAsyncInitPending = 0
    window.__nmAsyncInitNodes = new Set()
    const start = proto._startAsyncInit
    proto._startAsyncInit = function (nodeId, effectDef, options) {
      if (options && options.debounce) return start.call(this, nodeId, effectDef, options)
      window.__nmAsyncInitNodes.add(nodeId)
      const own = Object.prototype.hasOwnProperty.call(effectDef, 'asyncInit')
      const asyncInit = effectDef.asyncInit
      effectDef.asyncInit = function (context) {
        let pending
        try { pending = Promise.resolve(asyncInit.call(this, context)) } catch (err) { pending = Promise.reject(err) }
        window.__nmAsyncInitPending++
        const settle = () => { window.__nmAsyncInitPending-- }
        pending.then(settle, settle)
        return pending
      }
      try {
        return start.call(this, nodeId, effectDef, options)
      } finally {
        if (own) effectDef.asyncInit = asyncInit
        else delete effectDef.asyncInit
      }
    }
    proto.__nmTracksAsyncInit = true
  })
}

// Load `src` through the demo editor and wait until the page runs it.
async function runDsl (page, src, expectedPassCount) {
  await page.evaluate(({ src }) => {
    const programState = window.__noisemakerProgramState
    if (programState && Array.isArray(programState._structure)) programState._structure = []
    document.getElementById('status').textContent = ''
    const editor = document.getElementById('dsl-editor')
    editor.value = src
    editor.dispatchEvent(new Event('input', { bubbles: true }))
    document.getElementById('dsl-run-btn').click()
  }, { src })
  await page.waitForFunction(({ src, expectedPassCount }) => {
    const status = document.getElementById('status')?.textContent || ''
    if (/error|failed/i.test(status)) throw new Error('DSL compile failed: ' + status)
    const p = window.__noisemakerRenderingPipeline
    if (!(p && p.graph && p.graph.source === src.trim() && !p.isCompiling)) return false
    if (p.backend?.getName?.() !== 'WebGPU') return false
    if (typeof expectedPassCount === 'number' && p.graph.passes?.length !== expectedPassCount) return false
    return /compiled/i.test(status)
  }, { src, expectedPassCount }, { timeout: STATUS_TIMEOUT })
}

async function mintOne (page, opts, dslPath, dsl, expectedPassCount, programName) {
  await sizePage(page, opts.size)
  await installAsyncInitTracker(page)
  await page.evaluate(() => { window.__nmAsyncInitNodes = new Set() })
  await runDsl(page, dsl, expectedPassCount)

  const graph = await pageGraph(page)
  const meshes = await meshPlan(graph, dslPath)
  if (meshes) await applyMeshPlan(page, meshes)

  const loadedSize = await page.evaluate(() => {
    const p = window.__noisemakerRenderingPipeline
    const surf = p.surfaces && p.surfaces.get(p.graph?.renderSurface || 'o0')
    const info = surf && p.backend?.textures?.get(surf.read)
    return info ? [info.width, info.height] : null
  })
  if (!loadedSize || loadedSize[0] !== opts.size || loadedSize[1] !== opts.size) {
    throw new Error(`render surface is ${JSON.stringify(loadedSize)} after the DSL load, expected ${opts.size}x${opts.size}`)
  }

  const externalIds = [...new Set(graph.passes.flatMap((pass) =>
    Object.values(pass.inputs || {}).filter((id) => EXTERNAL_TEXTURE_ID.test(id))))]
  await page.waitForFunction((ids) => {
    const p = window.__noisemakerRenderingPipeline
    if (!p) return false
    if (p._asyncDebounceTimers && p._asyncDebounceTimers.size > 0) return false
    if (window.__nmAsyncInitPending > 0) return false
    return ids.every((id) => !!p.backend?.textures?.get(id))
  }, externalIds, { timeout: STATUS_TIMEOUT })
  const hostTextures = []
  for (const id of externalIds) {
    writeFileSync(join(opts.outDir, `${programName}.${id}.png`), await capture(page, id))
    hostTextures.push(id)
  }
  for (const id of await asyncOverlayIds(page)) {
    writeFileSync(join(opts.outDir, `${programName}.${id}.png`), await capture(page, id))
    hostTextures.push(id)
  }

  // Fresh pipeline state (see the header): clear every pipeline-written texture.
  const reset = await page.evaluate(({ keep }) => {
    const p = window.__noisemakerRenderingPipeline
    const backend = p?.backend
    if (!backend?.textures || typeof backend.clearTexture !== 'function') return { error: 'backend cannot clear textures' }
    const keepIds = new Set(keep)
    const isKept = (id) => keepIds.has(id) || /^global_mesh\d+_(positions|normals|uvs)$/.test(id) ||
      id === 'midiNoteGrid' || backend.textures.get(id)?.isExternal === true
    let cleared = 0
    for (const id of backend.textures.keys()) {
      if (isKept(id)) continue
      const tex = backend.textures.get(id)
      if (tex?.is3D || tex?.cube) continue
      backend.clearTexture(id)
      cleared++
    }
    for (const [name, surface] of p.surfaces.entries()) {
      if (backend.textures.get(`global_${name}_read`) && backend.textures.get(`global_${name}_write`)) {
        surface.read = `global_${name}_read`
        surface.write = `global_${name}_write`
      }
    }
    const staleGlobals = Object.keys(p.globalUniforms || {})
    p.globalUniforms = {}
    p.frameIndex = 0
    p.lastTime = 0
    return { cleared, staleGlobals }
  }, { keep: hostTextures })
  if (reset.error) throw new Error(`state reset failed: ${reset.error}`)

  if (opts.runSeconds > 0) {
    const everyFrames = opts.sampleEvery * 60
    const samples = Math.max(1, Math.floor((opts.runSeconds * 60) / everyFrames))
    for (let s = 0; s < samples; s++) {
      await page.evaluate(({ everyFrames, startFrame }) => {
        const p = window.__noisemakerRenderingPipeline
        for (let i = 0; i < everyFrames; i++) p.render(((startFrame + i + 1) / 600) % 1.0)
      }, { everyFrames, startFrame: s * everyFrames })
      const sec = (s + 1) * opts.sampleEvery
      writeFileSync(join(opts.outDir, `${programName}.golden.t${sec}.png`), await capture(page))
    }
  } else {
    await page.evaluate(({ time, frames }) => {
      if (window.__noisemakerSetPausedTime) window.__noisemakerSetPausedTime(time)
      const p = window.__noisemakerRenderingPipeline
      for (let i = 0; i < frames; i++) p.render(time)
    }, { time: opts.time, frames: opts.frames })
    writeFileSync(join(opts.outDir, `${programName}.golden.png`), await capture(page))
  }
  writeFileSync(join(opts.outDir, `${programName}.graph.json`), JSON.stringify(graph, null, 2) + '\n')
  return { hostTextures, reset }
}

// A fixture the reference cannot render (for example a texture size the WebGPU
// backend rejects) leaves its graph in the page, and the next fixture's resize
// would fail on it. Load a known-good program so the next fixture starts from a
// working pipeline; if that fails too, the caller restarts the session.
const RECOVERY_DSL = 'search synth\nsolid().write(o0)\nrender(o0)'
async function recoverPage (page) {
  await runDsl(page, RECOVERY_DSL, 2)
}

// ---- session ---------------------------------------------------------------

async function withSession (opts, fn) {
  process.env.SHADE_VIEWER_ROOT = REFERENCE_ROOT
  process.env.SHADE_VIEWER_PATH = VIEWER_PATH
  process.env.SHADE_EFFECTS_DIR = EFFECTS_DIR
  process.env.SHADE_GLOBALS_PREFIX = GLOBALS_PREFIX
  process.env.SHADE_HEADLESS = process.env.SHADE_HEADLESS ?? '1'
  const { BrowserSession } = await import(pathToFileURL(HARNESS).href)
  const session = new BrowserSession({ backend: 'webgpu' })
  await session.setup()
  try {
    const page = session.page
    await session.setBackend('webgpu')
    await page.setViewportSize({ width: opts.size, height: opts.size })
    await page.waitForFunction(() => !!document.getElementById('dsl-editor') && !!document.getElementById('dsl-run-btn'),
      null, { timeout: STATUS_TIMEOUT })
    const adapter = await page.evaluate(async () => {
      const a = await navigator.gpu?.requestAdapter()
      return a ? `${a.info?.vendor} ${a.info?.architecture} ${a.info?.description}` : null
    })
    process.stderr.write(`[batch-golden] golden WebGPU adapter: ${adapter}\n`)
    return await fn(session, page)
  } finally {
    await session.teardown()
  }
}

function parseArgs (argv) {
  const opts = { size: 256, time: 0.25, frames: 8, chunkSize: 60, list: null, runSeconds: 0, sampleEvery: 5, dslPaths: [] }
  const pos = []
  let rest = false
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (rest) { opts.dslPaths.push(a); continue }
    if (a === '--') rest = true
    else if (a === '--size') opts.size = parseInt(argv[++i], 10)
    else if (a === '--time') opts.time = parseFloat(argv[++i])
    else if (a === '--frames') opts.frames = parseInt(argv[++i], 10)
    else if (a === '--chunk-size') opts.chunkSize = parseInt(argv[++i], 10)
    else if (a === '--list') opts.list = argv[++i]
    else if (a === '--run-seconds') opts.runSeconds = parseInt(argv[++i], 10)
    else if (a === '--sample-every') opts.sampleEvery = parseInt(argv[++i], 10)
    else if (a.endsWith('.dsl')) opts.dslPaths.push(a)
    else pos.push(a)
  }
  opts.outDir = pos[0]
  return opts
}

async function main () {
  const opts = parseArgs(process.argv.slice(2))
  if (!opts.outDir) {
    console.error('usage: node parity/batch-golden.mjs <outDir> [--size 256] [--time 0.25] [--frames 8] ' +
      '[--chunk-size 60] [--run-seconds N --sample-every S] [--list names.txt] [--] prog.dsl...')
    process.exit(2)
  }
  mkdirSync(opts.outDir, { recursive: true })
  const listed = opts.list
    ? readFileSync(opts.list, 'utf8').split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'))
    : []
  const dslPaths = [...listed, ...opts.dslPaths].map(p => resolve(p))
  if (!dslPaths.length) {
    console.error('[batch-golden] no DSL paths given')
    process.exit(2)
  }

  // Expected pass counts come from the reference compileGraph, run in Node.
  const { bootstrapReference } = await import(pathToFileURL(join(ROOT, 'tools', 'reference-oracle.mjs')).href)
  const ref = await bootstrapReference()

  const minted = new Set()
  const failed = new Map()
  const total = dslPaths.length
  const t0All = Date.now()
  // Round 0 mints everything; the repair round retries every fixture not minted
  // yet (failed fixtures and fixtures of an aborted session) in fresh sessions.
  for (let round = 0; round < 2; round++) {
    const pending = dslPaths.filter(p => !minted.has(p))
    if (!pending.length) break
    const attempted = new Set()
    if (round > 0) process.stderr.write(`[batch-golden] repair round: retrying ${pending.length} fixture(s) in fresh sessions\n`)
    for (let start = 0; start < dslPaths.length; start += opts.chunkSize) {
      const chunk = dslPaths.slice(start, start + opts.chunkSize).filter(p => !minted.has(p))
      if (!chunk.length) continue
      let attempt = 0
      for (;;) {
        try {
          await withSession(opts, async (session, page) => {
            for (const dslPath of chunk) {
              if (minted.has(dslPath) || attempted.has(dslPath)) continue
              attempted.add(dslPath)
              const programName = basename(dslPath).replace(/\.dsl$/, '')
              const t0 = Date.now()
              try {
                const dsl = readFileSync(dslPath, 'utf8')
                const expectedPassCount = ref.compileGraph(dsl).passes.length
                for (const file of readdirSync(opts.outDir)) {
                  if (file.startsWith(`${programName}.`) && file.endsWith('.png')) unlinkSync(join(opts.outDir, file))
                }
                const info = await mintOne(page, opts, dslPath, dsl, expectedPassCount, programName)
                const notes = session.getConsoleMessages().map(m => m.text)
                session.clearConsoleMessages()
                process.stderr.write(`[batch-golden] (${minted.size + 1}/${total}) ${programName}: ok ${Date.now() - t0}ms` +
                  (info.hostTextures.length ? ` host=[${info.hostTextures.join(',')}]` : '') +
                  (notes.length ? ` [console: ${notes.join(' | ').slice(0, 400)}]` : '') + '\n')
                minted.add(dslPath)
                failed.delete(dslPath)
              } catch (err) {
                const msg = err?.message || String(err)
                process.stderr.write(`[batch-golden] ${programName}: FAILED after ${Date.now() - t0}ms: ${msg.slice(0, 600)}\n`)
                failed.set(dslPath, msg)
                if (/Target (page|closed)|Target crashed|has been closed/i.test(msg)) throw err
                // A failed recovery means the page is unusable: restart the session.
                await recoverPage(page)
              }
            }
          })
          break
        } catch (err) {
          process.stderr.write(`[batch-golden] session aborted: ${(err?.message || err).toString().slice(0, 300)}\n`)
          if (++attempt > 2) break
        }
      }
    }
  }
  for (const p of dslPaths) {
    if (!minted.has(p) && !failed.has(p)) failed.set(p, 'not minted (session aborted)')
  }
  process.stderr.write(`[batch-golden] minted=${minted.size} failed=${failed.size} total=${total} (${((Date.now() - t0All) / 1000).toFixed(1)}s)\n`)
  if (failed.size) process.stderr.write(`[batch-golden] FAILED: ${[...failed.keys()].map(p => basename(p, '.dsl')).join(' ')}\n`)
  process.stdout.write(`BATCH-GOLDEN: minted=${minted.size} failed=${failed.size} total=${total}\nDONE\n`)
  process.exit(failed.size ? 1 : 0)
}

main().catch(err => {
  process.stderr.write(`[batch-golden] FATAL: ${err?.stack || err}\n`)
  process.exit(1)
})
