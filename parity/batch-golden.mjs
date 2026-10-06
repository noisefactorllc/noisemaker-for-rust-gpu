#!/usr/bin/env node
// batch-golden.mjs — mint reference goldens on the reference engine's WebGPU backend.
//
// The golden of a fixture is the reference engine's own render of its DSL, in the
// reference demo page (/demo/shaders/), on the WebGPU backend, read back from the
// presented surface. One Chromium session (the vendored shade-mcp harness) serves
// a chunk of fixtures; each fixture runs this protocol:
//
//   1. pause the demo's render loop, pin the canvas and pipeline to SIZE x SIZE;
//   1b. a fixture with a Portable sidecar (<name>.portable.json, its WGSL in
//      <name>.<program>.wgsl; tools/portable.mjs) registers that user effect
//      with the page's CanvasRenderer.registerPortableEffect (and in the Node
//      realm that predicts the pass count) before its DSL loads;
//   2. load the DSL through the demo's editor and run button, wait until the
//      pipeline runs exactly that source (graph.source, pass count from the
//      reference compileGraph, compilation finished, WebGPU backend);
//   3. load the meshes a mesh fixture needs (built-in defaults of each
//      externalMesh step, then the fixture's .obj sidecar into mesh0, as the
//      reference demo host does);
//   3b. a fixture with a MIDI sidecar (<name>.midi.json: {"messages":
//      [[status, data1, data2], ...]}) connects a MIDI state to the renderer
//      (CanvasRenderer.setMidiState()) and delivers those raw messages to it
//      (MidiState.handleMessage), a deterministic stand-in for a device; every
//      other fixture runs with no MIDI state, as the headless demo page does;
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
// Timed fixtures (--run-seconds N --sample-every S, --run-frames N,
// --sample-every-frames K, --sample-frames a,b,c) skip the fresh-state reset's
// pinned clock: they step pipeline.render(t) with t = ((frame + 1) / 600) % 1
// and save <name>.golden.t<sec>.png every S seconds and
// <name>.golden.f<frames>.png every K frames and after each listed frame
// count (nm-render's sample schedule).
//
// Exact intermediate state: --dump-texture ID (repeatable) reads a backend
// texture, or a global surface's current read texture (ID = the surface name
// or global_<name>), back as raw little-endian float32 RGBA at every sample:
// <name>.golden[.<label>].<ID>.bin. --dump-passes also snapshots those
// textures after every executed pass of each sampled frame, copied in the
// frame's own command encoder: <name>.golden[.<label>].p<NNN>.<ID>.bin, the
// pass list in <name>.golden[.<label>].passes.json. Texels are decoded from
// the texture's format in the page (half floats widened exactly, 8-bit
// channels as byte / 255) and leave the page as a binary upload, like the
// PNG captures; nm-render --dump-texture/--dump-passes writes the same files.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/batch-golden.mjs <outDir> \
//       [--size 256] [--time 0.25] [--frames 8] [--chunk-size 60] \
//       [--run-seconds N --sample-every S] [--run-frames N] \
//       [--sample-every-frames K] [--sample-frames a,b,c] \
//       [--dump-texture ID ...] [--dump-passes] [--list names.txt] [--] prog.dsl...
//
// Exit 0 when every fixture minted, 1 otherwise. The last stdout lines are
// "BATCH-GOLDEN: minted=N failed=M total=T" and "DONE".

import { existsSync, mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { deflateSync } from 'node:zlib'

import { loadPortableDefinition, portableSidecar } from '../tools/portable.mjs'

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
//
// The pixels leave the page as a binary upload: the page POSTs the readback's
// bytes to /__nm_capture/<token>, which installCaptureRoute() intercepts and
// keeps as a Buffer. Images never travel as text.
const captureUploads = new Map()
let captureCounter = 0

async function installCaptureRoute (page) {
  await page.route('**/__nm_capture/*', async (route) => {
    const token = route.request().url().split('/__nm_capture/')[1]
    captureUploads.set(token, route.request().postDataBuffer())
    await route.fulfill({ status: 204 })
  })
}

async function capture (page, textureId = null) {
  const token = String(++captureCounter)
  const result = await page.evaluate(async ({ textureId, token }) => {
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
    const upload = await fetch(`/__nm_capture/${token}`, {
      method: 'POST',
      headers: { 'content-type': 'application/octet-stream' },
      body: new Blob([px.data])
    })
    if (upload.status !== 204) return { error: `capture upload failed (${upload.status})` }
    return { width: px.width, height: px.height }
  }, { textureId, token })
  if (result.error) throw new Error(`readback failed: ${result.error}`)
  const pixels = captureUploads.get(token)
  captureUploads.delete(token)
  if (!pixels || pixels.length !== result.width * result.height * 4) {
    throw new Error(`readback failed: received ${pixels ? pixels.length : 0} bytes for ${result.width}x${result.height}`)
  }
  return encodePng(result.width, result.height, pixels)
}

// ---- exact state probes ------------------------------------------------------

// In-page helpers of the state probes, installed once per page: resolving a
// probe id, decoding a texture's texels to float32 RGBA, the pass probe.
async function installProbeHelpers (page) {
  await page.evaluate(() => {
    if (window.__nmProbe) return
    const layouts = {
      r8unorm: [1, 'u8', 1], rg8unorm: [2, 'u8', 2], rgba8unorm: [4, 'u8', 4], 'rgba8unorm-srgb': [4, 'u8', 4],
      bgra8unorm: [4, 'bgra', 4], 'bgra8unorm-srgb': [4, 'bgra', 4],
      r16float: [2, 'f16', 1], rg16float: [4, 'f16', 2], rgba16float: [8, 'f16', 4],
      r32float: [4, 'f32', 1], rg32float: [8, 'f32', 2], rgba32float: [16, 'f32', 4],
      r32uint: [4, 'u32', 1], r32sint: [4, 'i32', 1]
    }
    const halfToFloat = (h) => {
      const sign = (h & 0x8000) ? -1 : 1
      const e = (h >> 10) & 0x1f
      const m = h & 0x3ff
      if (e === 0) return sign * m * 2 ** -24
      if (e === 31) return m ? NaN : sign * Infinity
      return sign * (1 + m / 1024) * 2 ** (e - 15)
    }
    const probe = {
      // A global surface (by name or global_<name>): its current read texture
      // (within a frame, the frame-local read binding); else a backend texture.
      resolve (p, id, inFrame) {
        let name = null
        if (p.surfaces?.has(id)) name = id
        else if (id.startsWith('global_') && p.surfaces?.has(id.slice(7))) name = id.slice(7)
        if (name !== null) {
          const frame = inFrame ? p.frameReadTextures?.get(name) : undefined
          return frame || p.surfaces.get(name).read || null
        }
        return p.backend.textures.has(id) ? id : null
      },
      // Read a GPUTexture (level 0) back as float32 RGBA.
      async readFloat (device, texture) {
        const layout = layouts[texture.format]
        if (!layout) throw new Error(`cannot probe a ${texture.format} texture`)
        const [bpp, kind, count] = layout
        const width = texture.width
        const height = texture.height
        const bytesPerRow = Math.ceil((width * bpp) / 256) * 256
        const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
        const encoder = device.createCommandEncoder()
        encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height, depthOrArrayLayers: 1 })
        device.queue.submit([encoder.finish()])
        await buffer.mapAsync(GPUMapMode.READ)
        const bytes = new Uint8Array(buffer.getMappedRange().slice(0))
        buffer.unmap()
        buffer.destroy()
        const view = new DataView(bytes.buffer)
        const out = new Float32Array(width * height * 4)
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const at = y * bytesPerRow + x * bpp
            const o = (y * width + x) * 4
            out[o + 3] = 1
            for (let c = 0; c < count; c++) {
              let v
              if (kind === 'u8') v = bytes[at + c] / 255
              else if (kind === 'bgra') v = bytes[at + [2, 1, 0, 3][c]] / 255
              else if (kind === 'f16') v = halfToFloat(view.getUint16(at + c * 2, true))
              else if (kind === 'f32') v = view.getFloat32(at + c * 4, true)
              else if (kind === 'u32') v = view.getUint32(at + c * 4, true)
              else v = view.getInt32(at + c * 4, true)
              out[o + c] = v
            }
          }
        }
        return { width, height, data: out }
      },
      async upload (token, data) {
        const res = await fetch(`/__nm_capture/${token}`, {
          method: 'POST',
          headers: { 'content-type': 'application/octet-stream' },
          body: new Blob([data.buffer])
        })
        if (res.status !== 204) throw new Error(`probe upload failed (${res.status})`)
      },
      // Arm the pass probe for the next frame: after every executed pass
      // (updateFrameSurfaceBindings runs once per executed pass and repeat
      // iteration) copy each id's texture in the frame's command encoder.
      arm (p, ids) {
        const snaps = []
        let ordinal = 0
        const original = Object.getPrototypeOf(p).updateFrameSurfaceBindings
        p.updateFrameSurfaceBindings = function (pass, state) {
          const result = original.call(this, pass, state)
          const backend = this.backend
          for (const id of ids) {
            const texId = probe.resolve(this, id, true)
            const rec = texId ? backend.textures.get(texId) : null
            const handle = rec?.handle
            if (!handle || rec.is3D || rec.cube || !(handle.usage & GPUTextureUsage.COPY_SRC)) continue
            const size = { width: handle.width, height: handle.height, depthOrArrayLayers: 1 }
            const snap = backend.device.createTexture({ size, format: handle.format, usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC })
            const encoder = backend.commandEncoder || backend.device.createCommandEncoder()
            encoder.copyTextureToTexture({ texture: handle }, { texture: snap }, size)
            if (encoder !== backend.commandEncoder) backend.device.queue.submit([encoder.finish()])
            snaps.push({ ordinal, passId: pass.id, id, texId, snap })
          }
          ordinal++
          return result
        }
        return snaps
      },
      disarm (p) { delete p.updateFrameSurfaceBindings }
    }
    window.__nmProbe = probe
  })
}

// Read the dump textures of the current state back as float32 RGBA and
// write <prefix><id>.bin for each.
async function dumpState (page, ids, prefix) {
  for (const id of ids) {
    const token = String(++captureCounter)
    const result = await page.evaluate(async ({ id, token }) => {
      const p = window.__noisemakerRenderingPipeline
      const texId = window.__nmProbe.resolve(p, id, false)
      const rec = texId ? p.backend.textures.get(texId) : null
      if (!rec?.handle) return { error: `no such texture or surface ${id}` }
      if (!(rec.handle.usage & GPUTextureUsage.COPY_SRC)) return { error: `texture ${texId} has no COPY_SRC usage` }
      const px = await window.__nmProbe.readFloat(p.backend.device, rec.handle)
      await window.__nmProbe.upload(token, px.data)
      return { width: px.width, height: px.height }
    }, { id, token })
    if (result.error) throw new Error(`dump texture ${id}: ${result.error}`)
    const bytes = captureUploads.get(token)
    captureUploads.delete(token)
    if (!bytes || bytes.length !== result.width * result.height * 16) {
      throw new Error(`dump texture ${id}: received ${bytes ? bytes.length : 0} bytes for ${result.width}x${result.height}`)
    }
    writeFileSync(`${prefix}${fileId(id)}.bin`, bytes)
  }
}

// Read the armed frame's pass snapshots back: <prefix>p<NNN>.<id>.bin and
// the pass list <prefix>passes.json.
async function dumpPassSnapshots (page, prefix) {
  const count = await page.evaluate(() => (window.__nmProbeSnaps || []).length)
  const passes = []
  for (let i = 0; i < count; i++) {
    const token = String(++captureCounter)
    const info = await page.evaluate(async ({ i, token }) => {
      const p = window.__noisemakerRenderingPipeline
      const s = window.__nmProbeSnaps[i]
      const px = await window.__nmProbe.readFloat(p.backend.device, s.snap)
      s.snap.destroy()
      await window.__nmProbe.upload(token, px.data)
      return { ordinal: s.ordinal, passId: s.passId, id: s.id, texId: s.texId, width: px.width, height: px.height }
    }, { i, token })
    const bytes = captureUploads.get(token)
    captureUploads.delete(token)
    if (!bytes || bytes.length !== info.width * info.height * 16) {
      throw new Error(`pass snapshot ${info.id}: received ${bytes ? bytes.length : 0} bytes`)
    }
    writeFileSync(`${prefix}p${String(info.ordinal).padStart(3, '0')}.${fileId(info.id)}.bin`, bytes)
    let entry = passes.find(e => e.ordinal === info.ordinal)
    if (!entry) passes.push(entry = { ordinal: info.ordinal, passId: info.passId, textures: {} })
    entry.textures[info.id] = info.texId
  }
  await page.evaluate(() => { window.__nmProbeSnaps = [] })
  writeFileSync(`${prefix}passes.json`, JSON.stringify(passes, null, 2) + '\n')
}

function fileId (id) {
  return id.replace(/[/\\]/g, '_')
}

// The timed run's samples in frame order: [{ frame, labels }] (nm-render's
// FixtureSpec::sample_schedule).
function sampleSchedule (opts) {
  const samples = []
  const add = (frame, label) => {
    if (frame <= 0) return
    const s = samples.find(x => x.frame === frame)
    if (s) { if (!s.labels.includes(label)) s.labels.push(label) } else samples.push({ frame, labels: [label] })
  }
  let total = opts.runFrames
  if (opts.runSeconds > 0) {
    const everyFrames = Math.max(1, Math.round(opts.sampleEvery * 60))
    const count = Math.max(1, Math.floor((opts.runSeconds * 60) / everyFrames))
    for (let s = 0; s < count; s++) add((s + 1) * everyFrames, `t${(s + 1) * opts.sampleEvery}`)
    total = Math.max(total, count * everyFrames)
  }
  if (opts.sampleEveryFrames > 0) {
    for (let f = opts.sampleEveryFrames; f <= total; f += opts.sampleEveryFrames) add(f, `f${f}`)
  }
  for (const f of opts.sampleFrames) add(f, `f${f}`)
  return samples.sort((a, b) => a.frame - b.frame)
}

function isTimed (opts) {
  return opts.runSeconds > 0 || opts.runFrames > 0 || opts.sampleFrames.length > 0
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

// Register a fixture's Portable effect with the page's renderer (once per
// session: the page's registries keep it).
async function registerPortable (page, definition) {
  const result = await page.evaluate(async (def) => {
    const r = window.__noisemakerCanvasRenderer
    if (!r || typeof r.registerPortableEffect !== 'function') return { error: 'the page renderer has no registerPortableEffect' }
    const func = def.func ?? def.name
    if (r.loadedEffects?.has(`user/${func}`)) return { ok: true }
    try {
      await r.registerPortableEffect(def)
      return { ok: true }
    } catch (err) {
      return { error: err?.message || String(err) }
    }
  }, definition)
  if (result.error) throw new Error(`registerPortableEffect failed: ${result.error}`)
}

// The MIDI messages of a fixture's sidecar, or null.
function midiMessages (dslPath) {
  const sidecar = dslPath.replace(/\.dsl$/, '.midi.json')
  if (!existsSync(sidecar)) return null
  const messages = JSON.parse(readFileSync(sidecar, 'utf8')).messages
  if (!Array.isArray(messages) || !messages.every(m => Array.isArray(m) && m.length > 0 &&
    m.every(b => Number.isInteger(b) && b >= 0 && b <= 255))) {
    throw new Error(`${sidecar}: "messages" must be an array of byte arrays`)
  }
  return messages
}

async function mintOne (page, opts, dslPath, dsl, expectedPassCount, programName, portable) {
  await sizePage(page, opts.size)
  await installAsyncInitTracker(page)
  await page.evaluate(() => {
    window.__nmAsyncInitNodes = new Set()
    // No MIDI state carried over from an earlier fixture of this session.
    const r = window.__noisemakerCanvasRenderer
    if (r && r._midiState) {
      r._midiState = null
      window.__noisemakerRenderingPipeline?.setMidiState?.(null)
    }
  })
  if (portable) await registerPortable(page, portable)
  await runDsl(page, dsl, expectedPassCount)

  const graph = await pageGraph(page)
  const meshes = await meshPlan(graph, dslPath)
  if (meshes) await applyMeshPlan(page, meshes)
  const midi = midiMessages(dslPath)
  if (midi) {
    await page.evaluate((messages) => {
      const state = window.__noisemakerCanvasRenderer.setMidiState()
      for (const m of messages) state.handleMessage(new Uint8Array(m))
    }, midi)
  }

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

  const probePasses = opts.dumpPasses && opts.dumpTextures.length > 0
  if (opts.dumpTextures.length) await installProbeHelpers(page)
  const base = join(opts.outDir, `${programName}.golden`)
  if (isTimed(opts)) {
    let frame = 0
    for (const sample of sampleSchedule(opts)) {
      await page.evaluate(({ from, to, probe, ids }) => {
        const p = window.__noisemakerRenderingPipeline
        for (let f = from; f < to; f++) {
          const armed = probe && f + 1 === to
          if (armed) window.__nmProbeSnaps = window.__nmProbe.arm(p, ids)
          try {
            p.render(((f + 1) / 600) % 1.0)
          } finally {
            if (armed) window.__nmProbe.disarm(p)
          }
        }
      }, { from: frame, to: sample.frame, probe: probePasses, ids: opts.dumpTextures })
      frame = sample.frame
      const png = await capture(page)
      for (const label of sample.labels) {
        writeFileSync(`${base}.${label}.png`, png)
        await dumpState(page, opts.dumpTextures, `${base}.${label}.`)
      }
      if (probePasses) await dumpPassSnapshots(page, `${base}.${sample.labels[0]}.`)
    }
  } else {
    await page.evaluate(({ time, frames, probe, ids }) => {
      if (window.__noisemakerSetPausedTime) window.__noisemakerSetPausedTime(time)
      const p = window.__noisemakerRenderingPipeline
      for (let i = 0; i < frames; i++) {
        const armed = probe && i + 1 === frames
        if (armed) window.__nmProbeSnaps = window.__nmProbe.arm(p, ids)
        try {
          p.render(time)
        } finally {
          if (armed) window.__nmProbe.disarm(p)
        }
      }
    }, { time: opts.time, frames: opts.frames, probe: probePasses, ids: opts.dumpTextures })
    writeFileSync(`${base}.png`, await capture(page))
    await dumpState(page, opts.dumpTextures, `${base}.`)
    if (probePasses) await dumpPassSnapshots(page, `${base}.`)
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
    await installCaptureRoute(page)
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
  const opts = {
    size: 256, time: 0.25, frames: 8, chunkSize: 60, list: null, runSeconds: 0, sampleEvery: 5,
    runFrames: 0, sampleEveryFrames: 0, sampleFrames: [], dumpTextures: [], dumpPasses: false, dslPaths: []
  }
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
    else if (a === '--run-seconds') opts.runSeconds = parseFloat(argv[++i])
    else if (a === '--sample-every') opts.sampleEvery = parseFloat(argv[++i])
    else if (a === '--run-frames') opts.runFrames = parseInt(argv[++i], 10)
    else if (a === '--sample-every-frames') opts.sampleEveryFrames = parseInt(argv[++i], 10)
    else if (a === '--sample-frames') opts.sampleFrames.push(...argv[++i].split(',').map(n => parseInt(n, 10)))
    else if (a === '--dump-texture') opts.dumpTextures.push(argv[++i])
    else if (a === '--dump-passes') opts.dumpPasses = true
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
      '[--chunk-size 60] [--run-seconds N --sample-every S] [--run-frames N] [--sample-every-frames K] ' +
      '[--sample-frames a,b,c] [--dump-texture ID ...] [--dump-passes] [--list names.txt] [--] prog.dsl...')
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

  // Portable sidecars, registered in the Node realm once (it predicts the
  // pass counts) and in each page session before their fixture.
  const portables = new Map()
  const portableFor = async (dslPath) => {
    if (!portables.has(dslPath)) {
      const sidecar = portableSidecar(dslPath)
      const def = sidecar ? loadPortableDefinition(sidecar) : null
      if (def) await new ref.CanvasRenderer().registerPortableEffect(structuredClone(def))
      portables.set(dslPath, def)
    }
    return portables.get(dslPath)
  }

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
                const portable = await portableFor(dslPath)
                const expectedPassCount = ref.compileGraph(dsl).passes.length
                for (const file of readdirSync(opts.outDir)) {
                  if (file.startsWith(`${programName}.`) && /\.(png|bin)$|\.passes\.json$/.test(file)) unlinkSync(join(opts.outDir, file))
                }
                const info = await mintOne(page, opts, dslPath, dsl, expectedPassCount, programName, portable)
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
