#!/usr/bin/env node
// convert-effects.mjs — generate the embedded effect catalog from the reference.
//
// Reads every shaders/effects/<namespace>/<effect>/definition.js of the reference
// engine at $NM_REFERENCE_ROOT and writes, under
// crates/noisemaker-effects/catalog/:
//
//   <namespace>/<effect>/definition.json   the definition instance, serialized with
//                                          JSON.stringify semantics (own enumerable
//                                          properties in JavaScript order)
//   <namespace>/<effect>/wgsl/<prog>.wgsl  byte copies of the reference WGSL
//   manifest.json                          byte copy of shaders/effects/manifest.json
//   share/meshes/*.obj                     byte copies of the built-in meshes that
//                                          definitions reference (builtinMeshes)
//   share/palettes.json                    byte copy of the reference palette table
//                                          (share/palettes.json; the palette enum)
//   share/fonts/<family>/*                 byte copies of the fonts the reference demo
//                                          host serves for canvas text (demo/font:
//                                          Nunito, filter/text's default family) and
//                                          their licenses (OFL.txt)
//
// Behavior that a definition carries as JavaScript functions cannot be serialized.
// Those functions are listed by name in the definition's "jsHooks" array so the
// runtime can bind its native implementations; any other non-JSON value
// (NaN, Infinity, Map, Set, typed arrays, symbols, bigints) fails the conversion.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node tools/convert-effects.mjs [--out DIR]
//
// --out writes the catalog somewhere else (the freshness gate,
// parity/check_effects.mjs, regenerates into a temporary directory and compares).

import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const SHADERS_DIR = join(REFERENCE_ROOT, 'shaders')
const EFFECTS_DIR = join(SHADERS_DIR, 'effects')
// demo/font/<family>/<file> copied to share/fonts/<family>/<file>
const DEMO_FONTS = [['Nunito', ['Nunito-VariableFont_wght.ttf', 'OFL.txt']]]

function parseArgs (argv) {
  const opts = { out: join(REPO, 'crates', 'noisemaker-effects', 'catalog') }
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') opts.out = resolve(argv[++i])
    else throw new Error(`unknown argument ${argv[i]}`)
  }
  return opts
}

function sortedDirs (dir) {
  return readdirSync(dir, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name)
    .sort()
}

// Collect function-valued members (own, then up the prototype chain until the
// Effect base class). The base-class methods are no-op defaults and never count.
function jsHooks (instance, Effect) {
  const hooks = []
  for (const key of Object.keys(instance)) {
    if (typeof instance[key] === 'function') hooks.push(key)
  }
  let proto = Object.getPrototypeOf(instance)
  while (proto && proto !== Effect.prototype && proto !== Object.prototype) {
    for (const key of Object.getOwnPropertyNames(proto)) {
      if (key !== 'constructor' && typeof proto[key] === 'function' && !hooks.includes(key)) hooks.push(key)
    }
    proto = Object.getPrototypeOf(proto)
  }
  // Config-object hooks are stored as _configX by the Effect constructor.
  for (const [configKey, hook] of [['_configOnInit', 'onInit'], ['_configOnUpdate', 'onUpdate'],
    ['_configOnDestroy', 'onDestroy'], ['_configAsyncInit', 'asyncInit']]) {
    if (typeof instance[configKey] === 'function' && !hooks.includes(hook)) hooks.push(hook)
  }
  return hooks.filter(h => !h.startsWith('_config'))
}

// Reject values JSON.stringify would silently corrupt.
function assertJsonSafe (value, path) {
  if (value === null || value === undefined) return
  const t = typeof value
  if (t === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${path}: non-finite number ${value}`)
    return
  }
  if (t === 'string' || t === 'boolean' || t === 'function') return
  if (t === 'bigint' || t === 'symbol') throw new Error(`${path}: unsupported ${t}`)
  if (value instanceof Map || value instanceof Set || ArrayBuffer.isView(value)) {
    throw new Error(`${path}: unsupported ${value.constructor.name}`)
  }
  if (Array.isArray(value)) {
    value.forEach((v, i) => {
      if (typeof v === 'function') throw new Error(`${path}[${i}]: function inside an array`)
      assertJsonSafe(v, `${path}[${i}]`)
    })
    return
  }
  for (const [k, v] of Object.entries(value)) assertJsonSafe(v, `${path}.${k}`)
}

async function main () {
  const opts = parseArgs(process.argv.slice(2))
  const { Effect } = await import(pathToFileURL(join(SHADERS_DIR, 'src', 'runtime', 'effect.js')).href)

  rmSync(opts.out, { recursive: true, force: true })
  mkdirSync(opts.out, { recursive: true })

  let effects = 0
  let programs = 0
  const meshes = new Set()
  for (const namespace of sortedDirs(EFFECTS_DIR)) {
    for (const name of sortedDirs(join(EFFECTS_DIR, namespace))) {
      const srcDir = join(EFFECTS_DIR, namespace, name)
      const defPath = join(srcDir, 'definition.js')
      if (!existsSync(defPath)) continue
      const mod = await import(pathToFileURL(defPath).href)
      const instance = typeof mod.default === 'function' ? new mod.default() : mod.default
      if (!instance || typeof instance !== 'object') throw new Error(`${namespace}/${name}: no definition instance`)

      assertJsonSafe(instance, `${namespace}/${name}`)
      const hooks = jsHooks(instance, Effect)
      const json = JSON.parse(JSON.stringify(instance))
      if (hooks.length) json.jsHooks = hooks

      const outDir = join(opts.out, namespace, name)
      mkdirSync(outDir, { recursive: true })
      writeFileSync(join(outDir, 'definition.json'), JSON.stringify(json, null, 2) + '\n')

      const wgslDir = join(srcDir, 'wgsl')
      if (existsSync(wgslDir)) {
        mkdirSync(join(outDir, 'wgsl'), { recursive: true })
        for (const file of readdirSync(wgslDir).filter(f => f.endsWith('.wgsl')).sort()) {
          copyFileSync(join(wgslDir, file), join(outDir, 'wgsl', file))
          programs++
        }
      }
      for (const meshPath of Object.values(instance.builtinMeshes || {})) meshes.add(meshPath)
      effects++
    }
  }

  copyFileSync(join(EFFECTS_DIR, 'manifest.json'), join(opts.out, 'manifest.json'))
  mkdirSync(join(opts.out, 'share'), { recursive: true })
  copyFileSync(join(REFERENCE_ROOT, 'share', 'palettes.json'), join(opts.out, 'share', 'palettes.json'))
  for (const meshPath of [...meshes].sort()) {
    const dest = join(opts.out, meshPath)
    mkdirSync(dirname(dest), { recursive: true })
    copyFileSync(join(SHADERS_DIR, meshPath), dest)
  }
  // The demo's text fonts (filter/text draws its default family, Nunito, from
  // these files) with the license that allows redistributing them.
  let fonts = 0
  for (const [family, files] of DEMO_FONTS) {
    mkdirSync(join(opts.out, 'share', 'fonts', family), { recursive: true })
    for (const file of files) {
      copyFileSync(join(REFERENCE_ROOT, 'demo', 'font', family, file), join(opts.out, 'share', 'fonts', family, file))
      fonts++
    }
  }

  console.error(`[convert-effects] ${effects} definitions, ${programs} WGSL programs, ${meshes.size} meshes, ${fonts} font files -> ${opts.out}`)
}

main().catch(err => {
  console.error(`[convert-effects] FAILED: ${err?.stack || err}`)
  process.exit(1)
})
