// portable.mjs — Portable effect sidecars of DSL fixtures.
//
// A fixture <name>.dsl that calls a user-defined effect ships its Portable
// definition beside it as <name>.portable.json; the WGSL and GLSL of each pass
// program P live in <name>.P.wgsl and <name>.P.glsl (attached as
// shaders[P].wgsl and shaders[P].glsl unless the definition carries that
// source inline). The minter (parity/batch-golden.mjs), the
// reference oracle (tools/reference-api.mjs) and the candidate
// (noisemaker_gpu::protocol::load_portable_definition) all load sidecars with
// this rule, then hand the definition to registerPortableEffect.

import { existsSync, readFileSync } from 'node:fs'
import { basename, dirname, join } from 'node:path'

// <name>.portable.json for <name>.dsl, or null when the fixture has none.
export function portableSidecar (dslPath) {
  const path = dslPath.replace(/\.dsl$/, '.portable.json')
  return path !== dslPath && existsSync(path) ? path : null
}

// The definition with its shader sources attached.
export function loadPortableDefinition (path) {
  const def = JSON.parse(readFileSync(path, 'utf8'))
  const name = basename(path).replace(/\.portable\.json$/, '')
  const programs = Array.isArray(def?.passes) ? def.passes.map(p => p?.program).filter(p => typeof p === 'string') : []
  for (const program of programs) {
    for (const language of ['wgsl', 'glsl']) {
      const file = join(dirname(path), `${name}.${program}.${language}`)
      if (!existsSync(file)) continue
      if (!def.shaders || typeof def.shaders !== 'object') def.shaders = {}
      if (!def.shaders[program] || typeof def.shaders[program] !== 'object') def.shaders[program] = {}
      if (def.shaders[program][language] === undefined) def.shaders[program][language] = readFileSync(file, 'utf8')
    }
  }
  return def
}
