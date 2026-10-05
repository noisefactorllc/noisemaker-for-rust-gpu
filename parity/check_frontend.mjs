#!/usr/bin/env node
// check_frontend.mjs — gate the Rust DSL frontend against the reference, stage by stage.
//
// For every fixture program, the reference output of a stage
// (tools/reference-oracle.mjs) must equal nm-render's output of the same stage
// (`nm-render dump <stage>`): same values, same member order, same errors. Member
// order is compared because the reference observes it (uniform packing, pass and
// texture creation order, enum positions).
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_frontend.mjs [stage...] \
//       [--programs file...] [--verbose]
//   stages default to: tokens ast validated expanded graph
//
// Env:
//   NM_RENDER   candidate binary (default target/release/nm-render)
//
// Exit 0 when every stage matches for every program; 1 otherwise.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const ALL_STAGES = ['tokens', 'ast', 'validated', 'expanded', 'graph']

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const NM_RENDER = process.env.NM_RENDER || join(ROOT, 'target', 'release', 'nm-render')

function defaultPrograms () {
  const files = []
  for (const dir of ['programs', 'corpus']) {
    const d = join(ROOT, 'parity', dir)
    for (const f of readdirSync(d).filter(f => f.endsWith('.dsl')).sort()) files.push(join(d, f))
  }
  return files
}

function parseArgs (argv) {
  const opts = { stages: [], programs: [], verbose: false }
  let inPrograms = false
  for (const a of argv) {
    if (a === '--programs') { inPrograms = true; continue }
    if (a === '--verbose') { opts.verbose = true; continue }
    if (inPrograms) opts.programs.push(resolve(a))
    else if (ALL_STAGES.includes(a)) opts.stages.push(a)
    else throw new Error(`unknown argument ${a}`)
  }
  if (!opts.stages.length) opts.stages = ALL_STAGES
  if (!opts.programs.length) opts.programs = defaultPrograms()
  return opts
}

function readJsonl (path) {
  const map = new Map()
  for (const line of readFileSync(path, 'utf8').split('\n')) {
    if (!line.trim()) continue
    const rec = JSON.parse(line)
    map.set(rec.program, rec)
  }
  return map
}

function typeName (v) {
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'array'
  return typeof v
}

// First difference between a (reference) and b (candidate), or null.
function diff (a, b, path = '$') {
  const ta = typeName(a)
  const tb = typeName(b)
  if (ta !== tb) return `${path}: type ${ta} (reference) vs ${tb} (candidate): ${JSON.stringify(a)?.slice(0, 120)} vs ${JSON.stringify(b)?.slice(0, 120)}`
  if (ta === 'array') {
    const n = Math.min(a.length, b.length)
    for (let i = 0; i < n; i++) {
      const d = diff(a[i], b[i], `${path}[${i}]`)
      if (d) return d
    }
    if (a.length !== b.length) return `${path}: length ${a.length} (reference) vs ${b.length} (candidate)`
    return null
  }
  if (ta === 'object') {
    const ka = Object.keys(a)
    const kb = Object.keys(b)
    for (const k of ka) {
      if (!(k in b)) return `${path}.${k}: missing in candidate (reference ${JSON.stringify(a[k])?.slice(0, 120)})`
    }
    for (const k of kb) {
      if (!(k in a)) return `${path}.${k}: extra in candidate (${JSON.stringify(b[k])?.slice(0, 120)})`
    }
    for (const k of ka) {
      const d = diff(a[k], b[k], `${path}.${k}`)
      if (d) return d
    }
    if (ka.join('\u0000') !== kb.join('\u0000')) {
      return `${path}: member order differs: reference [${ka.join(', ')}] vs candidate [${kb.join(', ')}]`
    }
    return null
  }
  if (ta === 'number') {
    if (a === b || (Number.isNaN(a) && Number.isNaN(b))) return null
    return `${path}: ${a} (reference) vs ${b} (candidate)`
  }
  return a === b ? null : `${path}: ${JSON.stringify(a)?.slice(0, 200)} (reference) vs ${JSON.stringify(b)?.slice(0, 200)} (candidate)`
}

function main () {
  const opts = parseArgs(process.argv.slice(2))
  if (!existsSync(NM_RENDER)) {
    console.error(`nm-render not found at ${NM_RENDER} (cargo build --release -p nm-render, or set NM_RENDER)`)
    process.exit(2)
  }
  const work = mkdtempSync(join(tmpdir(), 'nm-check-frontend-'))
  let failures = 0
  try {
    for (const stage of opts.stages) {
      const refOut = join(work, `${stage}.reference.jsonl`)
      const candOut = join(work, `${stage}.candidate.jsonl`)
      execFileSync('node', [join(ROOT, 'tools', 'reference-oracle.mjs'), stage, '--out', refOut, ...opts.programs],
        { stdio: ['ignore', 'inherit', opts.verbose ? 'inherit' : 'ignore'], maxBuffer: 1 << 30 })
      execFileSync(NM_RENDER, ['dump', stage, '--out', candOut, ...opts.programs],
        { stdio: ['ignore', 'inherit', opts.verbose ? 'inherit' : 'ignore'], maxBuffer: 1 << 30 })
      const ref = readJsonl(refOut)
      const cand = readJsonl(candOut)
      let ok = 0
      const bad = []
      for (const [program, r] of ref) {
        const c = cand.get(program)
        let d
        if (!c) d = 'no candidate record'
        else if ('error' in r || 'error' in c) d = diff({ error: r.error ?? null, result: 'error' in r ? null : r.result }, { error: c.error ?? null, result: 'error' in c ? null : c.result })
        else d = diff(r.result, c.result)
        if (d) bad.push(`  ${program}: ${d}`)
        else ok++
      }
      console.log(`STAGE ${stage}: ${ok}/${ref.size} match`)
      if (bad.length) {
        failures += bad.length
        console.log(bad.join('\n'))
      }
    }
  } finally {
    rmSync(work, { recursive: true, force: true })
  }
  process.exit(failures ? 1 : 0)
}

main()
