#!/usr/bin/env node
// check_api.mjs — gate the Rust port's equivalents of the reference engine's
// public API (shaders/src/index.js) against the reference itself:
// runtime/tags.js (tags, namespaces and their registration, with the parser
// and validator observing registered namespaces), palettes.js (PALETTES and
// samplePalette, bit for bit over every palette and a sweep of positions),
// lang/constants.js, runtime/effect.js (the Effect constructor, parameter
// categories), the renderer/canvas.js helpers over every catalog effect, the
// effect/op/starter/enum registries, VERSION/PHASE, and the resource
// allocator over the fixture pool.
//
// tools/reference-api.mjs runs the cases on the reference (Node, importing its
// modules directly) and writes the resolved inputs and its results; `nm-api
// cases` runs the same inputs with noisemaker-dsl. Every record must match
// exactly: same value (strings byte for byte, undefined/NaN/-0 distinguished,
// numbers bit for bit) and member order, or the same thrown error.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_api.mjs [--keep DIR] [--verbose]
//
// Env: NM_API  candidate binary (default target/release/nm-api)
//
// Exit 0 when every case matches; 1 otherwise.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { readJsonl, recordDiff } from '../tools/parity-diff.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const NM_API = process.env.NM_API || join(ROOT, 'target', 'release', 'nm-api')

function main () {
  const argv = process.argv.slice(2)
  const verbose = argv.includes('--verbose')
  const keepIndex = argv.indexOf('--keep')
  const keep = keepIndex >= 0 ? resolve(argv[keepIndex + 1]) : null
  if (!existsSync(NM_API)) {
    console.error(`nm-api not found at ${NM_API} (cargo build --release -p noisemaker-dsl --bin nm-api, or set NM_API)`)
    process.exit(2)
  }
  const work = keep || mkdtempSync(join(tmpdir(), 'nm-check-api-'))
  mkdirSync(work, { recursive: true })
  let failures = 0
  try {
    const cases = join(work, 'api.cases.jsonl')
    const expected = join(work, 'api.expected.jsonl')
    const candidate = join(work, 'api.candidate.jsonl')
    execFileSync('node', [join(ROOT, 'tools', 'reference-api.mjs'), 'api', '--cases', cases, '--expected', expected], {
      stdio: ['ignore', 'pipe', verbose ? 'inherit' : 'ignore'], maxBuffer: 1 << 30, env: process.env
    })
    execFileSync(NM_API, ['cases', cases, '--out', candidate], { stdio: ['ignore', 'inherit', verbose ? 'inherit' : 'pipe'], maxBuffer: 1 << 30 })
    const ref = readJsonl(expected)
    const cand = new Map(readJsonl(candidate).map(r => [r.id, r]))
    const totals = new Map()
    const bad = []
    let samples = 0
    for (const r of ref) {
      const d = recordDiff(r, cand.get(r.id))
      const t = totals.get(r.category) || { total: 0, match: 0 }
      t.total++
      if (d) bad.push(`  ${r.id}: ${d}`)
      else t.match++
      totals.set(r.category, t)
      if (r.id.startsWith('samplePaletteSweep:') && Array.isArray(r.result)) samples += r.result.length
    }
    for (const [cat, t] of [...totals].sort()) console.log(`CATEGORY ${cat}: ${t.match}/${t.total} match`)
    console.log(`PALETTE SAMPLES: ${samples} (every palette x ${samples / Math.max(1, ref.filter(r => r.id.startsWith('samplePaletteSweep:')).length)} positions, bit-exact)`)
    if (bad.length) {
      failures = bad.length
      console.log(bad.slice(0, verbose ? bad.length : 40).join('\n'))
      if (!verbose && bad.length > 40) console.log(`  ... ${bad.length - 40} more (--verbose lists all)`)
    }
    console.log(`TOTAL: ${ref.length - bad.length}/${ref.length} match`)
  } finally {
    if (!keep) rmSync(work, { recursive: true, force: true })
  }
  process.exit(failures ? 1 : 0)
}

main()
