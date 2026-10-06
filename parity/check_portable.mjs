#!/usr/bin/env node
// check_portable.mjs — gate the port's registerPortableEffect
// (noisemaker_dsl::Registry::register_portable_effect, behind
// CanvasRenderer::register_portable_effect) against the reference's
// CanvasRenderer.registerPortableEffect.
//
// tools/reference-api.mjs portable builds the scenarios and runs each in a
// fresh reference realm: the reference's own registration tests
// (shaders/tests/test_portable_registration.js: contract, parameters and
// enums, starter inference, invalid packages, bare-name preservation,
// duplicates, reserved names), every catalog effect re-registered as a
// Portable definition, the effect-definition validator's definitions (as
// given and with stub shaders), edge cases of every check, and the Portable
// parity fixtures (parity/portable) with their programs compiled and
// expanded into render graphs. `nm-api portable` runs each scenario on a
// fresh registry. For every step the thrown message or the registered effect
// (its Effect instance with shaders and starter), the registry lookups of
// its name (bare, user.<func>, user/<func>), its op, choice enums, starter
// op, aliases must match; for every scenario, the final registry state
// (every effect lookup key, ops, the merged enum tree, starter ops, parameter
// and effect aliases, loaded effects).
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_portable.mjs [--keep DIR] [--verbose]
//
// Env: NM_API  candidate binary (default target/release/nm-api)
//
// Exit 0 when every step and state matches; 1 otherwise.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { diff, readJsonl } from '../tools/parity-diff.mjs'

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
  const work = keep || mkdtempSync(join(tmpdir(), 'nm-check-portable-'))
  mkdirSync(work, { recursive: true })
  const bad = []
  let steps = 0
  let stepMatch = 0
  let errors = 0
  let registered = 0
  try {
    const cases = join(work, 'portable.cases.jsonl')
    const expected = join(work, 'portable.expected.jsonl')
    const candidate = join(work, 'portable.candidate.jsonl')
    execFileSync('node', [join(ROOT, 'tools', 'reference-api.mjs'), 'portable', '--cases', cases, '--expected', expected], {
      stdio: ['ignore', 'pipe', verbose ? 'inherit' : 'ignore'], maxBuffer: 1 << 30, env: process.env
    })
    execFileSync(NM_API, ['portable', cases, '--out', candidate], { stdio: ['ignore', 'inherit', verbose ? 'inherit' : 'pipe'], maxBuffer: 1 << 30 })
    const scenarios = readJsonl(cases)
    const ref = readJsonl(expected)
    const cand = new Map(readJsonl(candidate).map(r => [r.id, r]))
    let stateMatch = 0
    for (const [index, r] of ref.entries()) {
      const c = cand.get(r.id)
      const rs = r.result.steps
      const cs = c?.result?.steps || []
      const ops = scenarios[index].steps.map(s => s.op)
      let scenarioBad = 0
      for (let i = 0; i < rs.length; i++) {
        steps++
        if (ops[i] === 'register') {
          if ('error' in rs[i]) errors++
          else registered++
        }
        const d = diff(rs[i], cs[i], `steps[${i}]`)
        if (d) {
          scenarioBad++
          bad.push(`  ${r.id} step ${i} (${ops[i]}): ${d}`)
        } else {
          stepMatch++
        }
      }
      const d = c ? diff(r.result.state, c.result.state, 'state') : 'no candidate record'
      if (d) bad.push(`  ${r.id} final state: ${d}`)
      else stateMatch++
      console.log(`SCENARIO ${r.id}: ${rs.length - scenarioBad}/${rs.length} steps match, state ${d ? 'DIFFERS' : 'matches'}`)
    }
    console.log(`STEPS: ${stepMatch}/${steps} match (${registered} registrations, ${errors} rejections)`)
    console.log(`STATES: ${stateMatch}/${ref.length} match`)
    if (bad.length) console.log(bad.slice(0, verbose ? bad.length : 40).join('\n'))
    if (!verbose && bad.length > 40) console.log(`  ... ${bad.length - 40} more (--verbose lists all)`)
  } finally {
    if (!keep) rmSync(work, { recursive: true, force: true })
  }
  process.exit(bad.length ? 1 : 0)
}

main()
