#!/usr/bin/env node
// check_dsl_tools.mjs — gate the Rust DSL tooling against the reference: the
// unparser (unparse, applyParameterUpdates, formatValue, unparseCall), the
// transform API (listSteps, replaceEffect, getCompatibleReplacements,
// predictReplacement), the DSL error formatter and the effect-definition
// validator.
//
// For every case suite in parity/dsl-tools/*.json, tools/reference-dsl-tools.mjs
// evaluates the cases' arguments with the reference (compiling programs,
// capturing thrown errors, reading catalog definitions), writes the resolved
// cases and the reference's result for each; `nm-dsl cases` then runs the
// resolved cases. Every record must match exactly: same value (strings byte for
// byte, undefined/NaN/-0 distinguished) and same member order, or the same
// thrown error (name and message).
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_dsl_tools.mjs \
//       [suite.json...] [--rust-frontend] [--live-check] [--keep DIR] [--verbose]
//
// --rust-frontend  nm-dsl compiles the cases' DSL with the Rust frontend instead
//                  of using the reference's compile results (end to end).
// --live-check     the oracle also runs every case on the live reference values
//                  and reports cases the tagged encoding cannot carry.
// --keep DIR       keep the resolved/expected/candidate files in DIR.
//
// Env: NM_DSL  candidate binary (default target/release/nm-dsl)
//
// Exit 0 when every case matches; 1 otherwise.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const NM_DSL = process.env.NM_DSL || join(ROOT, 'target', 'release', 'nm-dsl')

function parseArgs (argv) {
  const opts = { suites: [], rustFrontend: false, liveCheck: false, keep: null, verbose: false }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--rust-frontend') opts.rustFrontend = true
    else if (a === '--live-check') opts.liveCheck = true
    else if (a === '--verbose') opts.verbose = true
    else if (a === '--keep') opts.keep = resolve(argv[++i])
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`)
    else opts.suites.push(resolve(a))
  }
  if (!opts.suites.length) {
    const dir = join(ROOT, 'parity', 'dsl-tools')
    opts.suites = readdirSync(dir).filter(f => f.endsWith('.json')).sort().map(f => join(dir, f))
  }
  return opts
}

function readJsonl (path) {
  return readFileSync(path, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l))
}

function typeName (v) {
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'array'
  return typeof v
}

const show = v => JSON.stringify(v)?.slice(0, 300)

// First difference between a (reference) and b (candidate), or null.
function diff (a, b, path = '$') {
  const ta = typeName(a)
  const tb = typeName(b)
  if (ta !== tb) return `${path}: type ${ta} (reference) vs ${tb} (candidate): ${show(a)} vs ${show(b)}`
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
    for (const k of ka) if (!Object.hasOwn(b, k)) return `${path}.${k}: missing in candidate (reference ${show(a[k])})`
    for (const k of kb) if (!Object.hasOwn(a, k)) return `${path}.${k}: extra in candidate (${show(b[k])})`
    for (const k of ka) {
      const d = diff(a[k], b[k], `${path}.${k}`)
      if (d) return d
    }
    if (ka.join('\u0000') !== kb.join('\u0000')) {
      return `${path}: member order differs: reference [${ka.join(', ')}] vs candidate [${kb.join(', ')}]`
    }
    return null
  }
  if (ta === 'string' && a !== b) {
    let i = 0
    while (i < a.length && a[i] === b[i]) i++
    return `${path}: strings differ at offset ${i}: reference ${JSON.stringify(a.slice(Math.max(0, i - 40), i + 80))} vs candidate ${JSON.stringify(b.slice(Math.max(0, i - 40), i + 80))}`
  }
  return a === b ? null : `${path}: ${show(a)} (reference) vs ${show(b)} (candidate)`
}

function recordDiff (r, c) {
  if (!c) return 'no candidate record'
  const pick = rec => ('error' in rec ? { error: rec.error } : { result: rec.result })
  return diff(pick(r), pick(c))
}

function main () {
  const opts = parseArgs(process.argv.slice(2))
  if (!existsSync(NM_DSL)) {
    console.error(`nm-dsl not found at ${NM_DSL} (cargo build --release -p noisemaker-dsl --bin nm-dsl, or set NM_DSL)`)
    process.exit(2)
  }
  const work = opts.keep || mkdtempSync(join(tmpdir(), 'nm-check-dsl-tools-'))
  mkdirSync(work, { recursive: true })
  const totals = new Map() // category -> {total, match}
  let failures = 0
  let liveMismatches = 0
  try {
    for (const suite of opts.suites) {
      const name = basename(suite).replace(/\.json$/, '')
      const resolved = join(work, `${name}.resolved.jsonl`)
      const expected = join(work, `${name}.expected.jsonl`)
      const candidate = join(work, `${name}.candidate.jsonl`)
      const oracleArgs = [join(ROOT, 'tools', 'reference-dsl-tools.mjs'), 'cases', suite, '--resolved', resolved, '--expected', expected]
      if (opts.liveCheck) oracleArgs.push('--live-check')
      const summary = JSON.parse(execFileSync('node', oracleArgs, {
        stdio: ['ignore', 'pipe', opts.verbose || opts.liveCheck ? 'inherit' : 'ignore'],
        maxBuffer: 1 << 30,
        env: process.env
      }).toString())
      liveMismatches += summary.liveMismatches || 0
      const nmArgs = ['cases', resolved, '--out', candidate]
      if (opts.rustFrontend) nmArgs.push('--rust-frontend')
      execFileSync(NM_DSL, nmArgs, { stdio: ['ignore', 'inherit', 'inherit'], maxBuffer: 1 << 30 })
      const ref = readJsonl(expected)
      const cand = new Map(readJsonl(candidate).map(r => [r.id, r]))
      const bad = []
      const suiteTotals = new Map()
      for (const r of ref) {
        const d = recordDiff(r, cand.get(r.id))
        const t = totals.get(r.category) || { total: 0, match: 0 }
        const s = suiteTotals.get(r.category) || { total: 0, match: 0 }
        t.total++
        s.total++
        if (d) bad.push(`  ${r.id}: ${d}`)
        else {
          t.match++
          s.match++
        }
        totals.set(r.category, t)
        suiteTotals.set(r.category, s)
      }
      const parts = [...suiteTotals].map(([cat, s]) => `${cat} ${s.match}/${s.total}`).join(', ')
      console.log(`SUITE ${name}: ${ref.length - bad.length}/${ref.length} match (${parts})`)
      if (bad.length) {
        failures += bad.length
        console.log(bad.slice(0, opts.verbose ? bad.length : 40).join('\n'))
        if (!opts.verbose && bad.length > 40) console.log(`  ... ${bad.length - 40} more (--verbose lists all)`)
      }
    }
  } finally {
    if (!opts.keep) rmSync(work, { recursive: true, force: true })
  }
  let total = 0
  let match = 0
  for (const [cat, t] of [...totals].sort()) {
    console.log(`CATEGORY ${cat}: ${t.match}/${t.total} match`)
    total += t.total
    match += t.match
  }
  console.log(`TOTAL: ${match}/${total} match`)
  if (opts.liveCheck) console.log(`LIVE-CHECK: ${liveMismatches} case(s) where live reference values and their encoding disagree`)
  process.exit(failures || liveMismatches ? 1 : 0)
}

main()
