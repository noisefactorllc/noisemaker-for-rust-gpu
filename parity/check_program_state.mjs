#!/usr/bin/env node
// check_program_state.mjs — gate the Rust ProgramState port
// (noisemaker_dsl::program_state) against the reference ProgramState
// (demo/shaders/lib/program-state.js).
//
// For every suite in parity/program-state/*.json, tools/reference-program-state.mjs
// runs the reference over the suite's scenarios (expanding its macros into
// concrete operations with the reference's own values) and writes the
// resolved scenarios and one record per operation: the result or thrown
// error, the events every listener saw, the console warnings and errors, the
// calls made on the mock pipeline, the pass uniforms that changed, and the
// ProgramState's internal state. `nm-program-state run` then runs the resolved
// scenarios through the Rust port. Every record must match exactly: same
// values (undefined/NaN/-0 distinguished by the tagged encoding), same member
// order, same events in the same order.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_program_state.mjs \
//       [suite.json...] [--only substring] [--keep DIR] [--verbose]
//
// Env: NM_PROGRAM_STATE  candidate binary (default target/release/nm-program-state)
//
// Exit 0 when every record matches; 1 otherwise.

import { execFileSync } from 'node:child_process'
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const CANDIDATE = process.env.NM_PROGRAM_STATE || join(ROOT, 'target', 'release', 'nm-program-state')

function parseArgs (argv) {
  const opts = { suites: [], only: null, keep: null, verbose: false }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--only') opts.only = argv[++i]
    else if (a === '--keep') opts.keep = resolve(argv[++i])
    else if (a === '--verbose') opts.verbose = true
    else if (a.startsWith('--')) throw new Error(`unknown option ${a}`)
    else opts.suites.push(resolve(a))
  }
  if (!opts.suites.length) {
    const dir = join(ROOT, 'parity', 'program-state')
    opts.suites = readdirSync(dir).filter(f => f.endsWith('.json')).sort().map(f => join(dir, f))
  }
  return opts
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

// JSON lines split on '\n' only (readline also breaks at U+2028/U+2029, which
// JSON.stringify leaves unescaped inside strings).
async function * lines (path) {
  let rest = ''
  for await (const chunk of createReadStream(path, { encoding: 'utf8', highWaterMark: 1 << 20 })) {
    rest += chunk
    let start = 0
    let nl
    while ((nl = rest.indexOf('\n', start)) !== -1) {
      const line = rest.slice(start, nl)
      start = nl + 1
      if (line.trim()) yield line
    }
    rest = rest.slice(start)
  }
  if (rest.trim()) yield rest
}

// The category of a record: its operation, with fixture scenarios apart.
function category (rec) {
  const suite = rec.s.split(':')[0]
  return `${suite}/${rec.op}`
}

async function compare (expected, candidate, opts, totals) {
  const exp = lines(expected)[Symbol.asyncIterator]()
  const cand = lines(candidate)[Symbol.asyncIterator]()
  const bad = []
  const scenariosBad = new Set()
  let n = 0
  for (;;) {
    const [e, c] = await Promise.all([exp.next(), cand.next()])
    if (e.done && c.done) break
    if (e.done) { bad.push(`extra candidate record: ${c.value.slice(0, 200)}`); continue }
    n++
    const ref = JSON.parse(e.value)
    const cat = category(ref)
    const t = totals.get(cat) || { total: 0, match: 0 }
    t.total++
    totals.set(cat, t)
    if (c.done) { bad.push(`${ref.s}#${ref.i} ${ref.op}: no candidate record`); scenariosBad.add(ref.s); continue }
    if (e.value === c.value) { t.match++; continue }
    const got = JSON.parse(c.value)
    const d = (got.s !== ref.s || got.i !== ref.i) ? `record out of step: candidate ${got.s}#${got.i}` : diff(ref, got)
    if (!d) { t.match++; continue }
    scenariosBad.add(ref.s)
    bad.push(`${ref.s}#${ref.i} ${ref.op}: ${d}`)
  }
  return { n, bad, scenariosBad }
}

async function main () {
  const opts = parseArgs(process.argv.slice(2))
  if (!existsSync(CANDIDATE)) {
    console.error(`nm-program-state not found at ${CANDIDATE} (cargo build --release -p noisemaker-dsl --bin nm-program-state, or set NM_PROGRAM_STATE)`)
    process.exit(2)
  }
  const work = opts.keep || mkdtempSync(join(tmpdir(), 'nm-check-program-state-'))
  mkdirSync(work, { recursive: true })
  const totals = new Map()
  let failures = 0
  try {
    for (const suite of opts.suites) {
      const name = basename(suite).replace(/\.json$/, '')
      const resolved = join(work, `${name}.resolved.jsonl`)
      const expected = join(work, `${name}.expected.jsonl`)
      const candidate = join(work, `${name}.candidate.jsonl`)
      const oracleArgs = [join(ROOT, 'tools', 'reference-program-state.mjs'), 'run', suite, '--resolved', resolved, '--expected', expected]
      if (opts.only) oracleArgs.push('--only', opts.only)
      const summary = JSON.parse(execFileSync('node', ['--max-old-space-size=8192', ...oracleArgs], {
        stdio: ['ignore', 'pipe', opts.verbose ? 'inherit' : 'ignore'],
        maxBuffer: 1 << 30,
        env: process.env
      }).toString())
      execFileSync(CANDIDATE, ['run', resolved, '--out', candidate], { stdio: ['ignore', 'inherit', opts.verbose ? 'inherit' : 'ignore'], maxBuffer: 1 << 30 })
      const { n, bad, scenariosBad } = await compare(expected, candidate, opts, totals)
      console.log(`SUITE ${name}: ${n - bad.length}/${n} records match (${summary.scenarios - scenariosBad.size}/${summary.scenarios} scenarios)`)
      if (bad.length) {
        failures += bad.length
        console.log(bad.slice(0, opts.verbose ? bad.length : 40).map(b => `  ${b}`).join('\n'))
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
  process.exit(failures ? 1 : 0)
}

main().catch(err => {
  console.error(`[check_program_state] FAILED: ${err?.stack || err}`)
  process.exit(1)
})
