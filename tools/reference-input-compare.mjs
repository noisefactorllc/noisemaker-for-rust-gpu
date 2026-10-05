// reference-input-compare.mjs — shared machinery of the noisemaker-input gates
// (parity/check_midi_state.mjs, check_audio_state.mjs, check_automation.mjs):
// run the reference (tools/reference-input.mjs) and the candidate
// (`nm-input-dump`) on the same scenarios, each writing one JSON record per
// line, and compare the two streams record by record.
//
// Records are equal when their text is equal, or else when they decode to the
// same value: same keys and array lengths, strings and booleans equal, and
// numbers identical (Object.is, after decoding the {"$num": ...} encoding of
// NaN, ±Infinity and -0). Nothing is rounded or tolerated.

import { spawnSync } from 'node:child_process'
import { closeSync, existsSync, mkdtempSync, openSync, readSync, rmSync, writeFileSync, writeSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runReference } from './reference-input.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
export const DUMP = process.env.NM_INPUT_DUMP || join(ROOT, 'target', 'release', 'nm-input-dump')

export function requireDump () {
  if (!existsSync(DUMP)) {
    console.error(`nm-input-dump not found at ${DUMP} (cargo build --release -p noisemaker-input --bin nm-input-dump)`)
    process.exit(2)
  }
}

// Reads a file line by line without holding it in memory.
function * lines (path) {
  const fd = openSync(path, 'r')
  const buffer = Buffer.alloc(1 << 20)
  let rest = ''
  try {
    for (;;) {
      const n = readSync(fd, buffer, 0, buffer.length, null)
      if (n === 0) break
      const text = rest + buffer.toString('utf8', 0, n)
      const parts = text.split('\n')
      rest = parts.pop()
      yield * parts
    }
    if (rest) yield rest
  } finally {
    closeSync(fd)
  }
}

function decodeNum (value) {
  if (value && typeof value === 'object' && !Array.isArray(value) && Object.keys(value).length === 1 && '$num' in value) {
    return { NaN: NaN, Infinity: Infinity, '-Infinity': -Infinity, '-0': -0 }[value.$num]
  }
  return value
}

// First difference between a (reference) and b (candidate), or null.
export function firstDiff (a, b, path = '') {
  a = decodeNum(a)
  b = decodeNum(b)
  if (typeof a === 'number' || typeof b === 'number') {
    return typeof a === typeof b && Object.is(a, b) ? null : `${path || '.'}: ${JSON.stringify(a) ?? String(a)} vs ${JSON.stringify(b) ?? String(b)}`
  }
  if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') {
    return a === b ? null : `${path || '.'}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`
  }
  if (Array.isArray(a) !== Array.isArray(b)) return `${path || '.'}: array vs object`
  if (Array.isArray(a)) {
    if (a.length !== b.length) return `${path}: length ${a.length} vs ${b.length}`
    for (let i = 0; i < a.length; i++) {
      const d = firstDiff(a[i], b[i], `${path}[${i}]`)
      if (d) return d
    }
    return null
  }
  const keys = new Set([...Object.keys(a), ...Object.keys(b)])
  for (const k of keys) {
    if (!(k in a)) return `${path}.${k}: extra in candidate`
    if (!(k in b)) return `${path}.${k}: missing in candidate`
    const d = firstDiff(a[k], b[k], `${path}.${k}`)
    if (d) return d
  }
  return null
}

// JSON for the candidate's input: every number that is not a safe integer is
// written {"$num": "<shortest round-trip decimal>"} (or NaN, ±Infinity, -0),
// which the candidate parses with a correctly rounded parser. (serde_json's
// default float parsing can be one ulp off.)
export function exactJson (value) {
  return JSON.stringify(value, (key, v) => {
    if (typeof v !== 'number' || (Number.isSafeInteger(v) && !Object.is(v, -0))) return v
    return { $num: Object.is(v, -0) ? '-0' : String(v) }
  })
}

// Runs both sides on `scenarios` of `kind` and compares their records.
// Returns {pass, total, failures, records, perScenario} where perScenario maps
// a scenario name to {records, failures}.
export async function compareKind (kind, scenarios, { verbose = false, maxReports = 20, label = kind } = {}) {
  const dir = mkdtempSync(join(tmpdir(), `nm-input-${kind}-`))
  try {
    const scenarioFile = join(dir, 'scenarios.json')
    writeFileSync(scenarioFile, exactJson(scenarios))

    const referenceFile = join(dir, 'reference.jsonl')
    const fd = openSync(referenceFile, 'w')
    try {
      await runReference(kind, scenarios, record => writeSync(fd, JSON.stringify(record) + '\n'))
    } finally {
      closeSync(fd)
    }

    const candidateFile = join(dir, 'candidate.jsonl')
    const out = openSync(candidateFile, 'w')
    let run
    try {
      run = spawnSync(DUMP, [kind, scenarioFile], { stdio: ['ignore', out, 'inherit'] })
    } finally {
      closeSync(out)
    }
    if (run.status !== 0) {
      console.log(`[FAIL] ${label}: nm-input-dump exited with ${run.status ?? run.signal}`)
    }

    const perScenario = new Map()
    let pass = 0
    let total = 0
    let reports = 0
    const reference = lines(referenceFile)
    const candidate = lines(candidateFile)
    for (;;) {
      const r = reference.next()
      const c = candidate.next()
      if (r.done && c.done) break
      total++
      const expected = r.done ? null : JSON.parse(r.value)
      const name = expected?.scenario ?? '<extra candidate record>'
      const stats = perScenario.get(name) ?? { records: 0, failures: 0 }
      perScenario.set(name, stats)
      stats.records++
      let diff = null
      if (r.done) diff = 'candidate has extra records'
      else if (c.done) diff = 'candidate stopped early'
      else if (r.value !== c.value) diff = firstDiff(expected, JSON.parse(c.value))
      if (diff) {
        stats.failures++
        if (verbose || reports < maxReports) {
          const where = expected ? `${expected.scenario} ${['step', 'evaluation', 'requirements'].filter(k => k in expected).map(k => `${k} ${expected[k]}`).join(' ')}` : name
          console.log(`[FAIL] ${label} ${where}: ${diff}`)
        }
        reports++
      } else {
        pass++
      }
      if (r.done || c.done) break
    }
    if (run.status !== 0) total++
    return { pass, total, perScenario }
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

// Seeded generator (mulberry32) shared by the gates.
export function mulberry32 (seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}
