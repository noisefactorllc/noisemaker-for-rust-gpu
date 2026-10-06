// parity-diff.mjs — structural comparison of reference and candidate records
// (tagged JSON, tools/reference-dsl-tools.mjs encoding): same values, strings
// byte for byte, the same member order.

import { readFileSync } from 'node:fs'

export function readJsonl (path) {
  return readFileSync(path, 'utf8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l))
}

function typeName (v) {
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'array'
  return typeof v
}

const show = v => JSON.stringify(v)?.slice(0, 300)

// The first difference between a (reference) and b (candidate), or null.
export function diff (a, b, path = '$') {
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
  if (ta === 'number' && !Object.is(a, b)) return `${path}: ${show(a)} (reference) vs ${show(b)} (candidate)`
  return a === b ? null : `${path}: ${show(a)} (reference) vs ${show(b)} (candidate)`
}

// A record's outcome: its error, or its result.
export function recordDiff (r, c) {
  if (!c) return 'no candidate record'
  const pick = rec => ('error' in rec ? { error: rec.error } : { result: rec.result })
  return diff(pick(r), pick(c))
}
