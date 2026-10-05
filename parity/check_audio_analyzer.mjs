#!/usr/bin/env node
// check_audio_analyzer.mjs — gate noisemaker-input's AudioAnalyzer against
// Chromium's own AnalyserNode.
//
// Oracle: tools/reference-input-analyser.mjs runs every scenario through the
// AnalyserNode of the headless Chromium that the reference checkout's
// Playwright installs (an OfflineAudioContext suspended at render-quantum
// boundaries). Candidate: `nm-input-dump analyser`, fed the same float32
// signal in irregular chunks (one frame, 37 frames, whole quanta) so partial
// render quanta are exercised too.
//
// Scenarios cover the analyser parameters the reference's AudioInputManager
// configures (fftSize 256, smoothingTimeConstant 0.8 and the values its
// setter allows, default decibel range) and the whole AnalyserNode surface:
// every fftSize from 32 to 32768, smoothing 0..1, decibel ranges, attribute
// changes and rejected assignments between reads, repeated reads within one
// render quantum, getter call orders, and every down-mix layout (mono, stereo,
// quad, 5.1, and the discrete 3- and 8-channel fallbacks). Signals: sines on
// and off bin centres, multi-tone, linear and exponential chirps, seeded white
// noise, silence gaps, clipping beyond ±1, impulses, DC, and denormal samples
// (Chromium down-mixes on its audio thread with denormals flushed to zero).
//
// Chromium 153 computes AnalyserNode FFTs with the rustfft crate
// (WebAudioRustFft, rustfft_ffi.rs) and the candidate uses the same crate
// version and arithmetic, so the gate requires exact equality: byte and float
// frequency data, byte and float time-domain data, and attribute assignment
// results, bit for bit.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_audio_analyzer.mjs [--verbose]
// Env:
//   NM_INPUT_DUMP   candidate binary (default target/release/nm-input-dump)
// Exit 0 when every read matches; 1 otherwise.

import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { runAnalyserOracle } from '../tools/reference-input-analyser.mjs'
import { exactJson } from '../tools/reference-input-compare.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const DUMP = process.env.NM_INPUT_DUMP || join(ROOT, 'target', 'release', 'nm-input-dump')
const VERBOSE = process.argv.includes('--verbose')

if (!existsSync(DUMP)) {
  console.error(`nm-input-dump not found at ${DUMP} (cargo build --release -p noisemaker-input --bin nm-input-dump)`)
  process.exit(2)
}

// ---------------------------------------------------------------- signals

function mulberry32 (seed) {
  return () => {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

const TAU = Math.PI * 2

// One channel of a named signal kind, `frames` long.
function channelSignal (kind, frames, sampleRate, rand, c) {
  const out = new Array(frames)
  for (let f = 0; f < frames; f++) {
    const t = f / sampleRate
    let v
    switch (kind) {
      case 'bin-sine': v = 0.5 * Math.sin(TAU * (8 + c) * f / 256); break
      case 'sine': v = 0.7 * Math.sin(TAU * (440 + 110 * c) * t); break
      case 'multitone': v = 0.4 * Math.sin(TAU * 220 * t) + 0.3 * Math.sin(TAU * (3000 + 500 * c) * t + 1) + 0.1 * Math.sin(TAU * 9000 * t); break
      case 'chirp': v = 0.6 * Math.sin(TAU * (100 * t + 0.5 * 8000 * t * t)); break
      case 'exp-chirp': v = 0.8 * Math.sin(TAU * 50 * (Math.pow(200, t) - 1) / Math.log(200)); break
      case 'noise': v = rand() * 2 - 1; break
      case 'quiet-noise': v = (rand() * 2 - 1) * 1e-4; break
      case 'silence': v = 0; break
      case 'clipping': v = 1.6 * Math.sin(TAU * 330 * t) + (f % 977 === 0 ? (c % 2 ? -3 : 3) : 0); break
      case 'square': v = Math.sin(TAU * 200 * t) >= 0 ? 1.3 : -1.3; break
      case 'impulses': v = f % 1500 === 0 ? 1 : 0; break
      case 'dc': v = 0.25 - 0.1 * c; break
      case 'gaps': v = Math.floor(f / 4096) % 3 === 1 ? 0 : 0.5 * Math.sin(TAU * 1000 * t) + 0.05 * (rand() * 2 - 1); break
      case 'denormal': v = (f % 7 === 0 ? 1 : -1) * (1e-39 + (f % 13) * 1e-40) + (f % 64 === 0 ? 1.5e-38 : 0); break
      default: throw new Error(`unknown signal ${kind}`)
    }
    out[f] = Math.fround(v)
  }
  return out
}

function signal (kinds, channels, quanta, sampleRate, seed) {
  const rand = mulberry32(seed)
  const frames = quanta * 128
  const perChannel = []
  for (let c = 0; c < channels; c++) {
    const kind = kinds[c % kinds.length]
    perChannel.push(channelSignal(kind, frames, sampleRate, rand, c))
  }
  const out = new Array(frames * channels)
  for (let f = 0; f < frames; f++) for (let c = 0; c < channels; c++) out[f * channels + c] = perChannel[c][f]
  return out
}

const ALL_GETTERS = ['byteFrequency', 'floatFrequency', 'byteTimeDomain', 'floatTimeDomain']

// Reads every `step` quanta from `from`, plus the last quantum before `quanta`.
function readEvents (quanta, step, from = 1, getters = ALL_GETTERS) {
  const events = []
  for (let q = from; q < quanta; q += step) events.push({ quantum: q, ops: [{ op: 'read', getters }] })
  if (events.at(-1)?.quantum !== quanta - 1) events.push({ quantum: quanta - 1, ops: [{ op: 'read', getters }] })
  return events
}

const MANAGER = { fftSize: 256, smoothingTimeConstant: 0.8, minDecibels: -100, maxDecibels: -30 }

function scenario (name, { kinds, channels = 1, quanta = 240, sampleRate = 48000, seed = 1, options = MANAGER, events }) {
  return {
    name,
    channels,
    sampleRate,
    samples: signal(kinds, channels, quanta, sampleRate, seed),
    options,
    events: events ?? readEvents(quanta, 7)
  }
}

const scenarios = []

// The reference AudioInputManager's analysers: fftSize 256, smoothing 0.8.
for (const kind of ['bin-sine', 'sine', 'multitone', 'chirp', 'exp-chirp', 'noise', 'quiet-noise', 'silence', 'clipping', 'square', 'impulses', 'dc', 'gaps']) {
  scenarios.push(scenario(`manager-mono-${kind}`, { kinds: [kind], seed: kind.length, sampleRate: 44100 }))
}
// The manager's main analyser sees the capture's channels and down-mixes them.
scenarios.push(scenario('manager-stereo', { kinds: ['sine', 'noise'], channels: 2, seed: 11 }))
scenarios.push(scenario('manager-quad', { kinds: ['multitone', 'chirp', 'noise', 'clipping'], channels: 4, seed: 12 }))
scenarios.push(scenario('manager-5.1', { kinds: ['sine', 'noise', 'chirp', 'dc', 'clipping', 'multitone'], channels: 6, seed: 13 }))
scenarios.push(scenario('discrete-3', { kinds: ['noise', 'sine', 'clipping'], channels: 3, seed: 14 }))
scenarios.push(scenario('discrete-8', { kinds: ['chirp', 'noise', 'sine', 'dc', 'clipping', 'square', 'multitone', 'gaps'], channels: 8, seed: 15 }))
// Denormal samples: copied as-is in mono, flushed by the down-mix arithmetic.
scenarios.push(scenario('denormal-mono', { kinds: ['denormal'], quanta: 60, events: readEvents(60, 5) }))
scenarios.push(scenario('denormal-stereo', { kinds: ['denormal', 'denormal'], channels: 2, quanta: 60, events: readEvents(60, 5) }))
scenarios.push(scenario('denormal-quad', { kinds: ['denormal', 'quiet-noise', 'denormal', 'dc'], channels: 4, quanta: 60, events: readEvents(60, 5) }))
scenarios.push(scenario('denormal-5.1', { kinds: ['denormal', 'denormal', 'denormal', 'dc', 'denormal', 'denormal'], channels: 6, quanta: 60, events: readEvents(60, 5) }))
// The smoothing values the manager's setter allows, and the AnalyserNode range.
for (const smoothing of [0, 0.3, 0.5, 0.99, 1]) {
  scenarios.push(scenario(`smoothing-${smoothing}`, { kinds: ['gaps'], seed: 20, options: { ...MANAGER, smoothingTimeConstant: smoothing } }))
}
// Every fftSize; large sizes need the window filled before most reads.
for (const fftSize of [32, 64, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768]) {
  const quanta = Math.max(120, fftSize / 128 + 40)
  const step = fftSize >= 8192 ? 61 : 13
  scenarios.push(scenario(`fft-${fftSize}`, {
    kinds: ['multitone', 'noise'], channels: 2, quanta, seed: fftSize, sampleRate: 48000,
    options: { fftSize, smoothingTimeConstant: 0.8, minDecibels: -100, maxDecibels: -30 },
    events: readEvents(quanta, step, fftSize >= 8192 ? Math.floor(quanta / 2) : 1)
  }))
}
// Decibel ranges.
for (const [min, max] of [[-80, -30], [-90, -10], [-100, 0], [-200, -1], [-35, -30], [-150, 40]]) {
  scenarios.push(scenario(`decibels-${min}-${max}`, { kinds: ['chirp'], seed: 30, options: { ...MANAGER, minDecibels: min, maxDecibels: max } }))
}
// Attribute changes, rejected assignments, repeated reads and getter orders.
{
  const read = getters => ({ op: 'read', getters })
  const set = (prop, value) => ({ op: 'set', prop, value })
  const events = [
    { quantum: 0, ops: [read(ALL_GETTERS)] },
    { quantum: 3, ops: [read(['floatTimeDomain', 'byteTimeDomain'])] },
    { quantum: 9, ops: [read(['floatFrequency', 'byteFrequency']), read(ALL_GETTERS)] },
    { quantum: 10, ops: [read(['byteTimeDomain'])] },
    { quantum: 17, ops: [read(ALL_GETTERS)] },
    { quantum: 20, ops: [set('smoothingTimeConstant', 0.2), read(ALL_GETTERS)] },
    { quantum: 30, ops: [set('fftSize', 1024), read(ALL_GETTERS), set('fftSize', 1024), read(['floatFrequency'])] },
    { quantum: 31, ops: [read(ALL_GETTERS)] },
    { quantum: 45, ops: [set('minDecibels', -20), set('maxDecibels', -120), set('minDecibels', -60), set('maxDecibels', -10), read(ALL_GETTERS)] },
    { quantum: 52, ops: [set('fftSize', 300), set('fftSize', 16), set('fftSize', 65536), set('smoothingTimeConstant', 1.5), set('smoothingTimeConstant', -0.1), read(ALL_GETTERS)] },
    { quantum: 60, ops: [set('fftSize', 64), read(ALL_GETTERS)] },
    { quantum: 61, ops: [set('fftSize', 2048), read(['byteFrequency']), read(['floatFrequency'])] },
    { quantum: 75, ops: [set('smoothingTimeConstant', 0), read(ALL_GETTERS), set('smoothingTimeConstant', 1), read(ALL_GETTERS)] },
    { quantum: 76, ops: [read(ALL_GETTERS)] },
    { quantum: 90, ops: [set('maxDecibels', 0), set('minDecibels', -100), read(ALL_GETTERS)] },
    { quantum: 119, ops: [read(ALL_GETTERS)] }
  ]
  scenarios.push(scenario('attribute-changes', { kinds: ['multitone', 'noise'], channels: 2, quanta: 120, seed: 40, events }))
}
// A long run at the default fftSize through the ring's wrap-around (65536
// samples = 512 quanta).
scenarios.push(scenario('ring-wrap', {
  kinds: ['chirp'], quanta: 1100, seed: 50,
  options: { fftSize: 2048, smoothingTimeConstant: 0.8, minDecibels: -100, maxDecibels: -30 },
  events: readEvents(1100, 97, 500)
}))

// ---------------------------------------------------------------- compare

function runCandidate (cases, chunkFrames) {
  const dir = mkdtempSync(join(tmpdir(), 'nm-input-analyser-'))
  try {
    const file = join(dir, 'scenarios.json')
    writeFileSync(file, exactJson(cases.map(c => ({ ...c, chunkFrames }))))
    return JSON.parse(execFileSync(DUMP, ['analyser', file], { encoding: 'utf8', maxBuffer: 1 << 30 }))
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
}

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)
function fromBits (bits) { u32[0] = bits; return f32[0] }

// Compares one read; returns a description of the first difference or null,
// accumulating statistics.
function compareRead (expected, actual, stats) {
  let first = null
  for (const getter of Object.keys(expected)) {
    const e = expected[getter]
    const a = actual?.[getter]
    if (!a || a.length !== e.length) { first ??= `${getter}: length ${e.length} vs ${a?.length}`; continue }
    for (let i = 0; i < e.length; i++) {
      if (e[i] === a[i]) continue
      stats.mismatches[getter] = (stats.mismatches[getter] || 0) + 1
      if (getter === 'floatFrequency' || getter === 'floatTimeDomain') {
        const d = Math.abs(fromBits(e[i]) - fromBits(a[i]))
        stats.maxFloatDiff[getter] = Math.max(stats.maxFloatDiff[getter] || 0, Number.isNaN(d) ? Infinity : d)
        first ??= `${getter}[${i}]: ${fromBits(e[i])} vs ${fromBits(a[i])}`
      } else {
        stats.maxByteDiff[getter] = Math.max(stats.maxByteDiff[getter] || 0, Math.abs(e[i] - a[i]))
        first ??= `${getter}[${i}]: ${e[i]} vs ${a[i]}`
      }
    }
  }
  return first
}

const oracle = await runAnalyserOracle(scenarios)
console.log(`[INFO] oracle: ${oracle.oracle.browser} on ${oracle.oracle.platform}`)

let pass = 0
let total = 0
let values = 0
for (const chunkFrames of [128, 37, 1]) {
  const candidate = runCandidate(scenarios, chunkFrames)
  for (let s = 0; s < scenarios.length; s++) {
    const expected = oracle.results[s]
    const actual = candidate[s]
    const stats = { mismatches: {}, maxFloatDiff: {}, maxByteDiff: {} }
    let ok = expected.events.length === actual.events.length
    if (!ok) console.log(`[FAIL] ${expected.name} (chunk ${chunkFrames}): ${expected.events.length} events vs ${actual.events.length}`)
    let reported = false
    for (let i = 0; i < expected.events.length && i < actual.events.length; i++) {
      const e = expected.events[i]
      const a = actual.events[i]
      if (e.frame !== e.quantum * 128) { ok = false; console.log(`[FAIL] ${expected.name}: oracle suspended at frame ${e.frame}, expected ${e.quantum * 128}`) }
      for (let r = 0; r < e.results.length; r++) {
        total++
        const er = e.results[r]
        const ar = a.results[r]
        let diff = null
        if (er.op !== ar?.op) diff = `op ${er.op} vs ${ar?.op}`
        else if (er.op === 'set') diff = er.ok === ar.ok && er.prop === ar.prop ? null : `set ${er.prop}: ok ${er.ok} vs ${ar.ok}`
        else {
          for (const v of Object.values(er.data)) values += v.length
          diff = compareRead(er.data, ar.data, stats)
        }
        if (diff) {
          ok = false
          if (!reported || VERBOSE) console.log(`[FAIL] ${expected.name} (chunk ${chunkFrames}) quantum ${e.quantum} op ${r}: ${diff}`)
          reported = true
        } else pass++
      }
    }
    if (!ok || VERBOSE) {
      console.log(`[${ok ? 'INFO' : 'FAIL'}] ${expected.name} (chunk ${chunkFrames}): mismatches ${JSON.stringify(stats.mismatches)}, max float diff ${JSON.stringify(stats.maxFloatDiff)}, max byte diff ${JSON.stringify(stats.maxByteDiff)}`)
    }
  }
}
console.log(`[INFO] ${scenarios.length} scenarios x 3 chunkings, ${values} compared values`)
console.log(`AUDIO_ANALYZER: ${pass}/${total}`)
process.exit(pass === total ? 0 : 1)
