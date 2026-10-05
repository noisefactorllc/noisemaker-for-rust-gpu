#!/usr/bin/env node
// check_audio_state.mjs — gate noisemaker-input's AudioState against the
// reference AudioState (shaders/src/runtime/external-input.js), step by step.
//
// The same scripted operations run through the unmodified reference (in Node,
// a fresh module instance per scenario) and through `nm-input-dump audio`.
// After EVERY step the gate compares, exactly, the operation's return value
// and the full state: the aggregate and every default-input and device
// channel state (low, mid, high, vol, raw, rawReady, the 16 FFT bins, the 128
// spectrum and waveform values as IEEE bits, the smoothing buffers, the reused
// analyser byte buffer and the smoothing length), and the registries (devices
// with name, connection and channel count, the name index, the inventory, the
// default channels and their connection).
//
// Scenarios: every AudioState sequence of the reference's
// test_external_input.js and test_audio.js; analyser updates (bin counts 0 to
// 1024, partial fills of the reused buffer, smoothing lengths including 0,
// fractions, NaN and Infinity, rolling-buffer growth and shrink); bands, raw
// samples and direct field writes with NaN, ±Infinity, -0 and out-of-range
// values; spectrum and waveform bytes of every length; the device registry
// (re-registration with new names and channel counts, disconnection, raw
// invalidation, inventories with duplicates, disconnected and unnamed
// entries, channel values with non-finite inputs) and getDeviceChannelState
// with every selector shape; default channels (counts 0-33, disconnect,
// re-register); registry-less states; and seeded fuzz runs.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_audio_state.mjs [--verbose]
// Env:
//   NM_INPUT_DUMP         candidate binary (default target/release/nm-input-dump)
//   NM_AUDIO_FUZZ_STEPS   steps per fuzz scenario (default 1500)
// Exit 0 when every step matches; 1 otherwise.

import { compareKind, mulberry32, requireDump } from '../tools/reference-input-compare.mjs'

const VERBOSE = process.argv.includes('--verbose')
const FUZZ_STEPS = Number(process.env.NM_AUDIO_FUZZ_STEPS || 1500)
requireDump()

const N = { NaN: { $num: 'NaN' }, Inf: { $num: 'Infinity' }, NegInf: { $num: '-Infinity' }, NegZero: { $num: '-0' } }
const fill = (n, v) => new Array(n).fill(v)
const ramp = n => Array.from({ length: n }, (_, i) => (i * 37 + 11) % 256)

const scenarios = []

// ---------------------------------------------------------------- reference tests

scenarios.push({
  name: 'reference-external-input',
  steps: [
    { op: 'updateFromAnalyser', target: 'root', analyser: { bins: 128, data: fill(128, 0) } },
    { op: 'updateFromAnalyser', target: 'root', analyser: { bins: 128, data: fill(128, 0) } },
    { op: 'setBands', target: 'root', low: 0.5, mid: 0.3, high: 0.8 },
    { op: 'setBands', target: 'root', low: -0.5, mid: 1.5, high: 0.5 },
    { op: 'reset', target: 'root' },
    { op: 'registerDevice', id: 'selected', name: 'Selected', channelCount: 1 },
    { op: 'registerDefaultChannels', count: 1 },
    { op: 'updateFromAnalyser', target: 'root', analyser: { bins: 128, data: fill(128, 128) } },
    { op: 'updateFromAnalyser', target: { device: 'selected', channel: 1 }, analyser: { bins: 128, data: fill(128, 128) } },
    { op: 'setRaw', target: 'root', value: 0.8 },
    { op: 'setRaw', target: { device: 'selected', channel: 1 }, value: -0.25 },
    { op: 'setRaw', target: { default: 1 }, value: 0.5 },
    { op: 'setSpectrum', target: 'root', data: fill(128, 128) },
    { op: 'setWaveform', target: 'root', data: fill(128, 255) },
    { op: 'resetAggregate', target: 'root' },
    { op: 'updateFromAnalyser', target: 'root', analyser: { bins: 128, data: fill(128, 32) } },
    { op: 'updateFromAnalyser', target: { device: 'selected', channel: 1 }, analyser: { bins: 128, data: fill(128, 32) } },
    { op: 'reset', target: 'root' },
    { op: 'setRaw', target: 'root', value: -0.75 }, { op: 'setRaw', target: 'root', value: 2 }, { op: 'setRaw', target: 'root', value: -2 },
    { op: 'registerDevice', id: 'interface-a', name: 'Interface', channelCount: 2 },
    { op: 'setChannelValues', id: 'interface-a', channel: 1, values: { raw: 0 } },
    { op: 'setChannelValues', id: 'interface-a', channel: 2, values: { raw: -0.5 } },
    { op: 'setDeviceRawUnavailable', id: 'interface-a' },
    { op: 'registerDevice', id: 'left', name: 'Interface', channelCount: 2 },
    { op: 'registerDevice', id: 'right', name: 'Interface', channelCount: 4 },
    { op: 'setChannelValues', id: 'left', channel: 2, values: { low: 0.2, mid: 0.3, high: 0.4, vol: 0.5, raw: -0.6 } },
    { op: 'setChannelValues', id: 'right', channel: 2, values: { low: 0.8, mid: 0.7, high: 0.6, vol: 0.5, raw: 0.4 } },
    { op: 'getDeviceChannelState', selector: { name: 'Interface', id: 'right', channel: 2 } },
    { op: 'getDeviceChannelState', selector: { name: 'Interface', channel: 2 } },
    { op: 'registerDevice', id: 'solo', name: 'Unique Interface', channelCount: 2 },
    { op: 'setChannelValues', id: 'solo', channel: 1, values: { low: 0.9, raw: 0.25 } },
    { op: 'getDeviceChannelState', selector: { name: 'Unique Interface', channel: 1 } },
    { op: 'disconnectDevice', id: 'solo' },
    { op: 'getDeviceChannelState', selector: { name: 'Unique Interface', id: 'solo', channel: 1 } },
    { op: 'devices' },
    { op: 'registerDevice', id: 'stereo', name: 'Stereo', channelCount: 2 },
    { op: 'getDeviceChannelState', selector: { name: 'Stereo', id: 'stereo', channel: 3 } },
    { op: 'setMaxBufferLength', target: 'root', value: 3 },
    { op: 'setBands', target: 'root', low: 0.9, mid: 0.9, high: 0.9 },
    { op: 'smooth', target: 'root', band: 'low', value: 0.9 },
    { op: 'smooth', target: 'root', band: 'low', value: 0.3 },
    { op: 'setWaveform', target: 'root', data: [...fill(64, 255), ...fill(64, 0)] },
    { op: 'reset', target: 'root' }
  ]
})

scenarios.push({
  name: 'reference-audio',
  steps: [
    { op: 'registerDevice', id: 'a', name: 'Interface', channelCount: 1 },
    { op: 'setChannelValues', id: 'a', channel: 1, values: { raw: 0 } },
    { op: 'registerDevice', id: 'b', name: 'Interface', channelCount: 2 },
    { op: 'setChannelValues', id: 'b', channel: 2, values: { low: 0.85, raw: 0.5 } },
    { op: 'registerDefaultChannels', count: 2 },
    { op: 'setField', target: { default: 1 }, field: 'low', value: 0.25 },
    { op: 'setField', target: { default: 2 }, field: 'low', value: 0.75 },
    { op: 'getDeviceChannelState', selector: { channel: 1 } },
    { op: 'getDeviceChannelState', selector: { channel: 3 } },
    { op: 'disconnectDefaultInput' },
    { op: 'getDeviceChannelState', selector: { channel: 2 } },
    { op: 'registerDefaultChannels', count: 1 },
    { op: 'getDeviceChannelState', selector: { channel: 2 } },
    { op: 'getDeviceChannelState', selector: { channel: 1 } },
    { op: 'registerDevice', id: 'left', name: 'Interface2', channelCount: 2 },
    { op: 'setChannelValues', id: 'left', channel: 1, values: { low: 0.8 } },
    { op: 'setDeviceInventory', devices: [{ id: 'left', name: 'Interface2', connected: true }, { id: 'right', name: 'Interface2', connected: true }] },
    { op: 'getDeviceChannelState', selector: { name: 'Interface2', channel: 1 } },
    { op: 'getDeviceChannelState', selector: { name: 'Interface2', id: 'left', channel: 1 } },
    { op: 'setDeviceInventory', devices: [{ id: 'left', name: 'Interface2', connected: true }, { id: 'right', name: 'Interface2', connected: false }] },
    { op: 'getDeviceChannelState', selector: { name: 'Interface2', channel: 1 } },
    { op: 'setDeviceInventory', devices: [{ id: 'right', name: 'Interface2', connected: true }] },
    { op: 'getDeviceChannelState', selector: { name: 'Interface2', channel: 1 } },
    { op: 'registerDefaultChannels', count: 32 },
    { op: 'setRaw', target: { default: 31 }, value: -0.6 },
    { op: 'setRaw', target: { default: 32 }, value: 0.8 },
    { op: 'registerDevice', id: 'mixer-a', name: 'mixer-a', channelCount: 32 },
    { op: 'registerDevice', id: 'mixer-b', name: 'mixer-b', channelCount: 32 },
    ...Array.from({ length: 32 }, (_, i) => ({ op: 'setChannelValues', id: 'mixer-a', channel: i + 1, values: { raw: (i + 1) / 16 - 1, low: (i + 1) / 32, mid: (33 - i - 1) / 32, high: i % 2 ? 0.8 : 0.2, vol: 0.5 } })),
    { op: 'setChannelValues', id: 'mixer-a', channel: 17, values: { raw: 1 } },
    { op: 'getDeviceChannelState', selector: { name: 'mixer-a', id: 'mixer-a', channel: 32 } }
  ]
})

// ---------------------------------------------------------------- analyser updates

{
  const steps = []
  for (const bins of [0, 1, 8, 15, 16, 17, 31, 32, 47, 48, 128, 256, 1024]) {
    steps.push({ op: 'updateFromAnalyser', target: 'root', analyser: { bins, data: ramp(bins) } })
  }
  // A shorter fill leaves the reused buffer's tail; a new size reallocates.
  steps.push({ op: 'updateFromAnalyser', target: 'root', analyser: { bins: 1024, data: fill(10, 255) } })
  steps.push({ op: 'updateFromAnalyser', target: 'root', analyser: { bins: 128, data: fill(5, 7) } })
  steps.push({ op: 'updateFromAnalyser', target: 'root', analyser: { bins: 128, data: [] } })
  steps.push({ op: 'updateFromAnalyser', target: 'root', analyser: null })
  for (const smoothing of [1, 2, 2.5, 0, -3, 10, 11, 0.5, N.NaN, 3, N.Inf, 1, N.NegInf, 5]) {
    for (let k = 0; k < 4; k++) {
      steps.push({ op: 'updateFromAnalyser', target: 'root', smoothing, analyser: { bins: 128, data: ramp(128).map(v => (v + k * 53) % 256) } })
    }
  }
  scenarios.push({ name: 'analyser-updates', steps })
}

// ---------------------------------------------------------------- values

{
  const values = [0, 0.25, 1, -1, 2, -2, 0.999999, 1e-300, -1e-300, N.NaN, N.Inf, N.NegInf, N.NegZero]
  const steps = []
  for (const v of values) {
    steps.push({ op: 'setRaw', target: 'root', value: v })
    steps.push({ op: 'setBands', target: 'root', low: v, mid: 0.5, high: v })
    steps.push({ op: 'setField', target: 'root', field: 'vol', value: v })
    steps.push({ op: 'setField', target: 'root', field: 'raw', value: v })
  }
  steps.push({ op: 'setRawUnavailable', target: 'root' }, { op: 'setField', target: 'root', field: 'rawReady', value: true })
  for (const len of [0, 1, 64, 127, 128, 129, 200]) {
    steps.push({ op: 'setSpectrum', target: 'root', data: ramp(len) }, { op: 'setWaveform', target: 'root', data: ramp(len).reverse() })
  }
  for (const v of [0.5, N.NaN, 0.25, N.Inf, 1]) {
    for (const band of ['low', 'mid', 'high']) steps.push({ op: 'smooth', target: 'root', band, value: v })
  }
  for (const length of [2, 0, 7, N.NaN, 1, 2.5]) {
    steps.push({ op: 'setMaxBufferLength', target: 'root', value: length })
    for (let k = 0; k < 3; k++) steps.push({ op: 'smooth', target: 'root', band: 'mid', value: k / 3 })
  }
  steps.push({ op: 'resetAggregate', target: 'root' }, { op: 'reset', target: 'root' })
  scenarios.push({ name: 'values', steps })
}

// ---------------------------------------------------------------- registry

{
  const steps = [
    { op: 'registerDevice', id: '', name: 'Empty' },
    { op: 'registerDevice', id: 'x', name: 'X' },
    { op: 'registerDevice', id: 'x', name: 'X', channelCount: 3 },
    { op: 'registerDevice', id: 'x', name: 'X', channelCount: 0 },
    { op: 'registerDevice', id: 'x', name: 'Y', channelCount: 5 },
    { op: 'registerDevice', id: 'x', name: 'Y', channelCount: 2 },
    { op: 'registerDevice', id: 'z', name: '', channelCount: 2 },
    { op: 'registerDevice', id: 'w', name: 'Y', channelCount: 40 },
    { op: 'devices' },
    { op: 'setChannelValues', id: 'x', channel: 1, values: { low: 0.4, mid: N.NaN, high: N.Inf, vol: 1.5, raw: N.NegInf } },
    { op: 'setChannelValues', id: 'x', channel: 2, values: { low: -0.5, mid: N.NegZero, raw: N.NegZero } },
    { op: 'setChannelValues', id: 'x', channel: 3, values: { low: 0.1 } },
    { op: 'setChannelValues', id: 'nope', channel: 1, values: { low: 0.1 } },
    { op: 'setChannelValues', id: 'w', channel: 40, values: { raw: 0.3 } },
    { op: 'setDeviceRawUnavailable', id: 'w' },
    { op: 'setDeviceRawUnavailable', id: 'nope' },
    { op: 'disconnectDevice', id: 'x' },
    { op: 'setChannelValues', id: 'x', channel: 1, values: { low: 0.7 } },
    { op: 'setDeviceRawUnavailable', id: 'x' },
    { op: 'disconnectDevice', id: 'nope' },
    { op: 'registerDevice', id: 'x', name: 'Y', channelCount: 2 },
    { op: 'registerDefaultChannels', count: 0 },
    { op: 'registerDefaultChannels', count: 33 },
    { op: 'registerDefaultChannels', count: 4 },
    { op: 'setChannelValues', id: 'x', channel: 1, values: { raw: 0.2 } },
    { op: 'setDeviceInventory', devices: [] },
    { op: 'setDeviceInventory', devices: [{ id: 'a', name: 'Y', connected: true }, { id: 'a', name: 'Y', connected: true }, { id: '', name: 'Y', connected: true }, { id: 'b', name: '', connected: true }, { id: 'c', name: 'Q', connected: false }, { id: 'd', name: 'Q', connected: true }] }
  ]
  const selectors = []
  for (const name of [undefined, '', 'X', 'Y', 'Q', 'Missing']) {
    for (const id of [undefined, '', 'x', 'w', 'd', 'missing']) {
      for (const channel of [undefined, 0, 1, 2, 3, 32, 33, 1.5, -1, N.NaN]) {
        const selector = {}
        if (name !== undefined) selector.name = name
        if (id !== undefined) selector.id = id
        if (channel !== undefined) selector.channel = channel
        selectors.push({ op: 'getDeviceChannelState', selector })
      }
    }
  }
  steps.push(...selectors)
  steps.push({ op: 'setDeviceInventory', devices: [{ id: 'w', name: 'Y', connected: true }] }, ...selectors.filter((_, i) => i % 3 === 0))
  for (const channel of [0, 1, 4, 5, 32, 1.5, N.NaN]) steps.push({ op: 'getDefaultChannelState', channel })
  steps.push({ op: 'disconnectDefaultInput' }, { op: 'getDefaultChannelState', channel: 1 }, { op: 'registerDefaultChannels', count: 2 }, { op: 'getDefaultChannelState', channel: 3 })
  steps.push({ op: 'updateFromAnalyser', target: { default: 2 }, analyser: { bins: 64, data: ramp(64) } })
  steps.push({ op: 'updateFromAnalyser', target: { device: 'w', channel: 3 }, smoothing: 2, analyser: { bins: 16, data: ramp(16) } })
  steps.push({ op: 'setSpectrum', target: { device: 'x', channel: 2 }, data: ramp(128) })
  steps.push({ op: 'reset', target: { device: 'w', channel: 3 } }, { op: 'reset', target: 'root' }, { op: 'devices' })
  scenarios.push({ name: 'registry', steps })
}

scenarios.push({
  name: 'no-registry',
  registry: false,
  steps: [
    { op: 'registerDevice', id: 'x', name: 'X', channelCount: 2 },
    { op: 'registerDefaultChannels', count: 2 },
    { op: 'getDefaultChannelState', channel: 1 },
    { op: 'getDeviceChannelState', selector: { channel: 1 } },
    { op: 'getDeviceChannelState', selector: {} },
    { op: 'getDeviceChannelState', selector: { id: 'x', channel: 1 } },
    { op: 'setChannelValues', id: 'x', channel: 1, values: { low: 0.5 } },
    { op: 'setDeviceInventory', devices: [{ id: 'x', name: 'X', connected: true }] },
    { op: 'getDeviceChannelState', selector: { name: 'X', channel: 1 } },
    { op: 'updateFromAnalyser', target: 'root', analyser: { bins: 32, data: ramp(32) } },
    { op: 'disconnectDefaultInput' }, { op: 'disconnectDevice', id: 'x' }, { op: 'devices' }, { op: 'reset', target: 'root' }
  ]
})

// ---------------------------------------------------------------- fuzz

function fuzz (seed, count) {
  const rand = mulberry32(seed)
  const pick = items => items[Math.floor(rand() * items.length)]
  const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1))
  const numbers = [0, 0.1, 0.5, 1, -1, 1.5, -0.3, N.NaN, N.Inf, N.NegZero]
  const value = () => rand() < 0.8 ? Math.round(rand() * 1000) / 500 - 0.5 : pick(numbers)
  const devices = [['a', 'Mixer'], ['b', 'Mixer'], ['c', 'Solo']]
  const steps = []
  const live = new Set()
  const targets = () => {
    const out = ['root']
    for (const t of live) out.push(t)
    return out
  }
  while (steps.length < count) {
    const r = rand()
    if (r < 0.08) {
      const [id, name] = pick(devices)
      const channels = int(0, 6)
      steps.push({ op: 'registerDevice', id, name: rand() < 0.1 ? 'Renamed' : name, channelCount: channels })
      for (const t of [...live]) if (t.device === id) live.delete(t)
      for (let c = 1; c <= Math.max(1, channels); c++) live.add({ device: id, channel: c })
    } else if (r < 0.12) {
      const count = int(0, 6)
      steps.push({ op: 'registerDefaultChannels', count })
      if (count >= 1 && count <= 32) {
        for (const t of [...live]) if ('default' in t) live.delete(t)
        for (let c = 1; c <= count; c++) live.add({ default: c })
      }
    } else if (r < 0.14) steps.push({ op: 'disconnectDevice', id: pick(devices)[0] })
    else if (r < 0.15) steps.push({ op: 'disconnectDefaultInput' })
    else if (r < 0.17) steps.push({ op: 'setDeviceRawUnavailable', id: pick(devices)[0] })
    else if (r < 0.19) steps.push({ op: 'setDeviceInventory', devices: devices.filter(() => rand() < 0.7).map(([id, name]) => ({ id, name, connected: rand() < 0.8 })) })
    else if (r < 0.3) {
      const values = {}
      for (const k of ['low', 'mid', 'high', 'vol', 'raw']) if (rand() < 0.5) values[k] = value()
      steps.push({ op: 'setChannelValues', id: pick(devices)[0], channel: int(0, 7), values })
    } else if (r < 0.45) {
      const bins = pick([128, 128, 64, 16, 8])
      steps.push({ op: 'updateFromAnalyser', target: pick(targets()), ...(rand() < 0.5 ? { smoothing: pick([1, 3, 5, 10, 2.5]) } : {}), analyser: { bins, data: Array.from({ length: rand() < 0.9 ? bins : int(0, bins) }, () => int(0, 255)) } })
    } else if (r < 0.52) steps.push({ op: 'setBands', target: pick(targets()), low: value(), mid: value(), high: value() })
    else if (r < 0.6) steps.push({ op: 'setRaw', target: pick(targets()), value: value() })
    else if (r < 0.62) steps.push({ op: 'setRawUnavailable', target: pick(targets()) })
    else if (r < 0.66) steps.push({ op: pick(['setSpectrum', 'setWaveform']), target: pick(targets()), data: Array.from({ length: int(0, 140) }, () => int(0, 255)) })
    else if (r < 0.7) steps.push({ op: 'setField', target: pick(targets()), field: pick(['low', 'mid', 'high', 'vol', 'raw']), value: value() })
    else if (r < 0.72) steps.push({ op: pick(['resetAggregate', 'reset']), target: pick(targets()) })
    else if (r < 0.9) {
      const selector = {}
      if (rand() < 0.6) selector.name = pick(['Mixer', 'Solo', 'Renamed', ''])
      if (rand() < 0.5) selector.id = pick(['a', 'b', 'c', 'z', ''])
      if (rand() < 0.9) selector.channel = pick([1, 2, 3, 6, 0, 33, 1.5])
      steps.push({ op: 'getDeviceChannelState', selector })
    } else if (r < 0.96) steps.push({ op: 'getDefaultChannelState', channel: int(0, 7) })
    else steps.push({ op: 'devices' })
  }
  return { name: `fuzz-${seed}`, steps }
}

for (const seed of [1, 2, 3]) scenarios.push(fuzz(seed, FUZZ_STEPS))

// ---------------------------------------------------------------- compare

const { pass, total, perScenario } = await compareKind('audio', scenarios, { verbose: VERBOSE, label: 'audio' })
for (const [name, stats] of perScenario) {
  if (stats.failures || VERBOSE) console.log(`[${stats.failures ? 'FAIL' : 'INFO'}] ${name}: ${stats.records - stats.failures}/${stats.records} steps`)
}
console.log(`[INFO] ${scenarios.length} scenarios, ${total} steps compared (state and return value after each)`)
console.log(`AUDIO_STATE: ${pass}/${total}`)
process.exit(pass === total ? 0 : 1)
