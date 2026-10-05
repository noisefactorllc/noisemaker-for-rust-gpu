#!/usr/bin/env node
// reference-input.mjs — run the unmodified reference external-input state and
// automation evaluators on parity scenarios, for the gates under parity/.
//
// The reference engine at $NM_REFERENCE_ROOT is imported as is:
// shaders/src/runtime/external-input.js (MidiState, AudioState) and
// shaders/src/runtime/pipeline.js (Pipeline.resolveUniformValue and
// Pipeline.getAudioInputRequirements on a GPU-less `new Pipeline(graph, null)`).
// Date.now() is replaced by the scenario's clock. Each scenario imports a fresh
// instance of external-input.js (a distinct module URL), so the module-level
// note-on counter (midiNoteOrder) starts at zero for every scenario.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node tools/reference-input.mjs \
//       <midi|audio|automation|math> <scenarios.json> [--out results.jsonl]
// Output: one JSON record per line (per step, evaluation or requirement set),
// each naming its scenario.
//
// The scenario formats are documented with the runners below; the Rust side is
// `nm-input-dump` (crates/noisemaker-input/src/bin/nm-input-dump). JSON
// conventions shared with it: a number JSON cannot carry exactly (NaN,
// ±Infinity, -0) is {"$num": "NaN" | "Infinity" | "-Infinity" | "-0"};
// Float32Array contents are arrays of IEEE bit patterns; JavaScript values in
// automation scenarios may use {"$undefined": true}, {"$function": "source"},
// and object identity {"$id": name, ...} / {"$ref": name} (including cycles).

import { closeSync, openSync, readFileSync, writeSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const RUNTIME = join(REFERENCE_ROOT, 'shaders', 'src', 'runtime')
const EXTERNAL_INPUT = pathToFileURL(join(RUNTIME, 'external-input.js')).href

let instance = 0
async function freshExternalInput () {
  return import(`${EXTERNAL_INPUT}?instance=${++instance}`)
}

// ---------------------------------------------------------------- encoding

export function num (x) {
  if (Number.isNaN(x)) return { $num: 'NaN' }
  if (x === Infinity) return { $num: 'Infinity' }
  if (x === -Infinity) return { $num: '-Infinity' }
  if (Object.is(x, -0)) return { $num: '-0' }
  return x
}

export function readNum (value) {
  if (typeof value === 'number') return value
  if (value && typeof value === 'object' && '$num' in value) {
    return { NaN: NaN, Infinity: Infinity, '-Infinity': -Infinity, '-0': -0 }[value.$num]
  }
  throw new Error(`not a number: ${JSON.stringify(value)}`)
}

const f32 = new Float32Array(1)
const u32 = new Uint32Array(f32.buffer)
export function bits (array) {
  return Array.from(array, v => { f32[0] = v; return u32[0] })
}

// JSON with the $id/$ref/$num/$undefined/$function conventions -> JS value.
export function decodeValue (json) {
  const ids = new Map()
  const collect = (node) => {
    if (Array.isArray(node)) { node.forEach(collect); return }
    if (!node || typeof node !== 'object') return
    if (typeof node.$id === 'string') {
      if (ids.has(node.$id)) throw new Error(`duplicate $id ${node.$id}`)
      ids.set(node.$id, '$array' in node ? [] : {})
    }
    for (const value of Object.values(node)) collect(value)
  }
  collect(json)
  const build = (node) => {
    if (Array.isArray(node)) return node.map(build)
    if (!node || typeof node !== 'object') return node
    if ('$ref' in node) {
      if (!ids.has(node.$ref)) throw new Error(`unknown $ref ${node.$ref}`)
      return ids.get(node.$ref)
    }
    if ('$num' in node) return readNum(node)
    if ('$undefined' in node) return undefined
    if ('$function' in node) {
      const fn = function () {}
      return fn
    }
    let target
    if (typeof node.$id === 'string') {
      target = ids.get(node.$id)
      if (Array.isArray(target)) {
        for (const item of node.$array) target.push(build(item))
        return target
      }
    } else {
      target = {}
    }
    for (const [key, value] of Object.entries(node)) {
      if (key === '$id') continue
      target[key] = build(value)
    }
    return target
  }
  return build(json)
}

// ---------------------------------------------------------------- MIDI
//
// Scenario: {name, registry (default true), time, steps: [...]}. Steps (all
// may carry `time`, the Date.now() value from that step on):
//   {op: 'message', data: [bytes], port?: {id, name}}
//   {op: 'register', id, name}     {op: 'disconnect', id}
//   {op: 'inventory', ports: [{id, name, connected}]}     {op: 'reset'}
//   {op: 'noteOn' | 'noteOff' | 'controlChange' | 'resetControllers' |
//        'clearNotes' | 'channelReset' | 'setField', target, channel, ...}
//     direct channel API on target 'root' | 'unscoped' | {port: id}
//   {op: 'zoneVoice', target, zone, members?}   {op: 'portState', name?, id?}
//   {op: 'ports'}
// Each step yields {ret, snapshot, grid}: the operation's return value, the
// state (root, unscoped and port states, each with the channels changed since
// the previous step; a channel exactly in its initial state is written
// "default"), and the note grid after updateNoteGrid() as sparse
// [index, bits] pairs.

const originName = origin => origin === null || origin === undefined
  ? null
  : (typeof origin === 'symbol' ? '<unscoped>' : `port:${origin}`)

function sparse (array, fn = v => v, isDefault = v => v === 0) {
  const out = []
  array.forEach((value, index) => { if (!isDefault(value)) out.push([index, fn(value)]) })
  return out
}

function midiChannelSnapshot (ch) {
  const note = n => ({ key: n.key, velocity: n.velocity, time: num(n.time), order: n.order, origin: originName(n.origin) })
  return {
    key: ch.key,
    velocity: ch.velocity,
    gate: ch.gate,
    time: num(ch.time),
    keys: sparse([...ch.keys]),
    cc: sparse([...ch.cc]),
    cc14: sparse([...ch.cc14]),
    ccPorts: sparse(ch._ccPorts, originName, v => v === null),
    cc14Ports: sparse(ch._cc14Ports, originName, v => v === null),
    pitchBend: ch.pitchBend,
    pressure: ch.pressure,
    polyPressure: sparse([...ch.polyPressure]),
    polyPressurePorts: sparse(ch._polyPressurePorts, originName, v => v === null),
    pitchBendPort: originName(ch._pitchBendPort),
    pressurePort: originName(ch._pressurePort),
    nrpn: [...ch.nrpn],
    rpn: [...ch.rpn],
    nrpnPorts: [...ch._nrpnPorts].map(([p, o]) => [p, originName(o)]),
    rpnPorts: [...ch._rpnPorts].map(([p, o]) => [p, originName(o)]),
    heldNotes: [...ch.heldNotes].map(([k, n]) => [k, note(n)]),
    selectors: { nrpn: [...ch._selectors.nrpn], rpn: [...ch._selectors.rpn] },
    parameterFamily: ch._parameterFamily
  }
}

// A channel exactly in its initial state is written as "default" (both sides
// decide by comparing with a fresh channel's snapshot).
let defaultChannelText = null
function midiChannelEntry (channel) {
  const snapshot = midiChannelSnapshot(channel)
  return JSON.stringify(snapshot) === defaultChannelText ? 'default' : snapshot
}

// `cache` holds each state's channel snapshots from the previous step (keyed
// 'root', 'unscoped', 'port:<id>'); a snapshot lists only the channels that
// changed since then, as [channel, snapshot] pairs. The first step lists every
// channel, so equal streams mean equal states after every step.
function midiSnapshot (state, key, cache) {
  const previous = cache.get(key) ?? []
  const current = []
  const changed = []
  for (let n = 1; n <= 16; n++) {
    const entry = midiChannelEntry(state.channels[n])
    const text = JSON.stringify(entry)
    current.push(text)
    if (previous[n - 1] !== text) changed.push([n, entry])
  }
  cache.set(key, current)
  const out = {
    clockCount: state.clockCount,
    mpeZones: { lower: state.mpeZones.lower, upper: state.mpeZones.upper },
    changed
  }
  if (!state._ports) return out
  out.ports = [...state._ports.values()].map(entry => ({
    id: entry.id, name: entry.name, connected: entry.connected, state: midiSnapshot(entry.state, `port:${entry.id}`, cache)
  }))
  out.unscoped = midiSnapshot(state._unscopedState, 'unscoped', cache)
  out.portInventory = state._portInventory ? [...state._portInventory] : null
  const idOf = s => s === null ? null : [...state._ports.values()].find(e => e.state === s)?.id ?? '<unknown>'
  out.portsByName = [...state._portsByName].map(([name, s]) => [name, idOf(s)])
  return out
}

function midiStateName (root, state) {
  if (state === null || state === undefined) return null
  if (state === root) return 'root'
  if (state === root._unscopedState) return 'unscoped'
  for (const entry of root._ports?.values() || []) if (entry.state === state) return `port:${entry.id}`
  return '<unknown>'
}

function midiTarget (root, target) {
  if (target === 'root' || target === undefined) return root
  if (target === 'unscoped') return root._unscopedState
  const entry = root._ports?.get(target.port)
  if (!entry) throw new Error(`no port ${target.port}`)
  return entry.state
}

function parameterChange (change) {
  if (!change) return null
  const out = { family: change.family, parameter: change.parameter, value: change.value }
  if (change.resetChannels) out.resetChannels = change.resetChannels
  return out
}

function midiStep (root, step) {
  switch (step.op) {
    case 'message':
      return parameterChange(root.handleMessage(Uint8Array.from(step.data), step.port ?? undefined))
    case 'register':
      return midiStateName(root, root.registerPort({ id: step.id, name: step.name }))
    case 'disconnect':
      root.disconnectPort(step.id); return null
    case 'inventory':
      root.setPortInventory(step.ports); return null
    case 'reset':
      root.reset(); return null
    case 'ports':
      return root.getPorts()
    case 'portState':
      return midiStateName(root, root.getPortState({ name: step.name, id: step.id }))
    case 'zoneVoice': {
      const state = midiTarget(root, step.target)
      const voice = state.getZoneVoice({ zone: step.zone, members: step.members })
      if (!voice) return null
      const owner = [root, root._unscopedState, ...[...(root._ports?.values() || [])].map(e => e.state)]
        .find(s => s && Object.values(s.channels).includes(voice.channel))
      return {
        key: voice.key, velocity: voice.velocity, time: num(voice.time), order: voice.order,
        origin: originName(voice.origin), channelNumber: voice.channelNumber,
        channelOwner: midiStateName(root, owner)
      }
    }
    default: {
      const channel = midiTarget(root, step.target).channels[step.channel]
      switch (step.op) {
        case 'noteOn': channel.noteOn(step.key, step.velocity); return null
        case 'noteOff': channel.noteOff(step.key ?? undefined); return null
        case 'controlChange': return parameterChange(channel.controlChange(step.controller, step.value))
        case 'resetControllers': channel.resetControllers(); return null
        case 'clearNotes': channel.clearNotes(); return null
        case 'channelReset': channel.reset(); return null
        case 'setField': channel[step.field] = step.value; return null
        default: throw new Error(`unknown MIDI op ${step.op}`)
      }
    }
  }
}

export async function runMidi (scenarios, emit) {
  const realNow = Date.now
  try {
    for (const scenario of scenarios) {
      const { MidiState, MidiChannelState } = await freshExternalInput()
      defaultChannelText = JSON.stringify(midiChannelSnapshot(new MidiChannelState()))
      let now = scenario.time ?? 0
      Date.now = () => now
      const root = new MidiState({ portRegistry: scenario.registry !== false })
      const cache = new Map()
      scenario.steps.forEach((step, index) => {
        if (step.time !== undefined) now = step.time
        const ret = midiStep(root, step)
        root.updateNoteGrid()
        emit({ scenario: scenario.name, step: index, ret, snapshot: midiSnapshot(root, 'root', cache), grid: sparse(bits(root.noteGrid)) })
      })
    }
  } finally {
    Date.now = realNow
  }
}

// ---------------------------------------------------------------- audio state
//
// Scenario: {name, registry (default true), steps: [...]}. Steps:
//   {op: 'setBands', target, low, mid, high}   {op: 'setRaw', target, value}
//   {op: 'setRawUnavailable', target}   {op: 'setField', target, field, value}
//   {op: 'registerDevice', id, name, channelCount?}
//   {op: 'setChannelValues', id, channel, values: {low?, mid?, high?, vol?, raw?}}
//   {op: 'setDeviceRawUnavailable', id}   {op: 'disconnectDevice', id}
//   {op: 'setDeviceInventory', devices: [{id, name, connected}]}
//   {op: 'registerDefaultChannels', count}   {op: 'disconnectDefaultInput'}
//   {op: 'getDefaultChannelState', channel}
//   {op: 'getDeviceChannelState', selector: {name?, id?, channel?}}
//   {op: 'updateFromAnalyser', target, analyser: null | {bins, data}, smoothing?}
//     the fake analyser writes data[0..min(len, data.length)] (a shorter
//     `data` leaves the reused buffer's tail as it was)
//   {op: 'setSpectrum' | 'setWaveform', target, data}
//   {op: 'smooth', target, band, value}   {op: 'setMaxBufferLength', target, value}
//   {op: 'resetAggregate' | 'reset', target}   {op: 'devices'}
// target: 'root' | {default: n} | {device: id, channel: n}.
// Each step yields {ret, snapshot}: the registries in full, and every state's
// own fields, or "unchanged" when they equal the previous step's.

// Fields of one AudioState itself (not its registries).
function audioOwnSnapshot (state) {
  return {
    low: num(state.low),
    mid: num(state.mid),
    high: num(state.high),
    vol: num(state.vol),
    raw: num(state.raw),
    rawReady: state.rawReady,
    fft: bits(state.fft),
    spectrum: bits(state.spectrum),
    waveform: bits(state.waveform),
    smoothing: {
      low: state._smoothingBuffers.low.map(num),
      mid: state._smoothingBuffers.mid.map(num),
      high: state._smoothingBuffers.high.map(num)
    },
    frequencyData: state._frequencyData ? [...state._frequencyData] : null,
    maxBufferLength: num(state._maxBufferLength)
  }
}

// `cache` holds each state's own snapshot text from the previous step (keyed
// 'root', 'default:<n>', 'device:<id>:<n>'); a state unchanged since then is
// written "unchanged". The first step writes every state in full, so equal
// streams mean equal states after every step.
function audioOwnEntry (state, key, cache) {
  const own = audioOwnSnapshot(state)
  const text = JSON.stringify(own)
  const unchanged = cache.get(key) === text
  cache.set(key, text)
  return unchanged ? 'unchanged' : own
}

function audioSnapshot (state, cache) {
  const out = { root: audioOwnEntry(state, 'root', cache) }
  if (!state._devices) return out
  out.devices = [...state._devices.values()].map(entry => ({
    id: entry.id,
    name: entry.name,
    connected: entry.connected,
    channelCount: entry.channelCount,
    channels: [...entry.channels].map(([n, s]) => [n, audioOwnEntry(s, `device:${entry.id}:${n}`, cache)])
  }))
  out.devicesByName = [...state._devicesByName].map(([name, entry]) => [name, entry === null ? null : entry.id])
  out.deviceInventory = state._deviceInventory ? [...state._deviceInventory] : null
  out.defaultChannels = [...state._defaultChannels].map(([n, s]) => [n, audioOwnEntry(s, `default:${n}`, cache)])
  out.defaultConnected = state._defaultConnected
  return out
}

function audioStateName (root, state) {
  if (state === null || state === undefined) return null
  if (state === root) return 'root'
  for (const [n, s] of root._defaultChannels || []) if (s === state) return `default:${n}`
  for (const entry of root._devices?.values() || []) {
    for (const [n, s] of entry.channels) if (s === state) return `device:${entry.id}:${n}`
  }
  return '<unknown>'
}

function audioTarget (root, target) {
  if (target === 'root' || target === undefined) return root
  const state = 'default' in target
    ? root._defaultChannels.get(target.default)
    : root._devices.get(target.device)?.channels.get(target.channel)
  if (!state) throw new Error(`no audio target ${JSON.stringify(target)}`)
  return state
}

function decodeValues (values) {
  const out = {}
  for (const [key, value] of Object.entries(values)) out[key] = readNum(value)
  return out
}

function audioStep (root, step) {
  switch (step.op) {
    case 'registerDevice': {
      const device = { id: step.id, name: step.name }
      if (step.channelCount !== undefined) device.channelCount = step.channelCount
      return !!root.registerDevice(device)
    }
    case 'setChannelValues': return root.setChannelValues(step.id, step.channel, decodeValues(step.values))
    case 'setDeviceRawUnavailable': root.setDeviceRawUnavailable(step.id); return null
    case 'disconnectDevice': root.disconnectDevice(step.id); return null
    case 'setDeviceInventory': root.setDeviceInventory(step.devices); return null
    case 'registerDefaultChannels': return root.registerDefaultChannels(step.count) !== null
    case 'disconnectDefaultInput': root.disconnectDefaultInput(); return null
    case 'getDefaultChannelState': return audioStateName(root, root.getDefaultChannelState(step.channel))
    case 'getDeviceChannelState': {
      const selector = {}
      for (const [key, value] of Object.entries(step.selector)) selector[key] = typeof value === 'object' && value !== null ? readNum(value) : value
      return audioStateName(root, root.getDeviceChannelState(selector))
    }
    case 'devices': return root.getDevices()
  }
  const state = audioTarget(root, step.target)
  switch (step.op) {
    case 'setBands': state.setBands(readNum(step.low), readNum(step.mid), readNum(step.high)); return null
    case 'setRaw': state.setRaw(readNum(step.value)); return null
    case 'setRawUnavailable': state.setRawUnavailable(); return null
    case 'setField': state[step.field] = typeof step.value === 'boolean' ? step.value : readNum(step.value); return null
    case 'updateFromAnalyser': {
      const analyser = step.analyser && {
        frequencyBinCount: step.analyser.bins,
        getByteFrequencyData (buffer) {
          const data = Uint8Array.from(step.analyser.data)
          buffer.set(data.subarray(0, Math.min(buffer.length, data.length)))
        }
      }
      if (step.smoothing === undefined) state.updateFromAnalyser(analyser)
      else state.updateFromAnalyser(analyser, readNum(step.smoothing))
      return null
    }
    case 'setSpectrum': state.setSpectrum(Uint8Array.from(step.data)); return null
    case 'setWaveform': state.setWaveform(Uint8Array.from(step.data)); return null
    case 'smooth': return num(state._smooth(step.band, readNum(step.value)))
    case 'setMaxBufferLength': state._maxBufferLength = readNum(step.value); return null
    case 'resetAggregate': state.resetAggregate(); return null
    case 'reset': state.reset(); return null
    default: throw new Error(`unknown audio op ${step.op}`)
  }
}

export async function runAudio (scenarios, emit) {
  for (const scenario of scenarios) {
    const { AudioState } = await freshExternalInput()
    const root = new AudioState({ deviceRegistry: scenario.registry !== false })
    const cache = new Map()
    scenario.steps.forEach((step, index) => {
      const ret = audioStep(root, step)
      emit({ scenario: scenario.name, step: index, ret, snapshot: audioSnapshot(root, cache) })
    })
  }
}

// ---------------------------------------------------------------- automation
//
// Scenario: {name, midi: null | {registry, time, steps}, audio: null |
// {registry, steps}, evaluations: [{value, time, spec?, wallTime}],
// requirements: [{passes: [{uniforms, audioTagged}]}]}. The MIDI and audio
// states are built with the steps above; each evaluation is
// Pipeline.resolveUniformValue(value, time, spec) with Date.now() returning
// wallTime; each requirement set is getAudioInputRequirements() of a graph
// with those passes (audio-tagged passes use an effect registered with
// tags: ['audio']). Values use the JavaScript value conventions.

const AUDIO_TAGGED_EFFECT = 'nmInputParity.audioTagged'

export async function runAutomation (scenarios, emit) {
  const { Pipeline } = await import(pathToFileURL(join(RUNTIME, 'pipeline.js')).href)
  const { registerEffect } = await import(pathToFileURL(join(RUNTIME, 'registry.js')).href)
  registerEffect(AUDIO_TAGGED_EFFECT, {
    name: 'Audio tagged parity effect', namespace: 'nmInputParity', func: 'audioTagged', tags: ['audio'], passes: []
  })
  const realNow = Date.now
  try {
    for (const scenario of scenarios) {
      const { MidiState, AudioState } = await freshExternalInput()
      let now = scenario.midi?.time ?? 0
      Date.now = () => now
      const pipeline = new Pipeline(null, null)
      if (scenario.midi) {
        const midi = new MidiState({ portRegistry: scenario.midi.registry !== false })
        for (const step of scenario.midi.steps) {
          if (step.time !== undefined) now = step.time
          midiStep(midi, step)
        }
        pipeline.setMidiState(midi)
      }
      if (scenario.audio) {
        const audio = new AudioState({ deviceRegistry: scenario.audio.registry !== false })
        for (const step of scenario.audio.steps) audioStep(audio, step)
        pipeline.setAudioState(audio)
      }
      scenario.evaluations.forEach((evaluation, index) => {
        const value = decodeValue(evaluation.value)
        const spec = evaluation.spec === undefined ? undefined : decodeValue(evaluation.spec)
        now = evaluation.wallTime
        let result
        try {
          const resolved = pipeline.resolveUniformValue(value, readNum(evaluation.time), spec)
          result = Object.is(resolved, value) ? { passthrough: true } : { value: num(resolved) }
        } catch (error) {
          result = { error: String(error?.message ?? error) }
        }
        emit({ scenario: scenario.name, evaluation: index, result })
      })
      ;(scenario.requirements || []).forEach((set, index) => {
        const passes = set.passes.map(pass => ({
          ...(pass.audioTagged ? { effectKey: AUDIO_TAGGED_EFFECT } : {}),
          uniforms: decodeValue(pass.uniforms)
        }))
        emit({ scenario: scenario.name, requirements: index, result: new Pipeline({ passes }, null).getAudioInputRequirements() })
      })
    }
  } finally {
    Date.now = realNow
  }
}

// ---------------------------------------------------------------- math
//
// Scenario: {name, inputs: [numbers]} -> {name, sin: [...], cos: [...],
// round: [...]}: Math.sin, Math.cos and Math.round of every input.

export function runMath (scenarios, emit) {
  for (const scenario of scenarios) {
    const inputs = scenario.inputs.map(readNum)
    emit({
      scenario: scenario.name,
      sin: inputs.map(x => num(Math.sin(x))),
      cos: inputs.map(x => num(Math.cos(x))),
      round: inputs.map(x => num(Math.round(x)))
    })
  }
}

// Runs scenarios of one kind, calling emit(record) for every record.
export async function runReference (kind, scenarios, emit) {
  switch (kind) {
    case 'midi': return runMidi(scenarios, emit)
    case 'audio': return runAudio(scenarios, emit)
    case 'automation': return runAutomation(scenarios, emit)
    case 'math': return runMath(scenarios, emit)
    default: throw new Error(`unknown scenario kind ${kind}`)
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  // The reference logs with console.*; keep stdout for records.
  console.log = (...args) => process.stderr.write(args.map(String).join(' ') + '\n')
  console.warn = console.log
  console.info = console.log
  const args = process.argv.slice(2)
  const outIndex = args.indexOf('--out')
  const out = outIndex >= 0 ? args.splice(outIndex, 2)[1] : null
  if (args.length !== 2) {
    console.error('usage: reference-input.mjs <midi|audio|automation|math> <scenarios.json> [--out results.jsonl]')
    process.exit(2)
  }
  const fd = out ? openSync(out, 'w') : 1
  await runReference(args[0], JSON.parse(readFileSync(args[1], 'utf8')), record => {
    writeSync(fd, JSON.stringify(record) + '\n')
  })
  if (out) closeSync(fd)
}
