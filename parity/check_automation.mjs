#!/usr/bin/env node
// check_automation.mjs — gate noisemaker-input's automation evaluators against
// the reference pipeline (shaders/src/runtime/pipeline.js).
//
// Reference: Pipeline.resolveUniformValue(value, time, paramSpec) on a
// GPU-less `new Pipeline(null, null)` whose external state is a reference
// MidiState/AudioState built by scripted steps (Date.now() stubbed), and
// Pipeline.getAudioInputRequirements() of graphs built from pass lists.
// Candidate: `nm-input-dump automation` (noisemaker_input::automation over the
// ported MidiState/AudioState). Every result must be identical: the same
// number bit for bit, or both a passthrough.
//
// Descriptors come from three sources:
//   - real compiler output: DSL programs compiled by the reference compiler
//     (osc/midi/audio in every mode, let-bound and inline nesting, device and
//     port selectors, zones, invalid descriptors carrying _invalid and _ast),
//     serialized with object identity preserved ($id/$ref);
//   - the descriptors of the reference's test_midi.js, test_audio.js,
//     test_oscillators.js, test_nested_automation.js and
//     test_midi_audio_integration.js;
//   - seeded generated descriptors: every kind, every field drawn from valid
//     values and adversarial ones (NaN, ±Infinity, -0, non-integers, numeric
//     strings, booleans, null, arrays, objects), nesting beyond the depth
//     limit, shared nodes and cycles, _ast-only typing and _invalid variants.
// Each is evaluated at many normalized times, wall-clock times and parameter
// specs (none, ranges, int ranges, non-finite and non-object specs), against
// no state, rich MIDI/audio states (ports, notes, CC/CC14, NRPN, bend,
// pressures, MPE zones, devices, default channels, raw readiness) and a
// registry-less MIDI state. Requirements cover generated pass sets with
// nested, shared, cyclic, valid and invalid selected descriptors and
// audio-tagged passes.
//
// The gate also compares Math.sin, Math.cos and Math.round with
// noisemaker_input::jsmath (V8's fdlibm port) over a few hundred thousand
// inputs from tiny to huge magnitudes, around multiples of pi/2 and at halves.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_automation.mjs [--verbose]
// Env:
//   NM_INPUT_DUMP   candidate binary (default target/release/nm-input-dump)
// Exit 0 when every result matches; 1 otherwise.

import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { compareKind, mulberry32, requireDump } from '../tools/reference-input-compare.mjs'

const VERBOSE = process.argv.includes('--verbose')
requireDump()
if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const LANG = join(resolve(process.env.NM_REFERENCE_ROOT), 'shaders', 'src', 'lang')

const T0 = 1_700_000_000_000

// ---------------------------------------------------------------- JS value encoding

// Encodes a JavaScript value with the gate's JSON conventions, giving every
// object reached more than once (shared or cyclic) an $id.
function encodeJs (root) {
  const counts = new Map()
  const count = (value) => {
    if (!value || (typeof value !== 'object')) return
    counts.set(value, (counts.get(value) || 0) + 1)
    if (counts.get(value) > 1) return
    for (const item of Array.isArray(value) ? value : Object.values(value)) count(item)
  }
  count(root)
  const ids = new Map()
  let next = 0
  const encode = (value) => {
    if (value === undefined) return { $undefined: true }
    if (typeof value === 'function') return { $function: value.toString() }
    if (typeof value === 'number') {
      if (Number.isNaN(value)) return { $num: 'NaN' }
      if (value === Infinity) return { $num: 'Infinity' }
      if (value === -Infinity) return { $num: '-Infinity' }
      if (Object.is(value, -0)) return { $num: '-0' }
      return value
    }
    if (!value || typeof value !== 'object') return value
    if (ids.has(value)) return { $ref: ids.get(value) }
    const shared = counts.get(value) > 1
    if (shared) ids.set(value, `n${next++}`)
    if (Array.isArray(value)) {
      const items = value.map(encode)
      return shared ? { $id: ids.get(value), $array: items } : items
    }
    const out = shared ? { $id: ids.get(value) } : {}
    for (const [key, item] of Object.entries(value)) out[key] = encode(item)
    return out
  }
  return encode(root)
}

// ---------------------------------------------------------------- external states

const left = { id: 'port-a', name: 'Keys' }
const right = { id: 'port-b', name: 'Keys' }
const pads = { id: 'port-c', name: 'Pads' }
const msg = (data, port, time) => ({ op: 'message', data, ...(port ? { port } : {}), ...(time !== undefined ? { time } : {}) })

const MIDI_RICH = {
  registry: true,
  time: T0,
  steps: [
    msg([0x90, 60, 100], left, T0 - 400), msg([0x90, 64, 90], undefined, T0 - 250), msg([0x91, 67, 127], right, T0 - 900),
    msg([0x94, 72, 50], pads, T0 - 10), msg([0xb0, 1, 64], left), msg([0xb0, 33, 99], left), msg([0xb0, 74, 127], right),
    msg([0xb1, 7, 100]), msg([0xb0, 99, 1], left), msg([0xb0, 98, 2], left), msg([0xb0, 6, 64], left), msg([0xb0, 38, 5], left),
    msg([0xb0, 101, 0], pads), msg([0xb0, 100, 0], pads), msg([0xb0, 6, 2], pads), msg([0xe0, 0, 96], left), msg([0xe1, 127, 127]),
    msg([0xd0, 77], left), msg([0xd1, 33], right), msg([0xa0, 60, 55], left), msg([0xa0, 64, 12]),
    ...[[101, 0], [100, 6], [6, 7]].map(([c, v]) => msg([0xb0, c, v], left)), msg([0x92, 62, 80], left, T0 - 120),
    msg([0x93, 65, 70], left, T0 - 60), msg([0xa2, 62, 30], left), msg([0xe2, 0, 32], left),
    ...[[101, 0], [100, 6], [6, 3]].map(([c, v]) => msg([0xbf, c, v], right)), msg([0x9e, 50, 60], right, T0 - 30),
    msg([0xf8]), msg([0xf8], left), { op: 'inventory', ports: [{ ...left, connected: true }, { ...right, connected: true }, { ...pads, connected: true }] }
  ]
}
const MIDI_PLAIN = {
  registry: false,
  time: T0,
  steps: [
    msg([0x90, 60, 100], undefined, T0 - 300), msg([0x93, 40, 20], undefined, T0 - 2000), msg([0xb0, 1, 127]), msg([0xb0, 33, 127]),
    { op: 'setField', target: 'root', channel: 5, field: 'key', value: 99 },
    { op: 'setField', target: 'root', channel: 5, field: 'gate', value: 1 },
    { op: 'setField', target: 'root', channel: 5, field: 'velocity', value: 127 },
    { op: 'setField', target: 'root', channel: 5, field: 'time', value: T0 - 500.5 }
  ]
}
const AUDIO_RICH = {
  registry: true,
  steps: [
    { op: 'setField', target: 'root', field: 'low', value: 0.7 }, { op: 'setField', target: 'root', field: 'mid', value: 1.5 },
    { op: 'setField', target: 'root', field: 'high', value: -0.25 }, { op: 'setField', target: 'root', field: 'vol', value: 0.65 },
    { op: 'setRaw', target: 'root', value: -0.35 },
    { op: 'registerDevice', id: 'dev-a', name: 'Interface', channelCount: 4 },
    { op: 'registerDevice', id: 'dev-b', name: 'Interface', channelCount: 2 },
    { op: 'registerDevice', id: 'dev-c', name: 'Mixer', channelCount: 32 },
    { op: 'setChannelValues', id: 'dev-a', channel: 2, values: { low: 0.35, mid: 0.2, high: 0.9, vol: 0.4, raw: -0.5 } },
    { op: 'setChannelValues', id: 'dev-b', channel: 2, values: { low: 0.85, raw: 0.5 } },
    { op: 'setChannelValues', id: 'dev-c', channel: 32, values: { raw: 0.8, low: 0.123 } },
    { op: 'setChannelValues', id: 'dev-c', channel: 31, values: { raw: -0.6 } },
    { op: 'registerDefaultChannels', count: 3 },
    { op: 'setRaw', target: { default: 1 }, value: 0.25 }, { op: 'setBands', target: { default: 2 }, low: 0.1, mid: 0.2, high: 0.3 },
    { op: 'registerDevice', id: 'dev-d', name: 'Gone', channelCount: 1 }, { op: 'disconnectDevice', id: 'dev-d' }
  ]
}
const AUDIO_UNREADY = {
  registry: true,
  steps: [
    { op: 'setBands', target: 'root', low: 0.3, mid: 0.6, high: 0.9 },
    { op: 'registerDevice', id: 'dev-a', name: 'Interface', channelCount: 2 },
    { op: 'setChannelValues', id: 'dev-a', channel: 1, values: { low: 0.5 } }
  ]
}

const STATES = [
  { name: 'none', midi: null, audio: null },
  { name: 'rich', midi: MIDI_RICH, audio: AUDIO_RICH },
  { name: 'plain', midi: MIDI_PLAIN, audio: AUDIO_UNREADY }
]

const TIMES = [0, 0.0125, 0.25, 0.5, 0.61, 0.75, 0.83, 1, -0.3, 2.7, 1e-9, 1e6, 0.123456789, NaN]
const WALL_TIMES = [T0, T0 + 1, T0 + 250, T0 + 999, T0 + 5000, T0 - 100]
const SPECS = [undefined, null, { min: 0, max: 1 }, { min: 10, max: 50 }, { min: -5, max: 5, type: 'int' },
  { min: 0, max: 100, type: 'int' }, { min: NaN, max: 1 }, { min: 'a', max: 2 }, { type: 'int' }, 5, 'spec',
  { min: -1e300, max: 1e300 }, { min: 3, max: 3, type: 'int' }, []]

// ---------------------------------------------------------------- compiled descriptors

const { compile } = await import(pathToFileURL(join(LANG, 'index.js')).href)
const { registerOp } = await import(pathToFileURL(join(LANG, 'ops.js')).href)
const { registerStarterOps } = await import(pathToFileURL(join(LANG, 'validator.js')).href)
registerOp('synth.nestedAutomationProbe', {
  name: 'nestedAutomationProbe',
  args: [{ name: 'amount', type: 'float', default: 0, min: 0, max: 1 }]
})
registerStarterOps(['synth.nestedAutomationProbe'])

const probe = (lets, amount) => `search synth\n${lets.join('\n')}\nnestedAutomationProbe(amount: ${amount}).write(o0)`
const PROGRAMS = [
  probe([], 'osc(type: oscKind.sine)'),
  probe([], 'osc(type: oscKind.tri, min: 0.2, max: 0.8, speed: 2, offset: 0.25, seed: 42)'),
  probe([], 'osc(type: oscKind.saw, min: -2, max: 10, speed: -3)'),
  probe([], 'osc(type: oscKind.sawInv, speed: 0)'),
  probe([], 'osc(type: oscKind.square, offset: 0.5)'),
  probe([], 'osc(type: oscKind.noise, seed: 37)'),
  probe([], 'osc(type: oscKind.noise2d, speed: 2, seed: 42)'),
  probe([], 'osc(type: 6, seed: 7, speed: 3.5)'),
  probe(['let rate = osc(type: oscKind.sine, min: 0.25, max: 0.75)', 'let carrier = osc(type: oscKind.saw, speed: rate)'], 'carrier'),
  probe(['let rate = osc(type: oscKind.sine)', 'let carrier = osc(type: oscKind.saw, speed: rate)'], 'carrier'),
  probe(['let rate = osc(type: oscKind.noise, seed: 37)', 'let carrier = osc(type: oscKind.sine, speed: rate)'], 'carrier'),
  probe(['let rate = osc(type: oscKind.noise2d, seed: 9)', 'let carrier = osc(type: oscKind.noise2d, speed: rate, offset: rate)'], 'carrier'),
  probe(['let rate = midi(channel: 1, mode: midiMode.gateVelocity)', 'let carrier = osc(type: oscKind.saw, speed: rate)'], 'carrier'),
  probe(['let rate = audio(band: audioBand.raw)', 'let carrier = osc(type: oscKind.saw, speed: rate)'], 'carrier'),
  probe(['let shape = osc(type: oscKind.sine)', 'let rate = midi(channel: 1, mode: midiMode.gateVelocity, min: shape, max: shape)', 'let carrier = osc(type: oscKind.saw, speed: rate)'], 'carrier'),
  probe(['let rate = audio(band: audioBand.raw, min: 0.25, max: 0.75)', 'let carrier = osc(type: oscKind.sine, min: rate, speed: rate)'], 'carrier'),
  probe(['let rate8 = osc(type: oscKind.sine)', 'let rate7 = osc(type: oscKind.sine, speed: rate8)', 'let rate6 = osc(type: oscKind.tri, speed: rate7)',
    'let rate5 = osc(type: oscKind.sine, speed: rate6)', 'let rate4 = osc(type: oscKind.noise, speed: rate5)', 'let rate3 = osc(type: oscKind.sine, speed: rate4)',
    'let rate2 = osc(type: oscKind.sine, speed: rate3)', 'let rate1 = osc(type: oscKind.sine, speed: rate2)', 'let carrier = osc(type: oscKind.saw, speed: rate1)'], 'carrier'),
  probe(['let first = osc(type: oscKind.sine, speed: second)', 'let second = osc(type: oscKind.tri, speed: first)'], 'first'),
  probe(['let movement = midi(channel: 1)', 'let gate = audio(band: audioBand.vol, min: movement)'], 'gate'),
  probe(['let gate = audio(band: audioBand.vol, min: "not-a-number")'], 'gate'),
  probe(['let gate = audio(band: audioBand.vol, channel: "not-a-channel", name: "Input")'], 'gate'),
  probe(['let inner = audio(band: audioBand.raw, channel: 2, name: "Inner Interface", id: "inner-id")',
    'let outer = audio(band: audioBand.low, min: inner, channel: 1, name: "Outer Interface", id: "outer-id")'], 'outer'),
  ...['noteChange', 'gateNote', 'gateVelocity', 'triggerNote', 'velocity'].flatMap(mode => [
    probe([], `midi(channel: 1, mode: midiMode.${mode}, min: 0.1, max: 0.9, sensitivity: 3)`),
    probe([], `midi(channel: 5, mode: midiMode.${mode}, name: "Keys", id: "port-a")`),
    probe([], `midi(channel: 0, mode: midiMode.${mode})`),
    probe([], `midi(channel: 2, mode: midiMode.${mode}, name: "Keys")`)
  ]),
  probe([], 'midi(1, midiMode.velocity, 0, 0.8)'),
  probe([], 'midi(name: "Keys", id: "port-b", 2, 1, 0.25, 0.75, 2)'),
  ...['cc', 'cc14'].flatMap(mode => [
    probe([], `midi(channel: 1, mode: midiMode.${mode})`),
    probe([], `midi(channel: 1, mode: midiMode.${mode}, cc: 33, name: "Keys", id: "port-a")`),
    probe([], `midi(channel: 2, mode: midiMode.${mode}, cc: 74)`),
    probe([], `midi(channel: 1, mode: midiMode.${mode}, cc: 200)`),
    probe(['let controller = 1', `let knob = midi(channel: 1, mode: midiMode.${mode}, cc: controller, name: "Keys", id: "port-a")`], 'osc(speed: knob)')
  ]),
  ...['nrpn', 'pitchBend', 'pressure', 'polyPressure'].flatMap(mode => [
    probe([], `midi(channel: 1, mode: midiMode.${mode}${mode === 'nrpn' ? ', nrpn: 130' : ''})`),
    probe([], `midi(channel: 1, mode: midiMode.${mode}${mode === 'nrpn' ? ', nrpn: 130' : ''}, name: "Keys", id: "port-a")`),
    probe([], `midi(zone: midiZone.lower, mode: midiMode.${mode}${mode === 'nrpn' ? ', nrpn: 130' : ''}, name: "Keys", id: "port-a")`),
    probe([], `midi(zone: midiZone.upper, members: 3, mode: midiMode.${mode}${mode === 'nrpn' ? ', nrpn: 130' : ''})`),
    probe(['let zone = midiZone.upper', 'let size = 7', 'let parameter = 1234', `let expression = midi(zone: zone, members: size, mode: midiMode.${mode}${mode === 'nrpn' ? ', nrpn: parameter' : ''}, name: "Keys", id: "port-b")`], 'osc(speed: expression)')
  ]),
  probe([], 'midi(zone: midiZone.lower, mode: midiMode.noteChange)'),
  probe([], 'midi(zone: midiZone.lower, members: 15, mode: midiMode.velocity, sensitivity: 2)'),
  probe([], 'midi(channel: 1, mode: midiMode.nrpn, nrpn: 16383)'),
  probe([], 'midi(zone: midiZone.upper, members: 16, mode: midiMode.pressure)'),
  probe([], 'midi(channel: 17, mode: midiMode.pitchBend)'),
  ...['low', 'mid', 'high', 'vol', 'raw'].flatMap(band => [
    probe([], `audio(band: audioBand.${band})`),
    probe([], `audio(band: audioBand.${band}, min: 0.2, max: 0.8)`),
    probe([], `audio(band: audioBand.${band}, channel: 2, name: "Interface", id: "dev-a")`),
    probe([], `audio(band: audioBand.${band}, channel: 2, name: "Interface")`),
    probe([], `audio(band: audioBand.${band}, channel: 32, name: "Mixer", id: "dev-c")`),
    probe([], `audio(band: audioBand.${band}, channel: 1)`),
    probe([], `audio(band: audioBand.${band}, channel: 2)`)
  ]),
  probe([], 'audio(band: 4)'),
  probe([], 'audio(band: 5)'),
  probe([], 'audio(band: audioBand.raw, channel: 0, name: "Interface", id: "dev-a")'),
  probe([], 'audio(band: audioBand.raw, channel: 33)'),
  probe([], 'audio(band: audioBand.raw, channel: true, name: "Interface")'),
  probe([], 'audio(audioBand.high, 0.2, 0.8, channel: 3, name: "Interface", id: "dev-a")'),
  probe([], 'audio(band: audioBand.raw, 0.25, 0.75, channel: 2, name: "Interface")'),
  probe(['let cv = audio(band: audioBand.raw, channel: 2, name: "Interface", id: "dev-a")'], 'osc(type: oscKind.sine, min: cv, max: cv, speed: cv, offset: cv, seed: cv)')
]

const compiled = []
for (const source of PROGRAMS) {
  let result
  try { result = compile(source) } catch { continue }
  const amount = result.plans?.[0]?.chain?.[0]?.args?.amount
  if (amount !== undefined) compiled.push(amount)
}

// ---------------------------------------------------------------- reference test descriptors

const midi = (fields) => ({ type: 'Midi', min: 0, max: 1, sensitivity: 1, ...fields })
const audio = (fields) => ({ type: 'Audio', min: 0, max: 1, ...fields })
const osc = (fields) => ({ type: 'Oscillator', oscType: 0, min: 0, max: 1, speed: 1, offset: 0, seed: 1, ...fields })
const REFERENCE_DESCRIPTORS = [
  ...[0, 1, 2, 3, 4].map(mode => midi({ channel: 1, mode })),
  midi({ channel: 5, mode: 2 }), midi({ channel: 1, mode: 0, min: 10, max: 20 }), midi({ channel: 1, mode: 1, min: 5, max: 10 }),
  midi({ channel: 1, mode: 3, sensitivity: 4 }),
  midi({ channel: 2, mode: 5, cc: 74, min: 0.2, max: 0.8, id: 'port-a', name: 'Keys' }),
  midi({ channel: 1, mode: 6, cc: 1 }), midi({ channel: 1, mode: 5 }),
  ...[[5, 128], [6, 32], [6, -1], [5, 0.5]].map(([mode, cc]) => midi({ channel: 1, mode, cc, min: 0.2 })),
  ...[5, 6].flatMap(mode => [0, 17, 1.5, true, '1', undefined].map(channel => midi({ channel, mode, min: 0.2 }))),
  ...[8, 9, 10].map(mode => midi({ channel: 2, mode, id: 'port-b', name: 'Keys' })),
  midi({ mode: 7, nrpn: 130, channel: 1, id: 'port-a', name: 'Keys' }), midi({ mode: 7, nrpn: 130, channel: 1 }),
  ...[0, 5, 8, 10].map(mode => midi({ zone: 0, mode, cc: 74 })),
  ...[0, 8].map(mode => midi({ zone: 0, mode, cc: 74, id: 'port-a', name: 'Keys' })),
  midi({ zone: 1, mode: 0 }), midi({ zone: 1, members: 1, mode: 0 }), midi({ zone: 0, members: 15, mode: 0 }),
  ...[{ channel: 17, mode: 8 }, { channel: 0, mode: 9 }, { channel: 1.5, mode: 10 }, { channel: 2, mode: 7, nrpn: 16383 },
    { zone: 0, channel: 2, mode: 9 }, { zone: 2, mode: 9 }, { zone: 0, members: 16, mode: 9 }, { channel: 2, members: 2, mode: 9 }]
    .map(fields => ({ type: 'Midi', min: 0.2, max: 1, ...fields })),
  audio({ band: 0 }), audio({ band: 0, min: 0.2, max: 0.8 }), audio({ band: 1 }), audio({ band: 2 }), audio({ band: 3, min: 0, max: 0.5 }),
  audio({ band: 4, min: 0.2, max: 0.8 }), audio({ band: 0, min: 0.2, max: 0.9, _invalid: true }),
  audio({ band: 4, channel: 1, name: 'Interface', id: 'dev-a' }), audio({ band: 0, channel: 2, name: 'Interface', id: 'dev-b' }),
  audio({ band: 4, channel: 2, name: 'Interface', id: 'dev-b' }), audio({ band: 0, channel: 1, name: 'Missing', id: 'gone', min: 0.25, max: 0.75 }),
  audio({ band: 0, channel: 1, name: 'Interface', min: 0.25, max: 0.75 }), audio({ band: 0, channel: 2 }), audio({ band: 0, channel: 3 }),
  audio({ band: 4, channel: 32, name: 'Mixer', id: 'dev-c' }), audio({ band: 4, channel: 31, name: 'Mixer', id: 'dev-c' }),
  audio({ band: 4, name: 'Interface', _ast: { type: 'Audio', band: { type: 'Member', path: ['audioBand', 'raw'] }, channel: { type: 'Number', value: 0 }, name: { type: 'String', value: 'Interface' } } }),
  ...[0, 1, 2, 3, 4, 5].flatMap(oscType => [7, 42].map(seed => osc({ oscType, seed }))),
  ...[1, 2, 3].flatMap(speed => [0, 0.25].flatMap(offset => [7, 42].map(seed => osc({ oscType: 6, speed, offset, seed })))),
  osc({ oscType: 6, speed: 3.5, offset: 0.25, seed: 7 }),
  audio({ band: 0, min: audio({ band: 0, channel: 1, name: 'Inner Interface', id: 'inner-id' }), max: 1, _invalid: true })
]

// ---------------------------------------------------------------- generated descriptors

function generated (seed, count) {
  const rand = mulberry32(seed)
  const pick = items => items[Math.floor(rand() * items.length)]
  const NUMBERS = [0, 1, 0.5, 0.25, -0.5, 2, 3, 4, 5, 6, 7, 10, 42, 1e-9, -3, 0.9999, 15, 16, 17, 31, 32, 33, 127, 128, 130, 16382, 16383,
    NaN, Infinity, -Infinity, -0, 1.5, 4.5, -20, 20, 9999, 1e300, -1e-300]
  const STRINGS = ['', '1', '2', '5', '16', '05', ' 3 ', 'abc', '0x1F', 'Infinity', '1e1', '-0', '.5', 'Interface', 'Keys', 'Mixer', 'Pads', 'port-a', 'port-b', 'dev-a', 'dev-c']
  const OTHERS = () => pick([true, false, null, undefined, [], [3], ['2'], [1, 2], [[4]], {}, { type: 'Other' }, [null], [true], [5.5]])
  const scalar = () => {
    const r = rand()
    if (r < 0.6) return pick(NUMBERS)
    if (r < 0.8) return pick(STRINGS)
    return OTHERS()
  }
  const shared = []
  const descriptor = (depth) => {
    if (shared.length && rand() < 0.1) return pick(shared)
    const kind = pick(['Oscillator', 'Midi', 'Audio'])
    const field = (valid) => {
      if (depth < 10 && rand() < 0.18) return descriptor(depth + 1)
      return rand() < 0.65 ? valid() : scalar()
    }
    const d = {}
    if (rand() < 0.9) d.type = kind
    if (kind === 'Oscillator') {
      if (rand() < 0.95) d.oscType = rand() < 0.85 ? Math.floor(rand() * 7) : scalar()
      for (const key of ['min', 'max', 'speed', 'offset', 'seed']) {
        if (rand() < 0.9) d[key] = field(() => key === 'seed' ? Math.floor(rand() * 100) : key === 'speed' ? pick([1, 2, -1, 0, 0.5, 3]) : Math.round(rand() * 100) / 100)
      }
    } else if (kind === 'Midi') {
      if (rand() < 0.85) d.mode = rand() < 0.85 ? Math.floor(rand() * 12) : scalar()
      if (rand() < 0.7) d.channel = rand() < 0.75 ? pick([1, 2, 3, 4, 5, 15, 16]) : scalar()
      if (rand() < 0.2) d.zone = rand() < 0.8 ? pick([0, 1]) : scalar()
      if (rand() < 0.15) d.members = rand() < 0.8 ? pick([1, 2, 3, 7, 15]) : scalar()
      if (rand() < 0.4) d.cc = rand() < 0.8 ? pick([1, 7, 33, 74, 0, 31, 127]) : scalar()
      if (rand() < 0.3) d.nrpn = rand() < 0.8 ? pick([130, 2, 0, 16382]) : scalar()
      for (const key of ['min', 'max', 'sensitivity']) if (rand() < 0.85) d[key] = field(() => Math.round(rand() * 100) / 100)
      if (rand() < 0.4) d.name = rand() < 0.8 ? pick(['Keys', 'Pads', 'Missing']) : scalar()
      if (rand() < 0.3) d.id = rand() < 0.8 ? pick(['port-a', 'port-b', 'port-c', 'nope']) : scalar()
    } else {
      if (rand() < 0.95) d.band = rand() < 0.85 ? Math.floor(rand() * 6) : scalar()
      for (const key of ['min', 'max']) if (rand() < 0.85) d[key] = field(() => Math.round(rand() * 100) / 100)
      if (rand() < 0.45) d.channel = rand() < 0.8 ? pick([1, 2, 3, 31, 32]) : scalar()
      if (rand() < 0.4) d.name = rand() < 0.8 ? pick(['Interface', 'Mixer', 'Gone', 'Missing']) : scalar()
      if (rand() < 0.3) d.id = rand() < 0.8 ? pick(['dev-a', 'dev-b', 'dev-c', 'dev-d', 'nope']) : scalar()
    }
    if (rand() < 0.15) d._invalid = pick([true, 1, 0, '', 'yes', null, false])
    if (rand() < 0.2 || d.type === undefined) {
      d._ast = { type: rand() < 0.7 ? kind : pick(['Audio', 'Midi', 'Oscillator', 'Number', 'Ident']) }
      if (rand() < 0.4) d._ast.name = rand() < 0.5 ? { type: 'String', value: 'Interface' } : 'Interface'
      if (rand() < 0.3) d._ast.id = { type: 'String', value: 'dev-a' }
      if (rand() < 0.4) d._ast.channel = { type: 'Number', value: 2 }
    }
    if (rand() < 0.15) shared.push(d)
    // Occasionally close a cycle through a numeric field.
    if (shared.length > 1 && rand() < 0.03) {
      const target = pick(shared)
      const key = target.type === 'Midi' || target._ast?.type === 'Midi' ? pick(['min', 'max', 'sensitivity']) : pick(['min', 'max', 'speed'])
      target[key] = d
    }
    return d
  }
  const out = []
  for (let i = 0; i < count; i++) out.push(rand() < 0.97 ? descriptor(0) : scalar())
  return out
}

// ---------------------------------------------------------------- scenarios

function evaluations (descriptors, rand, perDescriptor) {
  const pick = items => items[Math.floor(rand() * items.length)]
  const out = []
  for (const descriptor of descriptors) {
    const value = encodeJs(descriptor)
    for (let i = 0; i < perDescriptor; i++) {
      const spec = pick(SPECS)
      out.push({ value, time: pick(TIMES), wallTime: pick(WALL_TIMES), ...(spec === undefined ? {} : { spec: encodeJs(spec) }) })
    }
  }
  return out
}

const scenarios = []
const rand = mulberry32(1234)
for (const state of STATES) {
  scenarios.push({ name: `compiled-${state.name}`, midi: state.midi, audio: state.audio, evaluations: evaluations(compiled, rand, 24) })
  scenarios.push({ name: `reference-${state.name}`, midi: state.midi, audio: state.audio, evaluations: evaluations(REFERENCE_DESCRIPTORS, rand, 14) })
  scenarios.push({ name: `generated-${state.name}`, midi: state.midi, audio: state.audio, evaluations: evaluations(generated(77 + state.name.length, 2500), rand, 4) })
}

// Requirements of generated pass sets.
{
  const prand = mulberry32(99)
  const pick = items => items[Math.floor(prand() * items.length)]
  const pool = [...compiled, ...REFERENCE_DESCRIPTORS, ...generated(5, 400)].filter(d => d && typeof d === 'object')
  const requirements = []
  for (let i = 0; i < 600; i++) {
    const passes = []
    const n = 1 + Math.floor(prand() * 4)
    const reused = pick(pool)
    for (let p = 0; p < n; p++) {
      const uniforms = {}
      const m = Math.floor(prand() * 5)
      for (let u = 0; u < m; u++) {
        const r = prand()
        uniforms[`u${u}`] = r < 0.5 ? pick(pool) : r < 0.65 ? [pick(pool), 3, [pick(pool)]] : r < 0.75 ? { nested: pick(pool), plain: 2 } : r < 0.85 ? reused : pick([1, 'x', null, [0, 1]])
      }
      passes.push({ uniforms: encodeJs(uniforms), audioTagged: prand() < 0.1 })
    }
    requirements.push({ passes })
  }
  scenarios.push({ name: 'requirements', midi: null, audio: null, evaluations: [], requirements })
}

// ---------------------------------------------------------------- math inputs

const mathInputs = []
{
  const mrand = mulberry32(2024)
  const halfPi = Math.PI / 2
  for (let i = 0; i < 60000; i++) mathInputs.push((mrand() * 2 - 1) * 10)
  for (let i = 0; i < 40000; i++) mathInputs.push((mrand() * 2 - 1) * 2000)
  for (let i = 0; i < 30000; i++) mathInputs.push((mrand() < 0.5 ? -1 : 1) * Math.pow(10, mrand() * 308))
  for (let i = 0; i < 20000; i++) mathInputs.push((mrand() < 0.5 ? -1 : 1) * Math.pow(10, -mrand() * 320))
  for (let k = -2000; k <= 2000; k++) {
    const base = k * halfPi
    mathInputs.push(base, base * (1 + Number.EPSILON), base * (1 - Number.EPSILON), base + 1e-9, base - 1e-9)
  }
  for (let k = -200; k <= 200; k++) mathInputs.push(k + 0.5, k - 0.5, k + 0.49999999999999994, k * 0.25)
  mathInputs.push(0, -0, NaN, Infinity, -Infinity, Number.MIN_VALUE, -Number.MIN_VALUE, Number.MAX_VALUE, 2 ** 52, 2 ** 53, 2 ** 52 + 0.5,
    1e10, 1e300, 0.7853981633974483, 2.356194490192345, 3.9269908169872414, 1.5707963267948966, 6.283185307179586)
}

// ---------------------------------------------------------------- compare

let pass = 0
let total = 0
const automation = await compareKind('automation', scenarios, { verbose: VERBOSE, label: 'automation' })
pass += automation.pass
total += automation.total
for (const [name, stats] of automation.perScenario) {
  if (stats.failures || VERBOSE) console.log(`[${stats.failures ? 'FAIL' : 'INFO'}] ${name}: ${stats.records - stats.failures}/${stats.records} results`)
}
const math = await compareKind('math', [{ name: 'math', inputs: mathInputs }], { verbose: VERBOSE, label: 'math' })
pass += math.pass
total += math.total
const evaluationCount = scenarios.reduce((n, s) => n + s.evaluations.length, 0)
const requirementCount = scenarios.reduce((n, s) => n + (s.requirements?.length ?? 0), 0)
console.log(`[INFO] ${compiled.length} compiled descriptors, ${REFERENCE_DESCRIPTORS.length} reference descriptors, ${evaluationCount} evaluations, ${requirementCount} requirement sets, ${mathInputs.length} x 3 math results (one record)`)
console.log(`AUTOMATION: ${pass}/${total}`)
process.exit(pass === total ? 0 : 1)
