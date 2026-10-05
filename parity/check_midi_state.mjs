#!/usr/bin/env node
// check_midi_state.mjs — gate noisemaker-input's MidiState against the
// reference MidiState (shaders/src/runtime/external-input.js), step by step.
//
// The same scripted operations run through the unmodified reference (in Node,
// Date.now() stubbed to each step's time, a fresh module instance per scenario
// so the note-on counter starts at zero) and through `nm-input-dump midi`.
// After EVERY step the gate compares, exactly:
//   - the operation's return value (handleMessage's parameter change with any
//     MPE reset channels, registerPort, getPortState, getZoneVoice, getPorts);
//   - the full state snapshot: every channel of the aggregate, of the unscoped
//     state and of every port state (key, velocity, gate, time, keys, cc, cc14,
//     pitch bend, pressure, poly pressure, NRPN/RPN values, held notes with
//     time, order and origin, the origin of every CC/CC14/bend/pressure/poly
//     pressure/parameter value, parameter selectors and family), the clock
//     count, MPE zones, the port registry, the inventory and the name index;
//   - the 128x16 RGBA note grid after updateNoteGrid() (IEEE bits).
//
// Scenarios: every MIDI sequence of the reference's test_midi.js and
// test_external_input.js; system messages (clock, start/continue/stop, active
// sensing, reset, SysEx, song position, MTC, tune request); a matrix of
// malformed and short messages (every status class, lengths 1-4, data bytes
// 0/64/127/128/200/255 in every position); the direct channel API (noteOn,
// noteOff, controlChange, resetControllers, clearNotes, reset, field writes)
// on the aggregate, unscoped and port states; registry-less states; port
// topology (rename, duplicates, disconnect and reconnect by message,
// inventories with duplicate, disconnected and unnamed entries); MPE
// configuration (both managers, counts 0-17, overlaps, non-MSB data entry,
// NRPN 6) and zone voice queries; and seeded fuzz runs mixing all of it.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_midi_state.mjs [--verbose]
// Env:
//   NM_INPUT_DUMP        candidate binary (default target/release/nm-input-dump)
//   NM_MIDI_FUZZ_STEPS   steps per fuzz scenario (default 2500)
// Exit 0 when every step matches; 1 otherwise.

import { compareKind, mulberry32, requireDump } from '../tools/reference-input-compare.mjs'

const VERBOSE = process.argv.includes('--verbose')
const FUZZ_STEPS = Number(process.env.NM_MIDI_FUZZ_STEPS || 2500)
requireDump()

const T0 = 1_700_000_000_000

// Step builders.
const msg = (data, port) => ({ op: 'message', data, ...(port ? { port } : {}) })
const at = (time, step) => ({ ...step, time })
const port = (id, name) => ({ id, name })

// Advances time by `dt` for each step of a list.
function timed (steps, start = T0, dt = 7) {
  let time = start
  return steps.map(step => ({ ...step, time: (time += dt) }))
}

const scenarios = []

// ---------------------------------------------------------------- test_external_input.js

{
  const left = port('left-id', 'Launch Control XL')
  const right = port('right-id', 'Launch Control XL')
  scenarios.push({
    name: 'reference-external-input',
    time: T0,
    steps: timed([
      msg([0x90, 60, 100]), msg([0x94, 72, 64]), msg([0x80, 60, 0]), msg([0x90, 60, 100]), msg([0x90, 60, 0]),
      msg([0x90, 60, 100]), msg([0x95, 72, 80]), { op: 'reset' },
      msg([0x90, 60, 40], left), msg([0x90, 72, 100], right),
      { op: 'portState', name: 'Launch Control XL', id: 'left-id' },
      { op: 'portState', name: 'Launch Control XL', id: 'right-id' },
      { op: 'portState', name: 'Launch Control XL' },
      { op: 'register', id: 'port-id', name: 'Renamed Controller' },
      { op: 'portState', name: 'Old Controller Name', id: 'port-id' },
      msg([0x90, 60, 127], port('port-id', 'Launch Control XL')),
      { op: 'disconnect', id: 'port-id' },
      { op: 'portState', name: 'Launch Control XL', id: 'port-id' },
      { op: 'ports' },
      msg([0xf8]), msg([0xf8]), msg([0x90, 60, 100]), msg([0x80, 60, 0]), msg([0x90, 61, 100]), msg([0x90, 61, 0]),
      { op: 'reset' }, msg([0x90, 60, 100]), msg([0x95, 72, 80])
    ])
  })
  scenarios.push({
    name: 'reference-port-inventory',
    time: T0,
    steps: timed([
      { op: 'register', id: 'twin-a', name: 'Twin' },
      { op: 'inventory', ports: [{ id: 'twin-a', name: 'Twin', connected: true }, { id: 'twin-b', name: 'Twin', connected: true }] },
      msg([0xb0, 74, 127], port('twin-a', 'Twin')),
      { op: 'portState', name: 'Twin' }, { op: 'portState', id: 'twin-a' }, { op: 'portState', id: 'twin-b' },
      { op: 'inventory', ports: [{ id: 'twin-a', name: 'Twin', connected: true }, { id: 'twin-b', name: 'Twin', connected: false }] },
      { op: 'portState', name: 'Twin' },
      { op: 'inventory', ports: [{ id: 'twin-b', name: 'Twin', connected: true }] },
      { op: 'portState', name: 'Twin' }
    ])
  })
}

// ---------------------------------------------------------------- test_midi.js

{
  const left = port('left', 'Controller')
  const right = port('right', 'Controller')
  scenarios.push({
    name: 'reference-cc',
    time: T0,
    steps: timed([
      msg([0xb1, 74, 127], left), msg([0xb0, 74, 12], left), msg([0xb1, 74, 8], right), msg([0x91, 60, 30], left),
      msg([0x81, 60, 0], left), msg([0xb1, 74, 64], left), msg([0xb1, 74, 0], left)
    ])
  })
  const l = port('left', 'Left')
  const r = port('right', 'Right')
  scenarios.push({
    name: 'reference-cc14',
    time: T0,
    steps: timed([
      msg([0xb0, 1, 64], l), msg([0xb0, 33, 127], r), msg([0xb1, 33, 127], l), msg([0xb0, 33, 1], l),
      msg([0xb0, 1, 127], l), msg([0xb0, 33, 127], l), msg([0x90, 60, 127], l), msg([0x80, 60, 0], l),
      { op: 'disconnect', id: 'left' }, msg([0xb0, 33, 127], l), { op: 'reset' }, msg([0xb0, 33, 127], l),
      msg([0xb0, 1, 127]), msg([0x90, 60, 127]), { op: 'reset' }
    ])
  })
  const expressive = port('expressive', 'Expressive')
  scenarios.push({
    name: 'reference-expression',
    time: T0,
    steps: timed([
      { op: 'register', id: 'expressive', name: 'Expressive' },
      msg([0xe1, 127, 127], expressive), msg([0xd1, 91], expressive), msg([0x91, 60, 100], expressive),
      msg([0xa1, 60, 63], expressive), msg([0xa1, 61, 127], expressive),
      msg([0xe1, 128, 127], expressive), msg([0xe1, 0], expressive), msg([0xd1], expressive), msg([0xd1, 128], expressive),
      msg([0xa1, 60, 255], expressive), msg([0x81, 60, 0], expressive)
    ])
  })
  const a = port('nrpn-a', 'Controller')
  const b = port('nrpn-b', 'Controller')
  const cc = (p, channel, controller, value) => msg([0xb0 | (channel - 1), controller, value], p)
  const select = (p, parameter, channel = 1) => [cc(p, channel, 99, parameter >> 7), cc(p, channel, 98, parameter & 127)]
  scenarios.push({
    name: 'reference-nrpn',
    time: T0,
    steps: timed([
      cc(a, 1, 99, 1), cc(a, 1, 6, 20), cc(a, 1, 98, 2), cc(a, 1, 6, 64), cc(a, 1, 38, 127),
      ...select(a, 131), cc(a, 1, 38, 5), ...select(a, 130), cc(a, 1, 6, 32),
      cc(a, 1, 101, 0), cc(a, 1, 100, 0), cc(a, 1, 6, 12),
      ...select(b, 130), cc(b, 1, 38, 7), ...select(a, 130, 2), cc(a, 2, 38, 9), ...select(a, 16383), cc(a, 1, 6, 127)
    ])
  })
  scenarios.push({
    name: 'reference-nrpn-saturation',
    time: T0,
    steps: timed([
      ...[[99, 0], [98, 3], [6, 127], [38, 127], [96, 92], [97, 0], [6, 0], [97, 127], [6, 20], [38, 3],
        [74, 99], [7, 81], [10, 61], [1, 91], [64, 127]].map(([c, v]) => msg([0xb0, c, v])),
      msg([0xe0, 127, 127]), msg([0xd0, 80]), msg([0x90, 60, 100]), msg([0xa0, 60, 80]), msg([0xb0, 121, 0]), msg([0xb0, 6, 90])
    ])
  })
  const ma = port('mpe-a', 'MPE')
  const mb = port('mpe-b', 'MPE')
  scenarios.push({
    name: 'reference-mpe-voices',
    time: T0,
    steps: timed([
      msg([0xe1, 0, 96], ma), msg([0x91, 60, 80], ma), msg([0xa1, 60, 40], ma),
      { op: 'zoneVoice', target: { port: 'mpe-a' }, zone: 0 },
      msg([0x91, 62, 100], ma), msg([0xa1, 62, 70], ma), { op: 'zoneVoice', target: { port: 'mpe-a' }, zone: 0 },
      msg([0x81, 62, 0], ma), { op: 'zoneVoice', target: { port: 'mpe-a' }, zone: 0 },
      msg([0x92, 67, 110], ma), msg([0xb2, 74, 100], ma), msg([0x92, 70, 120], mb), msg([0xb2, 74, 10], mb),
      { op: 'zoneVoice', zone: 0 }, msg([0x82, 70, 0], mb), { op: 'zoneVoice', zone: 0 },
      msg([0xb2, 64, 127], ma), msg([0x92, 67, 0], ma), msg([0xb1, 123, 0], ma), msg([0x91, 64, 100], ma),
      msg([0xb1, 120, 0], ma), { op: 'zoneVoice', zone: 0 }
    ])
  })
  const zones = port('zones', 'Zones')
  const mcm = (master, count, p = zones) => [[101, 0], [100, 6], [6, count]].map(([c, v]) => msg([0xb0 | (master - 1), c, v], p))
  const zq = (zone, members) => ({ op: 'zoneVoice', target: { port: 'zones' }, zone, ...(members === undefined ? {} : { members }) })
  scenarios.push({
    name: 'reference-mpe-detection',
    time: T0,
    steps: timed([
      ...mcm(1, 10), msg([0x9a, 70, 100], zones), zq(0), ...mcm(16, 8), zq(0), msg([0x97, 66, 100], zones), zq(0), zq(1), zq(0, 15),
      ...mcm(16, 0), msg([0x9e, 71, 100], zones), zq(1), zq(1, 1), ...mcm(1, 15), msg([0x9f, 72, 100], zones), zq(0)
    ])
  })
  const re = port('reset-expression', 'Reset')
  scenarios.push({
    name: 'reference-disconnect-reset',
    time: T0,
    steps: timed([
      msg([0xb1, 99, 0], re), msg([0xb1, 98, 1], re), msg([0xb1, 6, 100], re), msg([0xe1, 127, 127], re), msg([0xd1, 90], re),
      msg([0x91, 60, 100], re), { op: 'disconnect', id: 'reset-expression' }, msg([0xb1, 38, 127], re), msg([0x91, 60, 100], re),
      { op: 'reset' }, { op: 'zoneVoice', zone: 0 }
    ])
  })
  const ca = port('config-a', 'A')
  const cb = port('config-b', 'B')
  scenarios.push({
    name: 'reference-zone-config-origin',
    time: T0,
    steps: timed([
      msg([0xe1, 127, 127], ca), msg([0xd1, 100], ca), msg([0x91, 60, 100], ca), msg([0xe2, 0, 32], cb), msg([0x92, 62, 100], cb),
      ...[[101, 0], [100, 6], [6, 3]].map(([c, v]) => msg([0xb0, c, v], ca))
    ])
  })
  const oa = port('origin-a', 'A')
  const ob = port('origin-b', 'B')
  scenarios.push({
    name: 'reference-port-reset-unscoped',
    time: T0,
    steps: timed([
      msg([0xd1, 100]), msg([0xb1, 1, 100]), msg([0xb1, 121, 0], oa), msg([0x91, 60, 80], oa), msg([0x91, 60, 100], ob),
      msg([0xa1, 60, 99], ob), msg([0x81, 60, 0], oa)
    ])
  })
  const mr = port('manager-reset', 'Manager')
  scenarios.push({
    name: 'reference-mpe-activation',
    time: T0,
    steps: timed([
      msg([0xd0, 95], mr), msg([0xb0, 74, 99], mr), msg([0x90, 60, 100], mr),
      ...[[101, 0], [100, 6], [6, 2]].map(([c, v]) => msg([0xb0, c, v], mr)), msg([0xb0, 6, 4], mr)
    ])
  })
  scenarios.push({
    name: 'reference-keyless-note-off',
    registry: false,
    time: T0,
    steps: timed([
      { op: 'noteOn', target: 'root', channel: 2, key: 60, velocity: 100 },
      { op: 'noteOn', target: 'root', channel: 2, key: 64, velocity: 100 },
      { op: 'noteOff', target: 'root', channel: 2 },
      { op: 'zoneVoice', zone: 0 }
    ])
  })
}

// ---------------------------------------------------------------- system messages

{
  const p = port('sys', 'System')
  const steps = []
  for (const source of [undefined, p]) {
    steps.push(msg([0xf8], source), msg([0xfa], source), msg([0xf8, 0, 0], source), msg([0xfb], source), msg([0xfc], source),
      msg([0xfe], source), msg([0xff], source), msg([0xf0, 0x7e, 0x7f, 0x06, 0x01, 0xf7], source), msg([0xf2, 0x10, 0x20], source),
      msg([0xf1, 0x33], source), msg([0xf6], source), msg([0xf3, 5], source), msg([0x40, 60, 100], source), msg([0x7f, 1, 2], source),
      msg([], source), msg([0x90, 60, 100, 99], source), msg([0x80, 60], source), msg([0xc0, 5], source), msg([0xc5, 5, 0], source))
  }
  scenarios.push({ name: 'system-messages', time: T0, steps: timed(steps) })
}

// ---------------------------------------------------------------- malformed matrix

{
  const p = port('malformed', 'Malformed')
  const values = [0, 64, 127, 128, 200, 255]
  const steps = []
  for (const status of [0x80, 0x90, 0xa0, 0xb0, 0xc0, 0xd0, 0xe0, 0x85, 0x9f, 0xbf, 0xef]) {
    for (const length of [1, 2, 3, 4]) {
      for (const v1 of values) {
        for (const v2 of length >= 3 ? values : [undefined]) {
          const data = [status]
          if (length >= 2) data.push(v1)
          if (length >= 3) data.push(v2)
          if (length >= 4) data.push(1)
          steps.push(msg(data, (v1 + (v2 ?? 0)) % 3 === 0 ? p : undefined))
          if (length === 1) break
        }
        if (length === 1) break
      }
    }
  }
  scenarios.push({ name: 'malformed-matrix', time: T0, steps: timed(steps, T0, 3) })
}

// ---------------------------------------------------------------- direct channel API

{
  const p = port('api', 'Api')
  const steps = [msg([0x90, 60, 100], p), msg([0x91, 61, 50])]
  const targets = ['root', 'unscoped', { port: 'api' }]
  for (const target of targets) {
    steps.push(
      { op: 'noteOn', target, channel: 1, key: 62, velocity: 90 },
      { op: 'noteOn', target, channel: 1, key: 62, velocity: 91 },
      { op: 'noteOn', target, channel: 2, key: 200, velocity: 255 },
      { op: 'controlChange', target, channel: 1, controller: 101, value: 0 },
      { op: 'controlChange', target, channel: 1, controller: 100, value: 6 },
      { op: 'controlChange', target, channel: 1, controller: 6, value: 5 },
      { op: 'controlChange', target, channel: 1, controller: 38, value: 9 },
      { op: 'controlChange', target, channel: 1, controller: 96, value: 0 },
      { op: 'controlChange', target, channel: 1, controller: 200, value: 1 },
      { op: 'controlChange', target, channel: 1, controller: 7, value: 128 },
      { op: 'controlChange', target, channel: 3, controller: 33, value: 12 },
      { op: 'controlChange', target, channel: 3, controller: 1, value: 3 },
      { op: 'setField', target, channel: 4, field: 'key', value: 99 },
      { op: 'setField', target, channel: 4, field: 'gate', value: 1 },
      { op: 'setField', target, channel: 4, field: 'velocity', value: 77 },
      { op: 'setField', target, channel: 4, field: 'time', value: T0 - 1234.5 },
      { op: 'noteOff', target, channel: 1, key: 62 },
      { op: 'noteOff', target, channel: 2, key: 200 },
      { op: 'resetControllers', target, channel: 1 },
      { op: 'clearNotes', target, channel: 1 },
      { op: 'channelReset', target, channel: 3 },
      { op: 'noteOff', target, channel: 1 },
      { op: 'zoneVoice', target, zone: 0 },
      { op: 'zoneVoice', target, zone: 1, members: 15 }
    )
  }
  steps.push(msg([0x80, 60, 0], p), msg([0xb0, 121, 0], p), msg([0xb1, 123, 0]))
  scenarios.push({ name: 'direct-channel-api', time: T0, steps: timed(steps) })
}

// ---------------------------------------------------------------- registry-less states

{
  const p = port('direct', 'Direct')
  scenarios.push({
    name: 'no-registry',
    registry: false,
    time: T0,
    steps: timed([
      msg([0x90, 60, 100]), msg([0x90, 61, 90], p), msg([0xb0, 1, 64], p), msg([0xb0, 33, 3]), msg([0xe0, 1, 2], p),
      msg([0xd0, 50]), msg([0xa0, 60, 40], p), msg([0x80, 60, 0], p), msg([0x80, 61, 0]),
      ...[[101, 0], [100, 6], [6, 7]].map(([c, v]) => msg([0xb0, c, v])),
      msg([0x93, 64, 80]), { op: 'zoneVoice', zone: 0 }, { op: 'zoneVoice', zone: 0, members: 2 },
      { op: 'register', id: 'x', name: 'X' }, { op: 'disconnect', id: 'x' }, { op: 'portState', name: 'X' },
      { op: 'portState', id: 'x' }, { op: 'portState' }, { op: 'ports' },
      { op: 'inventory', ports: [{ id: 'x', name: 'X', connected: true }] }, { op: 'portState', name: 'X' },
      msg([0xf8], p), { op: 'reset' }
    ])
  })
}

// ---------------------------------------------------------------- port topology

{
  const a = port('a', 'Keys')
  const b = port('b', 'Keys')
  const c = port('c', 'Pads')
  const named = name => ({ op: 'portState', name })
  const byId = id => ({ op: 'portState', id })
  scenarios.push({
    name: 'port-topology',
    time: T0,
    steps: timed([
      { op: 'ports' }, { op: 'portState' }, named('Keys'), byId('a'),
      msg([0x90, 60, 100], a), named('Keys'), msg([0x90, 62, 100], b), named('Keys'), named('Pads'),
      { op: 'register', id: 'c', name: 'Pads' }, named('Pads'), { op: 'register', id: '', name: 'Empty' },
      { op: 'register', id: 'd', name: '' }, named(''), byId(''), { op: 'ports' },
      { op: 'register', id: 'b', name: 'Drums' }, named('Keys'), named('Drums'),
      { op: 'disconnect', id: 'a' }, named('Keys'), byId('a'), { op: 'disconnect', id: 'zzz' },
      msg([0xb0, 7, 10], a), named('Keys'), byId('a'), { op: 'ports' },
      { op: 'inventory', ports: [{ id: 'a', name: 'Keys', connected: true }, { id: 'b', name: 'Drums', connected: true }, { id: 'e', name: 'Keys', connected: true }] },
      named('Keys'), named('Drums'), named('Pads'), byId('c'),
      { op: 'inventory', ports: [{ id: 'a', name: 'Keys', connected: true }, { id: 'a', name: 'Keys', connected: true }, { id: '', name: 'Keys', connected: true }, { id: 'f', name: '', connected: true }, { id: 'g', name: 'Keys', connected: false }] },
      named('Keys'), named(''), { op: 'inventory', ports: [] }, named('Keys'),
      { op: 'disconnect', id: 'c' }, msg([0x90, 64, 1], c), named('Pads'),
      { op: 'register', id: 'a', name: 'Keys' }, { op: 'register', id: 'a', name: 'Keys' }, { op: 'reset' }, { op: 'ports' }
    ])
  })
}

// ---------------------------------------------------------------- MPE configuration

{
  const p = port('mpe', 'MPE')
  const q = port('mpe2', 'MPE2')
  const steps = []
  const mcm = (master, count, source = p) => [[101, 0], [100, 6], [6, count]].map(([c, v]) => msg([0xb0 | (master - 1), c, v], source))
  const notes = source => [2, 5, 9, 12, 15, 16, 1].map((ch, i) => msg([0x90 | (ch - 1), 40 + i, 60 + i], source))
  const queries = target => [0, 1, 2, 3].flatMap(zone => [undefined, 0, 1, 7, 15, 16].map(members => ({
    op: 'zoneVoice', target, zone, ...(members === undefined ? {} : { members })
  })))
  for (const [master, count] of [[1, 5], [16, 3], [1, 0], [16, 15], [1, 14], [1, 16], [16, 17], [2, 4], [1, 15], [16, 1]]) {
    steps.push(...mcm(master, count), ...notes(p), ...queries({ port: 'mpe' }))
  }
  steps.push(...mcm(1, 6, q), ...notes(q), ...queries('root'))
  // Non-MSB data entry and NRPN 6 never configure zones.
  steps.push(msg([0xb0, 101, 0], p), msg([0xb0, 100, 6], p), msg([0xb0, 38, 9], p), msg([0xb0, 96, 0], p), msg([0xb0, 97, 0], p))
  steps.push(msg([0xb0, 99, 0], p), msg([0xb0, 98, 6], p), msg([0xb0, 6, 4], p), ...queries({ port: 'mpe' }))
  steps.push(...mcm(1, 4), msg([0xb0, 6, 9], p), msg([0xbf, 6, 2], p), ...queries('unscoped'))
  steps.push(...[[101, 0], [100, 6], [6, 3]].map(([c, v]) => msg([0xb0 | 15, c, v])), ...queries('unscoped'), ...queries('root'))
  scenarios.push({ name: 'mpe-configuration', time: T0, steps: timed(steps, T0, 2) })
}

// ---------------------------------------------------------------- fuzz

function fuzz (seed, count) {
  const rand = mulberry32(seed)
  const pick = items => items[Math.floor(rand() * items.length)]
  const int = (lo, hi) => lo + Math.floor(rand() * (hi - lo + 1))
  const ports = [null, null, port('fa', 'Keys'), port('fb', 'Pads'), port('fc', 'Keys'), port('fd', 'Extra'), port('', 'Nameless')]
  const controllers = [0, 1, 6, 7, 11, 32, 33, 38, 64, 74, 96, 97, 98, 99, 100, 101, 120, 121, 123]
  const targets = ['root', 'unscoped', { port: 'fa' }, { port: 'fb' }]
  let time = T0 + seed * 1000
  const steps = []
  // Ports used as direct-API targets must exist.
  steps.push({ op: 'register', id: 'fa', name: 'Keys', time }, { op: 'register', id: 'fb', name: 'Pads', time })
  while (steps.length < count) {
    time += int(0, 25)
    const source = pick(ports)
    const ch = rand() < 0.6 ? pick([0, 1, 2, 15]) : int(0, 15)
    const r = rand()
    let step
    if (r < 0.2) step = msg([0x90 | ch, int(36, 84), int(0, 127)], source)
    else if (r < 0.3) step = msg([0x80 | ch, int(36, 84), int(0, 127)], source)
    else if (r < 0.5) step = msg([0xb0 | ch, rand() < 0.8 ? pick(controllers) : int(0, 127), int(0, 127)], source)
    else if (r < 0.56) {
      // A parameter transaction: MPE configuration, RPN or NRPN.
      const mcmMaster = rand() < 0.4
      const c = mcmMaster ? pick([0, 15]) : ch
      const seq = mcmMaster
        ? [[101, 0], [100, 6], [6, int(0, 17)]]
        : rand() < 0.5
          ? [[101, int(0, 2)], [100, int(0, 8)], [6, int(0, 127)], [38, int(0, 127)]]
          : [[99, int(0, 3)], [98, int(0, 3)], [6, int(0, 127)], [pick([38, 96, 97]), int(0, 127)]]
      for (const [cc, value] of seq) steps.push({ ...msg([0xb0 | c, cc, value], source), time })
      continue
    } else if (r < 0.62) step = msg([0xe0 | ch, int(0, 127), int(0, 127)], source)
    else if (r < 0.66) step = msg([0xd0 | ch, int(0, 127)], source)
    else if (r < 0.71) step = msg([0xa0 | ch, int(36, 84), int(0, 127)], source)
    else if (r < 0.74) step = msg([0xf8], source)
    else if (r < 0.77) step = msg(pick([[0x90 | ch], [0x90 | ch, 130, 5], [0xb0 | ch, 7, 140], [0xc0 | ch, 3], [0xf0, 1, 0xf7], [0xe0 | ch, 3], [0xa0 | ch, 200, 1]]), source)
    else if (r < 0.79) step = { op: 'disconnect', id: pick(['fa', 'fb', 'fc', 'fd']) }
    else if (r < 0.81) {
      const p = pick(ports.filter(Boolean))
      step = { op: 'register', id: p.id, name: rand() < 0.2 ? 'Renamed' : p.name }
    } else if (r < 0.82) {
      step = { op: 'inventory', ports: ports.filter(Boolean).map(p => ({ ...p, connected: rand() < 0.8 })) }
    } else if (r < 0.825) step = { op: 'reset' }
    else if (r < 0.86) {
      const target = pick(targets)
      const channel = int(1, 16)
      const op = pick(['noteOn', 'noteOn', 'noteOff', 'noteOff', 'controlChange', 'resetControllers', 'clearNotes', 'channelReset', 'setField'])
      step = { op, target, channel }
      if (op === 'noteOn') Object.assign(step, { key: int(36, 84), velocity: int(1, 127) })
      if (op === 'noteOff' && rand() < 0.8) step.key = int(36, 84)
      if (op === 'controlChange') Object.assign(step, { controller: pick(controllers), value: int(0, 127) })
      if (op === 'setField') Object.assign(step, pick([{ field: 'gate', value: int(0, 1) }, { field: 'key', value: int(0, 127) }, { field: 'velocity', value: int(0, 127) }, { field: 'time', value: time - int(0, 3000) }]))
    } else if (r < 0.94) step = { op: 'zoneVoice', target: pick([...targets, 'root']), zone: int(0, 1), ...(rand() < 0.4 ? { members: int(0, 16) } : {}) }
    else if (r < 0.98) step = { op: 'portState', ...(rand() < 0.7 ? { name: pick(['Keys', 'Pads', 'Extra', 'Renamed', 'Nameless', '']) } : {}), ...(rand() < 0.5 ? { id: pick(['fa', 'fb', 'fc', 'fd', 'zz', '']) } : {}) }
    else step = { op: 'ports' }
    steps.push({ ...step, time })
  }
  return { name: `fuzz-${seed}`, time: T0, steps: steps.slice(0, count) }
}

for (const seed of [1, 2, 3, 4]) scenarios.push(fuzz(seed, FUZZ_STEPS))

// ---------------------------------------------------------------- compare

const { pass, total, perScenario } = await compareKind('midi', scenarios, { verbose: VERBOSE, label: 'midi' })
for (const [name, stats] of perScenario) {
  if (stats.failures || VERBOSE) console.log(`[${stats.failures ? 'FAIL' : 'INFO'}] ${name}: ${stats.records - stats.failures}/${stats.records} steps`)
}
console.log(`[INFO] ${scenarios.length} scenarios, ${total} steps compared (state, return value and note grid after each)`)
console.log(`MIDI_STATE: ${pass}/${total}`)
process.exit(pass === total ? 0 : 1)
