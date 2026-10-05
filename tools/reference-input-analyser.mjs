#!/usr/bin/env node
// reference-input-analyser.mjs — Chromium's own AnalyserNode as the oracle for
// noisemaker-input's AudioAnalyzer.
//
// The reference host fills its audio state from Web Audio AnalyserNodes in the
// browser. This tool replays analyser scenarios through the AnalyserNode of the
// headless Chromium that the reference checkout's Playwright installs: an
// OfflineAudioContext plays the scenario's float32 signal through an
// AnalyserNode, suspends at render-quantum boundaries (suspend(q * 128 / rate)
// lands exactly on frame q * 128: Chromium truncates when * rate and rounds up
// to a quantum), applies attribute assignments and calls the getters there.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node tools/reference-input-analyser.mjs \
//       <scenarios.json> [--out results.json]
//
// Scenario: {name, channels, sampleRate, samples: [interleaved float32],
//   options: {fftSize, smoothingTimeConstant, minDecibels, maxDecibels},
//   events: [{quantum, ops: [{op: 'set', prop, value} | {op: 'read', getters}]}]}
// with strictly increasing event quanta below the signal length.
// Output: {oracle: {browser, platform}, results: [{name, events: [{quantum,
//   frame, results: [{op: 'set', prop, ok} | {op: 'read', data: {getter:
//   values}}]}]}]}; float arrays are IEEE bit patterns.
//
// The browser is launched once, runs every scenario back to back, and is
// closed before the tool exits.

import { readFileSync, writeFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)

function parseArgs (argv) {
  const opts = { input: null, out: null }
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') opts.out = argv[++i]
    else if (!opts.input) opts.input = argv[i]
    else throw new Error(`unexpected argument ${argv[i]}`)
  }
  if (!opts.input) throw new Error('usage: reference-input-analyser.mjs <scenarios.json> [--out results.json]')
  return opts
}

// Runs inside the page: one scenario through an OfflineAudioContext.
async function runScenarioInPage ({ channels, sampleRate, samples, options, events }) {
  const frames = samples.length / channels
  const ctx = new OfflineAudioContext({ numberOfChannels: channels, length: frames, sampleRate })
  const buffer = ctx.createBuffer(channels, frames, sampleRate)
  for (let ch = 0; ch < channels; ch++) {
    const data = buffer.getChannelData(ch)
    for (let f = 0; f < frames; f++) data[f] = samples[f * channels + ch]
  }
  const source = ctx.createBufferSource()
  source.buffer = buffer
  const analyser = new AnalyserNode(ctx, {
    fftSize: options.fftSize,
    minDecibels: options.minDecibels,
    maxDecibels: options.maxDecibels,
    smoothingTimeConstant: options.smoothingTimeConstant
  })
  source.connect(analyser)
  analyser.connect(ctx.destination)
  const bits = array => Array.from(new Uint32Array(array.buffer))
  const out = []
  for (const event of events) {
    ctx.suspend(event.quantum * 128 / sampleRate).then(() => {
      const results = []
      for (const op of event.ops) {
        if (op.op === 'set') {
          let ok = true
          try { analyser[op.prop] = op.value } catch { ok = false }
          results.push({ op: 'set', prop: op.prop, ok })
        } else {
          const data = {}
          for (const getter of op.getters) {
            if (getter === 'byteFrequency') {
              const array = new Uint8Array(analyser.frequencyBinCount)
              analyser.getByteFrequencyData(array)
              data[getter] = Array.from(array)
            } else if (getter === 'floatFrequency') {
              const array = new Float32Array(analyser.frequencyBinCount)
              analyser.getFloatFrequencyData(array)
              data[getter] = bits(array)
            } else if (getter === 'byteTimeDomain') {
              const array = new Uint8Array(analyser.fftSize)
              analyser.getByteTimeDomainData(array)
              data[getter] = Array.from(array)
            } else if (getter === 'floatTimeDomain') {
              const array = new Float32Array(analyser.fftSize)
              analyser.getFloatTimeDomainData(array)
              data[getter] = bits(array)
            } else {
              throw new Error(`unknown getter ${getter}`)
            }
          }
          results.push({ op: 'read', data })
        }
      }
      out.push({ quantum: event.quantum, frame: Math.round(ctx.currentTime * sampleRate), results })
      ctx.resume()
    })
  }
  source.start(0)
  await ctx.startRendering()
  return out
}

export async function runAnalyserOracle (scenarios) {
  const { chromium } = await import(pathToFileURL(join(REFERENCE_ROOT, 'node_modules', 'playwright', 'index.mjs')).href)
  const browser = await chromium.launch({ headless: true })
  try {
    const page = await browser.newPage()
    await page.setContent('<!doctype html><title>analyser oracle</title>')
    const results = []
    for (const scenario of scenarios) {
      const events = await page.evaluate(runScenarioInPage, scenario)
      results.push({ name: scenario.name, events })
    }
    return { oracle: { browser: `Chromium ${browser.version()}`, platform: `${process.platform}-${process.arch}` }, results }
  } finally {
    await browser.close()
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const opts = parseArgs(process.argv.slice(2))
  const scenarios = JSON.parse(readFileSync(opts.input, 'utf8'))
  const output = JSON.stringify(await runAnalyserOracle(scenarios))
  if (opts.out) writeFileSync(opts.out, output)
  else process.stdout.write(output + '\n')
}
