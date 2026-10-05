#!/usr/bin/env node
// generate-coverage.mjs — generate the effect × mode coverage corpus.
//
// For every effect of the catalog this writes parity/coverage/<ns>_<func>.dsl, a
// program that runs the effect with its defaults, and one variant per discrete
// branch of the effect: every value of every choice-valued parameter (compile-time
// defines included) and the flipped value of every boolean parameter:
//
//   parity/coverage/<ns>_<func>__<param>_<value>.dsl
//
// The base program is the effect's own `defaultProgram` when the definition has
// one; otherwise a program built for the effect's role in a chain (generator,
// 2D filter, mixer, particle stage, volume generator/processor/renderer, mesh,
// loop). Every generated program is compiled by the unmodified reference
// compileGraph, and a variant is kept only when the reference graph carries the
// requested value on one of the effect's passes (as a uniform or as a define), so
// each kept fixture provably exercises its branch. Dropped candidates are listed
// with the reason on stderr.
//
// Usage:
//   NM_REFERENCE_ROOT=/path/to/noisemaker node tools/generate-coverage.mjs [--check]
//
// --check regenerates in memory and exits 1 when parity/coverage differs.

import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const CATALOG = join(ROOT, 'crates', 'noisemaker-effects', 'catalog')
const OUT = join(ROOT, 'parity', 'coverage')

console.log = (...args) => process.stderr.write(args.map(String).join(' ') + '\n')
console.warn = () => {}
console.info = () => {}

function sortedDirs (dir) {
  return readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory() && d.name !== 'share').map(d => d.name).sort()
}

const IDENT = /^[A-Za-z_][A-Za-z0-9_]*$/

function isStarterEffect (def) {
  const passes = def.passes || []
  if (passes.length === 0) return true
  const pipelineInputs = new Set(['inputTex', 'inputTex3d', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7'])
  return !passes.some(p => p.inputs && Object.values(p.inputs).some(v => pipelineInputs.has(v)))
}

function passInputs (def) {
  return new Set((def.passes || []).flatMap(p => Object.values(p.inputs || {})))
}

// The chain the effect runs in when its definition has no defaultProgram.
function baseProgram (ns, func, def) {
  const inputs = passInputs(def)
  const globals = def.globals || {}
  const surfaceParams = Object.entries(globals).filter(([, g]) => g.type === 'surface')
  const needsSurface = surfaceParams.filter(([, g]) => g.default !== 'inputTex' && g.default !== 'none')
  if (ns === 'synth3d') {
    return { search: ['synth3d', 'render'], text: `${func}(@ARGS@).render3d().write(o0)` }
  }
  if (ns === 'filter3d') {
    return { search: ['synth3d', 'filter3d', 'render'], text: `noise3d().${func}(@ARGS@).render3d().write(o0)` }
  }
  if (ns === 'render' && inputs.has('inputTex3d')) {
    return { search: ['synth3d', 'render'], text: `noise3d().${func}(@ARGS@).write(o0)` }
  }
  if (ns === 'render' && (func === 'pointsRender' || func === 'pointsBillboardRender')) {
    return { search: ['synth', 'render'], text: `solid().pointsEmit(stateSize: 128).${func}(@ARGS@).write(o0)` }
  }
  if (ns === 'render' && func === 'pointsEmit') {
    return { search: ['synth', 'render'], text: `solid().pointsEmit(@ARGS@).pointsRender().write(o0)` }
  }
  if (ns === 'render' && (func === 'meshRender')) {
    return { search: ['render'], text: `meshLoader().meshRender(@ARGS@).write(o0)` }
  }
  if (ns === 'render' && func === 'loopBegin') {
    return { search: ['synth', 'filter', 'render'], text: `noise(seed: 1, scaleX: 50, scaleY: 50).loopBegin(@ARGS@).blur().loopEnd().write(o0)` }
  }
  if (ns === 'render' && func === 'loopEnd') {
    return { search: ['synth', 'filter', 'render'], text: `noise(seed: 1, scaleX: 50, scaleY: 50).loopBegin().blur().loopEnd(@ARGS@).write(o0)` }
  }
  if (ns === 'points') {
    return { search: ['synth', 'points', 'render'], text: `solid().pointsEmit(stateSize: 128).${func}(@ARGS@).pointsRender().write(o0)` }
  }
  if (needsSurface.length > 0) {
    const surfaceArgs = needsSurface.map(([k]) => `${k}: o0`).join(', ')
    const starter = isStarterEffect(def)
    const head = starter ? `${func}(${surfaceArgs}@SEP@@ARGS@)` : `gradient(seed: 1).${func}(${surfaceArgs}@SEP@@ARGS@)`
    return { search: ['synth', ns], text: `noise(seed: 1, colorMode: 1).write(o0)\n${head}.write(o1)`, render: 'o1' }
  }
  if (isStarterEffect(def)) {
    return { search: [ns], text: `${func}(@ARGS@).write(o0)` }
  }
  return { search: ['synth', ns], text: `noise(seed: 1, scaleX: 50, scaleY: 50).${func}(@ARGS@).write(o0)` }
}

function templateSource (tpl, args) {
  const argText = args.map(([k, v]) => `${k}: ${v}`).join(', ')
  const search = [...new Set(tpl.search)].join(', ')
  const body = tpl.text
    .replace('@SEP@', argText ? ', ' : '')
    .replace('@ARGS@', argText)
  return `search ${search}\n${body}\nrender(${tpl.render || 'o0'})\n`
}

// Insert or replace keyword arguments in the first call of `func` in `src`.
function withCallArgs (src, func, args) {
  if (!args.length) return src
  const re = new RegExp(`(^|[^A-Za-z0-9_])${func}\\s*\\(`, 'g')
  const m = re.exec(src)
  if (!m) return null
  const open = m.index + m[0].length - 1
  let depth = 0
  let close = -1
  let quote = null
  for (let i = open; i < src.length; i++) {
    const c = src[i]
    if (quote) {
      if (c === '\\') { i++; continue }
      if (c === quote) quote = null
      continue
    }
    if (c === '"' || c === "'") { quote = c; continue }
    if (c === '(') depth++
    else if (c === ')') { depth--; if (depth === 0) { close = i; break } }
  }
  if (close < 0) return null
  const inner = src.slice(open + 1, close)
  // Split the existing argument list at depth-0 commas.
  const items = []
  let cur = ''
  depth = 0
  quote = null
  for (let i = 0; i < inner.length; i++) {
    const c = inner[i]
    if (quote) {
      cur += c
      if (c === '\\') { cur += inner[++i]; continue }
      if (c === quote) quote = null
      continue
    }
    if (c === '"' || c === "'") { quote = c; cur += c; continue }
    if (c === '(' || c === '[' || c === '{') depth++
    if (c === ')' || c === ']' || c === '}') depth--
    if (c === ',' && depth === 0) { items.push(cur); cur = ''; continue }
    cur += c
  }
  if (cur.trim()) items.push(cur)
  const kept = items.map(s => s.trim()).filter(Boolean)
    .filter(item => !args.some(([k]) => new RegExp(`^${k}\\s*:`).test(item)))
  if (kept.some(item => !/^[A-Za-z_][A-Za-z0-9_]*\s*:/.test(item))) return null // positional args: skip
  const merged = [...kept, ...args.map(([k, v]) => `${k}: ${v}`)]
  return src.slice(0, open + 1) + merged.join(', ') + src.slice(close)
}

function choiceToken (name, value) {
  if (IDENT.test(name)) return name
  const sanitized = name.replace(/\s+(.)/g, (_, c) => c.toUpperCase()).replace(/\s+/g, '').replace(/[^a-zA-Z0-9_]/g, '')
  if (IDENT.test(sanitized)) return sanitized
  return String(value)
}

function fileToken (s) {
  return String(s).replace(/[^A-Za-z0-9]+/g, '_').replace(/^_+|_+$/g, '') || 'x'
}

function sameValue (a, b) {
  if (typeof a === 'boolean' || typeof b === 'boolean') return (a === true || a === 1) === (b === true || b === 1)
  return Number(a) === Number(b)
}

async function main () {
  if (!process.env.NM_REFERENCE_ROOT) {
    process.stderr.write('NM_REFERENCE_ROOT is not set\n')
    process.exit(2)
  }
  const check = process.argv.includes('--check')
  const { bootstrapReference } = await import(pathToFileURL(join(ROOT, 'tools', 'reference-oracle.mjs')).href)
  const ref = await bootstrapReference()

  const files = new Map()
  const dropped = []
  const uncovered = []
  for (const ns of sortedDirs(CATALOG)) {
    for (const name of sortedDirs(join(CATALOG, ns))) {
      const def = JSON.parse(readFileSync(join(CATALOG, ns, name, 'definition.json'), 'utf8'))
      const func = def.func || name
      const effectKey = `${ns}.${func}`
      const tpl = def.defaultProgram ? null : baseProgram(ns, func, def)
      const build = (args) => {
        if (tpl) return templateSource(tpl, args)
        return withCallArgs(def.defaultProgram.trim() + '\n', func, args)
      }
      // The effect's output must reach the presented surface: some pass at or after
      // the effect's last pass writes graph.renderSurface.
      const reachesOutput = (graph) => {
        const last = graph.passes.map((p, i) => p.effectKey === effectKey ? i : -1).reduce((a, b) => Math.max(a, b), -1)
        if (last < 0 || !graph.renderSurface) return false
        const target = `global_${graph.renderSurface}`
        return graph.passes.some((p, i) => i >= last && Object.values(p.outputs || {}).includes(target))
      }
      const compiles = (src) => {
        try {
          const graph = ref.compileGraph(src)
          if (!reachesOutput(graph)) return { error: `the effect does not reach the rendered surface ${graph.renderSurface}` }
          return { graph }
        } catch (err) {
          const msg = err?.message || (err?.diagnostics || err?.errors || []).map(d => d.message).join('; ') || JSON.stringify(err)
          return { error: msg }
        }
      }
      const base = build([])
      const baseResult = base && compiles(base)
      if (!base || baseResult.error) {
        uncovered.push(`${effectKey}: base program does not compile: ${base ? baseResult.error : 'no call site in defaultProgram'}`)
        continue
      }
      const stem = `${ns}_${func}`
      files.set(`${stem}.dsl`, base)

      const effectPasses = (graph) => graph.passes.filter(p => p.effectKey === effectKey)
      for (const [param, spec] of Object.entries(def.globals || {})) {
        const variants = []
        if (spec.choices && typeof spec.choices === 'object') {
          for (const [choiceName, value] of Object.entries(spec.choices)) {
            if (choiceName.endsWith(':')) continue
            if (value === spec.default || (typeof value === 'number' && Number(spec.default) === value && spec.type !== 'member')) continue
            variants.push({ label: choiceName, token: choiceToken(choiceName, value), value })
          }
        } else if (spec.type === 'boolean') {
          const flipped = !(spec.default === true)
          variants.push({ label: String(flipped), token: String(flipped), value: flipped })
        }
        for (const v of variants) {
          const src = build([[param, v.token]])
          const fname = `${stem}__${fileToken(param)}_${fileToken(v.label)}.dsl`
          if (!src) { dropped.push(`${fname}: cannot place the argument in defaultProgram`); continue }
          const result = compiles(src)
          if (result.error) { dropped.push(`${fname}: reference compile failed: ${result.error.slice(0, 200)}`); continue }
          const uniform = spec.uniform || param
          const hit = effectPasses(result.graph).some(p => {
            if (spec.define) return p.program.includes(`__${spec.define}_${v.value}`) || (p.uniforms && sameValue(p.uniforms[uniform], v.value))
            return p.uniforms && uniform in p.uniforms && sameValue(p.uniforms[uniform], v.value)
          })
          if (!hit) { dropped.push(`${fname}: the reference graph does not carry ${param}=${JSON.stringify(v.value)}`); continue }
          files.set(fname, src)
        }
      }
    }
  }

  const names = [...files.keys()].sort()
  if (check) {
    const existing = existsSync(OUT) ? readdirSync(OUT).filter(f => f.endsWith('.dsl')).sort() : []
    const stale = []
    if (existing.join('\n') !== names.join('\n')) stale.push('file set differs')
    for (const n of names) {
      const p = join(OUT, n)
      if (!existsSync(p) || readFileSync(p, 'utf8') !== files.get(n)) stale.push(n)
    }
    if (stale.length) {
      process.stderr.write(`[generate-coverage] parity/coverage is stale (${stale.length}): ${stale.slice(0, 10).join(', ')}\n`)
      process.exit(1)
    }
    process.stderr.write(`[generate-coverage] parity/coverage is current (${names.length} programs)\n`)
    return
  }
  rmSync(OUT, { recursive: true, force: true })
  mkdirSync(OUT, { recursive: true })
  for (const n of names) writeFileSync(join(OUT, n), files.get(n))
  for (const d of dropped) process.stderr.write(`[generate-coverage] dropped ${d}\n`)
  for (const u of uncovered) process.stderr.write(`[generate-coverage] UNCOVERED ${u}\n`)
  process.stderr.write(`[generate-coverage] ${names.length} programs, ${dropped.length} dropped variants, ${uncovered.length} uncovered effects\n`)
}

main().catch(err => {
  process.stderr.write(`[generate-coverage] FAILED: ${err?.stack || err}\n`)
  process.exit(1)
})
