#!/usr/bin/env node
// reference-oracle.mjs — run the unmodified reference frontend and dump each stage.
//
// The reference engine at $NM_REFERENCE_ROOT is bootstrapped the way its host,
// CanvasRenderer, prepares it (manifest starter ops, standard enums, then every
// effect of the catalog in namespace/effect order: definition, WGSL shaders,
// registerEffectWithRuntime, choice enums, starter op). That is the same sequence
// noisemaker_dsl::Registry::with_catalog performs, so `nm-render dump` and this
// tool describe the same engine state.
//
// Usage:
//   node tools/reference-oracle.mjs <stage> --out results.jsonl <program.dsl>...
//   stage: tokens | ast | validated | expanded | graph
//
// Each line of the output is {"program": <file stem>, "stage": <stage>,
// "result": <stage output>} or {"program", "stage", "error": <thrown value>}.
// Stage outputs are serialized with JSON.stringify semantics; Maps become
// objects; graph program specs drop their shader source texts (glsl, wgsl, vertex,
// fragment), which the catalog freshness gate covers instead; `compiledAt` is
// dropped.

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs'
import { basename, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const SHADERS = join(REFERENCE_ROOT, 'shaders')
const EFFECTS = join(SHADERS, 'effects')

// The reference logs warnings with console.warn/console.log; keep stdout for
// callers and route every reference message to stderr.
console.log = (...args) => process.stderr.write(args.map(String).join(' ') + '\n')
console.warn = console.log
console.info = console.log

const STAGES = ['tokens', 'ast', 'validated', 'expanded', 'graph']
const SHADER_SOURCE_KEYS = new Set(['glsl', 'wgsl', 'vertex', 'fragment'])

function sortedDirs (dir) {
  return readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => d.name).sort()
}

export async function bootstrapReference () {
  const mod = await import(pathToFileURL(join(SHADERS, 'src', 'index.js')).href)
  const { registerEffect, registerOp, registerStarterOps, mergeIntoEnums, sanitizeEnumName, isStarterEffect } = mod
  // The host merges the standard enums of lang/std_enums.js. (index.js also
  // exports a `stdEnums`, but that one is the merged enum tree of lang/enums.js,
  // which is still empty here.)
  const { stdEnums } = await import(pathToFileURL(join(SHADERS, 'src', 'lang', 'std_enums.js')).href)
  const { registerParamAliases } = await import(pathToFileURL(join(SHADERS, 'src', 'lang', 'paramAliases.js')).href)
  const { registerEffectAlias } = await import(pathToFileURL(join(SHADERS, 'src', 'lang', 'effectAliases.js')).href)
  const manifest = JSON.parse(readFileSync(join(EFFECTS, 'manifest.json'), 'utf8'))

  // loadManifest(): starter ops from the manifest, then the standard enums.
  const starterNames = []
  for (const [effectId, entry] of Object.entries(manifest)) {
    if (entry.starter) {
      const parts = effectId.split('/')
      if (parts.length === 2) {
        starterNames.push(parts[1])
        starterNames.push(`${parts[0]}.${parts[1]}`)
      }
    }
  }
  registerStarterOps(starterNames)
  await mergeIntoEnums(stdEnums)

  for (const namespace of sortedDirs(EFFECTS)) {
    for (const effectName of sortedDirs(join(EFFECTS, namespace))) {
      const defPath = join(EFFECTS, namespace, effectName, 'definition.js')
      if (!existsSync(defPath)) continue
      const exported = (await import(pathToFileURL(defPath).href)).default
      const instance = typeof exported === 'function' ? new exported() : exported

      // loadEffectShaders(): one bucket per pass program; WGSL when the manifest lists it.
      const effectManifest = manifest[`${namespace}/${effectName}`]
      if (instance.passes && effectManifest) {
        if (!instance.shaders) instance.shaders = {}
        for (const pass of instance.passes) {
          if (!pass.program) continue
          const prog = pass.program
          const bucket = instance.shaders[prog] ?? (instance.shaders[prog] = {})
          if (effectManifest.wgsl?.[prog]) {
            const wgslPath = join(EFFECTS, namespace, effectName, 'wgsl', `${prog}.wgsl`)
            if (existsSync(wgslPath)) bucket.wgsl = readFileSync(wgslPath, 'utf8')
          }
        }
      }

      // registerEffectWithRuntime(effect)
      registerEffect(instance.func, instance)
      registerEffect(`${namespace}.${instance.func}`, instance)
      registerEffect(`${namespace}/${effectName}`, instance)
      registerEffect(`${namespace}.${effectName}`, instance)
      if (instance.func) {
        const choicesToRegister = {}
        const args = Object.entries(instance.globals || {}).map(([key, spec]) => {
          let enumPath = spec.enum || spec.enumPath
          if (spec.choices && !enumPath) {
            enumPath = `${namespace}.${instance.func}.${key}`
            if (!choicesToRegister[namespace]) choicesToRegister[namespace] = {}
            if (!choicesToRegister[namespace][instance.func]) choicesToRegister[namespace][instance.func] = {}
            choicesToRegister[namespace][instance.func][key] = {}
            for (const [name, val] of Object.entries(spec.choices)) {
              if (name.endsWith(':')) continue
              choicesToRegister[namespace][instance.func][key][name] = { type: 'Number', value: val }
              const sanitized = sanitizeEnumName(name)
              if (sanitized && sanitized !== name) {
                choicesToRegister[namespace][instance.func][key][sanitized] = { type: 'Number', value: val }
              }
            }
          }
          return {
            name: key,
            type: spec.type === 'vec4' ? 'color' : spec.type,
            default: spec.default,
            enum: enumPath,
            enumPath,
            min: spec.min,
            max: spec.max,
            uniform: spec.uniform,
            choices: spec.choices
          }
        })
        registerOp(`${namespace}.${instance.func}`, { name: instance.func, args })
        if (instance.paramAliases) registerParamAliases(`${namespace}.${instance.func}`, instance.paramAliases)
        if (instance.hidden && instance.deprecatedBy) registerEffectAlias(`${namespace}.${instance.func}`, instance.deprecatedBy)
        if (Object.keys(choicesToRegister).length > 0) await mergeIntoEnums(choicesToRegister)
      }

      // registerStarterOpForEffect(effect)
      const effect = { namespace, name: effectName, instance }
      if (isStarterEffect(effect)) {
        const func = instance.func || effectName
        registerStarterOps(func ? [func, `${namespace}.${func}`] : [])
      }
    }
  }
  return mod
}

function mapsToObjects (_key, value) {
  if (value instanceof Map) return Object.fromEntries(value)
  if (value instanceof Set) return [...value]
  return value
}

function plain (value) {
  const text = JSON.stringify(value, mapsToObjects)
  return text === undefined ? null : JSON.parse(text)
}

function thrown (err) {
  if (err instanceof Error) return { name: err.name, message: err.message }
  return plain(err)
}

export function normalizeGraph (graph) {
  const programs = {}
  for (const [id, spec] of Object.entries(graph.programs || {})) {
    const out = {}
    for (const [k, v] of Object.entries(spec)) if (!SHADER_SOURCE_KEYS.has(k)) out[k] = v
    programs[id] = out
  }
  const { compiledAt: _compiledAt, ...rest } = graph
  return plain({ ...rest, programs })
}

export function runStage (mod, stage, src) {
  const tokens = mod.lex(src)
  if (stage === 'tokens') return plain(tokens)
  const ast = mod.parse(tokens)
  if (stage === 'ast') return plain(ast)
  const validated = mod.validate(ast)
  if (stage === 'validated') return plain(validated)
  if (stage === 'expanded') return plain(mod.expand(validated))
  if (stage === 'graph') return normalizeGraph(mod.compileGraph(src))
  throw new Error(`unknown stage ${stage}`)
}

async function main () {
  const argv = process.argv.slice(2)
  const stage = argv.shift()
  if (!STAGES.includes(stage)) {
    console.error(`usage: node tools/reference-oracle.mjs <${STAGES.join('|')}> --out results.jsonl <program.dsl>...`)
    process.exit(2)
  }
  let out = null
  const files = []
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--out') out = argv[++i]
    else files.push(argv[i])
  }
  const mod = await bootstrapReference()
  const lines = []
  for (const file of files) {
    const program = basename(file).replace(/\.dsl$/, '')
    const src = readFileSync(file, 'utf8')
    let record
    try {
      record = { program, stage, result: runStage(mod, stage, src) }
    } catch (err) {
      record = { program, stage, error: thrown(err) }
    }
    lines.push(JSON.stringify(record))
  }
  const text = lines.join('\n') + (lines.length ? '\n' : '')
  if (out) writeFileSync(out, text)
  else process.stdout.write(text)
}

if (process.argv[1] && basename(process.argv[1]) === 'reference-oracle.mjs') {
  main().catch(err => {
    console.error(`[reference-oracle] FAILED: ${err?.stack || err}`)
    process.exit(1)
  })
}
