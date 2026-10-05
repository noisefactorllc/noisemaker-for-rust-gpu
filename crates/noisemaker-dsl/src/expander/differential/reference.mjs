// Reference side of the expander differential tests (src/expander/tests.rs).
//
// Runs the unmodified reference `expand` (runtime/expander.js) and
// `compileGraph` (runtime/compiler.js, evaluated from its own source with its
// stage-1 `compile` returning the given validated result) on synthetic effects
// and validated plans.
//
// stdin: {effects: [{keys: [...], def: {...}}], cases: [{name, source, input, options}]}
// Strings equal to UNDEF become undefined members/elements (present keys).
// stdout: one JSON line per case: {name, expanded|expandedError, graph|graphError}
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
const ROOT = process.env.NM_REFERENCE_ROOT
const RT = join(ROOT, 'shaders', 'src', 'runtime')
const { registerEffect } = await import(pathToFileURL(join(RT, 'registry.js')).href)
const { expand } = await import(pathToFileURL(join(RT, 'expander.js')).href)
const { allocateResources } = await import(pathToFileURL(join(RT, 'resources.js')).href)
// compiler.js imports the whole pipeline; evaluate its own source with the
// stage functions injected so compileGraph runs on a given validated result.
const compilerSrc = readFileSync(join(RT, 'compiler.js'), 'utf8')
  .replace(/^import .*$/gm, '').replace(/^export /gm, '')
let currentValidated = null
const compilerMod = new Function('compile', 'expand', 'allocateResources', 'createPipeline',
  compilerSrc + '\nreturn { compileGraph, extractTextureSpecs, hashSource, formatError }')(
  () => currentValidated, expand, allocateResources, null)
console.warn = () => {}
const UNDEF = '\u0000undefined'
function revive (v) {
  if (Array.isArray(v)) { for (let i = 0; i < v.length; i++) v[i] = v[i] === UNDEF ? undefined : revive(v[i]); return v }
  if (v && typeof v === 'object') { for (const k of Object.keys(v)) v[k] = v[k] === UNDEF ? undefined : revive(v[k]); return v }
  return v
}
const mapsToObjects = (_k, v) => (v instanceof Map ? Object.fromEntries(v) : v instanceof Set ? [...v] : v)
const plain = v => { const t = JSON.stringify(v, mapsToObjects); return t === undefined ? null : JSON.parse(t) }
const thrown = e => (e instanceof Error ? { name: e.name, message: e.message } : plain(e))
const spec = revive(JSON.parse(readFileSync(0, 'utf8')))
for (const { keys, def } of spec.effects) for (const k of keys) registerEffect(k, def)
const out = []
for (const c of spec.cases) {
  const rec = { name: c.name }
  // Each stage gets its own structured clone: the reference may share objects.
  try { rec.expanded = plain(expand(structuredClone(c.input), structuredClone(c.options || {}))) } catch (e) { rec.expandedError = thrown(e) }
  try {
    currentValidated = structuredClone(c.input)
    const { compiledAt: _drop, ...graph } = compilerMod.compileGraph(c.source || '', structuredClone(c.options || {}))
    rec.graph = plain(graph)
  } catch (e) { rec.graphError = thrown(e) }
  out.push(JSON.stringify(rec))
}
process.stdout.write(out.join('\n') + '\n')
