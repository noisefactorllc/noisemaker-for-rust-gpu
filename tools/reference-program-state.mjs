#!/usr/bin/env node
// reference-program-state.mjs — run the unmodified reference ProgramState
// (demo/shaders/lib/program-state.js, with its Emitter and
// extractEffectsFromDsl) over scripted scenarios, and record everything it
// does after every operation.
//
// The reference engine at $NM_REFERENCE_ROOT is bootstrapped like
// tools/reference-oracle.mjs does it (bootstrapReference), the state
// noisemaker_dsl::Registry::with_catalog() reproduces. Each scenario gets a
// fresh ProgramState over a mock renderer modeled on the reference tests:
// Object.create(CanvasRenderer.prototype) with `_currentDsl`, `_enums` and a
// `_pipeline` whose `graph` is the reference compileGraph() output and whose
// members (broadcastChainScopedParam, checkAsyncRegen, recreateTextures,
// collectDefaultUniforms, setUniform; each optional) only record their
// calls. The Rust driver (nm-program-state) builds the same mock
// (noisemaker_dsl::program_state::MockHost) over its own compile_graph().
//
// A suite (parity/program-state/*.json) is {scenarios: [...]}; a scenario is
// {name, host, ops}, {generate: "fixtures", dirs: [...]} for one fixture
// scenario per .dsl file, or {generate: "conversions", enums} for
// convertParameterForUniform over every catalog parameter spec (and
// resolveEnumValue over their enum paths). `host` configures the mock
// renderer: {renderer: attached?, enums: host|std|empty|none, convert:
// canvas|passthrough|absent, methods: [pipeline members present]}.
//
// Ops are concrete ProgramState calls (fromDsl, setValue, batch, insertStep,
// serialize, on/off/once with listeners described as action lists, ...),
// host operations (host.load compiles a program into the mock pipeline and
// sets currentDsl; host.loadGraph only replaces the graph; host.setDsl,
// host.setMethods, host.setEnums, host.clearPipeline, setRenderer, newState),
// and macros this tool expands with the reference's own values: fixture (the
// per-fixture sequence), edits, initControls (demo-ui's control
// initialization), reload (toDsl, recompile, reload), serializeRoundTrip,
// insertAndLoad, deleteAndLoad, effectProgram (the demo's one-effect
// program) and defineWalk (test-define-variants). An op's `"save": name`
// keeps its result for later `{"$var": name}` fields; `stepOf: effectKey`
// names the step of an effect.
//
// `run` writes:
//   --resolved  a {"suite"} header line, then one {name, host, ops} line per
//               scenario with every macro expanded into the concrete ops that
//               ran (values in the tagged encoding below) — the input of
//               `nm-program-state run`;
//   --expected  one record per concrete op:
//               {s, i, op, r | x, ev, log, calls, u, st}
//               r: the op's result, x: what it threw; ev: the events listeners
//               saw; log: console.warn/error calls; calls: mock pipeline
//               calls; u: passes whose uniforms changed ({p, u}); st: the
//               ProgramState's internal state.
// Unchanged parts repeat as {"$same": true} (per scenario: snapshot parts,
// recorder event payloads, query results, checkAsyncRegen step values per
// node, recreateTextures uniforms), which both tools apply identically.
//
// Values use a tagged JSON encoding: {"$js": "undefined" | "NaN" |
// "Infinity" | "-Infinity" | "-0" | "function"}, {"$js": "error", name,
// message}, {"$js": "object", members} (an object with an own "$js" key).
//
// Usage:
//   node tools/reference-program-state.mjs run <suite.json>... --resolved R --expected E [--only substring]

import { closeSync, openSync, readFileSync, readdirSync, writeSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { bootstrapReference } from './reference-oracle.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const SHADERS = join(REFERENCE_ROOT, 'shaders')
const TAG = '$js'

const EVENTS = ['change', 'stepchange', 'structurechange', 'reset', 'load', 'recompileNeeded', 'mediachange', 'textchange']
const ALL_METHODS = ['broadcastChainScopedParam', 'checkAsyncRegen', 'recreateTextures', 'collectDefaultUniforms', 'setUniform']
const DEFAULT_HOST = { renderer: true, enums: 'host', convert: 'canvas', methods: ALL_METHODS }

// ---------------------------------------------------------------------------
// Console capture
// ---------------------------------------------------------------------------

let captured = null
function stderr (...args) {
  process.stderr.write(args.map(a => (a instanceof Error ? `${a.name}: ${a.message}` : String(a))).join(' ') + '\n')
}
console.log = stderr
console.info = stderr
console.warn = (...args) => (captured ? captured.push({ level: 'warn', args: args.map(a => enc(a)) }) : stderr(...args))
console.error = (...args) => (captured ? captured.push({ level: 'error', args: args.map(a => enc(a)) }) : stderr(...args))

// ---------------------------------------------------------------------------
// Tagged encoding
// ---------------------------------------------------------------------------

function defineOwn (obj, key, value) {
  Object.defineProperty(obj, key, { value, enumerable: true, writable: true, configurable: true })
}

export function enc (v, depth = 0) {
  if (depth > 400) throw new Error('enc: value nests too deeply')
  if (v === undefined) return { [TAG]: 'undefined' }
  if (v === null || typeof v === 'boolean' || typeof v === 'string') return v
  if (typeof v === 'number') {
    if (Number.isNaN(v)) return { [TAG]: 'NaN' }
    if (v === Infinity) return { [TAG]: 'Infinity' }
    if (v === -Infinity) return { [TAG]: '-Infinity' }
    if (Object.is(v, -0)) return { [TAG]: '-0' }
    return v
  }
  if (typeof v === 'function') return { [TAG]: 'function' }
  if (typeof v !== 'object') throw new Error(`enc: unsupported ${typeof v}`)
  if (v instanceof Error) return { [TAG]: 'error', name: v.name, message: v.message }
  if (ArrayBuffer.isView(v)) return { [TAG]: v.constructor.name, values: Array.from(v, x => enc(x, depth + 1)) }
  if (v instanceof Map) return { [TAG]: 'Map', entries: [...v].map(([k, x]) => [enc(k, depth + 1), enc(x, depth + 1)]) }
  if (v instanceof Set) return { [TAG]: 'Set', values: [...v].map(x => enc(x, depth + 1)) }
  if (Array.isArray(v)) return Array.from(v, x => enc(x, depth + 1))
  const out = {}
  for (const key of Object.keys(v)) defineOwn(out, key, enc(v[key], depth + 1))
  if (Object.hasOwn(out, TAG)) return { [TAG]: 'object', members: out }
  return out
}

const ERROR_TYPES = { Error, TypeError, RangeError, SyntaxError, ReferenceError }

export function dec (v) {
  if (v === null || typeof v !== 'object') return v
  if (Array.isArray(v)) return v.map(dec)
  if (Object.hasOwn(v, TAG)) {
    switch (v[TAG]) {
      case 'undefined': return undefined
      case 'NaN': return NaN
      case 'Infinity': return Infinity
      case '-Infinity': return -Infinity
      case '-0': return -0
      case 'function': return function () {}
      case 'error': return makeError(v)
      case 'object': return decMembers(v.members)
      default: throw new Error(`dec: unknown tag ${v[TAG]}`)
    }
  }
  return decMembers(v)
}

function decMembers (obj) {
  const out = {}
  for (const key of Object.keys(obj)) defineOwn(out, key, dec(obj[key]))
  return out
}

function makeError (spec) {
  const Ctor = ERROR_TYPES[spec.name] || Error
  const err = new Ctor(spec.message)
  if (!ERROR_TYPES[spec.name]) err.name = spec.name
  return err
}

function thrownRec (err) {
  if (err instanceof Error) return { name: err.name, message: err.message }
  return { thrown: enc(err) }
}

const SAME = { $same: true }

// ---------------------------------------------------------------------------
// The mock renderer
// ---------------------------------------------------------------------------

let M = null // reference modules

function enumsFor (name) {
  switch (name) {
    case 'host': return M.hostEnums
    case 'std': return M.stdEnums
    case 'empty': return {}
    case 'none': return undefined
    default: throw new Error(`unknown enums ${name}`)
  }
}

function makeRenderer (cfg) {
  const r = Object.create(M.CanvasRenderer.prototype)
  r._currentDsl = ''
  r._pipeline = null
  r._enums = enumsFor(cfg.enums)
  if (cfg.convert === 'passthrough') r.convertParameterForUniform = value => value
  else if (cfg.convert === 'absent') r.convertParameterForUniform = undefined
  else if (cfg.convert !== 'canvas') throw new Error(`unknown convert ${cfg.convert}`)
  return r
}

function makePipeline (ctx, graph) {
  const p = { graph }
  const has = name => ctx.methods.includes(name)
  if (has('broadcastChainScopedParam')) {
    p.broadcastChainScopedParam = (pass, uniformName, scopedName) => {
      ctx.calls.push({ fn: 'broadcastChainScopedParam', pass: graph.passes.indexOf(pass), uniformName: enc(uniformName), scopedName: enc(scopedName) })
    }
  }
  if (has('checkAsyncRegen')) {
    p.checkAsyncRegen = (nodeId, effectKey, stepValues) => {
      ctx.calls.push({ fn: 'checkAsyncRegen', nodeId: enc(nodeId), effectKey: enc(effectKey), stepValues: enc(stepValues) })
    }
  }
  if (has('recreateTextures')) {
    p.recreateTextures = uniforms => { ctx.calls.push({ fn: 'recreateTextures', uniforms: enc(uniforms) }) }
  }
  if (has('collectDefaultUniforms')) {
    p.collectDefaultUniforms = () => {
      ctx.calls.push({ fn: 'collectDefaultUniforms' })
      const uniforms = {}
      for (const pass of graph.passes) if (pass.uniforms) Object.assign(uniforms, pass.uniforms)
      return uniforms
    }
  }
  if (has('setUniform')) {
    p.setUniform = (name, value) => { ctx.calls.push({ fn: 'setUniform', name: enc(name), value: enc(value) }) }
  }
  return p
}

function passSummary (pass) {
  return ({
    id: pass.id,
    nodeId: pass.nodeId,
    effectKey: pass.effectKey,
    stepIndex: pass.stepIndex,
    uniforms: pass.uniforms,
    scopedParams: pass.scopedParams,
    inheritsVolumeSize: pass.inheritsVolumeSize,
    uniformAliases: pass.uniformAliases
  })
}

// ---------------------------------------------------------------------------
// Scenario context
// ---------------------------------------------------------------------------

class Context {
  constructor (name, host) {
    this.name = name
    this.index = 0
    this.out = null
    this.resolved = []
    this.last = new Map() // $same compression
    this.vars = new Map() // saved results
    this.passJson = []
    this.newState(host)
  }

  newState (host) {
    const cfg = { ...DEFAULT_HOST, ...(host || {}) }
    this.methods = cfg.methods
    this.renderer = makeRenderer(cfg)
    this.calls = []
    this.events = []
    this.passJson = []
    this.listeners = new Map()
    this.state = new M.ProgramState({ renderer: cfg.renderer ? this.renderer : null })
    for (const event of EVENTS) {
      this.state.on(event, data => {
        this.events.push({ l: '*', e: event, d: this.same(`ev.${event}`, enc(data)) })
      })
    }
  }

  same (key, encoded) {
    const json = JSON.stringify(encoded)
    if (this.last.get(key) === json) return SAME
    this.last.set(key, json)
    return encoded
  }

  compressCalls (calls) {
    return calls.map(c => {
      if (c.fn === 'checkAsyncRegen') {
        return { ...c, stepValues: this.same(`call.checkAsyncRegen.${JSON.stringify(c.nodeId)}`, c.stepValues) }
      }
      if (c.fn === 'recreateTextures') return { ...c, uniforms: this.same('call.recreateTextures', c.uniforms) }
      return c
    })
  }

  uniformDeltas () {
    const graph = this.renderer._pipeline?.graph
    if (!graph || !Array.isArray(graph.passes)) return []
    const out = []
    graph.passes.forEach((pass, i) => {
      const u = enc(pass.uniforms)
      const json = JSON.stringify(u)
      if (this.passJson[i] !== json) {
        out.push({ p: i, u })
        this.passJson[i] = json
      }
    })
    return out
  }

  snapshot () {
    const s = this.state
    const steps = [...s._stepStates].map(([key, st]) => [key, {
      effectKey: st.effectKey,
      def: defSummary(st.effectDef),
      stepIndex: st.stepIndex,
      values: st.values
    }])
    const routing = {
      writeTargets: [...s._writeTargetOverrides],
      writeStepTargets: [...s._writeStepTargetOverrides],
      readSources: [...s._readSourceOverrides],
      read3dVol: [...s._read3dVolOverrides],
      read3dGeo: [...s._read3dGeoOverrides],
      write3dVol: [...s._write3dVolOverrides],
      write3dGeo: [...s._write3dGeoOverrides],
      renderTarget: s._renderTargetOverride
    }
    return {
      steps: this.same('st.steps', enc(steps)),
      structure: this.same('st.structure', enc(s._structure)),
      compiled: this.same('st.compiled', enc(s._compiled)),
      routing: this.same('st.routing', enc(routing)),
      media: this.same('st.media', enc([...s._mediaInputs])),
      text: this.same('st.text', enc([...s._textInputs])),
      batch: enc({ depth: s._batchDepth, changes: s._batchedChanges, recompilePending: s._recompilePending }),
      dsl: s._renderer ? enc(s._renderer.currentDsl) : null
    }
  }

  // Run one concrete op, record it, and return its outcome.
  exec (op) {
    this.resolved.push(op)
    captured = []
    let outcome
    try {
      outcome = { value: runOp(this, op) }
    } catch (error) {
      outcome = { error }
    }
    const log = captured
    captured = null
    const rec = { s: this.name, i: this.index++, op: op.op }
    if ('error' in outcome) rec.x = thrownRec(outcome.error)
    else rec.r = QUERY_OPS.has(op.op) ? this.same(`res.${op.op}`, encResult(op, outcome.value)) : encResult(op, outcome.value)
    rec.ev = this.events.splice(0)
    rec.log = log
    rec.calls = this.compressCalls(this.calls.splice(0))
    rec.u = this.uniformDeltas()
    rec.st = this.snapshot()
    writeSync(this.out, JSON.stringify(rec) + '\n')
    return { ...outcome, record: rec }
  }
}

const QUERY_OPS = new Set(['getStructure', 'getCompiled', 'getAllStepValues', 'serialize'])

function defSummary (def) {
  return def ? { func: def.func, namespace: def.namespace } : null
}

function encResult (op, value) {
  switch (op.op) {
    case 'batch': return value // already encoded nested results
    case 'getEffectDef': return enc(defSummary(value))
    case 'getAllMediaInputs':
    case 'getAllTextInputs': return enc([...value])
    default: return enc(value)
  }
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

function makeListener (ctx, id, actions) {
  return data => {
    for (const action of actions) {
      switch (action.do) {
        case 'record': ctx.events.push({ l: id, d: enc(data) }); break
        case 'throw': throw makeError(action)
        case 'on': ctx.state.on(action.event, ctx.listeners.get(action.id)); break
        case 'off': ctx.state.off(action.event, ctx.listeners.get(action.id)); break
        case 'once': ctx.state.once(action.event, ctx.listeners.get(action.id)); break
        case 'removeAllListeners': ctx.state.removeAllListeners(action.event); break
        case 'op': {
          try {
            const value = runOp(ctx, action.op)
            ctx.events.push({ l: id, n: action.op.op, r: encResult(action.op, value) })
          } catch (error) {
            ctx.events.push({ l: id, n: action.op.op, x: thrownRec(error) })
            if (action.rethrow) throw error
          }
          break
        }
        default: throw new Error(`unknown listener action ${action.do}`)
      }
    }
  }
}

function hostLoad (ctx, dsl, setDsl = true) {
  if (setDsl) ctx.renderer._currentDsl = dsl
  let graph
  // compileGraph's own warnings are the host's, not ProgramState's.
  const saved = captured
  captured = null
  try {
    graph = M.compileGraph(dsl)
  } catch (error) {
    ctx.renderer._pipeline = null
    ctx.passJson = []
    return { error: thrownRec(error) }
  } finally {
    captured = saved
  }
  ctx.renderer._pipeline = makePipeline(ctx, graph)
  ctx.passJson = graph.passes.map(p => JSON.stringify(enc(p.uniforms)))
  return { passes: graph.passes.map(passSummary) }
}

// A concrete op; returns its result (thrown errors propagate).
function runOp (ctx, op) {
  const s = ctx.state
  const a = name => dec(op[name])
  switch (op.op) {
    // host
    case 'host.load': return hostLoad(ctx, op.dsl)
    case 'host.loadGraph': return hostLoad(ctx, op.dsl, false)
    case 'host.setDsl': ctx.renderer._currentDsl = op.dsl; return undefined
    case 'host.clearPipeline': ctx.renderer._pipeline = null; ctx.passJson = []; return undefined
    case 'host.setMethods': ctx.methods = op.methods; return undefined
    case 'host.setEnums': ctx.renderer._enums = enumsFor(op.enums); return undefined
    case 'setRenderer': s.setRenderer(op.renderer ? ctx.renderer : null); return undefined
    case 'newState': ctx.newState(op.host); return undefined
    // state
    case 'fromDsl': return s.fromDsl(op.dsl)
    case 'toDsl': return s.toDsl()
    case 'wouldChangeStructure': return s.wouldChangeStructure(op.dsl)
    case 'getValue': return s.getValue(op.stepKey, op.paramName)
    case 'setValue': return s.setValue(op.stepKey, op.paramName, a('value'))
    case 'getStepValues': return s.getStepValues(op.stepKey)
    case 'setStepValues': return s.setStepValues(op.stepKey, a('values'))
    case 'batch': {
      const results = []
      s.batch(() => {
        for (const nested of op.ops) results.push(encResult(nested, runOp(ctx, nested)))
        if (op.throw) throw makeError(op.throw)
      })
      return results
    }
    case 'resetStep': return s.resetStep(op.stepKey)
    case 'setSkip': return s.setSkip(op.stepKey, a('skip'))
    case 'isSkipped': return s.isSkipped(op.stepKey)
    case 'deleteStep': return s.deleteStep(a('stepIndex'))
    case 'insertStep': return s.insertStep(a('afterStepIndex'), op.effectId)
    case 'getStructure': return s.getStructure()
    case 'getCompiled': return s.getCompiled()
    case 'getEffectDef': return s.getEffectDef(op.stepKey)
    case 'stepCount': return s.stepCount
    case 'getStepKeys': return s.getStepKeys()
    case 'getAllStepValues': return s.getAllStepValues()
    case 'setWriteTarget': return s.setWriteTarget(a('planIndex'), a('target'))
    case 'getWriteTarget': return s.getWriteTarget(a('planIndex'))
    case 'setWriteStepTarget': return s.setWriteStepTarget(a('stepIndex'), a('target'))
    case 'getWriteStepTarget': return s.getWriteStepTarget(a('stepIndex'))
    case 'setReadSource': return s.setReadSource(a('stepIndex'), a('source'))
    case 'getReadSource': return s.getReadSource(a('stepIndex'))
    case 'setRead3dVolume': return s.setRead3dVolume(a('stepIndex'), a('volume'))
    case 'setRead3dGeometry': return s.setRead3dGeometry(a('stepIndex'), a('geometry'))
    case 'setWrite3dVolume': return s.setWrite3dVolume(a('stepIndex'), a('volume'))
    case 'setWrite3dGeometry': return s.setWrite3dGeometry(a('stepIndex'), a('geometry'))
    case 'setRenderTarget': return s.setRenderTarget(a('target'))
    case 'getRenderTarget': return s.getRenderTarget()
    case 'clearRoutingOverrides': return s.clearRoutingOverrides()
    case 'setMediaInput': return s.setMediaInput(a('stepIndex'), a('metadata'))
    case 'getMediaInput': return s.getMediaInput(a('stepIndex'))
    case 'removeMediaInput': return s.removeMediaInput(a('stepIndex'))
    case 'getAllMediaInputs': return s.getAllMediaInputs()
    case 'setTextInput': return s.setTextInput(a('stepIndex'), a('metadata'))
    case 'getTextInput': return s.getTextInput(a('stepIndex'))
    case 'removeTextInput': return s.removeTextInput(a('stepIndex'))
    case 'getAllTextInputs': return s.getAllTextInputs()
    case 'applyToPipeline': return s.applyToPipeline()
    case 'serialize': return s.serialize()
    case 'deserialize': return s.deserialize(a('data'))
    case 'emit': return s.emit(op.event, a('data'))
    // listeners
    case 'defineListener': ctx.listeners.set(op.id, makeListener(ctx, op.id, op.actions)); return undefined
    case 'on': return s.on(op.event, ctx.listeners.get(op.id))
    case 'off': return s.off(op.event, ctx.listeners.get(op.id))
    case 'once': return s.once(op.event, ctx.listeners.get(op.id))
    case 'removeAllListeners': return s.removeAllListeners(op.event)
    // standalone helpers
    case 'extractEffectsFromDsl': return M.extractEffectsFromDsl(op.dsl)
    case 'convertParameterForUniform': {
      // CanvasRenderer.prototype.convertParameterForUniform over a renderer
      // whose _enums is the named tree.
      const r = Object.create(M.CanvasRenderer.prototype)
      r._enums = enumsFor(op.enums)
      return r.convertParameterForUniform(a('value'), a('spec'))
    }
    case 'resolveEnumValue': {
      const r = Object.create(M.CanvasRenderer.prototype)
      r._enums = enumsFor(op.enums)
      return r.resolveEnumValue(a('path'))
    }
    default: throw new Error(`unknown op ${op.op}`)
  }
}

// ---------------------------------------------------------------------------
// Macros (expanded with the reference's values)
// ---------------------------------------------------------------------------

const NOISE = 'search synth\nnoise().write(o0)\nrender(o0)'

function effectSteps (ctx) {
  return ctx.state.getStructure().filter(e => M.getEffect(e.effectKey)?.globals)
}

// Deterministic edits: up to four parameters of each of the first two effect
// steps that have globals.
function editOps (ctx) {
  const ops = []
  for (const effect of effectSteps(ctx).slice(0, 2)) {
    const def = M.getEffect(effect.effectKey)
    const stepKey = `step_${effect.stepIndex}`
    let n = 0
    for (const [name, spec] of Object.entries(def.globals)) {
      if (n >= 4) break
      const current = ctx.state.getValue(stepKey, name)
      let value
      switch (spec?.type) {
        case 'float': value = typeof current === 'number' ? current * 0.5 + 0.125 : '0.625'; break
        case 'int': {
          if (spec.choices) {
            const vals = Object.entries(spec.choices).filter(([k]) => !k.endsWith(':')).map(([, v]) => v)
            value = vals.find(v => v !== current) ?? current
          } else {
            value = typeof current === 'number' ? current + 1 : '3'
          }
          break
        }
        case 'boolean': value = !current; break
        case 'color': value = '#336699'; break
        case 'vec2': value = [0.25, '0.5']; break
        case 'vec3': value = [0.25, '0.5', 2]; break
        case 'vec4': value = [0.25, '0.5', 2, -0]; break
        default: continue
      }
      ops.push({ op: 'setValue', stepKey, paramName: name, value: enc(value) })
      n++
    }
  }
  return ops
}

// demo-ui's control initialization (_createControlGroup): every visible
// parameter of a step gets setValue(preserved ?? DSL arg ?? default), or its
// variable reference when the DSL automates it through a variable.
function initControlOps (ctx, effect) {
  const def = M.getEffect(effect.effectKey)
  const stepKey = `step_${effect.stepIndex}`
  const ops = []
  for (const [key, spec] of Object.entries(def.globals)) {
    if (spec.ui?.control === false || spec.ui?.hidden === true) continue
    let value
    const preserved = ctx.state.getValue(stepKey, key)
    if (preserved !== undefined) value = preserved
    else if (effect.args[key] !== undefined) value = effect.args[key]
    else value = structuredClone(spec.default)
    const rawKwarg = effect.rawKwargs?.[key]
    const automationValue = (value && typeof value === 'object') ? value : effect.args?.[key]
    const isAutomated = (automationValue && typeof automationValue === 'object' && (
      automationValue._varRef || ['Oscillator', 'Midi', 'Audio'].includes(automationValue.type) ||
      ['Oscillator', 'Midi', 'Audio'].includes(automationValue._ast?.type))) ||
      (rawKwarg && typeof rawKwarg === 'object' && ['Oscillator', 'Midi', 'Audio'].includes(rawKwarg.type))
    if (isAutomated) {
      if (rawKwarg && rawKwarg.type === 'Ident') ops.push({ op: 'setValue', stepKey, paramName: key, value: enc({ _varRef: rawKwarg.name }) })
      continue
    }
    ops.push({ op: 'setValue', stepKey, paramName: key, value: enc(value) })
  }
  return ops
}

// A one-effect program as the demo viewer builds it for ?effect=ns.func: a
// starter writes o0 directly, a 3D processor reads a noise3d volume and is
// rendered by render3d, anything else filters noise().
function effectDsl (effect) {
  const def = M.getEffect(effect)
  const ns = effect.split('.')[0]
  const func = def.func
  const consumes3d = (def.passes || []).some(p => p.inputs && Object.values(p.inputs).includes('inputTex3d'))
  if (def.outputTex3d && consumes3d) return `search synth3d, ${ns}, render\nnoise3d().${func}().render3d().write(o0)\nrender(o0)`
  if (M.isStarterEffect({ instance: def })) return `search ${ns}\n${func}().write(o0)\nrender(o0)`
  return `search synth, ${ns}\nnoise().${func}().write(o0)\nrender(o0)`
}

function expand (ctx, op) {
  switch (op.op) {
    case 'effectProgram': {
      const dsl = effectDsl(op.effect)
      ctx.exec({ op: 'host.load', dsl })
      ctx.exec({ op: 'fromDsl', dsl })
      return
    }
    case 'defineWalk': {
      // test-define-variants: walk each value of a define parameter; a
      // recompileNeeded regenerates the DSL and recompiles, as demo-ui does.
      const stepKey = op.stepKey || ctx.state.getStepKeys().find(k => ctx.state.getEffectDef(k)?.globals?.[op.param])
      if (!stepKey) throw new Error(`defineWalk: no step has ${op.param}`)
      for (const value of op.values) {
        const out = ctx.exec({ op: 'setValue', stepKey, paramName: op.param, value: enc(value) })
        if (out.record.ev.some(e => e.e === 'recompileNeeded')) expand(ctx, { op: 'reload', apply: true })
      }
      return
    }
    case 'edits':
      for (const e of editOps(ctx)) ctx.exec(e)
      return
    case 'initControls': {
      const effects = effectSteps(ctx)
      const effect = op.stepKey ? effects.find(e => `step_${e.stepIndex}` === op.stepKey) : effects[0]
      if (effect) ctx.exec({ op: 'batch', ops: initControlOps(ctx, effect) })
      return
    }
    case 'reload': {
      const out = ctx.exec({ op: 'toDsl' })
      if (typeof out.value === 'string' && out.value) {
        ctx.exec({ op: 'host.load', dsl: out.value })
        ctx.exec({ op: op.apply ? 'applyToPipeline' : 'fromDsl', ...(op.apply ? {} : { dsl: out.value }) })
      }
      return
    }
    case 'serializeRoundTrip': {
      const out = ctx.exec({ op: 'serialize' })
      if ('value' in out) ctx.exec({ op: 'deserialize', data: enc(out.value) })
      return
    }
    case 'insertAndLoad':
    case 'deleteAndLoad': {
      const concrete = op.op === 'insertAndLoad'
        ? { op: 'insertStep', afterStepIndex: op.afterStepIndex, effectId: op.effectId }
        : { op: 'deleteStep', stepIndex: op.stepIndex }
      const out = ctx.exec(concrete)
      if (out.value?.success) {
        ctx.exec({ op: 'host.load', dsl: out.value.newDsl })
        ctx.exec({ op: 'applyToPipeline' })
      }
      return
    }
    case 'fixture': {
      const src = op.dsl
      ctx.exec({ op: 'host.load', dsl: src })
      ctx.exec({ op: 'fromDsl', dsl: src })
      for (const q of ['getStructure', 'getCompiled', 'getAllStepValues', 'serialize', 'getStepKeys', 'stepCount']) ctx.exec({ op: q })
      ctx.exec({ op: 'toDsl' })
      expand(ctx, { op: 'initControls' })
      expand(ctx, { op: 'edits' })
      const effects = effectSteps(ctx)
      const first = effects[0]
      if (first) {
        const stepKey = `step_${first.stepIndex}`
        const def = M.getEffect(first.effectKey)
        const numeric = Object.entries(def.globals).find(([, spec]) => spec?.type === 'float' || spec?.type === 'int')
        const values = {}
        if (numeric) values[numeric[0]] = '2.5'
        values._custom = 7
        ctx.exec({ op: 'setStepValues', stepKey, values: enc(values) })
      }
      expand(ctx, { op: 'reload' })
      ctx.exec({ op: 'wouldChangeStructure', dsl: src })
      ctx.exec({ op: 'wouldChangeStructure', dsl: NOISE })
      if (first) {
        const stepKey = `step_${first.stepIndex}`
        ctx.exec({ op: 'setSkip', stepKey, skip: true })
        ctx.exec({ op: 'toDsl' })
        ctx.exec({ op: 'setSkip', stepKey, skip: false })
        ctx.exec({ op: 'resetStep', stepKey: effects[1] ? `step_${effects[1].stepIndex}` : stepKey })
      }
      expand(ctx, { op: 'serializeRoundTrip' })
      expand(ctx, { op: 'insertAndLoad', afterStepIndex: 0, effectId: 'filter/blur' })
      expand(ctx, { op: 'deleteAndLoad', stepIndex: 1 })
      ctx.exec({ op: 'toDsl' })
      ctx.exec({ op: 'getAllStepValues' })
      return
    }
    default: {
      const out = ctx.exec(resolveOp(ctx, op))
      if (op.save && 'value' in out) ctx.vars.set(op.save, { raw: out.value, encoded: enc(out.value) })
    }
  }
}

// Fields that hold strings as they are (other fields hold encoded values).
const STRING_FIELDS = new Set(['dsl', 'stepKey', 'paramName', 'effectId', 'event', 'id'])

// `{"$var": name}` fields take a saved result (`"save": name` on an earlier
// op); `stepOf: effectKey` names the first step of that effect.
function resolveOp (ctx, op) {
  const out = {}
  for (const [key, value] of Object.entries(op)) {
    if (key === 'save' || key === 'stepOf') continue
    if (value && typeof value === 'object' && !Array.isArray(value) && Object.hasOwn(value, '$var')) {
      const saved = ctx.vars.get(value.$var)
      if (!saved) throw new Error(`no saved result ${value.$var}`)
      out[key] = STRING_FIELDS.has(key) ? saved.raw : saved.encoded
    } else {
      out[key] = value
    }
  }
  if (op.stepOf) {
    const effect = ctx.state.getStructure().find(e => e.effectKey === op.stepOf)
    if (!effect) throw new Error(`no step of ${op.stepOf}`)
    out.stepKey = `step_${effect.stepIndex}`
  }
  return out
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

function fixtureScenarios (gen) {
  const out = []
  for (const dir of gen.dirs) {
    for (const f of readdirSync(join(ROOT, dir)).filter(f => f.endsWith('.dsl')).sort()) {
      const dsl = readFileSync(join(ROOT, dir, f), 'utf8')
      out.push({ name: `${dir.replace(/^parity\//, '')}/${f.replace(/\.dsl$/, '')}`, host: gen.host, ops: [{ op: 'fixture', dsl }] })
    }
  }
  return out
}

// Probe values for convertParameterForUniform: every JavaScript type the
// conversions branch on, plus the spec's own default and enum names.
const PROBES = [undefined, null, true, false, 0, -0, 1.5, -2.5, 2.5, NaN, Infinity, -Infinity, '3.7', ' 42px', 'abc', '',
  '#ff8000', '#abc', '#', 'red', [1, '2', null], [0.5], [], ['0x10', true, 3, 4, 5], {}, { value: 3 }]

function * catalogSpecs () {
  const root = join(SHADERS, 'effects')
  const dirs = d => readdirSync(d, { withFileTypes: true }).filter(e => e.isDirectory()).map(e => e.name).sort()
  for (const ns of dirs(root)) {
    for (const name of dirs(join(root, ns))) {
      const def = M.getEffect(`${ns}/${name}`)
      if (!def?.globals) continue
      for (const [key, spec] of Object.entries(def.globals)) yield { id: `${ns}/${name}.${key}`, spec }
    }
  }
}

// convertParameterForUniform over every catalog parameter spec (host enums),
// and resolveEnumValue over every enum path the specs name.
function conversionScenarios (gen) {
  const ops = []
  const paths = new Set()
  for (const { spec } of catalogSpecs()) {
    const values = [...PROBES, spec.default]
    const base = spec.enum || spec.enumPath
    if (spec.choices) for (const [name, v] of Object.entries(spec.choices)) values.push(name, v)
    if (typeof base === 'string') {
      values.push(`${base}.missing`, base)
      paths.add(base)
      // The members of the enum the spec names, bare and qualified.
      let node = M.hostEnums
      for (const seg of base.split('.').filter(Boolean)) node = node?.[seg]
      if (node && typeof node === 'object') {
        for (const name of Object.keys(node).slice(0, 12)) {
          values.push(name, `${base}.${name}`)
          paths.add(`${base}.${name}`)
        }
      }
    }
    if (typeof spec.default === 'string') paths.add(spec.default)
    for (const value of values) ops.push({ op: 'convertParameterForUniform', value: enc(value), spec: enc(spec), enums: gen.enums || 'host' })
  }
  for (const enums of ['host', 'std', 'empty', 'none']) {
    for (const path of [...paths, 'color.mono', 'palette.solaris', 'channel', 'channel.r.value', '..color..rgb', 'oscKind.noise', 'constructor', 'color.constructor', 3, true, null, undefined, {}, ['color', 'rgb']]) {
      ops.push({ op: 'resolveEnumValue', path: enc(path), enums })
    }
  }
  return [{ name: `conversions:${gen.enums || 'host'}`, host: { renderer: false }, ops }]
}

function suiteScenarios (suite) {
  const out = []
  for (const sc of suite.scenarios) {
    if (sc.generate === 'conversions') out.push(...conversionScenarios(sc))
    else if (sc.generate === 'fixtures') out.push(...fixtureScenarios(sc))
    else if (sc.generate) throw new Error(`unknown generator ${sc.generate}`)
    else out.push(sc)
  }
  return out
}

async function loadModules () {
  const mod = await bootstrapReference()
  const enums = await import(pathToFileURL(join(SHADERS, 'src', 'lang', 'enums.js')).href)
  const std = await import(pathToFileURL(join(SHADERS, 'src', 'lang', 'std_enums.js')).href)
  return {
    ...mod,
    // CanvasRenderer._enums after loading: the merged enum tree mergeIntoEnums returns.
    hostEnums: enums.default,
    stdEnums: std.stdEnums
  }
}

async function run (suites, resolvedPath, expectedPath, only) {
  M = await loadModules()
  const resolvedFd = openSync(resolvedPath, 'w')
  const expectedFd = openSync(expectedPath, 'w')
  let scenarios = 0
  let records = 0
  try {
    for (const suitePath of suites) {
      const suite = JSON.parse(readFileSync(suitePath, 'utf8'))
      const name = basename(suitePath).replace(/\.json$/, '')
      writeSync(resolvedFd, JSON.stringify({ suite: name }) + '\n')
      for (const sc of suiteScenarios(suite)) {
        const scName = `${name}:${sc.name}`
        if (only && !scName.includes(only)) continue
        const ctx = new Context(scName, sc.host)
        ctx.out = expectedFd
        for (const op of sc.ops) expand(ctx, op)
        writeSync(resolvedFd, JSON.stringify({ name: scName, host: sc.host || null, ops: ctx.resolved }) + '\n')
        scenarios++
        records += ctx.index
      }
    }
  } finally {
    closeSync(resolvedFd)
    closeSync(expectedFd)
  }
  process.stdout.write(JSON.stringify({ scenarios, records }) + '\n')
}

async function main () {
  const argv = process.argv.slice(2)
  const command = argv.shift()
  if (command !== 'run') {
    console.error('usage: node tools/reference-program-state.mjs run <suite.json>... --resolved R --expected E [--only substring]')
    process.exit(2)
  }
  const suites = []
  let resolvedPath = null
  let expectedPath = null
  let only = null
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--resolved') resolvedPath = argv[++i]
    else if (argv[i] === '--expected') expectedPath = argv[++i]
    else if (argv[i] === '--only') only = argv[++i]
    else suites.push(resolve(argv[i]))
  }
  if (!suites.length || !resolvedPath || !expectedPath) throw new Error('missing arguments')
  await run(suites, resolvedPath, expectedPath, only)
}

if (process.argv[1] && basename(process.argv[1]) === 'reference-program-state.mjs') {
  main().catch(err => {
    process.stderr.write(`[reference-program-state] FAILED: ${err?.stack || err}\n`)
    process.exit(1)
  })
}
