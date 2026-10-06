#!/usr/bin/env node
// reference-api.mjs — run the unmodified reference engine's public API over
// the case lists of parity/check_api.mjs and parity/check_portable.mjs.
//
// The reference at $NM_REFERENCE_ROOT is bootstrapped as tools/reference-oracle.mjs
// does it (the host's registration of every catalog effect), the state
// noisemaker_dsl::Registry::with_catalog() reproduces. Its modules are then
// imported directly: runtime/tags.js, palettes.js, lang/constants.js,
// runtime/effect.js, renderer/canvas.js, runtime/registry.js, lang/ops.js,
// lang/validator.js, lang/enums.js, runtime/resources.js and index.js.
//
//   api       one realm: every case in order (the registry cases mutate it,
//             the later cases observe the result). Cases cover runtime/tags.js
//             (constants, tag lookups and validation over every tag of the
//             catalog, namespace registration with every validation rule,
//             the parser and validator against registered namespaces),
//             palettes.js (samplePalette over every palette and a sweep of
//             positions), lang/constants.js, runtime/effect.js (the Effect
//             constructor over every catalog definition, categories over every
//             catalog effect), the canvas.js helpers over every catalog
//             effect, the effect/op/starter/enum registries, VERSION/PHASE and
//             analyzeLiveness/allocateResources over the fixture pool.
//   portable  one fresh realm per scenario (a worker thread each): the
//             registerPortableEffect scenarios of the reference's own tests
//             (shaders/tests/test_portable_registration.js), every catalog
//             effect re-registered as a Portable definition, the effect
//             validator's definitions, edge cases, and the Portable parity
//             fixtures (parity/portable) with their programs compiled. Each
//             step records the thrown error or the registered effect plus the
//             registry lookups of its name; each scenario ends with a digest
//             of the whole registry state (effect lookups, ops, enums, starter
//             ops, parameter and effect aliases, loaded effects).
//
// Both write --cases (the resolved inputs nm-api runs, tagged-JSON encoded as
// tools/reference-dsl-tools.mjs encodes values) and --expected (one
// {id, category, result | error} record per case).
//
// Usage:
//   node tools/reference-api.mjs api --cases C.jsonl --expected E.jsonl
//   node tools/reference-api.mjs portable --cases C.jsonl --expected E.jsonl

import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads'

import { bootstrapReference, normalizeGraph, runStage } from './reference-oracle.mjs'
import { decode, encode, useEffectClass } from './reference-dsl-tools.mjs'
import { loadPortableDefinition } from './portable.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const REFERENCE_ROOT = resolve(process.env.NM_REFERENCE_ROOT)
const SHADERS = join(REFERENCE_ROOT, 'shaders')
const EFFECTS = join(SHADERS, 'effects')
const CATALOG = join(ROOT, 'crates', 'noisemaker-effects', 'catalog')

const refModule = path => import(pathToFileURL(join(SHADERS, 'src', path)).href)

// Own property names of Object.prototype: the inputs where the reference's
// property reads see inherited members.
const PROTO_NAMES = ['constructor', '__defineGetter__', '__defineSetter__', 'hasOwnProperty', '__lookupGetter__',
  '__lookupSetter__', 'isPrototypeOf', 'propertyIsEnumerable', 'toString', 'valueOf', '__proto__', 'toLocaleString']

async function loadReference () {
  const mod = await bootstrapReference()
  const effect = await refModule('runtime/effect.js')
  useEffectClass(effect.Effect)
  return {
    mod,
    tags: await refModule('runtime/tags.js'),
    palettes: await refModule('palettes.js'),
    constants: await refModule('lang/constants.js'),
    effect,
    canvas: await refModule('renderer/canvas.js'),
    registry: await refModule('runtime/registry.js'),
    ops: await refModule('lang/ops.js'),
    validator: await refModule('lang/validator.js'),
    enums: await refModule('lang/enums.js'),
    paramAliases: await refModule('lang/paramAliases.js'),
    effectAliases: await refModule('lang/effectAliases.js'),
    resources: await refModule('runtime/resources.js'),
    renderers: []
  }
}

function sortedDirs (dir) {
  return readdirSync(dir, { withFileTypes: true }).filter(d => d.isDirectory()).map(d => d.name).sort()
}

// Every catalog effect id (namespace/effect directory), in catalog order.
function catalogIds () {
  const ids = []
  for (const ns of sortedDirs(EFFECTS)) {
    for (const name of sortedDirs(join(EFFECTS, ns))) {
      if (existsSync(join(EFFECTS, ns, name, 'definition.js'))) ids.push(`${ns}/${name}`)
    }
  }
  return ids
}

function catalogJson (id) {
  return JSON.parse(readFileSync(join(CATALOG, id, 'definition.json'), 'utf8'))
}

const plainObject = value => (value && typeof value === 'object' ? { ...value } : value)
const mapEntries = map => [...map].map(([k, v]) => [k, plainObject(v)])

// An effect registry value as the gates compare it.
function describe (inst) {
  if (inst === undefined) return undefined
  return { namespace: inst?.namespace, func: inst?.func, name: inst?.name }
}

// ---------------------------------------------------------------------------
// API cases
// ---------------------------------------------------------------------------

const CONSTANTS = {
  TAG_DEFINITIONS: R => R.tags.TAG_DEFINITIONS,
  VALID_TAGS: R => R.tags.VALID_TAGS,
  NAMESPACE_DESCRIPTIONS: R => Object.fromEntries(Object.keys(R.tags.NAMESPACE_DESCRIPTIONS).map(k => [k, { ...R.tags.NAMESPACE_DESCRIPTIONS[k] }])),
  VALID_NAMESPACES: R => [...R.tags.VALID_NAMESPACES],
  BUILTIN_NAMESPACE: R => R.tags.BUILTIN_NAMESPACE,
  IO_FUNCTIONS: R => R.tags.IO_FUNCTIONS,
  PALETTES: R => R.palettes.PALETTES,
  STARTER_BLOCK_CATEGORIES: R => R.constants.STARTER_BLOCK_CATEGORIES,
  BLOCK_CATEGORY_TYPES: R => R.constants.BLOCK_CATEGORY_TYPES,
  DEFAULT_BLOCK_CATEGORY_TYPE: R => R.constants.DEFAULT_BLOCK_CATEGORY_TYPE,
  DEFAULT_CATEGORY: R => R.effect.DEFAULT_CATEGORY,
  VERSION: R => R.mod.VERSION,
  PHASE: R => R.mod.PHASE,
  stdEnums: R => R.enums.default
}

// The effect argument of a canvas.js helper: a catalog effect or a definition.
function effectArg (R, a) {
  if (typeof a.catalog === 'string') return { instance: R.registry.getEffect(a.catalog) }
  return { instance: a.definition }
}

const API_OPS = {
  constant: (R, a) => CONSTANTS[a.name](R),
  // runtime/tags.js
  isValidTag: (R, a) => R.tags.isValidTag(a.tagId),
  getTagDefinition: (R, a) => plainObject(R.tags.getTagDefinition(a.tagId)),
  validateTags: (R, a) => R.tags.validateTags(a.tags),
  isIOFunction: (R, a) => R.tags.isIOFunction(a.funcName),
  isValidNamespace: (R, a) => R.tags.isValidNamespace(a.id),
  getNamespaceDescription: (R, a) => plainObject(R.tags.getNamespaceDescription(a.id)),
  registerNamespace: (R, a) => ({ ...R.tags.registerNamespace(a.id, a.descriptor) }),
  unregisterNamespace: (R, a) => R.tags.unregisterNamespace(a.id),
  parse: (R, a) => runStage(R.mod, 'ast', a.src),
  compile: (R, a) => runStage(R.mod, 'validated', a.src),
  // palettes.js
  samplePalette: (R, a) => R.palettes.samplePalette(a.name, a.t),
  samplePaletteSweep: (R, a) => a.ts.map(t => R.palettes.samplePalette(a.name, t)),
  // lang/constants.js
  isStarterBlockCategory: (R, a) => R.constants.isStarterBlockCategory(a.category),
  getBlockCategoryType: (R, a) => R.constants.getBlockCategoryType(a.category),
  // runtime/effect.js
  newEffect: (R, a) => ({ ...new R.effect.Effect(a.config) }),
  getUniformCategory: (R, a) => R.effect.getUniformCategory(a.spec),
  groupGlobalsByCategory: (R, a) => R.effect.groupGlobalsByCategory(a.globals, a.options),
  getCategories: (R, a) => R.effect.getCategories(a.globals),
  // renderer/canvas.js
  cloneParamValue: (R, a) => R.canvas.cloneParamValue(a.value),
  isValidIdentifier: (R, a) => R.canvas.isValidIdentifier(a.name),
  sanitizeEnumName: (R, a) => R.canvas.sanitizeEnumName(a.name),
  hasTexSurfaceParam: (R, a) => R.canvas.hasTexSurfaceParam(effectArg(R, a)),
  hasExplicitTexParam: (R, a) => R.canvas.hasExplicitTexParam(effectArg(R, a)),
  getVolGeoParams: (R, a) => R.canvas.getVolGeoParams(effectArg(R, a)),
  needsInputTex3d: (R, a) => R.canvas.needsInputTex3d(effectArg(R, a)),
  // The reference returns its falsy `outputTex3d` operand for effects
  // without one; the port's API is boolean.
  is3dGenerator: (R, a) => Boolean(R.canvas.is3dGenerator(effectArg(R, a))),
  is3dProcessor: (R, a) => Boolean(R.canvas.is3dProcessor(effectArg(R, a))),
  isStarterEffect: (R, a) => R.canvas.isStarterEffect(effectArg(R, a)),
  // runtime/registry.js, lang/ops.js, lang/validator.js, lang/enums.js
  registerEffect: (R, a) => R.registry.registerEffect(a.name, a.definition),
  unregisterEffect: (R, a) => R.registry.unregisterEffect(a.name),
  getEffect: (R, a) => describe(R.registry.getEffect(a.name)),
  getAllEffects: R => [...R.registry.getAllEffects()].map(([k, v]) => [String(k), describe(v)]),
  registerOp: (R, a) => R.ops.registerOp(a.name, a.spec),
  op: (R, a) => R.ops.ops[a.name],
  registerStarterOps: (R, a) => R.validator.registerStarterOps(a.names),
  isStarterOp: (R, a) => R.validator.isStarterOp(a.name),
  mergeIntoEnums: async (R, a) => { await R.enums.mergeIntoEnums(a.source) },
  // CanvasRenderer manifest and effect-string queries (a renderer whose
  // basePath is the reference's shaders directory)
  getEffectsFromManifest: (R, a) => R.host.getEffectsFromManifest(a.namespace, a.options),
  setLocale: async (R, a) => R.host.setLocale(a.locale),
  getEffectDescriptionSweep: async (R, a) => {
    await R.host.setLocale(a.locale)
    return a.ids.map(id => R.host.getEffectDescription(id))
  },
  localizeSweep: async (R, a) => {
    await R.host.setLocale(a.locale)
    return a.ids.map(id => R.host.localize(id, a.fallback))
  },
  // runtime/resources.js
  analyzeLiveness: (R, a) => mapEntries(R.resources.analyzeLiveness(a.passes)),
  allocateResources: (R, a) => mapEntries(R.resources.allocateResources(a.passes))
}

function apiCases (R) {
  const cases = []
  const seen = new Map()
  const add = (op, label, args = {}) => {
    let id = `${op}:${label}`
    const n = seen.get(id) || 0
    seen.set(id, n + 1)
    if (n) id += `#${n}`
    cases.push({ id, op, args })
  }
  const ids = catalogIds()
  const instances = ids.map(id => [id, R.registry.getEffect(id)])
  const catalogTags = new Set()
  const choiceNames = new Set()
  const globalKeys = new Set()
  const specs = []
  for (const [id, inst] of instances) {
    for (const t of inst.tags || []) catalogTags.add(t)
    for (const [key, spec] of Object.entries(inst.globals || {})) {
      globalKeys.add(key)
      specs.push([`${id}.${key}`, spec])
      for (const name of Object.keys(spec?.choices || {})) choiceNames.add(name)
    }
  }

  for (const name of Object.keys(CONSTANTS)) add('constant', name, { name })

  // --- runtime/tags.js ---------------------------------------------------
  const junk = ['', 'Color', 'COLOR', ' color', 'color ', '3D', 'video2', 'unknown', 'synth', 'user', 'io', 'é', 'colör']
  for (const t of new Set([...R.tags.VALID_TAGS, ...catalogTags, ...PROTO_NAMES, ...junk])) {
    add('isValidTag', t, { tagId: t })
    add('getTagDefinition', t, { tagId: t })
  }
  for (const [id, inst] of instances) add('validateTags', id, { tags: inst.tags })
  for (const [label, tags] of [['empty', []], ['valid', ['color', '3d', 'video']], ['mixed', ['color', 'bogus', 3, null, undefined, 'blur']],
    ['string', 'color'], ['null', null], ['undefined', undefined], ['object', { 0: 'color' }], ['nested', [['color']]],
    ['nan', [NaN]], ['proto', ['constructor', 'toString']], ['duplicates', ['bogus', 'bogus']]]) {
    add('validateTags', label, { tags })
  }
  const opNames = Object.keys(R.ops.ops)
  for (const f of new Set([...R.tags.IO_FUNCTIONS, ...opNames.map(o => o.split('.').pop()), 'read2d', 'Read', 'out', 'osc', ...PROTO_NAMES, ''])) {
    add('isIOFunction', f, { funcName: f })
  }
  const namespaceProbes = () => [...new Set([...R.tags.VALID_NAMESPACES, ...ids.map(i => i.split('/')[0]), 'notReal', 'Synth', '', ...PROTO_NAMES])]
  for (const ns of namespaceProbes()) {
    add('isValidNamespace', ns, { id: ns })
    add('getNamespaceDescription', ns, { id: ns })
  }

  // registerNamespace / unregisterNamespace: the reference's own test
  // sequence (shaders/tests/test_register_namespace.js), then every
  // validation rule over its whole input classes.
  const reg = (label, id, descriptor) => add('registerNamespace', label, { id, descriptor })
  const unreg = (label, id) => add('unregisterNamespace', label, { id })
  const state = label => {
    add('constant', `${label}/VALID_NAMESPACES`, { name: 'VALID_NAMESPACES' })
    add('constant', `${label}/NAMESPACE_DESCRIPTIONS`, { name: 'NAMESPACE_DESCRIPTIONS' })
  }
  reg('happy', 'myFooHappy', { description: 'Foo collection' })
  add('isValidNamespace', 'happy', { id: 'myFooHappy' })
  add('getNamespaceDescription', 'happy', { id: 'myFooHappy' })
  state('happy')
  unreg('happy', 'myFooHappy')
  add('isValidNamespace', 'happy-removed', { id: 'myFooHappy' })
  unreg('never', 'neverRegistered')
  reg('twice', 'myFooTwice', { description: 'twice' })
  unreg('twice-1', 'myFooTwice')
  unreg('twice-2', 'myFooTwice')
  for (const id of ['', null, undefined, 42, true, {}, ['abc'], 'Foo', '1foo', 'foo-bar', 'foo bar', 'foo.bar', 'foo_bar', '_foo',
    'fooé', 'a', 'aB1', 'z9', 'ab\n', '\tab']) {
    reg(`id/${JSON.stringify(id) ?? 'undefined'}`, id, { description: 'x' })
  }
  for (const kw of ['render', 'let', 'search', 'subchain', 'write', 'write3d', 'if', 'elif', 'else', 'break', 'continue', 'return', 'true', 'false']) {
    reg(`keyword/${kw}`, kw, { description: 'x' })
  }
  for (const fn of ['read', 'write', 'read3d', 'write3d', 'render', 'render3d', 'from', 'osc', 'midi', 'audio', 'null', 'undefined']) {
    reg(`reserved/${fn}`, fn, { description: 'x' })
  }
  for (const b of R.tags.VALID_NAMESPACES) {
    reg(`builtin/${b}`, b, { description: 'x' })
    unreg(`builtin/${b}`, b)
  }
  for (const [label, descriptor] of [['null', null], ['undefined', undefined], ['string', 'string'], ['number', 42], ['array', ['x']],
    ['empty', {}], ['emptyDescription', { description: '' }], ['numberDescription', { description: 42 }],
    ['nullDescription', { description: null }], ['objectDescription', { description: { text: 'x' } }]]) {
    reg(`descriptor/${label}`, 'myFooDesc', descriptor)
  }
  add('isValidNamespace', 'descriptor-rejected', { id: 'myFooDesc' })
  reg('idem-1', 'myFooIdem', { description: 'same' })
  reg('idem-2', 'myFooIdem', { description: 'same', extra: true })
  state('idem')
  reg('conflict-1', 'myFooConflict', { description: 'first' })
  reg('conflict-2', 'myFooConflict', { description: 'different' })
  state('conflict')
  unreg('idem', 'myFooIdem')
  unreg('conflict', 'myFooConflict')
  for (const p of PROTO_NAMES) unreg(`proto/${p}`, p)
  // The parser's search directive and the validator observe the live table.
  reg('dsl', 'myFooDsl', { description: 'Foo DSL test' })
  add('registerEffect', 'dsl', { name: 'myFooDsl/bar', definition: { name: 'bar', namespace: 'myFooDsl', func: 'bar' } })
  add('registerOp', 'dsl', { name: 'myFooDsl.bar', spec: { name: 'bar', args: [{ name: 'amount', type: 'float', default: 0.5, min: 0, max: 1 }] } })
  add('registerStarterOps', 'dsl', { names: ['myFooDsl.bar'] })
  for (const [label, src] of [
    ['search', 'search myFooDsl\nbar().write(o0)'],
    ['search-list', 'search synth, myFooDsl\nbar(amount: 0.25).write(o0)\nrender(o0)'],
    ['qualified', 'search synth\nmyFooDsl.bar().write(o0)\nrender(o0)'],
    ['program', 'search myFooDsl\nbar(amount: 0.75).write(o0)\nrender(o0)'],
    ['unknown', 'search notAnyNamespace\nfoo().write(o0)']
  ]) {
    add('parse', label, { src })
    add('compile', label, { src })
  }
  add('isStarterOp', 'dsl', { name: 'myFooDsl.bar' })
  reg('ephemeral', 'myFooEphemeral', { description: 'temp' })
  unreg('ephemeral', 'myFooEphemeral')
  add('parse', 'ephemeral', { src: 'search myFooEphemeral\nfoo().write(o0)' })
  unreg('dsl', 'myFooDsl')
  add('getEffect', 'dsl-kept', { name: 'myFooDsl/bar' })
  add('parse', 'dsl-unregistered', { src: 'search myFooDsl\nbar().write(o0)' })
  add('compile', 'dsl-unregistered', { src: 'search myFooDsl\nbar().write(o0)\nrender(o0)' })
  reg('late', 'zLate', { description: 'registered last' })
  state('late')
  add('parse', 'late-error', { src: 'search nope\nnoise().write(o0)' })

  // --- palettes.js -------------------------------------------------------
  const ts = Array.from({ length: 257 }, (_, i) => i / 256)
  ts.push(-0, -1, 1.5, 2, 10, 0.1, 1 / 3, 2 / 3, 1e6, 1e10, 1e300, -1e300, Number.MIN_VALUE, Number.EPSILON, NaN, Infinity, -Infinity)
  for (const name of Object.keys(R.palettes.PALETTES)) add('samplePaletteSweep', name, { name, ts })
  for (const name of ['bogus', '', 'Grayscale', ...PROTO_NAMES]) add('samplePalette', name, { name, t: 0.5 })

  // --- lang/constants.js -------------------------------------------------
  for (const c of [...Object.keys(R.constants.BLOCK_CATEGORY_TYPES), 'Unknown', '', 'synths', 'Color&FX', ...PROTO_NAMES, null, undefined, 3, true, ['Synths'], { Synths: 1 }]) {
    const label = typeof c === 'string' ? c : `${typeof c}:${JSON.stringify(c)}`
    add('isStarterBlockCategory', label, { category: c })
    add('getBlockCategoryType', label, { category: c })
  }

  // --- runtime/effect.js -------------------------------------------------
  for (const id of ids) add('newEffect', id, { config: catalogJson(id) })
  for (const [label, config] of [['empty', {}], ['falsy', { name: '', func: 0, tags: null, hidden: 0, deprecatedBy: '' }],
    ['truthy', { name: 'N', hidden: 'yes', deprecatedBy: 5, extra: 1, state: { kept: false }, uniforms: [1] }],
    ['hooks', { func: 'f', onInit: 'init', onUpdate: 1, onDestroy: [], asyncInit: {} }],
    ['all', { name: 'n', namespace: 'ns', func: 'f', description: 'd', tags: ['color'], globals: { a: {} }, passes: [{}], textures: { t: {} },
      outputTex3d: { width: 8 }, outputGeo: true, uniformLayout: { a: 'float' }, uniformLayouts: { p: {} }, paramAliases: { o: 'a' },
      openCategories: ['general'], defaultProgram: 'p', hidden: true, deprecatedBy: 'x' }]]) {
    add('newEffect', label, { config })
  }
  for (const [id, inst] of instances) {
    for (const options of [undefined, { includeHidden: false }, { includeHidden: true }]) {
      add('groupGlobalsByCategory', `${id}/${JSON.stringify(options) ?? 'default'}`, { globals: inst.globals, options })
    }
    add('getCategories', id, { globals: inst.globals })
  }
  for (const [label, spec] of specs) add('getUniformCategory', label, { spec })
  const syntheticGlobals = [
    ['none', undefined], ['null', null], ['empty', {}], ['number', 5], ['string', 'ab'], ['array', [{ ui: { category: 'x' } }, {}]],
    ['order', { a: { ui: { category: 'effect' } }, b: {}, c: { ui: { control: false } }, d: { ui: { category: 7 } }, e: { ui: { category: 'effect', hidden: true } } }],
    ['falsyCategory', { a: { ui: { category: '' } }, b: { ui: { category: 0 } }, c: { ui: { category: null } } }],
    ['numeric', { a: { ui: { category: '10' } }, b: { ui: { category: '2' } }, c: { ui: { category: 'general' } } }],
    ['hiddenString', { a: { ui: { hidden: 'true' } }, b: { ui: { control: 0 } } }],
    ['nullSpec', { a: null }], ['primitiveSpec', { a: 3, b: 'x' }],
    ...PROTO_NAMES.map(p => [`proto/${p}`, { a: { ui: { category: p } } }])
  ]
  for (const [label, globals] of syntheticGlobals) {
    add('groupGlobalsByCategory', label, { globals })
    add('groupGlobalsByCategory', `${label}/hidden`, { globals, options: { includeHidden: true } })
    add('getCategories', label, { globals })
  }
  for (const [label, spec] of [['undefined', undefined], ['null', null], ['noUi', {}], ['uiNull', { ui: null }], ['number', { ui: { category: 3 } }],
    ['object', { ui: { category: { a: 1 } } }], ['string', 'spec']]) {
    add('getUniformCategory', label, { spec })
  }

  // --- renderer/canvas.js ------------------------------------------------
  const helpers = ['hasTexSurfaceParam', 'hasExplicitTexParam', 'getVolGeoParams', 'needsInputTex3d', 'is3dGenerator', 'is3dProcessor', 'isStarterEffect']
  for (const id of ids) for (const h of helpers) add(h, id, { catalog: id })
  const syntheticDefs = [
    ['empty', {}],
    ['texSurface', { globals: { tex: { type: 'surface', default: 'inputTex' } } }],
    ['texExplicit', { globals: { tex: { type: 'surface', default: 'o1' } } }],
    ['texNoDefault', { globals: { tex: { type: 'surface' } } }],
    ['texOther', { globals: { tex: { type: 'float' } } }],
    ['volGeo', { globals: { b: { type: 'geometry' }, a: { type: 'volume' }, c: { type: 'volume' }, d: { type: 'geometry' } } }],
    ['inputTex3d', { passes: [{ inputs: { a: 'x' } }, { inputs: { b: 'inputTex3d' } }] }],
    ['noPasses', { passes: [] }],
    ['starterInputs', { passes: [{ inputs: { a: '_tmp', b: 'global_x' } }] }],
    ['filterInputs', { passes: [{ inputs: { a: 'o3' } }] }],
    ['gen3d', { func: 'noise3d', outputTex3d: { width: 8 } }],
    ['gen3dNoOutput', { func: 'noise3d' }],
    ['proc3d', { func: 'render3d', outputTex3d: true }],
    ['unknown3d', { func: 'other3d', outputTex3d: true }]
  ]
  for (const [label, definition] of syntheticDefs) for (const h of helpers) add(h, label, { definition })
  const values = [
    ['number', 1.5], ['negZero', -0], ['nan', NaN], ['string', 'x'], ['null', null], ['undefined', undefined], ['bool', true],
    ['array', [1, [2, 3], { a: 4 }]], ['object', { a: 1, b: [1, 2], c: { d: 'e' } }],
    ['special', { n: NaN, i: Infinity, z: -0, u: undefined, arr: [undefined, NaN, -0], s: ' ' }],
    ['numericKeys', { b: 1, 2: 'two', 1: 'one' }], ['emptyObject', {}], ['emptyArray', []]
  ]
  for (const [label, value] of values) add('cloneParamValue', label, { value })
  for (const [label, spec] of specs) if (spec && Object.hasOwn(spec, 'default')) add('cloneParamValue', label, { value: spec.default })
  const names = new Set([...choiceNames, ...globalKeys, '', ' ', 'Cell Scale', 'a  b c', '3d', 'x-y', '_ok', 'a b', 'a b', 'a﻿b',
    'a\u0085b', 'soft ßtep', 'x ıy', 'one\ttwo\nthree', 'trailing ', ' leading', 'é ok', 'ok é', '__proto__', 'constructor', 'a b-c d'])
  for (const name of names) {
    add('isValidIdentifier', name, { name })
    add('sanitizeEnumName', name, { name })
  }

  // --- runtime/registry.js, ops, starter ops, enums, index.js -----------------
  add('getAllEffects', 'catalog')
  for (const name of [...opNames, ...new Set(opNames.map(o => o.split('.').pop())), 'particles', 'render.particles', 'nope']) {
    add('isStarterOp', name, { name })
  }
  add('registerEffect', 'probe', { name: 'apiProbe/x', definition: { namespace: 'apiProbe', name: 'x', func: 'x' } })
  add('registerEffect', 'probe-again', { name: 'apiProbe/x', definition: { namespace: 'apiProbe', name: 'x', func: 'y' } })
  add('registerEffect', 'probe-2', { name: 'apiProbe.z', definition: { namespace: 'apiProbe', func: 'z' } })
  add('getEffect', 'probe', { name: 'apiProbe/x' })
  add('unregisterEffect', 'probe', { name: 'apiProbe/x' })
  add('unregisterEffect', 'probe-again', { name: 'apiProbe/x' })
  add('unregisterEffect', 'synth.noise', { name: 'synth.noise' })
  add('getEffect', 'synth.noise', { name: 'synth.noise' })
  add('getAllEffects', 'after')
  add('registerOp', 'probe', { name: 'apiProbe.z', spec: { name: 'z', args: [] } })
  add('op', 'probe', { name: 'apiProbe.z' })
  add('op', 'synth.noise', { name: 'synth.noise' })
  add('registerStarterOps', 'probe', { names: ['apiProbe.z', '', 'zStandalone'] })
  for (const name of ['apiProbe.z', 'z', 'zStandalone', 'other.zStandalone']) add('isStarterOp', `probe/${name}`, { name })
  add('mergeIntoEnums', 'probe', { source: { apiProbe: { Keep: { type: 'Number', value: 7 } }, synth: { noise: { type: { extra: { type: 'Number', value: 99 } } } } } })
  add('mergeIntoEnums', 'ignored', { source: null })
  add('mergeIntoEnums', 'protoKeys', { source: { constructor: { a: 1 }, prototype: { b: 2 }, ok: { c: { type: 'Number', value: 1 } } } })
  add('constant', 'stdEnums/after', { name: 'stdEnums' })

  // --- CanvasRenderer manifest and effect strings --------------------------
  const manifestIds = Object.keys(R.host.manifest)
  for (const ns of [...new Set(manifestIds.map(id => id.split('/')[0])), 'user', 'nope', '']) {
    for (const options of [undefined, { includeHidden: false }, { includeHidden: true }]) {
      add('getEffectsFromManifest', `${ns}/${JSON.stringify(options) ?? 'default'}`, { namespace: ns, options })
    }
  }
  const locales = [null, 'en', 'de', 'es', 'fr', 'it', 'ja', 'pt', 'xx', '']
  const stringIds = new Set()
  for (const f of readdirSync(EFFECTS).filter(f => /^strings\.[A-Za-z_-]+\.json$/.test(f)).sort()) {
    for (const id of Object.keys(JSON.parse(readFileSync(join(EFFECTS, f), 'utf8')))) stringIds.add(id)
  }
  const extraIds = ['nope', '', 'filter/nope#desc', ...PROTO_NAMES]
  for (const locale of locales) {
    const label = JSON.stringify(locale)
    add('setLocale', label, { locale })
    add('getEffectDescriptionSweep', label, { locale, ids: [...manifestIds, ...extraIds] })
    add('localizeSweep', `${label}/fallback`, { locale, ids: [...stringIds, ...extraIds], fallback: 'FALLBACK' })
    add('localizeSweep', `${label}/default`, { locale, ids: [...extraIds, 'synth/noise', '@ns/synth'], fallback: undefined })
  }

  // --- runtime/resources.js ----------------------------------------------
  for (const f of readdirSync(join(ROOT, 'parity', 'programs')).filter(f => f.endsWith('.dsl')).sort()) {
    let passes
    try {
      passes = R.mod.expand(R.mod.compile(readFileSync(join(ROOT, 'parity', 'programs', f), 'utf8'))).passes
    } catch { continue }
    const label = f.replace(/\.dsl$/, '')
    add('analyzeLiveness', label, { passes })
    add('allocateResources', label, { passes })
  }
  return cases
}

const CATEGORY = op => {
  if (['isValidTag', 'getTagDefinition', 'validateTags', 'isIOFunction', 'isValidNamespace', 'getNamespaceDescription', 'registerNamespace', 'unregisterNamespace', 'parse', 'compile'].includes(op)) return 'tags'
  if (op.startsWith('samplePalette')) return 'palettes'
  if (['isStarterBlockCategory', 'getBlockCategoryType'].includes(op)) return 'constants'
  if (['newEffect', 'getUniformCategory', 'groupGlobalsByCategory', 'getCategories'].includes(op)) return 'effect'
  if (['analyzeLiveness', 'allocateResources'].includes(op)) return 'resources'
  if (['getEffectsFromManifest', 'setLocale', 'getEffectDescriptionSweep', 'localizeSweep'].includes(op)) return 'renderer'
  if (['registerEffect', 'unregisterEffect', 'getEffect', 'getAllEffects', 'registerOp', 'op', 'registerStarterOps', 'isStarterOp', 'mergeIntoEnums'].includes(op)) return 'registry'
  if (op === 'constant') return 'constants'
  return 'canvas'
}

function thrownRecord (error) {
  if (error instanceof Error) return { name: error.name, message: error.message }
  return { thrown: encode(error) }
}

async function runOp (fn, R, args) {
  try {
    return { result: encode(await fn(R, args)) }
  } catch (error) {
    return { error: thrownRecord(error) }
  }
}

// The renderer's fetch() of its manifest and string catalogs, served from the
// reference checkout.
function serveReferenceFiles () {
  globalThis.fetch = async (url) => {
    const path = String(url)
    if (path.startsWith(`${SHADERS}/`) && existsSync(path)) {
      const text = readFileSync(path, 'utf8')
      return { ok: true, json: async () => JSON.parse(text), text: async () => text }
    }
    return { ok: false, json: async () => ({}), text: async () => '' }
  }
}

async function runApi (casesPath, expectedPath) {
  const R = await loadReference()
  serveReferenceFiles()
  // loadManifest() also registers the manifest's starter ops and merges the
  // standard enums again: no change to the bootstrapped registries.
  R.host = new R.mod.CanvasRenderer({ basePath: SHADERS })
  await R.host.loadManifest()
  const cases = apiCases(R)
  const resolved = []
  const expected = []
  for (const { id, op, args } of cases) {
    const encodedArgs = {}
    for (const [k, v] of Object.entries(args)) encodedArgs[k] = encode(v)
    const category = CATEGORY(op)
    resolved.push(JSON.stringify({ id, op, category, args: encodedArgs }))
    const decoded = {}
    for (const [k, v] of Object.entries(encodedArgs)) decoded[k] = decode(v)
    expected.push(JSON.stringify({ id, category, ...await runOp(API_OPS[op], R, decoded) }))
  }
  writeFileSync(casesPath, resolved.join('\n') + '\n')
  writeFileSync(expectedPath, expected.join('\n') + '\n')
  process.stdout.write(JSON.stringify({ cases: cases.length }) + '\n')
}

// ---------------------------------------------------------------------------
// Portable scenarios
// ---------------------------------------------------------------------------

const STUB_GLSL = '#version 300 es\nvoid main() {}'
const STUB_WGSL = '@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }'

// definition() of shaders/tests/test_portable_registration.js
function testDefinition (func, overrides = {}) {
  return {
    namespace: 'user', name: func, func,
    globals: {},
    passes: [{ name: 'main', program: 'main', inputs: {}, outputs: { fragColor: 'outputTex' } }],
    shaders: { main: { glsl: STUB_GLSL } },
    ...overrides
  }
}

// A catalog effect as a Portable definition: its JSON definition in the user
// namespace, with its WGSL attached.
function catalogPortable (id) {
  const def = catalogJson(id)
  def.namespace = 'user'
  const shaders = {}
  for (const pass of def.passes || []) {
    const prog = pass?.program
    if (typeof prog !== 'string') continue
    const path = join(CATALOG, id, 'wgsl', `${prog}.wgsl`)
    if (existsSync(path)) shaders[prog] = { wgsl: readFileSync(path, 'utf8') }
  }
  def.shaders = shaders
  return def
}

// The plain definitions the effect-definition validator suites check.
function validatorDefinitions () {
  const out = []
  const dir = join(ROOT, 'parity', 'dsl-tools')
  for (const f of readdirSync(dir).filter(f => f.endsWith('.json')).sort()) {
    const suite = JSON.parse(readFileSync(join(dir, f), 'utf8'))
    for (const c of suite.cases || []) {
      if (c.op !== 'validateEffectDefinition' || !c.definition || typeof c.definition !== 'object') continue
      if (Object.keys(c.definition).some(k => k.startsWith('$') && k !== '$js')) continue
      out.push([`${f.replace(/\.json$/, '')}/${c.id}`, c.definition])
    }
  }
  return out
}

// Definitions with a stub shader for every named program (so validation
// reaches the checks after the shader checks).
function withStubShaders (def) {
  if (!def || typeof def !== 'object' || Array.isArray(def) || !Array.isArray(def.passes) || def.shaders !== undefined) return null
  const shaders = {}
  for (const pass of def.passes) {
    if (pass && typeof pass.program === 'string' && pass.program) shaders[pass.program] = { wgsl: STUB_WGSL }
  }
  return { ...def, shaders }
}

function portableScenarios () {
  const S = []
  const scenario = (id, steps, setup = []) => S.push({ id, setup, steps })
  const reg = (definition, renderer = 0) => ({ op: 'register', definition, renderer })

  // shaders/tests/test_portable_registration.js
  scenario('test/contract', [reg(testDefinition('portableContract', {
    textures: { history: { width: 32, height: 32, format: 'rgba16f' } },
    outputTex3d: { width: 8, height: 8, depth: 8 },
    outputGeo: { count: 16 },
    uniformLayout: { time: 'float' }, uniformLayouts: { main: { time: 'float' } },
    passes: [{ name: 'main', program: 'main', type: 'compute', drawMode: 'points', count: 16, inputs: { previous: 'history' }, outputs: { fragColor: 'outputTex' } }],
    defaultProgram: 'search user\nportableContract().write(o0)\nrender(o0)'
  })), { op: 'register', definition: testDefinition('portableContract'), renderer: 0 }])
  const paramsDsl = 'search user\nportableParams(oldMode: SoftLight, pinned: Keep, primary: Keep).write(o0)\nrender(o0)'
  scenario('test/params', [
    reg(testDefinition('portableParams', {
      globals: {
        mode: { type: 'int', default: 0, uniform: 'modeUniform', choices: { 'Modes:': -1, 'Soft Light': 3 } },
        pinned: { type: 'int', default: 0, uniform: 'pinnedUniform', enumPath: 'portableExplicit', choices: { Keep: 99 } },
        primary: { type: 'int', default: 0, uniform: 'primaryUniform', enum: 'portableExplicit', enumPath: 'missing', choices: { Keep: 99 } }
      },
      paramAliases: { oldMode: 'mode' }
    })),
    { op: 'compile', dsl: paramsDsl },
    { op: 'graph', dsl: paramsDsl },
    { op: 'resolveEnumValue', path: 'user.portableParams.mode.SoftLight' },
    { op: 'resolveEnumValue', path: 'user.portableParams.mode.Soft Light' },
    { op: 'resolveEnumValue', path: 'portableExplicit.Keep' }
  ], [{ op: 'mergeIntoEnums', source: { portableExplicit: { Keep: { type: 'Number', value: 7 } } } }])
  const bindings = ['inputTex', 'inputTex3d', 'inputGeo', 'inputXyz', 'inputVel', 'inputRgba', 'src', 'o0', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7']
  scenario('test/starter-inference', [
    ...bindings.map((binding, index) => reg(testDefinition(`portableInput${index}`, {
      passes: [{ program: 'main', inputs: { source: binding }, outputs: { fragColor: 'outputTex' } }]
    }))),
    reg(testDefinition('portableExplicitFilter', { starter: false })),
    reg(testDefinition('portableExplicitStarter', { starter: true, passes: [{ program: 'main', inputs: { source: 'inputTex' } }] })),
    reg(testDefinition('portableInferredStarter')),
    reg(testDefinition('portableOtherInput', { passes: [{ program: 'main', inputs: { source: 'o8', b: 'inputTex2', c: 'global_o0' } }] }))
  ])
  const invalid = [null, [], 'definition', 42, true, testDefinition('bad-name'), testDefinition('portableInvalid', { namespace: 'synth' }),
    testDefinition('portableInvalid', { passes: [] }), testDefinition('portableInvalid', { passes: [null] }),
    testDefinition('portableInvalid', { passes: [{ program: 'main', inputs: { src: 42 } }] }),
    testDefinition('portableInvalid', { passes: [{ program: 'main', outputs: null }] }),
    testDefinition('portableInvalid', { passes: [{ program: 'main', outputs: { color: '' } }] }),
    testDefinition('portableInvalid', { shaders: {} }), testDefinition('portableInvalid', { shaders: { main: { glsl: ' ' } } }),
    testDefinition('portableInvalid', { passes: [{ program: 'a' }, { program: 'b' }], shaders: { a: { glsl: 'source' }, b: { wgsl: 'source' } } }),
    testDefinition('portableInvalid', { globals: { amount: null } }), testDefinition('portableInvalid', { starter: 'false' }),
    testDefinition('portableInvalid', { paramAliases: 'bad' }),
    testDefinition('portableInvalid', { paramAliases: { old: 42 } }),
    testDefinition('portableInvalid', { paramAliases: { old: 'absent' } }),
    testDefinition('portableInvalid', { globals: { mode: { type: 'int', choices: 'abc' } } }),
    testDefinition('portableInvalid', { globals: { mode: { type: 'int', choices: { Broken: {} } } } })]
  scenario('test/invalid', [...invalid.map(d => reg(d)), reg(testDefinition('portableInvalid'))])
  const nameOnly = testDefinition('portableNameOnly')
  delete nameOnly.func
  scenario('test/bare-name', [reg(nameOnly)], [{ op: 'registerEffect', name: 'portableNameOnly', definition: { namespace: 'synth', func: 'portableNameOnly' } }])
  scenario('test/duplicate', [reg(testDefinition('portableDuplicate')), reg(testDefinition('portableDuplicate', { starter: false }), 1)])
  const choices = { mode: { type: 'int', default: 0, choices: { Choice: 1 } } }
  const reserved = []
  for (const name of ['__proto__', 'constructor', 'prototype', 'toString', 'valueOf', 'hasOwnProperty']) {
    reserved.push(reg(testDefinition(name, { globals: choices })))
    reserved.push(reg(testDefinition('portableReserved', { globals: { [name]: choices.mode } })))
    reserved.push(reg(testDefinition('portableReserved', { globals: { mode: { ...choices.mode, choices: { [name]: 1 } } } })))
  }
  scenario('test/reserved', reserved)

  // Every catalog effect, re-registered as a Portable definition.
  scenario('catalog', catalogIds().map(id => reg(catalogPortable(id))))

  // The effect validator's definitions, as given and with stub shaders.
  const defs = validatorDefinitions()
  scenario('validator', defs.map(([, d]) => reg(decode(d))))
  scenario('validator/stub-shaders', defs.map(([, d]) => withStubShaders(decode(d))).filter(Boolean).map(d => reg(d)))

  // Edge cases of every check.
  const wgsl = testDefinition('edge', { shaders: { main: { wgsl: STUB_WGSL } } })
  const edge = (overrides, func = 'edge') => reg({ ...wgsl, name: func, func, ...overrides })
  scenario('edge/identity', [
    edge({ func: null, name: 'edgeFromName' }, 'edgeFromName'),
    reg({ ...wgsl, func: undefined, name: 'edgeUndefinedFunc' }),
    edge({ func: '' }), edge({ func: 3 }), edge({ func: 'My Effect' }), edge({ func: '1abc' }), edge({ func: 'ok', name: 'not ok' }),
    reg({ ...wgsl, func: undefined, name: undefined }),
    edge({ namespace: null }), edge({ namespace: 'User' }), edge({ namespace: undefined }, 'edgeNoNamespace'),
    edge({ starter: null }), edge({ starter: 0 }), edge({ starter: true }, 'edgeStarterTrue'), edge({ starter: false }, 'edgeStarterFalse')
  ])
  scenario('edge/passes', [
    edge({ passes: {} }), edge({ passes: 'main' }), edge({ passes: [{}] }), edge({ passes: [{ program: '' }] }), edge({ passes: [{ program: 3 }] }),
    edge({ passes: [[]] }), edge({ passes: [{ program: 'main', inputs: [] }] }), edge({ passes: [{ program: 'main', inputs: { a: '  ' } }] }),
    edge({ passes: [{ program: 'main', inputs: null }] }), edge({ passes: [{ program: 'main', outputs: { a: 'outputTex', b: 3 } }] }),
    edge({ passes: [{ program: 'main', inputs: undefined, outputs: undefined }] }, 'edgeNoBindings'),
    edge({ passes: [{ program: 'missing' }] }), edge({ passes: [{ program: 'constructor' }] }), edge({ passes: [{ program: '__proto__' }] }),
    edge({ shaders: null }), edge({ shaders: [] }), edge({ shaders: 'x' }), edge({ shaders: { main: null } }), edge({ shaders: { main: 'wgsl' } }),
    edge({ shaders: { main: { wgsl: '﻿　\n' } } }), edge({ shaders: { main: { wgsl: 5 } } }),
    edge({ shaders: { main: { glsl: 'g', wgsl: 'w' } } }, 'edgeBothLanguages'),
    edge({ passes: [{ program: 'a' }, { program: 'b' }], shaders: { a: { glsl: 'g', wgsl: 'w' }, b: { glsl: 'g' } } }),
    edge({ passes: [{ program: 'a' }, { program: 'b' }], shaders: { a: { wgsl: 'w' }, b: { glsl: 'g' } } }),
    edge({ passes: [{ program: 'a' }, { program: 'a' }], shaders: { a: { wgsl: 'w' } } }, 'edgeSharedProgram')
  ])
  scenario('edge/globals', [
    edge({ globals: null }), edge({ globals: [] }), edge({ globals: { a: [] } }), edge({ globals: { a: 'float' } }),
    edge({ globals: { a: { type: 'int', choices: [] } } }), edge({ globals: { a: { type: 'int', choices: { x: NaN } } } }),
    edge({ globals: { a: { type: 'int', choices: { x: Infinity } } } }), edge({ globals: { a: { type: 'int', choices: { x: '1' } } } }),
    edge({ globals: { a: { type: 'string', choices: { x: 1 } } } }), edge({ globals: { a: { type: 'string', choices: { x: 'one', y: null } } } }, 'edgeStringChoices'),
    edge({ globals: { a: { type: 'int', choices: { x: null, 'y:': 2, 'Soft Light': 3, 'a-b': 4, '3d': 5, SoftLight: 6 } } } }, 'edgeChoiceNames'),
    edge({ globals: { a: { type: 'vec4', default: [1, 0, 0, 1], uniform: 'tint' }, b: { type: 'float', enum: 'x.y', choices: { k: 1 } } } }, 'edgeArgTypes'),
    edge({ globals: { a: { type: 'float' } }, paramAliases: { old: 'a', older: 'a' } }, 'edgeAliases'),
    edge({ globals: { a: { type: 'float' } }, paramAliases: { old: 'toString' } }),
    edge({ globals: undefined, paramAliases: { old: 'a' } }),
    edge({ globals: { a: { type: 'float' } }, paramAliases: null }),
    edge({ globals: { 7: { type: 'int', choices: { 2: 1, b: 0 } }, b: { type: 'float' } } }, 'edgeNumericKeys')
  ])
  scenario('edge/metadata', [
    edge({ hidden: true, deprecatedBy: 'other' }, 'edgeDeprecated'),
    edge({ hidden: true, deprecatedBy: 5 }, 'edgeDeprecatedNumber'),
    edge({ hidden: 1 }, 'edgeHidden'),
    edge({ description: 'd', tags: ['color'], openCategories: ['general'], defaultProgram: 'search user\nedge().write(o0)', onInit: 'x', extra: { nested: [1, { deep: true }] } }, 'edgeMetadata'),
    edge({ extra: { list: [{ ok: 1 }, { hasOwnProperty: 1 }] } }),
    edge({ extra: [{ a: { __lookupGetter__: 1 } }] }),
    edge({ globals: { a: { ui: { prototype: 1 } } } }),
    edge({ textures: { _t: { width: 8, height: 8 } }, passes: [{ program: 'main', inputs: { inputTex: 'inputTex' }, outputs: { fragColor: '_t' } }] }, 'edgeFilter')
  ])
  scenario('edge/existing', [
    edge({}, 'noise'), edge({}, 'blur'), edge({}, 'write'), edge({}, 'render'), edge({}, 'user'), edge({}, 'noise')
  ], [{ op: 'registerOp', name: 'user.blur', spec: { name: 'blur', args: [] } },
    { op: 'registerEffect', name: 'user/write', definition: { namespace: 'user', func: 'write' } }])

  // The Portable parity fixtures: registered, then their programs compiled.
  const dir = join(ROOT, 'parity', 'portable')
  if (existsSync(dir)) {
    for (const f of readdirSync(dir).filter(f => f.endsWith('.portable.json')).sort()) {
      const name = f.replace(/\.portable\.json$/, '')
      const dsl = readFileSync(join(dir, `${name}.dsl`), 'utf8')
      scenario(`fixture/${name}`, [reg(loadPortableDefinition(join(dir, f))), { op: 'compile', dsl }, { op: 'graph', dsl }])
    }
  }
  return S
}

// The registry lookups of `func` after a step.
function observe (R, func) {
  const user = `user.${func}`
  return {
    bare: describe(R.registry.getEffect(func)),
    dot: describe(R.registry.getEffect(user)),
    slash: describe(R.registry.getEffect(`user/${func}`)),
    op: R.ops.ops[user],
    enums: R.enums.default?.user?.[func],
    starter: R.validator.isStarterOp(user),
    starterBare: R.validator.isStarterOp(func),
    paramAliases: R.paramAliases.getParamAliases(user),
    effectAlias: R.effectAliases.checkEffectAlias(user)
  }
}

// The registry state at the end of a scenario.
function digest (R) {
  const all = R.registry.getAllEffects()
  const ops = R.ops.ops
  const keys = Object.keys(ops)
  const effectKeys = [...all.keys()].map(String)
  return {
    effects: [...all].map(([k, v]) => [String(k), describe(v)]),
    ops: keys,
    userOps: Object.fromEntries(keys.filter(k => k.startsWith('user.')).map(k => [k, ops[k]])),
    enums: R.enums.default,
    starters: [...keys, ...effectKeys].filter(n => R.validator.isStarterOp(n)),
    paramAliases: keys.map(k => [k, R.paramAliases.getParamAliases(k)]).filter(([, a]) => Object.keys(a).length),
    effectAliases: keys.map(k => [k, R.effectAliases.checkEffectAlias(k)]).filter(([, a]) => a !== null),
    loaded: R.renderers.flatMap(r => [...r.loadedEffects.keys()])
  }
}

// A step's probe name: the Portable function name the definition asks for.
function probeName (def) {
  const func = def && typeof def === 'object' && !Array.isArray(def) ? (def.func ?? def.name) : undefined
  return typeof func === 'string' && /^[a-zA-Z_][a-zA-Z0-9_]*$/.test(func) && !PROTO_NAMES.includes(func) && func !== 'prototype' ? func : null
}

async function runScenario (s) {
  const R = await loadReference()
  const renderer = i => (R.renderers[i] ??= new R.mod.CanvasRenderer())
  renderer(0)
  for (const step of s.setup) {
    if (step.op === 'mergeIntoEnums') await R.enums.mergeIntoEnums(decode(step.source))
    else if (step.op === 'registerEffect') R.registry.registerEffect(step.name, decode(step.definition))
    else if (step.op === 'registerOp') R.ops.registerOp(step.name, decode(step.spec))
    else throw new Error(`unknown setup op ${step.op}`)
  }
  const outcomes = []
  let lastRenderer = null
  for (const step of s.steps) {
    let outcome
    if (step.op === 'register') {
      const r = renderer(step.renderer)
      try {
        const effect = await r.registerPortableEffect(decode(step.definition))
        lastRenderer = r
        outcome = { result: encode({ namespace: effect.namespace, name: effect.name, instance: { ...effect.instance } }) }
      } catch (error) {
        outcome = { error: thrownRecord(error) }
      }
      if (step.probe) outcome.observed = encode(observe(R, step.probe))
    } else if (step.op === 'compile' || step.op === 'graph') {
      try {
        const value = step.op === 'compile' ? runStage(R.mod, 'validated', step.dsl) : normalizeGraph(R.mod.compileGraph(step.dsl))
        outcome = { result: encode(value) }
      } catch (error) {
        outcome = { error: thrownRecord(error) }
      }
    } else if (step.op === 'resolveEnumValue') {
      outcome = { result: encode((lastRenderer || renderer(0)).resolveEnumValue(step.path)) }
    } else {
      throw new Error(`unknown step ${step.op}`)
    }
    outcomes.push(outcome)
  }
  return { steps: outcomes, state: encode(digest(R)) }
}

async function runPortable (casesPath, expectedPath) {
  // Validator definitions that are Effect instances decode as instances.
  useEffectClass((await refModule('runtime/effect.js')).Effect)
  const scenarios = portableScenarios()
  const resolved = scenarios.map(s => ({
    id: s.id,
    op: 'scenario',
    category: 'portable',
    setup: s.setup.map(step => Object.fromEntries(Object.entries(step).map(([k, v]) => [k, k === 'op' || k === 'name' ? v : encode(v)]))),
    steps: s.steps.map(step => step.op === 'register'
      ? { op: 'register', definition: encode(step.definition), renderer: step.renderer, probe: probeName(step.definition) }
      : step)
  }))
  // One fresh realm per scenario: a worker thread imports its own module graph.
  const results = []
  for (const s of resolved) {
    results.push(await new Promise((resolveResult, reject) => {
      const worker = new Worker(fileURLToPath(import.meta.url), { workerData: s, env: process.env, stderr: true })
      worker.stderr.on('data', d => process.stderr.write(d))
      worker.once('message', resolveResult)
      worker.once('error', reject)
      worker.once('exit', code => { if (code) reject(new Error(`scenario ${s.id}: worker exited with ${code}`)) })
    }))
  }
  writeFileSync(casesPath, resolved.map(s => JSON.stringify(s)).join('\n') + '\n')
  writeFileSync(expectedPath, resolved.map((s, i) => JSON.stringify({ id: s.id, category: 'portable', result: results[i] })).join('\n') + '\n')
  const steps = resolved.reduce((n, s) => n + s.steps.length, 0)
  process.stdout.write(JSON.stringify({ cases: resolved.length, steps }) + '\n')
}

async function main () {
  const argv = process.argv.slice(2)
  const command = argv.shift()
  let cases = null
  let expected = null
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--cases') cases = argv[++i]
    else if (argv[i] === '--expected') expected = argv[++i]
  }
  if (!['api', 'portable'].includes(command) || !cases || !expected) {
    console.error('usage: node tools/reference-api.mjs api|portable --cases C.jsonl --expected E.jsonl')
    process.exit(2)
  }
  if (command === 'api') await runApi(cases, expected)
  else await runPortable(cases, expected)
}

if (!isMainThread) {
  runScenario(workerData).then(r => parentPort.postMessage(r), err => {
    process.stderr.write(`[reference-api] scenario ${workerData.id} FAILED: ${err?.stack || err}\n`)
    process.exit(1)
  })
} else if (process.argv[1] && basename(process.argv[1]) === 'reference-api.mjs') {
  main().catch(err => {
    console.error(`[reference-api] FAILED: ${err?.stack || err}`)
    process.exit(1)
  })
}
