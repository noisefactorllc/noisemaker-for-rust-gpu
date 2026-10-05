// Synthetic effects and validated plans for the expander differential tests
// (src/expander/tests.rs). They reach the expander paths the effect catalog
// cannot: agent-state lanes, textures3d, outputTex/outputTex3d/outputXyz/...
// properties, member-typed defines, pass-level uniforms falling back to global
// defaults, surface defaults of every form, feedback targets, plan shapes the
// validator never emits, shader overrides, and the TypeErrors the reference
// raises on malformed definitions and plans.
const U = '\u0000undefined'
const sh = (...progs) => Object.fromEntries(progs.map(p => [p, { wgsl: `// ${p}`, glsl: `// ${p} glsl` }]))
const effects = [
  { keys: ['test.agents', 'agents'], def: {
    name: 'Agents', namespace: 'test', func: 'agents',
    globals: { count: { type: 'int', default: 4, uniform: 'count', min: 1, max: 8 }, stateSize: { type: 'int', default: 64, uniform: 'stateSize' } },
    textures: {
      global_xyz: { width: { param: 'stateSize', default: 64 }, height: { param: 'stateSize', default: 64 }, format: 'rgba32f' },
      global_vel: { width: { param: 'stateSize', default: 64 }, height: { param: 'stateSize', default: 64 } },
      trail: { width: 'screen', height: 'screen', mipmaps: true, persistent: false },
      sized: { width: { param: 'stateSize', default: 8 }, height: { screenDivide: 'stateSize', default: 2 } }
    },
    passes: [
      { program: 'init', inputs: { xyzIn: 'inputXyz', velIn: 'inputVel', rgbaIn: 'inputRgba', n: 'noise' }, outputs: { pos: 'outputXyz', vel: 'outputVel', col: 'outputRgba' } },
      { program: 'draw', inputs: { pos: 'global_xyz', vel: 'global_vel', trail: 'trail', life: 'global_life_data' }, outputs: { color: 'outputTex', t: 'trail' } }
    ],
    shaders: sh('init', 'draw') } },
  { keys: ['test.agentFilter'], def: {
    name: 'AgentFilter', namespace: 'test', func: 'agentFilter',
    passes: [{ program: 'step', inputs: { pos: 'inputXyz', vel: 'inputVel', col: 'inputRgba', src: 'inputTex', geo: 'inputGeo', v3: 'inputTex3d' },
      outputs: { a: 'inputXyz', b: 'inputVel', c: 'inputRgba', g: 'inputGeo', v: 'inputTex3d', color: 'outputTex' } }],
    shaders: sh('step') } },
  { keys: ['test.agentProps'], def: {
    name: 'AgentProps', namespace: 'test', func: 'agentProps',
    outputXyz: 'inputXyz', outputVel: 'inputVel', outputRgba: 'inputRgba',
    passes: [{ program: 'p', inputs: { src: 'inputTex' }, outputs: { color: 'outputTex' } }], shaders: sh('p') } },
  { keys: ['test.agentMake'], def: {
    name: 'AgentMake', namespace: 'test', func: 'agentMake',
    outputXyz: 'global_xyz', outputVel: 'velTex', outputRgba: 'global_points_trail', outputTex: 'shown',
    passes: [{ program: 'p', inputs: {}, outputs: { a: 'velTex' } }], shaders: sh('p') } },
  { keys: ['test.agentMake2'], def: {
    name: 'AgentMake2', namespace: 'test', func: 'agentMake2',
    outputXyz: 'posTex', outputVel: 'global_vel', outputRgba: 'colTex',
    passes: [{ program: 'p', inputs: { x: 'inputXyz' }, outputs: { a: 'posTex' } }], shaders: sh('p') } },
  { keys: ['test.vol'], def: {
    name: 'Vol', namespace: 'test', func: 'vol',
    globals: { volumeSize: { type: 'int', default: 32, uniform: 'volumeSize' } },
    textures3d: {
      cache: { width: 32, height: 32, depth: 32, format: 'rgba16f', filter: 'nearest' },
      global_state: { width: { param: 'volumeSize', default: 32 }, height: { param: 'volumeSize', power: 2, default: 1024 } },
      global_xyz: { width: 8 },
      bare: {}
    },
    passes: [{ program: 'gen', outputs: { color: 'outputTex3d', geo: 'inputGeo', back: 'inputTex3d', fb: 'feedback_o3' } }],
    outputTex: 'global_view', outputGeo: 'geoTex',
    shaders: sh('gen') } },
  { keys: ['test.passOutputTex'], def: { name: 'P', namespace: 'test', func: 'passOutputTex', outputTex: 'inputTex', outputTex3d: 'inputTex3d', outputGeo: 'inputGeo', passes: [{ program: 'p', inputs: { src: 'inputTex' }, outputs: { o: 'scratch' } }], shaders: sh('p') } },
  { keys: ['test.localOutputTex'], def: { name: 'L', namespace: 'test', func: 'localOutputTex', outputTex: 'shown', outputTex3d: 'cache3', passes: [{ program: 'p', outputs: { o: 'shown' } }], shaders: sh('p') } },
  { keys: ['test.feedbackUser'], def: { name: 'F', namespace: 'test', func: 'feedbackUser', passes: [{ program: 'p', inputs: { fb: 'feedback', self: 'selfTex', own: 'outputTex', o2: 'o2', o9x: 'o9x', ox: 'oops' }, outputs: { color: 'outputTex' } }], shaders: sh('p') } },
  { keys: ['test.globalInputs'], def: {
    name: 'G', namespace: 'test', func: 'globalInputs',
    globals: {
      a: { type: 'surface', default: 'none' }, b: { type: 'surface', default: 'inputTex' }, c: { type: 'surface', default: 'inputColor' },
      d: { type: 'surface', default: 'o2' }, e: { type: 'surface', default: 'global_shared' }, f: { type: 'surface', default: 'global_xyz' },
      g: { type: 'surface', default: 'localTex' }, s: { type: 'string', default: 'x' }, n: { type: 'surface' }
    },
    passes: [{ program: 'p', inputs: { a: 'a', b: 'b', c: 'c', d: 'd', e: 'e', f: 'f', g: 'g', sArg: 's', nArg: 'n', glob: 'global_xyz', loc: 'thing' }, outputs: { color: 'outputTex' } }],
    shaders: sh('p') } },
  { keys: ['test.badDefault'], def: { name: 'B', namespace: 'test', func: 'badDefault', globals: { k: { type: 'surface', default: 5 } }, passes: [{ program: 'p', inputs: { k: 'k' }, outputs: { color: 'outputTex' } }] } },
  { keys: ['test.nullDefault'], def: { name: 'B', namespace: 'test', func: 'nullDefault', globals: { k: { type: 'surface', default: null } }, passes: [{ program: 'p', inputs: { k: 'k' }, outputs: { color: 'outputTex' } }] } },
  { keys: ['test.arrDefault'], def: { name: 'B', namespace: 'test', func: 'arrDefault', globals: { k: { type: 'surface', default: ['o3'] } }, passes: [{ program: 'p', inputs: { k: 'k' }, outputs: { color: 'outputTex' } }] } },
  { keys: ['test.colorMode'], def: {
    name: 'C', namespace: 'test', func: 'colorMode',
    globals: {
      tex: { type: 'surface', default: 'none', colorModeUniform: 'texActive' },
      tex2: { type: 'surface', default: 'o1', colorModeUniform: 'tex2Active' },
      active: { type: 'int', default: 1, uniform: 'texActive' },
      level: { type: 'float', default: 0.5, uniform: 'level' },
      tex3: { type: 'surface', default: 'none', colorModeUniform: 'level', uniform: 'tex3u' }
    },
    passes: [{ program: 'p', inputs: { tex: 'tex', tex2: 'tex2', tex3: 'tex3' }, outputs: { color: 'outputTex' } }], shaders: sh('p') } },
  { keys: ['test.member'], def: {
    name: 'M', namespace: 'test', func: 'member',
    globals: {
      mode: { type: 'member', default: 'channel.g', define: 'MODE' },
      kind: { type: 'member', default: 'oscKind.saw', uniform: 'oscU', define: 'KIND' },
      bad: { type: 'member', default: 'nope.path', uniform: 'badU' },
      leaf: { type: 'member', default: 'palette', uniform: 'leafU' },
      zero: { type: 'member', default: 'channel.r.value', uniform: 'zeroU' },
      pal: { type: 'palette', default: 3 },
      ref: { type: 'member', default: 'midiZone.upper' },
      plain: { type: 'float', default: 2 },
      sel: { type: 'int', default: 0, uniform: 'sel', choices: { a: 0, b: 1 }, min: 0, max: 1 },
      sel2: { type: 'int', default: 0, uniform: 'sel2', choices: { a: 0 } },
      rng: { type: 'float', default: 1, min: null, uniform: 'rng' },
      flag: { type: 'boolean', default: false, define: 'FLAG' },
      vec: { type: 'vec3', default: [1, 2, 3], define: 'VEC' }
    },
    passes: [
      { program: 'm', uniforms: { refU: 'ref', plainU: 'plain', paletteOffset: 'pal', paletteMode: 'pal', k: 3, same: 'same', modeU: 'mode' }, outputs: { color: 'outputTex' },
        conditions: { runIf: [{ uniform: 'sel', equals: 1 }], skipIf: [{ uniform: 'sel2', equals: 0 }] },
        defines: { Z_LAST: 1, a_first: 2, B_mid: true, 'b-dash': 'x' }, entryPoint: 'main', workgroups: [8, 8, 1], storageBuffers: { b: 1 }, storageTextures: { t: 'x' },
        countUniform: 'count', clear: false, samplerTypes: { tex: 'nearest' }, viewport: { x: 0 }, drawMode: 'points', drawBuffers: 2, count: 10, repeat: 2, blend: true, name: 'mPass', type: 'render' },
      { program: 'unknownProgram', defines: { X: 1 }, outputs: { color: 'outputTex' } },
      { program: 'm', defines: { X: 2 }, inputs: {}, outputs: {} }
    ],
    uniformLayouts: { m: { a: 1 } }, uniformLayout: { base: 1 },
    shaders: sh('m', 'extra') } },
  { keys: ['test.noFunc'], def: { name: 'NF', passes: [{ program: 'p', outputs: { color: 'outputTex' } }], shaders: sh('p') } },
  { keys: ['test.noPasses'], def: { name: 'NP', namespace: 'test', func: 'noPasses', globals: { x: { type: 'float', default: 1, uniform: 'x' } } } },
  { keys: ['test.numTexRef'], def: { name: 'N', namespace: 'test', func: 'numTexRef', passes: [{ program: 'p', inputs: { x: 5 }, outputs: {} }] } },
  { keys: ['test.numOutRef'], def: { name: 'N', namespace: 'test', func: 'numOutRef', passes: [{ program: 'p', inputs: {}, outputs: { x: 5 } }] } },
  { keys: ['test.nullGlobal'], def: { name: 'N', namespace: 'test', func: 'nullGlobal', globals: { g: null }, passes: [] } },
  { keys: ['test.badConditions'], def: { name: 'N', namespace: 'test', func: 'badConditions', passes: [{ program: 'p', conditions: { runIf: 5 } }] } },
  { keys: ['test.nullCondition'], def: { name: 'N', namespace: 'test', func: 'nullCondition', passes: [{ program: 'p', conditions: { skipIf: [null] } }] } },
  { keys: ['test.objPasses'], def: { name: 'N', namespace: 'test', func: 'objPasses', passes: { a: 1 } } },
  { keys: ['test.strPasses'], def: { name: 'N', namespace: 'test', func: 'strPasses', passes: 'a\u{1F600}' } },
  { keys: ['test.numOutputTex'], def: { name: 'N', namespace: 'test', func: 'numOutputTex', outputTex: 7, passes: [] } },
  { keys: ['test.numOutputXyz'], def: { name: 'N', namespace: 'test', func: 'numOutputXyz', outputXyz: 7, passes: [] } },
  { keys: ['test.nullDim'], def: { name: 'N', namespace: 'test', func: 'nullDim', textures: { global_t: { width: null } }, passes: [] } },
  { keys: ['test.media'], def: { name: 'Md', namespace: 'test', func: 'media', externalTexture: 'imageTex', passes: [{ program: 'p', inputs: { imageTex: 'imageTex', again: 'imageTex' }, outputs: { color: 'outputTex' } }, { program: 'p', inputs: { imageTex: 'imageTex' }, outputs: { color: 'outputTex' } }], shaders: sh('p') } },
  { keys: ['test.volSize'], def: { name: 'VS', namespace: 'test', func: 'volSize', globals: { volumeSize: { type: 'int', default: 16, uniform: 'volumeSize' } }, textures: { cache: { width: { param: 'volumeSize', default: 16 }, height: { param: 'volumeSize', power: 2 } } }, passes: [{ program: 'p', inputs: { v: 'inputTex3d' }, outputs: { color: 'outputTex3d', g: 'geoOut' } }], outputGeo: 'geoOut', shaders: sh('p') } }
]
const step = (op, temp, from, args = {}, extra = {}) => ({ op, args, from, temp, ...extra })
const w = (temp, from, name, kind = 'output') => ({ op: '_write', args: { tex: { kind, name } }, from, temp, builtin: true })
const plan = (chain, write = { kind: 'output', name: 'o0' }, extra = {}) => ({ chain, write, write3d: null, final: chain.length && chain[chain.length - 1] ? chain[chain.length - 1].temp : null, states: [], ...extra })
const prog = (plans, render = 'o0', diagnostics = []) => ({ plans, diagnostics, render, vars: [], searchNamespaces: ['test'] })
const cases = [
  { name: 'agents_lanes', input: prog([plan([
    step('test.agents', 0, null), step('test.agentFilter', 1, 0),
    step('test.agentFilter', 2, 1, { _skip: true }),
    { op: '_subchain_begin', args: { name: 's', id: null }, from: 2, temp: 3, builtin: true },
    step('test.agentProps', 4, 3), step('test.agentMake', 5, 4), step('test.agentFilter', 6, 5),
    { op: '_subchain_end', args: { name: 's', id: null }, from: 6, temp: 7, builtin: true },
    w(8, 7, 'o0')])]) },
  { name: 'agents_two_pipelines', input: prog([
    plan([step('test.agents', 0, null, { stateSize: 32 }), step('test.agents', 1, 0, { stateSize: 16, count: 2 }), step('test.agentFilter', 2, 1), w(3, 2, 'o0')]),
    plan([step('test.agentProps', 4, null), step('test.agentMake', 5, 4), step('test.vol', 6, 5), w(7, 6, 'o1')], { kind: 'output', name: 'o1' })]) },
  { name: 'vol_paths', input: prog([
    plan([{ op: '_read3d', args: { tex3d: { kind: 'tex3d', type: 'VolRef', name: 'v1' }, geo: { kind: 'x', type: 'GeoRef', name: 'g1' } }, from: null, temp: 0, builtin: true },
      step('test.vol', 1, 0), step('test.passOutputTex', 2, 1), step('test.localOutputTex', 3, 2), step('test.volSize', 4, 3, { volumeSize: 8 }),
      { op: '_write3d', args: { tex3d: { kind: 'vol', name: 'vol2' }, geo: { kind: 'geo', name: 'geo2' } }, from: 4, temp: 5, builtin: true }], null),
    plan([{ op: '_read3d', args: { tex3d: { kind: 'tex3d', name: 'plain3d' }, geo: { kind: 'other', name: 'plainGeo' } }, from: null, temp: 6, builtin: true },
      step('test.volSize', 7, 6, { volumeSize: 4 }), w(8, 7, 'o2')], { kind: 'output', name: 'o2' }),
    plan([{ op: '_read3d', args: { tex3d: { kind: 'vol', name: 'vol2' }, geo: { kind: 'geo', name: 'geo2' } }, from: null, temp: 9, builtin: true },
      step('test.volSize', 10, 9), w(11, 10, 'o3')], { kind: 'output', name: 'o3' })]) },
  { name: 'vol_object_name', input: prog([plan([{ op: '_read3d', args: { tex3d: { kind: 'tex3d' }, geo: 'rawGeo' }, from: null, temp: 0, builtin: true }, step('test.volSize', 1, 0), w(2, 1, 'o0')])]) },
  { name: 'vol_null_size', input: prog([
    plan([step('test.volSize', 0, null, { volumeSize: null }), { op: '_write3d', args: { tex3d: { kind: 'vol', name: 'vol0' }, geo: { kind: 'geo', name: 'none' } }, from: 0, temp: 1, builtin: true }], null),
    plan([{ op: '_read3d', args: { tex3d: { kind: 'vol', name: 'vol0' }, geo: null }, from: null, temp: 2, builtin: true }, step('test.volSize', 3, 2), w(4, 3, 'o0')])]) },
  { name: 'global_inputs_defaults', input: prog([plan([step('test.agents', 0, null), step('test.globalInputs', 1, 0, {}), w(2, 1, 'o0')]), plan([{ op: '_read', args: { tex: { kind: 'xyz', name: 'xyz1' } }, from: null, temp: 3, builtin: true }, step('test.globalInputs', 4, 3, {}), w(5, 4, 'o1')], { kind: 'output', name: 'o1' })]) },
  { name: 'global_inputs_args', input: prog([plan([step('test.globalInputs', 0, null, { a: 'o3', b: 'global_q', c: 'none', d: 'plain', e: { kind: 'pipeline', name: 'inputColor' }, f: { kind: 'pipeline', name: 'other' }, g: { kind: 'feedback', name: 'o1' }, s: 'vol2', n: null }), w(1, 0, 'o0')])]) },
  { name: 'bad_default', input: prog([plan([step('test.badDefault', 0, null, {}), w(1, 0, 'o0')])]) },
  { name: 'null_default', input: prog([plan([step('test.nullDefault', 0, null, {}), w(1, 0, 'o0')])]) },
  { name: 'arr_default', input: prog([plan([step('test.arrDefault', 0, null, {}), w(1, 0, 'o0')])]) },
  { name: 'color_mode', input: prog([
    plan([step('test.colorMode', 0, null, { tex: { kind: 'output', name: 'o0' }, active: 0, level: 0.25 }), w(1, 0, 'o0')]),
    plan([step('test.colorMode', 2, null, { tex: { kind: 'output', name: 'none' }, tex2: { kind: 'output', name: 'o4' }, tex3: { kind: 'source', name: 's1' }, level: 0.75 }), w(3, 2, 'o1')], { kind: 'output', name: 'o1' }),
    plan([step('test.colorMode', 4, null, {}), w(5, 4, 'o2')], { kind: 'output', name: 'o2' }),
    plan([step('test.colorMode', 6, null), w(7, 6, 'o3')], { kind: 'output', name: 'o3' })]) },
  { name: 'member_and_pass_uniforms', input: prog([
    plan([step('test.member', 0, null, {}), w(1, 0, 'o0')]),
    plan([step('test.member', 2, null, { mode: 'oscKind.sine', kind: { value: 'channel.b' }, pal: 2, plain: 7, ref: 3, sel: 1, rng: 4, flag: true, vec: [4, 5, 6] }), w(3, 2, 'o1')], { kind: 'output', name: 'o1' }),
    plan([step('test.member', 4, null, { mode: 'bogus.path', pal: 0, kind: { value: U } }), w(5, 4, 'o2')], { kind: 'output', name: 'o2' })]) },
  { name: 'member_palette_fraction', input: prog([plan([step('test.member', 6, null, { mode: { _varRef: 'v', value: 2 }, pal: 1.5 }), w(7, 6, 'o3')], { kind: 'output', name: 'o3' })]) },
  { name: 'member_palette_ok', input: prog([plan([step('test.member', 0, null, { pal: 55, mode: null, flag: U }), w(1, 0, 'o0')])]) },
  { name: 'undefined_args', input: prog([plan([step('test.colorMode', 0, null, { level: U, active: U, tex: U }), step('test.member', 1, 0, { plain: U, sel: U, kind: U, zero: U, bad: U }), w(2, 1, 'o0')])]) },
  { name: 'agent_outputs_without_lanes', input: prog([
    plan([step('test.agentFilter', 0, null), step('test.agentMake2', 1, 0), step('test.agentFilter', 2, 1), w(3, 2, 'o0')]),
    plan([step('test.agents', 4, null), step('test.agentMake2', 5, 4), step('test.agentProps', 6, 5), step('test.agentFilter', 7, 6, { _skip: true }), w(8, 7, 'o1')], { kind: 'output', name: 'o1' })]) },
  { name: 'no_func_no_passes', input: prog([plan([step('test.noFunc', 0, null), step('test.noPasses', 1, 0, { x: 3 }), step('test.missing', 2, 1), w(3, 2, 'o0')])]) },
  { name: 'feedback_writes', input: prog([
    plan([step('test.feedbackUser', 0, null)], { kind: 'feedback', name: 'o3' }),
    plan([step('test.feedbackUser', 1, null)], 'o4'),
    plan([step('test.feedbackUser', 2, null)], { name: 'o5' }),
    plan([step('test.feedbackUser', 3, null), step('test.feedbackUser', 4, 3)], null),
    plan([step('test.feedbackUser', 5, null), w(6, 5, 'o6', 'feedback')], { kind: 'output', name: 'o6' }),
    plan([step('test.feedbackUser', 7, null)], ['arr']),
    plan([step('test.feedbackUser', 8, null), w(9, 8, 'o7')], { kind: 'output', name: 'o8' })], null) },
  { name: 'last_step_direct_write', input: prog([plan([step('test.agents', 0, null), step('test.member', 1, 0)], { kind: 'output', name: 'o2' })], null) },
  { name: 'media_steps', input: prog([plan([step('test.media', 0, null), step('test.media', 1, 0), w(2, 1, 'o0')]), plan([step('test.media', 3, null), w(4, 3, 'o1')])]) },
  { name: 'shader_overrides', input: prog([plan([step('test.member', 0, null), step('test.agents', 1, 0), w(2, 1, 'o0')])]),
    options: { shaderOverrides: { 0: { m: { wgsl: 'override m' }, extra2: { glsl: 'x' } }, 1: { init: 'notAnObject', draw: { wgsl: 'd2', uniformLayout: { z: 1 } } }, 7: { q: {} } } } },
  { name: 'type_numTexRef', input: prog([plan([step('test.numTexRef', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_numOutRef', input: prog([plan([step('test.numOutRef', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_nullGlobal', input: prog([plan([step('test.nullGlobal', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_badConditions', input: prog([plan([step('test.badConditions', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_nullCondition', input: prog([plan([step('test.nullCondition', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_objPasses', input: prog([plan([step('test.objPasses', 0, null), w(1, 0, 'o0')])]) },
  { name: 'string_passes', input: prog([plan([step('test.strPasses', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_numOutputTex', input: prog([plan([step('test.numOutputTex', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_numOutputXyz', input: prog([plan([step('test.numOutputXyz', 0, null), w(1, 0, 'o0')])]) },
  { name: 'type_nullDim', input: prog([plan([step('test.nullDim', 0, null), w(1, 0, 'o0')])]) },
  { name: 'branch_plan', input: prog([plan([step('test.noFunc', 0, null), w(1, 0, 'o0')]), { type: 'Branch', cond: true, then: [], elif: [], else: [] }]) },
  { name: 'plans_not_iterable', input: { plans: 3, diagnostics: [], render: 'o0' } },
  { name: 'plans_string', input: { plans: '', diagnostics: [], render: null } },
  { name: 'null_plan', input: prog([null]) },
  { name: 'null_step', input: prog([plan([null])]) },
  { name: 'no_input', input: null },
  { name: 'diag_warning', source: 'a☃b', input: prog([plan([step('test.noFunc', 0, null), w(1, 0, 'o0')])], 'o0', [{ code: 'S002', message: 'w', severity: 'warning' }]) },
  { name: 'diag_error', input: prog([plan([step('test.noFunc', 0, null), w(1, 0, 'o0')])], 'o0', [{ code: 'S002', message: 'w', severity: 'warning' }, { code: 'S001', message: 'e', severity: 'error' }]) },
  { name: 'diag_string', input: { plans: [], diagnostics: 'x', render: 'o0' } },
  { name: 'diag_null_entry', input: prog([], 'o0', [null]) },
  { name: 'diag_length_object', input: { plans: [], diagnostics: { length: 2 }, render: 'o0' } },
  { name: 'diag_length_string', input: { plans: [], diagnostics: { length: '0' }, render: 'o0' } },
  { name: 'diag_missing', input: { plans: [], render: 'o0' } },
  { name: 'render_object', input: prog([plan([step('test.noFunc', 0, null), w(1, 0, 'o0')])], { weird: true }) },
  { name: 'temp_args', input: prog([plan([step('test.vol', 0, null), step('test.globalInputs', 1, null, { a: { kind: 'temp', index: 0 }, b: { kind: 'temp', index: 9 }, c: { kind: 'temp', index: 1 } }), w(2, 1, 'o0')])]) },
  { name: 'step_without_from', input: prog([plan([{ op: 'test.noFunc', args: {}, temp: 0 }, w(1, 0, 'o0')])]) },
  { name: 'skip_variants', input: prog([plan([step('test.agents', 0, null), step('test.vol', 1, 0, { _skip: 1 }), step('test.vol', 2, 1, { _skip: true }), { op: '_subchain_begin', args: {}, from: 2, temp: 3, builtin: true }, { op: '_subchain_end', args: {}, from: 3, temp: 4, builtin: true }, w(5, 4, 'o0')])]) }
]
process.stdout.write(JSON.stringify({ effects, cases }))
