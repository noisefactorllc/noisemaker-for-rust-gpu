// Seeded random effects and validated plans for the expander differential
// tests (src/expander/tests.rs). NM_FUZZ_SEED and NM_FUZZ_COUNT select the
// sequence; the same seed always generates the same cases.
const U = '\u0000undefined'
let s = (+process.env.NM_FUZZ_SEED || 1) >>> 0
const rnd = () => { s |= 0; s = s + 0x6D2B79F5 | 0; let t = Math.imul(s ^ s >>> 15, 1 | s); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296 }
const pick = a => a[Math.floor(rnd() * a.length)]
const chance = p => rnd() < p
const int = (a, b) => a + Math.floor(rnd() * (b - a + 1))
const N = +process.env.NM_FUZZ_COUNT || 200
const GLOBAL_NAMES = ['g0', 'g1', 'g2', 'g3', 'volumeSize', 'stateSize', 'zoom', 'palette', 'tex', 'mix']
const UNIFORMS = ['u0', 'u1', 'volumeSize', 'stateSize', 'zoom', 'palette', 'mixAmt', 'texActive', 'paletteOffset', 'paletteMode']
const MEMBER_DEFAULTS = ['channel.g', 'oscKind.saw', 'palette.brushedMetal', 'nope.path', 'palette', 'channel.r', 'midiZone.upper', 'channel.r.value', 'audioBand']
const SURF_DEFAULTS = ['none', 'inputTex', 'inputColor', 'o2', 'vol1', 'global_shared', 'global_xyz', 'localTex', 'xyz3']
const IN_REFS = ['inputTex', 'inputTex', 'inputTex3d', 'inputGeo', 'inputXyz', 'inputVel', 'inputRgba', 'noise', 'midiNoteGrid', 'feedback', 'selfTex', 'outputTex', 'o0', 'o3x', 'global_xyz', 'global_vel', 'global_rgba', 'global_points_trail', 'global_life_data', 'global_state', 'local1', 'local2', 'imageTex']
const OUT_REFS = ['outputTex', 'outputTex', 'outputTex', 'outputTex3d', 'outputXyz', 'outputVel', 'outputRgba', 'inputTex3d', 'inputGeo', 'inputXyz', 'inputVel', 'inputRgba', 'global_xyz', 'global_vel', 'global_state', 'feedback_o2', 'local1', 'local2']
const TEX_NAMES = ['global_xyz', 'global_vel', 'global_rgba', 'global_points_trail', 'global_life_data', 'global_state', 'global_trail', 'local1', 'local2', 'cache']
const dim = () => pick([64, 'screen', 'input', '50%', { param: pick(['stateSize', 'volumeSize', 'zoom', 'g0']), default: 32 }, { screenDivide: pick(['zoom', 'g1']), default: 4 }, { param: 'volumeSize', power: 2, default: 1024 }, undefined])
function genGlobal () {
  const t = pick(['float', 'int', 'int', 'member', 'palette', 'surface', 'boolean', 'vec3', 'string'])
  const g = { type: t }
  if (t === 'float' || t === 'int') { g.default = int(-2, 9); if (chance(0.5)) { g.min = int(-5, 0); g.max = int(1, 10) } if (chance(0.3)) g.choices = { a: 0, b: 1 } }
  if (t === 'member') g.default = pick(MEMBER_DEFAULTS)
  if (t === 'palette') g.default = pick([0, 1, 3, 55, 56])
  if (t === 'surface' && chance(0.85) && typeof g.default !== 'string') g.default = 'none'
  if (t === 'surface') { g.default = pick([...SURF_DEFAULTS, undefined]); if (chance(0.3)) g.colorModeUniform = pick(['texActive', 'u1', 'g0']) }
  if (t === 'boolean') g.default = chance(0.5)
  if (t === 'vec3') g.default = [0.5, 0.25, 1]
  if (t === 'string') g.default = pick(['o1', 'abc', 'global_x'])
  if (g.default === undefined) delete g.default
  if (chance(0.6)) g.uniform = pick(UNIFORMS)
  if (chance(0.25) && t !== 'surface') g.define = pick(['DA', 'DB', 'DC'])
  return g
}
function genEffect (i) {
  const d = { name: `E${i}`, namespace: 'test', func: `e${i}` }
  if (chance(0.1)) delete d.func
  if (chance(0.1)) delete d.namespace
  const globals = {}
  for (let k = 0, n = int(0, 5); k < n; k++) globals[pick(GLOBAL_NAMES)] = genGlobal()
  if (Object.keys(globals).length || chance(0.5)) d.globals = globals
  if (chance(0.35)) { d.textures = {}; for (let k = 0, n = int(1, 3); k < n; k++) { const sp = { width: dim(), height: dim() }; if (chance(0.3)) sp.format = 'rgba32f'; if (chance(0.2)) sp.mipmaps = chance(0.5); if (chance(0.2)) sp.persistent = true; d.textures[pick(TEX_NAMES)] = sp } }
  if (chance(0.2)) { d.textures3d = {}; for (let k = 0, n = int(1, 2); k < n; k++) { const sp = { width: dim(), height: dim() }; if (chance(0.5)) sp.depth = pick([16, { param: 'volumeSize' }]); if (chance(0.3)) sp.filter = 'linear'; d.textures3d[pick(TEX_NAMES)] = sp } }
  for (const [p, vals] of [['outputTex', ['inputTex', 'global_view', 'shown']], ['outputTex3d', ['inputTex3d', 'global_state', 'cache']], ['outputGeo', ['inputGeo', 'geoBuffer']], ['outputXyz', ['inputXyz', 'global_xyz', 'posTex']], ['outputVel', ['inputVel', 'global_vel', 'velTex']], ['outputRgba', ['inputRgba', 'global_rgba', 'colTex']]]) if (chance(0.12)) d[p] = pick(vals)
  if (chance(0.08)) d.externalTexture = 'imageTex'
  const progs = ['p0', 'p1', 'p2']
  if (chance(0.95)) {
    d.passes = []
    for (let k = 0, n = int(1, 3); k < n; k++) {
      const pd = { program: pick(progs) }
      if (chance(0.9)) { pd.inputs = {}; for (let j = 0, m = int(0, 3); j < m; j++) pd.inputs[pick(['a', 'b', 'c', 'inputTex', 'src'])] = chance(0.12) ? pick(GLOBAL_NAMES) : pick(IN_REFS) }
      if (chance(0.95)) { pd.outputs = {}; for (let j = 0, m = int(1, 2); j < m; j++) pd.outputs[pick(['fragColor', 'color', 'o2', 'g'])] = pick(OUT_REFS) }
      if (chance(0.3)) { pd.uniforms = {}; for (let j = 0, m = int(1, 3); j < m; j++) pd.uniforms[pick([...UNIFORMS, 'x'])] = chance(0.3) ? int(0, 4) : pick([...GLOBAL_NAMES, ...UNIFORMS, 'nothing']) }
      if (chance(0.15)) pd.defines = { [pick(['M', 'a_b', 'Z'])]: int(0, 2), [pick(['K', 'k'])]: chance(0.5) }
      if (chance(0.15)) pd.conditions = { runIf: [{ uniform: pick(UNIFORMS), equals: 1 }], ...(chance(0.5) ? { skipIf: [{ uniform: pick(UNIFORMS), equals: 0 }] } : {}) }
      for (const f of ['entryPoint', 'drawMode', 'count', 'blend', 'name', 'type', 'clear', 'viewport', 'repeat']) if (chance(0.08)) pd[f] = pick([1, 'x', true, { x: 1 }])
      d.passes.push(pd)
    }
  }
  if (chance(0.85)) { d.shaders = {}; for (const p of progs) if (chance(0.8)) d.shaders[p] = { wgsl: `// ${p}`, ...(chance(0.3) ? { glsl: 'g' } : {}) } }
  if (chance(0.2)) d.uniformLayout = { u0: { slot: 0 } }
  if (chance(0.15)) d.uniformLayouts = { p0: { u1: { slot: 1 } } }
  return d
}
function argValue (temps) {
  const r = rnd()
  if (r < 0.25) return int(-1, 6)
  if (r < 0.32) return pick(['o3', 'global_q', 'none', 'plain', 'vol2', 'channel.b', 'oscKind.tri', 'nope'])
  if (r < 0.55) { const kind = pick(['temp', 'output', 'source', 'vol', 'geo', 'xyz', 'vel', 'rgba', 'pipeline', 'mesh', 'state', 'feedback']); if (kind === 'temp') return { kind, index: temps.length && chance(0.8) ? pick(temps) : int(0, 30) }; return { kind, name: kind === 'pipeline' ? pick(['inputTex', 'inputColor', 'x']) : pick(['o1', 'none', 'vol0', 'xyz1', 'geo2']) } }
  if (r < 0.62) return { value: int(0, 3) }
  if (r < 0.66) return { _varRef: 'v', value: int(0, 3) }
  if (r < 0.70) return null
  if (r < 0.74) return U
  if (r < 0.78) return { type: 'Oscillator', oscType: 0, min: 0, max: 1 }
  if (r < 0.82) return chance(0.5)
  if (r < 0.86) return [1, 2, 3]
  return chance(0.1) ? int(0, 2) + 0.5 : int(0, 3)
}
function genCase (ci, nEff) {
  let temp = 0
  const plans = []
  for (let p = 0, np = int(1, 4); p < np; p++) {
    const chain = []
    const temps = []
    let prev = null
    for (let k = 0, n = int(1, 6); k < n; k++) {
      const r = rnd()
      let st
      const t = temp++
      if (r < 0.08) st = { op: '_read', args: { tex: pick([{ kind: 'output', name: pick(['o0', 'o1']) }, { kind: 'xyz', name: 'xyz0' }]) }, from: null, temp: t, builtin: true }
      else if (r < 0.15) st = { op: '_read3d', args: { tex3d: pick([{ kind: 'vol', name: pick(['vol0', 'vol1', 'vol2']) }, { kind: 'tex3d', name: 'plainVol' }]), geo: { kind: 'geo', name: pick(['geo0', 'geo1']) } }, from: null, temp: t, builtin: true }
      else if (r < 0.25) st = { op: '_write', args: { tex: { kind: pick(['output', 'output', 'xyz', 'mesh']), name: pick(['o0', 'o1', 'o2', 'none']) } }, from: prev, temp: t, builtin: true }
      else if (r < 0.32) st = { op: '_write3d', args: { tex3d: { kind: 'vol', name: pick(['vol0', 'vol1', 'vol2', 'none']) }, geo: { kind: 'geo', name: pick(['geo0', 'geo1', 'none']) } }, from: prev, temp: t, builtin: true }
      else if (r < 0.36) st = { op: pick(['_subchain_begin', '_subchain_end']), args: { name: 's', id: null }, from: prev, temp: t, builtin: true }
      else {
        const args = {}
        for (let j = 0, m = int(0, 4); j < m; j++) args[pick([...GLOBAL_NAMES, 'other', 'tex2'])] = argValue(temps)
        if (chance(0.06)) args._skip = pick([true, 1])
        st = { op: chance(0.01) ? 'test.unknown' : `test.e${int(0, nEff - 1)}`, args, from: chance(0.85) ? prev : null, temp: t }
        if (chance(0.05)) delete st.args
      }
      chain.push(st)
      temps.push(t)
      prev = t
    }
    const write = pick([{ kind: 'output', name: pick(['o0', 'o1', 'o2']) }, { kind: 'output', name: pick(['o0', 'o1']) }, null, { kind: 'feedback', name: 'o3' }, 'o4'])
    plans.push({ chain, write, write3d: null, final: prev, states: [] })
  }
  if (chance(0.03)) plans.push({ type: 'Branch', cond: true, then: [], elif: [], else: [] })
  const diagnostics = chance(0.05) ? [{ code: 'S002', message: 'w', severity: 'warning' }] : []
  const c = { name: `fuzz_${ci}`, source: `src ${ci}`, input: { plans, diagnostics, render: pick(['o0', 'o1', null, null]), vars: [], searchNamespaces: ['test'] } }
  if (chance(0.1)) c.options = { shaderOverrides: { [int(0, temp)]: { p0: { wgsl: 'override' }, pX: { wgsl: 'new' } } } }
  return c
}
const nEff = 12
const effects = []
for (let i = 0; i < nEff; i++) effects.push({ keys: [`test.e${i}`, `e${i}`], def: genEffect(i) })
const cases = []
for (let ci = 0; ci < N; ci++) cases.push(genCase(ci, nEff))
process.stdout.write(JSON.stringify({ effects, cases }))
