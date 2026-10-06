#!/usr/bin/env node
// check_effects.mjs — catalog freshness gate.
//
// Regenerates the embedded effect catalog from the reference at
// $NM_REFERENCE_ROOT into a temporary directory (tools/convert-effects.mjs) and
// requires it to be byte-identical to crates/noisemaker-effects/catalog: every
// definition, every WGSL program, the manifest, the built-in meshes, the
// palette table, the string catalogs, the demo font and the demo's default
// media image (share/img/testcard.png, also compared directly with the
// reference's demo/shaders/img/testcard.png). A stale catalog, a hand-edited
// shader or definition, or a missing file fails the gate.
//
// It also reports whether the reference checkout is at the pinned commit
// (parity/reference.json); a different commit is an error unless
// NM_ALLOW_REFERENCE_DRIFT=1 (used when checking a candidate sync).
//
// Usage: NM_REFERENCE_ROOT=/path/to/noisemaker node parity/check_effects.mjs

import { execFileSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const CATALOG = join(ROOT, 'crates', 'noisemaker-effects', 'catalog')
// Catalog files that must be byte copies of these reference files.
const REFERENCE_COPIES = [['share/img/testcard.png', 'demo/shaders/img/testcard.png']]

if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set; point it at a checkout of the noisemaker reference repository')
  process.exit(2)
}

function walk (dir) {
  const out = []
  for (const entry of readdirSync(dir).sort()) {
    const p = join(dir, entry)
    if (statSync(p).isDirectory()) out.push(...walk(p))
    else out.push(p)
  }
  return out
}

const pin = JSON.parse(readFileSync(join(ROOT, 'parity', 'reference.json'), 'utf8')).commit
let head = null
try {
  head = execFileSync('git', ['-C', process.env.NM_REFERENCE_ROOT, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()
} catch { /* not a git checkout */ }
let failed = false
if (head !== pin) {
  const msg = `reference checkout is at ${head ?? 'an unknown revision'}, the catalog pin is ${pin}`
  if (process.env.NM_ALLOW_REFERENCE_DRIFT === '1') console.warn(`check_effects: note: ${msg}`)
  else { console.error(`check_effects: ${msg}`); failed = true }
}

const work = mkdtempSync(join(tmpdir(), 'nm-check-effects-'))
try {
  execFileSync('node', [join(ROOT, 'tools', 'convert-effects.mjs'), '--out', work], { stdio: ['ignore', 'ignore', 'inherit'] })
  const expected = walk(work).map(p => relative(work, p))
  const actual = walk(CATALOG).map(p => relative(CATALOG, p))
  const missing = expected.filter(p => !actual.includes(p))
  const extra = actual.filter(p => !expected.includes(p))
  const differ = expected.filter(p => actual.includes(p) &&
    !readFileSync(join(work, p)).equals(readFileSync(join(CATALOG, p))))
  for (const p of missing) console.error(`check_effects: missing ${p}`)
  for (const p of extra) console.error(`check_effects: unexpected ${p}`)
  for (const p of differ) console.error(`check_effects: differs ${p}`)
  if (missing.length || extra.length || differ.length) failed = true
  console.log(`CATALOG: ${expected.length} files, ${missing.length} missing, ${extra.length} unexpected, ${differ.length} differ`)
  for (const [catalogPath, referencePath] of REFERENCE_COPIES) {
    const ours = join(CATALOG, catalogPath)
    const theirs = join(process.env.NM_REFERENCE_ROOT, referencePath)
    if (!existsSync(ours) || !existsSync(theirs) || !readFileSync(ours).equals(readFileSync(theirs))) {
      console.error(`check_effects: ${catalogPath} is not a byte copy of the reference's ${referencePath}`)
      failed = true
    }
  }
} finally {
  rmSync(work, { recursive: true, force: true })
}
process.exit(failed ? 1 : 0)
