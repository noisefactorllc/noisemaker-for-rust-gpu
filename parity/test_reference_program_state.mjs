#!/usr/bin/env node
// test_reference_program_state.mjs — regression test for reference-program-state.mjs
// module interception under both canonical and existing system-alias paths.
//
// Creates no symbolic links. Uses an existing OS path alias (such as /tmp vs
// /private/tmp or /var vs /private/var on macOS) when NM_REFERENCE_ROOT is
// reachable through one.

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { existsSync, mkdtempSync, readFileSync, realpathSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { stableContracts } from '../tools/stable-contracts.mjs'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
if (!process.env.NM_REFERENCE_ROOT) {
  console.error('NM_REFERENCE_ROOT is not set')
  process.exit(2)
}

const configuredRoot = resolve(process.env.NM_REFERENCE_ROOT)
const canonicalRoot = realpathSync(configuredRoot)

function findExistingAlias (configured, canonical) {
  if (configured !== canonical) return configured
  const systemAliases = [
    ['/tmp', '/private/tmp'],
    ['/var', '/private/var'],
    ['/etc', '/private/etc']
  ]
  for (const [aliasPrefix, realPrefix] of systemAliases) {
    if (!canonical.startsWith(`${realPrefix}/`) || !existsSync(aliasPrefix)) continue
    try {
      if (realpathSync(aliasPrefix) !== realPrefix) continue
    } catch {
      continue
    }
    const candidate = `${aliasPrefix}${canonical.slice(realPrefix.length)}`
    if (existsSync(candidate) && realpathSync(candidate) === canonical) return candidate
  }
  return null
}

function runReference (refRoot, suiteRel, extraArgs, workDir, tag) {
  const resolved = join(workDir, `${tag}.resolved.jsonl`)
  const expected = join(workDir, `${tag}.expected.jsonl`)
  const res = spawnSync(
    'node',
    [
      '--max-old-space-size=8192',
      join(ROOT, 'tools', 'reference-program-state.mjs'),
      'run',
      join(ROOT, suiteRel),
      '--resolved', resolved,
      '--expected', expected,
      ...extraArgs
    ],
    {
      cwd: ROOT,
      env: { ...process.env, NM_REFERENCE_ROOT: refRoot },
      encoding: 'utf8',
      maxBuffer: 1 << 26
    }
  )
  return { res, resolved, expected }
}

const work = mkdtempSync(join(tmpdir(), 'nm-test-ref-ps-'))
try {
  const aliasRoot = findExistingAlias(configuredRoot, canonicalRoot)

  // 1. Canonical path spelling exits 0 on api.json.
  const canonRun = runReference(canonicalRoot, 'parity/program-state/api.json', [], work, 'canonical')
  assert.equal(canonRun.res.status, 0, `canonical api.json failed: ${canonRun.res.stderr}`)
  const canonSummary = JSON.parse(canonRun.res.stdout.trim())
  assert.ok(canonSummary.scenarios > 0 && canonSummary.records > 0, `unexpected summary: ${canonRun.res.stdout}`)

  // 2. Existing system-alias path spelling exits 0 and matches byte-for-byte.
  if (aliasRoot) {
    assert.notEqual(aliasRoot, canonicalRoot)
    assert.equal(realpathSync(aliasRoot), canonicalRoot)
    const aliasRun = runReference(aliasRoot, 'parity/program-state/api.json', [], work, 'alias')
    assert.equal(aliasRun.res.status, 0, `alias api.json failed (${aliasRoot}): ${aliasRun.res.stderr}`)
    const aliasSummary = JSON.parse(aliasRun.res.stdout.trim())
    assert.deepEqual(aliasSummary, canonSummary)
    assert.equal(
      readFileSync(aliasRun.resolved, 'utf8'),
      readFileSync(canonRun.resolved, 'utf8'),
      'resolved scenarios differ between alias and canonical reference paths'
    )
    assert.equal(
      readFileSync(aliasRun.expected, 'utf8'),
      readFileSync(canonRun.expected, 'utf8'),
      'expected records differ between alias and canonical reference paths'
    )
  }

  // 3. Stable negative compiler fixtures retain their expected SyntaxErrors.
  const contractRoot = aliasRoot || canonicalRoot
  const stableRun = runReference(
    contractRoot,
    'parity/program-state/fixtures.json',
    ['--only', 'stable-contract/'],
    work,
    'stable'
  )
  assert.equal(stableRun.res.status, 0, `stable-contract fixtures failed: ${stableRun.res.stderr}`)
  const records = readFileSync(stableRun.expected, 'utf8')
    .trim()
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
  for (const contract of stableContracts) {
    const rel = contract.path.replace(/^parity\//, '').replace(/\.dsl$/, '')
    const scenarioName = `fixtures:stable-contract/${rel}`
    const matching = records.filter(r => r.s === scenarioName)
    assert.ok(matching.length > 0, `missing records for ${scenarioName}`)
    const hasExpectedError = matching.some(r => {
      const text = JSON.stringify(r)
      return text.includes(contract.error.name) && text.includes(contract.error.message)
    })
    assert.ok(
      hasExpectedError,
      `expected ${contract.error.name}: ${contract.error.message} in ${scenarioName}`
    )
  }

  // 4. Direct module-identity helper check (covers both matching alias and non-matching sibling).
  const mod = await import(pathToFileURL(join(ROOT, 'tools', 'reference-program-state.mjs')).href)
  assert.equal(typeof mod.isSameCanonicalModule, 'function', 'isSameCanonicalModule helper must be exported')
  const langIndexCanonical = realpathSync(join(canonicalRoot, 'shaders', 'src', 'lang', 'index.js'))
  const langEnumsCanonical = realpathSync(join(canonicalRoot, 'shaders', 'src', 'lang', 'enums.js'))
  assert.equal(mod.isSameCanonicalModule(pathToFileURL(langIndexCanonical).href, langIndexCanonical), true)
  assert.equal(mod.isSameCanonicalModule(pathToFileURL(langEnumsCanonical).href, langIndexCanonical), false)
  if (aliasRoot) {
    const langIndexAlias = join(aliasRoot, 'shaders', 'src', 'lang', 'index.js')
    assert.equal(mod.isSameCanonicalModule(pathToFileURL(langIndexCanonical).href, langIndexAlias), true)
    assert.equal(mod.isSameCanonicalModule(pathToFileURL(langIndexAlias).href, langIndexCanonical), true)
  }

  console.log(
    `test_reference_program_state: ok (canonical=${canonicalRoot}, alias=${aliasRoot || 'none'}, stableContracts=${stableContracts.length})`
  )
} finally {
  rmSync(work, { recursive: true, force: true })
}
