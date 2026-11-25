// The 2.x branch's workflow contract.
//
// The test jobs run on every push to 2.x, and every release, notice or
// dispatch job runs only for main. A job runs only for main when its
// condition names refs/heads/main or a workflow_run head branch of main, or
// when every trigger of its workflow is a schedule (which runs only from the
// default branch) or a push or workflow_run filtered to main alone.
//
// Usage: node --test scripts/workflow-contract.test.mjs

import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { join, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

const WORKFLOWS = join(resolve(fileURLToPath(import.meta.url), '..', '..'), '.github', 'workflows')

// Workflows whose jobs are test jobs, and must run on every 2.x push.
const TEST_WORKFLOWS = ['tests.yml']
// Workflows that never run on push, by their test jobs. Each job's condition
// must admit 2.x, for a dispatch on 2.x.
const DISPATCHED_TEST_JOBS = { 'parity.yml': ['sweep'] }
// Workflows whose jobs release, notify or dispatch, and must run only for main.
const RELEASE_WORKFLOWS = ['export-kit.yml']

const MAIN_ONLY_CONDITIONS = [
  /github\.ref\s*==\s*'refs\/heads\/main'/,
  /github\.event\.workflow_run\.head_branch\s*==\s*'main'/
]

const unquote = s => s.trim().replace(/^['"]+|['"]+$/g, '')
const indentOf = s => s.length - s.replace(/^ +/, '').length

function lines (name) {
  const result = []
  for (let raw of readFileSync(join(WORKFLOWS, name), 'utf8').split(/\r?\n/)) {
    raw = raw.replace(/\s+#[^'"]*$/, '')
    if (raw.trim() && !raw.trimStart().startsWith('#')) {
      result.push([indentOf(raw), raw.trim()])
    }
  }
  return result
}

function partition (text, sep) {
  const at = text.indexOf(sep)
  return at < 0 ? [text, ''] : [text.slice(0, at), text.slice(at + sep.length)]
}

// [key, value, index] for each key one level below lines[start].
function children (all, start, indent) {
  const found = []
  let child = null
  for (let index = start + 1; index < all.length; index++) {
    const [depth, text] = all[index]
    if (depth <= indent) break
    if (child === null) child = depth
    if (depth === child && !text.startsWith('- ')) {
      const [key, value] = partition(text, ':')
      found.push([unquote(key), value.trim(), index])
    }
  }
  return found
}

// The scalar at lines[index]: inline, or a folded or literal block.
function block (all, index) {
  const [depth, text] = all[index]
  const value = partition(text, ':')[1].trim()
  if (!['>', '|'].includes(value.slice(0, 1))) return value
  const parts = []
  for (let later = index + 1; later < all.length; later++) {
    if (all[later][0] <= depth) break
    parts.push(all[later][1])
  }
  return parts.join(' ')
}

function top (all, key) {
  const index = all.findIndex(([depth, text]) => depth === 0 && unquote(partition(text, ':')[0]) === key)
  if (index < 0) throw new assert.AssertionError({ message: `no top-level '${key}'` })
  return index
}

// Map each trigger to its branch filter (null when it has none).
function triggers (name) {
  const all = lines(name)
  const result = {}
  for (const [event, , index] of children(all, top(all, 'on'), 0)) {
    let branches = null
    for (const [key, inline, child] of children(all, index, all[index][0])) {
      if (key !== 'branches') continue
      if (inline.startsWith('[')) {
        branches = inline.replace(/^\[+|\]+$/g, '').split(',').filter(b => b.trim()).map(unquote)
      } else {
        branches = []
        for (let later = child + 1; later < all.length; later++) {
          const [depth, text] = all[later]
          if (depth <= all[child][0] && !text.startsWith('- ')) break
          if (text.startsWith('- ')) branches.push(unquote(text.slice(2)))
        }
      }
    }
    result[event] = branches
  }
  return result
}

// Map each job to its condition ('' when it has none).
function jobs (name) {
  const all = lines(name)
  const result = {}
  for (const [job, , index] of children(all, top(all, 'jobs'), 0)) {
    let condition = ''
    for (const [key, , child] of children(all, index, all[index][0])) {
      if (key === 'if') condition = block(all, child)
    }
    result[job] = condition
  }
  return result
}

function runsOnlyForMain (name, condition) {
  // A condition with an alternative can hold outside main, so only the
  // triggers can make such a job main-only.
  if (!condition.includes('||') && MAIN_ONLY_CONDITIONS.some(pattern => pattern.test(condition))) {
    return true
  }
  for (const [event, branches] of Object.entries(triggers(name))) {
    if (event === 'schedule') continue
    if (['push', 'workflow_run'].includes(event) && branches?.length === 1 && branches[0] === 'main') continue
    return false
  }
  return true
}

function admits2x (condition) {
  return !condition.includes('refs/heads/main') || condition.includes('refs/heads/2.x')
}

for (const name of TEST_WORKFLOWS) {
  test(`the test jobs of ${name} run on every 2.x push`, () => {
    const found = triggers(name)
    // A push trigger with no branch filter runs on every branch.
    assert.ok('push' in found && (found.push === null || found.push.includes('2.x')),
      `${name} does not run on a 2.x push`)
    for (const [job, condition] of Object.entries(jobs(name))) {
      assert.ok(admits2x(condition), `${name} job ${job} runs only for main: ${condition}`)
    }
  })
}

for (const [name, names] of Object.entries(DISPATCHED_TEST_JOBS)) {
  test(`the test jobs of ${name} admit a 2.x dispatch, and never run on push`, () => {
    const found = triggers(name)
    assert.ok(!('push' in found), `${name} runs on push`)
    assert.ok('schedule' in found && 'workflow_dispatch' in found,
      `${name} lost its schedule or dispatch trigger`)
    const conditions = jobs(name)
    for (const job of names) {
      assert.ok(job in conditions, `${name} has no job ${job}`)
      assert.ok(admits2x(conditions[job]), `${name} job ${job} runs only for main: ${conditions[job]}`)
    }
  })
}

for (const name of RELEASE_WORKFLOWS) {
  test(`every release, notice or dispatch job of ${name} runs only for main`, () => {
    const found = jobs(name)
    assert.ok(Object.keys(found).length > 0, `${name} has no jobs`)
    for (const [job, condition] of Object.entries(found)) {
      assert.ok(runsOnlyForMain(name, condition), `${name} job ${job} can run outside main`)
    }
  })
}

test('the contract reader finds every workflow trigger and job', () => {
  for (const name of [...TEST_WORKFLOWS, ...Object.keys(DISPATCHED_TEST_JOBS), ...RELEASE_WORKFLOWS]) {
    assert.ok(Object.keys(triggers(name)).length > 0, `no triggers read from ${name}`)
    assert.ok(Object.keys(jobs(name)).length > 0, `no jobs read from ${name}`)
  }
  assert.ok('dispatch' in jobs('export-kit.yml'))
  assert.ok('push' in triggers('tests.yml'))
  assert.ok('sweep' in jobs('parity.yml'))
  assert.ok('schedule' in triggers('parity.yml'))
})
