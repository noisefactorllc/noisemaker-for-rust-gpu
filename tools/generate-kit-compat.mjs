#!/usr/bin/env node
// generate-kit-compat.mjs: the export kit's effect list.
//
// Writes export-kit/compat-effects.json: a JSON array of the `<namespace>/<name>`
// ids of every effect in the embedded catalog (crates/noisemaker-effects/catalog,
// listed as its build script lists it: every namespace directory except
// `share`, every effect directory holding a definition.json), sorted. The kit
// builder reads it (kit.config.json `compat.fromJsonList`) to write the kit's
// compat.json, the list of effects Noisedeck's export dialog lets this kit
// render. Every catalog effect renders here: the pixel parity sweep requires
// informative exact or strict evidence for each one (scripts/parity-summary).
//
// Usage:
//   node tools/generate-kit-compat.mjs           write the file
//   node tools/generate-kit-compat.mjs --check   exit 1 if the file is stale

import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const CATALOG = join(ROOT, 'crates', 'noisemaker-effects', 'catalog')
const OUT = join(ROOT, 'export-kit', 'compat-effects.json')

const args = process.argv.slice(2)
const check = args.includes('--check')
for (const arg of args) {
  if (arg !== '--check') {
    console.error(`generate-kit-compat: unknown argument ${arg}`)
    process.exit(2)
  }
}

const dirs = dir => readdirSync(dir).filter(name => statSync(join(dir, name)).isDirectory()).sort()

const ids = []
for (const namespace of dirs(CATALOG)) {
  if (namespace === 'share') continue
  for (const name of dirs(join(CATALOG, namespace))) {
    if (existsSync(join(CATALOG, namespace, name, 'definition.json'))) ids.push(`${namespace}/${name}`)
  }
}
ids.sort()
const text = JSON.stringify(ids, null, 2) + '\n'

if (check) {
  const current = existsSync(OUT) ? readFileSync(OUT, 'utf8') : null
  if (current !== text) {
    console.error(`generate-kit-compat: ${relative(ROOT, OUT)} is stale; run node tools/generate-kit-compat.mjs`)
    process.exit(1)
  }
  console.log(`KIT-COMPAT: ${ids.length} effects, ${relative(ROOT, OUT)} is current`)
} else {
  writeFileSync(OUT, text)
  console.log(`KIT-COMPAT: ${ids.length} effects -> ${relative(ROOT, OUT)}`)
}
