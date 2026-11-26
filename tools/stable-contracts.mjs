import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..')
export const stableContracts = JSON.parse(readFileSync(join(ROOT, 'parity', 'stable-contracts.json'), 'utf8'))

export function stableContractForFile (file) {
  const contract = stableContracts.find(c => resolve(ROOT, c.path) === resolve(file))
  if (!contract) return null
  const hash = createHash('sha256').update(readFileSync(file)).digest('hex')
  if (hash !== contract.sha256) throw new Error(`stable contract source changed: ${contract.path}`)
  return contract
}
