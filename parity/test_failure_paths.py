#!/usr/bin/env python3
"""Run parity orchestration against isolated process and browser boundaries."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


class ParityFailurePaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nm-parity-failures-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('parity', 'scripts', 'tools', 'parity/programs', 'parity/out'):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        for name in ('parity/sweep.py', 'parity/batch-golden.mjs',
                     'scripts/parity-summary', 'tools/reference-oracle.mjs', 'tools/portable.mjs'):
            shutil.copy2(ROOT / name, self.root / name)
        self.out = self.root / 'parity/out'
        self.reference = self.root / 'reference'
        self.write('reference/package.json', '{"type":"module"}')
        self.write('reference/node_modules/playwright/package.json', '{}')
        self.env = dict(os.environ, NM_REFERENCE_ROOT=str(self.reference))
        # A real, informative PNG with hand-selected black and white pixels.
        image = Image.new('RGBA', (2, 2))
        image.putdata([(0, 0, 0, 255), (255, 255, 255, 255)] * 2)
        image.save(self.root / 'pixels.png')
        for case in ('first', 'second'):
            self.write(f'parity/programs/{case}.dsl', 'search synth\nnoise().write(o0)\nrender(o0)\n')
        self.renderer = self.write('renderer', f'#!{sys.executable}\n' + '''
import json, os, shutil, sys
from pathlib import Path
for entry in json.loads(Path(sys.argv[2]).read_text()):
    if os.environ.get('RENDER_OUTPUT', '1') == '1':
        shutil.copyfile(Path(__file__).parent / 'pixels.png', entry['out'])
sys.exit(int(os.environ.get('RENDER_RC', '0')))
''')
        self.renderer.chmod(0o755)
        self.env['NM_RENDER'] = str(self.renderer)
        # Summary invokes this suite as a gate in a real checkout. At this
        # process boundary use a controllable exit status to avoid recursion.
        self.write('parity/test_failure_paths.py',
                   'import os\nraise SystemExit(int(os.environ.get("REGRESSION_RC", "0")))\n')

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def run_script(self, *args, **env):
        return subprocess.run(args, cwd=self.root, env={**self.env, **env},
                              capture_output=True, text=True, timeout=30)

    def seed(self):
        for case in ('first', 'second'):
            for suffix in ('golden.png', 'golden.f1.png', 'candidate.png', 'media.png'):
                shutil.copyfile(self.root / 'pixels.png', self.out / f'{case}.{suffix}')
            for suffix in ('graph.json', 'golden.f1.passes.json', 'golden.f1.state.bin'):
                (self.out / f'{case}.{suffix}').write_text('stale')
        (self.out / 'unrelated.golden.png').write_text('preserve')

    def fake_minter(self):
        # The process boundary can fail after emitting valid output, as a real
        # renderer can. The actual sweep and pixel grading remain in use.
        self.write('parity/batch-golden.mjs', '''
import { readFileSync, copyFileSync } from 'node:fs';
import { basename, join } from 'node:path';
const args = process.argv.slice(2);
const paths = readFileSync(args[args.indexOf('--list') + 1], 'utf8').trim().split('\\n');
if (process.env.MINT_OUTPUT !== '0') {
  for (const path of paths) copyFileSync('pixels.png', join(args[0], basename(path, '.dsl') + '.golden.png'));
}
process.exit(Number(process.env.MINT_RC || 0));
''')

    def sweep(self, **env):
        result = self.run_script(sys.executable, 'parity/sweep.py', 'first', 'second', **env)
        ledger = self.out / 'ledger.json'
        self.assertTrue(ledger.exists(), result.stderr)
        rows = json.loads(ledger.read_text())
        self.assertEqual([r['program'] for r in rows], ['first', 'second'])
        return result, rows

    def test_mint_failure_cannot_reuse_prior_success(self):
        self.fake_minter()
        initial, rows = self.sweep()
        self.assertEqual(initial.returncode, 0, initial.stderr)
        self.assertEqual([r['bucket'] for r in rows], ['EXACT', 'EXACT'])
        failed, rows = self.sweep(MINT_RC='7', MINT_OUTPUT='0')
        self.assertNotEqual(failed.returncode, 0, failed.stdout)
        self.assertEqual([r['bucket'] for r in rows], ['MISSING', 'MISSING'])

    def test_failed_mint_with_valid_output_still_fails(self):
        self.fake_minter()
        result, rows = self.sweep(MINT_RC='7')
        self.assertEqual([r['bucket'] for r in rows], ['MISSING', 'MISSING'])
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_failed_candidate_with_valid_output_still_fails(self):
        self.fake_minter()
        result, rows = self.sweep(RENDER_RC='9')
        self.assertEqual([r['bucket'] for r in rows], ['MISSING', 'MISSING'])
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_candidate_launch_failure_retains_every_case(self):
        self.fake_minter()
        self.seed()
        result, rows = self.sweep(NM_RENDER=str(self.root / 'absent-renderer'))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([r['bucket'] for r in rows], ['MISSING', 'MISSING'])

    def test_candidate_no_output_cannot_reuse_prior_success(self):
        self.fake_minter()
        self.assertEqual(self.sweep()[0].returncode, 0)
        result, rows = self.sweep(RENDER_RC='9', RENDER_OUTPUT='0')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([r['bucket'] for r in rows], ['MISSING', 'MISSING'])

    def test_minter_launch_failure_retains_every_case(self):
        self.seed()
        result, rows = self.sweep(PATH=str(self.root / 'no-executables'))
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([r['bucket'] for r in rows], ['MISSING', 'MISSING'])

    def reference_modules(self):
        self.write('reference/shaders/src/index.js', '''
export const registerStarterOps = () => {};
export const mergeIntoEnums = async () => {};
export const compileGraph = () => { throw new Error('fixture compile failed'); };
''')
        self.write('reference/shaders/src/lang/std_enums.js', 'export const stdEnums = {};')
        self.write('reference/shaders/src/lang/effectAliases.js', 'export const registerEffectAlias = () => {};')
        self.write('reference/shaders/effects/manifest.json', '{}')
        # Only browser operations are doubled. bootstrapReference, compile
        # invocation, repair loop and filesystem cleanup execute normally.
        self.write('reference/vendor/shade-mcp/harness/index.js', '''
export class BrowserSession {
  constructor () {
    this.page = { route: async () => {}, setViewportSize: async () => {},
      waitForFunction: async () => {}, evaluate: async () => 'test adapter' };
  }
  async setup () { if (process.env.FAIL_BROWSER) throw new Error('browser launch failed'); }
  async setBackend () {}
  async teardown () {}
}
''')

    def batch(self, **env):
        return self.run_script('node', 'parity/batch-golden.mjs', str(self.out), '--',
                               'parity/programs/first.dsl', 'parity/programs/second.dsl', **env)

    def assert_no_old_goldens(self):
        for case in ('first', 'second'):
            names = [p.name for p in self.out.glob(f'{case}.*') if '.candidate' not in p.name]
            self.assertEqual(names, [], 'stale reference artifacts survived setup failure')
        self.assertTrue((self.out / 'unrelated.golden.png').exists())

    def test_reference_import_failure_invalidates_all_old_artifacts(self):
        self.seed()
        result = self.batch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ERR_MODULE_NOT_FOUND', result.stderr)
        self.assert_no_old_goldens()

    def test_missing_reference_configuration_invalidates_all_old_artifacts(self):
        self.seed()
        result = self.batch(NM_REFERENCE_ROOT='')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('NM_REFERENCE_ROOT is not set', result.stderr)
        self.assert_no_old_goldens()

    def test_browser_launch_failure_invalidates_all_old_artifacts(self):
        self.reference_modules()
        self.seed()
        result = self.batch(FAIL_BROWSER='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('browser launch failed', result.stderr)
        self.assertIn('minted=0 failed=2 total=2', result.stdout)
        self.assert_no_old_goldens()

    def test_fixture_compile_failure_invalidates_all_old_artifacts(self):
        self.reference_modules()
        self.seed()
        result = self.batch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fixture compile failed', result.stderr)
        self.assert_no_old_goldens()

    def summary(self, **env):
        result = self.run_script('bash', 'scripts/parity-summary', 'first', 'second', **env)
        lines = [line for line in result.stdout.splitlines() if line.startswith('PARITY-SUMMARY ')]
        self.assertEqual(len(lines), 1, result.stderr)
        counts = json.loads(lines[0].split(' ', 1)[1])
        self.assertEqual(counts['expected'], 2)
        return result, counts

    def test_summary_propagates_failed_sweep_with_passing_pixels(self):
        self.fake_minter()
        result, counts = self.summary(MINT_RC='7')
        self.assertEqual(counts['exact'], 0)
        self.assertEqual(counts['strict'], 0)
        self.assertEqual(counts['missing'], 2)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_summary_excludes_pixels_from_failed_candidate_process(self):
        self.fake_minter()
        result, counts = self.summary(RENDER_RC='9')
        self.assertEqual(counts['exact'], 0)
        self.assertEqual(counts['strict'], 0)
        self.assertEqual(counts['missing'], 2)
        self.assertNotEqual(result.returncode, 0, result.stdout)

    def test_summary_checks_sweep_exit_independently_of_ledger(self):
        # Exercise the shell's subprocess contract independently of the
        # sweep's own failure classification.
        self.write('parity/sweep.py', '''
import json, pathlib, sys
ledger = pathlib.Path(sys.argv[sys.argv.index('--ledger') + 1])
ledger.write_text(json.dumps([
    {"program": "first", "verdict": "EXACT", "bucket": "EXACT"},
    {"program": "second", "verdict": "EXACT", "bucket": "EXACT"}
]))
sys.exit(7)
''')
        result, counts = self.summary()
        self.assertEqual(result.returncode, 7, result.stdout)

    def test_summary_fails_if_orchestration_regression_gate_fails(self):
        self.fake_minter()
        result, counts = self.summary(REGRESSION_RC='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(counts['skip'], 2)
        self.assertEqual(counts['executed'], 0)

    def test_summary_retains_expected_cases_when_sweep_cannot_write_ledger(self):
        self.write('parity/sweep.py', 'raise RuntimeError("sweep setup failed")\n')
        result, counts = self.summary()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(counts['skip'], 2)
        self.assertEqual(counts['executed'], 0)

    def test_summary_success_and_failure_do_not_reuse_goldens(self):
        self.fake_minter()
        success, counts = self.summary()
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(counts['exact'], 2)
        failed, counts = self.summary(MINT_RC='7', MINT_OUTPUT='0')
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(counts['exact'], 0)
        self.assertEqual(counts['missing'], 2)
        recovered, counts = self.summary()
        self.assertEqual(recovered.returncode, 0, recovered.stderr)
        self.assertEqual(counts['exact'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
