#!/usr/bin/env python3
"""The WebGL2 attribution gate accepts only differences the reference's own backends explain."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attribute_webgl2  # noqa: E402


def png(path, value):
    Image.fromarray(np.full((4, 4, 4), value, dtype=np.uint8), "RGBA").save(path)


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.goldens, self.candidates = root / "webgpu", root / "out"
        self.goldens.mkdir()
        self.candidates.mkdir()
        self.ledger = root / "ledger.json"

    def tearDown(self):
        self.temp.cleanup()

    def run_gate(self, rows):
        self.ledger.write_text(json.dumps(rows))
        return attribute_webgl2.main(["--ledger", str(self.ledger), "--webgpu-goldens", str(self.goldens),
                                      "--candidates", str(self.candidates)])

    def test_passing_cases_need_no_attribution(self):
        rows = [{"program": p, "verdict": v} for p, v in (("a", "EXACT"), ("b", "STRICT"))]
        rows.append({"program": "c", "verdict": "EXACT", "bucket": "UNINFORMATIVE"})
        self.assertEqual(self.run_gate(rows), 0)

    def test_a_failure_whose_candidate_equals_the_webgpu_golden_is_attributed(self):
        png(self.goldens / "x.golden.png", 10)
        png(self.candidates / "x.candidate.png", 10)
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "FAIL"}]), 0)

    def test_a_failure_whose_candidate_differs_from_the_webgpu_golden_fails(self):
        png(self.goldens / "x.golden.png", 10)
        png(self.candidates / "x.candidate.png", 11)
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "FAIL"}]), 1)

    def test_a_near_case_is_held_to_the_same_attribution(self):
        png(self.goldens / "x.golden.png", 10)
        png(self.candidates / "x.candidate.png", 12)
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "NEAR"}]), 1)

    def test_a_failure_without_a_webgpu_golden_fails(self):
        png(self.candidates / "x.candidate.png", 10)
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "FAIL"}]), 1)

    def test_a_failure_without_a_candidate_fails(self):
        png(self.goldens / "x.golden.png", 10)
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "FAIL"}]), 1)

    def test_an_attributed_near_case_passes(self):
        png(self.goldens / "x.golden.png", 10)
        png(self.candidates / "x.candidate.png", 10)
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "NEAR"}]), 0)

    def test_one_unattributed_case_fails_a_ledger_of_passes_and_attributed_cases(self):
        for case, cand in (("a", 10), ("b", 30)):
            png(self.goldens / f"{case}.golden.png", 10 if case == "a" else 20)
            png(self.candidates / f"{case}.candidate.png", cand)
        rows = [{"program": "p", "verdict": "EXACT"}, {"program": "a", "verdict": "FAIL"},
                {"program": "b", "verdict": "FAIL"}]
        self.assertEqual(self.run_gate(rows), 1)
        self.assertEqual(self.run_gate(rows[:2]), 0)

    def test_a_missing_case_fails(self):
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "MISSING"}]), 1)

    def test_a_timed_case_is_attributed_only_when_every_sample_is(self):
        for label, cand in (("t0", 10), ("t1", 20)):
            png(self.goldens / f"x.golden.{label}.png", 10 if label == "t0" else 20)
            png(self.candidates / f"x.candidate.{label}.png", cand)
        row = {"program": "x", "verdict": "FAIL", "samples": [{"label": "t0"}, {"label": "t1"}]}
        self.assertEqual(self.run_gate([row]), 0)
        png(self.candidates / "x.candidate.t1.png", 21)
        self.assertEqual(self.run_gate([row]), 1)

    def test_a_size_mismatch_is_not_attributed(self):
        png(self.goldens / "x.golden.png", 10)
        Image.fromarray(np.full((8, 8, 4), 10, dtype=np.uint8), "RGBA").save(self.candidates / "x.candidate.png")
        self.assertEqual(self.run_gate([{"program": "x", "verdict": "FAIL"}]), 1)

    def test_an_empty_ledger_fails(self):
        self.assertEqual(self.run_gate([]), 1)


if __name__ == "__main__":
    unittest.main()
