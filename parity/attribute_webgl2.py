#!/usr/bin/env python3
"""Gate the WebGL2 grade on attribution to the reference's own backends.

The WebGPU goldens are the parity gate. The reference's WebGL2 backend is the
upstream authority, and on some cases the reference's own WebGPU output still
differs from it, which no change in this port can resolve. So a WebGL2 case
that does not pass is accepted only when the port's candidate is pixel-identical
to the reference's WebGPU golden for the same case, minted in the same run: the
difference is then the reference's, between its two backends. Every other
non-passing WebGL2 case fails, as does a case with no golden or no candidate.

Pixel-identical means equal decoded RGBA, the test parity/sweep.py uses for
EXACT. A timed case is accepted only when every one of its samples is.

Usage: attribute_webgl2.py --ledger WEBGL2_LEDGER --webgpu-goldens DIR [--candidates DIR]
Prints one WEBGL2-ATTRIBUTION line; exit 0 when every case passes or is
attributed, 1 otherwise.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PASSING = {"EXACT", "STRICT", "UNINFORMATIVE"}


def pixels(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def identical(a, b):
    if not a.exists() or not b.exists():
        return False
    pa, pb = pixels(a), pixels(b)
    return pa.shape == pb.shape and bool(np.array_equal(pa, pb))


def pairs(row, goldens, candidates):
    case = row["program"]
    samples = row.get("samples")
    if samples:
        return [(goldens / f"{case}.golden.{s['label']}.png", candidates / f"{case}.candidate.{s['label']}.png")
                for s in samples]
    return [(goldens / f"{case}.golden.png", candidates / f"{case}.candidate.png")]


def attribute(rows, goldens, candidates):
    counts = {"cases": len(rows), "pass": 0, "attributed": 0, "unattributed": 0, "missing": 0}
    unattributed, missing = [], []
    for row in rows:
        bucket = row.get("bucket", row.get("verdict"))
        if bucket in PASSING:
            counts["pass"] += 1
        elif bucket == "MISSING":
            counts["missing"] += 1
            missing.append(row["program"])
        elif all(identical(g, c) for g, c in pairs(row, goldens, candidates)):
            counts["attributed"] += 1
        else:
            counts["unattributed"] += 1
            unattributed.append(row["program"])
    return counts, unattributed, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", required=True, help="the WebGL2 run's ledger")
    ap.add_argument("--webgpu-goldens", required=True, help="the WebGPU goldens kept from the gate run")
    ap.add_argument("--candidates", default=str(ROOT / "parity" / "out"), help="the WebGL2 run's candidates")
    args = ap.parse_args(argv)
    rows = json.loads(Path(args.ledger).read_text())
    counts, unattributed, missing = attribute(rows, Path(args.webgpu_goldens), Path(args.candidates))
    print("WEBGL2-ATTRIBUTION " + json.dumps(counts, separators=(",", ":")))
    if unattributed:
        print("attribute_webgl2: WebGL2 differences the reference's backends do not explain: "
              + " ".join(unattributed), file=sys.stderr)
    if missing:
        print("attribute_webgl2: cases with no WebGL2 golden or candidate: " + " ".join(missing), file=sys.stderr)
    return 0 if counts["cases"] and not unattributed and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
