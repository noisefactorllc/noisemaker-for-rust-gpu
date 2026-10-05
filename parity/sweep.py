#!/usr/bin/env python3
"""Corpus-wide pixel parity sweep: mint goldens, render candidates, grade, ledger.

Cases are parity/programs/*.dsl (the shared fixture pool) and parity/coverage/*.dsl
(the generated effect x mode corpus); a case id is the file stem. One run:

  1. MINT    goldens with the reference engine's WebGPU backend
             (parity/batch-golden.mjs), fresh, in this run -- a sweep never grades a
             golden from an earlier run. Host inputs the minter saves
             (<id>.<textureId>.png: media/text images, asyncInit overlays) are
             passed to the candidate.
  2. RENDER  candidates with nm-render in one batch process. By default the
             candidate compiles the DSL itself (the live Rust frontend);
             --from-graph renders the reference graph the golden page rendered
             instead (isolates the runtime from the compiler).
  3. GRADE   each case: decoded RGBA8 pixels compared with the golden.
             exact  -- identical pixels
             strict -- max-abs-diff <= 2.001 (8-bit units) and global SSIM >= 0.98
             near   -- beyond strict but within the case's recorded policy in
                       NEAR_POLICIES (each entry carries its mechanism)
             fail   -- outside every policy
             missing-- no golden or no candidate (mint or render failure)
             TIMED cases (stateful solvers) are graded on a sample series.
  4. LEDGER  parity/out/ledger.json (untracked), one row per case.

Usage:
  NM_REFERENCE_ROOT=/path/to/noisemaker python3 parity/sweep.py [case-id ...]
      [--from-graph] [--skip-mint] [--skip-render] [--jobs-chunk 60]

Env: NM_RENDER (candidate binary, default target/release/nm-render).
Exit 0 iff every case is exact or strict.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "parity" / "out"
SIZE = 256
TIME = 0.25
FRAMES = 8
STRICT_TOL = 2.001
STRICT_SSIM = 0.98

# Stateful solvers graded on timed samples (normalized dt = 1/600 per frame),
# matching the family's established split: (run seconds, sample every seconds).
TIMED = {
    "navierStokes": (30, 5),
    "temporalAberration": (30, 10),
}

# Per-case tolerances beyond the strict bar. Every entry must name the observed
# mechanism; entries are added only with evidence from this backend.
NEAR_POLICIES = {}


def discover(ids):
    cases = {}
    for sub in ("programs", "coverage"):
        for path in sorted((ROOT / "parity" / sub).glob("*.dsl")):
            cases[path.stem] = path
    if ids:
        unknown = [i for i in ids if i not in cases]
        if unknown:
            sys.exit("unknown case ids: " + " ".join(unknown))
        cases = {i: cases[i] for i in ids}
    return cases


def load_rgba(path):
    return np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0


def global_ssim(a, b):
    def luma(x):
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    la, lb = luma(a).ravel(), luma(b).ravel()
    mu_a, mu_b = la.mean(), lb.mean()
    var_a, var_b = la.var(), lb.var()
    cov = ((la - mu_a) * (lb - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den != 0 else 1.0


def grade_pair(golden, candidate):
    a, b = load_rgba(golden), load_rgba(candidate)
    if a.shape != b.shape:
        return {"error": "size_mismatch", "golden_shape": list(a.shape), "candidate_shape": list(b.shape)}
    diff = np.abs(a - b) * 255.0
    return {
        "exact": bool(np.array_equal(np.asarray(Image.open(golden).convert("RGBA")),
                                     np.asarray(Image.open(candidate).convert("RGBA")))),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "ssim": global_ssim(a, b),
    }


def host_textures(case_id):
    found = {}
    prefix = case_id + "."
    for p in sorted(OUT.glob(prefix + "*.png")):
        tex = p.name[len(prefix):-len(".png")]
        if tex in ("golden", "candidate") or tex.startswith("golden.t") or tex.startswith("candidate.t"):
            continue
        found[tex] = str(p)
    return found


def mint(cases, chunk):
    single = [str(p) for i, p in cases.items() if i not in TIMED]
    rc = 0
    if single:
        listing = OUT / "mint-list.txt"
        listing.write_text("\n".join(single) + "\n")
        rc |= subprocess.call(["node", str(ROOT / "parity" / "batch-golden.mjs"), str(OUT),
                               "--size", str(SIZE), "--time", str(TIME), "--frames", str(FRAMES),
                               "--chunk-size", str(chunk), "--list", str(listing)])
    for case_id, path in cases.items():
        if case_id in TIMED:
            run, every = TIMED[case_id]
            rc |= subprocess.call(["node", str(ROOT / "parity" / "batch-golden.mjs"), str(OUT),
                                   "--size", str(SIZE), "--run-seconds", str(run),
                                   "--sample-every", str(every), "--", str(path)])
    return rc


def render(cases, nm_render, from_graph):
    manifest = []
    for case_id, path in cases.items():
        entry = {"out": str(OUT / (case_id + ".candidate.png")), "size": SIZE, "time": TIME,
                 "frames": FRAMES, "hostTextures": host_textures(case_id)}
        if from_graph:
            entry["graph"] = str(OUT / (case_id + ".graph.json"))
        else:
            entry["dsl"] = str(path)
        if case_id in TIMED:
            run, every = TIMED[case_id]
            entry["runSeconds"] = run
            entry["sampleEvery"] = every
        manifest.append(entry)
    for case_id in cases:
        for p in OUT.glob(case_id + ".candidate*.png"):
            p.unlink()
    path = OUT / "render-manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return subprocess.call([nm_render, "batch", str(path)])


def grade(cases):
    rows = []
    for case_id in cases:
        policy = NEAR_POLICIES.get(case_id, {"tolerance": STRICT_TOL, "ssim_min": STRICT_SSIM})
        row = {"program": case_id, "policy": {"tolerance": policy["tolerance"], "ssim_min": policy["ssim_min"]}}
        if case_id in TIMED:
            run, every = TIMED[case_id]
            samples = []
            for sec in range(every, run + 1, every):
                g = OUT / f"{case_id}.golden.t{sec}.png"
                c = OUT / f"{case_id}.candidate.t{sec}.png"
                if not g.exists() or not c.exists():
                    samples = None
                    break
                m = grade_pair(g, c)
                m.update({"golden": str(g.relative_to(ROOT)), "candidate": str(c.relative_to(ROOT)), "t": sec})
                samples.append(m)
            if samples is None or any("error" in s for s in samples):
                row.update({"verdict": "MISSING"})
            else:
                row.update({"samples": samples, "exact": all(s["exact"] for s in samples),
                            "max_abs_diff": max(s["max_abs_diff"] for s in samples),
                            "ssim": min(s["ssim"] for s in samples)})
        else:
            g = OUT / f"{case_id}.golden.png"
            c = OUT / f"{case_id}.candidate.png"
            if not g.exists() or not c.exists():
                row.update({"verdict": "MISSING", "golden_present": g.exists(), "candidate_present": c.exists()})
            else:
                m = grade_pair(g, c)
                if "error" in m:
                    row.update({"verdict": "FAIL", **m})
                else:
                    row.update(m)
        if "verdict" not in row:
            if row["exact"]:
                row["verdict"] = "EXACT"
            elif row["max_abs_diff"] <= STRICT_TOL and row["ssim"] >= STRICT_SSIM:
                row["verdict"] = "STRICT"
            elif row["max_abs_diff"] <= policy["tolerance"] and row["ssim"] >= policy["ssim_min"]:
                row["verdict"] = "NEAR"
                row["mechanism"] = policy.get("mechanism")
            else:
                row["verdict"] = "FAIL"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--from-graph", action="store_true")
    ap.add_argument("--skip-mint", action="store_true", help="grade against goldens already in parity/out (development only)")
    ap.add_argument("--skip-render", action="store_true", help="grade candidates already in parity/out (development only)")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--ledger", default=str(OUT / "ledger.json"))
    args = ap.parse_args()
    if not os.environ.get("NM_REFERENCE_ROOT"):
        sys.exit("NM_REFERENCE_ROOT is not set")
    nm_render = os.environ.get("NM_RENDER", str(ROOT / "target" / "release" / "nm-render"))
    OUT.mkdir(parents=True, exist_ok=True)
    cases = discover(args.cases)

    if not args.skip_mint:
        print(f"[sweep] minting {len(cases)} goldens", file=sys.stderr)
        mint(cases, args.chunk)
    if not args.skip_render:
        print(f"[sweep] rendering {len(cases)} candidates ({'graph' if args.from_graph else 'dsl'})", file=sys.stderr)
        render(cases, nm_render, args.from_graph)
    rows = grade(cases)
    Path(args.ledger).write_text(json.dumps(rows, indent=1) + "\n")
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    for r in rows:
        if r["verdict"] in ("FAIL", "MISSING", "NEAR"):
            detail = f"max={r.get('max_abs_diff', float('nan')):.3f} ssim={r.get('ssim', float('nan')):.5f}" \
                if "max_abs_diff" in r else json.dumps({k: v for k, v in r.items() if k not in ("program", "policy", "verdict")})
            print(f"[{r['verdict']}] {r['program']}: {detail}")
    print("SWEEP " + json.dumps(counts, sort_keys=True))
    ok = all(r["verdict"] in ("EXACT", "STRICT") for r in rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
