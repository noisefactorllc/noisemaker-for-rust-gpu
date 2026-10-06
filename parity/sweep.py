#!/usr/bin/env python3
"""Corpus-wide pixel parity sweep: mint goldens, render candidates, grade, ledger.

Cases are parity/programs/*.dsl (the shared fixture pool), parity/coverage/*.dsl
(the generated effect x mode corpus) and parity/portable/*.dsl (user-defined
Portable effects: each registers its <id>.portable.json sidecar, WGSL in
<id>.<program>.wgsl, before its program runs, in the golden page and in the
candidate); a case id is the file stem. One run:

  1. MINT    goldens with the reference engine's WebGPU backend
             (parity/batch-golden.mjs), fresh, in this run -- a sweep never grades a
             golden from an earlier run. The minter also saves the host inputs the
             page produced (<id>.<textureId>.png: media/text images, asyncInit
             overlays).
  2. RENDER  candidates with nm-render in one batch process. By default the
             candidate runs the DSL through the port's demo host (the live Rust
             frontend, ProgramState and the demo's control initialization, then
             applyStepParameterValues) and produces every host input natively:
             the demo's default media image ($NM_REFERENCE_ROOT/demo/shaders/img/
             testcard.png), text canvases, asyncInit overlays and meshes (the
             fixture's .obj sidecar into mesh0, as the minter loads it).
             --captured-host-inputs grades with the minter's saved host-input PNGs
             in place of the natively produced ones (isolates the renderer from
             the host-input ports); --from-graph renders the reference graph the
             golden page rendered with the saved host inputs (isolates the runtime
             from the compiler).
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
      [--captured-host-inputs | --from-graph] [--skip-mint] [--skip-render]
      [--chunk 60] [--out DIR] [--golden-dir DIR]

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
# mechanism; entries are added only with evidence from this backend. NEAR is
# outside the published contract: scripts/parity-summary still fails on it.
#
# Evidence for every entry below (Apple M4, Metal): substituting the Metal code
# Chromium's Tint generates for the failing pass, compiled with Dawn's
# MTLCompileOptions, renders each case byte-identical to its golden. wgpu-hal
# compiles every Metal library with preserveInvariance on and fast math; Dawn
# compiles with relaxed math and enables invariance only for @invariant
# shaders. Without invariance Metal orders fused multiply-adds and sum
# groupings by program order, with it by expression depth, so the two engines
# round the same WGSL differently. Neither option is reachable through wgpu's
# public API.
#
# Visual inspection (2026-10-06, golden | candidate | diff for all 19 cases):
# indistinguishable. The differences are scattered single pixels: 3-4 levels
# along craquelure cell edges, 2 pixels on the landscapes, and isolated root
# flips on Newton basin boundaries and in wormhole's noise field; no
# structural, color or shape difference. Tolerated for now by operator
# decision, pending a Metal compile-option fix (wgpu-hal); still reported as
# NEAR, outside the published contract.
_INVARIANCE_FMA = ("Metal FMA grouping differs under wgpu-hal's preserveInvariance "
                   "(Dawn compiles without it); exact once the pass compiles without invariance")
_LANDSCAPE_FMA = ("ray-origin z fused as fma(fma(up.z, ty, right.z * tx), span, a) under "
                  "preserveInvariance vs fma(fma(right.z, tx, up.z * ty), span, a) in Dawn")
_VERTEX_INVARIANCE = ("vertex-stage preserveInvariance changes the warped coordinate's rounding; "
                      "exact once only the vertex stage compiles without invariance")
_NEWTON_BASINS = ("Newton-iteration basin boundaries amplify single-ulp differences between "
                  "naga's and Tint's Metal code under different math modes into root flips on "
                  "isolated pixels; needs both Tint's code and Dawn's compile options to match")

NEAR_POLICIES = {
    "craquelure": {"tolerance": 3.001, "ssim_min": 0.99998, "mechanism": _INVARIANCE_FMA},
    "craquelureBig": {"tolerance": 4.001, "ssim_min": 0.99998, "mechanism": _INVARIANCE_FMA},
    "filter_craquelure": {"tolerance": 3.001, "ssim_min": 0.99998, "mechanism": _INVARIANCE_FMA},
    "heightmap3d_landscape": {"tolerance": 3.001, "ssim_min": 0.99999, "mechanism": _LANDSCAPE_FMA},
    "render_renderLandscape3d": {"tolerance": 3.001, "ssim_min": 0.99999, "mechanism": _LANDSCAPE_FMA},
    "synth3d_heightmap3d": {"tolerance": 3.001, "ssim_min": 0.99999, "mechanism": _LANDSCAPE_FMA},
    "synth3d_heightmap3d__volumeSize_x128": {"tolerance": 3.001, "ssim_min": 0.99999, "mechanism": _LANDSCAPE_FMA},
    "classicNoisedeck_fractal__type_newton": {"tolerance": 65.001, "ssim_min": 0.99994, "mechanism": _INVARIANCE_FMA},
    "filter_wormhole": {"tolerance": 189.001, "ssim_min": 0.9998, "mechanism": _VERTEX_INVARIANCE},
    "filter_wormhole__wrap_mirror": {"tolerance": 173.001, "ssim_min": 0.99985, "mechanism": _VERTEX_INVARIANCE},
    "newton": {"tolerance": 246.001, "ssim_min": 0.998, "mechanism": _NEWTON_BASINS},
    "synth_newton": {"tolerance": 246.001, "ssim_min": 0.998, "mechanism": _NEWTON_BASINS},
    "synth_newton__invert_true": {"tolerance": 247.001, "ssim_min": 0.998, "mechanism": _NEWTON_BASINS},
    "synth_newton__outputMode_iteration": {"tolerance": 204.001, "ssim_min": 0.9978, "mechanism": _NEWTON_BASINS},
    "synth_newton__outputMode_rootIndex": {"tolerance": 212.001, "ssim_min": 0.998, "mechanism": _NEWTON_BASINS},
    "synth_newton__poi_octoFlower8": {"tolerance": 252.001, "ssim_min": 0.9968, "mechanism": _NEWTON_BASINS},
    "synth_newton__poi_pentaSpiral5": {"tolerance": 213.001, "ssim_min": 0.9985, "mechanism": _NEWTON_BASINS},
    "synth_newton__poi_spiralJunction3": {"tolerance": 89.001, "ssim_min": 0.99996, "mechanism": _NEWTON_BASINS},
    "synth_newton__poi_starCenter5": {"tolerance": 207.001, "ssim_min": 0.9988, "mechanism": _NEWTON_BASINS},
}


def discover(ids):
    cases = {}
    for sub in ("programs", "coverage", "portable"):
        for path in sorted((ROOT / "parity" / sub).glob("*.dsl")):
            cases[path.stem] = path
    if ids:
        unknown = [i for i in ids if i not in cases]
        if unknown:
            sys.exit("unknown case ids: " + " ".join(unknown))
        cases = {i: cases[i] for i in ids}
    return cases


def rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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


def host_textures(case_id, golden_dir):
    found = {}
    prefix = case_id + "."
    for p in sorted(golden_dir.glob(prefix + "*.png")):
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


def render(cases, nm_render, mode, golden_dir):
    manifest = []
    for case_id, path in cases.items():
        entry = {"out": str(OUT / (case_id + ".candidate.png")), "size": SIZE, "time": TIME,
                 "frames": FRAMES}
        sidecar = path.with_suffix(".obj")
        if sidecar.exists():
            entry["obj"] = str(sidecar)
        portable = path.with_suffix(".portable.json")
        if portable.exists() and mode != "graph":
            entry["portable"] = str(portable)
        if mode == "graph":
            entry["graph"] = str(golden_dir / (case_id + ".graph.json"))
            entry["hostTextures"] = host_textures(case_id, golden_dir)
        else:
            entry["dsl"] = str(path)
            if mode == "captured":
                entry["hostTextures"] = host_textures(case_id, golden_dir)
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


def grade(cases, golden_dir):
    rows = []
    for case_id in cases:
        policy = NEAR_POLICIES.get(case_id, {"tolerance": STRICT_TOL, "ssim_min": STRICT_SSIM})
        row = {"program": case_id, "policy": {"tolerance": policy["tolerance"], "ssim_min": policy["ssim_min"]}}
        if case_id in TIMED:
            run, every = TIMED[case_id]
            samples = []
            for sec in range(every, run + 1, every):
                g = golden_dir / f"{case_id}.golden.t{sec}.png"
                c = OUT / f"{case_id}.candidate.t{sec}.png"
                if not g.exists() or not c.exists():
                    samples = None
                    break
                m = grade_pair(g, c)
                m.update({"golden": rel(g), "candidate": rel(c), "t": sec})
                samples.append(m)
            if samples is None or any("error" in s for s in samples):
                row.update({"verdict": "MISSING"})
            else:
                row.update({"samples": samples, "exact": all(s["exact"] for s in samples),
                            "max_abs_diff": max(s["max_abs_diff"] for s in samples),
                            "ssim": min(s["ssim"] for s in samples)})
        else:
            g = golden_dir / f"{case_id}.golden.png"
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
    inputs = ap.add_mutually_exclusive_group()
    inputs.add_argument("--from-graph", action="store_true",
                        help="render the golden page's graph with the minter's host-input PNGs")
    inputs.add_argument("--captured-host-inputs", action="store_true",
                        help="run the DSL with the minter's host-input PNGs instead of native host inputs")
    ap.add_argument("--out", default=None, help="output directory (default parity/out)")
    ap.add_argument("--golden-dir", default=None,
                    help="grade against goldens minted earlier in DIR (implies --skip-mint; development only)")
    ap.add_argument("--skip-mint", action="store_true", help="grade against goldens already in parity/out (development only)")
    ap.add_argument("--skip-render", action="store_true", help="grade candidates already in parity/out (development only)")
    ap.add_argument("--chunk", type=int, default=60)
    ap.add_argument("--ledger", default=None, help="ledger path (default <out>/ledger.json)")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = Path(args.out).resolve()
    if args.ledger is None:
        args.ledger = str(OUT / "ledger.json")
    if not os.environ.get("NM_REFERENCE_ROOT"):
        sys.exit("NM_REFERENCE_ROOT is not set")
    nm_render = os.environ.get("NM_RENDER", str(ROOT / "target" / "release" / "nm-render"))
    OUT.mkdir(parents=True, exist_ok=True)
    cases = discover(args.cases)
    golden_dir = Path(args.golden_dir).resolve() if args.golden_dir else OUT
    mode = "graph" if args.from_graph else ("captured" if args.captured_host_inputs else "native")

    if not args.skip_mint and args.golden_dir is None:
        print(f"[sweep] minting {len(cases)} goldens", file=sys.stderr)
        mint(cases, args.chunk)
    if not args.skip_render:
        label = {"graph": "graph + captured host inputs", "captured": "dsl + captured host inputs",
                 "native": "dsl + native host inputs"}[mode]
        print(f"[sweep] rendering {len(cases)} candidates ({label})", file=sys.stderr)
        render(cases, nm_render, mode, golden_dir)
    rows = grade(cases, golden_dir)
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
