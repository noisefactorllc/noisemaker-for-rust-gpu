#!/usr/bin/env python3
"""Corpus-wide pixel parity sweep: mint goldens, render candidates, grade, ledger.

Cases are parity/programs/*.dsl (the shared fixture pool), parity/coverage/*.dsl
(the generated effect x mode corpus), parity/portable/*.dsl (user-defined
Portable effects: each registers its <id>.portable.json sidecar, WGSL in
<id>.<program>.wgsl, before its program runs, in the golden page and in the
candidate), parity/timed/*.dsl (the timed tier, generated with the coverage
corpus: every effect that evolves across frames, every oscillator kind and every
shared or curated program running a stateful, particle or simulation effect,
rendered with the timed protocol of parity/timed/manifest.json) and
parity/curated/*.dsl (the sibling ports' author-curated programs, provenance in
parity/curated/sources.json); a case id is the file stem. One run:

  1. MINT    goldens with the reference engine's WebGPU backend
             (parity/batch-golden.mjs), fresh, in this run -- a sweep never grades a
             golden from an earlier run. The minter also saves the host inputs the
             page produced (<id>.<textureId>.png: media/text images, asyncInit
             overlays).
  2. RENDER  candidates with nm-render in one batch process. By default the
             candidate runs the DSL through the port's demo host (the live Rust
             frontend, ProgramState and the demo's control initialization, then
             applyStepParameterValues) and produces every host input natively:
             the demo's default media image (the catalog's byte copy of
             demo/shaders/img/testcard.png), text canvases, asyncInit overlays
             and meshes (the fixture's .obj sidecar into mesh0, as the minter
             loads it); a
             fixture's .midi.json sidecar (raw MIDI messages) reaches a MIDI
             state connected after the program loads, in both engines.
             --captured-host-inputs grades with the minter's saved host-input PNGs
             in place of the natively produced ones (isolates the renderer from
             the host-input ports); --from-graph renders the reference graph the
             golden page rendered with the saved host inputs (isolates the runtime
             from the compiler).
  3. GRADE   each case: decoded RGBA8 pixels compared with the golden.
             exact  -- identical pixels
             strict -- max-abs-diff <= 2.001 (8-bit units) and global SSIM >= 0.98
             near   -- beyond strict but within the case's recorded policy in
                       NEAR_POLICIES (each entry carries its mechanism), or,
                       for a timed case in AMPLIFIED_POLICIES, byte-identical
                       early samples and statistically matching later ones
             fail   -- outside every policy
             missing-- no golden or no candidate (mint or render failure)
             TIMED cases (the timed tier and the stateful solvers below) are
             graded on their sample series: every sample must pass.
             INFORMATIVE: a golden is parity evidence only when it shows
             structure: its most frequent RGBA value covers at most 99% of
             the pixels (INFORMATIVE_MAX_DOMINANT) and its luminance standard
             deviation is at least one 8-bit level (INFORMATIVE_MIN_LUMA_STD).
             A timed case is informative when any of its samples is. A case
             that passes on an uninformative golden is reported apart
             ("uninformative"), never as exact or strict evidence; a failure
             is a failure either way. Every catalog effect needs at least one
             informative case of its own (its coverage fixtures and its timed
             case) graded exact or strict; effects without one are listed.
  4. LEDGER  parity/out/ledger.json (untracked), one row per case, with the
             golden statistics, the informative flag and the summary bucket;
             parity/out/ledger.effects.json maps every catalog effect to its
             informative exact or strict cases (full runs only).

Usage:
  NM_REFERENCE_ROOT=/path/to/noisemaker python3 parity/sweep.py [case-id ...]
      [--captured-host-inputs | --from-graph] [--skip-mint] [--skip-render]
      [--chunk 60] [--out DIR] [--golden-dir DIR]

Env: NM_RENDER (candidate binary, default target/release/nm-render).
Exit 0 iff no case is near, failing or missing and every catalog effect has
informative exact or strict evidence (the effect check runs only when no case
ids are given).
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
INFORMATIVE_MAX_DOMINANT = 0.99
INFORMATIVE_MIN_LUMA_STD = 1.0
CATALOG = ROOT / "crates" / "noisemaker-effects" / "catalog"
TIMED_MANIFEST = ROOT / "parity" / "timed" / "manifest.json"

# Stateful solvers graded on long timed runs (normalized dt = 1/600 per frame),
# matching the family's established split; the values are sample schedules
# (sample_schedule), as the timed tier's manifest protocol is.
TIMED = {
    "navierStokes": {"runSeconds": 30, "sampleEvery": 5},
    "temporalAberration": {"runSeconds": 30, "sampleEvery": 10},
}


def load_timed_manifest():
    """The timed tier: {case id: schedule}, {case id: effect key}."""
    if not TIMED_MANIFEST.exists():
        return {}, {}
    manifest = json.loads(TIMED_MANIFEST.read_text())
    protocol = manifest["protocol"]
    schedules = {cid: dict(protocol) for cid in manifest["cases"]}
    effects = {cid: c.get("effect") for cid, c in manifest["cases"].items()}
    return schedules, effects


def js_number(x):
    """A number as JavaScript prints it (integral values without a fraction)."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def sample_schedule(schedule):
    """[(frame, [labels])] in frame order: the minter's and nm-render's schedule
    (batch-golden.mjs sampleSchedule, FixtureSpec::sample_schedule)."""
    samples = {}

    def add(frame, label):
        if frame <= 0:
            return
        labels = samples.setdefault(frame, [])
        if label not in labels:
            labels.append(label)

    run_seconds = schedule.get("runSeconds", 0)
    every = schedule.get("sampleEvery", 5)
    total = schedule.get("runFrames", 0)
    if run_seconds > 0:
        every_frames = max(1, int(every * 60 + 0.5))
        count = max(1, int((run_seconds * 60) // every_frames))
        for s in range(count):
            add((s + 1) * every_frames, "t" + js_number((s + 1) * every))
        total = max(total, count * every_frames)
    k = schedule.get("sampleEveryFrames", 0)
    if k > 0:
        for f in range(k, total + 1, k):
            add(f, f"f{f}")
    for f in schedule.get("sampleFrames", []):
        add(f, f"f{f}")
    return sorted(samples.items())


def sample_labels(schedule):
    return [label for _, labels in sample_schedule(schedule) for label in labels]


def schedule_args(schedule):
    """batch-golden.mjs options for a schedule."""
    args = []
    if schedule.get("runSeconds", 0) > 0:
        args += ["--run-seconds", js_number(schedule["runSeconds"]), "--sample-every", js_number(schedule.get("sampleEvery", 5))]
    if schedule.get("runFrames", 0) > 0:
        args += ["--run-frames", str(schedule["runFrames"])]
    if schedule.get("sampleEveryFrames", 0) > 0:
        args += ["--sample-every-frames", str(schedule["sampleEveryFrames"])]
    if schedule.get("sampleFrames"):
        args += ["--sample-frames", ",".join(str(f) for f in schedule["sampleFrames"])]
    return args

# Per-case tolerances beyond the strict bar, {case id: {"tolerance", "ssim_min",
# "mechanism"}}. Every entry must name the observed mechanism; entries are
# added only with evidence from this backend. NEAR is outside the published
# contract: scripts/parity-summary still fails on it. No case needs one: on
# Metal the port compiles every shader as the reference's Chromium does (Tint
# at Chromium's Dawn revision generating the MSL, Metal compiling it with
# Dawn's options; crates/noisemaker-tint, crates/noisemaker-gpu/src/backend/
# shaders.rs), which made every earlier compile-option case byte-exact.
NEAR_POLICIES = {}

# Amplified divergence, {case id: {"exact", "mean_tol", "hist_tv",
# "mechanism"}}: a 1-ulp difference in one pass of a simulation that its own
# dynamics amplify until the trajectories decorrelate. Such a case passes as
# NEAR (outside the published contract) only when the samples of `exact` are
# byte-identical (the early trajectory) and every later sample matches the
# golden statistically: per-channel mean within `mean_tol` 8-bit levels and
# 32-bin luminance-histogram total variation within `hist_tv`. No case needs
# one (the flock, life and reaction-diffusion simulations are byte-exact over
# their timed runs).
AMPLIFIED_POLICIES = {}


def discover(ids):
    cases = {}
    for sub in ("programs", "coverage", "portable", "timed", "curated"):
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


def golden_stats(golden):
    """The informativeness statistics of a golden (see the header)."""
    a = np.ascontiguousarray(np.asarray(Image.open(golden).convert("RGBA")))
    _, counts = np.unique(a.view(np.uint32).ravel(), return_counts=True)
    rgb = a[..., :3].astype(np.float64)
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    dominant = float(counts.max() / counts.sum())
    luma_std = float(luma.std())
    return {"distinct": int(counts.size), "dominant": round(dominant, 6), "luma_std": round(luma_std, 4),
            "informative": dominant <= INFORMATIVE_MAX_DOMINANT and luma_std >= INFORMATIVE_MIN_LUMA_STD}


def histogram_tv(a, b):
    """Total variation distance of the 32-bin luminance histograms."""
    def hist(x):
        luma = 255.0 * (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
        h, _ = np.histogram(luma, bins=32, range=(0.0, 256.0))
        return h / h.sum()
    return float(0.5 * np.abs(hist(a) - hist(b)).sum())


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
        "mean_diff": float(np.abs(a.reshape(-1, 4).mean(axis=0) - b.reshape(-1, 4).mean(axis=0)).max() * 255.0),
        "hist_tv": histogram_tv(a, b),
    }


def host_textures(case_id, golden_dir):
    found = {}
    prefix = case_id + "."
    for p in sorted(golden_dir.glob(prefix + "*.png")):
        tex = p.name[len(prefix):-len(".png")]
        if tex.split(".")[0] in ("golden", "candidate"):
            continue
        found[tex] = str(p)
    return found


def mint(cases, chunk, schedules):
    single = [str(p) for i, p in cases.items() if i not in schedules]
    rc = 0
    if single:
        listing = OUT / "mint-list.txt"
        listing.write_text("\n".join(single) + "\n")
        rc |= subprocess.call(["node", str(ROOT / "parity" / "batch-golden.mjs"), str(OUT),
                               "--size", str(SIZE), "--time", str(TIME), "--frames", str(FRAMES),
                               "--chunk-size", str(chunk), "--list", str(listing)])
    # One minter run per distinct timed schedule.
    groups = {}
    for case_id, path in cases.items():
        if case_id in schedules:
            key = json.dumps(schedules[case_id], sort_keys=True)
            groups.setdefault(key, []).append(str(path))
    for n, (key, paths) in enumerate(sorted(groups.items())):
        listing = OUT / f"mint-list-timed-{n}.txt"
        listing.write_text("\n".join(paths) + "\n")
        rc |= subprocess.call(["node", str(ROOT / "parity" / "batch-golden.mjs"), str(OUT),
                               "--size", str(SIZE), "--chunk-size", str(chunk), "--list", str(listing)]
                              + schedule_args(json.loads(key)))
    return rc


def render(cases, nm_render, mode, golden_dir, schedules):
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
        if case_id in schedules:
            entry.update(schedules[case_id])
        manifest.append(entry)
    for case_id in cases:
        for p in OUT.glob(case_id + ".candidate*.png"):
            p.unlink()
    path = OUT / "render-manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    return subprocess.call([nm_render, "batch", str(path)])


def amplified_ok(row, policy):
    """The amplified-divergence criterion (AMPLIFIED_POLICIES)."""
    samples = row.get("samples") or []
    if not samples:
        return False
    for smp in samples:
        if smp["label"] in policy["exact"]:
            if not smp["exact"]:
                return False
        elif smp["mean_diff"] > policy["mean_tol"] or smp["hist_tv"] > policy["hist_tv"]:
            return False
    return True


def verdict_of(row, policy, amplified=None):
    if row["exact"]:
        return "EXACT"
    if row["max_abs_diff"] <= STRICT_TOL and row["ssim"] >= STRICT_SSIM:
        return "STRICT"
    if amplified is not None:
        return "NEAR" if amplified_ok(row, amplified) else "FAIL"
    if row["max_abs_diff"] <= policy["tolerance"] and row["ssim"] >= policy["ssim_min"]:
        return "NEAR"
    return "FAIL"


def grade(cases, golden_dir, schedules):
    rows = []
    for case_id in cases:
        policy = NEAR_POLICIES.get(case_id, {"tolerance": STRICT_TOL, "ssim_min": STRICT_SSIM})
        row = {"program": case_id, "policy": {"tolerance": policy["tolerance"], "ssim_min": policy["ssim_min"]}}
        if case_id in schedules:
            samples = []
            for label in sample_labels(schedules[case_id]):
                g = golden_dir / f"{case_id}.golden.{label}.png"
                c = OUT / f"{case_id}.candidate.{label}.png"
                if not g.exists() or not c.exists():
                    samples = None
                    break
                m = grade_pair(g, c)
                m.update({"golden": rel(g), "candidate": rel(c), "label": label, "golden_stats": golden_stats(g)})
                samples.append(m)
            if samples is None or any("error" in s for s in samples):
                row.update({"verdict": "MISSING", "informative": False})
            else:
                row.update({"samples": samples, "exact": all(s["exact"] for s in samples),
                            "max_abs_diff": max(s["max_abs_diff"] for s in samples),
                            "ssim": min(s["ssim"] for s in samples),
                            "informative": any(s["golden_stats"]["informative"] for s in samples)})
                worst = max(samples, key=lambda s: (not s["exact"], s["max_abs_diff"], -s["ssim"]))
                row["worst_sample"] = worst["label"]
                first_off = next((s["label"] for s in samples if not s["exact"]), None)
                if first_off is not None:
                    row["first_inexact_sample"] = first_off
        else:
            g = golden_dir / f"{case_id}.golden.png"
            c = OUT / f"{case_id}.candidate.png"
            if not g.exists() or not c.exists():
                row.update({"verdict": "MISSING", "golden_present": g.exists(), "candidate_present": c.exists(),
                            "informative": False})
            else:
                stats = golden_stats(g)
                row.update({"golden_stats": stats, "informative": stats["informative"]})
                m = grade_pair(g, c)
                if "error" in m:
                    row.update({"verdict": "FAIL", **m})
                else:
                    row.update(m)
        amplified = AMPLIFIED_POLICIES.get(case_id)
        if amplified is not None:
            row["policy"] = {k: v for k, v in amplified.items() if k != "mechanism"}
        if "verdict" not in row:
            row["verdict"] = verdict_of(row, policy, amplified)
            if row["verdict"] == "NEAR":
                row["mechanism"] = (amplified or policy).get("mechanism")
        rows.append(row)
    return rows


def catalog_effects():
    """{effect key: coverage stem} for every catalog effect."""
    effects = {}
    for d in sorted(CATALOG.glob("*/*/definition.json")):
        ns, name = d.parent.parent.name, d.parent.name
        func = json.loads(d.read_text()).get("func") or name
        effects[f"{ns}.{func}"] = f"{ns}_{func}"
    return effects


def effect_evidence(rows, timed_effects):
    """({effect key: [informative exact/strict case ids of its own]},
    {effect key: [its informative cases]}) over the effect's coverage fixtures
    (<ns>_<func>, <ns>_<func>__*) and its timed cases."""
    by_case = {r["program"]: r for r in rows}
    evidence = {}
    informative = {}
    for key, stem in catalog_effects().items():
        own = [c for c in by_case if c == stem or c.startswith(stem + "__")]
        own += [c for c, e in timed_effects.items() if e == key and c in by_case and not c.startswith("timed_osc_")]
        informative[key] = sorted(c for c in own if by_case[c]["informative"])
        evidence[key] = [c for c in informative[key] if by_case[c]["verdict"] in ("EXACT", "STRICT")]
    return evidence, informative


def effects_path(ledger):
    """Where the per-effect evidence of a ledger goes: <ledger stem>.effects.json
    beside a .json ledger, <ledger>.effects.json otherwise."""
    text = str(ledger)
    return Path(text[:-len(".json")] + ".effects.json" if text.endswith(".json") else text + ".effects.json")


def bucket(row):
    """The case's summary bucket: a pass on an uninformative golden is not
    evidence."""
    v = row["verdict"]
    if v in ("EXACT", "STRICT") and not row.get("informative"):
        return "UNINFORMATIVE"
    return v


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
    timed_schedules, timed_effects = load_timed_manifest()
    schedules = {**TIMED, **timed_schedules}
    schedules = {c: s for c, s in schedules.items() if c in cases}

    if not args.skip_mint and args.golden_dir is None:
        print(f"[sweep] minting {len(cases)} goldens", file=sys.stderr)
        mint(cases, args.chunk, schedules)
    if not args.skip_render:
        label = {"graph": "graph + captured host inputs", "captured": "dsl + captured host inputs",
                 "native": "dsl + native host inputs"}[mode]
        print(f"[sweep] rendering {len(cases)} candidates ({label})", file=sys.stderr)
        render(cases, nm_render, mode, golden_dir, schedules)
    rows = grade(cases, golden_dir, schedules)
    for r in rows:
        r["bucket"] = bucket(r)
    full = not args.cases
    evidence, informative_cases = effect_evidence(rows, timed_effects) if full else ({}, {})
    missing_effects = sorted(k for k, v in evidence.items() if not v)
    Path(args.ledger).write_text(json.dumps(rows, indent=1) + "\n")
    if full:
        effects_path(args.ledger).write_text(json.dumps(evidence, indent=1) + "\n")
    counts = {}
    buckets = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        buckets[r["bucket"]] = buckets.get(r["bucket"], 0) + 1
    for r in rows:
        if r["verdict"] in ("FAIL", "MISSING", "NEAR"):
            detail = f"max={r.get('max_abs_diff', float('nan')):.3f} ssim={r.get('ssim', float('nan')):.5f}" \
                if "max_abs_diff" in r else json.dumps({k: v for k, v in r.items() if k not in ("program", "policy", "verdict", "bucket")})
            if "first_inexact_sample" in r:
                detail += f" first-inexact={r['first_inexact_sample']} worst={r['worst_sample']}"
            print(f"[{r['verdict']}] {r['program']}: {detail}")
    uninformative = sorted(r["program"] for r in rows if not r.get("informative") and r["verdict"] != "MISSING")
    for c in uninformative:
        r = next(x for x in rows if x["program"] == c)
        stats = r.get("golden_stats") or max((s["golden_stats"] for s in r.get("samples", [])),
                                             key=lambda st: st["luma_std"], default={})
        print(f"[UNINFORMATIVE] {c}: {r['verdict'].lower()} on a golden with dominant={stats.get('dominant')} "
              f"luma_std={stats.get('luma_std')} distinct={stats.get('distinct')}")
    by_case = {r["program"]: r for r in rows}
    for k in missing_effects:
        cases = informative_cases.get(k, [])
        if cases:
            print(f"[NO-EVIDENCE] {k}: its informative cases are outside the contract: " +
                  ", ".join(f"{c} ({by_case[c]['verdict'].lower()})" for c in cases))
        else:
            print(f"[NO-EVIDENCE] {k}: no informative case of its own")
    print("SWEEP " + json.dumps(counts, sort_keys=True))
    informative_counts = {k: v for k, v in buckets.items()}
    print("SWEEP-INFORMATIVE " + json.dumps(informative_counts, sort_keys=True))
    if full:
        print("SWEEP-EFFECTS " + json.dumps({"effects": len(evidence), "evidenced": len(evidence) - len(missing_effects),
                                             "missing": len(missing_effects),
                                             "uninformed": sum(1 for k in missing_effects if not informative_cases[k])},
                                            sort_keys=True))
    ok = all(r["verdict"] in ("EXACT", "STRICT") for r in rows) and not missing_effects
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
