#!/usr/bin/env python3
"""Review-proofing gate. Reads a run directory (run_meta.json, manifest.jsonl,
per-unit outputs) and exits non-zero if anything would make the numbers unsafe
to report.

  A1  clean final run      --no-resume, no skips, single run_started, no fail/timeout
  B1  external validity    every declared claim has runs on its independent datasets
  E1  stats artifacts      stats/omnibus.json + stats/posthoc.json + stats/wins.json
  T1  eval-window freeze   ONE eval_hash per (suite, dataset) across all models/seeds
  C1  contamination split  stats/contamination_split.json exists AND verified:true
  D1  device discipline    every unit of a model ran on the single declared device
  V1  cell coverage        no (dataset, model) unit missing from the grid
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load(outdir):
    manifest = []
    mf = outdir / "manifest.jsonl"
    if mf.exists():
        manifest = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
    units = []
    for p in outdir.glob("*__*__*__seed*.json"):
        try:
            units.append(json.loads(p.read_text()))
        except Exception:
            pass
    return manifest, units


def check_A1(outdir, manifest):
    mp = outdir / "run_meta.json"
    if not mp.exists():
        return False, "run_meta.json missing"
    meta = json.loads(mp.read_text())
    if not meta.get("no_resume"):
        return False, "final pass ran WITHOUT --no-resume"
    bad = [r["unit"] for r in manifest if r.get("status") != "ok"]
    if bad:
        return False, f"{len(bad)} unit(s) not ok: {bad[:3]}..."
    stale = [r["unit"] for r in manifest if r.get("started") != meta.get("run_started")]
    if stale:
        return False, f"{len(stale)} carry-over unit(s): {stale[:3]}..."
    n_expected = meta.get("n_units")
    if n_expected is not None and len(manifest) != n_expected:
        return False, f"manifest has {len(manifest)} records, run declared {n_expected}"
    return True, f"clean: --no-resume, {len(manifest)} units, all ok, single run"


def check_B1(units, cfg):
    claims = cfg.get("comparative_claims", [])
    if not claims:
        return False, "no comparative_claims declared"
    have = {(u["suite"], u["dataset"]) for u in units} | \
           {u["dataset"] for u in units}
    fails = []
    for c in claims:
        if c.get("waive"):
            if not c.get("waiver_justification"):
                fails.append(f"'{c['id']}' waived without justification")
            continue
        needed = c.get("independent_datasets", [])
        hit = [d for d in needed if d in have or tuple(d.split("/")) in have]
        if not needed:
            fails.append(f"'{c['id']}' lists no independent_datasets")
        elif not hit:
            fails.append(f"'{c['id']}' has no run on any of {needed}")
    return (False, "; ".join(fails)) if fails else \
           (True, f"{len(claims)} claim(s) covered")


def check_E1(outdir, cfg):
    arts = cfg.get("stats_artifacts",
                   ["stats/omnibus.json", "stats/posthoc.json", "stats/wins.json"])
    missing = [x for x in arts if not (outdir / x).exists()]
    return (False, f"missing: {missing}") if missing else (True, f"present: {arts}")


def check_W1_evalhash(units):
    seen = defaultdict(set)
    for u in units:
        seen[(u["suite"], u["dataset"])].add(u["eval_hash"])
    bad = {k: sorted(v) for k, v in seen.items() if len(v) > 1}
    if bad:
        return False, f"eval_hash differs within {list(bad)[:3]} — models saw different windows"
    return True, f"one eval_hash per dataset across {len(seen)} datasets"


def check_C1_contamination(outdir):
    p = outdir / "stats" / "contamination_split.json"
    if not p.exists():
        return False, "stats/contamination_split.json missing"
    payload = json.loads(p.read_text())
    if not payload.get("verified"):
        return False, "contamination matrix still PROVISIONAL — human-verify it first"
    return True, "verified clean/contaminated split present"


def check_D1_device(units, cfg_models):
    bad = []
    for u in units:
        want = cfg_models.get(u["model"], {}).get("device")
        if want and u.get("device") != want:
            bad.append(u["model"])
    if bad:
        return False, f"device drift in: {sorted(set(bad))}"
    return True, "single declared device per model"


def check_V1_coverage(outdir, cfg):
    """No (dataset, model) cell may be missing. In the full run six AutoARIMA
    timeouts silently removed six datasets from EVERY statistic — and they were
    the datasets where the foundation models did best, so the loss biased the
    result. Declare genuinely impossible cells in gate_config.yaml under
    allowed_missing_cells with a justification; anything else fails."""
    p = outdir / "stats" / "coverage.json"
    if not p.exists():
        return False, "stats/coverage.json missing — rerun the stats step"
    cov = json.loads(p.read_text())
    allowed = cfg.get("allowed_missing_cells", {})
    offending = {d: ms for d, ms in cov.get("missing_cells", {}).items()
                 if [m for m in ms if m not in allowed.get(d, [])]}
    if offending:
        return False, (f"{len(offending)} dataset(s) with undeclared missing units: "
                       f"{dict(list(offending.items())[:3])}")
    return True, (f"all cells present across {cov['n_datasets_total']} datasets"
                  + (f" ({len(allowed)} declared exceptions)" if allowed else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    outdir = Path(a.outdir)
    root = Path(__file__).resolve().parents[1]
    cfgp = Path(a.config) if a.config else root / "configs" / "gate_config.yaml"
    cfg = yaml.safe_load(cfgp.read_text()) if cfgp.exists() else {}
    models_cfg = yaml.safe_load((root / "configs" / "models.yaml").read_text())["models"]

    manifest, units = load(outdir)
    results = [
        ("A1 clean-final-run", *check_A1(outdir, manifest)),
        ("B1 external-validity", *check_B1(units, cfg)),
        ("E1 stats-artifacts", *check_E1(outdir, cfg)),
        ("W1 eval-window-freeze", *check_W1_evalhash(units)),
        ("C1 contamination-verified", *check_C1_contamination(outdir)),
        ("D1 device-discipline", *check_D1_device(units, models_cfg)),
        ("V1 cell-coverage", *check_V1_coverage(outdir, cfg)),
    ]
    print("=" * 68)
    print(f"REVIEW-PROOFING GATE | outdir={outdir}")
    print("=" * 68)
    ok = True
    for name, passed, msg in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name:26s} {msg}")
        ok = ok and passed
    print("=" * 68)
    if not ok:
        print("GATE FAILED — do not freeze these numbers into figures.")
        sys.exit(1)
    print("GATE PASSED — numbers are clean to freeze.")


if __name__ == "__main__":
    main()
