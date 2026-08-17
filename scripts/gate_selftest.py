#!/usr/bin/env python3
"""Unit-test the gate at Stage 0: build a synthetic CLEAN run (expect exit 0)
and a synthetic DIRTY run (expect exit 1). Fails loudly if either expectation
breaks — run BEFORE any real compute is spent."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "review_gate.py"

MODELS = ["snaive", "chronos_bolt_small"]
DATASETS = [("monash", "nn5_daily"), ("monash", "hospital"),
            ("monash", "fred_md"), ("monash", "covid_deaths"),
            ("monash", "m4_hourly"), ("monash", "electricity_weekly"),
            ("monash", "traffic_weekly"),
            ("gifteval", "bizitobs_l2c+H+short"), ("gifteval", "jena_weather+H+short"),
            ("gifteval", "LOOP_SEATTLE+H+short"), ("gifteval", "restaurant+D+short"),
            ("gifteval", "hierarchical_sales+D+short"), ("gifteval", "solar+H+short"),
            ("gifteval", "ett1+H+short")]


def unit(suite, ds, model, seed, ehash, device):
    return {"suite": suite, "dataset": ds, "model": model, "seed": seed,
            "device": device, "eval_hash": ehash, "horizon": 8,
            "metrics": {"mase_mean": 1.0, "mase_median": 1.0, "smape_mean": 0.2,
                        "n_series_scored": 10, "n_series_degenerate_scale": 0,
                        "per_series_mase": [1.0] * 10},
            "oom_backoffs": 0, "peak_vram_gb": None, "wall_s": 1.0}


def build(outdir: Path, dirty: bool):
    outdir.mkdir(parents=True)
    started = "2026-07-21T00:00:00+00:00"
    manifest = []
    for suite, ds in DATASETS:
        for m in MODELS:
            uid = f"{suite}__{ds}__{m}__seed0"
            ehash = "abc123"
            device = "cpu" if m == "snaive" else "cuda"
            if dirty and ds == "hospital" and m == "snaive":
                ehash = "DRIFTED"          # T1 must catch differing eval windows
            (outdir / f"{uid}.json").write_text(
                json.dumps(unit(suite, ds, m, 0, ehash, device)))
            manifest.append({"unit": uid, "suite": suite, "dataset": ds, "model": m,
                             "seed": 0, "status": "ok", "started": started,
                             "finished": started, "wall_s": 1.0, "no_resume": not dirty})
    if dirty:
        manifest[0]["status"] = "timeout"  # A1 must catch an incomplete unit
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_started": started, "no_resume": not dirty,
        "n_units": len(manifest), "models": MODELS}))
    with (outdir / "manifest.jsonl").open("w") as fh:
        for r in manifest:
            fh.write(json.dumps(r) + "\n")
    sdir = outdir / "stats"; sdir.mkdir()
    arts = ["omnibus.json", "posthoc.json", "wins.json",
            "head_to_head_classical.json"]
    for name in arts:
        (sdir / name).write_text("{}")
    (sdir / "coverage.json").write_text(json.dumps(
        {"n_datasets_total": len(DATASETS), "n_datasets_complete": len(DATASETS),
         "missing_cells": {} if not dirty
                          else {"monash/hospital": ["autoarima"]},  # T4 must catch
         "models_with_full_coverage": MODELS}))
    if not dirty:
        (sdir / "contamination_split.json").write_text(
            json.dumps({"verified": True, "per_model": {}}))
    # dirty run: contamination_split.json entirely missing (the gate must catch it)


def run_gate(outdir: Path) -> int:
    p = subprocess.run([sys.executable, str(GATE), str(outdir)],
                       capture_output=True, text=True)
    print(p.stdout)
    return p.returncode


def main():
    tmp = Path(tempfile.mkdtemp(prefix="t2_gate_selftest_"))
    try:
        clean, dirty = tmp / "clean", tmp / "dirty"
        build(clean, dirty=False)
        build(dirty, dirty=True)
        rc_clean, rc_dirty = run_gate(clean), run_gate(dirty)
        ok = rc_clean == 0 and rc_dirty != 0
        print(f"[selftest] clean rc={rc_clean} (want 0) | dirty rc={rc_dirty} (want !=0)"
              f" -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
