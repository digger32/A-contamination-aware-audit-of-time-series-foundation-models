#!/usr/bin/env python3
"""Merge per-unit JSONs into <outdir>/results.csv and emit the per-model
clean/contaminated split scaffold.

The contamination column is joined from configs/contamination_verified.yaml if it
exists, otherwise from configs/contamination.yaml with EVERY status downgraded to
'provisional:<status>' — the stats step refuses to build the clean/contaminated
figures from provisional codes (level-2 rule), while wins/ranks over ALL datasets
still proceed.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def contamination_lookup():
    ver = ROOT / "configs" / "contamination_verified.yaml"
    prov = ROOT / "configs" / "contamination.yaml"
    if ver.exists():
        cfg, verified = yaml.safe_load(ver.read_text()), True
    else:
        cfg, verified = yaml.safe_load(prov.read_text()), False
    table = cfg.get("models", {})

    def status(model, dataset):
        cell = (table.get(model) or {}).get(dataset, "unknown")
        # cells may be a plain status string, or a {status, evidence} mapping —
        # the assembled verified matrix keeps the source alongside each code so
        # the manuscript can cite it, so both forms must be accepted here.
        s = cell.get("status", "unknown") if isinstance(cell, dict) else cell
        return s if verified else f"provisional:{s}"
    return status, verified


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    a = ap.parse_args()
    outdir = Path(a.outdir)

    status, verified = contamination_lookup()
    rows = []
    for p in sorted(outdir.glob("*__*__*__seed*.json")):
        u = json.loads(p.read_text())
        m = u["metrics"]
        rows.append({
            "suite": u["suite"], "dataset": u["dataset"], "model": u["model"],
            "seed": u["seed"], "device": u["device"], "eval_hash": u["eval_hash"],
            "horizon": u["horizon"], "mase_mean": m["mase_mean"],
            "mase_median": m["mase_median"], "smape_mean": m["smape_mean"],
            "n_series": m["n_series_scored"], "oom_backoffs": u["oom_backoffs"],
            "peak_vram_gb": u["peak_vram_gb"], "wall_s": u["wall_s"],
            "contamination": status(u["model"], u["dataset"]),
        })
    if not rows:
        sys.exit(f"[aggregate] no unit JSONs in {outdir}")
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "results.csv", index=False)
    (outdir / "contamination_status.json").write_text(json.dumps(
        {"verified_matrix_used": verified,
         "n_units": len(df),
         "counts": df["contamination"].value_counts().to_dict()}, indent=2))
    print(f"[aggregate] {len(df)} units -> {outdir/'results.csv'} | "
          f"contamination matrix: {'VERIFIED' if verified else 'PROVISIONAL only'}")


if __name__ == "__main__":
    main()
