#!/usr/bin/env python3
"""Diff a run's GIFT-Eval numbers against the official leaderboard values for
the same checkpoints, producing external/validation.json and a printed table.

This is the paper's external-validity evidence for the PROTOCOL itself: if our
seasonal-naive, AutoARIMA and foundation-model numbers reproduce the official
per-dataset MASE, the wrappers and the evaluation windows are right. Exact
agreement is not expected everywhere — we cap each config at max_series_per_config
series (seeded subset), so datasets with more instances than the cap will differ
by a sampling margin; that is reported rather than hidden.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.05  # |diff| in MASE beyond which we call it a mismatch worth explaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--external", default=str(ROOT / "external"))
    a = ap.parse_args()
    outdir, ext = Path(a.outdir), Path(a.external)

    ref = json.loads((ext / "gifteval_reference_mase.json").read_text())
    ours = defaultdict(dict)
    for r in csv.DictReader(open(outdir / "results.csv")):
        if r["suite"] != "gifteval" or not r["mase_mean"]:
            continue
        ours[r["model"]].setdefault(r["dataset"].replace("+", "/"), []).append(
            float(r["mase_mean"]))
    ours = {m: {d: sum(v) / len(v) for d, v in dd.items()} for m, dd in ours.items()}

    report, n_cmp, n_match = {}, 0, 0
    print(f"{'model':20s} {'config':28s} {'official':>9s} {'ours':>8s} {'diff':>8s}")
    for m, dd in sorted(ours.items()):
        if m not in ref:
            continue
        rows = []
        for d, v in sorted(dd.items()):
            if d not in ref[m]:
                continue
            diff = v - ref[m][d]
            n_cmp += 1; n_match += abs(diff) <= TOL
            rows.append({"dataset": d, "official": ref[m][d], "ours": v,
                         "diff": diff, "within_tol": abs(diff) <= TOL})
            print(f"{m:20s} {d:28s} {ref[m][d]:9.3f} {v:8.3f} {diff:+8.3f}"
                  f"{'' if abs(diff) <= TOL else '  <-- check'}")
        report[m] = rows
    payload = {"tolerance_mase": TOL, "n_compared": n_cmp,
               "n_within_tolerance": n_match,
               "agreement_rate": round(n_match / n_cmp, 4) if n_cmp else None,
               "per_model": report}
    (ext / "validation.json").write_text(json.dumps(payload, indent=2))
    print(f"\n[validate] {n_match}/{n_cmp} comparisons within {TOL} MASE "
          f"-> {ext/'validation.json'}")


if __name__ == "__main__":
    main()
