#!/usr/bin/env python3
"""Statistics over <outdir>/results.csv. Emits under <outdir>/stats/:
  omnibus.json   Friedman over models across datasets (pooled + per suite)
  posthoc.json   Nemenyi matrix + avg ranks (input to the CD diagram)
  wins.json      per-model: datasets where it beats snaive; Wilcoxon vs snaive;
                 bootstrap CI of the paired mean MASE difference vs snaive
  contamination_split.json   per-model aggregates on clean vs contaminated
                 subsets — ONLY when the verified matrix was used; otherwise a
                 refusal record (provisional codes never reach a figure).
Unit of analysis: dataset (per-dataset mean MASE, averaged over seeds first).
"""
import argparse
import json
from pathlib import Path

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats as st

BASELINE = "snaive"
B_BOOT = 10000
RNG = np.random.default_rng(0)


def pivot(df):
    g = (df.groupby(["suite", "dataset", "model"], as_index=False)["mase_mean"]
           .mean())
    return g.pivot_table(index=["suite", "dataset"], columns="model",
                         values="mase_mean")


def coverage_report(wide, sdir):
    """Which (dataset, model) cells are missing. The full run lost six datasets
    from EVERY statistic because one model (AutoARIMA) timed out on them and the
    old code dropped any row with a missing cell — and those datasets were
    exactly where the foundation models did best, so the loss was not neutral.
    Coverage is now reported explicitly and asserted by the gate."""
    missing = {}
    for (suite, ds), row in wide.iterrows():
        absent = [m for m in wide.columns if pd.isna(row[m])]
        if absent:
            missing[f"{suite}/{ds}"] = absent
    payload = {"n_datasets_total": len(wide),
               "n_datasets_complete": int((~wide.isna().any(axis=1)).sum()),
               "missing_cells": missing,
               "models_with_full_coverage":
                   [m for m in wide.columns if not wide[m].isna().any()]}
    (sdir / "coverage.json").write_text(json.dumps(payload, indent=2))
    return payload


def boot_ci(diff):
    idx = RNG.integers(0, len(diff), (B_BOOT, len(diff)))
    means = diff[idx].mean(axis=1)
    return [float(np.quantile(means, q)) for q in (0.025, 0.975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    sdir = outdir / "stats"; sdir.mkdir(exist_ok=True)

    df = pd.read_csv(outdir / "results.csv")
    wide_all = pivot(df)                         # all datasets, NaN where a unit failed
    cov = coverage_report(wide_all, sdir)
    wide = wide_all.dropna(axis=0, how="any")    # complete cases: rank tests only
    models = list(wide_all.columns)
    if cov["missing_cells"]:
        print(f"[stats] WARNING: {len(cov['missing_cells'])} dataset(s) have missing "
              f"cells -> rank tests use {cov['n_datasets_complete']} complete "
              f"datasets; wins/Wilcoxon use pairwise-complete data. "
              f"See stats/coverage.json")

    # --- omnibus: Friedman across datasets (complete cases, by construction) - #
    omnibus = {}
    for scope, w in [("pooled", wide)] + [(s, wide.loc[[s]])
                                          for s in wide.index.get_level_values(0).unique()]:
        if len(w) >= 3 and len(models) >= 3:
            fr = st.friedmanchisquare(*[w[m].to_numpy() for m in models])
            omnibus[scope] = {"n_datasets": len(w), "statistic": float(fr.statistic),
                              "p_value": float(fr.pvalue),
                              "n_datasets_excluded_incomplete": len(wide_all) - len(wide)}
    (sdir / "omnibus.json").write_text(json.dumps(omnibus, indent=2))

    # --- post-hoc: Nemenyi + average ranks --------------------------------- #
    ranks = wide.rank(axis=1).mean(axis=0)
    try:
        import scikit_posthocs as sp
        nem = sp.posthoc_nemenyi_friedman(wide.to_numpy())
        nem.index = nem.columns = models
        posthoc = {"avg_ranks": {m: float(ranks[m]) for m in models},
                   "nemenyi_p": {m: {k: float(nem.loc[m, k]) for k in models}
                                 for m in models},
                   "n_datasets": len(wide)}
    except ImportError:
        posthoc = {"avg_ranks": {m: float(ranks[m]) for m in models},
                   "nemenyi_p": None, "note": "scikit-posthocs not installed"}
    (sdir / "posthoc.json").write_text(json.dumps(posthoc, indent=2))

    # --- wins vs seasonal-naive + Wilcoxon + bootstrap CI ------------------- #
    # PAIRWISE-complete: each model is compared with the baseline on every
    # dataset where BOTH ran, so one model's failure no longer erases a dataset
    # from another model's comparison.
    wins = {}
    for m in models:
        if m == BASELINE:
            continue
        pair = wide_all[[m, BASELINE]].dropna()
        diff = (pair[m] - pair[BASELINE]).to_numpy()   # <0: model beats snaive
        if len(diff) == 0:
            continue
        entry = {"n_datasets": len(diff),
                 "wins": int((diff < 0).sum()),
                 "losses": int((diff > 0).sum()),
                 "loss_datasets": [f"{s}/{d}" for (s, d), v
                                   in zip(pair.index, diff) if v > 0],
                 "mean_diff": float(diff.mean()),
                 "mean_diff_ci95": boot_ci(diff)}
        if len(diff) >= 6 and np.any(diff != 0):
            entry["wilcoxon_p"] = float(st.wilcoxon(diff).pvalue)
        wins[m] = entry
    (sdir / "wins.json").write_text(json.dumps(wins, indent=2))

    # --- head-to-head vs the BEST classical baseline per dataset ------------ #
    # The harder and more informative comparison: seasonal-naive is a floor,
    # the real question is whether a foundation model beats the best of the
    # tuned classical methods available on that dataset.
    CLASSICAL = [c for c in ("snaive", "autoets", "autoarima") if c in models]
    best_classical = wide_all[CLASSICAL].min(axis=1)
    h2h = {}
    for m in models:
        if m in CLASSICAL:
            continue
        pair = pd.concat([wide_all[m], best_classical.rename("best_classical")],
                         axis=1).dropna()
        d = (pair[m] - pair["best_classical"]).to_numpy()
        if len(d) == 0:
            continue
        h2h[m] = {"n_datasets": len(d),
                  "wins": int((d < 0).sum()), "losses": int((d > 0).sum()),
                  "loss_datasets": [f"{s}/{ds}" for (s, ds), v
                                    in zip(pair.index, d) if v > 0],
                  "mean_diff": float(d.mean()), "mean_diff_ci95": boot_ci(d)}
        if len(d) >= 6 and np.any(d != 0):
            h2h[m]["wilcoxon_p"] = float(st.wilcoxon(d).pvalue)
    (sdir / "head_to_head_classical.json").write_text(json.dumps(h2h, indent=2))

    # --- clean/contaminated split (verified matrix only) -------------------- #
    # Comparison is the RATIO of the model's MASE to the best classical method on
    # the SAME dataset, not raw MASE. Raw means are confounded by which datasets
    # land in which group: covid_deaths alone carries MASE ~6 for every method,
    # so a group containing it looks bad regardless of contamination. The ratio
    # removes dataset difficulty and is what the claim is about.
    cst = json.loads((outdir / "contamination_status.json").read_text())
    if cst["verified_matrix_used"]:
        CLASSICAL_REF = [c for c in ("snaive", "autoets", "autoarima") if c in models]
        best_cl = wide_all[CLASSICAL_REF].min(axis=1)
        contam = {}
        for r in df.itertuples():
            contam[(r.suite, r.dataset, r.model)] = r.contamination
        rows = {}
        for m in models:
            if m in CLASSICAL_REF:
                continue
            groups = defaultdict(list)
            for (suite, ds) in wide_all.index:
                v, b = wide_all.loc[(suite, ds), m], best_cl.loc[(suite, ds)]
                cstat = contam.get((suite, ds, m), "unknown")
                if pd.notna(v) and pd.notna(b) and b > 0:
                    groups[cstat].append(v / b)
            rows[m] = {cstat: {"n": len(vals),
                               "ratio_to_best_classical": round(float(np.mean(vals)), 4),
                               "wins": int(sum(1 for x in vals if x < 1))}
                       for cstat, vals in sorted(groups.items())}
        payload = {"verified": True,
                   "metric": "mean(model MASE / best classical MASE on the same "
                             "dataset); below 1 means the model wins",
                   "note": "'unknown' holds the GIFT-Eval configs, which the Monash "
                           "corpus matrix does not cover, plus any undetermined cell",
                   "per_model": rows}
    else:
        payload = {"verified": False,
                   "refusal": "contamination matrix is PROVISIONAL — verify it into "
                              "configs/contamination_verified.yaml before any "
                              "clean/contaminated figure is built"}
    (sdir / "contamination_split.json").write_text(json.dumps(payload, indent=2))

    print(f"[stats] omnibus/posthoc/wins/contamination_split -> {sdir}")
    for m, e in wins.items():
        p = e.get("wilcoxon_p")
        print(f"  {m:20s} beats {BASELINE} on {e['wins']}/{e['n_datasets']} datasets"
              + (f" | Wilcoxon p={p:.4f}" if p is not None else ""))


if __name__ == "__main__":
    main()
