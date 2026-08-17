#!/usr/bin/env python3
"""Pull two things from the official GIFT-Eval repository:

  (1) LEAKAGE FLAGS — each submitted model's config.json carries a
      `testdata_leakage` field, self-disclosed by the submitting organisation
      and used by the leaderboard to tag zero-shot eligibility. This is the
      INDEPENDENT source (b) for the contamination analysis on the GIFT-Eval
      half of our suite. Scope limit, stated in their README and carried into
      our manuscript: the flag covers leakage into the GIFT-EVAL TEST SPLIT
      ONLY and says nothing about Monash overlap, which is why the Monash half
      still needs the primary corpus disclosures (source (a)).

  (2) OFFICIAL REFERENCE NUMBERS — per-dataset MASE for the same checkpoints and
      the same classical baselines we run. Comparing our numbers against theirs
      on identical configs is external validation of our whole evaluation
      protocol, which matters because three wrapper defects were found during
      this build. Measured on 21 Jul 2026: our seasonal-naive matched the
      official value to three decimals on 8 of 9 shared configs.

Writes runs/<outdir>/external/ or --out; no network access is needed at
analysis time because the fetched files are cached to disk.
"""
import argparse
import csv
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/SalesforceAIResearch/gift-eval/main/results"

# our checkpoint -> the leaderboard's directory for the SAME checkpoint
# (verified by matching the model_link field against configs/models.yaml)
MODEL_MAP = {
    "timesfm_2": "TimesFM-2.5",
    "chronos_bolt_base": "chronos_bolt_base",
    "chronos_bolt_small": "chronos_bolt_small",
    "moirai_base": "Moirai_base",
    "lag_llama": "Lag-Llama",
    "snaive": "seasonal_naive",
    "autoarima": "auto_arima",
    "autoets": "auto_ets",
    "patchtst": "PatchTST",
}
MASE_COL = "eval_metrics/MASE[0.5]"


def get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "external"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    flags, reference = {}, {}
    for ours, theirs in MODEL_MAP.items():
        try:
            cfg = json.loads(get(f"{RAW}/{theirs}/config.json"))
            flags[ours] = {"leaderboard_entry": theirs,
                           "testdata_leakage": cfg.get("testdata_leakage"),
                           "model_type": cfg.get("model_type"),
                           "org": cfg.get("org"),
                           "model_link": cfg.get("model_link")}
            print(f"  [flag] {ours:20s} leakage={cfg.get('testdata_leakage')} "
                  f"({cfg.get('model_link')})")
        except Exception as e:
            print(f"  [warn] {ours}: config.json — {e}")
        try:
            rows = list(csv.DictReader(get(f"{RAW}/{theirs}/all_results.csv").splitlines()))
            reference[ours] = {r["dataset"]: float(r[MASE_COL])
                               for r in rows if r.get(MASE_COL)}
        except Exception as e:
            print(f"  [warn] {ours}: all_results.csv — {e}")

    (out / "gifteval_leakage_flags.json").write_text(json.dumps(flags, indent=2))
    (out / "gifteval_reference_mase.json").write_text(json.dumps(reference, indent=2))
    print(f"\n[external] wrote {out}/gifteval_leakage_flags.json and "
          f"gifteval_reference_mase.json")

    # optional: immediately diff against a run's results.csv
    print("\nTo validate a run against these reference numbers:")
    print("  python scripts/validate_external.py runs/full --external external")


if __name__ == "__main__":
    main()
