#!/usr/bin/env python3
"""Assemble configs/contamination_verified.yaml from the primary sources already
gathered, so the human decision is reduced to the handful of genuinely ambiguous
cells rather than 75 of them.

Sources, per model (each cell carries the file it came from):
  chronos_bolt_{small,base}  the authors' in-domain / zero-shot config split
  lag_llama                  data/dataset_list.py ALL_DATASETS
  timesfm_2                  membership in Salesforce/GiftEvalPretrain, the
                             corpus named on the timesfm-2.5 model card
  moirai_base                membership in Salesforce/lotsa_data (LOTSA)

CANDIDATE cells (a corpus member whose name differs only in missing-value
treatment, e.g. nn5_daily_with_missing vs our nn5_daily) are written as
`unknown` by default. That is the conservative reading. If you judge the
underlying series to be the same data — which is the usual conclusion, since the
missing-value variant is the same source series — pass --candidates-as
contaminated and the file records that decision explicitly.

Run after check_corpus_membership.py and draft_contamination.py. Review the
result, then it is ready for the gate (provisional: false is written only when
--approve is passed, so nothing is promoted behind your back).
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-as", choices=["unknown", "contaminated", "clean"],
                    default="unknown")
    ap.add_argument("--approve", action="store_true",
                    help="write provisional: false — only after you have read the cells")
    a = ap.parse_args()

    mem = json.loads((ROOT / "configs" / "corpus_membership.json").read_text())
    draft = yaml.safe_load((ROOT / "configs" / "contamination_draft.yaml").read_text())
    out = {"provisional": not a.approve,
           "candidates_resolved_as": a.candidates_as,
           "sources": {
               "chronos_bolt_small": "amazon-science/chronos-forecasting :: "
                                     "scripts/evaluation/configs/{in-domain,zero-shot}.yaml",
               "chronos_bolt_base": "same as chronos_bolt_small",
               "lag_llama": "time-series-foundation-models/lag-llama :: "
                            "data/dataset_list.py ALL_DATASETS",
               "timesfm_2": "membership in Salesforce/GiftEvalPretrain "
                            "(named on the timesfm-2.5-200m model card, Data section)",
               "moirai_base": "membership in Salesforce/lotsa_data (LOTSA, Woo et al. 2024)",
           },
           "scope_note": "The GIFT-Eval leaderboard's testdata_leakage flag covers "
                         "the GIFT-EVAL TEST SPLIT ONLY and is not used for these "
                         "Monash cells; a model flagged as non-leaking there may "
                         "still have trained on Monash data, and one here does.",
           "models": {}}

    MAP = {"PRESENT": "contaminated", "ABSENT": "clean"}
    for model, corpus in (("timesfm_2", "GiftEvalPretrain"),
                          ("moirai_base", "lotsa_data")):
        cells = {}
        for ds, v in mem[corpus]["datasets"].items():
            st = MAP.get(v["status"], a.candidates_as)
            cells[ds] = {"status": st, "evidence": v["evidence"]}
        out["models"][model] = cells

    for model in ("chronos_bolt_small", "chronos_bolt_base", "lag_llama"):
        out["models"][model] = draft["models"][model]

    p = ROOT / "configs" / "contamination_verified.yaml"
    p.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True))
    n_unknown = sum(1 for m in out["models"].values()
                    for c in m.values() if c["status"] == "unknown")
    print(f"[verified] {p}")
    print(f"  provisional: {out['provisional']} | candidates as: {a.candidates_as}")
    print(f"  cells still 'unknown': {n_unknown}")
    if out["provisional"]:
        print("  the gate will still BLOCK final — rerun with --approve once you "
              "have read the cells")


if __name__ == "__main__":
    main()
