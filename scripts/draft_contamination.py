#!/usr/bin/env python3
"""Draft the contamination matrix from PRIMARY sources, with evidence per cell.

Source (a), per model, fetched live from each project's own repository — not
from anyone's recollection:
  chronos    scripts/evaluation/configs/{in-domain,zero-shot}.yaml — the authors'
             own split between datasets inside the training corpus and datasets
             deliberately held out. Authoritative and machine-readable.
  lag_llama  data/dataset_list.py ALL_DATASETS — the pretraining dataset list.
  moirai     LOTSA corpus; no single machine-readable manifest in the repo, so
             cells are emitted as NEEDS-SOURCE for manual reading of the paper's
             corpus appendix.
  timesfm    corpus not enumerated in a machine-readable form; emitted as
             NEEDS-SOURCE. If it stays unresolvable that is itself a reportable
             finding about disclosure practice.
Source (b) — the GIFT-Eval leaderboard's self-disclosed testdata_leakage flag —
is applied to the GIFT-Eval half by fetch_external_reference.py; its scope is
the GIFT-Eval test split only, so it never overrides a Monash cell here.

Output: configs/contamination_draft.yaml, every cell carrying `status` plus
`evidence` (the file and entry it came from). NOTHING is auto-promoted to the
verified matrix: copy the file to contamination_verified.yaml, resolve every
NEEDS-SOURCE and every AMBIGUOUS name mapping by reading the cited source, set
`provisional: false`, and only then will the gate let FINAL through.
"""
import json
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CHRONOS = ("https://raw.githubusercontent.com/amazon-science/chronos-forecasting/"
           "main/scripts/evaluation/configs/{}.yaml")
LAGLLAMA = ("https://raw.githubusercontent.com/time-series-foundation-models/"
            "lag-llama/main/data/dataset_list.py")

# Our Monash dataset -> the name used in each upstream source. Entries marked
# AMBIGUOUS need a human decision (the upstream name does not uniquely identify
# our config) and are never resolved automatically.
CHRONOS_NAMES = {
    "m3_monthly": "monash_m3_monthly", "m3_quarterly": "monash_m3_quarterly",
    "m4_hourly": "m4_hourly", "m4_weekly": "m4_weekly",
    "tourism_monthly": "monash_tourism_monthly",
    "tourism_quarterly": "monash_tourism_quarterly",
    "nn5_daily": "nn5", "nn5_weekly": "monash_nn5_weekly",
    "hospital": "monash_hospital", "fred_md": "monash_fred_md",
    "car_parts": "monash_car_parts", "covid_deaths": "monash_covid_deaths",
    "electricity_weekly": "monash_electricity_weekly",
    "traffic_weekly": "AMBIGUOUS:monash_traffic",  # hourly or weekly? decide by hand
    "us_births": "NOT-LISTED",
}
LAGLLAMA_NAMES = {  # ALL_DATASETS entries matching our Monash configs
    "traffic_weekly": "AMBIGUOUS:traffic", "electricity_weekly": "AMBIGUOUS:electricity_hourly",
}


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def chronos_split():
    out = {}
    for cfg, status in (("in-domain", "contaminated"), ("zero-shot", "clean")):
        text = get(CHRONOS.format(cfg))
        names = [e.get("name") for e in yaml.safe_load(text) if isinstance(e, dict)]
        for n in names:
            out[n] = (status, f"chronos-forecasting/scripts/evaluation/configs/{cfg}.yaml")
    return out


def main():
    mon = yaml.safe_load((ROOT / "configs" / "datasets_monash.yaml").read_text())["datasets"]
    draft = {"provisional": True,
             "note": "Drafted from primary sources; resolve every NEEDS-SOURCE and "
                     "AMBIGUOUS cell by reading the cited file, then save as "
                     "contamination_verified.yaml with provisional: false.",
             "models": {}}

    print("== chronos (in-domain vs zero-shot configs) ==")
    csplit = chronos_split()
    chronos_cells = {}
    for ds in mon:
        up = CHRONOS_NAMES.get(ds, "NOT-LISTED")
        if up.startswith("AMBIGUOUS"):
            chronos_cells[ds] = {"status": "unknown",
                                 "evidence": f"AMBIGUOUS upstream name '{up.split(':')[1]}' "
                                             f"— decide by hand against the Monash archive"}
        elif up == "NOT-LISTED":
            chronos_cells[ds] = {"status": "unknown",
                                 "evidence": "absent from both chronos config files "
                                             "— NEEDS-SOURCE (check the paper's corpus table)"}
        elif up in csplit:
            status, src = csplit[up]
            chronos_cells[ds] = {"status": status, "evidence": f"{src} :: {up}"}
        else:
            chronos_cells[ds] = {"status": "unknown",
                                 "evidence": f"mapped name '{up}' not found in either config"}
        print(f"  {ds:20s} {chronos_cells[ds]['status']}")
    draft["models"]["chronos_bolt_small"] = chronos_cells
    draft["models"]["chronos_bolt_base"] = chronos_cells

    print("\n== lag_llama (ALL_DATASETS) ==")
    text = get(LAGLLAMA)
    listed = text.split("ALL_DATASETS")[1] if "ALL_DATASETS" in text else ""
    ll = {}
    for ds in mon:
        up = LAGLLAMA_NAMES.get(ds)
        if up and up.startswith("AMBIGUOUS"):
            ll[ds] = {"status": "unknown",
                      "evidence": f"AMBIGUOUS upstream '{up.split(':')[1]}' in "
                                  f"lag-llama/data/dataset_list.py — decide by hand"}
        elif ds in listed:
            ll[ds] = {"status": "contaminated",
                      "evidence": f"lag-llama/data/dataset_list.py :: ALL_DATASETS"}
        else:
            ll[ds] = {"status": "clean",
                      "evidence": "absent from ALL_DATASETS in "
                                  "lag-llama/data/dataset_list.py"}
        print(f"  {ds:20s} {ll[ds]['status']}")
    draft["models"]["lag_llama"] = ll

    for m in ("moirai_base", "timesfm_2"):
        draft["models"][m] = {ds: {"status": "unknown",
                                   "evidence": "NEEDS-SOURCE: no machine-readable corpus "
                                               "manifest; read the model's corpus appendix"}
                              for ds in mon}
    print(f"\n== moirai_base, timesfm_2: all cells NEEDS-SOURCE ==")

    out = ROOT / "configs" / "contamination_draft.yaml"
    out.write_text(yaml.safe_dump(draft, sort_keys=False, allow_unicode=True))
    print(f"\n[draft] {out}\nResolve NEEDS-SOURCE / AMBIGUOUS by hand, then save as "
          f"configs/contamination_verified.yaml with provisional: false.")


if __name__ == "__main__":
    main()
