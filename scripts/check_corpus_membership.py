#!/usr/bin/env python3
"""Check whether our evaluation datasets sit inside the pretraining corpora that
TimesFM 2.5 and Moirai were trained on. Run on the server (needs Hugging Face
access); the sandbox cannot reach huggingface.co.

Why this is the decisive step. The two corpora are Hugging Face datasets whose
member datasets are listed as directories, so membership is a fact we can read
rather than infer:

  Salesforce/GiftEvalPretrain — the corpus named on the timesfm-2.5-200m model
      card ("Data" section: GiftEvalPretrain, Wikimedia Pageviews, Google Trends,
      synthetic). It is curated to exclude GIFT-EVAL evaluation data, which is
      what the leaderboard's leakage=No flag reflects. It says NOTHING about
      Monash, and it is drawn from LOTSA, which does contain Monash datasets —
      so TimesFM's status on our Monash half is an open question that this
      listing answers.
  Salesforce/lotsa_data — the LOTSA corpus behind Moirai.

Output: configs/corpus_membership.json, listing for each of our datasets whether
a matching member exists, with the matched member name as evidence. Fuzzy name
variants are reported as CANDIDATE and still need a human decision; only exact
matches are reported as definite.
"""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CORPORA = {
    "GiftEvalPretrain": ("Salesforce/GiftEvalPretrain",
                         "timesfm-2.5 model card, Data section"),
    "lotsa_data": ("Salesforce/lotsa_data",
                   "LOTSA corpus behind Moirai (Woo et al. 2024)"),
}
# our dataset -> name fragments that would indicate the same underlying data
ALIASES = {
    "m3_monthly": ["m3_monthly"], "m3_quarterly": ["m3_quarterly"],
    "m4_hourly": ["m4_hourly"], "m4_weekly": ["m4_weekly"],
    "tourism_monthly": ["tourism_monthly"], "tourism_quarterly": ["tourism_quarterly"],
    "nn5_daily": ["nn5_daily"], "nn5_weekly": ["nn5_weekly"],
    "hospital": ["hospital"], "fred_md": ["fred_md"], "car_parts": ["car_parts", "carparts"],
    "covid_deaths": ["covid_deaths"], "electricity_weekly": ["electricity_weekly"],
    "traffic_weekly": ["traffic_weekly"], "us_births": ["us_births"],
}


def list_members(repo_id):
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo_id, repo_type="dataset")
    members = set()
    for f in files:
        head = f.split("/")[0]
        if head and not head.startswith(".") and "." not in head:
            members.add(head)
    return sorted(members)


def main():
    mon = yaml.safe_load((ROOT / "configs" / "datasets_monash.yaml").read_text())["datasets"]
    report = {}
    for label, (repo, provenance) in CORPORA.items():
        print(f"== {label} ({repo}) ==")
        try:
            members = list_members(repo)
        except Exception as e:
            print(f"  [error] {e}\n  (needs `hf auth login`; skipping)")
            report[label] = {"error": str(e)}
            continue
        print(f"  {len(members)} member datasets listed")
        entry = {"repo": repo, "provenance": provenance,
                 "n_members": len(members), "datasets": {}}
        for ds in mon:
            frags = ALIASES.get(ds, [ds])
            exact = [m for m in members if m in frags or m.replace("monash_", "") in frags]
            fuzzy = [m for m in members
                     if not exact and any(re.sub(r"[^a-z0-9]", "", f) in
                                          re.sub(r"[^a-z0-9]", "", m) for f in frags)]
            if exact:
                st, ev = "PRESENT", f"{repo} :: {exact[0]}"
            elif fuzzy:
                st, ev = "CANDIDATE", f"{repo} :: {fuzzy} — confirm by hand"
            else:
                st, ev = "ABSENT", f"no member of {repo} matches {frags}"
            entry["datasets"][ds] = {"status": st, "evidence": ev}
            print(f"    {ds:20s} {st:10s} {ev[:70]}")
        report[label] = entry

    out = ROOT / "configs" / "corpus_membership.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[membership] {out}")
    print("PRESENT  -> that model's cell for the dataset is 'contaminated'")
    print("ABSENT   -> 'clean' for the corpus checked (state the corpus in the paper)")
    print("CANDIDATE-> decide by hand before it may enter contamination_verified.yaml")


if __name__ == "__main__":
    main()
