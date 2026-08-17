#!/usr/bin/env python3
"""Prepare raw data into data/prepared/<suite>/<dataset>.npz + .meta.json.

Skip-if-exists by default, --force to rebuild (frozen-artifact rule: a
post-freeze re-run must never silently replace what a gated run's eval_hash
referenced). Series capping is SEEDED (fixed seed 20260721) so the subset is
identical across every rebuild and every model.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tsfmbench import data as D  # noqa: E402

SUBSET_SEED = 20260721


def prep_monash(force: bool):
    cfg = D.load_cfg("datasets_monash.yaml")
    raw = ROOT / "data" / "raw" / "monash"
    for name, c in cfg["datasets"].items():
        out = D.prepared_path("monash", name)
        if out.exists() and not force:
            print(f"[skip] monash/{name}")
            continue
        tsf = raw / c["file"]
        if not tsf.exists():
            print(f"[MISS] monash/{name}: {tsf} not found — run download_data.py first")
            continue
        series = D.read_tsf(tsf)
        rng = np.random.default_rng(SUBSET_SEED)
        if len(series) > c["max_series"]:
            idx = sorted(rng.choice(len(series), c["max_series"], replace=False))
            series = [series[i] for i in idx]
        names = [s[0] for s in series]
        arrays = [s[1] for s in series]
        D.save_prepared("monash", name, names, arrays,
                        {"horizon": c["horizon"], "seasonality": c["seasonality"],
                         "source": f"zenodo:{c['zenodo']}", "subset_seed": SUBSET_SEED})
        print(f"[ok] monash/{name}: {len(arrays)} series")


def prep_gifteval(force: bool):
    """Materialise GIFT-Eval configs into the npz layout. API verified against
    the installed salesforce-gift-eval source (2026-07-21): Dataset(name, term,
    to_univariate) where `name` is the PATH inside the GIFT_EVAL storage dir
    (freq is a subdirectory, e.g. 'jena_weather/H'); horizon = prediction_length
    (term-dependent); seasonality derived from ds.freq via gluonts;
    test_data yields rolling (input, label) instance pairs, label length ==
    prediction_length."""
    cfg = D.load_cfg("datasets_gifteval.yaml")
    import os
    os.environ.setdefault("GIFT_EVAL", str(ROOT / "data" / "raw" / "gifteval"))
    storage = Path(os.environ["GIFT_EVAL"])
    try:
        from gift_eval.data import Dataset as GiftDataset
        from gluonts.time_feature import get_seasonality
    except ImportError:
        print("[MISS] salesforce-gift-eval not installed — see requirements.txt; "
              "GIFT-Eval preparation skipped")
        return
    rng = np.random.default_rng(SUBSET_SEED)
    for token in cfg["datasets"]:
        out_name = token.replace("/", "+")
        out = D.prepared_path("gifteval", out_name)
        if out.exists() and not force:
            print(f"[skip] gifteval/{out_name}")
            continue
        *name_parts, term = token.split("/")
        candidates = ["/".join(name_parts)] + name_parts[:1]  # 'ds/freq', then 'ds'
        name = next((c for c in candidates if (storage / c).is_dir()), None)
        if name is None:
            avail = sorted(p.name for p in storage.iterdir()) if storage.exists() else []
            print(f"[MISS] gifteval/{token}: none of {candidates} in {storage} "
                  f"(available: {avail})")
            continue
        gd = GiftDataset(name=name, term=term, to_univariate=True)
        horizon = int(gd.prediction_length)
        seasonality = int(get_seasonality(gd.freq))
        # Direct reimplementation of gift-eval's windowing (split(offset=-h*w) +
        # generate_instances(h, windows=w, distance=h)), reading rows in python
        # format. Reason: datasets>=5 numpy formatting of ragged multi-series
        # rows crashes gluonts' splitter (IndexError, shape ()); verified
        # 2026-07-21 that this path is instance-identical to gd.test_data
        # wherever gd.test_data works, and survives where it crashes.
        h, w = horizon, int(gd.windows)
        raw = gd.hf_dataset.with_format("python")
        full, n_short = [], 0
        for row in raw:
            t = np.asarray(row["target"], dtype=np.float64)
            chans = t[None, :] if t.ndim == 1 else t
            rid = str(row.get("item_id", len(full)))
            for ci, y in enumerate(chans):
                base = len(y) - h * w
                if base < 1:
                    n_short += 1
                    continue
                for k in range(1, w + 1):
                    full.append((f"{rid}|c{ci}|w{k}", y[: base + k * h]))
        if n_short:
            print(f"  [note] {token}: {n_short} channel(s) shorter than h*w skipped")
        if len(full) > cfg["max_series_per_config"]:
            idx = sorted(rng.choice(len(full), cfg["max_series_per_config"], replace=False))
            full = [full[i] for i in idx]
        D.save_prepared("gifteval", out_name, [f[0] for f in full],
                        [f[1] for f in full],
                        {"horizon": horizon, "seasonality": seasonality,
                         "freq": gd.freq, "windows": int(gd.windows),
                         "source": f"hf:{cfg['hf_repo']}:{name}:{term}",
                         "subset_seed": SUBSET_SEED})
        print(f"[ok] gifteval/{out_name}: {len(full)} instances, h={horizon}, m={seasonality}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--suite", choices=["monash", "gifteval", "all"], default="all")
    a = ap.parse_args()
    if a.suite in ("monash", "all"):
        prep_monash(a.force)
    if a.suite in ("gifteval", "all"):
        prep_gifteval(a.force)
