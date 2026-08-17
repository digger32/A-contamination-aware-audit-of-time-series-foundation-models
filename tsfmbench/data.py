"""Dataset loading and evaluation-window construction.

One protocol for both suites: for each series, the TARGET is the final `horizon`
points, the CONTEXT is everything before (models may truncate context to their
own cap; the target is never touched). The eval_hash fingerprints the exact
evaluation windows so the gate can assert every model saw identical targets.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_cfg(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


# ----------------------------------------------------------------------------- #
# .tsf parser (Monash format). Minimal reimplementation of the reference loader
# from rakshitha123/TSForecasting (CC-BY-4.0): header lines '@attribute ...',
# '@data', then one line per series: name:...:v1,v2,...  Missing values are '?'.
# ----------------------------------------------------------------------------- #
def read_tsf(path: Path):
    series = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        in_data = False
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower().startswith("@data"):
                    in_data = True
                continue
            parts = line.split(":")
            vals = parts[-1].split(",")
            arr = np.array([np.nan if v == "?" else float(v) for v in vals],
                           dtype=np.float64)
            series.append((parts[0], arr))
    return series


def prepared_path(suite: str, dataset: str) -> Path:
    return DATA / "prepared" / suite / f"{dataset}.npz"


def save_prepared(suite, dataset, names, arrays, meta):
    p = prepared_path(suite, dataset)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **{f"s{i}": a for i, a in enumerate(arrays)})
    tmp.rename(p)  # atomic
    meta = dict(meta, names=list(names), n_series=len(arrays))
    p.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def load_prepared(suite: str, dataset: str):
    """Returns (list_of_1d_arrays, meta_dict). Skip-if-missing is the caller's
    problem: a missing prepared file is a hard error, never silently re-derived
    (frozen-artifact rule)."""
    p = prepared_path(suite, dataset)
    meta = json.loads(p.with_suffix(".meta.json").read_text())
    with np.load(p) as z:
        arrays = [z[f"s{i}"] for i in range(meta["n_series"])]
    return arrays, meta


def eval_windows(arrays, horizon: int, min_context: int = 8):
    """Split each series into (context, target). Series shorter than
    min_context + horizon are dropped (counted in meta upstream)."""
    ctx, tgt, kept = [], [], []
    for i, a in enumerate(arrays):
        a = a[~np.isnan(a)]
        if len(a) < min_context + horizon:
            continue
        ctx.append(a[:-horizon])
        tgt.append(a[-horizon:])
        kept.append(i)
    return ctx, tgt, kept


def eval_hash(dataset: str, horizon: int, ctx, tgt) -> str:
    """Fingerprint of the exact evaluation protocol: dataset, horizon, per-series
    context lengths and target values. Any drift (different subset, different
    cutoff, different horizon) changes the hash; the gate asserts one hash per
    (suite, dataset) across all models."""
    h = hashlib.sha256()
    h.update(f"{dataset}|{horizon}|{len(ctx)}".encode())
    for c, t in zip(ctx, tgt):
        h.update(np.int64(len(c)).tobytes())
        h.update(np.round(t, 6).tobytes())
    return h.hexdigest()[:16]
