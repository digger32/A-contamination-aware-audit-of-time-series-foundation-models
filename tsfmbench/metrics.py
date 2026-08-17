"""Scaled error metrics. MASE per Hyndman & Koehler (2006): MAE on the target
scaled by the in-sample one-step seasonal-naive MAE computed on the CONTEXT only
(no target leakage into the scale). sMAPE in the standard symmetric form.
Series with a degenerate scale (constant context) are excluded and counted."""
import numpy as np


def mase(context: np.ndarray, target: np.ndarray, forecast: np.ndarray,
         seasonality: int) -> float:
    m = max(1, int(seasonality))
    if len(context) <= m:
        return np.nan
    scale = np.mean(np.abs(context[m:] - context[:-m]))
    if not np.isfinite(scale) or scale == 0:
        return np.nan
    return float(np.mean(np.abs(target - forecast)) / scale)


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(target) + np.abs(forecast)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(target - forecast) / denom))


def score_all(ctx, tgt, fcst, seasonality):
    """Per-series metric lists + aggregates. fcst: list of arrays, len == len(tgt)."""
    per_mase, per_smape, degenerate = [], [], 0
    for c, t, f in zip(ctx, tgt, fcst):
        f = np.asarray(f, dtype=np.float64)
        v = mase(c, t, f, seasonality)
        if np.isnan(v):
            degenerate += 1
            continue
        per_mase.append(v)
        per_smape.append(smape(t, f))
    return {
        "mase_mean": float(np.mean(per_mase)) if per_mase else None,
        "mase_median": float(np.median(per_mase)) if per_mase else None,
        "smape_mean": float(np.mean(per_smape)) if per_smape else None,
        "n_series_scored": len(per_mase),
        "n_series_degenerate_scale": degenerate,
        "per_series_mase": [round(x, 6) for x in per_mase],
    }
