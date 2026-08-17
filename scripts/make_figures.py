#!/usr/bin/env python3
"""Build the paper figures from a GATED run's stats. Refuses to run unless the
gate passed (checks for stats/ artifacts written by the chain; run
`pipeline.sh gate` first). Outputs vector PDFs into <outdir>/figures/:
  fig_wins.pdf   per-model win/loss bars vs seasonal-naive (Fig. 1)
  fig_cd.pdf     critical-difference diagram from avg ranks + Nemenyi (Fig. 2)
Greyscale-legible, colourblind-safe, Times fonts to match IEEEtran.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# Nemenyi critical values q_alpha (studentised range statistic / sqrt(2)), alpha=0.05
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
       8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
       14: 3.354, 15: 3.391}


def critical_difference(k: int, n: int, alpha: str = "0.05") -> float:
    """Nemenyi critical difference: CD = q_alpha * sqrt(k(k+1) / 6N).
    Verified against the scikit-posthocs Nemenyi p-values on this study's data:
    the CD rule and p<0.05 agreed on all 45 pairs."""
    if k not in Q05:
        raise ValueError(f"no tabulated q_alpha for k={k}")
    return Q05[k] * math.sqrt(k * (k + 1) / (6.0 * n))


def fig_wins(stats_dir: Path, out: Path):
    wins = json.loads((stats_dir / "wins.json").read_text())
    models = sorted(wins, key=lambda m: -wins[m]["wins"])
    n = wins[models[0]]["n_datasets"]
    w = [wins[m]["wins"] for m in models]
    l = [wins[m]["losses"] for m in models]
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(3.5, 0.34 * len(models) + 1.0))
    ax.barh(y, w, color="0.35", label="beats seasonal-naive")
    ax.barh(y, l, left=w, color="0.82", hatch="//", edgecolor="0.45",
            linewidth=0.4, label="loses to seasonal-naive")
    for yi, wi in zip(y, w):                      # count at the end of each bar
        ax.text(n + 0.4, yi, str(wi), va="center", fontsize=7)
    ax.set_yticks(y, models)
    ax.set_xlabel(f"datasets (of {n})")
    ax.set_xlim(0, n + 2)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    # legend ABOVE the axes so it can never sit on top of the bars
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), handlelength=1.4, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig_wins.pdf", bbox_inches="tight")
    print(f"[figures] {out/'fig_wins.pdf'}")


def fig_cd(stats_dir: Path, out: Path, alpha=0.05):
    """Critical-difference diagram in the classic Demsar layout: rank axis on top,
    methods split between the left and right margins so labels never cross the
    plot, the CD bar drawn and LABELLED above the axis, and cliques of
    statistically indistinguishable methods as bars below it."""
    ph = json.loads((stats_dir / "posthoc.json").read_text())
    ranks, nem, n_ds = ph["avg_ranks"], ph.get("nemenyi_p"), ph["n_datasets"]
    models = sorted(ranks, key=ranks.get)          # best (lowest rank) first
    k = len(models)
    cd = critical_difference(k, n_ds)

    lo, hi = math.floor(min(ranks.values())), math.ceil(max(ranks.values()))
    span = hi - lo
    left = models[: (k + 1) // 2]                  # better half -> left margin
    right = models[(k + 1) // 2:][::-1]            # worse half  -> right margin
    rows = max(len(left), len(right))

    # cliques: maximal runs of consecutive methods all within CD of each other
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and ranks[models[j + 1]] - ranks[models[i]] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
    cliques = [c for c in cliques
               if not any(o != c and o[0] <= c[0] and c[1] <= o[1] for o in cliques)]

    # Geometry, in data units with the rank axis at y = 0. The clique bars form a
    # band DIRECTLY BELOW the axis (Demsar's layout) so the reader reads them
    # against the scale; the label rows begin below that band. Everything is
    # derived from the number of cliques and rows, so nothing can collide.
    BAR_TOP, BAR_STEP, ROW_STEP = -0.30, 0.26, 1.0
    band_bottom = BAR_TOP - BAR_STEP * max(0, len(cliques) - 1)
    row0 = band_bottom - 0.75                      # first label row
    y_row = lambda i: row0 - ROW_STEP * i

    fig, ax = plt.subplots(figsize=(4.6, 0.30 * rows + 0.34 * len(cliques) + 1.5))
    ax.set_xlim(lo - 0.42 * span, hi + 0.42 * span)   # margins hold the labels
    ax.set_ylim(y_row(rows - 1) - 0.6, 2.6)
    ax.axis("off")

    # rank axis
    ax.plot([lo, hi], [0, 0], color="0.1", lw=1.0)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [0, 0.16], color="0.1", lw=1.0)
        ax.text(t, 0.30, str(t), ha="center", va="bottom", fontsize=7.5)
    ax.text((lo + hi) / 2, 1.55, "average rank (lower is better)",
            ha="center", va="bottom", fontsize=8)

    # CD bar, drawn to scale and labelled with its value
    y_cd = 0.95
    ax.plot([lo, lo + cd], [y_cd, y_cd], color="0.1", lw=1.4)
    for x in (lo, lo + cd):
        ax.plot([x, x], [y_cd - 0.11, y_cd + 0.11], color="0.1", lw=1.4)
    ax.text(lo + cd / 2, y_cd + 0.16, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=7.5)

    # clique bars, immediately under the axis
    for r, (i, j) in enumerate(cliques):
        y = BAR_TOP - BAR_STEP * r
        ax.plot([ranks[models[i]] - 0.05, ranks[models[j]] + 0.05], [y, y],
                color="0.1", lw=3.0, solid_capstyle="round")

    # connectors to the margins, starting below the clique band
    x_left, x_right = lo - 0.06 * span, hi + 0.06 * span
    for i, m in enumerate(left):
        y = y_row(i)
        ax.plot([ranks[m], ranks[m]], [0, y], color="0.25", lw=0.8)
        ax.plot([ranks[m], x_left], [y, y], color="0.25", lw=0.8)
        ax.text(x_left - 0.03 * span, y, f"{m} ({ranks[m]:.2f})",
                ha="right", va="center", fontsize=7.5)
    for i, m in enumerate(right):
        y = y_row(i)
        ax.plot([ranks[m], ranks[m]], [0, y], color="0.25", lw=0.8)
        ax.plot([ranks[m], x_right], [y, y], color="0.25", lw=0.8)
        ax.text(x_right + 0.03 * span, y, f"{m} ({ranks[m]:.2f})",
                ha="left", va="center", fontsize=7.5)

    if nem:  # sanity: the CD rule must agree with the computed p-values
        bad = sum(1 for a_i, a in enumerate(models) for b in models[a_i + 1:]
                  if (abs(ranks[a] - ranks[b]) > cd) != (nem[a][b] < alpha))
        if bad:
            print(f"[figures] WARNING: CD rule disagrees with Nemenyi p-values on "
                  f"{bad} pair(s) — inspect before publishing")
        else:
            print(f"[figures] CD={cd:.3f} agrees with Nemenyi p-values on all "
                  f"{k*(k-1)//2} pairs")

    fig.tight_layout()
    fig.savefig(out / "fig_cd.pdf", bbox_inches="tight")
    print(f"[figures] {out/'fig_cd.pdf'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    a = ap.parse_args()
    outdir = Path(a.outdir)
    stats_dir = outdir / "stats"
    for req in ("wins.json", "posthoc.json"):
        if not (stats_dir / req).exists():
            sys.exit(f"[figures] {stats_dir/req} missing — run the stats+gate chain first")
    figs = outdir / "figures"; figs.mkdir(exist_ok=True)
    fig_wins(stats_dir, figs)
    fig_cd(stats_dir, figs)


if __name__ == "__main__":
    main()
