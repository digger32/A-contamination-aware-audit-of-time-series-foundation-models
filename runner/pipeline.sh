#!/usr/bin/env bash
# Pipeline — one command per stage, run from PROJECT ROOT inside tmux:
#   tmux new -s bench   ->   bash runner/pipeline.sh <stage>   ->   Ctrl-b d
# Stages:
#   check     verify every dataset/checkpoint is reachable (NO downloads)
#   external  cache GIFT-Eval leakage flags + official reference MASE
#   data      download + prepare both suites
#   selftest  gate unit-test on synthetic clean/dirty runs (Stage 0 requirement)
#   smoke     1 dataset x 2 models, full chain run->aggregate->stats->gate
#   micro     MICRO timing slice: 1 unit per model on the heaviest dataset;
#             read wall_s from the manifest and extrapolate BEFORE full
#   pilot     2 seeds x 2 datasets x all models
#   full      complete grid, resume ENABLED
#   final     fresh outdir, --no-resume, then aggregate+stats+gate automatically
#   stats     aggregate + stats on an existing outdir (OUT=runs/... override)
#   gate      gate only (OUT=runs/... override)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=${PY:-python3}
STAGE="${1:?usage: pipeline.sh <check|data|selftest|smoke|micro|pilot|full|final|stats|gate>}"

chain() {  # aggregate -> stats -> external validation -> gate on $1
  $PY scripts/aggregate.py "$1"
  $PY scripts/stats.py "$1"
  [ -f external/gifteval_reference_mase.json ] && \
    $PY scripts/validate_external.py "$1" || \
    echo "[chain] no external reference cached; run: python scripts/fetch_external_reference.py"
  $PY scripts/review_gate.py "$1"
}

case "$STAGE" in
  check)
    $PY scripts/download_data.py --check ;;
  external)
    $PY scripts/fetch_external_reference.py ;;
  data)
    $PY scripts/download_data.py
    $PY scripts/prepare_data.py ;;
  selftest)
    $PY scripts/gate_selftest.py ;;
  smoke)
    $PY scripts/gate_selftest.py
    OUT=runs/smoke; rm -rf "$OUT"
    $PY runner/bench_runner.py --outdir "$OUT" --datasets nn5_weekly \
        --models snaive,autoets,xgb_lags --max-seeds 1 --timeout-s 600
    $PY scripts/aggregate.py "$OUT"
    $PY scripts/stats.py "$OUT" || true   # 1 dataset: omnibus legitimately degenerate
    $PY scripts/review_gate.py "$OUT" || true  # smoke is resume-mode: A1 FAIL is expected
    echo "[smoke] done — inspect $OUT (gate FAIL on A1/B1 here is EXPECTED)" ;;
  micro)
    OUT=runs/micro; rm -rf "$OUT"
    $PY runner/bench_runner.py --outdir "$OUT" --datasets m4_hourly \
        --max-seeds 1 --timeout-s 5400
    $PY - "$OUT" <<'EOF'
import json, sys, yaml
from pathlib import Path
out = Path(sys.argv[1])
models = yaml.safe_load(Path("configs/models.yaml").read_text())["models"]
mon = yaml.safe_load(Path("configs/datasets_monash.yaml").read_text())["datasets"]
gif = yaml.safe_load(Path("configs/datasets_gifteval.yaml").read_text())["datasets"]
n_ds = len(mon) + len(gif)
rows = [json.loads(l) for l in (out/"manifest.jsonl").read_text().splitlines()]
total = 0
print(f"{'model':22s} {'wall_s/unit':>12s} {'units':>6s} {'est_total_h':>12s}")
for r in rows:
    m = r["model"]; n_units = n_ds * models[m].get("n_seeds", 1)
    est = r["wall_s"] * n_units / 3600
    total += est
    print(f"{m:22s} {r['wall_s']:12.1f} {n_units:6d} {est:12.2f}")
print(f"{'TOTAL (upper bound: heaviest dataset everywhere)':50s} {total:10.1f} h")
EOF
    echo "[micro] extrapolation above is from MEASURED wall_s — trust it, not arithmetic" ;;
  pilot)
    OUT=runs/pilot
    $PY runner/bench_runner.py --outdir "$OUT" \
        --datasets nn5_weekly,hospital --max-seeds 2 --timeout-s 5400
    chain "$OUT" || true ;;
  full)
    OUT=runs/full
    $PY runner/bench_runner.py --outdir "$OUT" --timeout-s 5400
    chain "$OUT" || true ;;
  final)
    OUT=runs/final_$(date -u +%Y%m%d)
    if [ -d "$OUT" ]; then echo "[final] $OUT exists — refusing (fresh dir required)"; exit 1; fi
    $PY runner/bench_runner.py --outdir "$OUT" --no-resume --timeout-s 5400
    chain "$OUT"
    echo "[final] GATE PASSED on $OUT — numbers are freezable" ;;
  stats)
    chain "${OUT:?set OUT=runs/...}" ;;
  gate)
    $PY scripts/review_gate.py "${OUT:?set OUT=runs/...}" ;;
  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
