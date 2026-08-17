#!/usr/bin/env python3
"""Job-based runner. Units = suite x dataset x model x seed, each in its OWN
subprocess (hang/OOM/segfault contained to the unit). Orchestration is the
canonical house pattern: RESUME (skip existing outputs), per-unit HARD TIMEOUT,
manifest.jsonl + run_meta.json for the gate, --no-resume for the final pass.

Launch (always inside tmux):
    tmux new -s bench
    bash runner/pipeline.sh full        # or: final / smoke / pilot / micro
    # detach: Ctrl-b d
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORKER_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # D4 OOM lesson
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",    # small-n thread rule
    "MKL_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
}


def load_axes():
    import yaml
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"]
    mon = yaml.safe_load((ROOT / "configs" / "datasets_monash.yaml").read_text())
    gif = yaml.safe_load((ROOT / "configs" / "datasets_gifteval.yaml").read_text())
    datasets = [("monash", d) for d in mon["datasets"]] + \
               [("gifteval", d.replace("/", "+")) for d in gif["datasets"]]
    return models, datasets


def unit_id(suite, dataset, model, seed):
    return f"{suite}__{dataset}__{model}__seed{seed}"


def unit_out_path(outdir, suite, dataset, model, seed):
    return Path(outdir) / f"{unit_id(suite, dataset, model, seed)}.json"


def append_manifest(outdir, record):
    with (Path(outdir) / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
def run_unit(suite, dataset, model, seed, out_path: Path):
    """One unit: load prepared data, forecast with `model`, score, atomic write."""
    import yaml
    import numpy as np
    from tsfmbench.data import load_prepared, eval_windows, eval_hash
    from tsfmbench.metrics import score_all

    mcfg = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())["models"][model]
    device = mcfg.get("device", "cpu")
    t0 = time.time()

    arrays, meta = load_prepared(suite, dataset)
    horizon, seasonality = meta["horizon"], meta["seasonality"]
    ctx, tgt, _ = eval_windows(arrays, horizon)
    ehash = eval_hash(dataset, horizon, ctx, tgt)

    backoffs = 0
    if mcfg["kind"] == "classical":
        from tsfmbench.models_classical import forecast_statsforecast
        fcst = forecast_statsforecast(model, ctx, horizon, seasonality)
    elif model == "xgb_lags":
        from tsfmbench.models_classical import forecast_xgb_lags, LAG_CAP
        n_lags = min(LAG_CAP, max(8, 3 * seasonality))  # cap: m=360 gave 1080 lags -> OOM
        fcst = forecast_xgb_lags(ctx, horizon, seasonality, seed, n_lags,
                                 n_jobs=mcfg.get("n_jobs", 1))
    elif model == "patchtst":
        from tsfmbench.models_tsfm import forecast_patchtst
        fcst, backoffs = forecast_patchtst(ctx, horizon, seed,
                                           mcfg["max_epochs"], mcfg["batch"], device)
    else:  # tsfm
        from tsfmbench import models_tsfm as M
        fn = {"chronos_bolt_small": M.forecast_chronos_bolt,
              "chronos_bolt_base": M.forecast_chronos_bolt,
              "timesfm_2": M.forecast_timesfm,
              "moirai_base": M.forecast_moirai}.get(model)
        if fn is not None:
            fcst, backoffs = fn(mcfg["hf"], ctx, horizon, mcfg["context_cap"],
                                mcfg["batch_series"], device)
        elif model == "lag_llama":
            fcst, backoffs = M.forecast_lag_llama(mcfg["hf"], ctx, horizon,
                                                  mcfg["context_cap"], mcfg["batch_series"],
                                                  device, mcfg["num_samples"], seed)
        else:
            raise ValueError(model)

    peak_vram = None
    try:
        import torch
        if torch.cuda.is_available() and device == "cuda":
            peak_vram = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:
        pass

    result = {
        "suite": suite, "dataset": dataset, "model": model, "seed": seed,
        "device": device, "eval_hash": ehash, "horizon": horizon,
        "python": sys.executable,
        "metrics": score_all(ctx, tgt, fcst, seasonality),
        "oom_backoffs": backoffs, "peak_vram_gb": peak_vram,
        "wall_s": round(time.time() - t0, 1),
    }
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result))
    tmp.rename(out_path)  # atomic


# --------------------------------------------------------------------------- #
def run_orchestrator(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    models, datasets = load_axes()

    want_models = args.models.split(",") if args.models else list(models)
    want_ds = set(args.datasets.split(",")) if args.datasets else None
    max_seeds = args.max_seeds

    units = []
    for suite, ds in datasets:
        if want_ds and ds not in want_ds:
            continue
        for m in want_models:
            n_seeds = min(models[m].get("n_seeds", 1), max_seeds)
            for s in range(n_seeds):
                units.append((suite, ds, m, s))

    run_started = datetime.now(timezone.utc).isoformat()
    (outdir / "run_meta.json").write_text(json.dumps({
        "run_started": run_started, "no_resume": args.no_resume,
        "models": want_models, "n_units": len(units),
        "timeout_s": args.timeout_s, "max_seeds": max_seeds,
        "datasets": sorted({d for _, d in datasets if not want_ds or d in want_ds}),
    }, indent=2))

    print(f"[runner] {len(units)} units | outdir={outdir} | "
          f"no_resume={args.no_resume} | timeout={args.timeout_s}s", flush=True)

    logdir = outdir / "logs"; logdir.mkdir(exist_ok=True)
    n_done = n_skip = n_fail = n_timeout = 0
    for suite, ds, m, s in units:
        out_path = unit_out_path(outdir, suite, ds, m, s)
        uid = unit_id(suite, ds, m, s)
        if out_path.exists() and not args.no_resume:
            n_skip += 1
            print(f"[skip] {uid}", flush=True)
            continue
        if out_path.exists() and args.no_resume:
            out_path.unlink()

        cmd = [models[m].get("python") or sys.executable,
               os.path.abspath(__file__), "--worker",
               "--suite", suite, "--dataset", ds, "--model", m,
               "--seed", str(s), "--outdir", str(outdir)]
        env = dict(os.environ, **WORKER_ENV)
        t0 = time.time(); status = "ok"
        with open(logdir / f"{uid}.log", "w") as lf:
            try:
                subprocess.run(cmd, timeout=args.timeout_s, check=True,
                               stdout=lf, stderr=subprocess.STDOUT, env=env)
            except subprocess.TimeoutExpired:
                status = "timeout"; n_timeout += 1
                print(f"[TIMEOUT] {uid} > {args.timeout_s}s — batch continues", flush=True)
            except subprocess.CalledProcessError as e:
                status = f"fail(rc={e.returncode})"; n_fail += 1
                print(f"[FAIL] {uid} rc={e.returncode} — batch continues | "
                      f"log: {logdir / (uid + '.log')}", flush=True)
            else:
                n_done += 1
                print(f"[ok] {uid} ({time.time() - t0:.1f}s)", flush=True)

        append_manifest(outdir, {
            "unit": uid, "suite": suite, "dataset": ds, "model": m, "seed": s,
            "status": status, "started": run_started,
            "finished": datetime.now(timezone.utc).isoformat(),
            "wall_s": round(time.time() - t0, 1), "no_resume": args.no_resume,
        })

    print(f"[runner] done | ok={n_done} skip={n_skip} fail={n_fail} "
          f"timeout={n_timeout}", flush=True)
    if n_fail or n_timeout:
        print("[runner] some units incomplete — inspect manifest before freezing.",
              flush=True)


def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--models", default="", help="comma list; empty = all from models.yaml")
    ap.add_argument("--datasets", default="", help="comma list; empty = all")
    ap.add_argument("--max-seeds", dest="max_seeds", type=int, default=99)
    ap.add_argument("--outdir", default="runs/dev")
    ap.add_argument("--timeout-s", dest="timeout_s", type=int, default=5400)
    ap.add_argument("--no-resume", dest="no_resume", action="store_true")
    ap.add_argument("--suite"); ap.add_argument("--dataset")
    ap.add_argument("--model"); ap.add_argument("--seed", type=int)
    return ap


if __name__ == "__main__":
    a = build_argparser().parse_args()
    if a.worker:
        run_unit(a.suite, a.dataset, a.model, a.seed,
                 unit_out_path(a.outdir, a.suite, a.dataset, a.model, a.seed))
    else:
        run_orchestrator(a)
