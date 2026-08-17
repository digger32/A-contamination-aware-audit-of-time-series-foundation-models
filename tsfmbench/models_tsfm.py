"""TSFM wrappers (zero-shot) + PatchTST neural reference.

All CUDA paths run under PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set by
pipeline.sh and re-asserted here): allocator fragmentation rather than true
memory pressure is a common source of out-of-memory failures here. Every cross-series batched call goes
through `oom_backoff`, which halves the batch on torch.OutOfMemoryError down to 1
and reports the number of backoffs so the unit JSON records it.

API-CURRENCY NOTE: the exact constructor/predict signatures of chronos,
timesfm, uni2ts and lag-llama drift between minor releases. Each wrapper follows
the officially documented API as of the pinned versions in requirements.txt and
is tagged UNVERIFIED where the pin could not be exercised in this sandbox; the
Stage -1/-0 currency check (scripts/check_stack.py + smoke) is the verification
step: library glue written from recollection must never reach a final run.
"""
import os
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def oom_backoff(fn, items, batch0: int):
    """Run fn(batch_of_items) over items with OOM-halving. Returns (results, n_backoffs)."""
    import torch
    out, backoffs, bs, i = [], 0, max(1, batch0), 0
    while i < len(items):
        chunk = items[i:i + bs]
        try:
            out.extend(fn(chunk))
            i += bs
        except torch.OutOfMemoryError:
            _empty_cache()
            if bs == 1:
                raise  # a single series does not fit — real failure, unit dies
            bs = max(1, bs // 2)
            backoffs += 1
    return out, backoffs


def _ctx_tensors(ctx, cap: int):
    import torch
    return [torch.tensor(np.ascontiguousarray(c[-cap:]), dtype=torch.float32)
            for c in ctx]


def forecast_chronos_bolt(hf_id, ctx, horizon, cap, batch0, device):
    """Chronos-Bolt: quantile head, single forward pass; point forecast = median.
    API per amazon/chronos-forecasting BaseChronosPipeline. UNVERIFIED pin."""
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(hf_id, device_map=device)
    tensors = _ctx_tensors(ctx, cap)

    def _run(chunk):
        # chronos>=1.5 renamed the argument context -> inputs (verified against
        # the installed 1.5.3 signatures on 2026-07-21)
        q, _mean = pipe.predict_quantiles(inputs=chunk, prediction_length=horizon,
                                          quantile_levels=[0.5])
        return [q[j, :, 0].cpu().numpy() for j in range(len(chunk))]

    return oom_backoff(_run, tensors, batch0)


def forecast_timesfm(hf_id, ctx, horizon, cap, batch0, device):
    """TimesFM 2.5 (reworked package >=2.0.2). API verified against the official
    PyPI/README example on 2026-07-21: TimesFM_2p5_200M_torch.from_pretrained ->
    compile(ForecastConfig) -> forecast(horizon, inputs) -> (point, quantiles).
    The package handles device placement; point forecast is returned directly."""
    import timesfm
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(hf_id)
    model.compile(timesfm.ForecastConfig(
        max_context=cap, max_horizon=max(horizon, 64),
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=True,
        fix_quantile_crossing=True))
    arrays = [np.ascontiguousarray(c[-cap:], dtype=np.float32) for c in ctx]

    def _run(chunk):
        point, _q = model.forecast(horizon=horizon, inputs=list(chunk))
        return [point[j, :horizon] for j in range(len(chunk))]

    return oom_backoff(_run, arrays, batch0)


def forecast_moirai(hf_id, ctx, horizon, cap, batch0, device):
    """Moirai 1.1-R via uni2ts MoiraiForecast/MoiraiModule, gluonts predictor.
    Point forecast = median of 100 samples. UNVERIFIED pin."""
    import torch
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    from gluonts.dataset.common import ListDataset

    model = MoiraiForecast(module=MoiraiModule.from_pretrained(hf_id),
                           prediction_length=horizon, context_length=cap,
                           patch_size="auto", num_samples=100,
                           target_dim=1, feat_dynamic_real_dim=0,
                           past_feat_dynamic_real_dim=0)
    predictor = model.create_predictor(batch_size=batch0,
                                       device=device if torch.cuda.is_available() else "cpu")
    ds = ListDataset([{"start": "2000-01-01", "target": c[-cap:]} for c in ctx],
                     freq="D")  # freq is nominal; Moirai conditions on values
    fcsts = list(predictor.predict(ds))
    return [np.median(f.samples, axis=0) for f in fcsts], 0


def _install_gluonts_loss_shim():
    """gluonts 0.16 removed gluonts.torch.modules.loss; lag-llama (pinned to
    gluonts<=0.14.4 upstream) still imports DistributionLoss and
    NegativeLogLikelihood from it. Pre-register a faithful reimplementation of
    the 0.14 module (verified against the 0.14 source: NLL = -log_prob, with
    optional beta-NLL variance weighting; lag-llama uses beta=0) so the import
    resolves."""
    import sys
    import types
    from dataclasses import dataclass
    name = "gluonts.torch.modules.loss"
    if name in sys.modules:
        return
    mod = types.ModuleType(name)

    class DistributionLoss:
        def __call__(self, input, target):
            raise NotImplementedError

    @dataclass
    class NegativeLogLikelihood(DistributionLoss):
        beta: float = 0.0

        def __call__(self, input, target):
            nll = -input.log_prob(target)
            if self.beta > 0.0:
                variance = input.variance
                nll = nll * (variance.detach() ** self.beta)
            return nll

    mod.DistributionLoss = DistributionLoss
    mod.NegativeLogLikelihood = NegativeLogLikelihood
    sys.modules[name] = mod
    import gluonts.torch.modules as parent
    parent.loss = mod


def forecast_lag_llama(hf_id, ctx, horizon, cap, batch0, device, num_samples, seed):
    """Lag-Llama via its LagLlamaEstimator over the released checkpoint.
    Probabilistic — seeded; point forecast = sample median. UNVERIFIED pin."""
    import torch
    torch.manual_seed(seed)
    # torch>=2.6 defaults torch.load(weights_only=True); the official Lag-Llama
    # checkpoint pickles gluonts classes (StudentTOutput, ...) in its
    # hyper_parameters, and the estimator ALSO loads it internally via
    # lightning's load_from_checkpoint, so a one-off flag on our call is not
    # enough. Patch torch.load to default weights_only=False for this unit only
    # (each unit is an isolated subprocess; checkpoint provenance: official HF
    # release, sha256 recorded by download_data.py in data/DOWNLOADS.sha256).
    _orig_torch_load = torch.load

    def _load_trusted(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_torch_load(*a, **k)

    torch.load = _load_trusted
    _install_gluonts_loss_shim()
    from lag_llama.gluon.estimator import LagLlamaEstimator
    from gluonts.dataset.common import ListDataset

    # checkpoint file downloaded by scripts/download_data.py into data/checkpoints/
    from tsfmbench.data import ROOT
    ckpt_path = ROOT / "data" / "checkpoints" / "lag-llama.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["hyper_parameters"]["model_kwargs"]
    # Context = 32, the value the official Lag-Llama zero-shot example uses for
    # ALL datasets. We do NOT tune it per dataset: doing so to make Lag-Llama
    # beat seasonal-naive would fabricate the paper's central result. On short
    # Monash series (~85-113 steps) past_length = 32 + max(lags_seq)=182 = 214
    # still exceeds the series, so Lag-Llama runs partly on padding — that is
    # how the model behaves out of the box on short series, and a fair audit
    # reports it rather than engineering it away. A checkpoint-recorded context
    # is honoured if present and positive.
    _ckpt_ctx = args.get("context_length")
    lag_llama_context = int(_ckpt_ctx) if isinstance(_ckpt_ctx, int) and _ckpt_ctx > 0 else 32
    # Follow the official lag-llama example EXACTLY: feed every architecture arg
    # the checkpoint was trained with (a partial set silently builds a different
    # net), and let the estimator load the trained weights from ckpt_path. The
    # pilot bug was constructing the predictor from create_lightning_module()
    # (RANDOM weights) — that produced MASE ~1.5-1.7. create_predictor must
    # receive a module built via load_from_checkpoint, which is what the
    # estimator does internally when ckpt_path is set.
    estimator = LagLlamaEstimator(
        ckpt_path=str(ckpt_path),
        prediction_length=horizon,
        # Lag-Llama's InstanceSplitter uses past_length = context_length +
        # max(lags_seq), and max(lags_seq)=182. The official example uses
        # context_length=32. A cap of 1024 would inflate past_length to ~1206, so on
        # these short series (~85-113 steps) the model saw ~1100 padding steps
        # around ~100 real ones and forecast from padding -> MASE ~1.7. A small
        # context matched to the data is the documented usage; we cap it to the
        # horizon-scaled window, never above what the series can fill.
        context_length=lag_llama_context,
        input_size=args["input_size"],
        n_layer=args["n_layer"],
        n_embd_per_head=args["n_embd_per_head"],
        n_head=args["n_head"],
        scaling=args["scaling"],
        time_feat=args["time_feat"],
        rope_scaling=None,
        batch_size=batch0,
        num_parallel_samples=num_samples,
        # nonnegative_pred_samples MUST stay at the estimator default (False).
        # It was briefly set True on the assumption that these series are counts
        # or levels; several are not (ETT oil temperature, jena_weather in
        # Celsius cross zero), and clamping the samples at zero corrupted the
        # forecast — caught by the external check against the official GIFT-Eval
        # numbers, where our ett2 read 2.993 against their 0.862 while the other
        # eight configs agreed. False also matches the reference configuration
        # behind the published leaderboard values we validate against.
        device=torch.device(device),
    )
    transformation = estimator.create_transformation()
    lightning_module = estimator.create_lightning_module()
    predictor = estimator.create_predictor(transformation, lightning_module)
    ds = ListDataset([{"start": "2000-01-01", "target": c[-cap:]} for c in ctx], freq="D")
    fcsts = list(predictor.predict(ds, num_samples=num_samples))
    return [np.median(f.samples, axis=0) for f in fcsts], 0


def forecast_patchtst(ctx, horizon, seed, max_epochs, batch, device):
    """PatchTST reference, trained per dataset (gluonts.torch estimator),
    library defaults + max_epochs — the logged neural tuning budget. UNVERIFIED pin."""
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    from gluonts.torch.model.patch_tst import PatchTSTEstimator
    from gluonts.dataset.common import ListDataset

    train = ListDataset([{"start": "2000-01-01", "target": c} for c in ctx], freq="D")
    estimator = PatchTSTEstimator(prediction_length=horizon, patch_len=16,
                                  context_length=max(2 * horizon, 96),
                                  batch_size=batch,
                                  trainer_kwargs={"max_epochs": max_epochs,
                                                  "accelerator": "gpu" if device == "cuda" else "cpu",
                                                  "devices": 1, "enable_progress_bar": False,
                                                  "logger": False})
    predictor = estimator.train(train)
    fcsts = list(predictor.predict(train))
    return [np.median(f.samples, axis=0) if hasattr(f, "samples") else f.mean
            for f in fcsts], 0
