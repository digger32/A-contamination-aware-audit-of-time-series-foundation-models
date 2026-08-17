# Reproducibility

This document maps the repository to the reproducibility checklist that the
conference requires (Pineau et al., v2.0), so that each item can be located in
the code rather than taken on trust.

## Models and algorithms
- Every forecaster's exact configuration is in `configs/models.yaml`.
- The per-unit evaluation procedure is `run_unit()` in `runner/bench_runner.py`;
  it is the same object as the algorithm stated in the paper.
- The tuning budget of each method is fixed, disclosed and logged: automatic
  model selection for the statistical methods, one fixed gradient-boosting
  configuration, library defaults plus a stated epoch count for the neural
  reference, and zero-shot inference with no tuning for the foundation models.

## Datasets
- All datasets are public: the Monash Time Series Forecasting Archive (Zenodo,
  CC-BY-4.0; record identifiers in `configs/datasets_monash.yaml`) and GIFT-Eval
  (Hugging Face, `Salesforce/GiftEval`).
- Series subsets are capped with a fixed seed, and the identical subset is fed
  to every model. This is enforced, not assumed: each dataset's evaluation
  windows are fingerprinted and the gate fails if two models saw different ones.
- A SHA-256 digest of every downloaded file is recorded in
  `data/DOWNLOADS.sha256` during preparation.
- Pretraining-corpus overlap is recorded per (model, dataset) with a source for
  every entry; unverified codes cannot reach a figure, because the gate blocks
  on a matrix still marked provisional.

## Experiments
- A unit is one (suite, dataset, model, seed) combination. Units run in
  independent subprocesses with a per-unit timeout, so a hang or an
  out-of-memory failure is contained and logged rather than taking down the run.
- Resume is available for development. The reported numbers come from a final
  pass into a fresh directory with resume disabled, which the gate verifies.
- Randomness: seeds are set per unit for methods with run-to-run variation;
  deterministic methods declare a single seed.
- Environment: `requirements.txt`, with the core packages additionally pinned
  through a constraints file so that later installations cannot silently upgrade
  them.
- Compute: wall-clock time and peak GPU memory are recorded for every unit in
  the run manifest and in each unit's output. The study ran on a single NVIDIA
  A800 80 GB GPU with a 32-core host and 32 GB of RAM.

## Statistical analysis
- The unit of analysis is the dataset.
- Friedman omnibus test; Nemenyi post-hoc with a critical-difference diagram;
  Wilcoxon signed-rank against the baseline; bootstrap confidence intervals
  (10 000 resamples) on paired mean differences; win and loss counts with the
  losing datasets named.
- Comparisons are pairwise-complete, so one method's failure on a dataset does
  not remove that dataset from every other method's comparison. Coverage is
  reported explicitly in `stats/coverage.json` and asserted by the gate.
- All statistics are written under `runs/<name>/stats/`, and the figures are
  generated only from those files.

## Automated verification
`scripts/review_gate.py` runs at the end of the pipeline and exits non-zero,
blocking the results from being frozen, unless all of the following hold:
a clean final pass with resume disabled and no incomplete unit; every
comparative claim backed by runs on its declared independent datasets; the
statistical artefacts present at the paths the configuration names; one
evaluation-window fingerprint per dataset across all models; a verified
contamination matrix; one declared device per model; and no missing cell in the
grid. The gate is unit-tested against synthetic clean and dirty runs by
`scripts/gate_selftest.py`.

`scripts/validate_external.py` additionally compares this repository's results
for the shared configurations against the published per-dataset values of the
official GIFT-Eval leaderboard, which validates the evaluation protocol itself
against an independent implementation.
