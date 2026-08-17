# A contamination-aware audit of time-series foundation models

Anonymous code release accompanying a submission under review. It reproduces
every number, table and figure in the paper.

Five released time-series foundation-model checkpoints are evaluated zero-shot
against seasonal-naive, automatically selected ETS and ARIMA, gradient boosting
on lag features, and a PatchTST reference, on 27 datasets from two public
suites, under identical evaluation windows and a disclosed tuning budget for
every method. For each (model, dataset) pair the release also records whether
that dataset lies inside the model's own pretraining corpus, taken from the
model developers' published corpus listings, so that accuracy can be reported
separately on data a model has and has not seen.

## Layout

```
configs/      datasets, model roster, contamination matrix, gate configuration
tsfmbench/    data loading, evaluation windows, metrics, model wrappers
runner/       pipeline.sh (one command per stage) and the job-based runner
scripts/      download, prepare, aggregate, statistics, gate, figures,
              external validation, corpus-membership checks
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # see the note on conflicting pins below

bash runner/pipeline.sh check          # verify every dataset is reachable
bash runner/pipeline.sh data           # download and prepare both suites
bash runner/pipeline.sh external       # cache the official reference values
bash runner/pipeline.sh selftest       # unit-test the review gate
bash runner/pipeline.sh smoke          # one dataset, full chain, a few minutes
bash runner/pipeline.sh micro          # measured timing, to size the full grid
bash runner/pipeline.sh full           # complete grid, resume enabled
bash runner/pipeline.sh final          # fresh directory, resume disabled, gated
```

Long stages should be run under `tmux` or `nohup`. The final stage writes to
`runs/final_<date>/` and ends by running the review gate, which exits non-zero
if anything would make the numbers unsafe to report.

## Contamination analysis

```bash
python scripts/fetch_external_reference.py    # leakage flags + reference values
python scripts/check_corpus_membership.py     # membership in the pretraining corpora
python scripts/draft_contamination.py         # draft the matrix from primary sources
python scripts/assemble_contamination.py --candidates-as contaminated --approve
```

Each cell of the matrix carries the file and entry it came from. Cells that the
published sources do not settle stay marked unknown rather than being guessed,
and the paper reports a robustness check over the remaining ambiguity. Until the
matrix is explicitly approved, the gate refuses to let a final run through.

## What is and is not in this repository

Tracked: all code, all configurations, and the approved contamination matrix
together with the corpus listings it was derived from, since these are the
evidence for the paper's central claim.

Not tracked, because they are large and regenerable: the downloaded datasets and
checkpoints (`data/`), the per-unit outputs and statistics of a run (`runs/`),
and the cached leaderboard reference values (`external/`). Each is produced by
the corresponding pipeline stage above.

## Figures

```bash
python scripts/make_figures.py runs/final_<date>
```

produces the win-and-loss chart and the critical-difference diagram. The
diagram's critical distance is computed from the Nemenyi statistic and drawn to
scale; the script cross-checks it against the computed post-hoc p-values on
every pair and warns if the two ever disagree.

## Note on dependency pins

Several foundation-model libraries pin older versions of numpy, torch or gluonts
than the rest of the stack requires. Install the core packages first, then those
libraries with `--no-deps`, adding any missing runtime dependencies
individually. Do not relax the core pins to satisfy a single library: if one
model cannot run on the shared stack, give it its own environment and set
`python: /path/to/that/env/bin/python` in its entry in `configs/models.yaml`.
The runner will launch only that model's units through that interpreter and
record which interpreter produced each unit.

## Licence

Released under the MIT Licence; see `LICENSE`.
