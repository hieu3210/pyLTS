Linguistic Time Series (HA-LTS) reproduction notes

This folder contains files used to reproduce selected experiments from the paper.

Files generated:
- `NEW/docs/experiments_extracted.txt`: text excerpt extracted from lts.pdf (Experiments section).
- `NEW/Datasets/alabama.txt`: dataset copied from original `LTS/Datasets`.
- `NEW/src/lts_model.py`: simplified HA-LTS implementation following the paper's rules (intervals -> LLRs -> rules).
- `NEW/examples/run_lts.py`: simple runner that prints next-5 forecasts.
- `NEW/examples/evaluate_lts.py`: rolling-origin MAE evaluation comparing persistence vs HA-LTS.
- `NEW/examples/plot_lts.py`: creates `NEW/docs/plots/lts_forecast.png` showing forecasts and baselines.

Quick reproduction

Create and activate the project's venv, then run:

```bash
.venv/bin/python NEW/examples/plot_lts.py
```

This will print MAE values and save `NEW/docs/plots/lts_forecast.png`.

Notes and next work

- The implementation is a minimal, interpretable approximation of the HA-LTS rules from the paper. It does not yet implement the full SQM quantification (alpha/beta parameters) — that is a planned improvement.
- To reproduce the full experiment suite, we need to re-create additional datasets referenced in the paper and tune HA parameters (PSO tuner exists in the repo).
