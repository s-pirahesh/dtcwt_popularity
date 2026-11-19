# Changelog

Versions follow Semantic Versioning: a new capability raises the minor number
and no earlier behaviour changes. Every version has its own tag and its own DOI
on Zenodo (see the README).

## 4.1.1

Completes 4.1.0. The methods, the evaluation protocol and the run programs of
4.1.0 are unchanged. Release 4.1.0 holds the code only; its documentation and
the programs that build the tables and figures came with 4.1.1.

### Added

- `tools/prediction_numbers.py`: collects the numbers of runs A and B into
  `prediction_numbers.csv`.
- `tools/prediction_figures.py`: the figures and tables of the study from the
  result files (main text, supplementary tables, test part, held-out data and
  the aggregate frame).
- `tools/prediction_test_analysis.py`, `tools/prediction_block_sensitivity.py`,
  `tools/prediction_mechanism_read.py`, `tools/prediction_aggregate_analysis.py`:
  the decision files of the test part, the held-out data and the mechanism
  control, the sensitivity of the verdicts to the block length, and the
  aggregate view. They read result files only.
- `tools/compare_prediction_outputs.py` and its unit test: compare regenerated
  tables, figures and decision files with stored copies.
- `README.md`, `REPRODUCE.md` (sections 18 to 25), `CITATION.cff` and this file.

### Changed

- The default result folder of the prediction programs is `results/prediction`.
  A result folder written by 4.1.0 must be renamed to `results/prediction`.
- `requirements.txt`: comments only.
- `setup.py` and `.zenodo.json`: version 4.1.1 and the description of the
  release.

## 4.1.0

Adds forecasting in the wavelet-coefficient domain and the programs of its
evaluation. Nothing of 4.0.x changes in behaviour.

### Added

- `methods/wspi_forecast.py`: coefficient-domain forecasting methods on top of
  WSPI (WSPI-F, WSPI-F-YW, WSPI-F2, WSPI-FT, WSPI-FG, WSPI-FA, WSPI-S, WSPI-FS1
  to WSPI-FS3), the window-mean level methods WSPI-M and WSPI-H, the mechanism
  control WSPI-HL, and the moving average MA64. The module header gives the
  formulas and the score bounds.
- `tools/prediction_protocol.py`, `tools/run_prediction.py`,
  `tools/prediction_resume.py`: batched scorers of the forecasting methods and
  of the time-domain forecasts (Persistence, Holt, AR(2), MA64, SMA, EWMA), the
  runner for horizons of 1, 8 and 24 slots, resume after an interruption, the
  bound check and the gate profile.
- `tools/learned_baseline.py`: the learned baselines GBM and GBM+W.
- `tools/run_prediction_plan.py`: run plans A, B, C, C_sens, C_nlms, D and E, one
  job per scenario and horizon, and `--confirm` for the held-out data.
- `tools/run_prediction_stats.py`: paired tests of the runs (tuning and test
  part, extra references, runs B, D and E, held-out data, `--block-rule`).
- `tools/prediction_decision.py`: decision rules for the gate and for the claims
  of a gain.
- `tools/check_reproduction.py`, `tools/estimate_runtime.py`,
  `tools/estimate_plan_runtime.py`, `tools/make_timing_sample.py`: reproduction
  check against stored horizon-1 results and run-time estimators.
- `experiments/select_gate_synthetic.py`, `experiments/nlms_convergence.py`.
- `data/converters/wikipedia_converter.py`: converter of the Wikipedia Web
  Traffic data. The held-out Yellow Taxi 2024 data use the existing converter.
- Unit tests `tools/test_*.py` for each of the programs above.

### Changed

- `evaluation/protocol_v5.py`: new optional argument `horizon` (default 1, which
  is the behaviour of 4.0.x; for `h` slots the ground truth of a window is the
  sum of the counts of the `h` slots from the test slot on).
- `requirements.txt`: added `statsmodels==0.15.0`, `scikit-learn==1.9.1` and
  `joblib==1.6.0`.
- `methods/__init__.py`: ASCII status messages (Windows console encoding).

## 4.0.0 to 4.0.3

Code of the paper "WSPI: A Wavelet-Structured Popularity Index for Stable
Content Ranking via Trend-Noise Separation": the methods, the baselines, the
evaluation protocol V5, the programs of every experiment and the programs that
build the tables and figures. Release 4.0.0 is the archive cited by the paper.
4.0.1 to 4.0.3 correct the documentation and the table and figure numbers of the
final layout.
