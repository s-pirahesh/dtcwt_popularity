# WSPI: Wavelet-Structured Popularity Index

This repository contains the code of the paper *"WSPI: A Wavelet-Structured
Popularity Index for Stable Content Ranking via Trend-Noise Separation"*
(Scientific Reports, revised version). The methods rank data items by their
share of demand from the recent request history, for example to guide caching,
prefetching and capacity placement.

The paper proposes three training-free, wavelet-based methods:

| Method | Idea |
|---|---|
| **WSPI** | Dual-tree complex wavelet transform (DTCWT) of the last *N* slots. Score `P = mu_L * exp(alpha*R - beta*W_E)`: `mu_L` is a recency-weighted mean of the magnitudes of the approximation (trend) coefficients, `R` the share of energy in the trend band and `W_E` the normalised wavelet entropy of the band energies. Defaults: `N = 64`, `J = 3`, `alpha = beta = 1`, filters `near_sym_a` / `qshift_a` (`methods/wspi_assessment.py`). |
| **DTCWT+AF** | Access-frequency (recency-weighted) score of the magnitudes of the DTCWT approximation coefficients, plus 0.1 times that of the finest detail coefficients; `J = 3`, same filters as WSPI. |
| **DWT+AF** | The same with the discrete wavelet transform (db4, `J = 3`). |

They are compared with six time-domain baselines: AF, EWMA, RRD, VSE,
CompoundPop and PFRF (`baselines/`).

## Installation

Python 3.12 was used for all reported results.

```
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions of the reported runs. Note that
`dtcwt` 0.14.0 needs NumPy < 2.

## Quick start

Score one series with WSPI:

```python
import numpy as np
from methods.wspi_assessment import WSPIAssessment

wspi = WSPIAssessment()
history = np.random.poisson(5, size=64).astype(float)   # last 64 slots
print(wspi.assess_single(history))
```

Evaluate all nine methods on a prepared dataset (evaluation protocol of the paper):

```
python tools/run_v5_eval.py --data data/datasets/youtube_hourly.csv --min-obs 50 --causal-universe --out results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/youtube_hourly
```

How to get the datasets and how to rebuild every experiment, table and figure
of the paper: see [REPRODUCE.md](REPRODUCE.md).

## Repository layout

| Folder | Content |
|---|---|
| `methods/` | WSPI, DTCWT+AF, DWT+AF and the ablation variants of WSPI |
| `baselines/` | The six time-domain baselines |
| `evaluation/` | Evaluation protocol (`protocol_v5.py`, `fast_evaluator.py`), metrics (NDCG@K, Spearman rho, RSI@K, rank distortion), window sweep and responsiveness analysis |
| `data/converters/` | Converters from the public raw data to the standard CSV format (see [data/README.md](data/README.md)) |
| `prepare_data.py` | Command-line front end of the converters |
| `tools/` | One program per experiment (`run_*.py`), statistics (`stats_report.py`), audits and unit tests (`test_*.py`) |
| `scripts/` | Programs that build the figures and tables of the paper and its Supplementary Information from the result files |
| `results/` | Output of the experiments (CSV and JSON; not kept in Git) |

## Evaluation protocol in short

Every slot *t* the methods score all items from the window of the *N* slots
before *t*, and the ranking is compared with the true counts of slot *t*.
Baselines use a 7-slot window by default and the wavelet-based methods 64
slots; a second configuration gives every method the same 64-slot window. Only
windows common to all methods are compared. The item list is causal: an item
enters only when its counts up to *t* reach the threshold. Statistics use
circular block bootstrap confidence intervals and block-level Wilcoxon tests
with Holm correction (`tools/stats_report.py`).

## Which code produced the results of the paper

All results of the revised paper come from `evaluation/protocol_v5.py`, run by
`tools/run_v5_eval.py` and the other `tools/run_*.py` programs. It scores all
items of a window at once with the batched scorers of
`evaluation/fast_evaluator.py`: the six baselines, and the three wavelet-based
methods with `J = 3` and left `reflect` padding only for series shorter than
the window (the first windows). The robustness column of the main tables comes
from `ProtocolV5Evaluator._robustness_v5` (seed 42); the wider perturbation
study is `tools/run_robustness_v5.py` (seeds 42 to 46).

The per-item classes are kept as references; the unit tests check that the
batched scorers give the same scores.

- `methods/wspi_assessment.py` is the WSPI index of the paper.
- `methods/dtcwt_assessment.py` and `methods/dwt_assessment.py` are the classes
  of the first submission. They pad short series with `edge`, and
  `DWTAssessment` lowers the level for short series; the protocol of the paper
  does neither.
- `methods/hybrid_assessment.py` and `methods/wspi_ablation.py` hold an earlier
  WSPI formula with a slope term and a clip. They are not used for any result.
- `evaluation/scenarios.py`, `evaluation/temporal_evaluator.py` and
  `evaluation/results_analyzer.py` are the evaluation of the first submission;
  `evaluation/fast_evaluator.py` in `compat` mode reproduces it.
- Only the flat entries of `WAVELET_CONFIG` in `config.py` are read (by the
  per-item classes); the other blocks of `config.py` are not read.

## Citation

If you use this code, please cite the paper and the archived release of this
repository on Zenodo: <https://doi.org/10.5281/zenodo.23062475> (this DOI
always points to the latest version; every version also has its own DOI).

## License

MIT, see [LICENSE](LICENSE).
