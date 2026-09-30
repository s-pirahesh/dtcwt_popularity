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
| **DTCWT+AF** | Recency-weighted approximation coefficients of the DTCWT (`methods/dtcwt_assessment.py`). |
| **DWT+AF** | Recency-weighted approximation coefficients of the discrete wavelet transform, db4 (`methods/dwt_assessment.py`). |

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

## Citation

If you use this code, please cite the paper and the archived release of this
repository (see `.zenodo.json`).

## License

MIT, see [LICENSE](LICENSE).
