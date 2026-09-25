r"""
Window-length sweep methods for protocol V5 (revision-srep-v5, task T2.2 / E1)
==============================================================================
Adds three simple smoothers and builds the full method set for one window
length N.  No existing module is changed.

New smoothers (all vectorised over items, one row per item, oldest -> newest):

  SMA       mean of the N-slot window.
  EWMA-eq   exponential smoothing with the memory-equivalent weight
            alpha = 2 / (N + 1), started at the first sample of the window
            (same recursion as ``fast_evaluator.batch_ewma``).
  Holt      row-wise replica of ``methods.forecasting_baselines.HoltForecast``
            (alpha = 0.3, gamma = 0.1, horizon h = 7; the settings of the
            predcmp run).  Score = sum_{k=1..h} max(0, l + k b).

Settings approved by Sajjad (25 Sep 2026, chat 5):
  - grid N in {7, 16, 32, 64, 128}; the three wavelet-based methods use J = 3
    and are run only for N >= 16 (J = 3 needs at least 16 samples);
  - min_observations: baselines and the three smoothers 3 rows at every N;
    wavelet-based methods min(32, N / 2) rows (16 -> 8, 32 -> 16, 64 -> 32,
    128 -> 32);
  - evaluation horizon 1 slot (unchanged).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .fast_evaluator import (FastMethod, batch_af, batch_compound, batch_ewma,
                             batch_pfrf, batch_rrd, batch_vse)

SWEEP_WINDOWS = (7, 16, 32, 64, 128)
BASELINES = ('AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF')
SMOOTHERS = ('SMA', 'EWMA-eq', 'Holt')
WAVELET_BASED = ('DWT+AF', 'DTCWT+AF', 'WSPI')
BASELINE_MIN_OBS = 3
HOLT_ALPHA, HOLT_GAMMA, HOLT_H = 0.3, 0.1, 7


def batch_sma(X: np.ndarray) -> np.ndarray:
    return X.astype(np.float64).mean(axis=1)


def make_batch_ewma_eq(window: int):
    alpha = 2.0 / (window + 1.0)

    def f(X):
        return batch_ewma(X, alpha=alpha)
    f.__name__ = f'ewma_eq_a{alpha:.4f}'
    f.alpha = alpha
    return f


def make_batch_holt(alpha: float = HOLT_ALPHA, gamma: float = HOLT_GAMMA,
                    horizon: int = HOLT_H):
    ks = np.arange(1, horizon + 1, dtype=np.float64)

    def f(X):
        X = X.astype(np.float64)
        n = X.shape[1]
        if n == 0:
            return np.zeros(X.shape[0])
        if n == 1:
            return np.clip(X[:, 0], 0, None) * horizon
        level = X[:, 0].copy()
        trend = X[:, 1] - X[:, 0]
        for t in range(1, n):
            prev = level
            level = alpha * X[:, t] + (1 - alpha) * (level + trend)
            trend = gamma * (level - prev) + (1 - gamma) * trend
        fc = level[:, None] + ks[None, :] * trend[:, None]
        out = np.clip(fc, 0, None).sum(axis=1)
        bad = ~np.isfinite(fc).all(axis=1)          # HoltForecast fallback
        out[bad] = X[bad, -1] * horizon
        return out
    f.__name__ = f'holt_a{alpha}_g{gamma}_h{horizon}'
    return f


def wavelet_min_obs(window: int) -> int:
    return int(min(32, window // 2))


def build_sweep_methods(window: int, level: int = 3,
                        names: Optional[List[str]] = None) -> Dict[str, FastMethod]:
    """All methods of the sweep at window length ``window`` (protocol V5)."""
    from .protocol_v5 import make_v5_dtcwt_af, make_v5_dwt, make_v5_wspi
    fns = {'AF': batch_af, 'EWMA': batch_ewma, 'RRD': batch_rrd, 'VSE': batch_vse,
           'CompoundPop': batch_compound, 'PFRF': batch_pfrf,
           'SMA': batch_sma, 'EWMA-eq': make_batch_ewma_eq(window), 'Holt': make_batch_holt()}
    out = {}
    for n in BASELINES + SMOOTHERS:
        if names and n not in names:
            continue
        out[n] = FastMethod(n, window, BASELINE_MIN_OBS, fns[n])
    if window >= 2 ** (level + 1):
        makers = {'DWT+AF': make_v5_dwt, 'DTCWT+AF': make_v5_dtcwt_af, 'WSPI': make_v5_wspi}
        for n in WAVELET_BASED:
            if names and n not in names:
                continue
            out[n] = FastMethod(n, window, wavelet_min_obs(window),
                                makers[n](window=window, level=level))
    return out
