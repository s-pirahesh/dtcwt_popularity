r"""
Batch scorers of the prediction methods (protocol V5)
=====================================================
Row-wise (batch) versions of the prediction methods, for the evaluator
``evaluation/protocol_v5.py``.  Every method uses an N = 64 slot window and
needs 32 observed slots, the setting of the three wavelet-based methods, so
that all methods share the windows of WSPI and every paired test is window by
window.

Methods
-------
  Coefficient domain (DTCWT near_sym_a / qshift_a, J = 3), see
  ``methods/wspi_forecast.py`` for the definitions:
    WSPI-F     NLMS on the trend-band coefficients, no clamp, no gate
    WSPI-F-YW  Yule-Walker AR(2), no clamp, no gate
    WSPI-F2    Yule-Walker AR(2), clamp c = 4, structural gate (R(1-W_E))^0.3
    WSPI-FT    Theil-Sen, clamp c = 4, no gate
    WSPI-FG    Yule-Walker AR(2), clamp c = 4, forecastability (backtest) gate
               with k = 2 origins in the same window (source W)
    WSPI-FA    ARMA(1,1) by maximum likelihood (statsmodels), clamp c = 4,
               no gate; falls back to persistence on a failed fit

  Value forecasters on the raw window (the score is the sum of the forecasts
  over the horizon h, clipped at zero):
    Persistence, Holt (alpha 0.3, gamma 0.1), ARYW (AR(2) Yule-Walker),
    MA64 (mean of the last 64 slots times h), ARIMA (ARMA(1,1), statsmodels
    MLE; slow, row by row)

Horizon.  A coefficient-domain method forecasts m = ceil(h / 2^(J-1)) trend-band
coefficients (two DTCWT trees are interleaved in time, so two consecutive
coefficients are 2^(J-1) slots apart): m = 1, 2, 6 for h = 1, 8, 24.  For h = 1
this is the setting of the first paper; the scorers for h = 1 are the scorers
of the prediction runs of the thesis, unchanged.

Every batch scorer reproduces the per-item reference class
(``methods/wspi_forecast.py``, ``methods/forecasting_baselines.py``) on full
64-slot windows; ``tools/test_prediction.py`` checks this for h = 1, 8 and 24.
The only intended difference is the padding of the first windows (k < 64): the
series is reflect-padded on the left to 64 slots, exactly as WSPI under
protocol V5 (the per-item classes pad to the next power of two).

``TimedScorer`` wraps a scorer and measures the time spent inside it
(``scorer_timing.csv`` of ``tools/run_prediction.py``).

Author: Sajjad Pirahesh
Date: November 2026
"""
from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import dtcwt

from evaluation.fast_evaluator import FastMethod, _dtcwt_forward_rows
from evaluation.protocol_v5 import _pad_left_reflect, make_v5_wspi
from evaluation.sweep_methods import make_batch_holt
from methods.wspi_forecast import (AR_ORDER, BACKTEST_K, CLAMP, GATE_GAMMA, MA_WINDOW,
                                   NLMS_MU, ZERO_REL, arma11_forecast,
                                   n_forecast_coefficients)

LEVEL = 3
WINDOW = 64
MIN_OBS = 32
PHI_D = 1.0
ORDER = AR_ORDER

COEF_METHODS = ('WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT', 'WSPI-FG', 'WSPI-FA')
VALUE_METHODS = ('Persistence', 'Holt', 'ARYW', 'MA64', 'ARIMA')
ALL_METHODS = ('WSPI',) + COEF_METHODS + VALUE_METHODS
# Run A (all six scenarios) and run B (YouTube and hourly taxi only; slow methods)
RUN_A_METHODS = ('WSPI', 'WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
                 'Persistence', 'Holt', 'ARYW', 'MA64')
RUN_B_METHODS = ('WSPI', 'WSPI-FA', 'ARIMA')


# =============================================================================
# Row-wise predictor primitives (replicas of the per-item functions)
# =============================================================================

def ar2_yw_rows(seq: np.ndarray, steps: int) -> np.ndarray:
    """Row-wise replica of ``methods.wspi_forecast.aryw_forecast`` (order 2).

    AR(2) by Yule-Walker on the demeaned sequence, recursive forecast.
    Fallback (copy of the last value): r0 <= 1e-12, singular system,
    non-finite phi or sum|phi| > 4.  Needs n >= 4 (true for 16 coefficients).
    """
    seq = np.asarray(seq, dtype=np.float64)
    nr, n = seq.shape
    last = seq[:, -1]
    if n < ORDER + 2:
        return np.repeat(last[:, None], steps, axis=1)
    mu = seq.mean(axis=1)
    x = seq - mu[:, None]
    r0 = np.einsum('ij,ij->i', x, x) / n
    r1 = np.einsum('ij,ij->i', x[:, :n - 1], x[:, 1:]) / n
    r2 = np.einsum('ij,ij->i', x[:, :n - 2], x[:, 2:]) / n
    a = r0 + 1e-8
    det = a * a - r1 * r1
    with np.errstate(divide='ignore', invalid='ignore'):
        phi1 = (a * r1 - r1 * r2) / det
        phi2 = (a * r2 - r1 * r1) / det
    bad = (r0 <= 1e-12) | (det == 0) | ~np.isfinite(phi1) | ~np.isfinite(phi2) \
        | ((np.abs(phi1) + np.abs(phi2)) > 4)
    h1, h2 = x[:, -1].copy(), x[:, -2].copy()        # newest, second newest
    out = np.empty((nr, steps))
    for s in range(steps):
        nxt = phi1 * h1 + phi2 * h2
        out[:, s] = nxt + mu
        h2, h1 = h1, nxt
    out[bad] = last[bad, None]
    return out


def nlms_rows(seq: np.ndarray, steps: int, mu: float = NLMS_MU,
              eps: float = 1e-8) -> np.ndarray:
    """Row-wise replica of ``methods.wspi_forecast.nlms_forecast`` (order 2)."""
    seq = np.asarray(seq, dtype=np.float64)
    nr, n = seq.shape
    p = ORDER
    if n < p + 3:
        return np.repeat(seq[:, -1:], steps, axis=1)
    d = np.diff(seq, axis=1)
    w = np.zeros((nr, p))
    for k in range(p + 1, n):
        x = d[:, k - 1 - p:k - 1][:, ::-1]           # most recent increment first
        y_hat = seq[:, k - 1] + np.einsum('ij,ij->i', w, x)
        e = seq[:, k] - y_hat
        w = w + mu * (e / (np.einsum('ij,ij->i', x, x) + eps))[:, None] * x
    deltas = [d[:, -2], d[:, -1]]                      # oldest, newest
    last = seq[:, -1].copy()
    out = np.empty((nr, steps))
    for s in range(steps):
        d_hat = w[:, 0] * deltas[-1] + w[:, 1] * deltas[-2]
        last = last + d_hat
        out[:, s] = last
        deltas.append(d_hat)
    return out


def theilsen_rows(seq: np.ndarray, steps: int) -> np.ndarray:
    """Row-wise replica of ``methods.wspi_forecast.theilsen_forecast``."""
    seq = np.asarray(seq, dtype=np.float64)
    nr, n = seq.shape
    if n < 3:
        return np.repeat(seq[:, -1:], steps, axis=1)
    x = np.arange(n, dtype=np.float64)
    xi, xj = np.triu_indices(n, k=1)
    slopes = (seq[:, xj] - seq[:, xi]) / (x[xj] - x[xi])
    slope = np.median(slopes, axis=1)
    intercept = np.median(seq - slope[:, None] * x[None, :], axis=1)
    j = np.arange(1, steps + 1, dtype=np.float64)
    return intercept[:, None] + slope[:, None] * (n - 1 + j)[None, :]


def recency_mean_rows(seq: np.ndarray) -> np.ndarray:
    """mu_L of every row: recency-weighted (2^-k) mean."""
    n = seq.shape[1]
    w = 2.0 ** -np.arange(n)[::-1]
    return (seq * w).sum(axis=1) / w.sum()


def clamp_rows(raw: np.ndarray, anchor: np.ndarray, c: float = CLAMP) -> np.ndarray:
    """anchor * clip(raw / anchor, 1/c, c) per row; zero where anchor <= 0;
    then clipped at zero (replica of ``clamp_forecast``)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = raw / np.where(anchor > 0, anchor, 1.0)[:, None]
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.clip(ratio, 1.0 / c, c)
    fc = np.where(anchor[:, None] > 0, anchor[:, None] * ratio, 0.0)
    return np.clip(fc, 0.0, None)


def weighted_abs_rows(err: np.ndarray) -> np.ndarray:
    """sum_i 2^-(m-i) |err_i| per row (recency weights of the m forecasts)."""
    m = err.shape[1]
    w = 2.0 ** -np.arange(m)[::-1]
    return (np.abs(err) * w).sum(axis=1)


class _ArmaRows:
    """ARMA(1,1) forecast of every row (statsmodels MLE, one fit per row).

    ``n_forecasts`` and ``n_fallbacks`` count the fits and how many of them
    fell back to persistence."""

    def __init__(self, steps: int):
        self.steps = int(steps)
        self.n_forecasts = 0
        self.n_fallbacks = 0

    def __call__(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.float64)
        out = np.empty((seq.shape[0], self.steps))
        for i in range(seq.shape[0]):
            fc, fell_back = arma11_forecast(seq[i], self.steps)
            out[i] = fc
            self.n_forecasts += 1
            self.n_fallbacks += int(fell_back)
        return out


# =============================================================================
# Gates
# =============================================================================

def structural_gate(R: np.ndarray, WE: np.ndarray, gamma: float = GATE_GAMMA) -> np.ndarray:
    """Gate of WSPI-F2: (R (1 - W_E))^gamma, clipped to [0, 1]."""
    raw = np.clip(R * (1.0 - WE), 0.0, 1.0)
    return raw ** gamma


def backtest_gate_rows(a: np.ndarray, m: int, k: int = BACKTEST_K) -> np.ndarray:
    """Forecastability gate g in [0, 1] of WSPI-FG, source W (same window).

    For the origins t = n-m-k+1, ..., n-m the clamped AR(2) forecast of the
    prefix a[:, :t] and the reference mu_L(prefix) (repeated m times) are
    compared with the coefficients a[:, t:t+m], with the recency weights
    2^-(m-i).  g = max(0, 1 - S_F / S_mu); g = 0 if S_mu is zero up to
    rounding (S_mu <= ZERO_REL * mu_L(a) * k_eff).  Replica of
    ``methods.wspi_forecast.backtest_gate``; k is reduced to n-m-3 for short
    sequences (no backtest: g = 0)."""
    a = np.asarray(a, dtype=np.float64)
    nr, n = a.shape
    k_eff = min(int(k), n - m - 3)
    if k_eff < 1:
        return np.zeros(nr)
    s_f = np.zeros(nr)
    s_mu = np.zeros(nr)
    for t in range(n - m - k_eff + 1, n - m + 1):      # prefix a[:, :t]
        pre = a[:, :t]
        f = clamp_rows(ar2_yw_rows(pre, m), pre[:, -1])
        r = np.repeat(recency_mean_rows(pre)[:, None], m, axis=1)
        y = a[:, t:t + m]
        s_f += weighted_abs_rows(f - y)
        s_mu += weighted_abs_rows(r - y)
    return gate_from_sums(s_f, s_mu, recency_mean_rows(a) * k_eff)


def gate_from_sums(s_f: np.ndarray, s_mu: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """g = max(0, 1 - S_F / S_mu); g = 0 when S_mu is zero up to rounding."""
    pos = s_mu > ZERO_REL * scale
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(pos, np.maximum(0.0, 1.0 - s_f / np.where(pos, s_mu, 1.0)), 0.0)


# =============================================================================
# DTCWT features shared by the coefficient-domain methods
# =============================================================================

class CoefFeatures:
    """|L_J|, R and W_E of a batch (same formulas as ``make_batch_wspi``)."""

    def __init__(self, Xp: np.ndarray, tr, level: int = LEVEL):
        low, highs = _dtcwt_forward_rows(tr, Xp, level)
        self.a = np.abs(low)
        e_low = (self.a ** 2).sum(axis=1)
        e_h = np.stack([(np.abs(h) ** 2).sum(axis=1) for h in highs], axis=1)
        e_tot = e_low + e_h.sum(axis=1)
        self.R = np.where(e_tot > 0, e_low / np.where(e_tot > 0, e_tot, 1.0), 0.0)
        E = np.concatenate([e_low[:, None], e_h], axis=1)
        tot = E.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            P = E / np.where(tot > 0, tot, 1.0)[:, None]
            term = np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0)
        ent = -term.sum(axis=1)
        self.WE = np.where(tot > 0, ent / np.log2(E.shape[1]), 0.0)


def make_coef_scorer(name: str, horizon: int = 1, window: int = WINDOW,
                     level: int = LEVEL, alpha: float = 1.0, beta: float = 1.0,
                     gamma: float = GATE_GAMMA, k: int = BACKTEST_K) -> Callable:
    """Batch scorer of one coefficient-domain method (see module docstring).

    ``gamma`` is the exponent of the WSPI-F2 gate (the method uses 0.3; gamma = 1
    is only used to describe the gate).  ``k`` is the number of backtest origins
    of WSPI-FG (the method uses 2).  The scorer of WSPI-FA has a method
    ``fallback_counts()`` that returns (fits, fallbacks to persistence)."""
    if name not in COEF_METHODS:
        raise ValueError(name)
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    tgt = max(window, 2 ** (level + 1))
    m = n_forecast_coefficients(horizon, level)
    arma = _ArmaRows(m) if name == 'WSPI-FA' else None

    def f(X):
        F = CoefFeatures(_pad_left_reflect(np.asarray(X, dtype=np.float64), tgt), tr, level)
        a = F.a
        if name == 'WSPI-F':
            raw = nlms_rows(a, m)
        elif name == 'WSPI-FT':
            raw = theilsen_rows(a, m)
        elif name == 'WSPI-FA':
            raw = arma(a)
        else:
            raw = ar2_yw_rows(a, m)
        struct = np.exp(alpha * F.R - beta * F.WE)
        if name in ('WSPI-F', 'WSPI-F-YW'):
            # no clamp: forecast clipped at 0, mu_L of the extended sequence
            fc = np.clip(raw, 0.0, None)
            return recency_mean_rows(np.concatenate([a, fc], axis=1)) * struct
        # clamped methods: clamp relative to the last observed coefficient
        fc = clamp_rows(raw, a[:, -1])
        mu_L = recency_mean_rows(a)
        mu_hat = recency_mean_rows(np.concatenate([a, fc], axis=1))
        ok = np.isfinite(mu_L) & (mu_L > 1e-12)
        mu_hat = np.where(np.isfinite(mu_hat) & (mu_hat > 1e-12), mu_hat, mu_L)
        if name == 'WSPI-F2':
            g = structural_gate(F.R, F.WE, gamma)
        elif name == 'WSPI-FG':
            g = backtest_gate_rows(a, m, k)
        else:
            g = np.ones_like(mu_L)
        lam = PHI_D * g
        with np.errstate(divide='ignore', invalid='ignore'):
            blended = mu_L * np.power(np.where(ok, mu_hat / np.where(ok, mu_L, 1.0), 1.0), lam)
        return np.where(ok, blended * struct, 0.0)

    f.__name__ = f'{name.lower().replace("-", "_")}_h{horizon}_W{window}'
    if gamma != GATE_GAMMA:
        f.__name__ += f'_g{gamma:g}'
    if arma is not None:
        f.fallback_counts = lambda: (arma.n_forecasts, arma.n_fallbacks)
    return f


# =============================================================================
# Value forecasters
# =============================================================================

def _safe_sum_rows(fc: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    good = np.isfinite(fc).all(axis=1)
    out = np.clip(np.where(np.isfinite(fc), fc, 0.0), 0.0, None).sum(axis=1)
    return np.where(good, out, fallback)


def make_persistence(horizon: int = 1) -> Callable:
    def f(X):
        last = np.asarray(X, dtype=np.float64)[:, -1]
        return _safe_sum_rows(np.repeat(last[:, None], horizon, axis=1), last)
    f.__name__ = f'persistence_h{horizon}'
    return f


def make_aryw(horizon: int = 1) -> Callable:
    """Replica of ARYWForecast(order=2): fallback to the mean when the
    Yule-Walker fit is rejected (phi = 0), naive for very short series."""
    def f(X):
        X = np.asarray(X, dtype=np.float64)
        nr, n = X.shape
        naive = X[:, -1] * horizon
        if n < max(ORDER + 2, 4):
            return naive
        mu = X.mean(axis=1)
        x = X - mu[:, None]
        r0 = np.einsum('ij,ij->i', x, x) / n
        r1 = np.einsum('ij,ij->i', x[:, :n - 1], x[:, 1:]) / n
        r2 = np.einsum('ij,ij->i', x[:, :n - 2], x[:, 2:]) / n
        a = r0 + 1e-8
        det = a * a - r1 * r1
        with np.errstate(divide='ignore', invalid='ignore'):
            phi1 = (a * r1 - r1 * r2) / det
            phi2 = (a * r2 - r1 * r1) / det
        bad = (r0 <= 1e-12) | (det == 0) | ~np.isfinite(phi1) | ~np.isfinite(phi2) \
            | ((np.abs(phi1) + np.abs(phi2)) > 4)
        phi1 = np.where(bad, 0.0, phi1)
        phi2 = np.where(bad, 0.0, phi2)
        h1, h2 = x[:, -1].copy(), x[:, -2].copy()
        fc = np.empty((nr, horizon))
        for s in range(horizon):
            nxt = phi1 * h1 + phi2 * h2
            fc[:, s] = nxt + mu
            h2, h1 = h1, nxt
        return _safe_sum_rows(fc, naive)
    f.__name__ = f'aryw_h{horizon}'
    return f


def make_ma(horizon: int = 1, window: int = MA_WINDOW) -> Callable:
    """MA64: mean of the last ``window`` slots times h (shorter windows use
    all their slots).  Replica of ``MovingAverageForecast``."""
    def f(X):
        X = np.asarray(X, dtype=np.float64)
        return X[:, -window:].mean(axis=1) * horizon
    f.__name__ = f'ma{window}_h{horizon}'
    return f


def make_arima(horizon: int = 1) -> Callable:
    """ARIMAForecast(p=1, q=1) row by row (statsmodels MLE; slow)."""
    from methods.forecasting_baselines import ARIMAForecast
    obj = ARIMAForecast(p=1, q=1, horizon=horizon)

    def f(X):
        X = np.asarray(X, dtype=np.float64)
        out = np.empty(X.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(X.shape[0]):
                out[i] = obj.assess_single(X[i])
        return out
    f.__name__ = f'arima11_h{horizon}'
    return f


# =============================================================================
# Method set and timing
# =============================================================================

class TimedScorer:
    """Wraps a scorer and measures the time spent inside it.

    Counts every call, including the calls of the robustness test.  The name
    of the wrapped scorer is kept (the evaluator writes it into the run
    metadata).  For WSPI-FA the fits and fallbacks of the inner scorer are
    reported as well."""

    def __init__(self, scorer: Callable):
        self.scorer = scorer
        self.__name__ = getattr(scorer, '__name__', 'scorer')
        self.seconds = 0.0
        self.calls = 0
        self.rows = 0

    def __call__(self, X):
        t0 = time.perf_counter()
        try:
            return self.scorer(X)
        finally:
            self.seconds += time.perf_counter() - t0
            self.calls += 1
            self.rows += int(np.shape(X)[0])

    def stats(self) -> dict:
        out = dict(scorer=self.__name__, calls=self.calls, rows=self.rows,
                   seconds=self.seconds,
                   ms_per_row=(1000.0 * self.seconds / self.rows) if self.rows else float('nan'))
        counts = getattr(self.scorer, 'fallback_counts', None)
        if counts is not None:
            n_fit, n_fb = counts()
            out.update(n_fits=n_fit, n_fallbacks=n_fb,
                       fallback_share=(n_fb / n_fit) if n_fit else float('nan'))
        return out


def build_prediction_methods(horizon: int = 1, names: Optional[List[str]] = None,
                             window: int = WINDOW, min_obs: int = MIN_OBS,
                             timed: bool = False) -> Dict[str, FastMethod]:
    """FastMethod objects of the named methods (default: run A).

    timed=True wraps every scorer in a ``TimedScorer``."""
    makers = {
        'WSPI': lambda: make_v5_wspi(window=window, level=LEVEL),
        'Persistence': lambda: make_persistence(horizon),
        'Holt': lambda: make_batch_holt(horizon=horizon),
        'ARYW': lambda: make_aryw(horizon),
        'MA64': lambda: make_ma(horizon),
        'ARIMA': lambda: make_arima(horizon),
    }
    for n in COEF_METHODS:
        makers[n] = (lambda nn: (lambda: make_coef_scorer(nn, horizon, window)))(n)
    names = list(names) if names else list(RUN_A_METHODS)
    out = {}
    for n in names:
        if n not in makers:
            raise ValueError(f'unknown method {n}')
        scorer = makers[n]()
        out[n] = FastMethod(n, window, min_obs, TimedScorer(scorer) if timed else scorer)
    return out


def gate_features(X: np.ndarray, horizon: int = 1, window: int = WINDOW,
                  level: int = LEVEL, k: int = BACKTEST_K):
    """R, W_E, the gate of WSPI-F2 and the gate of WSPI-FG of every row
    (for the gate profile)."""
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    F = CoefFeatures(_pad_left_reflect(np.asarray(X, dtype=np.float64),
                                       max(window, 2 ** (level + 1))), tr, level)
    m = n_forecast_coefficients(horizon, level)
    return F.R, F.WE, structural_gate(F.R, F.WE), backtest_gate_rows(F.a, m, k)
