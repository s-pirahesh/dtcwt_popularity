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

Supplementary methods (run C and its sensitivity run; the methods of run A and
run B are unchanged):
    WSPI-S       smoothing control: clamped level forecast (mean of the last two
                 trend-band coefficients, no trend), lam = 1
    WSPI-FS<k>   WSPI-F2 with the structure-and-skill gate
                 (R(1-W_E))^0.3 * min(1, k S_mu / S_F), k in 1, 1.5, 2, 3
    SMA<N>       mean of the last N slots times h, N in 8, 16, 32, 64 (SMA64 = MA64)
    EWMA<N>      exponential mean over the window, alpha = 2 / (N + 1), started at
                 the first slot (the EWMA recursion of the first paper), times h
    WSPI-F2_c<c>_g<gamma>_p<phi_d>
                 WSPI-F2 with clamp c, gate exponent gamma and trust phi_d
                 (sensitivity: c in 2, 4, 8; gamma in 0.1, 0.3, 0.5; phi_d in 0.8, 1)
    GBM, GBM+W   learned baselines, trained before the run on the tuning windows
                 (``tools/learned_baseline.py``); passed to
                 ``build_prediction_methods`` with ``learned=``

Window-mean level methods (run D; the methods of runs A, B and C are unchanged):
    WSPI-M       every forecast trend-band coefficient is the plain mean of the
                 window coefficients, no clamp, lam = 1 (mean-reversion control)
    WSPI-H       the same, clamped to [mu_L / c, c mu_L] with c = 4, lam = 1;
                 P_H / WSPI lies in [1 - s_m (1 - 1/c), 1 + s_m (c - 1)], inside
                 [1/c, c] (``methods.wspi_forecast.level_anchor_bounds``)

Mechanism control (run E; the methods of runs A to D are unchanged):
    WSPI-HL      WSPI-H without the structural factor exp(alpha R - beta W_E)
                 (alpha = beta = 0): the score is the clamped level mu_hat alone;
                 P_HL / mu_L lies in [1 - s_m (1 - 1/c), 1 + s_m (c - 1)]

Author: Sajjad Pirahesh
Date: November 2026
"""
from __future__ import annotations

import time
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import dtcwt

from evaluation.fast_evaluator import FastMethod, _dtcwt_forward_rows, batch_ewma
from evaluation.protocol_v5 import _pad_left_reflect, make_v5_wspi
from evaluation.sweep_methods import make_batch_holt
from methods.wspi_forecast import (AR_ORDER, BACKTEST_K, CLAMP, GATE_GAMMA, MA_WINDOW,
                                   NLMS_MU, SKILL_KAPPAS, ZERO_REL, arma11_forecast,
                                   level_anchor_bounds, n_forecast_coefficients)

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

# Supplementary methods (run C, sensitivity run C_sens)
FS_METHODS = tuple(f'WSPI-FS{float(k):g}' for k in SKILL_KAPPAS)
MA_LENGTHS = (8, 16, 32, 64)
SMA_METHODS = tuple(f'SMA{n}' for n in MA_LENGTHS)
EWMA_METHODS = tuple(f'EWMA{n}' for n in MA_LENGTHS)
LEARNED_METHODS = ('GBM', 'GBM+W')
SENS_CLAMPS = (2.0, 4.0, 8.0)
SENS_GAMMAS = (0.1, 0.3, 0.5)
SENS_PHI_D = (0.8, 1.0)


def sens_name(c: float, gamma: float, phi_d: float) -> str:
    """Name of one sensitivity configuration of WSPI-F2."""
    return f'WSPI-F2_c{float(c):g}_g{float(gamma):g}_p{float(phi_d):g}'


SENS_METHODS = tuple(sens_name(c, g, p) for c in SENS_CLAMPS for g in SENS_GAMMAS
                     for p in SENS_PHI_D)
RUN_C_METHODS = ('WSPI', 'WSPI-S') + FS_METHODS + SMA_METHODS + EWMA_METHODS + LEARNED_METHODS
RUN_C_SENS_METHODS = ('WSPI',) + SENS_METHODS

# Window-mean level methods (run D)
LEVEL_METHODS = ('WSPI-M', 'WSPI-H')
RUN_D_METHODS = ('WSPI',) + LEVEL_METHODS

# Mechanism control (run E): WSPI-H without the structural factor
MECHANISM_METHODS = ('WSPI-HL',)
RUN_E_METHODS = ('WSPI', 'WSPI-H') + MECHANISM_METHODS


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
# Supplementary methods (run C)
# =============================================================================

def level_rows(seq: np.ndarray, steps: int) -> np.ndarray:
    """Row-wise replica of ``methods.wspi_forecast.level_forecast``: the mean
    of the last two values, repeated ``steps`` times."""
    seq = np.asarray(seq, dtype=np.float64)
    level = seq[:, -2:].mean(axis=1)
    return np.repeat(level[:, None], steps, axis=1)


def backtest_sums_rows(a: np.ndarray, m: int, k: int = BACKTEST_K):
    """S_F, S_mu and k_eff of the WSPI-FG backtest (same loop as
    ``backtest_gate_rows``).  Replica of ``methods.wspi_forecast.backtest_sums``."""
    a = np.asarray(a, dtype=np.float64)
    nr, n = a.shape
    k_eff = min(int(k), n - m - 3)
    s_f = np.zeros(nr)
    s_mu = np.zeros(nr)
    if k_eff < 1:
        return s_f, s_mu, 0
    for t in range(n - m - k_eff + 1, n - m + 1):
        pre = a[:, :t]
        f = clamp_rows(ar2_yw_rows(pre, m), pre[:, -1])
        r = np.repeat(recency_mean_rows(pre)[:, None], m, axis=1)
        y = a[:, t:t + m]
        s_f += weighted_abs_rows(f - y)
        s_mu += weighted_abs_rows(r - y)
    return s_f, s_mu, k_eff


def skill_factor_rows(s_f: np.ndarray, s_mu: np.ndarray, scale: np.ndarray,
                      kappa: float) -> np.ndarray:
    """min(1, kappa S_mu / S_F) per row; a sum below ZERO_REL * scale is zero;
    1 where S_F is zero.  Replica of ``methods.wspi_forecast.skill_factor``."""
    tol = ZERO_REL * scale
    pos_f = s_f > tol
    s_mu = np.where(s_mu > tol, s_mu, 0.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = kappa * s_mu / np.where(pos_f, s_f, 1.0)
    return np.where(pos_f, np.minimum(1.0, ratio), 1.0)


def parse_sens_name(name: str):
    """(c, gamma, phi_d) of a sensitivity name, or None."""
    if not name.startswith('WSPI-F2_c'):
        return None
    try:
        c, g, p = name[len('WSPI-F2_c'):].split('_')
        if not (g.startswith('g') and p.startswith('p')):
            return None
        return float(c), float(g[1:]), float(p[1:])
    except ValueError:
        return None


def make_control_scorer(name: str, horizon: int = 1, window: int = WINDOW,
                        level: int = LEVEL, alpha: float = 1.0, beta: float = 1.0,
                        clamp: float = CLAMP, gamma: float = GATE_GAMMA,
                        phi_d: float = PHI_D, kappa: Optional[float] = None,
                        k: int = BACKTEST_K) -> Callable:
    """Batch scorer of a clamped coefficient-domain method of run C:

      'WSPI-S'   level forecast, lam = phi_d
      'WSPI-FS'  AR(2) forecast, lam = phi_d * (R(1-W_E))^gamma * min(1, kappa S_mu / S_F)
      'WSPI-F2'  AR(2) forecast, lam = phi_d * (R(1-W_E))^gamma, with any clamp, gamma
                 and phi_d (sensitivity).  With c = 4, gamma = 0.3, phi_d = 1 it gives
                 the same floating-point result as ``make_coef_scorer('WSPI-F2')``.

    Same steps as the clamped branch of ``make_coef_scorer``."""
    if name not in ('WSPI-S', 'WSPI-FS', 'WSPI-F2'):
        raise ValueError(name)
    if name == 'WSPI-FS' and (kappa is None or not kappa > 0):
        raise ValueError('WSPI-FS needs kappa > 0')
    if clamp < 1.0 or not (0.0 <= phi_d <= 1.0) or gamma <= 0.0:
        raise ValueError('clamp >= 1, 0 <= phi_d <= 1 and gamma > 0 are required')
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    tgt = max(window, 2 ** (level + 1))
    m = n_forecast_coefficients(horizon, level)

    def f(X):
        F = CoefFeatures(_pad_left_reflect(np.asarray(X, dtype=np.float64), tgt), tr, level)
        a = F.a
        raw = level_rows(a, m) if name == 'WSPI-S' else ar2_yw_rows(a, m)
        struct = np.exp(alpha * F.R - beta * F.WE)
        fc = clamp_rows(raw, a[:, -1], clamp)
        mu_L = recency_mean_rows(a)
        mu_hat = recency_mean_rows(np.concatenate([a, fc], axis=1))
        ok = np.isfinite(mu_L) & (mu_L > 1e-12)
        mu_hat = np.where(np.isfinite(mu_hat) & (mu_hat > 1e-12), mu_hat, mu_L)
        if name == 'WSPI-S':
            g = np.ones_like(mu_L)
        else:
            g = structural_gate(F.R, F.WE, gamma)
            if name == 'WSPI-FS':
                s_f, s_mu, k_eff = backtest_sums_rows(a, m, k)
                g = g * skill_factor_rows(s_f, s_mu, mu_L * k_eff, kappa)
        lam = phi_d * g
        with np.errstate(divide='ignore', invalid='ignore'):
            blended = mu_L * np.power(np.where(ok, mu_hat / np.where(ok, mu_L, 1.0), 1.0), lam)
        return np.where(ok, blended * struct, 0.0)

    if name == 'WSPI-S':
        tag = 'wspi_s'
    elif name == 'WSPI-FS':
        tag = f'wspi_fs{kappa:g}'
    else:
        tag = f'wspi_f2_c{clamp:g}_g{gamma:g}_p{phi_d:g}'
    f.__name__ = f'{tag}_h{horizon}_W{window}'
    return f


def make_sma(horizon: int = 1, n: int = MA_WINDOW) -> Callable:
    """SMA<n>: mean of the last n slots times h (``make_ma`` with window n)."""
    f = make_ma(horizon, window=n)
    f.__name__ = f'sma{n}_h{horizon}'
    return f


def make_ewma(horizon: int = 1, n: int = 16) -> Callable:
    """EWMA<n>: exponential mean of the window with alpha = 2 / (n + 1),
    started at the first slot (``batch_ewma`` of the first paper), times h."""
    alpha = 2.0 / (n + 1.0)

    def f(X):
        return batch_ewma(np.asarray(X, dtype=np.float64), alpha) * horizon
    f.__name__ = f'ewma{n}_h{horizon}'
    return f


def mean_rows(seq: np.ndarray, steps: int) -> np.ndarray:
    """Row-wise replica of ``methods.wspi_forecast.mean_forecast``: the plain
    mean of every row, repeated ``steps`` times."""
    seq = np.asarray(seq, dtype=np.float64)
    return np.repeat(seq.mean(axis=1)[:, None], steps, axis=1)


def make_level_scorer(name: str, horizon: int = 1, window: int = WINDOW,
                      level: int = LEVEL, alpha: float = 1.0, beta: float = 1.0,
                      clamp: float = CLAMP) -> Callable:
    """Batch scorer of a window-mean level method (run D):

      'WSPI-M'   forecast = mean of the window coefficients, no clamp, lam = 1
                 (same steps as the unclamped branch of ``make_coef_scorer``)
      'WSPI-H'   the same forecast clamped relative to mu_L with c = ``clamp``,
                 lam = 1 (same steps as the clamped branch, anchor mu_L)
      'WSPI-HL'  WSPI-H without the structural factor (run E): the score is
                 the clamped level mu_hat alone

    Replica of ``methods.wspi_forecast.get_level_methods`` and
    ``get_mechanism_methods``."""
    if name not in LEVEL_METHODS + MECHANISM_METHODS:
        raise ValueError(name)
    if clamp < 1.0:
        raise ValueError('clamp >= 1 is required')
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    tgt = max(window, 2 ** (level + 1))
    m = n_forecast_coefficients(horizon, level)

    def f(X):
        F = CoefFeatures(_pad_left_reflect(np.asarray(X, dtype=np.float64), tgt), tr, level)
        a = F.a
        raw = mean_rows(a, m)
        struct = np.exp(alpha * F.R - beta * F.WE)
        if name == 'WSPI-M':
            fc = np.clip(raw, 0.0, None)
            return recency_mean_rows(np.concatenate([a, fc], axis=1)) * struct
        mu_L = recency_mean_rows(a)
        fc = clamp_rows(raw, mu_L, clamp)                  # anchor mu_L, not a_n
        mu_hat = recency_mean_rows(np.concatenate([a, fc], axis=1))
        ok = np.isfinite(mu_L) & (mu_L > 1e-12)
        mu_hat = np.where(np.isfinite(mu_hat) & (mu_hat > 1e-12), mu_hat, mu_L)
        with np.errstate(divide='ignore', invalid='ignore'):
            blended = mu_L * np.where(ok, mu_hat / np.where(ok, mu_L, 1.0), 1.0)
        if name == 'WSPI-HL':
            struct = 1.0                                   # alpha = beta = 0
        return np.where(ok, blended * struct, 0.0)

    tag = {'WSPI-M': 'wspi_m', 'WSPI-H': f'wspi_h_c{clamp:g}',
           'WSPI-HL': f'wspi_hl_c{clamp:g}'}[name]
    f.__name__ = f'{tag}_h{horizon}_W{window}'
    return f


def control_clamp(name: str) -> Optional[float]:
    """Clamp c of a clamped supplementary method (for the bound check), or None."""
    if name in ('WSPI-S', 'WSPI-H') or name in FS_METHODS:
        return CLAMP
    p = parse_sens_name(name)
    return p[0] if p else None


def control_bounds(name: str, horizon: int, n_trend: int = 16,
                   level: int = LEVEL):
    """(lower, upper) limits of P / WSPI for a supplementary method whose limits
    differ from ``methods.wspi_forecast.ratio_bounds`` (clamp anchored at a_n),
    or None.  WSPI-H: ``level_anchor_bounds`` (anchor mu_L), n_trend = 16
    trend-band coefficients of a 64-slot window with J = 3."""
    if name == 'WSPI-H':
        return level_anchor_bounds(n_trend, n_forecast_coefficients(horizon, level), CLAMP)
    return None


def supplementary_maker(name: str, horizon: int = 1, window: int = WINDOW) -> Optional[Callable]:
    """Factory of the scorer of a supplementary method, or None if the name
    is not one (learned methods are built in ``tools/learned_baseline.py``)."""
    if name == 'WSPI-S':
        return lambda: make_control_scorer('WSPI-S', horizon, window)
    if name in LEVEL_METHODS or name in MECHANISM_METHODS:
        return lambda: make_level_scorer(name, horizon, window)
    if name in FS_METHODS:
        kappa = float(name[len('WSPI-FS'):])
        return lambda: make_control_scorer('WSPI-FS', horizon, window, kappa=kappa)
    if name in SMA_METHODS:
        return lambda: make_sma(horizon, int(name[3:]))
    if name in EWMA_METHODS:
        return lambda: make_ewma(horizon, int(name[4:]))
    p = parse_sens_name(name)
    if p is not None:
        c, g, ph = p
        return lambda: make_control_scorer('WSPI-F2', horizon, window, clamp=c, gamma=g, phi_d=ph)
    return None


def iter_eligible_windows(fe, window: int = WINDOW, min_obs: int = MIN_OBS):
    """(k, item rows) of every window that ``ProtocolV5Evaluator.run_method``
    evaluates for a method with this window and min_obs (same rule, with the
    horizon of ``fe``)."""
    cum = np.zeros(len(fe.items)) if fe.causal_universe else None
    for k in range(fe.num_slots + 1):
        if cum is not None and k > 0:
            cum += fe.V[:, k - 1]
        if k + fe.horizon - 1 > fe.num_slots:
            break
        lo, hi = fe.window_bounds(k, window)
        if fe.any_cum[hi] - fe.any_cum[lo] == 0 or not fe.any_row[k]:
            continue
        ok = (fe.C[:, hi] - fe.C[:, lo]) >= min_obs
        if cum is not None:
            ok &= cum >= fe.universe_min_count
        idx = np.where(ok)[0]
        if len(idx) < 2:
            continue
        yield k, idx


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
                             timed: bool = False,
                             learned: Optional[Dict[str, Callable]] = None) -> Dict[str, FastMethod]:
    """FastMethod objects of the named methods (default: run A).

    timed=True wraps every scorer in a ``TimedScorer``.  The scorers of the
    learned methods (GBM, GBM+W) are trained elsewhere and given in ``learned``."""
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
    for n in names:
        if n in makers:
            continue
        if n in LEARNED_METHODS:
            if not learned or n not in learned:
                raise ValueError(f'{n} is a learned method: train it first '
                                 f'(tools/learned_baseline.py) and pass it in learned=')
            makers[n] = (lambda sc: (lambda: sc))(learned[n])
            continue
        mk = supplementary_maker(n, horizon, window)
        if mk is not None:
            makers[n] = mk
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
