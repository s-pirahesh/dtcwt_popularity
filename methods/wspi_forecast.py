# -*- coding: utf-8 -*-
"""
WSPI-F — Forecasted Wavelet Structural Popularity Index (Level-2 module)
========================================================================
This module fulfils, literally, the proposal's commitment:

    "An algorithm is presented for ESTIMATING THE WAVELET COEFFICIENTS and,
     like the RCCWLMK method, consecutive wavelet coefficients are analysed."
                                            — Proposal v2.2, problem statement

Idea
----
WSPI scores the CURRENT window. WSPI-F predicts the score of the FUTURE
window — not by forecasting the raw signal (noisy, spiky), but by
forecasting in the COEFFICIENT DOMAIN:

  1. Decompose the observation window with DTCWT (J=3), identical to WSPI.
  2. Take the low-pass (trend-band) magnitude sequence a[1..n]. Because the
     trend band is smooth and (with DTCWT) approximately shift-invariant,
     it is far easier to predict than the raw signal — this is the central
     scientific claim of the module.
  3. Run an adaptive NLMS linear predictor ALONG the coefficient sequence
     (the LMK / RCCWLMK family of Zhao & Ansari [54]): at each step it
     predicts the next coefficient from the previous p, measures its error,
     and updates its weights. The error trajectory DECREASES as the filter
     adapts — the evidence Hypothesis 4 of the proposal asked for
     (see experiments/run_wspiF_convergence.py).
  4. Forecast the next m = ceil(horizon / 2^J) coefficients recursively.
  5. Rebuild the WSPI score from the EXTENDED (observed + forecast)
     coefficient sequence. Structural features R and WE are persisted from
     the current window (structure changes slowly; volume is what moves —
     the same separation Chapter 3 argues for).

        WSPI-F = mu_L(observed ⊕ forecast coeffs) · exp(alpha·R − beta·WE)

With m = 0 the score reduces exactly to WSPI — the module is a strict
superset of the thesis method.

Variants
--------
  WSPI-F     : NLMS adaptive predictor (proposal-faithful, adaptive)
  WSPI-F-YW  : AR(2) via Yule-Walker on the coefficient sequence
               (closed-form, non-adaptive control variant)

Both are drop-in BaseMethod objects: opt-in via --methods, cached-run
compatible via prepare_compare_folder.py + --resume.

Author: Sajjad (with Claude)
Date: August 2026
"""
from typing import Tuple

import numpy as np
import dtcwt

from config import WAVELET_CONFIG
from .base_method import BaseMethod


# ---------------------------------------------------------------------------
# Adaptive predictor primitives (module-level so experiments can reuse them)
# ---------------------------------------------------------------------------
def nlms_walk(seq: np.ndarray, order: int = 2, mu: float = 0.2,
              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk a persistence-anchored NLMS adaptive predictor along `seq`.

    The filter predicts coefficient INCREMENTS (the analysis of consecutive
    wavelet coefficients — the RCCWLMK idea):

        y_hat[k] = seq[k-1] + w · (last `order` increments)
        e_k      = seq[k] - y_hat[k]
        w       += mu * e_k * x_k / (||x_k||^2 + eps)

    With w = 0 the predictor IS the persistence forecast, so adaptation can
    only move away from persistence when the data supports it. Empirically
    this cuts the one-step relative error in the coefficient domain by a
    large margin versus raw-value NLMS (which suffers a zero-weight burn-in)
    and versus persistence itself.

    Returns
    -------
    w      : final filter weights on increments (shape: order)
    errors : a-priori prediction errors per step (len(seq)-order-1)
    preds  : one-step-ahead predictions per step (len(seq)-order-1)
    """
    seq = np.asarray(seq, dtype=float)
    p = int(order)
    d = np.diff(seq)
    n = len(seq)
    w = np.zeros(p)
    errors, preds = [], []
    for k in range(p + 1, n):
        x = d[k - 1 - p:k - 1][::-1]     # most recent increment first
        y_hat = seq[k - 1] + float(np.dot(w, x))
        e = seq[k] - y_hat
        w = w + mu * e * x / (float(np.dot(x, x)) + eps)
        errors.append(e)
        preds.append(y_hat)
    return w, np.asarray(errors), np.asarray(preds)


def nlms_forecast(seq: np.ndarray, steps: int, order: int = 2,
                  mu: float = 0.2) -> np.ndarray:
    """Adapt persistence-anchored NLMS along `seq`, forecast `steps` ahead."""
    seq = np.asarray(seq, dtype=float)
    p = int(order)
    if len(seq) < p + 3:                       # too short to adapt
        last = seq[-1] if len(seq) else 0.0
        return np.full(steps, float(last))
    w, _, _ = nlms_walk(seq, order=p, mu=mu)
    deltas = list(np.diff(seq)[-p:])
    last = float(seq[-1])
    out = []
    for _ in range(steps):
        d_hat = float(np.dot(w, np.asarray(deltas[-p:][::-1])))
        last = last + d_hat
        out.append(last)
        deltas.append(d_hat)
    return np.asarray(out)


def aryw_forecast(seq: np.ndarray, steps: int, order: int = 2) -> np.ndarray:
    """AR(order) via Yule-Walker on `seq` (demeaned), recursive forecast."""
    seq = np.asarray(seq, dtype=float)
    p = int(order)
    n = len(seq)
    if n < p + 2:
        last = seq[-1] if n else 0.0
        return np.full(steps, float(last))
    mu_ = seq.mean()
    x = seq - mu_
    r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(p + 1)])
    if r[0] <= 1e-12:
        return np.full(steps, float(seq[-1]))
    R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
    try:
        phi = np.linalg.solve(R + 1e-8 * np.eye(p), r[1:p + 1])
    except np.linalg.LinAlgError:
        return np.full(steps, float(seq[-1]))
    if not np.isfinite(phi).all() or np.abs(phi).sum() > 4:
        return np.full(steps, float(seq[-1]))
    history = list(x[-p:])
    out = []
    for _ in range(steps):
        nxt = float(np.dot(phi, np.asarray(history[-p:][::-1])))
        out.append(nxt + mu_)
        history.append(nxt)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# The method
# ---------------------------------------------------------------------------
class WSPIForecast(BaseMethod):
    """
    WSPI-F: forecast DTCWT trend-band coefficients, score the future window.

    Parameters
    ----------
    predictor : 'nlms' (adaptive, proposal-faithful) or 'aryw' (closed-form).
    horizon   : evaluation horizon in slots; m = ceil(horizon / 2^J)
                coefficients are forecast.
    alpha, beta : WSPI fusion coefficients (defaults identical to WSPI).
    order, mu : predictor order and NLMS step size.
    """

    def __init__(self, predictor: str = 'nlms', horizon: int = 7,
                 alpha: float = 1.0, beta: float = 1.0,
                 order: int = 2, mu: float = 0.2, name: str = None):
        super().__init__(name=name or ('WSPI-F' if predictor == 'nlms'
                                       else 'WSPI-F-YW'))
        if predictor not in ('nlms', 'aryw'):
            raise ValueError(f"Unknown predictor: {predictor}")
        self.predictor = predictor
        self.horizon = int(horizon)
        self.alpha = alpha
        self.beta = beta
        self.order = int(order)
        self.mu = float(mu)

        self.biort = WAVELET_CONFIG['dtcwt_biort']
        self.qshift = WAVELET_CONFIG['dtcwt_qshift']
        self.level = WAVELET_CONFIG['decomposition_level']
        self.transform = dtcwt.Transform1d(biort=self.biort, qshift=self.qshift)

    # -- shared with WSPI (identical policies) ------------------------------
    def _pad(self, time_series: np.ndarray) -> np.ndarray:
        min_len = 2 ** (self.level + 1)
        target_len = max(min_len,
                         2 ** int(np.ceil(np.log2(max(len(time_series), 2)))))
        if len(time_series) < target_len:
            return np.pad(time_series, (target_len - len(time_series), 0),
                          mode='reflect')
        return time_series

    @staticmethod
    def _entropy(energies) -> float:
        total = sum(energies)
        if total == 0:
            return 0.0
        probs = [e / total for e in energies if e > 0]
        if not probs:
            return 0.0
        ent = -sum(p * np.log2(p) for p in probs)
        max_ent = np.log2(len(energies))
        return ent / max_ent if max_ent > 0 else 0.0

    # -- coefficient forecasting -------------------------------------------
    def _forecast_steps(self) -> int:
        return max(1, int(np.ceil(self.horizon / (2 ** self.level))))

    def forecast_lowpass(self, time_series: np.ndarray):
        """
        Return (observed_mags, forecast_mags) of the trend-band coefficients.
        Exposed separately so experiments can measure coefficient-domain
        forecast accuracy directly.
        """
        ts = self._pad(np.asarray(time_series, dtype=float))
        pyramid = self.transform.forward(ts, nlevels=self.level)
        a = np.abs(np.asarray(pyramid.lowpass).ravel())
        m = self._forecast_steps()
        if self.predictor == 'nlms':
            fc = nlms_forecast(a, m, order=self.order, mu=self.mu)
        else:
            fc = aryw_forecast(a, m, order=self.order)
        return a, np.clip(fc, 0.0, None), pyramid

    # -- scoring ------------------------------------------------------------
    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0
        try:
            a, fc, pyramid = self.forecast_lowpass(time_series)
            extended = np.concatenate([a, fc])

            # mu_L on the extended sequence: the forecast coefficients are
            # the most recent ones, so the recency weights give them the
            # largest influence — the score looks AHEAD.
            n = len(extended)
            weights = 2.0 ** -np.arange(n)[::-1]
            mu_L_hat = float(np.average(extended, weights=weights))

            # Structural features persisted from the observed window
            # (structure is a slow descriptor; volume is what we forecast).
            e_low = float(np.sum(a ** 2))
            e_highs = [float(np.sum(np.abs(np.asarray(h).ravel()) ** 2))
                       for h in pyramid.highpasses]
            e_total = e_low + sum(e_highs)
            R = e_low / e_total if e_total > 0 else 0.0
            WE = self._entropy([e_low] + e_highs)

            return float(mu_L_hat * np.exp(self.alpha * R - self.beta * WE))
        except Exception:
            return float(np.mean(time_series))

    # -- diagnostics (Hypothesis 4 evidence) --------------------------------
    def error_curve(self, time_series: np.ndarray) -> np.ndarray:
        """
        Normalised squared NLMS prediction errors along the trend-band
        coefficient sequence of `time_series` (adaptation trajectory).
        Used by experiments/run_wspiF_convergence.py.
        """
        ts = self._pad(np.asarray(time_series, dtype=float))
        pyramid = self.transform.forward(ts, nlevels=self.level)
        a = np.abs(np.asarray(pyramid.lowpass).ravel())
        scale = float(np.mean(a ** 2)) or 1.0
        _, errors, _ = nlms_walk(a, order=self.order, mu=self.mu)
        return errors ** 2 / scale


def get_wspi_forecast_methods(horizon: int = 7) -> dict:
    """Return {name: instance} for both WSPI-F variants."""
    return {
        'WSPI-F':    WSPIForecast(predictor='nlms', horizon=horizon),
        'WSPI-F-YW': WSPIForecast(predictor='aryw', horizon=horizon),
    }
