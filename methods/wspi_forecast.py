# -*- coding: utf-8 -*-
"""
Coefficient-domain forecasting on top of WSPI
=============================================
WSPI scores the observed window:

    WSPI = mu_L * exp(alpha * R - beta * W_E)

where mu_L is the recency-weighted (2^-k) mean of the magnitudes a of the
DTCWT trend-band coefficients, R is the energy share of the trend band and
W_E is the normalised wavelet entropy.

The methods of this module forecast the next m trend-band coefficients, extend
the sequence a, and re-compute the recency-weighted mean on the extended
sequence (mu_hat).  The score is a geometric blend of the two means:

    P_F = mu_L^(1 - lam) * mu_hat^lam * exp(alpha * R - beta * W_E),   lam in [0, 1]

Structure
---------
  - Every method has this form.  The methods differ in the coefficient
    forecaster and in lam.
  - Lemma 1: if all m forecast coefficients equal mu_L, then mu_hat = mu_L.
    So WSPI is the member of this form that forecasts mu_L.
  - lam = 0 gives exactly WSPI (same floating-point operations).
  - Number of forecast coefficients: m = ceil(h / 2^(J-1)).  The two DTCWT
    trees are interleaved in time, so two consecutive trend-band coefficients
    are 2^(J-1) slots apart, not 2^J.  For h = 1 the rule gives m = 1.

Methods
-------
  name       forecaster         clamp   lam
  WSPI-F     NLMS (order 2)     no      1
  WSPI-F-YW  AR(2) Yule-Walker  no      1
  WSPI-F2    AR(2) Yule-Walker  c = 4   phi_d * (R (1 - W_E))^0.3  (structural gate)
  WSPI-FT    Theil-Sen          c = 4   1
  WSPI-FG    AR(2) Yule-Walker  c = 4   phi_d * g  (forecastability gate, below)
  WSPI-FA    ARMA(1,1), MLE     c = 4   1

Clamp.  A forecast coefficient is kept inside [a_n / c, c * a_n], where a_n is
the last observed coefficient.  The last element of the extended sequence has
weight 1 out of about 2, so one coefficient moves mu_hat by up to 50 percent.
The clamp bounds this leverage.

Forecastability gate (WSPI-FG).  The gate asks whether the forecaster beat the
implicit WSPI forecast (repeat mu_L) inside the observed window itself.  For
each of k backtest origins t (prefix a[:t], coefficients of the same window),
the forecast f and the reference r = mu_L(prefix) are compared with the
coefficients that followed:

    e_F(t)  = sum_i 2^-(m-i) |f_i - y_i|        e_mu(t) = sum_i 2^-(m-i) |r - y_i|
    g = max(0, 1 - sum_t e_F / sum_t e_mu)       (g = 0 if the sum of e_mu is zero)

g lies in [0, 1].  g = 0 gives exactly WSPI for that item.  The sum of e_mu is
treated as zero if it is below 1e-9 * mu_L(a) * k (rounding on constant
windows).  The method uses k = 2.  No data after the observed window is used.

Bounds.  With the clamp and lam in [0, 1], for n observed and m forecast
coefficients (W_n = 2 - 2^(1-n)):

    2^-m W_n / W_(n+m)  <=  mu_hat / mu_L  <=  B(m, c) W_n / W_(n+m)
    B(m, c) = c (2 - 2^(1-m)) + 2^-m

and P_F / WSPI lies between the lam-th powers of these two limits.  Both
limits are attained.  The methods without a clamp (WSPI-F, WSPI-F-YW) have no
such bound.

Every class is a per-item reference implementation (one series in, one score
out).  The tests in tools/test_wspi_forecast.py check it against WSPI.

Author: Sajjad Pirahesh
Date: November 2026
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import dtcwt

from config import WAVELET_CONFIG
from .base_method import BaseMethod

LEVEL = 3
CLAMP = 4.0                 # c
GATE_GAMMA = 0.3            # exponent of the structural gate of WSPI-F2
AR_ORDER = 2
NLMS_MU = 0.2
BACKTEST_K = 2              # backtest origins of WSPI-FG
ZERO_REL = 1e-9             # relative tolerance for "no change in the backtest"
ARMA_MIN_LENGTH = 8         # shorter sequences fall back to persistence
ARMA_MAXITER = 50
MA_WINDOW = 64

PREDICTORS = ('nlms', 'aryw', 'theilsen', 'arma11')
GATES = ('none', 'structural', 'backtest')


# =============================================================================
# Window arithmetic
# =============================================================================

def coefficient_step(level: int = LEVEL) -> int:
    """Slots between two consecutive trend-band coefficients: 2^(J-1)."""
    return 2 ** (level - 1)


def n_forecast_coefficients(horizon: int, level: int = LEVEL) -> int:
    """m = ceil(h / 2^(J-1)): coefficients needed to cover h slots."""
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")
    return int(np.ceil(int(horizon) / coefficient_step(level)))


def recency_weights(n: int) -> np.ndarray:
    """2^-k weights, last element weight 1."""
    return 2.0 ** -np.arange(n)[::-1]


def recency_mean(seq: np.ndarray) -> float:
    """mu_L: recency-weighted mean (same formula as WSPIAssessment)."""
    seq = np.asarray(seq, dtype=float)
    return float(np.average(seq, weights=recency_weights(len(seq))))


def bound_factor(m: int, c: float) -> float:
    """B(m, c) = c (2 - 2^(1-m)) + 2^-m."""
    return float(c * (2.0 - 2.0 ** (1 - m)) + 2.0 ** (-m))


def ratio_bounds(n: int, m: int, c: float, lam: float = 1.0) -> Tuple[float, float]:
    """Lower and upper limit of P_F / WSPI for n observed and m forecast
    coefficients, clamp c and blend weight lam (see module docstring)."""
    w_n = 2.0 - 2.0 ** (1 - n)
    w_nm = 2.0 - 2.0 ** (1 - (n + m))
    lower = 2.0 ** (-m) * w_n / w_nm
    upper = bound_factor(m, c) * w_n / w_nm
    return float(lower ** lam), float(upper ** lam)


# =============================================================================
# Coefficient forecasters
# =============================================================================

def nlms_walk(seq: np.ndarray, order: int = AR_ORDER, mu: float = NLMS_MU,
              eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Persistence-anchored NLMS along ``seq``.

    The filter predicts increments: y_hat[k] = seq[k-1] + w . (last increments).
    With w = 0 it is the persistence forecast.

    Returns the final weights, the a-priori errors and the one-step
    predictions of every step."""
    seq = np.asarray(seq, dtype=float)
    p = int(order)
    d = np.diff(seq)
    w = np.zeros(p)
    errors, preds = [], []
    for k in range(p + 1, len(seq)):
        x = d[k - 1 - p:k - 1][::-1]            # most recent increment first
        y_hat = seq[k - 1] + float(np.dot(w, x))
        e = seq[k] - y_hat
        w = w + mu * e * x / (float(np.dot(x, x)) + eps)
        errors.append(e)
        preds.append(y_hat)
    return w, np.asarray(errors), np.asarray(preds)


def nlms_forecast(seq: np.ndarray, steps: int, order: int = AR_ORDER,
                  mu: float = NLMS_MU) -> np.ndarray:
    """Adapt the NLMS filter along ``seq`` and forecast ``steps`` ahead."""
    seq = np.asarray(seq, dtype=float)
    p = int(order)
    if len(seq) < p + 3:                          # too short to adapt
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


def aryw_forecast(seq: np.ndarray, steps: int, order: int = AR_ORDER) -> np.ndarray:
    """AR(order) by Yule-Walker on the demeaned ``seq``, recursive forecast.

    Falls back to repeating the last value for sequences shorter than
    order + 2, a constant sequence, a singular system, non-finite
    coefficients or sum |phi| > 4."""
    seq = np.asarray(seq, dtype=float)
    p = int(order)
    n = len(seq)
    if n < p + 2:
        last = seq[-1] if n else 0.0
        return np.full(steps, float(last))
    mean = seq.mean()
    x = seq - mean
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
        out.append(nxt + mean)
        history.append(nxt)
    return np.asarray(out)


def theilsen_forecast(seq: np.ndarray, steps: int) -> np.ndarray:
    """Theil-Sen robust line (median of pairwise slopes), extended ``steps``."""
    seq = np.asarray(seq, dtype=float)
    n = len(seq)
    if n < 3:
        last = seq[-1] if n else 0.0
        return np.full(steps, float(last))
    x = np.arange(n, dtype=float)
    xi, xj = np.triu_indices(n, k=1)
    slopes = (seq[xj] - seq[xi]) / (x[xj] - x[xi])
    slopes = slopes[np.isfinite(slopes)]
    if slopes.size == 0:
        return np.full(steps, float(seq[-1]))
    slope = float(np.median(slopes))
    intercept = float(np.median(seq - slope * x))
    return np.asarray([intercept + slope * (n - 1 + j) for j in range(1, steps + 1)])


def arma11_forecast(seq: np.ndarray, steps: int) -> Tuple[np.ndarray, bool]:
    """ARMA(1,1) by maximum likelihood (statsmodels), same settings as
    ``ARIMAForecast(p=1, q=1)`` of forecasting_baselines.py.

    Returns (forecast, fell_back).  It falls back to repeating the last value
    (fell_back = True) for sequences shorter than 8, constant sequences, a
    failed fit, or a non-finite forecast.  A missing statsmodels package
    raises ImportError."""
    from statsmodels.tsa.arima.model import ARIMA
    import warnings

    seq = np.asarray(seq, dtype=float)
    persist = np.full(steps, float(seq[-1]) if len(seq) else 0.0)
    if len(seq) < ARMA_MIN_LENGTH or np.allclose(seq, seq[0]):
        return persist, True
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = ARIMA(seq, order=(1, 0, 1), enforce_stationarity=False,
                          enforce_invertibility=False)
            fitted = model.fit(method_kwargs={'maxiter': ARMA_MAXITER})
            fc = np.asarray(fitted.forecast(steps=steps), dtype=float)
    except Exception:
        return persist, True
    if fc.shape != (steps,) or not np.isfinite(fc).all():
        return persist, True
    return fc, False


def clamp_forecast(raw: np.ndarray, anchor: float, c: Optional[float]) -> np.ndarray:
    """anchor * clip(raw / anchor, 1/c, c), zero if anchor <= 0, then clipped
    at zero.  c = None only clips at zero (no bound)."""
    raw = np.asarray(raw, dtype=float)
    if c is None:
        return np.clip(raw, 0.0, None)
    if anchor <= 0.0:
        return np.zeros(len(raw), dtype=float)
    ratio = raw / anchor
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    ratio = np.clip(ratio, 1.0 / c, c)
    return np.clip(anchor * ratio, 0.0, None)


# =============================================================================
# Forecastability gate of WSPI-FG
# =============================================================================

def backtest_gate(a: np.ndarray, steps: int,
                  predict: Callable[[np.ndarray, int], np.ndarray],
                  k: int = BACKTEST_K, c: Optional[float] = CLAMP) -> float:
    """Gate g in [0, 1] from k backtest origins inside the coefficients ``a``.

    For t = n-m-k+1, ..., n-m the forecaster ``predict`` is fitted on the
    prefix a[:t], forecasts m = ``steps`` coefficients (clamped to the last
    prefix coefficient) and is compared with a[t:t+m].  The reference is the
    repeated mu_L of the prefix.  Needs prefixes of at least 4 coefficients,
    so k is reduced to n-m-3 if the window is short (no backtest: g = 0)."""
    a = np.asarray(a, dtype=float)
    n = len(a)
    k_eff = min(int(k), n - steps - 3)
    if k_eff < 1:
        return 0.0
    w = recency_weights(steps)
    s_f = 0.0
    s_mu = 0.0
    for t in range(n - steps - k_eff + 1, n - steps + 1):
        prefix = a[:t]
        f = clamp_forecast(predict(prefix, steps), float(prefix[-1]), c)
        ref = recency_mean(prefix)
        y = a[t:t + steps]
        s_f += float(np.sum(w * np.abs(f - y)))
        s_mu += float(np.sum(w * np.abs(ref - y)))
    if s_mu <= ZERO_REL * recency_mean(a) * k_eff:
        return 0.0
    return float(max(0.0, 1.0 - s_f / s_mu))


# =============================================================================
# The methods
# =============================================================================

class WSPIForecast(BaseMethod):
    """Coefficient-domain forecast blended with WSPI (see module docstring).

    Parameters
    ----------
    predictor : 'nlms', 'aryw', 'theilsen' or 'arma11'.
    gate : 'none' (lam = phi_d), 'structural' (lam = phi_d * (R (1 - W_E))^gate_gamma)
        or 'backtest' (lam = phi_d * g, forecastability gate).
    horizon : evaluation horizon in slots; m = ceil(h / 2^(J-1)).
    clamp : c >= 1, or None for no clamp.
    phi_d : global trust in [0, 1].  phi_d = 0 gives exactly WSPI.
    backtest_k : number of backtest origins of the 'backtest' gate.
    alpha, beta : WSPI coefficients on R and W_E.
    order, mu : predictor order (nlms, aryw) and NLMS step size.

    Counters ``n_forecasts`` and ``n_fallbacks`` count the forecasts of the
    'arma11' predictor and how many of them fell back to persistence.
    """

    def __init__(self, predictor: str = 'aryw', gate: str = 'none', horizon: int = 1,
                 clamp: Optional[float] = CLAMP, phi_d: float = 1.0,
                 gate_gamma: float = GATE_GAMMA, backtest_k: int = BACKTEST_K,
                 alpha: float = 1.0, beta: float = 1.0, order: int = AR_ORDER,
                 mu: float = NLMS_MU, name: Optional[str] = None):
        super().__init__(name=name or 'WSPI-forecast')
        if predictor not in PREDICTORS:
            raise ValueError(f"Unknown predictor: {predictor}")
        if gate not in GATES:
            raise ValueError(f"Unknown gate: {gate}")
        if clamp is not None and clamp < 1.0:
            raise ValueError("clamp (c) must be >= 1 or None")
        if not (0.0 <= phi_d <= 1.0):
            raise ValueError("phi_d must be in [0, 1]")
        if gate == 'structural' and gate_gamma <= 0.0:
            raise ValueError("gate_gamma must be > 0")
        if gate == 'backtest' and int(backtest_k) < 1:
            raise ValueError("backtest_k must be >= 1")
        if predictor == 'arma11':
            import statsmodels  # noqa: F401  (fail early if it is missing)

        self.predictor = predictor
        self.gate = gate
        self.horizon = int(horizon)
        self.clamp = None if clamp is None else float(clamp)
        self.phi_d = float(phi_d)
        self.gate_gamma = float(gate_gamma)
        self.backtest_k = int(backtest_k)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.order = int(order)
        self.mu = float(mu)

        self.biort = WAVELET_CONFIG['dtcwt_biort']
        self.qshift = WAVELET_CONFIG['dtcwt_qshift']
        self.level = WAVELET_CONFIG['decomposition_level']
        self.transform = dtcwt.Transform1d(biort=self.biort, qshift=self.qshift)
        self.n_forecast = n_forecast_coefficients(self.horizon, self.level)

        self.n_forecasts = 0
        self.n_fallbacks = 0
        self._last_fallback = False

    # -- same policies as WSPIAssessment --------------------------------------
    def _pad(self, time_series: np.ndarray) -> np.ndarray:
        min_len = 2 ** (self.level + 1)
        target_len = max(min_len, 2 ** int(np.ceil(np.log2(max(len(time_series), 2)))))
        if len(time_series) < target_len:
            return np.pad(time_series, (target_len - len(time_series), 0), mode='reflect')
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

    def reset_counters(self) -> None:
        self.n_forecasts = 0
        self.n_fallbacks = 0

    # -- forecasting ----------------------------------------------------------
    def _predict(self, seq: np.ndarray, steps: int) -> np.ndarray:
        if self.predictor == 'nlms':
            return nlms_forecast(seq, steps, order=self.order, mu=self.mu)
        if self.predictor == 'aryw':
            return aryw_forecast(seq, steps, order=self.order)
        if self.predictor == 'theilsen':
            return theilsen_forecast(seq, steps)
        fc, self._last_fallback = arma11_forecast(seq, steps)
        return fc

    def forecast_lowpass(self, time_series: np.ndarray):
        """Return (a, f, pyramid): observed trend-band magnitudes, the
        forecast coefficients (clamped if a clamp is set) and the pyramid."""
        ts = self._pad(np.asarray(time_series, dtype=float))
        pyramid = self.transform.forward(ts, nlevels=self.level)
        a = np.abs(np.asarray(pyramid.lowpass).ravel())
        self._last_fallback = False
        raw = self._predict(a, self.n_forecast)
        if self.predictor == 'arma11':
            self.n_forecasts += 1
            self.n_fallbacks += int(self._last_fallback)
        anchor = float(a[-1]) if len(a) else 0.0
        return a, clamp_forecast(raw, anchor, self.clamp), pyramid

    def _structure(self, a: np.ndarray, pyramid) -> Tuple[float, float]:
        e_low = float(np.sum(a ** 2))
        e_highs = [float(np.sum(np.abs(np.asarray(h).ravel()) ** 2))
                   for h in pyramid.highpasses]
        e_total = e_low + sum(e_highs)
        R = e_low / e_total if e_total > 0 else 0.0
        return R, self._entropy([e_low] + e_highs)

    def gate_from(self, a: np.ndarray, R: float, WE: float) -> float:
        """Gate value in [0, 1] (1 if gate = 'none')."""
        if self.gate == 'none':
            return 1.0
        if self.gate == 'structural':
            raw = max(0.0, min(1.0, R * (1.0 - WE)))
            return float(raw ** self.gate_gamma)
        return backtest_gate(a, self.n_forecast, self._predict, self.backtest_k, self.clamp)

    def gate_value(self, time_series: np.ndarray) -> float:
        """Gate of one window (for gate profiles)."""
        a, _, pyramid = self.forecast_lowpass(time_series)
        R, WE = self._structure(a, pyramid)
        return self.gate_from(a, R, WE)

    # -- scoring --------------------------------------------------------------
    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0
        a, f, pyramid = self.forecast_lowpass(time_series)
        R, WE = self._structure(a, pyramid)
        mu_L = recency_mean(a)
        if not np.isfinite(mu_L) or mu_L <= 1e-12:
            return 0.0
        mu_hat = recency_mean(np.concatenate([a, f]))
        if not np.isfinite(mu_hat) or mu_hat <= 1e-12:
            mu_hat = mu_L
        lam = self.phi_d * self.gate_from(a, R, WE)
        blended = mu_L * (mu_hat / mu_L) ** lam           # lam = 0: exactly mu_L
        return float(blended * np.exp(self.alpha * R - self.beta * WE))

    def ratio_bounds(self, n: int, lam: Optional[float] = None) -> Tuple[float, float]:
        """Limits of P_F / WSPI for n observed coefficients (clamped methods)."""
        if self.clamp is None:
            raise ValueError("methods without a clamp have no bound")
        return ratio_bounds(n, self.n_forecast, self.clamp,
                            self.phi_d if lam is None else lam)


class MovingAverageForecast(BaseMethod):
    """MA64: mean of the last ``window`` slots times the horizon.

    The factor h does not change a ranking.  It puts the score on the scale of
    the ground truth, the sum of the next h slots.  Shorter series use all
    their slots.  (This class lives here so that forecasting_baselines.py
    stays unchanged.)"""

    def __init__(self, window: int = MA_WINDOW, horizon: int = 1,
                 name: Optional[str] = None):
        super().__init__(name=name or f'MA{int(window)}')
        if int(window) < 1 or int(horizon) < 1:
            raise ValueError("window and horizon must be >= 1")
        self.window = int(window)
        self.horizon = int(horizon)

    def assess_single(self, time_series: np.ndarray) -> float:
        ts = np.asarray(time_series, dtype=float)
        if ts.size == 0:
            return 0.0
        return float(np.mean(ts[-self.window:]) * self.horizon)


def get_coefficient_domain_methods(horizon: int = 1, include_arma: bool = True) -> Dict[str, WSPIForecast]:
    """The six coefficient-domain methods with their fixed settings.

    include_arma=False leaves out WSPI-FA (needs statsmodels)."""
    out = {
        'WSPI-F': WSPIForecast(predictor='nlms', gate='none', horizon=horizon,
                               clamp=None, name='WSPI-F'),
        'WSPI-F-YW': WSPIForecast(predictor='aryw', gate='none', horizon=horizon,
                                  clamp=None, name='WSPI-F-YW'),
        'WSPI-F2': WSPIForecast(predictor='aryw', gate='structural', horizon=horizon,
                                clamp=CLAMP, gate_gamma=GATE_GAMMA, name='WSPI-F2'),
        'WSPI-FT': WSPIForecast(predictor='theilsen', gate='none', horizon=horizon,
                                clamp=CLAMP, name='WSPI-FT'),
        'WSPI-FG': WSPIForecast(predictor='aryw', gate='backtest', horizon=horizon,
                                clamp=CLAMP, backtest_k=BACKTEST_K, name='WSPI-FG'),
    }
    if include_arma:
        out['WSPI-FA'] = WSPIForecast(predictor='arma11', gate='none', horizon=horizon,
                                      clamp=CLAMP, name='WSPI-FA')
    return out
