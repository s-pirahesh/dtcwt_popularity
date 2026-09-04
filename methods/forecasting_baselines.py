# -*- coding: utf-8 -*-
"""
Explicit Value-Forecasting Baselines
====================================
Prediction-based popularity methods for the defense-gap experiment:
each method explicitly FORECASTS the next-horizon demand and uses the
sum of forecasted values as the popularity score. The evaluation
protocol (future ground truth) then measures true forecasting quality
with the exact same 4-layer metrics as all other methods.

Methods:
  - PersistenceForecast : naive — next horizon repeats the last observation
  - HoltForecast        : Holt double exponential smoothing (level + trend)
  - ARYWForecast        : AR(p) fitted by Yule-Walker (fast, closed-form)
  - ARIMAForecast       : ARMA(p,q) via statsmodels MLE, sliding window
                          (the Hassine et al. 2017 [25] approach; SLOW)

All classes only need the standard BaseMethod interface:
    assess_single(time_series) -> float
so they plug into the existing pipeline with zero evaluator changes.

Author: Sajjad (with Claude)
Date: August 2026
"""
import warnings
import numpy as np

from .base_method import BaseMethod


def _safe_sum(values, fallback: float) -> float:
    """Sum forecast values; clip negatives to 0; fall back on non-finite."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).all():
        return float(fallback)
    return float(np.sum(np.clip(arr, 0.0, None)))


class PersistenceForecast(BaseMethod):
    """
    Naive persistence: y_hat(t+1..t+h) = y(t).
    The mandatory baseline of every forecasting paper.
    Score = h * last observation.
    """

    def __init__(self, horizon: int = 7):
        super().__init__(name='Persistence')
        self.horizon = int(horizon)

    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0
        last = float(time_series[-1])
        return _safe_sum([last] * self.horizon, fallback=last)


class HoltForecast(BaseMethod):
    """
    Holt's linear (double exponential) smoothing — closed form, no deps.
        l_t = a*y_t + (1-a)*(l_{t-1} + b_{t-1})
        b_t = g*(l_t - l_{t-1}) + (1-g)*b_{t-1}
        y_hat(t+k) = l_t + k*b_t
    Score = sum_{k=1..h} max(0, l + k*b).
    """

    def __init__(self, alpha: float = 0.3, gamma: float = 0.1, horizon: int = 7):
        super().__init__(name='Holt')
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.horizon = int(horizon)

    def assess_single(self, time_series: np.ndarray) -> float:
        ts = np.asarray(time_series, dtype=float)
        n = len(ts)
        if n == 0:
            return 0.0
        if n == 1:
            return _safe_sum([ts[0]] * self.horizon, fallback=ts[0])

        level = ts[0]
        trend = ts[1] - ts[0]
        for t in range(1, n):
            prev_level = level
            level = self.alpha * ts[t] + (1 - self.alpha) * (level + trend)
            trend = self.gamma * (level - prev_level) + (1 - self.gamma) * trend

        forecasts = [level + k * trend for k in range(1, self.horizon + 1)]
        return _safe_sum(forecasts, fallback=ts[-1] * self.horizon)


class ARYWForecast(BaseMethod):
    """
    AR(p) fitted by Yule-Walker equations (closed-form, fast — no MLE
    iterations). Demeans the window, fits AR coefficients from sample
    autocovariances, forecasts recursively h steps ahead.
    Score = sum of the h forecasts.
    """

    def __init__(self, order: int = 2, horizon: int = 7):
        super().__init__(name='ARYW')
        self.order = int(order)
        self.horizon = int(horizon)

    def _yule_walker(self, x: np.ndarray) -> np.ndarray:
        """Return AR coefficients phi_1..phi_p via Yule-Walker."""
        p = self.order
        n = len(x)
        # autocovariances r_0..r_p
        r = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(p + 1)])
        if r[0] <= 1e-12:
            return np.zeros(p)          # constant / all-zero window
        R = np.array([[r[abs(i - j)] for j in range(p)] for i in range(p)])
        try:
            phi = np.linalg.solve(R + 1e-8 * np.eye(p), r[1:p + 1])
        except np.linalg.LinAlgError:
            return np.zeros(p)
        # crude stability guard
        if not np.isfinite(phi).all() or np.abs(phi).sum() > 4:
            return np.zeros(p)
        return phi

    def assess_single(self, time_series: np.ndarray) -> float:
        ts = np.asarray(time_series, dtype=float)
        n = len(ts)
        naive = float(ts[-1]) * self.horizon if n else 0.0
        if n < max(self.order + 2, 4):
            return _safe_sum([], fallback=naive) if n == 0 else naive

        mu = ts.mean()
        x = ts - mu
        phi = self._yule_walker(x)

        history = list(x[-self.order:])
        forecasts = []
        for _ in range(self.horizon):
            nxt = float(np.dot(phi, history[::-1]))
            forecasts.append(nxt + mu)
            history.append(nxt)
            history = history[-self.order:]
        return _safe_sum(forecasts, fallback=naive)


class ARIMAForecast(BaseMethod):
    """
    Sliding-window ARMA(p,q) via statsmodels MLE — the approach of
    Hassine et al. (2017) [25] cited in the proposal.

    WARNING: one MLE fit per (item, window). This is orders of magnitude
    slower than every other method in the lineup. Run it selectively
    (e.g. youtube first) and measure its wall-clock — the runtime itself
    is a thesis result (Table 4-8 evidence).
    """

    def __init__(self, p: int = 1, q: int = 1, horizon: int = 7):
        super().__init__(name='ARIMA')
        self.p = int(p)
        self.q = int(q)
        self.horizon = int(horizon)
        self._sm_ok = None   # lazy statsmodels availability check

    def _statsmodels_available(self) -> bool:
        if self._sm_ok is None:
            try:
                from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
                self._sm_ok = True
            except ImportError:
                warnings.warn(
                    "statsmodels not installed — ARIMAForecast falls back to "
                    "Yule-Walker AR. Install with: pip install statsmodels",
                    ImportWarning)
                self._sm_ok = False
        return self._sm_ok

    def assess_single(self, time_series: np.ndarray) -> float:
        ts = np.asarray(time_series, dtype=float)
        n = len(ts)
        naive = float(ts[-1]) * self.horizon if n else 0.0
        if n < 8:
            return naive
        # degenerate (constant / all-zero) windows: MLE is pointless
        if np.allclose(ts, ts[0]):
            return naive

        if not self._statsmodels_available():
            return ARYWForecast(order=max(self.p, 1),
                                horizon=self.horizon).assess_single(ts)

        from statsmodels.tsa.arima.model import ARIMA
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = ARIMA(ts, order=(self.p, 0, self.q),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                fitted = model.fit(method_kwargs={'maxiter': 50})
                forecasts = fitted.forecast(steps=self.horizon)
            return _safe_sum(forecasts, fallback=naive)
        except Exception:
            # non-convergence, singular matrices, etc. — never crash the run
            return naive


def get_all_forecasting_methods(horizon: int = 7) -> dict:
    """Return {name: instance} for all explicit forecasting baselines."""
    return {
        'Persistence': PersistenceForecast(horizon=horizon),
        'Holt':        HoltForecast(horizon=horizon),
        'ARYW':        ARYWForecast(order=2, horizon=horizon),
        'ARIMA':       ARIMAForecast(p=1, q=1, horizon=horizon),
    }
