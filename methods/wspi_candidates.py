"""
WSPI Candidate Formulas (new methods, NOT replacements)
=======================================================
Two redesigned structural popularity indices proposed after the
sensitivity / ablation analysis showed that, in the original formula
    P = mu_L * exp(clip(alpha*S_L + beta*R - gamma*WE, -3, 3))
the clip operator was inert and the trend-slope term S_L was weak /
slightly harmful, while R and WE carried the real discriminative signal.

Both candidates keep the proven backbone unchanged:
  - the same DTCWT decomposition (J=3, shift-invariant),
  - the same trend-weighted lowpass volume mu_L,
  - the same multiplicative-exponential fusion.

What changes:
  - clip is removed (sensitivity proved it never activates);
  - the broken/weak terms are dropped or replaced;
  - coefficients are named alpha, beta, gamma IN ORDER.

----------------------------------------------------------------------
Candidate 1 — WSPI-2  (minimal / distilled)
    P = mu_L * exp(alpha*R - beta*WE)            alpha=1, beta=1
    Keeps only the two terms the sensitivity analysis proved effective.

Candidate 2 — WSPI-3  (smoothness-corrected, keeps three modulators)
    P = mu_L * exp(alpha*Sm + beta*R - gamma*WE)  alpha=beta=gamma=1
    Replaces the noisy slope S_L with a TREND-SMOOTHNESS term Sm in (0,1]:
        rough = std(diff(level)) / (mean(level) + eps)
        Sm    = 1 / (1 + rough)
    Sm -> 1 for a smooth / flat envelope (max stability);
    Sm -> 0 for a jagged / spiky envelope (min stability).
    This is a TIME-DOMAIN stability signal that complements the
    frequency-domain R and WE. (An earlier efficiency-ratio design was
    discarded: it rewarded net displacement, so it wrongly ranked a
    flat series last and a late spike high.)

Author: Sajjad (with assistance)
"""
from typing import List
import numpy as np

from methods.hybrid_assessment import HybridAssessment


class _CandidateBase(HybridAssessment):
    """Shared feature extraction; subclasses implement _fuse()."""

    def _extract_features(self, time_series: np.ndarray):
        """Return (mu_L, Sm, R, WE) using the SAME pipeline as WSPI."""
        # 1. Padding: 'reflect' to minimise edge artifacts (identical to WSPI)
        min_len = 2 ** (self.level + 1)
        target_len = max(min_len, 2 ** int(np.ceil(np.log2(len(time_series)))))
        if len(time_series) < target_len:
            ts_proc = np.pad(time_series, (target_len - len(time_series), 0),
                             mode='reflect')
        else:
            ts_proc = time_series

        pyramid = self.transform.forward(ts_proc, nlevels=self.level)

        # A) Trend-weighted lowpass volume mu_L  (unchanged)
        lowpass_mags = np.abs(np.asarray(pyramid.lowpass).ravel())
        n = len(lowpass_mags)
        weights = 2.0 ** -np.arange(n)[::-1]
        mu_L = float(np.average(lowpass_mags, weights=weights))

        # Sm) Trend smoothness in (0, 1]: flat/smooth -> 1, jagged -> 0
        diffs = np.diff(lowpass_mags)
        mean_lvl = float(np.mean(lowpass_mags))
        rough = float(np.std(diffs)) / (mean_lvl + 1e-8)
        Sm = 1.0 / (1.0 + rough)

        # C) Energy-concentration ratio R  (unchanged)
        e_low = float(np.sum(lowpass_mags ** 2))
        e_highs = [
            float(np.sum(np.abs(np.asarray(h).ravel()) ** 2))
            for h in pyramid.highpasses
        ]
        e_total = e_low + sum(e_highs)
        R = e_low / e_total if e_total > 0 else 0.0

        # D) Wavelet entropy WE  (unchanged)
        WE = self._calculate_entropy([e_low] + e_highs)

        return mu_L, Sm, R, WE

    def _fuse(self, mu_L, Sm, R, WE) -> float:
        raise NotImplementedError

    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0
        try:
            mu_L, Sm, R, WE = self._extract_features(time_series)
            return float(self._fuse(mu_L, Sm, R, WE))
        except Exception:
            # Same fallback contract as WSPI
            return float(np.mean(time_series))


class WSPI2(_CandidateBase):
    """Candidate 1: P = mu_L * exp(alpha*R - beta*WE).  No slope, no clip."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 name: str = 'WSPI-2'):
        # Map onto parent signature; parent's gamma is unused here.
        super().__init__(alpha_slope=alpha, beta_ratio=beta, gamma_entropy=0.0)
        self.a = alpha
        self.b = beta
        self.name = name

    def _fuse(self, mu_L, Sm, R, WE) -> float:
        return mu_L * np.exp(self.a * R - self.b * WE)


class WSPI3(_CandidateBase):
    """Candidate 2: P = mu_L * exp(alpha*Sm + beta*R - gamma*WE).  No clip."""

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 gamma: float = 1.0, name: str = 'WSPI-3'):
        super().__init__(alpha_slope=alpha, beta_ratio=beta,
                         gamma_entropy=gamma)
        self.a = alpha
        self.b = beta
        self.g = gamma
        self.name = name

    def _fuse(self, mu_L, Sm, R, WE) -> float:
        return mu_L * np.exp(self.a * Sm + self.b * R - self.g * WE)
