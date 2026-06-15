"""
WSPI — Wavelet Structural Popularity Index (final formula)
==========================================================
Proposed structural popularity index. After an extensive empirical study
(ablation, coefficient sensitivity, several alternative fusion designs, a
variance-domination analysis, and a slope partial-correlation diagnostic on
both datasets) the index was reduced to its minimal, fully-justified form:

        WSPI = mu_L * exp( alpha * R  -  beta * WE )          alpha = beta = 1

Why this form (each choice is backed by evidence, not assertion):
  * mu_L : recency-weighted (2^-k) mean of the DTCWT low-pass envelope —
           the trend-level magnitude; the dominant term.
  * R    : energy-concentration ratio  e_low / e_total  — rewards signals whose
           energy sits in the trend band (smooth / stable). Sensitivity showed
           a clear, monotone contribution.
  * WE   : normalised wavelet entropy across scales — penalises energy spread
           across scales (spiky / chaotic). Also a clear contribution.
  * The trend-slope term was removed: its partial correlation with future
    popularity, controlling for mu_L, is NEGATIVE on both datasets
    (YouTube -0.27, Taxi -0.35), i.e. it is redundant with mu_L and the
    residual is misleading (mean-reverting series).
  * The clip operator was removed: the sweep showed it never activates
    (identical metrics to 6 decimals across c = 1..5).
  * DTCWT (not DWT) is essential: swapping to DWT collapses RSI and inflates
    rank distortion (shift-invariance is the real source of stability).

Ablation flags (use_R / use_WE / use_dtcwt) reproduce every ablation variant
from this single class.

Author: Sajjad (with assistance)
"""
from typing import List
import numpy as np
import dtcwt
import pywt

from config import WAVELET_CONFIG
from .base_method import BaseMethod


class _DWTPyramid:
    """Container mimicking the dtcwt Pyramid interface (for the DWT ablation)."""
    __slots__ = ('lowpass', 'highpasses')


class WSPIAssessment(BaseMethod):
    """
    Final WSPI:  mu_L * exp(alpha*R - beta*WE).

    Parameters
    ----------
    alpha, beta : coefficients on R and WE (defaults 1.0, 1.0).
    use_R, use_WE : drop a term for ablation (sets its contribution to 0).
    use_dtcwt : if False, decompose with DWT (db4) instead of DTCWT — for the
                "WSPI with DWT" ablation. All other steps are identical.
    name : label shown in evaluation logs / result tables.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0,
                 use_R: bool = True, use_WE: bool = True,
                 use_dtcwt: bool = True, name: str = 'WSPI'):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.use_R = use_R
        self.use_WE = use_WE
        self.use_dtcwt = use_dtcwt

        self.biort = WAVELET_CONFIG['dtcwt_biort']
        self.qshift = WAVELET_CONFIG['dtcwt_qshift']
        self.level = WAVELET_CONFIG['decomposition_level']
        self.transform = dtcwt.Transform1d(biort=self.biort, qshift=self.qshift)
        self.dwt_wavelet = WAVELET_CONFIG['dwt_wavelet']

    # ------------------------------------------------------------------
    def _calculate_entropy(self, energies: List[float]) -> float:
        """Normalised Shannon entropy across scale energies (same as before)."""
        total = sum(energies)
        if total == 0:
            return 0.0
        probs = [e / total for e in energies if e > 0]
        if not probs:
            return 0.0
        entropy = -sum(p * np.log2(p) for p in probs)
        max_ent = np.log2(len(energies))
        return entropy / max_ent if max_ent > 0 else 0.0

    def _decompose_dwt(self, signal: np.ndarray) -> _DWTPyramid:
        coeffs = pywt.wavedec(signal, self.dwt_wavelet, level=self.level,
                              mode='symmetric')
        p = _DWTPyramid()
        p.lowpass = coeffs[0]
        p.highpasses = list(reversed(coeffs[1:]))   # level-1 detail first
        return p

    # ------------------------------------------------------------------
    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0
        try:
            # Padding: 'reflect' to a power of two (identical policy to before)
            min_len = 2 ** (self.level + 1)
            target_len = max(min_len, 2 ** int(np.ceil(np.log2(max(len(time_series), 2)))))
            if len(time_series) < target_len:
                ts = np.pad(time_series, (target_len - len(time_series), 0), mode='reflect')
            else:
                ts = time_series

            if self.use_dtcwt:
                pyramid = self.transform.forward(ts, nlevels=self.level)
            else:
                pyramid = self._decompose_dwt(ts)

            lowpass_mags = np.abs(np.asarray(pyramid.lowpass).ravel())

            # mu_L : recency-weighted low-pass mean
            n = len(lowpass_mags)
            weights = 2.0 ** -np.arange(n)[::-1]
            mu_L = float(np.average(lowpass_mags, weights=weights))

            # R : energy concentration in the trend band
            e_low = float(np.sum(lowpass_mags ** 2))
            e_highs = [float(np.sum(np.abs(np.asarray(h).ravel()) ** 2))
                       for h in pyramid.highpasses]
            e_total = e_low + sum(e_highs)
            R = e_low / e_total if e_total > 0 else 0.0

            # WE : normalised wavelet entropy
            WE = self._calculate_entropy([e_low] + e_highs)

            exponent = 0.0
            if self.use_R:
                exponent += self.alpha * R
            if self.use_WE:
                exponent -= self.beta * WE

            return float(mu_L * np.exp(exponent))
        except Exception:
            return float(np.mean(time_series))
