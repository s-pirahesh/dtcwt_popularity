"""
WSPI — Wavelet-Structured Popularity Index (index of the paper)
================================================================
        WSPI = mu_L * exp( alpha * R  -  beta * W_E )        alpha = beta = 1

  * mu_L : recency-weighted (2^-k) mean of the magnitudes of the DTCWT
           lowpass (trend) coefficients; the dominant term.
  * R    : share of the coefficient energy in the lowpass band.
  * W_E  : normalised wavelet entropy of the band energies.
Defaults: J = 3, filters near_sym_a / qshift_a.  An earlier version had a
slope term and a clip; both were removed (see the paper).

This is the per-item class.  The runs of the paper use the batched version
in ``evaluation/fast_evaluator.py`` through ``evaluation/protocol_v5.py``
(``make_v5_wspi``); the unit tests check that both give the same score.
Ablation flags (use_R / use_WE / use_dtcwt) give the ablation variants.

Author: Sajjad Pirahesh
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
