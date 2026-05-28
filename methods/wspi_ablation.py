"""
WSPI Ablation Variants
======================
Subclass of HybridAssessment (WSPI) that allows disabling individual
components or swapping DTCWT for DWT.

Six variants for the ablation study (Reviewer Comment #5):
  - Full WSPI                 (baseline)
  - WSPI - WE                 (gamma = 0)
  - WSPI - R                  (beta = 0)
  - WSPI - S_L                (alpha = 0)
  - WSPI with DWT             (DTCWT -> DWT)
  - WSPI no clip              (c -> infinity)

Author: Sajjad (with assistance)
"""
import numpy as np
import pywt
from typing import Dict
from config import WAVELET_CONFIG
from methods.hybrid_assessment import HybridAssessment


class _DWTPyramid:
    """Lightweight container that mimics dtcwt Pyramid attributes."""
    __slots__ = ('lowpass', 'highpasses')


class WSPIAblation(HybridAssessment):
    """
    Ablation-enabled variant of WSPI.

    Extra parameters
    ----------------
    use_dtcwt : bool
        If False, the wavelet decomposition uses DWT (db4) instead of DTCWT.
        Lowpass / highpass coefficients are then real-valued; the rest of the
        feature extraction pipeline is identical.
    use_clip : bool
        If False, the exponent (alpha*S_L + beta*R - gamma*WE) is NOT clipped
        before exponentiation.  Useful for testing the contribution of the
        clamp operator to robustness (Section 4-F of the paper).
    clip_c : float
        Symmetric clip bound, applied only when use_clip=True.  Default 3.0.
    variant_name : str
        Human-readable name shown in evaluation logs and result tables.
    """

    def __init__(self,
                 alpha_slope:   float = 1.0,
                 beta_ratio:    float = 0.5,
                 gamma_entropy: float = 0.5,
                 use_dtcwt:     bool  = True,
                 use_clip:      bool  = True,
                 clip_c:        float = 3.0,
                 variant_name:  str   = 'WSPI'):
        super().__init__(alpha_slope=alpha_slope,
                         beta_ratio=beta_ratio,
                         gamma_entropy=gamma_entropy)
        self.use_dtcwt = use_dtcwt
        self.use_clip  = use_clip
        self.clip_c    = clip_c
        # Override BaseMethod.name so evaluator logs the variant
        self.name      = variant_name
        # DWT wavelet (only used when use_dtcwt=False)
        self.dwt_wavelet = WAVELET_CONFIG['dwt_wavelet']

    # ---------------------------------------------------------------------
    # Decomposition backends
    # ---------------------------------------------------------------------
    def _decompose_dtcwt(self, signal: np.ndarray):
        """Forward DTCWT — returns a dtcwt Pyramid (parent uses self.transform)."""
        return self.transform.forward(signal, nlevels=self.level)

    def _decompose_dwt(self, signal: np.ndarray) -> _DWTPyramid:
        """
        DWT decomposition that returns a Pyramid-like object so the rest of
        assess_single() can stay backend-agnostic.

        pywt.wavedec returns [cA_L, cD_L, ..., cD_1]; the parent code expects:
            pyramid.lowpass    -> cA_L
            pyramid.highpasses -> [cD_1, cD_2, ..., cD_L]  (highpass[0] = level 1)
        """
        coeffs = pywt.wavedec(signal, self.dwt_wavelet,
                              level=self.level, mode='symmetric')
        pyramid = _DWTPyramid()
        pyramid.lowpass    = coeffs[0]                    # cA_L
        pyramid.highpasses = list(reversed(coeffs[1:]))   # cD_1 first
        return pyramid

    # ---------------------------------------------------------------------
    # Main entry point — replaces HybridAssessment.assess_single
    # ---------------------------------------------------------------------
    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0:
            return 0.0

        # 1. Padding (same policy as parent WSPI)
        min_len    = 2 ** (self.level + 1)
        target_len = max(min_len,
                         2 ** int(np.ceil(np.log2(max(len(time_series), 2)))))
        if len(time_series) < target_len:
            ts_proc = np.pad(time_series,
                             (target_len - len(time_series), 0),
                             mode='reflect')
        else:
            ts_proc = time_series

        try:
            # 2. Decompose with the chosen backend
            pyramid = (self._decompose_dtcwt(ts_proc) if self.use_dtcwt
                       else self._decompose_dwt(ts_proc))

            # 3. Feature extraction (identical to parent WSPI)
            # FIX: newer dtcwt returns shape (n, 1); flatten to 1-D.
            lowpass_mags = np.abs(np.asarray(pyramid.lowpass).ravel())

            # μ_L  — weighted volume
            n = len(lowpass_mags)
            weights = 2.0 ** -np.arange(n)[::-1]
            mu_L = np.average(lowpass_mags, weights=weights)

            # S_L  — normalised slope
            slope_L = self._calculate_trend_slope(
                np.asarray(pyramid.lowpass).ravel()
            )

            # R    — energy concentration ratio
            e_low   = float(np.sum(lowpass_mags ** 2))
            e_highs = [
                float(np.sum(np.abs(np.asarray(h).ravel()) ** 2))
                for h in pyramid.highpasses
            ]
            e_total = e_low + sum(e_highs)
            R       = e_low / e_total if e_total > 0 else 0.0

            # WE   — wavelet entropy
            WE = self._calculate_entropy([e_low] + e_highs)

            # 4. Exponential fusion (with optional clip)
            exponent = (self.alpha * slope_L
                        + self.beta  * R
                        - self.gamma * WE)
            if self.use_clip:
                exponent = float(np.clip(exponent, -self.clip_c, +self.clip_c))

            return float(mu_L * np.exp(exponent))

        except Exception:
            return float(np.mean(time_series))

    def get_metadata(self) -> Dict:
        return {
            'name':       self.name,
            'alpha':      self.alpha,
            'beta':       self.beta,
            'gamma':      self.gamma,
            'use_dtcwt':  self.use_dtcwt,
            'use_clip':   self.use_clip,
            'clip_c':     self.clip_c,
        }
