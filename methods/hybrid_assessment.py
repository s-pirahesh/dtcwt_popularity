"""
WSPI: Wavelet Structural Popularity Index (Gold Version)
========================================================
Core contribution: A structural, wavelet-based popularity metric.

Final Refinements (Stability & Robustness):
1. Numerical Stability: Added epsilon to slope normalization to prevent explosion for small means.
2. Range Control: Reduced clamping range to [-3, 3] to prevent extreme score multipliers.
3. Weighting: Exponential weighting (2^-i) strictly enforced for volume calculation.

Formula:
    Score = μ_L * exp( clip( α*Slope_Norm + β*R - γ*WE, -3, 3 ) )
"""

import numpy as np
import dtcwt
from typing import List, Dict
from config import WAVELET_CONFIG
from .base_method import BaseMethod

class HybridAssessment(BaseMethod):
    def __init__(self, alpha_slope: float = 1.0, beta_ratio: float = 0.5, gamma_entropy: float = 0.5):
        super().__init__(name="WSPI (Structural)")
        self.alpha = alpha_slope
        self.beta = beta_ratio
        self.gamma = gamma_entropy
        
        self.biort = WAVELET_CONFIG['dtcwt_biort']
        self.qshift = WAVELET_CONFIG['dtcwt_qshift']
        self.level = WAVELET_CONFIG['decomposition_level']
        self.transform = dtcwt.Transform1d(biort=self.biort, qshift=self.qshift)

    def _calculate_trend_slope(self, coeffs: np.ndarray) -> float:
        """Calculates normalized linear regression slope (Scale-Invariant)."""
        n = len(coeffs)
        if n < 2: return 0.0
        
        y = np.abs(coeffs)
        x = np.arange(n)
        
        # Vectorized Slope Calculation
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_xx = np.sum(x * x)
        
        denom = (n * sum_xx) - (sum_x * sum_x)
        if denom == 0: return 0.0
        
        slope = ((n * sum_xy) - (sum_x * sum_y)) / denom
        
        # Normalize by mean (Relative Growth Rate)
        # Stability Fix: Add epsilon to prevent explosion when mean is near zero
        mean_val = np.mean(y)
        epsilon = 1e-8
        
        # If mean is too small, slope normalization becomes unstable.
        # We cap the effective mean to avoid division by zero.
        effective_mean = mean_val if mean_val > epsilon else epsilon
            
        return slope / effective_mean

    def _calculate_entropy(self, energies: List[float]) -> float:
        """Calculates Normalized Shannon Entropy."""
        total = sum(energies)
        if total == 0: return 0.0
        probs = [e/total for e in energies if e > 0]
        if not probs: return 0.0
        entropy = -sum(p * np.log2(p) for p in probs)
        max_ent = np.log2(len(energies))
        return entropy / max_ent if max_ent > 0 else 0.0

    def assess_single(self, time_series: np.ndarray) -> float:
        if len(time_series) == 0: return 0.0
        
        # 1. Padding: 'reflect' mode to minimize edge artifacts
        min_len = 2 ** (self.level + 1)
        target_len = max(min_len, 2 ** int(np.ceil(np.log2(len(time_series)))))
        if len(time_series) < target_len:
            ts_proc = np.pad(time_series, (target_len - len(time_series), 0), mode='reflect')
        else:
            ts_proc = time_series
            
        try:
            # 2. Transform
            pyramid = self.transform.forward(ts_proc, nlevels=self.level)
            
            # 3. Feature Extraction
            lowpass_mags = np.abs(pyramid.lowpass)
            
            # A) Volume (μ_L) with Exponential Weighting
            # Weights: 2^-(N-i) -> Recent items get higher weight
            n = len(lowpass_mags)
            weights = 2.0 ** -np.arange(n)[::-1] 
            mu_L = np.average(lowpass_mags, weights=weights)
            
            # B) Slope (Persistence) - Normalized & Stabilized
            slope_L = self._calculate_trend_slope(pyramid.lowpass)
            
            # C) Stability Ratio (R)
            e_low = np.sum(lowpass_mags ** 2)
            e_highs = [np.sum(np.abs(h)**2) for h in pyramid.highpasses]
            e_total = e_low + sum(e_highs)
            R = e_low / e_total if e_total > 0 else 0.0
            
            # D) Entropy (WE)
            all_energies = [e_low] + e_highs
            WE = self._calculate_entropy(all_energies)
            
            # 4. Exponential Fusion
            exponent = (self.alpha * slope_L) + (self.beta * R) - (self.gamma * WE)
            
            # Stability Clamp: Reduced range [-3, 3] to avoid extreme multipliers
            # exp(3) ~= 20, exp(-3) ~= 0.05 (Sufficient dynamic range)
            exponent = np.clip(exponent, -3.0, 3.0)
            
            final_score = mu_L * np.exp(exponent)
            
            return float(final_score)
            
        except Exception as e:
            # Fallback to simple mean if transform fails
            return float(np.mean(time_series))

    def assess_batch(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        return np.array([self.assess_single(ts) for ts in time_series_list])

    def get_metadata(self) -> Dict:
        return {
            'name': self.name,
            'formula': 'μ_L * exp(clip(α*Slope + β*R - γ*WE, -3, 3))',
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma
        }