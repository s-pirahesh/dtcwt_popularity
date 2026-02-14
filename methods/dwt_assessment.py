"""
DWT-based Popularity Assessment (Contribution 1)
Strategy: Trend + Viral Shock Detection
Combines Approximation (Trend) with Level 1 Details (Viral Spikes).
"""
import numpy as np
import pywt
from typing import List, Dict, Optional
from config import WAVELET_CONFIG
from .base_method import BaseMethod

class DWTAssessment(BaseMethod):
    """
    Popularity assessment method based on DWT with a combined approach (Trend + Shock).

    This class calculates and combines popularity based on two separate components:

    1. Trend Component:
       - Extracted from approximation coefficients at the lowest decomposition level ($cA_L$).
       - Represents stable, long-term popularity and overall user behavior.
       - High-frequency noise in this component is removed.

    2. Shock Component:
       - Extracted from detail coefficients at level 1 ($cD_1$).
       - Level 1 wavelet is most sensitive to rapid changes.
       - Represents sudden changes, trending news, and viral content.

    Final score formula:
        Score = AF(Trend) + β * AF(Shock)

    Where AF is a time-weighted access frequency function.
    """

    def __init__(self, wavelet: str = None, level: int = None,
                 detail_weight: float = 0.1, mode: str = 'symmetric'):
        """
        Initialize the DWT assessment class.

        Args:
            wavelet (str): Name of the mother wavelet (e.g., 'db4', 'sym4').
                           If None, it is read from config.py.
            level (int): Decomposition level.
                         Level 3 is usually suitable for daily/weekly data.
            detail_weight (float): Beta coefficient (β) that determines the weight of
                                   sudden shock impacts. Default is 0.1 (10% impact).
            mode (str): Padding method for signal edges (default: 'symmetric').
        """
        super().__init__(name="DWT+AF (Trend+Shock)")
        self.wavelet = wavelet or WAVELET_CONFIG['dwt_wavelet']
        self.level = level or WAVELET_CONFIG['decomposition_level']
        self.detail_weight = detail_weight
        self.mode = mode
    
    def _apply_weighted_af(self, coeffs_array: np.ndarray) -> float:
        """
        Apply Access Frequency (AF) logic with temporal weighting on wavelet coefficients.

        This method calculates signal energy with higher weight for newer coefficients.

        Formula:
            Score = Σ (|Coeff[t-i]| * 2^-i)

        Args:
            coeffs_array (np.ndarray): Array of wavelet coefficients (from old to new).

        Returns:
            float: Calculated score (weighted energy).
        """
        if len(coeffs_array) == 0:
            return 0.0

        # Reverse the array so index 0 represents "present time" (t)
        reversed_coeffs = coeffs_array[::-1]
        score = 0.0

        for i, val in enumerate(reversed_coeffs):
            # Calculate energy (absolute value) with exponential weight decay in the past
            weight = 2.0 ** (-i)
            score += weight * abs(val)

        return float(score)

    def _safe_level(self, n: int) -> int:
        """
        Return the maximum decomposition level that avoids boundary-effect
        warnings from pywt for a signal of length n.

        pywt fires UserWarning when level > floor(log2(n / (filter_len - 1))).
        For db4: filter_len = 8, so safe threshold is n >= 7 * 2^level.

        If the requested self.level is already safe, it is returned unchanged.
        Otherwise the largest safe level (>= 1) is returned.
        """
        wavelet_obj  = pywt.Wavelet(self.wavelet)
        filter_len   = wavelet_obj.dec_len          # e.g. 8 for db4
        import math
        if filter_len <= 1 or n <= 0:
            return 1
        max_safe = int(math.floor(math.log2(n / (filter_len - 1)))) if n >= filter_len else 1
        return max(1, min(self.level, max_safe))

    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Calculate the final popularity score for a time series.

        Execution steps:
        1. Adaptive level: choose the highest DWT level that is safe for
           this signal length (avoids pywt boundary-effect warnings).
        2. Padding: ensure minimum length of 2^level.
        3. Decomposition: [cA_L, cD_L, …, cD_1].
        4. Extraction: trend (cA_L) and shock (cD_1).
        5. Scoring: weighted AF on each component.
        6. Fusion: Score = AF(trend) + β * AF(shock).

        Args:
            time_series (np.ndarray): Visit counts per time-slot.

        Returns:
            float: Final popularity score.
        """
        if len(time_series) == 0:
            return 0.0

        # 1. Adaptive level — never request more than the signal can support
        safe_lvl = self._safe_level(len(time_series))

        # 2. Minimum-length padding (now using safe_lvl, not self.level)
        min_len = 2 ** safe_lvl
        if len(time_series) < min_len:
            pad_width = min_len - len(time_series)
            ts_to_process = np.pad(time_series, (pad_width, 0), mode='edge')
        else:
            ts_to_process = time_series

        try:
            # 3. Wavelet decomposition at the (possibly reduced) safe level
            coeffs = pywt.wavedec(ts_to_process, self.wavelet,
                                  level=safe_lvl, mode=self.mode)

            # 4. Extract components
            approx_coeffs = coeffs[0]   # trend  (cA_L)
            detail_l1     = coeffs[-1]  # shock  (cD_1)

            # 5. Calculate scores
            trend_score = self._apply_weighted_af(approx_coeffs)
            shock_score = self._apply_weighted_af(detail_l1)

            # 6. Final fusion
            return float(trend_score + self.detail_weight * shock_score)

        except Exception as e:
            # Fallback: weighted sum (no print — caller handles logging)
            return float(np.sum(time_series))

    def assess_batch(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Evaluate a batch of time series.

        Args:
            time_series_list: List of time series arrays.

        Returns:
            np.ndarray: Array of calculated scores.
        """
        return np.array([self.assess_single(ts) for ts in time_series_list])

    def get_metadata(self) -> Dict:
        """
        Get method metadata information for saving in log files.

        Returns:
            Dict: Dictionary containing set parameters (wavelet name, level, beta coefficient).
        """
        return {
            'name': self.name,
            'wavelet': self.wavelet,
            'level': self.level,
            'detail_weight': self.detail_weight,
            'strategy': 'Trend (cA_L) + Weighted Shock (cD_1)'
        }