"""
DTCWT-based Popularity Assessment (Contribution 2)
Strategy: Stable Trend + Shock Detection using Dual-Tree Complex Wavelet Transform
"""
import numpy as np
import dtcwt
from typing import List, Dict, Optional
from config import WAVELET_CONFIG
from .base_method import BaseMethod

class DTCWTAssessment(BaseMethod):
    """
    Popularity assessment method based on Dual-Tree Complex Wavelet Transform (DTCWT).

    This method is proposed as the main contribution in this research to overcome
    the limitations of conventional DWT.

    Key advantages over DWT:
    1. Shift Invariance:
       Minor changes in data arrival time (e.g., short time delay) do not cause
       drastic changes in coefficient energy. This property produces "stable"
       popularity scores without flickering.

    2. Richer Information (Magnitude & Phase):
       Using complex numbers ($z = x + iy$) allows us to calculate the "true energy"
       of oscillations using magnitude ($|z|$), without the direction of oscillation
       (positive/negative) having a negative impact.

    Combined Strategy (Stable Trend + Shock):
    This class calculates the score based on the combination of two components:
    1. Stable Trend Component: magnitude of approximation coefficients at final level ($|Lowpass|$).
    2. Precise Shock Component: magnitude of detail coefficients at level 1 ($|Highpass_1|$).

    Final Formula:
        Score = AF(|Trend|) + β * AF(|Shock|)
    """
    
    def __init__(self, biort: str = None, qshift: str = None, level: int = None,
                 detail_weight: float = 0.1):
        """
        Initialize DTCWT transform and strategy parameters.

        Args:
            biort (str): Name of biorthogonal filters for the first stage (default: 'near_sym_a').
            qshift (str): Name of q-shift filters for higher stages (default: 'qshift_a').
            level (int): Decomposition level. Usually 3 or 4.
            detail_weight (float): Beta coefficient (β) that determines the weight of the
                                   impact of instantaneous shocks (Highpass). Default is 0.1.
        """
        super().__init__(name="DTCWT+AF (Stable Trend)")
        
        self.biort = biort or WAVELET_CONFIG['dtcwt_biort']
        self.qshift = qshift or WAVELET_CONFIG['dtcwt_qshift']
        self.level = level or WAVELET_CONFIG['decomposition_level']
        self.detail_weight = detail_weight

        # Create 1D transform object from dtcwt library
        self.transform = dtcwt.Transform1d(biort=self.biort, qshift=self.qshift)
    
    def _apply_weighted_af_magnitude(self, complex_coeffs: np.ndarray) -> float:
        """
        Calculate AF score (Access Frequency) on the magnitude of complex coefficients.

        This method first calculates the magnitude of coefficients ($|z|$) and then
        applies temporal weighting.

        Formula:
            Mag[t] = sqrt(Re[t]^2 + Im[t]^2)
            Score = Σ (Mag[t-i] * 2^-i)

        Args:
            complex_coeffs (np.ndarray): Array of complex coefficients (Complex64 or Complex128).

        Returns:
            float: Weighted energy score.
        """
        if len(complex_coeffs) == 0:
            return 0.0

        # 1. Calculate magnitude for each complex coefficient
        # This removes phase information and keeps only the signal "intensity"
        magnitudes = np.abs(complex_coeffs)

        # 2. Reverse the array (index 0 = most recent time)
        reversed_mags = magnitudes[::-1]

        score = 0.0
        # 3. Apply temporal weighting (Time Decay)
        for i, val in enumerate(reversed_mags):
            weight = 2.0 ** (-i)
            score += weight * val

        return float(score)

    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Calculate final DTCWT popularity score for a time series.

        Execution steps:
        1. Precise Padding: DTCWT requires specific data lengths (power of 2).
        2. Transform: Apply complex wavelet transform up to specified level.
        3. Extraction: Extract Trend (Lowpass) and Shock (Highpass L1) coefficients.
        4. Calculation: Calculate magnitude and apply AF on each component.
        5. Fusion: Combine scores with beta coefficient.

        Args:
            time_series (np.ndarray): Time series of visits.

        Returns:
            float: Popularity score.
        """
        if len(time_series) == 0:
            return 0.0

        # 1. Data length management (Strict Padding for DTCWT)
        # dtcwt library requires input length to be a power of 2 for optimal performance.
        # Also, for decomposition up to level L, length must be at least 2^(L+1).
        min_len = 2 ** (self.level + 1)
        curr_len = len(time_series)

        # Find the nearest power of 2 that is >= current length
        target_len = max(min_len, 2 ** int(np.ceil(np.log2(curr_len))))

        if curr_len < target_len:
            # Padding is done by repeating edge values on the left (past).
            # This ensures actual data stays on the right (present time) without index confusion.
            pad_width = target_len - curr_len
            ts_to_process = np.pad(time_series, (pad_width, 0), mode='edge')
        else:
            ts_to_process = time_series

        try:
            # 2. DTCWT Transform (Forward Transform)
            # Output pyramid contains:
            # - pyramid.lowpass: array of final approximation coefficients
            # - pyramid.highpasses: list of detail coefficient tuples (from level 1 to Level)
            pyramid = self.transform.forward(ts_to_process, nlevels=self.level)

            # 3. Extract Trend score (Lowpass)
            # These coefficients represent the main and smoothed trend.
            trend_score = 0.0
            if pyramid.lowpass is not None:
                trend_score = self._apply_weighted_af_magnitude(pyramid.lowpass)

            # 4. Extract Shock score (Highpass Level 1)
            # pyramid.highpasses[0] corresponds to level 1 (high frequency).
            # This level is most sensitive to sudden changes (spikes).
            shock_score = 0.0
            if pyramid.highpasses and len(pyramid.highpasses) > 0:
                # Note: In DTCWT 1D, highpass coefficients are an array (not tuple like 2D)
                shock_coeffs = pyramid.highpasses[0]
                shock_score = self._apply_weighted_af_magnitude(shock_coeffs)

            # 5. Final fusion
            # Shock score with small coefficient (e.g., 0.1) is added to only affect critical moments.
            final_score = trend_score + (self.detail_weight * shock_score)

            return float(final_score)

        except Exception as e:
            # Fallback mechanism if complex calculations fail
            print(f"Warning: DTCWT failed, fallback to raw sum. Error: {e}")
            return float(np.sum(time_series))

    def assess_batch(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Evaluate a batch of time series.
        """
        return np.array([self.assess_single(ts) for ts in time_series_list])

    def get_metadata(self) -> Dict:
        """
        Get method metadata information for logging.
        """
        return {
            'name': self.name,
            'biort': self.biort,
            'qshift': self.qshift,
            'level': self.level,
            'detail_weight': self.detail_weight,
            'strategy': 'Stable Trend (|Lowpass|) + Weighted Shock (|Highpass_1|)'
        }