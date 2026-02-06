"""
DTCWT-based Popularity Assessment (Contribution 2)
Main Innovation: Dual-Tree Complex Wavelet Transform for popularity measurement

Author: Sajjad
Date: February 2025
"""
import numpy as np
import dtcwt
from typing import List, Dict, Optional
from config import WAVELET_CONFIG
from .base_method import BaseMethod


class DTCWTAssessment(BaseMethod):
    """
    DTCWT-based popularity assessment method
    
    Innovation: Replace DWT with DTCWT for better shift invariance
    Claim: 10-15% improvement over DWT due to stability and directional selectivity
    """
    
    def __init__(self, biort: str = None, qshift: str = None, level: int = None):
        """
        Initialize DTCWT assessment
        
        Args:
            biort: Biorthogonal filters for level 1 (default: near_sym_a)
            qshift: Q-shift filters for levels >= 2 (default: qshift_a)
            level: Decomposition level (default: 3)
        """
        super().__init__(name="DTCWT+AF")
        
        self.biort = biort or WAVELET_CONFIG['dtcwt_biort']
        self.qshift = qshift or WAVELET_CONFIG['dtcwt_qshift']
        self.level = level or WAVELET_CONFIG['decomposition_level']
        
        # Initialize DTCWT transform
        self.transform = dtcwt.Transform1d(biort=self.biort, qshift=self.qshift)
    
    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Assess popularity using DTCWT coefficients
        
        Args:
            time_series: 1D array of access counts
            
        Returns:
            Popularity score
        """
        if len(time_series) == 0:
            return 0.0
        
        # DTCWT requires minimum length
        min_length = 2 ** (self.level + 1)
        if len(time_series) < min_length:
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        try:
            # Perform DTCWT forward transform
            pyramid = self.transform.forward(time_series, nlevels=self.level)
            
            score = 0.0
            
            # Score from approximation coefficients (lowpass)
            approx = pyramid.lowpass
            weight_approx = 2 ** 0
            score += weight_approx * np.mean(np.abs(approx))
            
            # Score from detail coefficients (highpass)
            for i, detail_level in enumerate(pyramid.highpass):
                weight = 2 ** (-(i + 1))
                magnitude = np.abs(detail_level)
                level_score = np.mean(magnitude)
                score += weight * level_score
            
            return float(score)
        
        except Exception as e:
            print(f"Warning: DTCWT failed: {e}")
            return float(np.mean(time_series))
    
    def assess_batch(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Assess popularity for multiple items
        
        Args:
            time_series_list: List of time series
            
        Returns:
            Array of popularity scores
        """
        return np.array([self.assess_single(ts) for ts in time_series_list])
    
    def decompose(self, time_series: np.ndarray) -> Dict:
        """
        Perform DTCWT decomposition
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary with decomposition results
        """
        min_length = 2 ** (self.level + 1)
        if len(time_series) < min_length:
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        pyramid = self.transform.forward(time_series, nlevels=self.level)
        
        return {
            'lowpass': pyramid.lowpass,
            'highpass': pyramid.highpass,
            'levels': len(pyramid.highpass),
            'biort': self.biort,
            'qshift': self.qshift,
        }
    
    def get_metadata(self) -> Dict:
        """Get method metadata"""
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'biort': self.biort,
            'qshift': self.qshift,
            'level': self.level,
            'min_signal_length': 2 ** (self.level + 1),
        }
    
    def compare_shift_invariance(self, time_series: np.ndarray) -> Dict:
        """
        Compare shift-invariance with DWT
        
        Args:
            time_series: Input signal
            
        Returns:
            Comparison statistics
        """
        from .dwt_assessment import DWTAssessment
        
        # Get DTCWT score
        dtcwt_score = self.assess_single(time_series)
        
        # Get DWT score
        dwt = DWTAssessment()
        dwt_score = dwt.assess_single(time_series)
        
        # Analyze shift invariance
        shifted = np.roll(time_series, 1)
        dtcwt_shifted = self.assess_single(shifted)
        dwt_shifted = dwt.assess_single(shifted)
        
        # Compute stability
        dtcwt_stability = abs(dtcwt_score - dtcwt_shifted) / (dtcwt_score + 1e-10)
        dwt_stability = abs(dwt_score - dwt_shifted) / (dwt_score + 1e-10)
        
        return {
            'dtcwt_score': dtcwt_score,
            'dwt_score': dwt_score,
            'dtcwt_stability': dtcwt_stability,
            'dwt_stability': dwt_stability,
            'improvement': (dtcwt_score - dwt_score) / (dwt_score + 1e-10),
        }


def compute_dtcwt_af_formula(lowpass: np.ndarray, highpass: List[np.ndarray]) -> float:
    """
    Compute AF formula adapted for DTCWT coefficients
    
    Args:
        lowpass: Approximation coefficients
        highpass: List of detail coefficients (complex)
        
    Returns:
        DTCWT-AF score
    """
    score = 0.0
    
    # Approximation contribution
    score += np.mean(np.abs(lowpass))
    
    # Detail contributions (weighted by level)
    for i, detail in enumerate(highpass):
        weight = 2 ** (-(i + 1))
        magnitude = np.abs(detail)
        score += weight * np.mean(magnitude)
    
    return score
