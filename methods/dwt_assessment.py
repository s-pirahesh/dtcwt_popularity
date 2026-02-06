"""
DWT-based Popularity Assessment (Contribution 1)
Uses Discrete Wavelet Transform with AF formula

Author: Sajjad
Date: February 2025
"""
import numpy as np
import pywt
from typing import List, Optional, Dict
from config import WAVELET_CONFIG
from .base_method import BaseMethod


class DWTAssessment(BaseMethod):
    """
    DWT-based popularity assessment method
    
    Innovation: Apply AF formula on wavelet coefficients instead of raw signal
    Claim: Better trend/noise separation than direct AF
    """
    
    def __init__(self, wavelet: str = None, level: int = None, mode: str = 'symmetric'):
        """
        Initialize DWT assessment
        
        Args:
            wavelet: Wavelet family (default: db4)
            level: Decomposition level (default: 3)
            mode: Signal extension mode (default: 'symmetric')
        """
        super().__init__(name="DWT+AF")
        
        self.wavelet = wavelet or WAVELET_CONFIG['dwt_wavelet']
        self.level = level or WAVELET_CONFIG['decomposition_level']
        self.mode = mode
    
    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Assess popularity of a single item
        
        Args:
            time_series: 1D array of access counts
            
        Returns:
            Popularity score
        """
        if len(time_series) == 0:
            return 0.0
        
        # Ensure minimum length for decomposition
        min_length = 2 ** self.level
        if len(time_series) < min_length:
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        try:
            # Perform DWT decomposition
            coeffs = pywt.wavedec(time_series, self.wavelet, level=self.level, mode=self.mode)
            
            # Apply AF formula: Score = sum(2^-i * mean(|coeffs_i|))
            score = 0.0
            for i, coeff in enumerate(coeffs):
                weight = 2 ** (-i)
                level_contribution = weight * np.mean(np.abs(coeff))
                score += level_contribution
            
            return float(score)
            
        except Exception as e:
            print(f"Warning: DWT failed for series length {len(time_series)}: {e}")
            return float(np.mean(time_series))
    
    def assess_batch(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Assess popularity for multiple items
        
        Args:
            time_series_list: List of 1D arrays
            
        Returns:
            Array of popularity scores
        """
        return np.array([self.assess_single(ts) for ts in time_series_list])
    
    def decompose(self, time_series: np.ndarray) -> Dict:
        """
        Perform DWT decomposition and return detailed results
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary with decomposition results
        """
        min_length = 2 ** self.level
        if len(time_series) < min_length:
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        coeffs = pywt.wavedec(time_series, self.wavelet, level=self.level, mode=self.mode)
        
        return {
            'approximation': coeffs[0],
            'details': coeffs[1:],
            'levels': len(coeffs) - 1,
            'wavelet': self.wavelet,
            'mode': self.mode
        }
    
    def get_feature_vector(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from DWT coefficients
        
        Args:
            time_series: Input time series
            
        Returns:
            Feature vector
        """
        decomp = self.decompose(time_series)
        
        features = []
        
        # Approximation coefficient statistics
        approx = decomp['approximation']
        features.extend([
            np.mean(approx),
            np.std(approx),
            np.max(approx),
            np.min(approx),
        ])
        
        # Detail coefficient statistics for each level
        for detail in decomp['details']:
            features.extend([
                np.mean(np.abs(detail)),
                np.std(detail),
                np.max(np.abs(detail)),
            ])
        
        return np.array(features)
    
    def get_metadata(self) -> Dict:
        """Get method metadata"""
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'wavelet': self.wavelet,
            'level': self.level,
            'mode': self.mode,
            'min_signal_length': 2 ** self.level,
        }


def compute_af_formula(coeffs: List[np.ndarray]) -> float:
    """
    Compute Access Frequency formula on wavelet coefficients
    
    Args:
        coeffs: List of coefficient arrays from wavelet decomposition
        
    Returns:
        AF score
    """
    score = 0.0
    for i, coeff in enumerate(coeffs):
        weight = 2 ** (-i)
        level_score = np.mean(np.abs(coeff))
        score += weight * level_score
    
    return score
