"""
DWT-based Popularity Assessment (Contribution 1)
Uses Discrete Wavelet Transform with AF formula
"""
import numpy as np
import pywt
from typing import List, Optional
from config import WAVELET_CONFIG


class DWTAssessment:
    """
    DWT-based popularity assessment method
    
    Innovation: Apply AF formula on wavelet coefficients instead of raw signal
    Claim: Better trend/noise separation than direct AF
    """
    
    def __init__(self, wavelet: str = None, level: int = None):
        """
        Initialize DWT assessment
        
        Args:
            wavelet: Wavelet family (default: db4)
            level: Decomposition level (default: 3)
        """
        self.wavelet = wavelet or WAVELET_CONFIG['dwt_wavelet']
        self.level = level or WAVELET_CONFIG['decomposition_level']
    
    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Assess popularity of a single item
        
        Args:
            time_series: 1D array of access counts
            
        Returns:
            Popularity score
        """
        # Handle edge cases
        if len(time_series) == 0:
            return 0.0
        
        # Ensure minimum length for decomposition
        min_length = 2 ** self.level
        if len(time_series) < min_length:
            # Pad with zeros if too short
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        try:
            # Perform DWT decomposition
            coeffs = pywt.wavedec(time_series, self.wavelet, level=self.level)
            
            # Apply AF formula: Score = sum(2^-i * mean(A_i))
            # where A_i are approximation coefficients at level i
            score = 0.0
            
            for i, coeff in enumerate(coeffs):
                # Weight decreases with level (2^-i)
                weight = 2 ** (-i)
                # Mean of coefficients at this level
                level_score = np.mean(np.abs(coeff))
                score += weight * level_score
            
            return float(score)
        
        except Exception as e:
            # Fallback to simple mean if decomposition fails
            return float(np.mean(time_series))
    
    def batch_assess(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Assess popularity for multiple items
        
        Args:
            time_series_list: List of time series
            
        Returns:
            Array of popularity scores
        """
        scores = np.array([self.assess_single(ts) for ts in time_series_list])
        return scores
    
    def decompose(self, time_series: np.ndarray) -> dict:
        """
        Get detailed decomposition (for analysis/visualization)
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary with coefficients at each level
        """
        # Ensure minimum length
        min_length = 2 ** self.level
        if len(time_series) < min_length:
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        coeffs = pywt.wavedec(time_series, self.wavelet, level=self.level)
        
        result = {
            'approximation': coeffs[0],
            'details': coeffs[1:],
            'levels': len(coeffs),
            'wavelet': self.wavelet
        }
        
        return result
    
    def get_feature_vector(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from DWT coefficients
        Useful for ML-based prediction
        
        Args:
            time_series: Input time series
            
        Returns:
            Feature vector (flattened coefficients + statistics)
        """
        decomp = self.decompose(time_series)
        
        features = []
        
        # Add approximation coefficient statistics
        approx = decomp['approximation']
        features.extend([
            np.mean(approx),
            np.std(approx),
            np.max(approx),
            np.min(approx),
        ])
        
        # Add detail coefficient statistics for each level
        for detail in decomp['details']:
            features.extend([
                np.mean(np.abs(detail)),
                np.std(detail),
                np.max(np.abs(detail)),
            ])
        
        return np.array(features)


def compute_af_formula(coeffs: List[np.ndarray]) -> float:
    """
    Compute Access Frequency formula on wavelet coefficients
    
    Formula: AF = sum(2^-i * mean(|coeffs_i|))
    
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
