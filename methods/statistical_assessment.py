"""
Statistical-based Popularity Assessment (Contribution 3)
Uses higher-order statistics: Skewness + Kurtosis

Author: Sajjad
Date: February 2025
"""
import numpy as np
from scipy import stats
from typing import List, Dict
from config import ASSESSMENT_CONFIG
from .base_method import BaseMethod


class StatisticalAssessment(BaseMethod):
    """
    Statistical popularity assessment using Skewness and Kurtosis
    
    Innovation: Novel formula combining mean, skewness, and kurtosis
    Claim: Higher-order statistics capture trend direction and burstiness
    """
    
    def __init__(self, alpha: float = None, beta: float = None, gamma: float = None):
        """
        Initialize statistical assessment
        
        Args:
            alpha: Weight for mean (default: 1.0)
            beta: Weight for skewness (default: 0.5)
            gamma: Weight for kurtosis (default: 0.3)
        """
        super().__init__(name="Statistical")
        
        self.alpha = alpha or ASSESSMENT_CONFIG['stat_alpha']
        self.beta = beta or ASSESSMENT_CONFIG['stat_beta']
        self.gamma = gamma or ASSESSMENT_CONFIG['stat_gamma']
    
    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Assess popularity using statistical features
        
        Args:
            time_series: 1D array of access counts
            
        Returns:
            Popularity score
        """
        if len(time_series) == 0:
            return 0.0
        
        if len(time_series) < 3:
            # Not enough data for skewness/kurtosis
            return float(np.mean(time_series))
        
        try:
            # Compute statistical features
            mean_val = np.mean(time_series)
            skewness = stats.skew(time_series)
            kurtosis = stats.kurtosis(time_series)
            
            # Handle NaN values
            if np.isnan(skewness):
                skewness = 0.0
            if np.isnan(kurtosis):
                kurtosis = 0.0
            
            # Compute score using weighted formula
            score = (
                self.alpha * mean_val +
                self.beta * skewness +
                self.gamma * kurtosis
            )
            
            return float(score)
        
        except Exception as e:
            print(f"Warning: Statistical calculation failed: {e}")
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
    
    def extract_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Extract all statistical features
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary of statistical features
        """
        if len(time_series) < 3:
            return {
                'mean': float(np.mean(time_series)) if len(time_series) > 0 else 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'min': float(np.min(time_series)) if len(time_series) > 0 else 0.0,
                'max': float(np.max(time_series)) if len(time_series) > 0 else 0.0,
            }
        
        return {
            'mean': float(np.mean(time_series)),
            'std': float(np.std(time_series)),
            'skewness': float(stats.skew(time_series)),
            'kurtosis': float(stats.kurtosis(time_series)),
            'min': float(np.min(time_series)),
            'max': float(np.max(time_series)),
            'median': float(np.median(time_series)),
            'q25': float(np.percentile(time_series, 25)),
            'q75': float(np.percentile(time_series, 75)),
        }
    
    def get_metadata(self) -> Dict:
        """Get method metadata"""
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'formula': 'α*mean + β*skewness + γ*kurtosis',
        }
