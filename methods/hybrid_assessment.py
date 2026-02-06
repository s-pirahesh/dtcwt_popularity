"""
Hybrid Popularity Assessment (Contribution 4)
Combines DTCWT + Statistical + Advanced Features

Author: Sajjad
Date: February 2025
"""
import numpy as np
from typing import List, Dict, Optional
from .dtcwt_assessment import DTCWTAssessment
from .statistical_assessment import StatisticalAssessment
from .base_method import BaseMethod
from config import ASSESSMENT_CONFIG


class HybridAssessment(BaseMethod):
    """
    Hybrid popularity assessment combining multiple methods
    
    Combines:
    - DTCWT features (shift invariance)
    - Statistical features (skewness, kurtosis)
    
    Claim: Optimal trade-off between accuracy, speed, and robustness
    """
    
    def __init__(self,
                 version: str = '3.0',
                 alpha: float = None,
                 beta: float = None,
                 gamma: float = None,
                 noise_penalty: float = None):
        """
        Initialize hybrid assessment
        
        Args:
            version: Version identifier ('3.0' or '3.1')
            alpha: Weight for mean/DTCWT base score
            beta: Weight for skewness
            gamma: Weight for kurtosis
            noise_penalty: Penalty for high-frequency noise
        """
        super().__init__(name=f"Hybrid V{version}")
        
        self.version = version
        
        # Component methods
        self.dtcwt = DTCWTAssessment()
        self.statistical = StatisticalAssessment()
        
        # Weights
        self.alpha = alpha or ASSESSMENT_CONFIG['hybrid_alpha']
        self.beta = beta or ASSESSMENT_CONFIG['hybrid_beta']
        self.gamma = gamma or ASSESSMENT_CONFIG['hybrid_gamma']
        self.noise_penalty = noise_penalty or ASSESSMENT_CONFIG['hybrid_noise_penalty']
    
    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Assess popularity using hybrid method
        
        Args:
            time_series: 1D array of access counts
            
        Returns:
            Hybrid popularity score
        """
        if len(time_series) == 0:
            return 0.0
        
        try:
            # Get DTCWT score
            dtcwt_score = self.dtcwt.assess_single(time_series)
            
            # Get statistical features
            if len(time_series) >= 3:
                from scipy import stats
                mean_val = np.mean(time_series)
                skewness = stats.skew(time_series)
                kurtosis = stats.kurtosis(time_series)
                
                # Handle NaN
                if np.isnan(skewness):
                    skewness = 0.0
                if np.isnan(kurtosis):
                    kurtosis = 0.0
            else:
                mean_val = np.mean(time_series)
                skewness = 0.0
                kurtosis = 0.0
            
            # Compute hybrid score
            score = (
                self.alpha * dtcwt_score +
                self.beta * skewness +
                self.gamma * kurtosis
            )
            
            # Apply noise penalty if needed
            if len(time_series) > 1:
                noise_level = np.std(time_series) / (mean_val + 1e-10)
                score -= self.noise_penalty * noise_level
            
            return float(score)
        
        except Exception as e:
            print(f"Warning: Hybrid assessment failed: {e}")
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
    
    def extract_all_features(self, time_series: np.ndarray) -> Dict:
        """
        Extract all features used in hybrid method
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # DTCWT features
        dtcwt_decomp = self.dtcwt.decompose(time_series)
        features['dtcwt_score'] = self.dtcwt.assess_single(time_series)
        features['dtcwt_levels'] = dtcwt_decomp['levels']
        
        # Statistical features
        stat_features = self.statistical.extract_features(time_series)
        features.update(stat_features)
        
        # Noise level
        if len(time_series) > 1:
            mean_val = np.mean(time_series)
            features['noise_level'] = np.std(time_series) / (mean_val + 1e-10)
        else:
            features['noise_level'] = 0.0
        
        return features
    
    def get_metadata(self) -> Dict:
        """Get method metadata"""
        return {
            'name': self.name,
            'class': self.__class__.__name__,
            'version': self.version,
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'noise_penalty': self.noise_penalty,
            'components': ['DTCWT', 'Statistical'],
        }
