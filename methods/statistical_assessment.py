"""
Statistical-based Popularity Assessment (Contribution 3)
Uses higher-order statistics: Skewness + Kurtosis
"""
import numpy as np
from scipy import stats
from typing import List, Dict
from config import ASSESSMENT_CONFIG


class StatisticalAssessment:
    """
    Statistical popularity assessment using Skewness and Kurtosis
    
    Innovation: Novel formula combining mean, skewness, and kurtosis
    Claim: Higher-order statistics capture trend direction and burstiness
    
    Formula: Score = α*mean + β*skewness + γ*kurtosis
    - Skewness: Captures asymmetry (trend direction)
    - Kurtosis: Captures burstiness (outliers/spikes)
    """
    
    def __init__(self, alpha: float = None, beta: float = None, gamma: float = None):
        """
        Initialize statistical assessment
        
        Args:
            alpha: Weight for mean (default: 1.0)
            beta: Weight for skewness (default: 0.5)
            gamma: Weight for kurtosis (default: 0.3)
        """
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
        # Handle edge cases
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
            # Fallback to mean
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
    
    def extract_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Extract all statistical features
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary of statistical features
        """
        if len(time_series) == 0:
            return {
                'mean': 0.0, 'std': 0.0, 'var': 0.0,
                'skewness': 0.0, 'kurtosis': 0.0,
                'min': 0.0, 'max': 0.0, 'median': 0.0
            }
        
        features = {
            # Basic statistics
            'mean': np.mean(time_series),
            'std': np.std(time_series),
            'var': np.var(time_series),
            'min': np.min(time_series),
            'max': np.max(time_series),
            'median': np.median(time_series),
            
            # Higher-order statistics
            'skewness': stats.skew(time_series) if len(time_series) >= 3 else 0.0,
            'kurtosis': stats.kurtosis(time_series) if len(time_series) >= 3 else 0.0,
            
            # Percentiles
            'q25': np.percentile(time_series, 25),
            'q75': np.percentile(time_series, 75),
            'iqr': np.percentile(time_series, 75) - np.percentile(time_series, 25),
            
            # Coefficient of variation
            'cv': np.std(time_series) / (np.mean(time_series) + 1e-10),
        }
        
        return features
    
    def get_feature_vector(self, time_series: np.ndarray) -> np.ndarray:
        """
        Get feature vector for ML
        
        Args:
            time_series: Input time series
            
        Returns:
            Feature vector
        """
        features = self.extract_features(time_series)
        
        # Convert to ordered array
        feature_vector = np.array([
            features['mean'],
            features['std'],
            features['skewness'],
            features['kurtosis'],
            features['min'],
            features['max'],
            features['median'],
            features['cv'],
        ])
        
        return feature_vector
    
    def analyze_trend_direction(self, time_series: np.ndarray) -> str:
        """
        Analyze trend direction using skewness
        
        Args:
            time_series: Input time series
            
        Returns:
            Trend direction: 'increasing', 'decreasing', or 'stable'
        """
        if len(time_series) < 3:
            return 'stable'
        
        skewness = stats.skew(time_series)
        
        if skewness > 0.5:
            return 'increasing'  # Right-skewed: more small values, trend up
        elif skewness < -0.5:
            return 'decreasing'  # Left-skewed: more large values, trend down
        else:
            return 'stable'
    
    def analyze_burstiness(self, time_series: np.ndarray) -> str:
        """
        Analyze burstiness using kurtosis
        
        Args:
            time_series: Input time series
            
        Returns:
            Burstiness level: 'high', 'medium', or 'low'
        """
        if len(time_series) < 3:
            return 'low'
        
        kurtosis = stats.kurtosis(time_series)
        
        # Kurtosis > 0: heavy-tailed (bursty)
        # Kurtosis < 0: light-tailed (smooth)
        
        if kurtosis > 1.0:
            return 'high'  # Very bursty
        elif kurtosis > 0.0:
            return 'medium'  # Moderately bursty
        else:
            return 'low'  # Smooth
    
    def compare_configurations(self, time_series: np.ndarray) -> Dict:
        """
        Compare different weight configurations
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary of scores with different configurations
        """
        features = self.extract_features(time_series)
        
        configurations = {
            'mean_only': features['mean'],
            'mean_skew': features['mean'] + 0.5 * features['skewness'],
            'mean_kurt': features['mean'] + 0.3 * features['kurtosis'],
            'all_features': (
                self.alpha * features['mean'] +
                self.beta * features['skewness'] +
                self.gamma * features['kurtosis']
            ),
        }
        
        return configurations


def compute_statistical_score(time_series: np.ndarray,
                              alpha: float = 1.0,
                              beta: float = 0.5,
                              gamma: float = 0.3) -> float:
    """
    Standalone function to compute statistical score
    
    Args:
        time_series: Input time series
        alpha: Weight for mean
        beta: Weight for skewness
        gamma: Weight for kurtosis
        
    Returns:
        Popularity score
    """
    if len(time_series) < 3:
        return float(np.mean(time_series))
    
    mean_val = np.mean(time_series)
    skewness = stats.skew(time_series)
    kurtosis = stats.kurtosis(time_series)
    
    # Handle NaN
    skewness = 0.0 if np.isnan(skewness) else skewness
    kurtosis = 0.0 if np.isnan(kurtosis) else kurtosis
    
    score = alpha * mean_val + beta * skewness + gamma * kurtosis
    
    return float(score)
