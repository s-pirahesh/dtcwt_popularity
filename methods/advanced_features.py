"""
Advanced Features for Popularity Assessment (V3.1)
Includes: Shannon Entropy, Hurst Exponent, Sample Entropy
"""
import numpy as np
from scipy import stats
from typing import Dict, Optional


class AdvancedFeatures:
    """
    Advanced feature extraction for enhanced popularity assessment
    
    Features:
    1. Shannon Entropy: Complexity/randomness measure
    2. Hurst Exponent: Long-term memory and trend persistence
    3. Sample Entropy: Regularity measure (optional, slower)
    4. Permutation Entropy: Pattern complexity (optional)
    """
    
    @staticmethod
    def shannon_entropy(time_series: np.ndarray, bins: int = 10) -> float:
        """
        Compute Shannon Entropy
        
        Measures complexity/randomness of the signal
        - High entropy: Random, unpredictable
        - Low entropy: Regular, predictable
        
        Args:
            time_series: Input time series
            bins: Number of bins for histogram
            
        Returns:
            Shannon entropy value
        """
        if len(time_series) == 0:
            return 0.0
        
        # Create histogram
        hist, _ = np.histogram(time_series, bins=bins, density=True)
        
        # Normalize to get probabilities
        hist = hist / np.sum(hist)
        
        # Remove zero probabilities
        hist = hist[hist > 0]
        
        # Compute entropy: H = -sum(p * log2(p))
        entropy = -np.sum(hist * np.log2(hist))
        
        return float(entropy)
    
    @staticmethod
    def hurst_exponent(time_series: np.ndarray, max_lag: Optional[int] = None) -> float:
        """
        Compute Hurst Exponent using R/S analysis
        
        Measures long-term memory and trend persistence:
        - H < 0.5: Mean-reverting (anti-persistent)
        - H = 0.5: Random walk (no memory)
        - H > 0.5: Trending (persistent)
        
        Args:
            time_series: Input time series
            max_lag: Maximum lag for analysis (default: len/2)
            
        Returns:
            Hurst exponent (0 to 1)
        """
        if len(time_series) < 20:
            return 0.5  # Default to random walk for short series
        
        ts = np.array(time_series)
        N = len(ts)
        
        if max_lag is None:
            max_lag = N // 2
        
        # Ensure max_lag is reasonable
        max_lag = min(max_lag, N // 2)
        
        # Compute R/S statistics for different lags
        lags = range(2, max_lag + 1)
        rs_values = []
        
        for lag in lags:
            # Divide series into non-overlapping segments
            n_segments = N // lag
            
            if n_segments == 0:
                continue
            
            rs_list = []
            
            for i in range(n_segments):
                segment = ts[i*lag:(i+1)*lag]
                
                # Mean-adjusted series
                mean_adj = segment - np.mean(segment)
                
                # Cumulative sum
                cumsum = np.cumsum(mean_adj)
                
                # Range
                R = np.max(cumsum) - np.min(cumsum)
                
                # Standard deviation
                S = np.std(segment)
                
                # R/S ratio
                if S > 0:
                    rs_list.append(R / S)
            
            if len(rs_list) > 0:
                rs_values.append(np.mean(rs_list))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Fit log(R/S) vs log(lag) to get Hurst exponent
        try:
            log_lags = np.log(list(lags)[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Linear regression: log(R/S) = H * log(lag) + c
            coeffs = np.polyfit(log_lags, log_rs, 1)
            hurst = coeffs[0]
            
            # Ensure Hurst is in valid range [0, 1]
            hurst = np.clip(hurst, 0.0, 1.0)
            
            return float(hurst)
        
        except:
            return 0.5
    
    @staticmethod
    def sample_entropy(time_series: np.ndarray, m: int = 2, r: float = None) -> float:
        """
        Compute Sample Entropy (slower but more robust than Shannon)
        
        Measures regularity/complexity of time series
        - Low value: Regular, predictable
        - High value: Complex, irregular
        
        Args:
            time_series: Input time series
            m: Pattern length (default: 2)
            r: Tolerance (default: 0.2 * std)
            
        Returns:
            Sample entropy value
        """
        if len(time_series) < 10:
            return 0.0
        
        ts = np.array(time_series)
        N = len(ts)
        
        if r is None:
            r = 0.2 * np.std(ts)
        
        if r == 0:
            return 0.0
        
        def _maxdist(xi, xj):
            """Maximum distance between patterns"""
            return max([abs(ua - va) for ua, va in zip(xi, xj)])
        
        def _phi(m):
            """Count pattern matches"""
            patterns = np.array([ts[i:i+m] for i in range(N - m + 1)])
            C = 0
            
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j and _maxdist(patterns[i], patterns[j]) <= r:
                        C += 1
            
            return C / (len(patterns) * (len(patterns) - 1))
        
        try:
            return float(-np.log(_phi(m + 1) / _phi(m)))
        except:
            return 0.0
    
    @staticmethod
    def permutation_entropy(time_series: np.ndarray, m: int = 3, tau: int = 1) -> float:
        """
        Compute Permutation Entropy
        
        Measures complexity based on permutation patterns
        
        Args:
            time_series: Input time series
            m: Pattern length (default: 3)
            tau: Time delay (default: 1)
            
        Returns:
            Permutation entropy value
        """
        if len(time_series) < m * tau:
            return 0.0
        
        ts = np.array(time_series)
        N = len(ts)
        
        # Extract all patterns
        patterns = []
        for i in range(N - (m-1) * tau):
            # Extract pattern
            pattern = [ts[i + j*tau] for j in range(m)]
            # Get permutation (rank)
            perm = tuple(np.argsort(pattern))
            patterns.append(perm)
        
        # Count unique patterns
        from collections import Counter
        pattern_counts = Counter(patterns)
        
        # Compute probabilities
        total = len(patterns)
        probs = np.array([count / total for count in pattern_counts.values()])
        
        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize by maximum entropy
        max_entropy = np.log2(np.math.factorial(m))
        
        return float(entropy / max_entropy)
    
    @staticmethod
    def compute_all_features(time_series: np.ndarray,
                            enable_sample_entropy: bool = False,
                            enable_perm_entropy: bool = False) -> Dict[str, float]:
        """
        Compute all advanced features
        
        Args:
            time_series: Input time series
            enable_sample_entropy: Whether to compute sample entropy (slow)
            enable_perm_entropy: Whether to compute permutation entropy
            
        Returns:
            Dictionary of all advanced features
        """
        features = {
            'shannon_entropy': AdvancedFeatures.shannon_entropy(time_series),
            'hurst_exponent': AdvancedFeatures.hurst_exponent(time_series),
        }
        
        if enable_sample_entropy:
            features['sample_entropy'] = AdvancedFeatures.sample_entropy(time_series)
        
        if enable_perm_entropy:
            features['permutation_entropy'] = AdvancedFeatures.permutation_entropy(time_series)
        
        return features
    
    @staticmethod
    def interpret_hurst(hurst: float) -> str:
        """
        Interpret Hurst exponent value
        
        Args:
            hurst: Hurst exponent value
            
        Returns:
            Interpretation string
        """
        if hurst < 0.4:
            return "Mean-reverting (strong anti-persistence)"
        elif hurst < 0.5:
            return "Mean-reverting (weak anti-persistence)"
        elif hurst < 0.6:
            return "Random walk (no memory)"
        elif hurst < 0.7:
            return "Trending (weak persistence)"
        else:
            return "Trending (strong persistence)"
    
    @staticmethod
    def interpret_entropy(entropy: float, max_entropy: float = 3.32) -> str:
        """
        Interpret Shannon entropy value
        
        Args:
            entropy: Entropy value
            max_entropy: Maximum possible entropy (log2(bins))
            
        Returns:
            Interpretation string
        """
        ratio = entropy / max_entropy
        
        if ratio < 0.3:
            return "Very regular (low complexity)"
        elif ratio < 0.6:
            return "Moderately regular"
        elif ratio < 0.8:
            return "Moderately complex"
        else:
            return "Highly complex (near random)"


# Utility functions for feature caching
def compute_advanced_features_cached(time_series: np.ndarray,
                                     cache_key: str = None) -> Dict[str, float]:
    """
    Compute advanced features with optional caching
    
    Args:
        time_series: Input time series
        cache_key: Key for caching (optional)
        
    Returns:
        Dictionary of advanced features
    """
    # If cache_key is provided, check cache first
    # This will be integrated with the caching module
    
    return AdvancedFeatures.compute_all_features(time_series)
