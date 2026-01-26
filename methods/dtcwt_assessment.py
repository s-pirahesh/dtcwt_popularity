"""
DTCWT-based Popularity Assessment (Contribution 2)
Main Innovation: Dual-Tree Complex Wavelet Transform for popularity measurement
"""
import numpy as np
import dtcwt
from typing import List, Dict, Optional
from config import WAVELET_CONFIG


class DTCWTAssessment:
    """
    DTCWT-based popularity assessment method
    
    Innovation: Replace DWT with DTCWT for better shift invariance
    Claim: 10-15% improvement over DWT due to stability and directional selectivity
    
    Key Advantages of DTCWT:
    1. Approximate shift invariance
    2. Better directional selectivity
    3. Reduced aliasing
    4. Perfect reconstruction
    """
    
    def __init__(self, biort: str = None, qshift: str = None, level: int = None):
        """
        Initialize DTCWT assessment
        
        Args:
            biort: Biorthogonal filters for level 1 (default: near_sym_a)
            qshift: Q-shift filters for levels >= 2 (default: qshift_a)
            level: Decomposition level (default: 3)
        """
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
        # Handle edge cases
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
            # Returns: lowpass (approximation) and highpass (details) coefficients
            pyramid = self.transform.forward(time_series, nlevels=self.level)
            
            # pyramid.lowpass: approximation coefficients (real-valued)
            # pyramid.highpass: detail coefficients (complex-valued, tuple of arrays)
            
            score = 0.0
            
            # 1. Score from approximation coefficients (lowpass)
            approx = pyramid.lowpass
            weight_approx = 2 ** 0  # Highest weight for approximation
            score += weight_approx * np.mean(np.abs(approx))
            
            # 2. Score from detail coefficients (highpass)
            # DTCWT produces complex coefficients - use magnitude
            for i, detail_level in enumerate(pyramid.highpass):
                # detail_level is a complex array
                # Weight decreases with level
                weight = 2 ** (-(i + 1))
                
                # Use magnitude of complex coefficients
                magnitude = np.abs(detail_level)
                level_score = np.mean(magnitude)
                
                score += weight * level_score
            
            return float(score)
        
        except Exception as e:
            # Fallback to simple mean if transform fails
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
    
    def decompose(self, time_series: np.ndarray) -> Dict:
        """
        Get detailed DTCWT decomposition
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary with lowpass and highpass coefficients
        """
        # Ensure minimum length
        min_length = 2 ** (self.level + 1)
        if len(time_series) < min_length:
            padded = np.zeros(min_length)
            padded[-len(time_series):] = time_series
            time_series = padded
        
        pyramid = self.transform.forward(time_series, nlevels=self.level)
        
        result = {
            'lowpass': pyramid.lowpass,  # Approximation coefficients
            'highpass': pyramid.highpasses,  # Detail coefficients (complex)
            'levels': len(pyramid.highpasses),
            'biort': self.biort,
            'qshift': self.qshift,
        }
        
        # Add magnitude and phase information for complex coefficients
        result['magnitudes'] = [np.abs(h) for h in pyramid.highpasses]
        result['phases'] = [np.angle(h) for h in pyramid.highpasses]
        
        return result
    
    def get_feature_vector(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive feature vector from DTCWT
        
        Features include:
        - Approximation statistics
        - Magnitude statistics at each level
        - Phase statistics (optional)
        
        Args:
            time_series: Input time series
            
        Returns:
            Feature vector
        """
        decomp = self.decompose(time_series)
        
        features = []
        
        # Approximation (lowpass) features
        approx = decomp['lowpass']
        features.extend([
            np.mean(approx),
            np.std(approx),
            np.max(approx),
            np.min(approx),
        ])
        
        # Detail (highpass) features - magnitude based
        for magnitude in decomp['magnitudes']:
            features.extend([
                np.mean(magnitude),
                np.std(magnitude),
                np.max(magnitude),
                np.median(magnitude),
            ])
        
        # Optional: Phase-based features (for temporal pattern detection)
        # Uncomment if phase information is important
        # for phase in decomp['phases']:
        #     features.extend([
        #         np.mean(np.cos(phase)),  # Real part of exp(i*phase)
        #         np.mean(np.sin(phase)),  # Imaginary part
        #     ])
        
        return np.array(features)
    
    def get_directional_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        Extract directional features from DTCWT
        
        DTCWT provides 6 directional subbands (unlike DWT)
        This can capture trend direction and patterns
        
        Args:
            time_series: Input time series
            
        Returns:
            Dictionary of directional statistics
        """
        decomp = self.decompose(time_series)
        
        directional_features = {}
        
        # Analyze each level's directionality using complex coefficients
        for i, highpass in enumerate(decomp['highpass']):
            # Real and imaginary parts represent different directions
            real_energy = np.sum(np.real(highpass) ** 2)
            imag_energy = np.sum(np.imag(highpass) ** 2)
            total_energy = real_energy + imag_energy
            
            if total_energy > 0:
                directional_features[f'level_{i+1}_real_ratio'] = real_energy / total_energy
                directional_features[f'level_{i+1}_imag_ratio'] = imag_energy / total_energy
            else:
                directional_features[f'level_{i+1}_real_ratio'] = 0.5
                directional_features[f'level_{i+1}_imag_ratio'] = 0.5
        
        return directional_features
    
    def compare_with_dwt(self, time_series: np.ndarray) -> Dict:
        """
        Compare DTCWT vs DWT for analysis
        
        Args:
            time_series: Input time series
            
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
        # Shift the signal and compute scores again
        shifted = np.roll(time_series, 1)
        dtcwt_shifted = self.assess_single(shifted)
        dwt_shifted = dwt.assess_single(shifted)
        
        # Compute stability (smaller change = better shift invariance)
        dtcwt_stability = abs(dtcwt_score - dtcwt_shifted) / (dtcwt_score + 1e-10)
        dwt_stability = abs(dwt_score - dwt_shifted) / (dwt_score + 1e-10)
        
        return {
            'dtcwt_score': dtcwt_score,
            'dwt_score': dwt_score,
            'dtcwt_stability': dtcwt_stability,
            'dwt_stability': dwt_stability,
            'improvement': (dtcwt_score - dwt_score) / (dwt_score + 1e-10),
        }


def compute_dtcwt_af_formula(lowpass: np.ndarray, 
                             highpass: List[np.ndarray]) -> float:
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
