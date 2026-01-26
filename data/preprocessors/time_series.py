"""
Time Series Preprocessing Utilities
"""
import numpy as np
from scipy import signal
from typing import Tuple, Optional


class TimeSeriesPreprocessor:
    """
    Preprocessing utilities for time series data
    """
    
    @staticmethod
    def normalize(ts: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize time series
        
        Args:
            ts: Input time series
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized time series
        """
        if method == 'zscore':
            mean = np.mean(ts)
            std = np.std(ts)
            if std == 0:
                return ts - mean
            return (ts - mean) / std
        
        elif method == 'minmax':
            min_val = np.min(ts)
            max_val = np.max(ts)
            if max_val == min_val:
                return np.zeros_like(ts)
            return (ts - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            median = np.median(ts)
            mad = np.median(np.abs(ts - median))
            if mad == 0:
                return ts - median
            return (ts - median) / mad
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def remove_trend(ts: np.ndarray, method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove trend from time series
        
        Args:
            ts: Input time series
            method: Detrending method ('linear', 'polynomial', 'moving_average')
            
        Returns:
            Tuple of (detrended_series, trend)
        """
        if method == 'linear':
            trend = signal.detrend(ts, type='linear')
            original_trend = ts - trend
            return trend, original_trend
        
        elif method == 'polynomial':
            x = np.arange(len(ts))
            coeffs = np.polyfit(x, ts, deg=2)
            trend_line = np.polyval(coeffs, x)
            detrended = ts - trend_line
            return detrended, trend_line
        
        elif method == 'moving_average':
            window = min(7, len(ts) // 4)
            if window < 2:
                return ts, np.zeros_like(ts)
            
            trend = np.convolve(ts, np.ones(window)/window, mode='same')
            detrended = ts - trend
            return detrended, trend
        
        else:
            raise ValueError(f"Unknown detrending method: {method}")
    
    @staticmethod
    def handle_missing(ts: np.ndarray, method: str = 'linear') -> np.ndarray:
        """
        Handle missing values (NaN or 0) in time series
        
        Args:
            ts: Input time series
            method: Interpolation method ('linear', 'forward_fill', 'backward_fill', 'mean')
            
        Returns:
            Time series with missing values filled
        """
        ts = ts.copy()
        
        # Find missing indices
        missing_mask = np.isnan(ts) | (ts == 0)
        
        if not np.any(missing_mask):
            return ts
        
        if method == 'linear':
            # Linear interpolation
            valid_indices = np.where(~missing_mask)[0]
            if len(valid_indices) == 0:
                return ts
            
            missing_indices = np.where(missing_mask)[0]
            ts[missing_indices] = np.interp(missing_indices, valid_indices, ts[valid_indices])
        
        elif method == 'forward_fill':
            # Forward fill
            for i in range(1, len(ts)):
                if missing_mask[i]:
                    ts[i] = ts[i-1]
        
        elif method == 'backward_fill':
            # Backward fill
            for i in range(len(ts)-2, -1, -1):
                if missing_mask[i]:
                    ts[i] = ts[i+1]
        
        elif method == 'mean':
            # Fill with mean of valid values
            valid_mean = np.mean(ts[~missing_mask])
            ts[missing_mask] = valid_mean
        
        return ts
    
    @staticmethod
    def smooth(ts: np.ndarray, window_size: int = 3, 
               method: str = 'moving_average') -> np.ndarray:
        """
        Smooth time series
        
        Args:
            ts: Input time series
            window_size: Size of smoothing window
            method: Smoothing method ('moving_average', 'gaussian', 'median')
            
        Returns:
            Smoothed time series
        """
        if len(ts) < window_size:
            return ts
        
        if method == 'moving_average':
            return np.convolve(ts, np.ones(window_size)/window_size, mode='same')
        
        elif method == 'gaussian':
            # Gaussian smoothing
            sigma = window_size / 6.0
            x = np.arange(-window_size//2, window_size//2 + 1)
            gaussian = np.exp(-x**2 / (2*sigma**2))
            gaussian = gaussian / gaussian.sum()
            return np.convolve(ts, gaussian, mode='same')
        
        elif method == 'median':
            # Median filter
            return signal.medfilt(ts, kernel_size=window_size)
        
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    @staticmethod
    def detect_outliers(ts: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using z-score method
        
        Args:
            ts: Input time series
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean mask indicating outliers
        """
        mean = np.mean(ts)
        std = np.std(ts)
        
        if std == 0:
            return np.zeros(len(ts), dtype=bool)
        
        z_scores = np.abs((ts - mean) / std)
        return z_scores > threshold
    
    @staticmethod
    def remove_outliers(ts: np.ndarray, threshold: float = 3.0,
                       replacement: str = 'median') -> np.ndarray:
        """
        Remove outliers from time series
        
        Args:
            ts: Input time series
            threshold: Z-score threshold
            replacement: How to replace outliers ('median', 'mean', 'interpolate')
            
        Returns:
            Time series with outliers removed
        """
        ts = ts.copy()
        outliers = TimeSeriesPreprocessor.detect_outliers(ts, threshold)
        
        if not np.any(outliers):
            return ts
        
        if replacement == 'median':
            ts[outliers] = np.median(ts[~outliers])
        elif replacement == 'mean':
            ts[outliers] = np.mean(ts[~outliers])
        elif replacement == 'interpolate':
            valid_indices = np.where(~outliers)[0]
            outlier_indices = np.where(outliers)[0]
            ts[outlier_indices] = np.interp(outlier_indices, valid_indices, ts[valid_indices])
        
        return ts
    
    @staticmethod
    def ensure_length(ts: np.ndarray, target_length: int, 
                     pad_value: float = 0.0) -> np.ndarray:
        """
        Ensure time series has specific length (pad or truncate)
        
        Args:
            ts: Input time series
            target_length: Desired length
            pad_value: Value to use for padding
            
        Returns:
            Time series of target_length
        """
        current_length = len(ts)
        
        if current_length == target_length:
            return ts
        elif current_length > target_length:
            # Truncate (keep most recent values)
            return ts[-target_length:]
        else:
            # Pad at the beginning
            padded = np.full(target_length, pad_value)
            padded[-current_length:] = ts
            return padded
