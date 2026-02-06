"""
Base Method for Popularity Assessment
Abstract base class for all popularity assessment methods

Author: Sajjad
Date: February 2025
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional


class BaseMethod(ABC):
    """
    Abstract base class for all popularity assessment methods
    
    All concrete methods (DWT, DTCWT, Statistical, Hybrid) must inherit from this
    class and implement the required methods.
    """
    
    def __init__(self, name: str = "BaseMethod"):
        """
        Initialize base method
        
        Args:
            name: Display name of the method
        """
        self.name = name
    
    @abstractmethod
    def assess_single(self, time_series: np.ndarray) -> float:
        """
        Assess popularity for a single time series
        
        This is the main method that all subclasses MUST implement.
        
        Args:
            time_series: 1D numpy array of access counts over time
            
        Returns:
            Popularity score (float)
        """
        pass
    
    def assess_batch(self, time_series_list: List[np.ndarray]) -> np.ndarray:
        """
        Assess popularity for multiple time series (batch processing)
        
        Default implementation: calls assess_single for each series.
        Subclasses can override for optimized batch processing.
        
        Args:
            time_series_list: List of 1D numpy arrays
            
        Returns:
            1D numpy array of popularity scores
        """
        return np.array([self.assess_single(ts) for ts in time_series_list])
    
    def assess(self, time_series: np.ndarray) -> float:
        """
        Alias for assess_single (for compatibility)
        
        Args:
            time_series: 1D numpy array of access counts
            
        Returns:
            Popularity score
        """
        return self.assess_single(time_series)
    
    def get_name(self) -> str:
        """Get the name of this method"""
        return self.name
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return self.__str__()
    
    def validate_time_series(self, time_series: np.ndarray) -> None:
        """
        Validate input time series
        
        Raises ValueError if input is invalid.
        
        Args:
            time_series: Input to validate
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(time_series, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(time_series)}")
        
        if time_series.ndim != 1:
            raise ValueError(f"Expected 1D array, got {time_series.ndim}D")
        
        if len(time_series) == 0:
            raise ValueError("Time series cannot be empty")
        
        if not np.isfinite(time_series).all():
            raise ValueError("Time series contains non-finite values (NaN or Inf)")
    
    def get_metadata(self) -> Dict:
        """
        Get method metadata
        
        Returns:
            Dictionary of metadata
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
        }
