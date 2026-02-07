"""
Base Method for Popularity Assessment
Abstract base class for all popularity assessment methods

All concrete methods (DWT, DTCWT, Statistical, Hybrid, Baselines) must inherit 
from this class and implement the required methods.

Author: Sajjad
Date: February 2025
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Optional


class BaseMethod(ABC):
    """
    Abstract base class for all popularity assessment methods
    
    All concrete methods must implement:
    - assess_single(time_series) -> float
    
    Optional methods with default implementations:
    - assess_batch(time_series_list) -> np.ndarray
    - assess(time_series) -> float (alias for assess_single)
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
        
        This is the MAIN method that all subclasses MUST implement.
        
        Args:
            time_series: 1D numpy array of access counts over time
            
        Returns:
            Popularity score (float)
            
        Example:
            >>> method = MyMethod()
            >>> time_series = np.array([1, 2, 3, 4, 5])
            >>> score = method.assess_single(time_series)
            >>> print(score)
            3.0
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
            
        Example:
            >>> method = MyMethod()
            >>> series_list = [np.array([1,2,3]), np.array([4,5,6])]
            >>> scores = method.assess_batch(series_list)
            >>> print(scores)
            [2. 5.]
        """
        return np.array([self.assess_single(ts) for ts in time_series_list])
    
    def assess(self, time_series: np.ndarray) -> float:
        """
        Alias for assess_single (for compatibility with older code)
        
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
            
        Example:
            >>> method = MyMethod()
            >>> method.validate_time_series(np.array([1, 2, 3]))  # OK
            >>> method.validate_time_series([1, 2, 3])  # Raises ValueError
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
            
        Example:
            >>> method = MyMethod()
            >>> metadata = method.get_metadata()
            >>> print(metadata)
            {'name': 'MyMethod', 'class': 'MyMethod'}
        """
        return {
            'name': self.name,
            'class': self.__class__.__name__,
        }


# Example usage
if __name__ == '__main__':
    # Example implementation
    class SimpleAverage(BaseMethod):
        """Simple average-based assessment"""
        
        def __init__(self):
            super().__init__(name="SimpleAverage")
        
        def assess_single(self, time_series: np.ndarray) -> float:
            """Return the average"""
            return float(np.mean(time_series))
    
    # Test
    print("Testing BaseMethod...")
    
    method = SimpleAverage()
    print(f"Method: {method}")
    
    # Test single
    ts = np.array([1, 2, 3, 4, 5])
    score = method.assess_single(ts)
    print(f"Single score: {score}")
    
    # Test batch
    ts_list = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    scores = method.assess_batch(ts_list)
    print(f"Batch scores: {scores}")
    
    # Test validation
    try:
        method.validate_time_series(np.array([1, 2, 3]))
        print("✓ Validation passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    print("\n✓ All tests passed!")
