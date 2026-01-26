"""
Base data loader interface
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class BaseDataLoader(ABC):
    """
    Abstract base class for all dataset loaders.
    Ensures consistent interface across different datasets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loader with dataset configuration
        
        Args:
            config: Dictionary containing:
                - path: Path to dataset file
                - time_col: Name of timestamp column
                - item_col: Name of item identifier column
                - count_col: Name of access count column (optional)
        """
        self.config = config
        self.path = config['path']
        self.time_col = config['time_col']
        self.item_col = config['item_col']
        self.count_col = config.get('count_col', None)
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load dataset from disk
        
        Returns:
            DataFrame with columns [timestamp, item_id, ...]
        """
        pass
    
    def get_all_items(self, data: pd.DataFrame) -> List:
        """
        Extract unique item identifiers
        
        Args:
            data: Loaded DataFrame
            
        Returns:
            List of unique item IDs
        """
        return data[self.item_col].unique().tolist()
    
    def create_time_series(self, data: pd.DataFrame, item_id: Any, 
                          window_size: int, 
                          aggregation: str = 'day') -> np.ndarray:
        """
        Create time series of access counts for a specific item
        
        Args:
            data: Loaded DataFrame
            item_id: Identifier of item
            window_size: Number of time steps
            aggregation: Time aggregation level ('hour', 'day', 'week')
            
        Returns:
            1D numpy array of access counts
        """
        # Filter data for specific item
        item_data = data[data[self.item_col] == item_id].copy()
        
        if len(item_data) == 0:
            return np.zeros(window_size)
        
        # Convert timestamp to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(item_data[self.time_col]):
            item_data[self.time_col] = pd.to_datetime(item_data[self.time_col])
        
        # Set time as index
        item_data = item_data.set_index(self.time_col)
        
        # Resample based on aggregation level
        freq_map = {'hour': 'H', 'day': 'D', 'week': 'W'}
        freq = freq_map.get(aggregation, 'D')
        
        if self.count_col:
            # Sum counts if count column exists
            time_series = item_data[self.count_col].resample(freq).sum()
        else:
            # Count occurrences if no count column
            time_series = item_data.resample(freq).size()
        
        # Fill missing values with 0
        time_series = time_series.fillna(0)
        
        # Return last window_size values (or pad with zeros if shorter)
        if len(time_series) >= window_size:
            return time_series.values[-window_size:]
        else:
            # Pad with zeros at the beginning
            padded = np.zeros(window_size)
            padded[-len(time_series):] = time_series.values
            return padded
    
    def get_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute basic statistics about the dataset
        
        Args:
            data: Loaded DataFrame
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_records': len(data),
            'num_unique_items': data[self.item_col].nunique(),
            'time_range': (data[self.time_col].min(), data[self.time_col].max()),
        }
        
        if self.count_col:
            stats['total_accesses'] = data[self.count_col].sum()
            stats['avg_accesses_per_item'] = data[self.count_col].sum() / data[self.item_col].nunique()
        
        return stats
