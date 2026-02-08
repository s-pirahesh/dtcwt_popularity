"""
Uber Dataset Loader
Loads converted Uber/NYC Yellow Taxi data

Extends BaseLoader with proper inheritance

Author: Sajjad
Date: February 2026
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class UberLoader(BaseLoader):
    """
    Loader for converted Uber/NYC Taxi data
    
    Expected format (from converter):
        - timestamp (datetime)
        - item_id (str) - location ID
        - count (int) - number of trips
        - [optional features]
    """
    
    def __init__(self, config: dict):
        """
        Initialize Uber loader
        
        Args:
            config: Dataset configuration dict with keys:
                - path: Path to converted CSV file
                - time_col: Timestamp column name (default: 'timestamp')
                - item_col: Item ID column name (default: 'item_id')
                - count_col: Count column name (default: 'count')
        """
        super().__init__(config)
        logger.info(f"UberLoader initialized for {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV (implements BaseLoader abstract method)
        
        Returns:
            DataFrame with columns: timestamp, item_id, count, [features]
        """
        logger.info(f"Loading Uber data from {self.data_path}")
        
        # Read CSV
        df = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded {len(df):,} records")
        
        # Convert timestamp
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # Ensure item_id is string
        df[self.item_col] = df[self.item_col].astype(str)
        
        # Sort by timestamp
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        # Validate
        self.validate_data(df)
        
        # Store
        self.data = df
        
        # Log stats
        logger.info(f"Locations: {df[self.item_col].nunique():,}")
        logger.info(f"Date range: {df[self.time_col].min()} to {df[self.time_col].max()}")
        
        return df
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get date range
        
        Returns:
            (min_date, max_date)
        """
        if self.data is None:
            self.load_data()
        
        return (self.data[self.time_col].min(), self.data[self.time_col].max())
    
    def get_popularity_time_series(self, 
                                   item_id: str,
                                   fill_missing: bool = False,
                                   fill_value: float = 0) -> pd.Series:
        """
        Get time series for a specific location
        
        Args:
            item_id: Location ID (string)
            fill_missing: Fill missing time slots
            fill_value: Value to fill missing slots with
            
        Returns:
            Series with index=timestamp and value=count
        """
        if self.data is None:
            self.load_data()
        
        # Filter data for this location
        loc_data = self.data[self.data[self.item_col] == item_id].copy()
        
        if loc_data.empty:
            logger.warning(f"Location {item_id} not found in data")
            return pd.Series(dtype=float)
        
        # Set index and get count series
        loc_data = loc_data.set_index(self.time_col).sort_index()
        ts = loc_data[self.count_col]
        
        # Fill missing time slots if requested
        if fill_missing:
            # Determine frequency from data
            time_diffs = ts.index.to_series().diff().dropna()
            freq = time_diffs.mode()[0] if len(time_diffs) > 0 else None
            
            if freq:
                # Create complete time range
                full_range = pd.date_range(
                    start=ts.index.min(),
                    end=ts.index.max(),
                    freq=freq
                )
                
                # Reindex with fill
                ts = ts.reindex(full_range, fill_value=fill_value)
        
        return ts
    
    def get_statistics(self) -> dict:
        """
        Get dataset statistics
        
        Returns:
            Dictionary with statistics
        """
        if self.data is None:
            self.load_data()
        
        df = self.data
        
        stats = {
            'total_records': len(df),
            'unique_locations': df[self.item_col].nunique(),
            'date_range': f"{df[self.time_col].min()} to {df[self.time_col].max()}",
            'date_range_days': (df[self.time_col].max() - df[self.time_col].min()).days,
            'total_trips': df[self.count_col].sum(),
            'avg_trips_per_record': df[self.count_col].mean(),
            'min_trips': df[self.count_col].min(),
            'max_trips': df[self.count_col].max()
        }
        
        # Add feature statistics if available
        feature_cols = [col for col in df.columns 
                       if col not in [self.time_col, self.item_col, self.count_col]]
        
        if feature_cols:
            stats['features'] = {}
            for col in feature_cols:
                stats['features'][col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return stats
    
    def get_top_locations(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N locations by total trips
        
        Args:
            n: Number of top locations
            
        Returns:
            DataFrame with location_id and total_trips
        """
        if self.data is None:
            self.load_data()
        
        top = self.data.groupby(self.item_col)[self.count_col].sum().sort_values(ascending=False).head(n)
        
        return top.reset_index().rename(columns={self.item_col: 'location_id', self.count_col: 'total_trips'})
    
    def get_temporal_pattern(self, 
                            item_id: str,
                            groupby: str = 'hour') -> pd.Series:
        """
        Get temporal pattern for a location
        
        Args:
            item_id: Location ID
            groupby: 'hour', 'dayofweek', 'month'
            
        Returns:
            Average trips grouped by time unit
        """
        if self.data is None:
            self.load_data()
        
        loc_data = self.data[self.data[self.item_col] == item_id].copy()
        
        if groupby == 'hour':
            loc_data['time_unit'] = loc_data[self.time_col].dt.hour
        elif groupby == 'dayofweek':
            loc_data['time_unit'] = loc_data[self.time_col].dt.dayofweek
        elif groupby == 'month':
            loc_data['time_unit'] = loc_data[self.time_col].dt.month
        else:
            raise ValueError(f"Invalid groupby: {groupby}")
        
        pattern = loc_data.groupby('time_unit')[self.count_col].mean()
        
        return pattern

    def load_for_temporal_evaluation(self, config):
        """
        Load data for incremental temporal evaluation
        
        Args:
            config: EvaluationConfig instance
            
        Returns:
            tuple: (data DataFrame, items numpy array)
        """
        import numpy as np
        
        # 1. Load full dataset
        if self.data is None:
            logger.info("Loading full dataset...")
            data = self.load_data()
        else:
            data = self.data.copy()
        
        logger.info(f"Initial data: {len(data):,} records")
        
        # 2. Filter by date range
        if hasattr(config, 'start_date') and config.start_date:
            start = pd.to_datetime(config.start_date)
            data = data[data[self.time_col] >= start]
            logger.info(f"After start_date filter: {len(data):,} records")
        
        if hasattr(config, 'end_date') and config.end_date:
            end = pd.to_datetime(config.end_date)
            data = data[data[self.time_col] <= end]
            logger.info(f"After end_date filter: {len(data):,} records")
        
        # 3. Select items
        if config.num_items and config.num_items > 0:
            # Calculate popularity
            item_popularity = data.groupby(self.item_col)[self.count_col].sum().sort_values(ascending=False)
            
            logger.info(f"Total unique items: {len(item_popularity)}")
            
            # Selection strategy
            selection_strategy = getattr(config, 'item_selection', 'top')
            
            if selection_strategy == 'top':
                selected_items = item_popularity.nlargest(config.num_items).index.values
            elif selection_strategy == 'random':
                selected_items = item_popularity.sample(
                    n=min(config.num_items, len(item_popularity)),
                    random_state=42
                ).index.values
            elif selection_strategy == 'bottom':
                selected_items = item_popularity.nsmallest(config.num_items).index.values
            else:
                selected_items = item_popularity.nlargest(config.num_items).index.values
            
            logger.info(f"Selected {len(selected_items)} items")
            
            # Filter data
            data = data[data[self.item_col].isin(selected_items)]
            items = selected_items
            
        else:
            # All items
            items = data[self.item_col].unique()
            logger.info(f"Using all {len(items)} items")
        
        # 4. Final sort
        data = data.sort_values(self.time_col).reset_index(drop=True)
        
        # 5. Final log
        logger.info("="*60)
        logger.info("DATA LOADED FOR INCREMENTAL EVALUATION")
        logger.info(f"  Records: {len(data):,}")
        logger.info(f"  Items: {len(items)}")
        logger.info(f"  Date range: {data[self.time_col].min()} to {data[self.time_col].max()}")
        logger.info("="*60)
        
        return data, items

# Factory function (like get_movielens_loader)
def get_uber_loader(config: dict = None) -> UberLoader:
    """
    Factory function for creating UberLoader
    
    Args:
        config: Dataset configuration (optional)
    
    Returns:
        UberLoader instance
    
    Example:
        >>> loader = get_uber_loader()
        >>> data = loader.load_data()
    """
    if config is None:
        from config import DATASETS
        config = DATASETS.get('uber', {
            'name': 'uber',
            'path': 'data/datasets/uber_hourly.csv',
            'time_col': 'timestamp',
            'item_col': 'item_id',
            'count_col': 'count'
        })
    return UberLoader(config)
