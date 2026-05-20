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


class YellowTaxiLoader(BaseLoader):
    """
    Loader for converted NYC Yellow Taxi data
    
    Expected format (from converter):
        - timestamp (datetime)
        - item_id (str) - location ID
        - count (int) - number of trips
        - [optional features]
    """
    
    def __init__(self, config: dict):
        """
        Initialize NYC Yellow Taxi loader

        Args:
            config: Dataset configuration dict with keys:
                - path: Path to converted CSV file
                - time_col: Timestamp column name (default: 'timestamp')
                - item_col: Item ID column name (default: 'item_id')
                - count_col: Count column name (default: 'count')
        """
        super().__init__(config)
        logger.info(f"YellowTaxiLoader initialized for {self.data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV (implements BaseLoader abstract method)
        
        Returns:
            DataFrame with columns: timestamp, item_id, count, [features]
        """
        logger.info(f"Loading NYC Yellow Taxi data from {self.data_path}")
        
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
        Load data for incremental/temporal evaluation.

        Returns the FULL dataset (including pre-evaluation-range rows) for
        selected items so the evaluator can build warm training windows.
        Item selection is based on the evaluation range only.

        Returns:
            tuple: (data DataFrame with all dates, selected_items array)
        """
        import numpy as np

        # 1. Load full dataset
        if self.data is None:
            logger.info("Loading full dataset...")
            full_data = self.load_data()
        else:
            full_data = self.data.copy()

        logger.info(f"Initial data: {len(full_data):,} records")

        # 2. Narrow to eval range for item SELECTION only
        eval_data = full_data.copy()
        if hasattr(config, 'start_date') and config.start_date:
            start = pd.to_datetime(config.start_date)
            eval_data = eval_data[eval_data[self.time_col] >= start]
            logger.info(f"Eval-range start: {len(eval_data):,} records")

        if hasattr(config, 'end_date') and config.end_date:
            end = pd.to_datetime(config.end_date)
            eval_data = eval_data[eval_data[self.time_col] <= end]
            logger.info(f"Eval-range end: {len(eval_data):,} records")

        # 3. Select items from eval range
        use_pre = getattr(config, 'use_pre_range_data', True)

        if config.num_items and config.num_items > 0:
            item_popularity = (eval_data
                               .groupby(self.item_col)[self.count_col]
                               .sum()
                               .sort_values(ascending=False))
            logger.info(f"Total unique items in eval range: {len(item_popularity)}")
            # Filter by min_observations (same as temporal_evaluator._select_items_from_data)
            min_obs = getattr(config, 'min_observations', 0)
            if min_obs > 0:
                item_popularity = item_popularity[item_popularity >= min_obs]
            logger.info(f"After min_obs={min_obs} filter: {len(item_popularity)} items")


            selection_strategy = getattr(config, 'item_selection', 'top')
            if selection_strategy == 'top':
                selected_items = item_popularity.nlargest(config.num_items).index.values
            elif selection_strategy == 'random':
                selected_items = item_popularity.sample(
                    n=min(config.num_items, len(item_popularity)),
                    random_state=42
                ).index.values
            elif selection_strategy == 'stratified':
                # Use config.strata_thresholds (same as temporal_evaluator) for
                # consistent cross-mode item selection. Seed=42 for reproducibility.
                strata_t = getattr(config, 'strata_thresholds', None)
                if strata_t and len(strata_t) >= 3:
                    t0, t1, t2 = float(strata_t[0]), float(strata_t[1]), float(strata_t[2])
                else:
                    t0, t1, t2 = float(item_popularity.quantile(0.25)), \
                                 float(item_popularity.quantile(0.50)), \
                                 float(item_popularity.quantile(0.75))
                tiers = [
                    item_popularity[item_popularity <  t0],
                    item_popularity[(item_popularity >= t0) & (item_popularity <  t1)],
                    item_popularity[(item_popularity >= t1) & (item_popularity <  t2)],
                    item_popularity[item_popularity >= t2],
                ]
                n_each = max(1, config.num_items // 4)
                selected_items = []
                for tier in tiers:
                    if len(tier) > 0:
                        selected_items.extend(
                            tier.sample(n=min(n_each, len(tier)),
                                        random_state=42).index.tolist()
                        )
                taken = set(selected_items)
                remaining = [i for i in item_popularity.index if i not in taken]
                selected_items.extend(remaining[:config.num_items - len(selected_items)])
                selected_items = np.array(selected_items[:config.num_items])
            elif selection_strategy == 'bottom':
                selected_items = item_popularity.nsmallest(config.num_items).index.values
            else:
                selected_items = item_popularity.nlargest(config.num_items).index.values

            items = selected_items
            logger.info(f"Selected {len(items)} items ({selection_strategy})")
        else:
            items = eval_data[self.item_col].unique()
            logger.info(f"Using all {len(items)} items")

        # 4. Return ALL rows for selected items (pre-range included when use_pre=True)
        if use_pre:
            data = full_data[full_data[self.item_col].isin(items)]
            logger.info(f"Pre-range data included: {len(data):,} records "
                        f"({data[self.time_col].min()} → {data[self.time_col].max()})")
        else:
            data = eval_data[eval_data[self.item_col].isin(items)]
            logger.info(f"Pre-range data excluded: {len(data):,} records")

        data = data.sort_values(self.time_col).reset_index(drop=True)

        logger.info("="*60)
        logger.info("DATA LOADED FOR TEMPORAL EVALUATION")
        logger.info(f"  Records:    {len(data):,}")
        logger.info(f"  Items:      {len(items)}")
        logger.info(f"  Full range: {data[self.time_col].min()} → {data[self.time_col].max()}")
        logger.info(f"  Pre-range:  {use_pre}")
        logger.info("="*60)

        return data, items

# Factory function (like get_movielens_loader)
def get_yellow_taxi_loader(config: dict = None) -> YellowTaxiLoader:
    """
    Factory function for creating YellowTaxiLoader
    
    Args:
        config: Dataset configuration (optional)
    
    Returns:
        YellowTaxiLoader instance
    
    Example:
        >>> loader = get_yellow_taxi_loader()
        >>> data = loader.load_data()
    """
    if config is None:
        from config import DATASETS
        config = DATASETS.get('yellow_taxi', {
            'name': 'yellow_taxi',
            'path': 'data/datasets/yellow_taxi_hourly.csv',
            'time_col': 'timestamp',
            'item_col': 'item_id',
            'count_col': 'count'
        })
    return YellowTaxiLoader(config)
