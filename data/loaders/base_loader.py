# -*- coding: utf-8 -*-
"""
Base Data Loader
Abstract base class for loading datasets.

This class defines the common interface for all data loaders.

Author: Sajjad
Date: February 2025
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    Abstract base class for loading datasets.

    All data loaders must inherit from this class and implement
    the abstract methods.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Dataset configuration (file path and column names)
        """
        self.config = config
        self.data_path = Path(self.config['path'])
        self.time_col = self.config['time_col']
        self.item_col = self.config['item_col']
        self.count_col = self.config['count_col']
        self.dataset_name = self.config.get('name', self.data_path.stem)
        self.data: Optional[pd.DataFrame] = None
        
        logger.info(f"Initializing {self.dataset_name} loader")
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        Load raw data from file.

        This method should:
        1. Read the data file
        2. Convert time column to datetime
        3. Have required columns (time_col/item_col/count_col)

        Returns:
            DataFrame with standard dataset columns
        """
        pass

    @abstractmethod
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Get dataset time range.

        Returns:
            (start_date, end_date)
        """
        pass
    
    def filter_by_date(self,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Filter data by date range.

        Args:
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None

        Returns:
            Filtered DataFrame
        """
        if self.data is None:
            self.data = self.load_data()

        df = self.data.copy()

        # Filter start date
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df[self.time_col] >= start_dt]
            logger.info(f"Filtered from {start_date}: {len(df):,} records")

        # Filter end date
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df[self.time_col] <= end_dt]
            logger.info(f"Filtered until {end_date}: {len(df):,} records")

        return df
    
    def aggregate_by_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data on a daily basis.

        Args:
            df: DataFrame with timestamp column

        Returns:
            DataFrame with columns [time_col, item_col, count_col]
        """
        # Extract date (without time)
        df = df.copy()
        df[self.time_col] = df[self.time_col].dt.date

        # Daily aggregation
        daily = (
            df.groupby([self.time_col, self.item_col])[self.count_col]
            .sum()
            .reset_index()
        )

        # Convert date to datetime
        daily[self.time_col] = pd.to_datetime(daily[self.time_col])

        # Sort by date and item
        daily = daily.sort_values([self.time_col, self.item_col]).reset_index(drop=True)

        logger.info(f"Aggregated to daily: {len(daily):,} records")
        logger.info(
            f"Date range: {daily[self.time_col].min()} to {daily[self.time_col].max()}"
        )
        logger.info(f"Unique items: {daily[self.item_col].nunique():,}")

        return daily
    
    def get_item_list(self,
                     df: pd.DataFrame,
                     num_items: Optional[int] = None,
                     selection: str = 'top') -> List[int]:
        """
        Get list of items based on selection strategy.

        Args:
            df: DataFrame with item_id column
            num_items: Number of items (None = all)
            selection: Selection method ('top', 'random', 'stratified')

        Returns:
            List of selected item_id values
        """
        # Count occurrences of each item
        item_counts = (
            df.groupby(self.item_col)[self.count_col]
            .sum()
            .sort_values(ascending=False)
        )

        # If num_items is not defined, return all
        if num_items is None:
            items = item_counts.index.tolist()
            logger.info(f"Selected all {len(items):,} items")
            return items

        # Limit to maximum available
        num_items = min(num_items, len(item_counts))

        if selection == 'top':
            # Most popular items
            items = item_counts.head(num_items).index.tolist()
            logger.info(f"Selected top {num_items:,} items")

        elif selection == 'random':
            # Random selection
            items = item_counts.sample(n=num_items, random_state=42).index.tolist()
            logger.info(f"Selected {num_items:,} random items")

        elif selection == 'stratified':
            # Stratified selection (uniform distribution from strata)
            # Divide into 4 strata (quartiles)
            quartiles = [0, 0.25, 0.5, 0.75, 1.0]
            thresholds = item_counts.quantile(quartiles).values

            items_per_stratum = num_items // 4
            items = []

            for i in range(4):
                lower = thresholds[i]
                upper = thresholds[i+1]

                # Items of this stratum
                stratum_items = item_counts[
                    (item_counts >= lower) & (item_counts < upper)
                ]

                # Random selection
                n_select = min(items_per_stratum, len(stratum_items))
                selected = stratum_items.sample(n=n_select, random_state=42).index.tolist()
                items.extend(selected)

            logger.info(f"Selected {len(items):,} stratified items")

        else:
            raise ValueError(f"Unknown selection method: {selection}")

        return items
    
    def prepare_temporal_data(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             num_items: Optional[int] = None,
                             item_selection: str = 'top') -> pd.DataFrame:
        """
        Complete preparation of temporal data.

        This method performs the complete preparation workflow:
        1. Load data
        2. Filter by date range
        3. Select items
        4. Aggregate daily

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            num_items: Number of items
            item_selection: Selection method

        Returns:
            DataFrame with columns [time_col, item_col, count_col]
        """
        logger.info("="*70)
        logger.info("DATA PREPARATION")
        logger.info("="*70)

        # 1. Filter by date
        df = self.filter_by_date(start_date, end_date)

        # 2. Select items
        items = self.get_item_list(df, num_items, item_selection)
        df = df[df[self.item_col].isin(items)]
        logger.info(f"Filtered to {len(items):,} items: {len(df):,} records")

        # 3. Daily aggregation
        daily_data = self.aggregate_by_day(df)

        # 4. Final statistics
        logger.info("="*70)
        logger.info(f"Dataset:         {self.dataset_name}")
        logger.info(f"Items:           {daily_data[self.item_col].nunique():,}")
        logger.info(f"Days:            {daily_data[self.time_col].nunique():,}")
        logger.info(f"Records:         {len(daily_data):,}")
        logger.info(
            f"Date range:      {daily_data[self.time_col].min()} "
            f"to {daily_data[self.time_col].max()}"
        )
        logger.info("="*70 + "\n")

        return daily_data
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate data integrity.

        Args:
            df: DataFrame to validate

        Returns:
            True if valid

        Raises:
            ValueError: If data is invalid
        """
        # Check for required columns
        required_cols = [self.time_col, self.item_col, self.count_col]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Check that dataset is not empty
        if len(df) == 0:
            raise ValueError("Empty dataset")

        # Check timestamp column type
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            raise ValueError(f"{self.time_col} column must be datetime")

        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found: {null_counts[null_counts > 0]}")

        logger.info("✓ Data validation passed")
        return True
    
    def get_statistics(self) -> dict:
        """
        Get overall dataset statistics.

        Returns:
            Dictionary with various statistics
        """
        if self.data is None:
            self.data = self.load_data()

        stats = {
            'total_records': len(self.data),
            'unique_items': self.data[self.item_col].nunique(),
            'date_range': self.get_date_range(),
            'duration_days': (
                self.data[self.time_col].max() - self.data[self.time_col].min()
            ).days,
        }

        return stats
