"""
NY Yellow Taxi Dataset Converter
Converts NYC Yellow Taxi trip records to standard popularity format

Dataset: NYC Yellow Taxi Trip Records
Source: NYC TLC (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
"""
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from .base_converter import BaseConverter, ConverterFactory


class YellowTaxiConverter(BaseConverter):
    """
    Converter for NYC Yellow Taxi data
    
    Input: Parquet files
        - tpep_pickup_datetime: pickup time
        - PULocationID: pickup location (1-263)
        - fare_amount, trip_distance, passenger_count (optional features)
    
    Output: Standard CSV
        - timestamp: pickup time (aggregated by time slot)
        - item_id: pickup location zone ID (as string)
        - count: number of trips
        - avg_fare, avg_distance, avg_passengers (optional features)
    
    Dataset-Specific Parameters:
        --yellow-taxi-granularity: Time slot size (5min/15min/30min/hourly/daily)
        --yellow-taxi-min-trips-per-location: Minimum total trips to keep a location
        --yellow-taxi-extract-features: Extract additional features (fare, distance, etc.)
    """
    
    # ==========================================
    # Metadata
    # ==========================================
    DATASET_NAME = 'yellow_taxi'
    SUPPORTED_FILE_TYPES = ['.parquet']
    DESCRIPTION = 'NYC Yellow Taxi trip data converter'
    
    # Time granularity mapping 
    GRANULARITY_MAP = {
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        'hourly': '1h',
        'daily': '1D'
    }
    
    # Valid date range for filtering anomalies
    MIN_VALID_DATE = '2009-01-01'
    MAX_VALID_DATE = datetime.now()
    
    # ==========================================
    # Dataset-Specific Parameters
    # ==========================================
    @classmethod
    def get_specific_params(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'granularity': {
                'type': str,
                'default': '15min',
                'choices': ['5min', '15min', '30min', 'hourly', 'daily'],
                'help': 'Time slot size for aggregation (recommended: 15min)'
            },
            'min_trips_per_location': {
                'type': int,
                'default': 100,
                'help': 'Minimum total trips to keep a location (filter low-activity zones)'
            },
            'extract_features': {
                'type': bool,
                'default': False,
                'help': 'Extract additional features (avg fare, distance, passenger count)'
            },
            'start_date': {
                'type': str,
                'default': None,
                'help': 'Filter start date (YYYY-MM-DD format)'
            },
            'end_date': {
                'type': str,
                'default': None,
                'help': 'Filter end date (YYYY-MM-DD format)'
            }
        }
    
    def __init__(self,
                 granularity: str = '15min',
                 min_trips_per_location: int = 100,
                 extract_features: bool = False,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 **kwargs):
        """
        Initialize
        
        Args:
            granularity: Time slot size
            min_trips_per_location: Minimum trips to keep a location
            extract_features: Extract additional features
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
            **kwargs: BaseConverter parameters
        """
        super().__init__(**kwargs)
        
        # Validation
        if granularity not in self.GRANULARITY_MAP:
            raise ValueError(
                f"Invalid granularity '{granularity}'. "
                f"Choose from: {list(self.GRANULARITY_MAP.keys())}"
            )
        
        self.granularity = granularity
        self.pandas_freq = self.GRANULARITY_MAP[granularity]
        self.min_trips = min_trips_per_location
        self.extract_features = extract_features
        
        # Date filtering
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
    
    def _convert_single_file(self,
                            file_path: Path,
                            **kwargs) -> pd.DataFrame:
        """
        Convert single Parquet file
        
        Args:
            file_path: Path to .parquet file
            **kwargs: Additional parameters
            
        Returns:
            Standard DataFrame
        """
        self.log(f"Reading Yellow Taxi file: {file_path.name}")
        
        # ==========================================
        # Step 1: Read Parquet
        # ==========================================
        df = self._read_parquet_file(file_path)
        
        # ==========================================
        # Step 2: Clean data
        # ==========================================
        df = self._clean_data(df)
        
        # ==========================================
        # Step 3: Aggregate by time slot and location
        # ==========================================
        df_agg = self._aggregate_trips(df)
        
        return df_agg
    
    def _read_parquet_file(self, file_path: Path) -> pd.DataFrame:
        """Read Parquet file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read Parquet
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        # Select required columns
        required_cols = [
            'tpep_pickup_datetime',
            'PULocationID'
        ]
        
        optional_cols = []
        if self.extract_features:
            optional_cols = [
                'fare_amount',
                'trip_distance',
                'passenger_count',
                'total_amount'
            ]
        
        # Check required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Select columns
        available_optional = [col for col in optional_cols if col in df.columns]
        selected_cols = required_cols + available_optional
        
        df = df[selected_cols].copy()
        
        # Rename standard columns
        df = df.rename(columns={
            'tpep_pickup_datetime': 'timestamp',
            'PULocationID': 'location_id'
        })
        
        self.log(f"  Loaded {len(df):,} trips")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        initial_count = len(df)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Remove invalid location IDs
        df = df[df['location_id'].notna()]
        df = df[df['location_id'] > 0]
        df['location_id'] = df['location_id'].astype(int)
        
        # Remove unrealistic dates (FIXED: stricter filtering)
        # Use MIN_VALID_DATE and MAX_VALID_DATE
        df = df[df['timestamp'] >= self.MIN_VALID_DATE]
        df = df[df['timestamp'] <= self.MAX_VALID_DATE]
        
        # User-specified date range filter
        if self.start_date:
            df = df[df['timestamp'] >= self.start_date]
            self.log(f"  Filtered to start_date >= {self.start_date}")
        
        if self.end_date:
            df = df[df['timestamp'] <= self.end_date]
            self.log(f"  Filtered to end_date <= {self.end_date}")
        
        # Clean features (if extracting)
        if self.extract_features:
            if 'fare_amount' in df.columns:
                df = df[df['fare_amount'] >= 0]
                df = df[df['fare_amount'] <= 500]  # $500 max
            
            if 'trip_distance' in df.columns:
                df = df[df['trip_distance'] >= 0]
                df = df[df['trip_distance'] <= 100]  # 100 miles max
            
            if 'passenger_count' in df.columns:
                df = df[df['passenger_count'] >= 1]
                df = df[df['passenger_count'] <= 6]
        
        removed = initial_count - len(df)
        if removed > 0:
            self.log(f"  Cleaned: {removed:,} invalid records removed ({removed/initial_count*100:.1f}%)")
        
        return df
    
    def _aggregate_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate trips by time slot and location"""
        self.log(f"  Aggregating by {self.granularity} slots...")
        
        # Round timestamps to time slots (FIXED: use new pandas freq)
        df['time_slot'] = df['timestamp'].dt.floor(self.pandas_freq)
        
        # Prepare aggregation
        agg_dict = {'timestamp': 'count'}  # Count trips
        
        if self.extract_features:
            # Add feature aggregations
            if 'fare_amount' in df.columns:
                agg_dict['fare_amount'] = 'mean'
            if 'trip_distance' in df.columns:
                agg_dict['trip_distance'] = 'mean'
            if 'passenger_count' in df.columns:
                agg_dict['passenger_count'] = 'mean'
            if 'total_amount' in df.columns:
                agg_dict['total_amount'] = 'mean'
        
        # Aggregate
        grouped = df.groupby(['time_slot', 'location_id']).agg(agg_dict).reset_index()
        
        # Rename columns
        grouped = grouped.rename(columns={
            'time_slot': 'timestamp',
            'timestamp': 'count'
        })
        
        # Convert location_id to item_id (FIXED: no duplicate column)
        grouped['item_id'] = grouped['location_id'].astype(str)
        
        # Remove location_id column (keep only item_id)
        grouped = grouped.drop(columns=['location_id'])
        
        # Reorder columns: timestamp, item_id, count, [features...]
        base_cols = ['timestamp', 'item_id', 'count']
        feature_cols = [col for col in grouped.columns if col not in base_cols]
        grouped = grouped[base_cols + feature_cols]
        
        # Round feature values
        if self.extract_features:
            for col in feature_cols:
                grouped[col] = grouped[col].round(2)
        
        self.log(f"  Created {len(grouped):,} time-location records")
        
        return grouped
    
    def convert(self,
                input_path: Union[str, Path, List[Union[str, Path]]],
                output_path: Union[str, Path],
                **kwargs) -> pd.DataFrame:
        """
        Override convert to add location filtering
        """
        # Standard conversion
        df = super().convert(input_path, output_path, **kwargs)
        
        # Filter locations
        df = self._filter_locations(df)
        
        # Final save
        self.log(f"Final save: {output_path}")
        self._save_output(df, Path(output_path))
        
        return df
    
    def _filter_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter locations with minimum trip threshold"""
        # Calculate total trips per location (item_id)
        location_totals = df.groupby('item_id')['count'].sum()
        
        # Filter
        valid_locations = location_totals[location_totals >= self.min_trips].index
        df_filtered = df[df['item_id'].isin(valid_locations)].copy()
        
        removed_locations = len(location_totals) - len(valid_locations)
        
        self.log(
            f"\nFilter locations (min {self.min_trips} trips):\n"
            f"  Kept: {len(valid_locations)} locations\n"
            f"  Removed: {removed_locations} locations"
        )
        
        return df_filtered
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extended statistics
        
        Args:
            df: Converted DataFrame
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_records': len(df),
            'unique_locations': df['item_id'].nunique(),
            'time_span': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'total_trips': df['count'].sum(),
            'avg_trips_per_slot': df['count'].mean(),
            'granularity': self.granularity
        }
        
        if self.extract_features:
            feature_stats = {}
            for col in ['fare_amount', 'trip_distance', 'passenger_count']:
                if col in df.columns:
                    feature_stats[f'{col}_mean'] = df[col].mean()
                    feature_stats[f'{col}_std'] = df[col].std()
            stats['features'] = feature_stats
        
        return stats


# ==========================================
# Register in Factory
# ==========================================
ConverterFactory.register('yellow_taxi', YellowTaxiConverter)


# ==========================================
# CLI Interface (for standalone testing)
# ==========================================
if __name__ == '__main__':
    """
    Usage examples:
    
    # Single file
    python yellow_taxi_converter.py yellow_2024_01.parquet output.csv
    
    # With options
    python yellow_taxi_converter.py yellow_2024_01.parquet output.csv \
        --granularity hourly --min-trips 200 --extract-features
    """
    import sys
    import argparse
    from glob import glob
    
    parser = argparse.ArgumentParser(description='NYC Yellow Taxi Converter')
    parser.add_argument('input', help='Input Parquet file(s) - use quotes for wildcards')
    parser.add_argument('output', help='Output CSV file')
    
    # Add arguments
    YellowTaxiConverter.add_arguments(parser, prefix=False)
    
    args = parser.parse_args()
    
    # Extract parameters
    params = YellowTaxiConverter.extract_params_from_args(args, prefix=False)
    
    # Process wildcards
    if '*' in args.input or '?' in args.input:
        input_files = sorted(glob(args.input))
        if not input_files:
            print(f"Error: No files found matching '{args.input}'")
            sys.exit(1)
        print(f"Found {len(input_files)} files")
    else:
        input_files = args.input
    
    # Convert
    converter = YellowTaxiConverter(**params)
    df = converter.convert(input_files, args.output)
    
    # Show statistics
    print("\nStatistics:")
    stats = converter.get_statistics(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")