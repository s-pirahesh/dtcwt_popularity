"""
YouTube Video Views Dataset Converter
Converts hourly view counts from YouTube videos to standard popularity format

Dataset: Statistics Observation of Random YouTube Video (Kaggle)
Source: https://www.kaggle.com/datasets/nnqkfdjq/statistics-observation-of-random-youtube-video
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from .base_converter import BaseConverter, ConverterFactory


class YouTubeConverter(BaseConverter):
    """
    Converter for YouTube hourly view data
    
    Input: CSV file
        - Time: snapshot time (hourly)
        - videoId: video ID
        - viewCount: cumulative views
        - viewCount_diff: new views since last snapshot
        - likeCount, dislikeCount, commentCount, etc.
    
    Output: Standard CSV
        - timestamp: snapshot time (aggregated if needed)
        - item_id: videoId (str)
        - count: new views (from viewCount_diff)
        - avg_likes, avg_dislikes, avg_comments (optional features)
    
    Dataset-Specific Parameters:
        --youtube-granularity: Time aggregation level (hourly/daily/none)
        --youtube-min-views-per-video: Minimum total views to keep a video
        --youtube-extract-features: Extract additional features (likes, dislikes, etc.)
        --youtube-start-date: Filter start date (YYYY-MM-DD)
        --youtube-end-date: Filter end date (YYYY-MM-DD)
    """
    
    # ==========================================
    # Metadata
    # ==========================================
    DATASET_NAME = 'youtube'
    SUPPORTED_FILE_TYPES = ['.csv']
    DESCRIPTION = 'YouTube hourly video views converter'
    
    # Granularity mapping
    GRANULARITY_MAP = {
        'none': None,  # Keep hourly
        'hourly': '1h',
        'daily': '1D'
    }
    
    # Valid date range
    MIN_VALID_DATE = '2018-05-01'
    MAX_VALID_DATE = datetime.now()
    
    # ==========================================
    # Dataset-Specific Parameters
    # ==========================================
    @classmethod
    def get_specific_params(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'granularity': {
                'type': str,
                'default': 'none',
                'choices': ['none', 'hourly', 'daily'],
                'help': 'Time aggregation level (recommended: none for hourly data)'
            },
            'min_views_per_video': {
                'type': int,
                'default': 100,
                'help': 'Minimum total views to keep a video (filter low-view videos)'
            },
            'extract_features': {
                'type': bool,
                'default': False,
                'help': 'Extract additional features (avg likes, dislikes, comments)'
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
                 granularity: str = 'none',
                 min_views_per_video: int = 100,
                 extract_features: bool = False,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 **kwargs):
        """
        Initialize
        
        Args:
            granularity: Time aggregation level
            min_views_per_video: Minimum views to keep a video
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
        self.min_views = min_views_per_video
        self.extract_features = extract_features
        
        # Date filtering
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
    
    def _convert_single_file(self,
                            file_path: Path,
                            **kwargs) -> pd.DataFrame:
        """
        Convert single CSV file
        
        Args:
            file_path: Path to .csv file
            **kwargs: Additional parameters
            
        Returns:
            Standard DataFrame
        """
        self.log(f"Reading YouTube file: {file_path.name}")
        
        # Step 1: Read CSV
        df = self._read_csv_file(file_path)
        
        # Step 2: Clean data
        df = self._clean_data(df)
        
        # Step 3: Aggregate if needed
        df_agg = self._aggregate_views(df)
        
        return df_agg
    
    def _read_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Read CSV file"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Select required columns
        required_cols = [
            'Time', 'videoId', 'viewCount_diff'
        ]  # Use diff for new views (popularity rate)
        
        optional_cols = []
        if self.extract_features:
            optional_cols = [
                'likeCount_diff', 'dislikeCount_diff', 'commentCount_diff'
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
            'Time': 'timestamp',
            'videoId': 'item_id',
            'viewCount_diff': 'count'
        })
        
        self.log(f"  Loaded {len(df):,} snapshots")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data"""
        initial_count = len(df)
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove invalid timestamps
        df = df.dropna(subset=['timestamp'])
        
        # Remove invalid item_ids
        df = df[df['item_id'].notna()]
        
        # Remove unrealistic dates
        df = df[df['timestamp'] >= self.MIN_VALID_DATE]
        df = df[df['timestamp'] <= self.MAX_VALID_DATE]
        
        # User-specified date range filter
        if self.start_date:
            df = df[df['timestamp'] >= self.start_date]
            self.log(f"  Filtered to start_date >= {self.start_date}")
        
        if self.end_date:
            df = df[df['timestamp'] <= self.end_date]
            self.log(f"  Filtered to end_date <= {self.end_date}")
        
        # Clean count (views diff)
        df = df[df['count'] >= 0]  # No negative views
        
        # Clean features (if extracting)
        if self.extract_features:
            for col in ['likeCount_diff', 'dislikeCount_diff', 'commentCount_diff']:
                if col in df.columns:
                    df = df[df[col] >= 0]  # No negatives
        
        removed = initial_count - len(df)
        if removed > 0:
            self.log(f"  Cleaned: {removed:,} invalid records removed ({removed/initial_count*100:.1f}%)")
        
        return df
    
    def _aggregate_views(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate views by time slot and video"""
        if self.granularity == 'none':
            self.log("  No aggregation (keeping hourly data)...")
            return df  # No aggregation needed
        
        self.log(f"  Aggregating by {self.granularity} slots...")
        
        # Round timestamps to slots
        df['time_slot'] = df['timestamp'].dt.floor(self.pandas_freq)
        
        # Prepare aggregation
        agg_dict = {'count': 'sum'}  # Sum new views
        
        if self.extract_features:
            if 'likeCount_diff' in df.columns:
                agg_dict['likeCount_diff'] = 'mean'
            if 'dislikeCount_diff' in df.columns:
                agg_dict['dislikeCount_diff'] = 'mean'
            if 'commentCount_diff' in df.columns:
                agg_dict['commentCount_diff'] = 'mean'
        
        # Aggregate
        grouped = df.groupby(['time_slot', 'item_id']).agg(agg_dict).reset_index()
        
        # Rename columns
        grouped = grouped.rename(columns={
            'time_slot': 'timestamp'
        })
        
        # Round feature values
        if self.extract_features:
            for col in ['likeCount_diff', 'dislikeCount_diff', 'commentCount_diff']:
                if col in grouped.columns:
                    grouped[col] = grouped[col].round(2)
        
        self.log(f"  Created {len(grouped):,} time-video records")
        
        return grouped
    
    def convert(self,
                input_path: Union[str, Path, List[Union[str, Path]]],
                output_path: Union[str, Path],
                **kwargs) -> pd.DataFrame:
        """
        Override convert to add video filtering
        """
        # Standard conversion
        df = super().convert(input_path, output_path, **kwargs)
        
        # Filter videos
        df = self._filter_videos(df)
        
        # Final save
        self.log(f"Final save: {output_path}")
        self._save_output(df, Path(output_path))
        
        return df
    
    def _filter_videos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter videos with minimum views threshold"""
        # Calculate total views per video (item_id)
        video_totals = df.groupby('item_id')['count'].sum()
        
        # Filter
        valid_videos = video_totals[video_totals >= self.min_views].index
        df_filtered = df[df['item_id'].isin(valid_videos)].copy()
        
        removed_videos = len(video_totals) - len(valid_videos)
        
        self.log(
            f"\nFilter videos (min {self.min_views} views):\n"
            f"  Kept: {len(valid_videos)} videos\n"
            f"  Removed: {removed_videos} videos"
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
            'unique_videos': df['item_id'].nunique(),
            'time_span': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'total_views': df['count'].sum(),
            'avg_views_per_slot': df['count'].mean(),
            'granularity': self.granularity
        }
        
        if self.extract_features:
            feature_stats = {}
            for col in ['likeCount_diff', 'dislikeCount_diff', 'commentCount_diff']:
                if col in df.columns:
                    feature_stats[f'{col}_mean'] = df[col].mean()
                    feature_stats[f'{col}_std'] = df[col].std()
            stats['features'] = feature_stats
        
        return stats


# ==========================================
# Register in Factory
# ==========================================
ConverterFactory.register('youtube', YouTubeConverter)


# ==========================================
# CLI Interface (for standalone testing)
# ==========================================
if __name__ == '__main__':
    """
    Usage examples:
    
    # Single file
    python youtube_converter.py count_observation_upload.csv output.csv
    
    # With options
    python youtube_converter.py count_observation_upload.csv output.csv \
        --granularity daily --min-views 200 --extract-features
    """
    import sys
    import argparse
    from glob import glob
    
    parser = argparse.ArgumentParser(description='YouTube Views Converter')
    parser.add_argument('input', help='Input CSV file(s) - use quotes for wildcards')
    parser.add_argument('output', help='Output CSV file')
    
    # Add arguments
    YouTubeConverter.add_arguments(parser, prefix=False)
    
    args = parser.parse_args()
    
    # Extract parameters
    params = YouTubeConverter.extract_params_from_args(args, prefix=False)
    
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
    converter = YouTubeConverter(**params)
    df = converter.convert(input_files, args.output)
    
    # Show statistics
    print("\nStatistics:")
    stats = converter.get_statistics(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")