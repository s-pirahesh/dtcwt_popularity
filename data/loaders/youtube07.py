"""
YouTube07 Dataset Loader
Dataset: YouTube-07 video access traces
"""
import pandas as pd
from . import BaseDataLoader


class YouTube07Loader(BaseDataLoader):
    """
    Loader for YouTube-07 dataset
    
    Expected format:
    - video_id: Video identifier
    - timestamp: Access timestamp
    - view_count: Number of views (optional)
    """
    
    def load(self) -> pd.DataFrame:
        """
        Load YouTube-07 dataset
        
        Returns:
            DataFrame with columns [timestamp, video_id, view_count]
        """
        try:
            # Try loading with header
            data = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {self.path}. "
                "Please download YouTube-07 dataset."
            )
        
        # Verify required columns exist
        required_cols = [self.time_col, self.item_col]
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Available columns: {data.columns.tolist()}"
            )
        
        # Convert timestamp to datetime
        data[self.time_col] = pd.to_datetime(data[self.time_col])
        
        # Sort by timestamp
        data = data.sort_values(self.time_col)
        
        return data
    
    @staticmethod
    def download_info():
        """Return information about downloading this dataset"""
        return {
            'name': 'YouTube-07',
            'url': 'http://trace.eas.asu.edu/yudata/YouTube-07',
            'description': 'YouTube video access traces from 2007',
            'size': '~1.5 GB',
            'format': 'CSV',
        }
