"""
MovieLens Dataset Loader
"""
import pandas as pd
from . import BaseDataLoader


class MovieLensLoader(BaseDataLoader):
    """
    Loader for MovieLens dataset
    
    Expected format:
    - movie_id: Movie identifier
    - timestamp: Rating timestamp
    - rating_count: Number of ratings (aggregated)
    """
    
    def load(self) -> pd.DataFrame:
        """Load MovieLens dataset"""
        try:
            data = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {self.path}. "
                "Please download MovieLens dataset."
            )
        
        # Verify required columns
        required_cols = [self.time_col, self.item_col]
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Convert timestamp (could be Unix timestamp or datetime string)
        if data[self.time_col].dtype == 'int64':
            # Unix timestamp
            data[self.time_col] = pd.to_datetime(data[self.time_col], unit='s')
        else:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
        
        # Sort by timestamp
        data = data.sort_values(self.time_col)
        
        return data
    
    @staticmethod
    def download_info():
        """Dataset information"""
        return {
            'name': 'MovieLens',
            'url': 'https://grouplens.org/datasets/movielens/',
            'description': 'Movie rating dataset',
            'versions': ['20M', '25M', 'Latest'],
        }
