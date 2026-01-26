"""
Youku Dataset Loader
Youku Video Dataset - Primary dataset for this research
"""
import pandas as pd
import sqlite3
from pathlib import Path
from . import BaseDataLoader


class YoukuLoader(BaseDataLoader):
    """
    Loader for Youku video dataset
    
    Can load from:
    1. CSV file (preprocessed)
    2. SQLite database (original format)
    
    Expected format:
    - video_id (uploader): Video/uploader identifier
    - timestamp: Access timestamp
    - view_count: Number of views
    - category: Video category (optional)
    - tags: Video tags (optional)
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.db_path = config.get('db_path', None)
        self.use_database = config.get('use_database', False)
    
    def load(self) -> pd.DataFrame:
        """
        Load Youku dataset from CSV or database
        
        Returns:
            DataFrame with columns [timestamp, video_id, view_count, category, tags]
        """
        if self.use_database and self.db_path:
            return self._load_from_database()
        else:
            return self._load_from_csv()
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load from CSV file"""
        try:
            data = pd.read_csv(self.path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found at {self.path}. "
                "Please prepare Youku dataset CSV."
            )
        
        # Verify required columns
        required_cols = [self.time_col, self.item_col]
        missing_cols = set(required_cols) - set(data.columns)
        
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Convert timestamp
        data[self.time_col] = pd.to_datetime(data[self.time_col])
        
        # Sort by timestamp
        data = data.sort_values(self.time_col)
        
        return data
    
    def _load_from_database(self) -> pd.DataFrame:
        """
        Load from SQLite database
        Uses the YoukuDB_full.sql structure
        """
        if not Path(self.db_path).exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        
        # Query to extract time series data
        query = """
        SELECT 
            v.video_id,
            v.uploader,
            v.category,
            w.watch_date as timestamp,
            COUNT(*) as view_count
        FROM videos v
        JOIN watches w ON v.video_id = w.video_id
        GROUP BY v.video_id, w.watch_date
        ORDER BY w.watch_date
        """
        
        try:
            data = pd.read_sql_query(query, conn)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        except Exception as e:
            raise ValueError(f"Error loading from database: {e}")
        finally:
            conn.close()
        
        # Rename columns to match expected format
        data = data.rename(columns={
            'video_id': self.item_col,
            'uploader': 'uploader_id',
        })
        
        return data
    
    def extract_graph_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract graph-related features for GSP analysis
        
        Args:
            data: Loaded Youku data
            
        Returns:
            DataFrame with graph features (user similarity, category, tags)
        """
        # This will be used for Graph Signal Processing analysis
        features = pd.DataFrame()
        
        # Extract category relationships
        if 'category' in data.columns:
            category_counts = data.groupby([self.item_col, 'category']).size()
            features['category_factor'] = category_counts
        
        # Extract tag relationships (if available)
        if 'tags' in data.columns:
            # Process tags for similarity computation
            pass
        
        return features
    
    @staticmethod
    def download_info():
        """Dataset information"""
        return {
            'name': 'Youku Video Dataset',
            'description': 'Chinese video sharing platform dataset',
            'format': 'SQLite database or CSV',
            'features': [
                'video_id', 'uploader', 'category', 
                'tags', 'watch_date', 'views'
            ],
            'note': 'Primary dataset for PhD research on content popularity'
        }
    
    def get_statistics(self, data: pd.DataFrame):
        """Extended statistics for Youku dataset"""
        stats = super().get_statistics(data)
        
        # Add Youku-specific stats
        if 'category' in data.columns:
            stats['num_categories'] = data['category'].nunique()
            stats['category_distribution'] = data['category'].value_counts().to_dict()
        
        if 'uploader' in data.columns:
            stats['num_uploaders'] = data['uploader'].nunique()
        
        return stats
