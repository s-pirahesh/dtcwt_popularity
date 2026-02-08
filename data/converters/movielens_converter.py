"""
MovieLens Dataset Converter
تبدیل فایل ratings.csv به فرمت استاندارد

Dataset: MovieLens (ml-25m, ml-32m)
Source: https://grouplens.org/datasets/movielens/
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from .base_converter import BaseConverter, ConverterFactory


class MovieLensConverter(BaseConverter):
    """
    Converter برای دیتاست MovieLens
    
    Input: ratings.csv
        - userId, movieId, rating, timestamp
        - timestamp: Unix timestamp (seconds)
        - rating: 0.5 to 5.0
    
    Output: Standard CSV
        - timestamp: datetime
        - item_id: movieId
        - count: 1 per rating (or aggregated)
        - rating: (optional)
        - user_id: (optional)
    
    Dataset-Specific Parameters:
        --movielens-aggregate-by: Temporal aggregation (hour/day/week/none)
        --movielens-keep-rating: Keep rating column
        --movielens-keep-user: Keep user ID column
        --movielens-min-rating: Minimum rating filter
    """
    
    # ==========================================
    # Metadata
    # ==========================================
    DATASET_NAME = 'movielens'
    SUPPORTED_FILE_TYPES = ['.csv']
    DESCRIPTION = 'MovieLens ratings dataset converter'
    
    # ==========================================
    # Dataset-Specific Parameters
    # ==========================================
    @classmethod
    def get_specific_params(cls) -> Dict[str, Dict[str, Any]]:
        return {
            'aggregate_by': {
                'type': str,
                'default': None,
                'choices': ['hour', 'day', 'week', 'none'],
                'help': 'Temporal aggregation level (default: no aggregation)'
            },
            'keep_rating': {
                'type': bool,
                'default': False,
                'help': 'Keep rating column in output'
            },
            'keep_user': {
                'type': bool,
                'default': False,
                'help': 'Keep user ID column in output'
            },
            'min_rating': {
                'type': float,
                'default': None,
                'help': 'Minimum rating to include (filter)'
            }
        }
    
    def __init__(self, 
                 aggregate_by: Optional[str] = None,
                 keep_rating: bool = False,
                 keep_user: bool = False,
                 min_rating: Optional[float] = None,
                 **kwargs):
        """
        مقداردهی اولیه
        
        Args:
            aggregate_by: تجمیع زمانی ('hour', 'day', 'week', None)
            keep_rating: نگهداری ستون rating
            keep_user: نگهداری ستون userId
            min_rating: حداقل rating برای فیلتر
            **kwargs: پارامترهای BaseConverter
        """
        super().__init__(**kwargs)
        
        # تنظیم پارامترها
        self.aggregate_by = aggregate_by if aggregate_by != 'none' else None
        self.keep_rating = keep_rating
        self.keep_user = keep_user
        self.min_rating = min_rating
    
    def _convert_single_file(self, 
                            file_path: Path,
                            **kwargs) -> pd.DataFrame:
        """
        تبدیل فایل ratings.csv
        
        Args:
            file_path: مسیر ratings.csv
            **kwargs: پارامترهای اضافی
            
        Returns:
            DataFrame استاندارد
        """
        self.log(f"Reading MovieLens file: {file_path.name}")
        
        # ==========================================
        # مرحله 1: خواندن CSV
        # ==========================================
        df = pd.read_csv(
            file_path,
            dtype={
                'userId': 'int32',
                'movieId': 'int32',
                'rating': 'float32',
                'timestamp': 'int64'
            }
        )
        
        initial_count = len(df)
        self.log(f"  Ratings count: {initial_count:,}")
        
        # ==========================================
        # مرحله 2: تبدیل timestamp
        # ==========================================
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # ==========================================
        # مرحله 3: فیلتر rating (اختیاری)
        # ==========================================
        if self.min_rating is not None:
            df = df[df['rating'] >= self.min_rating].copy()
            removed = initial_count - len(df)
            if removed > 0:
                self.log(f"  Filter rating < {self.min_rating}: {removed:,} records removed")
        
        # ==========================================
        # مرحله 4: تبدیل به فرمت استاندارد
        # ==========================================
        df['item_id'] = df['movieId'].astype(str)
        df['count'] = 1  # هر rating = 1 دسترسی
        
        # ==========================================
        # مرحله 5: تجمیع زمانی (اختیاری)
        # ==========================================
        if self.aggregate_by:
            df = self._aggregate_temporal(df)
        
        # ==========================================
        # مرحله 6: انتخاب ستون‌های خروجی
        # ==========================================
        output_columns = ['timestamp', 'item_id', 'count']
        
        if self.keep_rating:
            output_columns.append('rating')
        
        if self.keep_user:
            df['user_id'] = df['userId'].astype(str)
            output_columns.append('user_id')
        
        df_output = df[output_columns].copy()
        
        # ==========================================
        # مرحله 7: مرتب‌سازی
        # ==========================================
        df_output = df_output.sort_values('timestamp').reset_index(drop=True)
        
        self.log(f"  OK: Conversion completed: {len(df_output):,} records")
        
        return df_output
    
    def _aggregate_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تجمیع بر اساس بازه زمانی
        
        Args:
            df: DataFrame با timestamp دقیق
            
        Returns:
            DataFrame تجمیع شده
        """
        self.log(f"  Aggregating by: {self.aggregate_by}")
        
        # گرد کردن timestamp
        freq_map = {
            'hour': '1h',
            'day': '1D',
            'week': '1W'
        }
        freq = freq_map.get(self.aggregate_by, 'D')
        
        df['time_bucket'] = df['timestamp'].dt.floor(freq)
        
        # تجمیع
        agg_dict = {
            'count': 'sum',
        }
        
        if self.keep_rating and 'rating' in df.columns:
            agg_dict['rating'] = 'mean'
        
        if self.keep_user and 'userId' in df.columns:
            # تعداد کاربران یکتا در این بازه
            df['user_count'] = df.groupby(['time_bucket', 'item_id'])['userId'].transform('nunique')
            agg_dict['user_count'] = 'first'
        
        df_agg = df.groupby(['time_bucket', 'item_id'], as_index=False).agg(agg_dict)
        
        # تغییر نام
        df_agg.rename(columns={'time_bucket': 'timestamp'}, inplace=True)
        
        self.log(f"  OK: Aggregated from {len(df):,} to {len(df_agg):,} records")
        
        return df_agg
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        آمار تکمیلی
        
        Args:
            df: DataFrame تبدیل شده
            
        Returns:
            دیکشنری آمار
        """
        stats = {
            'total_records': len(df),
            'unique_movies': df['item_id'].nunique(),
            'time_span': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'date_range_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'total_interactions': df['count'].sum() if 'count' in df.columns else len(df)
        }
        
        if 'rating' in df.columns:
            stats['avg_rating'] = df['rating'].mean()
            stats['rating_std'] = df['rating'].std()
        
        if 'user_id' in df.columns:
            stats['unique_users'] = df['user_id'].nunique()
        
        return stats


# ==========================================
# ثبت در Factory
# ==========================================
ConverterFactory.register('movielens', MovieLensConverter)
