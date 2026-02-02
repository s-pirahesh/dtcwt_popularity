# -*- coding: utf-8 -*-
"""
MovieLens Data Loader
بارگذاری و پردازش دیتاست MovieLens 25M

MovieLens 25M contains:
- 25 million ratings
- 62,000 movies
- 162,000 users
- Time span: 1995-2018

Author: Sajjad
Date: February 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from datetime import datetime
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MovieLensLoader(BaseLoader):
    """
    بارگذاری دیتاست MovieLens 25M
    
    فایل ratings.csv شامل:
    - userId: شناسه کاربر
    - movieId: شناسه فیلم
    - rating: امتیاز (0.5-5.0)
    - timestamp: زمان Unix (seconds)
    """
    
    def __init__(self, data_dir: str = 'data/raw/movielens'):
        """
        Args:
            data_dir: مسیر دایرکتوری MovieLens
        """
        super().__init__(data_dir, 'MovieLens 25M')
        
        self.ratings_file = self.data_dir / 'ratings.csv'
        self.movies_file = self.data_dir / 'movies.csv'
        
        # بررسی وجود فایل‌ها
        if not self.ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {self.ratings_file}")
        
        logger.info(f"MovieLens data directory: {self.data_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """
        بارگذاری ratings از فایل CSV
        
        Returns:
            DataFrame با ستون‌های [timestamp, item_id, user_id, rating]
        """
        logger.info(f"Loading MovieLens ratings from {self.ratings_file}")
        
        # بارگذاری CSV
        df = pd.read_csv(
            self.ratings_file,
            dtype={
                'userId': np.int32,
                'movieId': np.int32,
                'rating': np.float32,
                'timestamp': np.int64
            }
        )
        
        logger.info(f"Loaded {len(df):,} ratings")
        
        # تبدیل timestamp از Unix seconds به datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # تغییر نام ستون‌ها به استاندارد
        df = df.rename(columns={
            'userId': 'user_id',
            'movieId': 'item_id'
        })
        
        # مرتب‌سازی بر اساس زمان
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # اعتبارسنجی
        self.validate_data(df)
        
        # ذخیره برای استفاده بعدی
        self.data = df
        
        # نمایش آمار
        logger.info(f"Users:  {df['user_id'].nunique():,}")
        logger.info(f"Movies: {df['item_id'].nunique():,}")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        دریافت بازه زمانی دیتاست
        
        Returns:
            (start_date, end_date)
        """
        if self.data is None:
            self.data = self.load_data()
        
        start = self.data['timestamp'].min()
        end = self.data['timestamp'].max()
        
        return (start, end)
    
    def load_movies_metadata(self) -> pd.DataFrame:
        """
        بارگذاری اطلاعات فیلم‌ها (اختیاری)
        
        Returns:
            DataFrame با ستون‌های [movieId, title, genres]
        """
        if not self.movies_file.exists():
            logger.warning(f"Movies file not found: {self.movies_file}")
            return pd.DataFrame()
        
        movies = pd.read_csv(self.movies_file)
        logger.info(f"Loaded {len(movies):,} movies metadata")
        
        return movies
    
    def get_genre_statistics(self) -> pd.DataFrame:
        """
        آمار ژانر فیلم‌ها (اختیاری)
        
        Returns:
            DataFrame با تعداد فیلم‌ها در هر ژانر
        """
        movies = self.load_movies_metadata()
        
        if movies.empty:
            return pd.DataFrame()
        
        # تفکیک ژانرها (جدا شده با |)
        all_genres = []
        for genres in movies['genres']:
            all_genres.extend(genres.split('|'))
        
        # شمارش
        genre_counts = pd.Series(all_genres).value_counts()
        
        return genre_counts.to_frame('count')
    
    def filter_by_rating(self, min_rating: float = 3.0) -> pd.DataFrame:
        """
        فیلتر بر اساس امتیاز (اختیاری)
        
        Args:
            min_rating: حداقل امتیاز
        
        Returns:
            DataFrame فیلتر شده
        """
        if self.data is None:
            self.data = self.load_data()
        
        df = self.data[self.data['rating'] >= min_rating].copy()
        logger.info(f"Filtered by rating >= {min_rating}: {len(df):,} records")
        
        return df
    
    def get_popular_items(self, top_n: int = 100) -> pd.DataFrame:
        """
        دریافت محبوب‌ترین فیلم‌ها
        
        Args:
            top_n: تعداد فیلم‌های برتر
        
        Returns:
            DataFrame با [item_id, count, avg_rating]
        """
        if self.data is None:
            self.data = self.load_data()
        
        popular = self.data.groupby('item_id').agg({
            'rating': ['count', 'mean']
        }).reset_index()
        
        popular.columns = ['item_id', 'count', 'avg_rating']
        popular = popular.sort_values('count', ascending=False).head(top_n)
        
        return popular
    
    def get_temporal_distribution(self) -> pd.DataFrame:
        """
        توزیع زمانی ratings
        
        Returns:
            DataFrame با [year, month, count]
        """
        if self.data is None:
            self.data = self.load_data()
        
        df = self.data.copy()
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        
        temporal = df.groupby(['year', 'month']).size().reset_index(name='count')
        
        return temporal
    
    def create_sample_dataset(self, 
                            sample_ratio: float = 0.01,
                            output_file: str = None) -> pd.DataFrame:
        """
        ایجاد نمونه کوچک از دیتاست (برای تست)
        
        Args:
            sample_ratio: نسبت نمونه (0.01 = 1%)
            output_file: مسیر ذخیره (اختیاری)
        
        Returns:
            DataFrame نمونه
        """
        if self.data is None:
            self.data = self.load_data()
        
        # نمونه‌برداری تصادفی
        sample = self.data.sample(frac=sample_ratio, random_state=42)
        sample = sample.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Created sample with {len(sample):,} records ({sample_ratio*100:.1f}%)")
        
        # ذخیره اگر خواسته شده
        if output_file:
            sample.to_csv(output_file, index=False)
            logger.info(f"Sample saved to {output_file}")
        
        return sample


# تابع کمکی برای ایجاد loader
def get_movielens_loader(data_dir: str = 'data/raw/movielens') -> MovieLensLoader:
    """
    Factory function برای ایجاد MovieLensLoader
    
    Args:
        data_dir: مسیر دایرکتوری داده
    
    Returns:
        MovieLensLoader instance
    
    Example:
        >>> loader = get_movielens_loader()
        >>> data = loader.prepare_temporal_data(
        ...     start_date='2018-01-01',
        ...     end_date='2018-12-31',
        ...     num_items=1000
        ... )
    """
    return MovieLensLoader(data_dir)
