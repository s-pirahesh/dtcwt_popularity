# -*- coding: utf-8 -*-
"""
MovieLens Data Loader
بارگذاری و پردازش دیتاست MovieLens 32M

MovieLens 32M contains:
- 32 million ratings
- 87,585 movies
- 200,948 users
- Time span: 1995-2023

Author: Sajjad
Date: February 2026
"""

import pandas as pd
from typing import Tuple
from datetime import datetime
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class MovieLensLoader(BaseLoader):
    """
    بارگذاری دیتاست MovieLens 32M
    
    فایل خروجی آماده‌سازی شامل:
    - timestamp: زمان (datetime)
    - item_id: شناسه فیلم
    - count: تعداد تعاملات
    """
    
    def __init__(self, config: dict):
        """
        Args:
            config: تنظیمات دیتاست MovieLens
        """
        super().__init__(config)
    
    def load_data(self) -> pd.DataFrame:
        """
        بارگذاری داده‌های پردازش‌شده از فایل CSV
        
        Returns:
            DataFrame با ستون‌های [timestamp, item_id, count, ...]
        """
        logger.info(f"Loading MovieLens data from {self.data_path}")
        
        # بارگذاری CSV
        df = pd.read_csv(self.data_path)
        
        logger.info(f"Loaded {len(df):,} records")
        
        # تبدیل ستون زمان به datetime
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        
        # مرتب‌سازی بر اساس زمان
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        # اعتبارسنجی
        self.validate_data(df)
        
        # ذخیره برای استفاده بعدی
        self.data = df
        
        # نمایش آمار
        logger.info(f"Movies: {df[self.item_col].nunique():,}")
        logger.info(
            f"Date range: {df[self.time_col].min()} to {df[self.time_col].max()}"
        )
        
        return df
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        دریافت بازه زمانی دیتاست
        
        Returns:
            (start_date, end_date)
        """
        if self.data is None:
            self.data = self.load_data()
        
        start = self.data[self.time_col].min()
        end = self.data[self.time_col].max()
        
        return (start, end)
    
    def load_movies_metadata(self) -> pd.DataFrame:
        """
        بارگذاری اطلاعات فیلم‌ها (اختیاری)
        
        Returns:
            DataFrame با ستون‌های [movieId, title, genres]
        """
        logger.warning("Movies metadata not available in processed dataset.")
        return pd.DataFrame()
    
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
        
        if 'rating' not in self.data.columns:
            logger.warning("Rating column not available; returning full dataset.")
            return self.data.copy()

        df = self.data[self.data['rating'] >= min_rating].copy()
        logger.info(f"Filtered by rating >= {min_rating}: {len(df):,} records")
        
        return df
    
    def get_popular_items(self, top_n: int = 100) -> pd.DataFrame:
        """
        دریافت محبوب‌ترین فیلم‌ها
        
        Args:
            top_n: تعداد فیلم‌های برتر
        
        Returns:
            DataFrame با [item_id, count, avg_rating (optional)]
        """
        if self.data is None:
            self.data = self.load_data()
        
        if 'rating' in self.data.columns:
            popular = self.data.groupby(self.item_col).agg({
                'rating': ['count', 'mean']
            }).reset_index()
            popular.columns = [self.item_col, 'count', 'avg_rating']
        else:
            popular = (
                self.data.groupby(self.item_col)[self.count_col]
                .sum()
                .reset_index()
                .rename(columns={self.count_col: 'count'})
            )

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
        df['year'] = df[self.time_col].dt.year
        df['month'] = df[self.time_col].dt.month
        
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
        sample = sample.sort_values(self.time_col).reset_index(drop=True)
        
        logger.info(f"Created sample with {len(sample):,} records ({sample_ratio*100:.1f}%)")
        
        # ذخیره اگر خواسته شده
        if output_file:
            sample.to_csv(output_file, index=False)
            logger.info(f"Sample saved to {output_file}")
        
        return sample

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
                        f"({data[self.time_col].min().date()} → "
                        f"{data[self.time_col].max().date()})")
        else:
            data = eval_data[eval_data[self.item_col].isin(items)]
            logger.info(f"Pre-range data excluded: {len(data):,} records")

        data = data.sort_values(self.time_col).reset_index(drop=True)

        logger.info("="*60)
        logger.info("DATA LOADED FOR TEMPORAL EVALUATION")
        logger.info(f"  Records:    {len(data):,}")
        logger.info(f"  Items:      {len(items)}")
        logger.info(f"  Full range: {data[self.time_col].min().date()} → "
                    f"{data[self.time_col].max().date()}")
        logger.info(f"  Pre-range:  {use_pre}")
        logger.info("="*60)

        return data, items

# تابع کمکی برای ایجاد loader
def get_movielens_loader(config: dict = None) -> MovieLensLoader:
    """
    Factory function برای ایجاد MovieLensLoader
    
    Args:
        config: تنظیمات دیتاست (اختیاری)
    
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
    if config is None:
        from config import DATASETS
        config = DATASETS['movielens']
    return MovieLensLoader(config)
