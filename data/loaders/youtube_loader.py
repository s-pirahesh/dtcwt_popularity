# -*- coding: utf-8 -*-
"""
YouTube Data Loader
بارگذاری و پردازش دیتاست YouTube (Statistics Observation - hourly views)

Author: Sajjad (with Grok assistance)
Date: February 2026
"""

import pandas as pd
from typing import Tuple, Optional, Dict, Any
from datetime import datetime
import logging

from .base_loader import BaseLoader

logger = logging.getLogger(__name__)


class YouTubeLoader(BaseLoader):
    """
    بارگذاری دیتاست YouTube (پس از پردازش توسط converter)
    
    ستون‌های مورد انتظار در فایل CSV:
    - timestamp     : زمان (datetime)
    - item_id       : videoId (str)
    - count         : تعداد بازدیدهای جدید (viewCount_diff)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading YouTube data from {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df):,} records")
        
        # تبدیل زمان به datetime
        df[self.time_col] = pd.to_datetime(df[self.time_col], errors='coerce')
        
        # حذف رکوردهای نامعتبر زمانی
        df = df.dropna(subset=[self.time_col])
        
        # مرتب‌سازی زمانی
        df = df.sort_values(self.time_col).reset_index(drop=True)
        
        self.validate_data(df)
        self.data = df
        
        logger.info(f"Videos: {df[self.item_col].nunique():,}")
        logger.info(f"Date range: {df[self.time_col].min()} to {df[self.time_col].max()}")
        
        return df

    def get_date_range(self) -> Tuple[datetime, datetime]:
        if self.data is None:
            self.data = self.load_data()
        return (self.data[self.time_col].min(), self.data[self.time_col].max())

    # -------------------------------------------------------------------------
    # متد اصلی که توسط run_popularity_assessment.py فراخوانی می‌شود
    # (دقیقاً مشابه آنچه در movielens_loader وجود دارد)
    # -------------------------------------------------------------------------
    def load_for_temporal_evaluation(self, config) -> tuple:
        """
        آماده‌سازی داده برای ارزیابی زمانی incremental
        
        این متد توسط run_popularity_assessment.py فراخوانی می‌شود.
        
        Returns:
            (data: pd.DataFrame, selected_items: np.ndarray)
        """
        if self.data is None:
            logger.info("Loading full YouTube dataset...")
            data = self.load_data()
        else:
            data = self.data.copy()
        
        logger.info(f"Initial records: {len(data):,}")

        # 1. فیلتر بازه زمانی
        if hasattr(config, 'start_date') and config.start_date:
            start = pd.to_datetime(config.start_date)
            data = data[data[self.time_col] >= start]
            logger.info(f"After start_date filter: {len(data):,} records")

        if hasattr(config, 'end_date') and config.end_date:
            end = pd.to_datetime(config.end_date)
            data = data[data[self.time_col] <= end]
            logger.info(f"After end_date filter: {len(data):,} records")

        # 2. انتخاب آیتم‌ها (ویدیوها)
        if hasattr(config, 'num_items') and config.num_items and config.num_items > 0:
            item_popularity = data.groupby(self.item_col)[self.count_col].sum().sort_values(ascending=False)
            logger.info(f"Total unique videos: {len(item_popularity)}")

            selection_strategy = getattr(config, 'item_selection', 'top')

            if selection_strategy == 'top':
                selected_items = item_popularity.nlargest(config.num_items).index.values
            elif selection_strategy == 'random':
                selected_items = item_popularity.sample(
                    n=min(config.num_items, len(item_popularity)),
                    random_state=42
                ).index.values
            elif selection_strategy == 'bottom':
                selected_items = item_popularity.nsmallest(config.num_items).index.values
            else:
                selected_items = item_popularity.nlargest(config.num_items).index.values

            logger.info(f"Selected {len(selected_items)} videos")
            data = data[data[self.item_col].isin(selected_items)]
        else:
            selected_items = data[self.item_col].unique()
            logger.info(f"Using all {len(selected_items)} videos")

        # 3. مرتب‌سازی نهایی
        data = data.sort_values(self.time_col).reset_index(drop=True)

        # لاگ نهایی
        logger.info("="*60)
        logger.info("DATA PREPARED FOR TEMPORAL EVALUATION")
        logger.info(f"  Records: {len(data):,}")
        logger.info(f"  Videos: {len(selected_items)}")
        logger.info(f"  Date range: {data[self.time_col].min().date()} to {data[self.time_col].max().date()}")
        logger.info("="*60)

        return data, selected_items


def get_youtube_loader(config: Optional[Dict[str, Any]] = None) -> YouTubeLoader:
    """
    Factory function برای ایجاد YouTubeLoader
    """
    if config is None:
        from config import DATASETS
        config = DATASETS.get('youtube', {})
    return YouTubeLoader(config)