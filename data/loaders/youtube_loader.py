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
        Prepare data for incremental/temporal evaluation.

        Critical: returns the FULL dataset (including pre-evaluation-range rows)
        for the selected items.  The evaluator needs pre-range data to build
        warm training windows for early slots (use_pre_range_data=True behaviour).

        Selection of items is based on the evaluation range [start_date, end_date]
        so that popularity ranking reflects the period being studied, but the
        returned DataFrame contains ALL available rows for those items.

        Returns:
            (data: pd.DataFrame, selected_items: np.ndarray)
              data            — all rows for selected items (full time span)
              selected_items  — array of item IDs chosen for evaluation
        """
        if self.data is None:
            logger.info("Loading full YouTube dataset...")
            full_data = self.load_data()
        else:
            full_data = self.data.copy()

        logger.info(f"Initial records: {len(full_data):,}")

        # 1. Determine evaluation range for item SELECTION only
        #    (we do NOT cut the data here — pre-range rows are kept below)
        eval_data = full_data.copy()
        if hasattr(config, 'start_date') and config.start_date:
            start = pd.to_datetime(config.start_date)
            eval_data = eval_data[eval_data[self.time_col] >= start]
            logger.info(f"Eval-range start filter: {len(eval_data):,} records")

        if hasattr(config, 'end_date') and config.end_date:
            end = pd.to_datetime(config.end_date)
            eval_data = eval_data[eval_data[self.time_col] <= end]
            logger.info(f"Eval-range end filter: {len(eval_data):,} records")

        # 2. Select items based on eval-range popularity
        use_pre = getattr(config, 'use_pre_range_data', True)

        if hasattr(config, 'num_items') and config.num_items and config.num_items > 0:
            item_popularity = (eval_data
                               .groupby(self.item_col)[self.count_col]
                               .sum()
                               .sort_values(ascending=False))
            logger.info(f"Total unique videos in eval range: {len(item_popularity)}")
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
                # Stratified: use config.strata_thresholds — same boundaries as
                # temporal_evaluator.StratificationSystem so both modes select
                # from identical strata. Fixed random_state=42 for reproducibility.
                strata_t = getattr(config, 'strata_thresholds', None)
                if strata_t and len(strata_t) >= 3:
                    t0, t1, t2 = float(strata_t[0]), float(strata_t[1]), float(strata_t[2])
                else:
                    # Fallback: quartile-based boundaries
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
                        n_take = min(n_each, len(tier))
                        selected_items.extend(
                            tier.sample(n=n_take, random_state=42).index.tolist()
                        )
                # Fill remaining slots from any leftover items
                taken = set(selected_items)
                remaining = [i for i in item_popularity.index if i not in taken]
                selected_items.extend(remaining[:config.num_items - len(selected_items)])
                import numpy as np
                selected_items = np.array(selected_items[:config.num_items])
            elif selection_strategy == 'bottom':
                selected_items = item_popularity.nsmallest(config.num_items).index.values
            else:
                selected_items = item_popularity.nlargest(config.num_items).index.values

            logger.info(f"Selected {len(selected_items)} videos ({selection_strategy})")
        else:
            selected_items = eval_data[self.item_col].unique()
            logger.info(f"Using all {len(selected_items)} videos")

        # 3. Return ALL rows for selected items (pre-range included when use_pre=True)
        if use_pre:
            data = full_data[full_data[self.item_col].isin(selected_items)]
            logger.info(f"Pre-range data included: {len(data):,} records "
                        f"(full span: {data[self.time_col].min().date()} → "
                        f"{data[self.time_col].max().date()})")
        else:
            data = eval_data[eval_data[self.item_col].isin(selected_items)]
            logger.info(f"Pre-range data excluded: {len(data):,} records")

        data = data.sort_values(self.time_col).reset_index(drop=True)

        logger.info("="*60)
        logger.info("DATA PREPARED FOR TEMPORAL EVALUATION")
        logger.info(f"  Records:    {len(data):,}")
        logger.info(f"  Videos:     {len(selected_items)}")
        logger.info(f"  Full range: {data[self.time_col].min().date()} → "
                    f"{data[self.time_col].max().date()}")
        logger.info(f"  Pre-range:  {use_pre}")
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