# -*- coding: utf-8 -*-
"""
Base Data Loader
کلاس پایه انتزاعی برای بارگذاری دیتاست‌ها

این کلاس interface مشترک برای همه data loaders را تعریف می‌کند.

Author: Sajjad
Date: February 2025
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    کلاس پایه انتزاعی برای بارگذاری دیتاست‌ها
    
    همه data loaders باید از این کلاس ارث‌بری کنند و متدهای
    انتزاعی را پیاده‌سازی نمایند.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: پیکربندی دیتاست (مسیر فایل و نام ستون‌ها)
        """
        self.config = config
        self.data_path = Path(self.config['path'])
        self.time_col = self.config['time_col']
        self.item_col = self.config['item_col']
        self.count_col = self.config['count_col']
        self.dataset_name = self.config.get('name', self.data_path.stem)
        self.data: Optional[pd.DataFrame] = None
        
        logger.info(f"Initializing {self.dataset_name} loader")
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        بارگذاری داده‌های خام از فایل
        
        این متد باید:
        1. فایل داده را بخواند
        2. ستون زمان را به datetime تبدیل کند
        3. ستون‌های ضروری را داشته باشد (time_col/item_col/count_col)
        
        Returns:
            DataFrame با ستون‌های استاندارد دیتاست
        """
        pass
    
    @abstractmethod
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        دریافت بازه زمانی دیتاست
        
        Returns:
            (start_date, end_date)
        """
        pass
    
    def filter_by_date(self, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        فیلتر کردن داده‌ها بر اساس بازه زمانی
        
        Args:
            start_date: تاریخ شروع (YYYY-MM-DD) یا None
            end_date: تاریخ پایان (YYYY-MM-DD) یا None
        
        Returns:
            DataFrame فیلتر شده
        """
        if self.data is None:
            self.data = self.load_data()
        
        df = self.data.copy()
        
        # فیلتر تاریخ شروع
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df[self.time_col] >= start_dt]
            logger.info(f"Filtered from {start_date}: {len(df):,} records")
        
        # فیلتر تاریخ پایان
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df[self.time_col] <= end_dt]
            logger.info(f"Filtered until {end_date}: {len(df):,} records")
        
        return df
    
    def aggregate_by_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تجمیع داده‌ها به صورت روزانه
        
        Args:
            df: DataFrame با timestamp
        
        Returns:
            DataFrame با ستون‌های [time_col, item_col, count_col]
        """
        # استخراج تاریخ (بدون ساعت)
        df = df.copy()
        df[self.time_col] = df[self.time_col].dt.date
        
        # تجمیع روزانه
        daily = (
            df.groupby([self.time_col, self.item_col])[self.count_col]
            .sum()
            .reset_index()
        )
        
        # تبدیل date به datetime
        daily[self.time_col] = pd.to_datetime(daily[self.time_col])
        
        # مرتب‌سازی
        daily = daily.sort_values([self.time_col, self.item_col]).reset_index(drop=True)
        
        logger.info(f"Aggregated to daily: {len(daily):,} records")
        logger.info(
            f"Date range: {daily[self.time_col].min()} to {daily[self.time_col].max()}"
        )
        logger.info(f"Unique items: {daily[self.item_col].nunique():,}")
        
        return daily
    
    def get_item_list(self, 
                     df: pd.DataFrame,
                     num_items: Optional[int] = None,
                     selection: str = 'top') -> List[int]:
        """
        دریافت لیست آیتم‌ها بر اساس استراتژی انتخاب
        
        Args:
            df: DataFrame با ستون item_id
            num_items: تعداد آیتم‌ها (None = همه)
            selection: روش انتخاب ('top', 'random', 'stratified')
        
        Returns:
            لیست item_id های انتخاب شده
        """
        # شمارش تعداد دفعات هر آیتم
        item_counts = (
            df.groupby(self.item_col)[self.count_col]
            .sum()
            .sort_values(ascending=False)
        )
        
        # اگر num_items تعریف نشده، همه را برگردان
        if num_items is None:
            items = item_counts.index.tolist()
            logger.info(f"Selected all {len(items):,} items")
            return items
        
        # محدود کردن به حداکثر موجود
        num_items = min(num_items, len(item_counts))
        
        if selection == 'top':
            # محبوب‌ترین‌ها
            items = item_counts.head(num_items).index.tolist()
            logger.info(f"Selected top {num_items:,} items")
        
        elif selection == 'random':
            # تصادفی
            items = item_counts.sample(n=num_items, random_state=42).index.tolist()
            logger.info(f"Selected {num_items:,} random items")
        
        elif selection == 'stratified':
            # طبقه‌بندی شده (توزیع یکنواخت از strata)
            # تقسیم به 4 طبقه
            quartiles = [0, 0.25, 0.5, 0.75, 1.0]
            thresholds = item_counts.quantile(quartiles).values
            
            items_per_stratum = num_items // 4
            items = []
            
            for i in range(4):
                lower = thresholds[i]
                upper = thresholds[i+1]
                
                # آیتم‌های این طبقه
                stratum_items = item_counts[
                    (item_counts >= lower) & (item_counts < upper)
                ]
                
                # انتخاب تصادفی
                n_select = min(items_per_stratum, len(stratum_items))
                selected = stratum_items.sample(n=n_select, random_state=42).index.tolist()
                items.extend(selected)
            
            logger.info(f"Selected {len(items):,} stratified items")
        
        else:
            raise ValueError(f"Unknown selection method: {selection}")
        
        return items
    
    def prepare_temporal_data(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             num_items: Optional[int] = None,
                             item_selection: str = 'top') -> pd.DataFrame:
        """
        آماده‌سازی کامل داده‌های زمانی
        
        این متد جریان کامل آماده‌سازی را انجام می‌دهد:
        1. بارگذاری داده
        2. فیلتر تاریخ
        3. انتخاب آیتم‌ها
        4. تجمیع روزانه
        
        Args:
            start_date: تاریخ شروع (YYYY-MM-DD)
            end_date: تاریخ پایان (YYYY-MM-DD)
            num_items: تعداد آیتم‌ها
            item_selection: روش انتخاب
        
        Returns:
            DataFrame با ستون‌های [time_col, item_col, count_col]
        """
        logger.info("="*70)
        logger.info("DATA PREPARATION")
        logger.info("="*70)
        
        # 1. فیلتر تاریخ
        df = self.filter_by_date(start_date, end_date)
        
        # 2. انتخاب آیتم‌ها
        items = self.get_item_list(df, num_items, item_selection)
        df = df[df[self.item_col].isin(items)]
        logger.info(f"Filtered to {len(items):,} items: {len(df):,} records")
        
        # 3. تجمیع روزانه
        daily_data = self.aggregate_by_day(df)
        
        # 4. آمار نهایی
        logger.info("="*70)
        logger.info(f"Dataset:         {self.dataset_name}")
        logger.info(f"Items:           {daily_data[self.item_col].nunique():,}")
        logger.info(f"Days:            {daily_data[self.time_col].nunique():,}")
        logger.info(f"Records:         {len(daily_data):,}")
        logger.info(
            f"Date range:      {daily_data[self.time_col].min()} "
            f"to {daily_data[self.time_col].max()}"
        )
        logger.info("="*70 + "\n")
        
        return daily_data
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        اعتبارسنجی داده‌ها
        
        Args:
            df: DataFrame برای اعتبارسنجی
        
        Returns:
            True اگر معتبر باشد
        
        Raises:
            ValueError: اگر داده نامعتبر باشد
        """
        # بررسی ستون‌های ضروری
        required_cols = [self.time_col, self.item_col, self.count_col]
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # بررسی خالی نبودن
        if len(df) == 0:
            raise ValueError("Empty dataset")
        
        # بررسی timestamp
        if not pd.api.types.is_datetime64_any_dtype(df[self.time_col]):
            raise ValueError(f"{self.time_col} column must be datetime")
        
        # بررسی مقادیر null
        null_counts = df[required_cols].isnull().sum()
        if null_counts.any():
            raise ValueError(f"Null values found: {null_counts[null_counts > 0]}")
        
        logger.info("✓ Data validation passed")
        return True
    
    def get_statistics(self) -> dict:
        """
        دریافت آمار کلی دیتاست
        
        Returns:
            dict با آمار مختلف
        """
        if self.data is None:
            self.data = self.load_data()
        
        stats = {
            'total_records': len(self.data),
            'unique_items': self.data[self.item_col].nunique(),
            'date_range': self.get_date_range(),
            'duration_days': (
                self.data[self.time_col].max() - self.data[self.time_col].min()
            ).days,
        }
        
        return stats
