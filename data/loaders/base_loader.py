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
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """
    کلاس پایه انتزاعی برای بارگذاری دیتاست‌ها
    
    همه data loaders باید از این کلاس ارث‌بری کنند و متدهای
    انتزاعی را پیاده‌سازی نمایند.
    """
    
    def __init__(self, data_dir: str, dataset_name: str):
        """
        Args:
            data_dir: مسیر دایرکتوری داده‌ها
            dataset_name: نام دیتاست
        """
        self.data_dir = Path(data_dir)
        self.dataset_name = dataset_name
        self.data: Optional[pd.DataFrame] = None
        
        logger.info(f"Initializing {self.dataset_name} loader")
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """
        بارگذاری داده‌های خام از فایل
        
        این متد باید:
        1. فایل داده را بخواند
        2. timestamp را به datetime تبدیل کند
        3. ستون‌های ضروری را داشته باشد: ['timestamp', 'item_id', 'user_id']
        
        Returns:
            DataFrame با ستون‌های [timestamp, item_id, user_id, ...]
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
            df = df[df['timestamp'] >= start_dt]
            logger.info(f"Filtered from {start_date}: {len(df):,} records")
        
        # فیلتر تاریخ پایان
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df['timestamp'] <= end_dt]
            logger.info(f"Filtered until {end_date}: {len(df):,} records")
        
        return df
    
    def aggregate_by_day(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        تجمیع داده‌ها به صورت روزانه
        
        Args:
            df: DataFrame با timestamp
        
        Returns:
            DataFrame با ستون‌های [date, item_id, count]
        """
        # استخراج تاریخ (بدون ساعت)
        df['date'] = df['timestamp'].dt.date
        
        # تجمیع روزانه
        daily = df.groupby(['date', 'item_id']).size().reset_index(name='count')
        
        # تبدیل date به datetime
        daily['date'] = pd.to_datetime(daily['date'])
        
        # مرتب‌سازی
        daily = daily.sort_values(['date', 'item_id']).reset_index(drop=True)
        
        logger.info(f"Aggregated to daily: {len(daily):,} records")
        logger.info(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
        logger.info(f"Unique items: {daily['item_id'].nunique():,}")
        
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
        item_counts = df['item_id'].value_counts()
        
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
            DataFrame با ستون‌های [date, item_id, count]
        """
        logger.info("="*70)
        logger.info("DATA PREPARATION")
        logger.info("="*70)
        
        # 1. فیلتر تاریخ
        df = self.filter_by_date(start_date, end_date)
        
        # 2. انتخاب آیتم‌ها
        items = self.get_item_list(df, num_items, item_selection)
        df = df[df['item_id'].isin(items)]
        logger.info(f"Filtered to {len(items):,} items: {len(df):,} records")
        
        # 3. تجمیع روزانه
        daily_data = self.aggregate_by_day(df)
        
        # 4. آمار نهایی
        logger.info("="*70)
        logger.info(f"Dataset:         {self.dataset_name}")
        logger.info(f"Items:           {daily_data['item_id'].nunique():,}")
        logger.info(f"Days:            {daily_data['date'].nunique():,}")
        logger.info(f"Records:         {len(daily_data):,}")
        logger.info(f"Date range:      {daily_data['date'].min()} to {daily_data['date'].max()}")
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
        required_cols = ['timestamp', 'item_id', 'user_id']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # بررسی خالی نبودن
        if len(df) == 0:
            raise ValueError("Empty dataset")
        
        # بررسی timestamp
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            raise ValueError("timestamp column must be datetime")
        
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
            'unique_users': self.data['user_id'].nunique(),
            'unique_items': self.data['item_id'].nunique(),
            'date_range': self.get_date_range(),
            'duration_days': (
                self.data['timestamp'].max() - self.data['timestamp'].min()
            ).days,
        }
        
        return stats
