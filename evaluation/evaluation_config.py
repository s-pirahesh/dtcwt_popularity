# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Configuration
Supports all datasets with adaptive parameters
Author: Sajjad
Date: February 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
from datetime import timedelta


@dataclass
class EvaluationConfig:
    """
    پیکربندی جامع برای ارزیابی زمانی
    سازگار با همه دیتاست‌ها (MovieLens, YouTube, Youku, NYC Taxi, ...)
    """
    
    # === پارامترهای زمانی ===
    start_date: Optional[str] = None        # 'YYYY-MM-DD' یا None (از ابتدا)
    end_date: Optional[str] = None          # 'YYYY-MM-DD' یا None (تا انتها)
    
    # Time Granularity (اندازه یک time slot)
    # مثال:
    #   MovieLens daily: timedelta(days=1)
    #   MovieLens weekly: timedelta(days=7)
    #   Youku (5-min): timedelta(minutes=5)
    #   NYC Yellow Taxi (15-min): timedelta(minutes=15)
    time_granularity: str = 'daily'        # 'daily', 'hourly', 'minute', یا custom
    slot_duration_minutes: Optional[int] = None  # برای custom granularity
    
    # Window Parameters (به تعداد time slots)
    window_size: int = 30                   # تعداد time slots برای training
    prediction_horizon: int = 7             # (legacy) فقط برای compatibility
    step_size: int = 1                      # گام لغزش (همیشه 1 slot)
    use_pre_range_data: bool = True         # استفاده از داده قبل از start_date
    
    # === پارامترهای آیتم ===
    num_items: Optional[int] = None         # تعداد آیتم‌ها (None = همه)
    item_selection: str = 'top'             # 'top', 'random', 'stratified'
    min_observations: int = 10              # حداقل مشاهدات برای یک آیتم
    
    # === پارامترهای Stratification ===
    strata_thresholds: Optional[List[int]] = None  # [Q1, Q2, Q3] یا None (خودکار)
    strata_names: List[str] = field(default_factory=lambda: ['cold_start', 'low', 'medium', 'high'])
    
    # === پارامترهای روش‌ها ===
    methods: Optional[List[str]] = None     # لیست روش‌ها یا None (همه)
    
    # === پارامترهای Wavelet (مهم!) ===
    wavelet_config: Dict = field(default_factory=lambda: {
        'dwt': {
            'wavelet': 'db4',
            'level': 'auto',               # auto = log2(window_size)
            'mode': 'symmetric'
        },
        'dtcwt': {
            'biort': 'near_sym_a',
            'qshift': 'qshift_a',
            'level': 'auto',               # auto = log2(window_size) - 1
        }
    })
    
    # === پارامترهای ذخیره‌سازی ===
    # نتایج میانی: همیشه Parquet (غیرقابل تغییر - حجم زیاد)
    # شامل: detailed scores, stratum summaries
    
    # نتایج نهایی: قابل انتخاب
    # شامل: method comparison, final reports
    final_format: str = 'csv'                  # فرمت نتایج نهایی: 'csv' or 'parquet'
    compression: str = 'snappy'                # فشرده‌سازی Parquet: 'snappy', 'gzip', 'brotli'
    save_detailed: bool = True                 # ذخیره نتایج تفصیلی (Parquet)
    save_summary: bool = True                  # ذخیره خلاصه (Parquet)
    output_dir: Optional[Path] = None          # دایرکتوری خروجی
    
    # === پارامترهای Performance ===
    parallel: bool = True                  # پردازش موازی
    num_cores: int = -1                    # -1 = همه هسته‌ها
    batch_size: int = 100                  # تعداد آیتم‌ها در هر batch
    
    # === پارامترهای Logging ===
    verbose: bool = True
    progress_bar: bool = True
    log_interval: int = 100                # هر 100 پنجره log کن
    
    # === Dataset name ===
    dataset_name: str = 'movielens'
    
    # === Run naming ===
    run_name: Optional[str] = None          # نام دلخواه برای این run (یا None برای خودکار)
    use_timestamp: bool = True              # اضافه کردن timestamp به نام

    # === Frozen Evaluation Protocol — Decision Layer ===
    # K values used for NDCG@K, Coverage@K, and RSI@K
    k_list: List[int] = field(default_factory=lambda: [5, 10, 20])

    # === Frozen Evaluation Protocol — Robustness Layer ===
    # Number of stable low-popularity items selected for noise injection
    robustness_sample_size: int = 50
    # Spike magnitude relative to item's historical mean (10x = 1000% increase)
    spike_multiplier: float = 10.0
    
    def __post_init__(self):
        """محاسبه خودکار پارامترها و اعتبارسنجی"""
        
        # 1. تنظیم خودکار wavelet levels
        if self.wavelet_config['dwt']['level'] == 'auto':
            max_level = int(np.log2(self.window_size))
            # محافظه‌کارانه: 1 level کمتر از maximum
            self.wavelet_config['dwt']['level'] = min(max_level - 1, 5)
            self.wavelet_config['dwt']['level'] = max(self.wavelet_config['dwt']['level'], 2)
        
        if self.wavelet_config['dtcwt']['level'] == 'auto':
            max_level = int(np.log2(self.window_size)) - 1
            # محافظه‌کارانه: 1 level کمتر از maximum
            self.wavelet_config['dtcwt']['level'] = min(max_level - 1, 4)
            self.wavelet_config['dtcwt']['level'] = max(self.wavelet_config['dtcwt']['level'], 2)
        
        # 2. اعتبارسنجی window_size
        self._validate_window_size()
        
        # 3. تنظیم output directory با run name
        if self.output_dir is None:
            from pathlib import Path
            from datetime import datetime
            
            # ایجاد نام منحصر به فرد برای این run
            if self.run_name is None:
                # نام خودکار بر اساس پارامترها
                self.run_name = self._generate_run_name()
            
            # اضافه کردن timestamp
            if self.use_timestamp:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                run_dir_name = f"{self.run_name}_{timestamp}"
            else:
                run_dir_name = self.run_name
            
            self.output_dir = Path(__file__).parent.parent / 'results' / self.dataset_name / run_dir_name
        
        # 4. ایجاد دایرکتوری‌ها
        self._create_directories()
        
        # 5. اعتبارسنجی پارامترها
        self._validate_parameters()
    
    def _validate_window_size(self):
        """بررسی اینکه window_size برای wavelet کافی است"""
        
        # بررسی DWT
        dwt_level = self.wavelet_config['dwt']['level']
        min_size_dwt = 2 ** (dwt_level + 1)
        if self.window_size < min_size_dwt:
            raise ValueError(
                f"Window size ({self.window_size}) too small for DWT level {dwt_level}. "
                f"Minimum required: {min_size_dwt}"
            )
        
        # بررسی DTCWT
        dtcwt_level = self.wavelet_config['dtcwt']['level']
        min_size_dtcwt = 2 ** (dtcwt_level + 2)
        if self.window_size < min_size_dtcwt:
            raise ValueError(
                f"Window size ({self.window_size}) too small for DTCWT level {dtcwt_level}. "
                f"Minimum required: {min_size_dtcwt}"
            )
    
    def _create_directories(self):
        """ایجاد ساختار دایرکتوری‌ها"""
        
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        if self.save_detailed:
            (self.output_dir / 'detailed').mkdir(exist_ok=True)
        
        if self.save_summary:
            (self.output_dir / 'summary').mkdir(exist_ok=True)
        
        (self.output_dir / 'comparison').mkdir(exist_ok=True)
        (self.output_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_dir / 'visualization').mkdir(exist_ok=True)
    
    def _validate_parameters(self):
        """اعتبارسنجی پارامترها"""
        
        if self.window_size < 4:
            raise ValueError("window_size must be at least 4")
        
        if self.prediction_horizon < 1:
            raise ValueError("prediction_horizon must be at least 1")
        
        if self.step_size != 1:
            raise ValueError("step_size must be 1 for true sliding window")
        
        if self.final_format not in ['csv', 'parquet']:
            raise ValueError(f"Invalid final_format: {self.final_format}. Must be 'csv' or 'parquet'")
        
        if self.item_selection not in ['top', 'random', 'stratified']:
            raise ValueError(f"Invalid item_selection: {self.item_selection}")
    
    def _generate_run_name(self) -> str:
        """
        ایجاد نام توصیفی برای run بر اساس پارامترها
        
        Returns:
            نام run (مثلاً: w30_h7_n1000_top)
        """
        parts = []
        
        # window size
        parts.append(f"w{self.window_size}")
        
        # horizon
        parts.append(f"h{self.prediction_horizon}")
        
        # num items
        if self.num_items is not None:
            parts.append(f"n{self.num_items}")
        else:
            parts.append("nall")
        
        # item selection
        # Explicit abbreviation — avoid 'str' which looks like Python builtin
        _abbrev = {'top': 'top', 'random': 'rnd', 'stratified': 'strat'}
        parts.append(_abbrev.get(self.item_selection, self.item_selection[:4]))
        
        return "_".join(parts)
    
    def get_num_windows(self, total_days: int) -> int:
        """محاسبه تعداد پنجره‌های ممکن"""
        return max(0, total_days - self.window_size - self.prediction_horizon + 1)
    
    def to_dict(self) -> dict:
        """تبدیل به dictionary برای ذخیره"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'window_size': self.window_size,
            'prediction_horizon': self.prediction_horizon,
            'step_size': self.step_size,
            'num_items': self.num_items,
            'item_selection': self.item_selection,
            'min_observations': self.min_observations,
            'strata_thresholds': self.strata_thresholds,
            'strata_names': self.strata_names,
            'methods': self.methods,
            'wavelet_config': self.wavelet_config,
            'final_format': self.final_format,
            'compression': self.compression,
            'dataset_name': self.dataset_name,
            # Frozen Evaluation Protocol
            'k_list': self.k_list,
            'robustness_sample_size': self.robustness_sample_size,
            'spike_multiplier': self.spike_multiplier,
        }
    
    def save_config(self, filepath: Path):
        """ذخیره پیکربندی"""
        import json
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def __repr__(self):
        """نمایش خلاصه پیکربندی"""
        return (
            f"EvaluationConfig(\n"
            f"  Dataset:          {self.dataset_name}\n"
            f"  Time Range:       {self.start_date} to {self.end_date}\n"
            f"  Window:           {self.window_size} days, Horizon: {self.prediction_horizon} days\n"
            f"  Items:            {self.num_items or 'all'}, Selection: {self.item_selection}\n"
            f"  DWT Level:        {self.wavelet_config['dwt']['level']}, "
            f"DTCWT Level: {self.wavelet_config['dtcwt']['level']}\n"
            f"  Output:           {self.final_format} ({self.compression})\n"
            f"  Parallel:         {self.parallel} (cores: {self.num_cores})\n"
            f"  --- Frozen Evaluation Protocol ---\n"
            f"  K values:         {self.k_list}\n"
            f"  Robustness:       {self.robustness_sample_size} items, "
            f"{self.spike_multiplier}x spike\n"
            f")"
        )
    
    def get_slot_duration(self) -> timedelta:
        """
        محاسبه مدت زمان یک time slot
        
        Returns:
            timedelta object representing one time slot
        """
        if self.time_granularity == 'daily':
            return timedelta(days=1)
        elif self.time_granularity == 'hourly':
            return timedelta(hours=1)
        elif self.time_granularity == 'minute':
            return timedelta(minutes=1)
        elif self.time_granularity == 'weekly':
            return timedelta(days=7)
        elif self.time_granularity == 'custom':
            if self.slot_duration_minutes is None:
                raise ValueError("slot_duration_minutes must be set for custom granularity")
            return timedelta(minutes=self.slot_duration_minutes)
        else:
            raise ValueError(f"Unknown time_granularity: {self.time_granularity}")


# پیکربندی‌های پیش‌فرض برای دیتاست‌های مختلف

def get_movielens_config(**kwargs) -> EvaluationConfig:
    """پیکربندی پیش‌فرض برای MovieLens"""
    defaults = {
        'dataset_name': 'movielens',
        'window_size': 30,
        'prediction_horizon': 7,
        # Thresholds in mean ratings/day (after mean-per-slot fix):
        #   cold_start : < 1   rating/day  (rarely-rated, new items)
        #   low        : 1-5   ratings/day
        #   medium     : 5-20  ratings/day
        #   high       : >= 20 ratings/day (consistently popular)
        # 'strata_thresholds': [1, 5, 20],
        'strata_thresholds': None,
        'min_observations': 10,
    }
    defaults.update(kwargs)
    return EvaluationConfig(**defaults)

def get_yellow_taxi_config(**kwargs) -> EvaluationConfig:
    """
    Configuration for NYC Yellow Taxi dataset (hourly granularity).

    Usage:
        config = get_yellow_taxi_config(
            num_items=50,
            window_size=168,             # 7 days × 24 hours
            start_date='2014-01-01',
            end_date='2014-06-30',
        )

    Strata thresholds (mean trips/hour per zone, hourly data):
        cold_start : <  5   low-traffic outer-borough zones
        low        :  5–30  mixed-use neighbourhoods
        medium     : 30–100 busy commercial / entertainment areas
        high       : ≥ 100  major hubs (Midtown, Penn Station, JFK, LGA)

    These values are calibrated from the NYC TLC hourly distribution
    where the most active zone peaks near 230 trips/hour and roughly
    7 % of zones exceed 100 trips/hour.

    Pass strata_thresholds=None to auto-compute Q1/Q2/Q3 from the data.
    This always produces 4 non-empty strata regardless of the date range
    or subset of zones, at the cost of less interpretable boundaries.

    Note: window_size is in TIME SLOTS (hours for hourly data).

    Data source: NYC Taxi & Limousine Commission, Yellow Taxi Trip Records.
        https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    """
    defaults = {
        'dataset_name':      'yellow_taxi',
        'time_granularity':  'hourly',
        'window_size':       168,        # 7 days × 24 h — sensible default
        'prediction_horizon': 24,        # predict next 24 hours
        # Fixed thresholds calibrated for NYC hourly data.
        # Set to None for automatic Q1/Q2/Q3 from the actual distribution.
        'strata_thresholds': None, 
        # 'strata_thresholds' : [2, 5, 100],
        'min_observations':  24,         # at least 1 full day of data
    }
    defaults.update(kwargs)
    return EvaluationConfig(**defaults)


# (youtube07 config removed — not yet implemented)

def get_youtube_config(**kwargs) -> EvaluationConfig:
    """
    Default config for YouTube (hourly granularity).
    Thresholds in mean views/hour per video:
      cold_start : < 50    views/hour
      low        : 50-500  views/hour
      medium     : 500-5000 views/hour
      high       : >= 5000 views/hour
    """
    defaults = {
        'dataset_name': 'youtube',
        'time_granularity': 'hourly',
        'window_size': 30,
        'prediction_horizon': 3,
        # 'strata_thresholds': [50, 500, 5000],
        'strata_thresholds': None,
        'min_observations': 50,
    }
    defaults.update(kwargs)
    return EvaluationConfig(**defaults)

def get_youku_config(**kwargs) -> EvaluationConfig:
    """پیکربندی پیش‌فرض برای Youku (5-minute granularity)"""
    defaults = {
        'dataset_name': 'youku',
        'time_granularity': 'custom',
        'slot_duration_minutes': 5,     # 5-minute slots
        'window_size': 288,              # 288 slots = 24 hours
        # Thresholds in mean views/5-min-slot:
        #   cold<50, low 50-500, medium 500-5000, high>=5000
        'strata_thresholds': [50, 500, 5000],
        'min_observations': 20,
    }
    defaults.update(kwargs)
    return EvaluationConfig(**defaults)
