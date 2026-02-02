# -*- coding: utf-8 -*-
"""
Results Analyzer - Analysis of Saved Results
Analyze evaluation results without re-running simulation
Author: Sajjad
Date: February 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json


class ResultsAnalyzer:
    """
    تحلیل نتایج ذخیره شده بدون شبیه‌سازی مجدد
    
    Features:
    - بارگذاری نتایج از Parquet/CSV
    - تحلیل برای همه داده‌ها یا بخش‌های خاص
    - مقایسه روش‌ها
    - خلاصه آماری
    - فیلتر بر اساس stratum, time range, top-k
    """
    
    def __init__(self, run_dir: Path):
        """
        Args:
            run_dir: مسیر دایرکتوری نتایج (مثلاً results/movielens/run_20250201_143052/)
        """
        self.run_dir = Path(run_dir)
        
        if not self.run_dir.exists():
            raise ValueError(f"Run directory not found: {run_dir}")
        
        # بارگذاری metadata
        self.config = self._load_config()
        self.thresholds = self._load_thresholds()
        self.runtime_stats = self._load_runtime_stats()
        
        # ذخیره روش‌های موجود
        self.available_methods = self._detect_methods()
        
        # cache برای نتایج بارگذاری شده
        self._cache = {}
    
    def _load_config(self) -> dict:
        """بارگذاری پیکربندی"""
        config_path = self.run_dir / 'metadata' / 'config.json'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_thresholds(self) -> dict:
        """بارگذاری thresholds"""
        thresh_path = self.run_dir / 'metadata' / 'thresholds.json'
        if thresh_path.exists():
            with open(thresh_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_runtime_stats(self) -> dict:
        """بارگذاری آمار زمان اجرا"""
        stats_path = self.run_dir / 'metadata' / 'runtime_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _detect_methods(self) -> List[str]:
        """شناسایی روش‌های موجود"""
        methods = []
        detailed_dir = self.run_dir / 'detailed'
        
        if not detailed_dir.exists():
            return methods
        
        # نتایج میانی همیشه Parquet
        for file in detailed_dir.glob('*_scores.parquet'):
            method_name = file.stem.replace('_scores', '')
            methods.append(method_name)
        
        return sorted(methods)
    
    def load_detailed_scores(self, method_name: str, 
                           use_cache: bool = True) -> pd.DataFrame:
        """
        بارگذاری امتیازهای تفصیلی یک روش
        نتایج میانی همیشه Parquet
        
        Args:
            method_name: نام روش
            use_cache: استفاده از cache
        
        Returns:
            DataFrame
        """
        if use_cache and method_name in self._cache:
            return self._cache[method_name]
        
        # نتایج میانی همیشه Parquet
        filename = f'{method_name}_scores.parquet'
        filepath = self.run_dir / 'detailed' / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results not found for method: {method_name}")
        
        df = pd.read_parquet(filepath)
        
        # تبدیل timestamp به datetime
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        if use_cache:
            self._cache[method_name] = df
        
        return df
    
    def load_stratum_summary(self, method_name: str) -> pd.DataFrame:
        """
        بارگذاری خلاصه stratum
        نتایج میانی همیشه Parquet
        """
        filename = f'{method_name}_stratum_summary.parquet'
        filepath = self.run_dir / 'summary' / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Summary not found for method: {method_name}")
        
        df = pd.read_parquet(filepath)
        
        # تبدیل timestamp
        if 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def load_method_comparison(self) -> pd.DataFrame:
        """
        بارگذاری مقایسه روش‌ها - خودکار (CSV یا Parquet)
        
        Returns:
            DataFrame
        """
        # سعی کردن هر دو فرمت
        for fmt in ['csv', 'parquet']:
            filename = f"method_comparison.{fmt}"
            filepath = self.run_dir / 'comparison' / filename
            
            if filepath.exists():
                if fmt == 'csv':
                    return pd.read_csv(filepath, encoding='utf-8')
                else:
                    return pd.read_parquet(filepath)
        
        raise FileNotFoundError("Method comparison file not found (tried both CSV and Parquet)")
    
    def filter_by_percentile(self, df: pd.DataFrame, 
                            top_percent: float = 20.0) -> pd.DataFrame:
        """
        فیلتر top-k% بر اساس actual_count
        
        Args:
            df: DataFrame
            top_percent: درصد بالایی (مثلاً 20 برای top 20%)
        
        Returns:
            DataFrame فیلتر شده
        """
        # محاسبه threshold
        threshold = df.groupby('item_id')['actual_count'].sum().quantile(
            1 - top_percent / 100
        )
        
        # آیتم‌های بالای threshold
        top_items = df.groupby('item_id')['actual_count'].sum()
        top_items = top_items[top_items >= threshold].index
        
        return df[df['item_id'].isin(top_items)]
    
    def filter_by_stratum(self, df: pd.DataFrame, 
                         stratum: str) -> pd.DataFrame:
        """
        فیلتر بر اساس stratum
        
        Args:
            df: DataFrame
            stratum: 'cold_start', 'low', 'medium', 'high'
        
        Returns:
            DataFrame فیلتر شده
        """
        stratum_map = {'cold_start': 0, 'low': 1, 'medium': 2, 'high': 3}
        stratum_id = stratum_map.get(stratum)
        
        if stratum_id is None:
            raise ValueError(f"Invalid stratum: {stratum}")
        
        return df[df['stratum'] == stratum_id]
    
    def filter_by_time_range(self, df: pd.DataFrame,
                            start_date: str = None,
                            end_date: str = None) -> pd.DataFrame:
        """
        فیلتر بر اساس بازه زمانی
        
        Args:
            df: DataFrame
            start_date: تاریخ شروع (YYYY-MM-DD)
            end_date: تاریخ پایان (YYYY-MM-DD)
        
        Returns:
            DataFrame فیلتر شده
        """
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def calculate_overall_metrics(self, method_name: str,
                                 filter_top_percent: Optional[float] = None,
                                 filter_stratum: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> Dict:
        """
        محاسبه معیارهای کلی با فیلترهای دلخواه
        
        Args:
            method_name: نام روش
            filter_top_percent: فیلتر top-k% (None = همه)
            filter_stratum: فیلتر stratum (None = همه)
            start_date: تاریخ شروع (None = از ابتدا)
            end_date: تاریخ پایان (None = تا انتها)
        
        Returns:
            dict حاوی معیارها
        """
        # بارگذاری داده
        df = self.load_detailed_scores(method_name)
        
        # اعمال فیلترها
        if filter_top_percent:
            df = self.filter_by_percentile(df, filter_top_percent)
        
        if filter_stratum:
            df = self.filter_by_stratum(df, filter_stratum)
        
        if start_date or end_date:
            df = self.filter_by_time_range(df, start_date, end_date)
        
        if len(df) == 0:
            return {}
        
        # محاسبه معیارها
        from .metrics import MetricsCalculator
        
        scores = df['popularity_score'].values
        actuals = df['actual_count'].values
        
        metrics = MetricsCalculator.calculate_all_metrics(scores, actuals)
        
        # حذف arrays (فقط scalars)
        return {
            'spearman': metrics['spearman'],
            'kendall': metrics['kendall'],
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'mape': metrics['mape'],
            'ndcg': metrics['ndcg'],
            'coverage': metrics['coverage'],
            'num_samples': len(df),
        }
    
    def compare_methods(self, 
                       filter_top_percent: Optional[float] = None,
                       filter_stratum: Optional[str] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        مقایسه همه روش‌ها با فیلترهای دلخواه
        
        Args:
            filter_top_percent: فیلتر top-k%
            filter_stratum: فیلتر stratum
            start_date: تاریخ شروع
            end_date: تاریخ پایان
        
        Returns:
            DataFrame مقایسه
        """
        results = []
        
        for method_name in self.available_methods:
            metrics = self.calculate_overall_metrics(
                method_name,
                filter_top_percent=filter_top_percent,
                filter_stratum=filter_stratum,
                start_date=start_date,
                end_date=end_date
            )
            
            if metrics:
                results.append({
                    'method': method_name,
                    **metrics
                })
        
        df = pd.DataFrame(results)
        
        if len(df) > 0:
            # مرتب‌سازی بر اساس spearman
            df = df.sort_values('spearman', ascending=False)
        
        return df
    
    def get_temporal_evolution(self, method_name: str,
                              metric: str = 'mae',
                              window_agg: str = 'mean') -> pd.DataFrame:
        """
        تکامل زمانی یک معیار
        
        Args:
            method_name: نام روش
            metric: نام معیار
            window_agg: نحوه aggregation ('mean', 'median', 'std')
        
        Returns:
            DataFrame با timestamp و metric value
        """
        df = self.load_detailed_scores(method_name)
        
        if metric not in df.columns:
            raise ValueError(f"Metric not found: {metric}")
        
        # گروه‌بندی بر اساس window
        if window_agg == 'mean':
            result = df.groupby('window_id')[metric].mean()
        elif window_agg == 'median':
            result = df.groupby('window_id')[metric].median()
        elif window_agg == 'std':
            result = df.groupby('window_id')[metric].std()
        else:
            raise ValueError(f"Invalid aggregation: {window_agg}")
        
        # اضافه کردن timestamp
        timestamps = df.groupby('window_id')['date'].first()
        
        result_df = pd.DataFrame({
            'date': timestamps,
            metric: result
        })
        
        return result_df.reset_index(drop=True)
    
    def get_stratum_comparison(self, method_name: str,
                              metric: str = 'spearman_corr') -> pd.DataFrame:
        """
        مقایسه عملکرد در strata مختلف
        
        Args:
            method_name: نام روش
            metric: نام معیار
        
        Returns:
            DataFrame با stratum و metric value
        """
        summary = self.load_stratum_summary(method_name)
        
        if metric not in summary.columns:
            raise ValueError(f"Metric not found: {metric}")
        
        # میانگین در هر stratum
        result = summary.groupby('stratum_name')[metric].agg(['mean', 'std', 'min', 'max'])
        
        return result.reset_index()
    
    def print_summary(self):
        """چاپ خلاصه این run"""
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(f"Run Directory:     {self.run_dir}")
        print(f"Dataset:           {self.config.get('dataset_name', 'N/A')}")
        print(f"Window Size:       {self.config.get('window_size', 'N/A')} days")
        print(f"Prediction Horizon: {self.config.get('prediction_horizon', 'N/A')} days")
        print(f"Number of Items:   {self.config.get('num_items', 'all')}")
        print(f"Available Methods: {', '.join(self.available_methods)}")
        
        if self.runtime_stats:
            duration = self.runtime_stats.get('total_duration', 0)
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            print(f"Total Duration:    {hours}h {minutes}m")
        
        print("="*70 + "\n")
