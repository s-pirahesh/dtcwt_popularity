# -*- coding: utf-8 -*-
"""
Temporal Evaluator - Main Evaluation Engine
Complete sliding window evaluation with stratification
Author: Sajjad
Date: February 2025
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime, timedelta
import multiprocessing as mp
from functools import partial

from .evaluation_config import EvaluationConfig
from .stratification import StratificationSystem
from .metrics import MetricsCalculator
from .wavelet_validator import WaveletWindowValidator
from .storage import StorageSystem


class TemporalEvaluator:
    """
    ارزیابی زمانی کامل با sliding window
    
    Features:
    - True sliding window (step=1)
    - Automatic stratification
    - Comprehensive metrics
    - Parquet storage
    - Parallel processing
    """
    
    def __init__(self, data_loader, methods: Dict, config: EvaluationConfig):
        """
        Args:
            data_loader: DataLoader instance (MovieLensLoader, YouTubeLoader, etc.)
            methods: {method_name: method_instance}
            config: EvaluationConfig instance
        """
        self.data_loader = data_loader
        self.methods = methods
        self.config = config
        
        # سیستم‌های جانبی
        self.stratification = StratificationSystem(config)
        self.storage = StorageSystem(config)
        
        # اعتبارسنجی wavelet
        self._validate_wavelet_config()
        
        # آماده‌سازی داده
        self.data = None
        self.items = None
        self.num_windows = 0
        
        # آمار اجرا
        self.runtime_stats = {
            'start_time': None,
            'end_time': None,
            'total_windows_processed': 0,
            'total_items_processed': 0,
            'total_calculations': 0,
        }
    
    def _validate_wavelet_config(self):
        """اعتبارسنجی پیکربندی wavelet"""
        
        if self.config.verbose:
            print("\n" + "="*70)
            print("WAVELET CONFIGURATION VALIDATION")
            print("="*70)
        
        # DWT
        WaveletWindowValidator.print_validation_report(
            self.config.window_size,
            self.config.wavelet_config['dwt']['level'],
            'dwt'
        )
        
        # DTCWT
        WaveletWindowValidator.print_validation_report(
            self.config.window_size,
            self.config.wavelet_config['dtcwt']['level'],
            'dtcwt'
        )
    
    def prepare_data(self):
        """آماده‌سازی داده برای ارزیابی"""
        
        if self.config.verbose:
            print("\n" + "="*70)
            print("DATA PREPARATION")
            print("="*70)
        
        # 1. بارگذاری داده
        print("Loading data...")
        self.data = self.data_loader.load_data()
        
        if self.config.verbose:
            print(f"  Total records: {len(self.data):,}")
            print(f"  Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
        # 2. فیلتر زمانی
        if self.config.start_date or self.config.end_date:
            self.data = self._apply_time_filter(self.data)
            
            if self.config.verbose:
                print(f"  After time filter: {len(self.data):,} records")
        
        # 3. انتخاب آیتم‌ها
        self.items = self._select_items()
        
        if self.config.verbose:
            print(f"  Selected items: {len(self.items):,}")
        
        # 4. فیلتر داده فقط برای آیتم‌های انتخابی
        self.data = self.data[self.data['item_id'].isin(self.items)]
        
        if self.config.verbose:
            print(f"  Final records: {len(self.data):,}")
        
        # 5. محاسبه تعداد پنجره‌ها
        total_days = (self.data['timestamp'].max() - self.data['timestamp'].min()).days + 1
        self.num_windows = self.config.get_num_windows(total_days)
        
        if self.config.verbose:
            print(f"  Total days: {total_days}")
            print(f"  Number of windows: {self.num_windows:,}")
            print(f"  Total calculations: {self.num_windows * len(self.items) * len(self.methods):,}")
        
        print("="*70 + "\n")
    
    def _apply_time_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """اعمال فیلتر زمانی"""
        
        if self.config.start_date:
            start = pd.to_datetime(self.config.start_date)
            data = data[data['timestamp'] >= start]
        
        if self.config.end_date:
            end = pd.to_datetime(self.config.end_date)
            data = data[data['timestamp'] <= end]
        
        return data
    
    def _select_items(self) -> np.ndarray:
        """انتخاب آیتم‌ها بر اساس استراتژی"""
        
        # محاسبه تعداد دسترسی هر آیتم
        item_counts = self.data.groupby('item_id')['count'].sum()
        
        # فیلتر بر اساس حداقل مشاهدات
        item_counts = item_counts[item_counts >= self.config.min_observations]
        
        if self.config.num_items is None:
            # همه آیتم‌ها
            selected = item_counts.index.values
        
        elif self.config.item_selection == 'top':
            # Top-N آیتم‌ها
            selected = item_counts.nlargest(self.config.num_items).index.values
        
        elif self.config.item_selection == 'random':
            # Random N آیتم
            n = min(self.config.num_items, len(item_counts))
            selected = np.random.choice(item_counts.index.values, size=n, replace=False)
        
        elif self.config.item_selection == 'stratified':
            # Stratified sampling
            selected = self._stratified_sampling(item_counts, self.config.num_items)
        
        else:
            raise ValueError(f"Invalid item_selection: {self.config.item_selection}")
        
        return selected
    
    def _stratified_sampling(self, item_counts: pd.Series, n: int) -> np.ndarray:
        """نمونه‌برداری طبقه‌بندی شده"""
        
        # طبقه‌بندی اولیه
        strata = self.stratification.stratify_items(
            pd.DataFrame({'item_id': item_counts.index, 'count': item_counts.values})
        )
        
        # تعداد آیتم از هر stratum (متناسب با اندازه)
        total_items = sum(len(items) for items in strata.values())
        selected = []
        
        for stratum_name, stratum_items in strata.items():
            if len(stratum_items) == 0:
                continue
            
            # نسبت این stratum
            ratio = len(stratum_items) / total_items
            n_from_stratum = int(n * ratio)
            
            # نمونه‌برداری
            n_sample = min(n_from_stratum, len(stratum_items))
            sampled = np.random.choice(stratum_items, size=n_sample, replace=False)
            selected.extend(sampled)
        
        return np.array(selected)
    
    def evaluate(self):
        """اجرای ارزیابی کامل"""
        
        # 1. آماده‌سازی
        if self.data is None:
            self.prepare_data()
        
        # 2. شروع زمان‌سنجی
        self.runtime_stats['start_time'] = time.time()
        
        # 3. ارزیابی هر روش
        for method_name, method in self.methods.items():
            if self.config.verbose:
                print(f"\n{'='*70}")
                print(f"EVALUATING: {method_name}")
                print(f"{'='*70}")
            
            self._evaluate_method(method_name, method)
        
        # 4. پایان زمان‌سنجی
        self.runtime_stats['end_time'] = time.time()
        self.runtime_stats['total_duration'] = (
            self.runtime_stats['end_time'] - self.runtime_stats['start_time']
        )
        
        # 5. مقایسه روش‌ها
        self._compare_methods()
        
        # 6. ذخیره آمار
        self._save_runtime_stats()
        
        # 7. خلاصه
        if self.config.verbose:
            self._print_summary()
    
    def _evaluate_method(self, method_name: str, method):
        """
        ارزیابی یک روش در همه پنجره‌ها
        
        Args:
            method_name: نام روش
            method: instance روش
        """
        all_results = []
        stratum_summaries = []
        
        # محاسبه تاریخ شروع و پایان
        min_date = self.data['timestamp'].min()
        max_date = self.data['timestamp'].max()
        
        # پیشرفت
        if self.config.progress_bar:
            pbar = tqdm(total=self.num_windows, desc=f"{method_name}")
        
        # برای هر پنجره (true sliding window)
        for window_idx in range(self.num_windows):
            # محاسبه بازه زمانی این پنجره
            train_start = min_date + timedelta(days=window_idx)
            train_end = train_start + timedelta(days=self.config.window_size)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.prediction_horizon)
            
            # استخراج داده پنجره
            train_window = self.data[
                (self.data['timestamp'] >= train_start) & 
                (self.data['timestamp'] < train_end)
            ]
            
            test_window = self.data[
                (self.data['timestamp'] >= test_start) & 
                (self.data['timestamp'] < test_end)
            ]
            
            if len(train_window) == 0 or len(test_window) == 0:
                if self.config.progress_bar:
                    pbar.update(1)
                continue
            
            # طبقه‌بندی آیتم‌ها در این پنجره
            strata = self.stratification.stratify_items(train_window)
            
            # ارزیابی برای هر آیتم
            window_results = self._evaluate_window(
                method, method_name, window_idx,
                train_window, test_window,
                test_start, strata
            )
            
            all_results.extend(window_results)
            
            # خلاصه stratum
            stratum_summary = self._calculate_stratum_summary(
                window_results, window_idx, test_start
            )
            stratum_summaries.extend(stratum_summary)
            
            # به‌روزرسانی آمار
            self.runtime_stats['total_windows_processed'] += 1
            self.runtime_stats['total_items_processed'] += len(window_results)
            self.runtime_stats['total_calculations'] += 1
            
            # پیشرفت
            if self.config.progress_bar:
                pbar.update(1)
            
            # لاگ
            if self.config.verbose and window_idx % self.config.log_interval == 0:
                print(f"  Window {window_idx}/{self.num_windows}: {len(window_results)} items")
        
        if self.config.progress_bar:
            pbar.close()
        
        # ذخیره نتایج
        if self.config.verbose:
            print(f"  Saving results...")
        
        self.storage.save_detailed_scores(method_name, all_results)
        self.storage.save_stratum_summary(method_name, stratum_summaries)
        
        if self.config.verbose:
            print(f"  Completed: {len(all_results):,} total records")
    
    def _evaluate_window(self, method, method_name: str, window_idx: int,
                        train_window: pd.DataFrame, test_window: pd.DataFrame,
                        test_start: datetime, strata: Dict) -> List[Dict]:
        """
        ارزیابی یک پنجره برای همه آیتم‌ها
        
        Args:
            method: روش ارزیابی
            method_name: نام روش
            window_idx: شماره پنجره
            train_window: داده آموزش
            test_window: داده تست
            test_start: تاریخ شروع تست
            strata: طبقه‌بندی آیتم‌ها
        
        Returns:
            لیست نتایج برای هر آیتم
        """
        results = []
        
        # تبدیل به timestamp (milliseconds)
        timestamp_ms = int(test_start.timestamp() * 1000)
        
        # محاسبه امتیاز برای هر آیتم
        for item_id in self.items:
            # داده این آیتم در پنجره آموزش
            item_train = train_window[train_window['item_id'] == item_id]
            
            if len(item_train) < self.config.min_observations:
                continue
            
            # محاسبه امتیاز (assessment, not prediction)
            try:
                # برای روش‌های مختلف
                if hasattr(method, 'assess_single'):
                    # DTCWT, DWT
                    time_series = item_train['count'].values
                    score = method.assess_single(time_series)
                
                elif hasattr(method, 'calculate'):
                    # Baselines (AF, LFU, etc.)
                    score = method.calculate(item_train['count'].values)
                
                else:
                    # سایر روش‌ها
                    score = method.predict(item_train)
                
                # اگر score لیست است، میانگین بگیر
                if isinstance(score, (list, np.ndarray)):
                    score = float(np.mean(score))
                else:
                    score = float(score)
            
            except Exception as e:
                if self.config.verbose:
                    print(f"    Warning: Failed to score item {item_id}: {e}")
                score = 0.0
            
            # دسترسی واقعی در پنجره تست
            item_test = test_window[test_window['item_id'] == item_id]
            actual_count = item_test['count'].sum() if len(item_test) > 0 else 0
            
            # تعیین stratum
            train_count = item_train['count'].sum()
            stratum_label = self.stratification.get_stratum_label(train_count)
            
            # ذخیره نتیجه
            result = {
                'window_id': window_idx,
                'timestamp': timestamp_ms,
                'item_id': int(item_id),
                'stratum': stratum_label,
                'popularity_score': score,
                'actual_count': int(actual_count),
                'train_count': int(train_count),
            }
            
            results.append(result)
        
        # محاسبه معیارها برای این پنجره
        if len(results) > 0:
            scores = np.array([r['popularity_score'] for r in results])
            actuals = np.array([r['actual_count'] for r in results])
            
            metrics = MetricsCalculator.calculate_all_metrics(scores, actuals)
            
            # اضافه کردن معیارها به هر رکورد
            for i, result in enumerate(results):
                result['mae'] = metrics['mae']
                result['mape'] = metrics['mape']
                result['rank_predicted'] = int(metrics['rank_predicted'][i])
                result['rank_actual'] = int(metrics['rank_actual'][i])
        
        return results
    
    def _calculate_stratum_summary(self, window_results: List[Dict],
                                   window_idx: int, timestamp: datetime) -> List[Dict]:
        """
        محاسبه خلاصه برای هر stratum در این پنجره
        
        Args:
            window_results: نتایج این پنجره
            window_idx: شماره پنجره
            timestamp: timestamp این پنجره
        
        Returns:
            لیست خلاصه برای هر stratum
        """
        if len(window_results) == 0:
            return []
        
        df = pd.DataFrame(window_results)
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        summaries = []
        
        # برای هر stratum
        for stratum_label in range(4):  # 0, 1, 2, 3
            stratum_data = df[df['stratum'] == stratum_label]
            
            if len(stratum_data) == 0:
                continue
            
            scores = stratum_data['popularity_score'].values
            actuals = stratum_data['actual_count'].values
            
            metrics = MetricsCalculator.calculate_all_metrics(scores, actuals)
            
            summary = {
                'window_id': window_idx,
                'timestamp': timestamp_ms,
                'stratum': stratum_label,
                'stratum_name': self.stratification.get_stratum_name(stratum_label),
                'num_items': len(stratum_data),
                'mean_mae': metrics['mae'],
                'mean_mape': metrics['mape'],
                'spearman_corr': metrics['spearman'],
                'kendall_tau': metrics['kendall'],
                'ndcg': metrics['ndcg'],
                'coverage': metrics['coverage'],
            }
            
            summaries.append(summary)
        
        return summaries
    
    def _compare_methods(self):
        """مقایسه همه روش‌ها"""
        
        if self.config.verbose:
            print(f"\n{'='*70}")
            print("COMPARING METHODS")
            print(f"{'='*70}")
        
        # بارگذاری نتایج همه روش‌ها
        method_summaries = {}
        
        for method_name in self.methods.keys():
            try:
                summary = self.storage.load_stratum_summary(method_name)
                method_summaries[method_name] = summary
            except FileNotFoundError:
                if self.config.verbose:
                    print(f"  Warning: No summary found for {method_name}")
        
        if len(method_summaries) == 0:
            return
        
        # مقایسه کلی (میانگین روی همه پنجره‌ها و strata)
        comparison_data = []
        
        for method_name, summary_df in method_summaries.items():
            try:
                comparison_data.append({
                    'method': method_name,
                    'mean_mae': summary_df['mean_mae'].mean(),
                    'mean_mape': summary_df['mean_mape'].mean(),
                    'mean_spearman': summary_df['spearman_corr'].mean(),
                    'mean_kendall': summary_df['kendall_tau'].mean(),
                    'mean_ndcg': summary_df['ndcg'].mean(),
                    'mean_coverage': summary_df['coverage'].mean(),
                })
            except KeyError as e:
                if self.config.verbose:
                    print(f"  Warning: {method_name} summary missing column {e}")
                    print(f"  Available columns: {list(summary_df.columns)}")
                    print(f"  Please delete 'results/' folder and re-run to regenerate with correct format")
                continue
        
        comparison_df = pd.DataFrame(comparison_data)
        
        if len(comparison_df) == 0:
            if self.config.verbose:
                print("  No valid method summaries found for comparison")
                print("  Please delete 'results/' folder and re-run")
            return
        
        comparison_df = comparison_df.sort_values('mean_spearman', ascending=False)
        
        # ذخیره
        self.storage.save_method_comparison(comparison_df)
        
        # چاپ
        if self.config.verbose:
            print("\nOverall Performance (averaged across all windows & strata):")
            print(comparison_df.to_string(index=False))
            print(f"{'='*70}\n")
    
    def _save_runtime_stats(self):
        """ذخیره آمار زمان اجرا"""
        
        stats = {
            **self.runtime_stats,
            'config': self.config.to_dict(),
            'num_methods': len(self.methods),
            'num_items': len(self.items),
            'num_windows': self.num_windows,
            'avg_time_per_window': (
                self.runtime_stats['total_duration'] / self.num_windows
                if self.num_windows > 0 else 0
            ),
        }
        
        self.storage.save_runtime_stats(stats)
        self.config.save_config(self.config.output_dir / 'metadata' / 'config.json')
        self.stratification.save_thresholds(
            self.config.output_dir / 'metadata' / 'thresholds.json'
        )
    
    def _print_summary(self):
        """چاپ خلاصه نهایی"""
        
        duration = self.runtime_stats['total_duration']
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        print(f"\n{'='*70}")
        print("EVALUATION COMPLETED")
        print(f"{'='*70}")
        print(f"Total Duration:        {hours}h {minutes}m {seconds}s")
        print(f"Windows Processed:     {self.runtime_stats['total_windows_processed']:,}")
        print(f"Items Processed:       {self.runtime_stats['total_items_processed']:,}")
        print(f"Total Calculations:    {self.runtime_stats['total_calculations']:,}")
        print(f"Output Directory:      {self.config.output_dir}")
        print(f"{'='*70}\n")
        
        # اندازه فایل‌ها
        self.storage.print_storage_summary()
