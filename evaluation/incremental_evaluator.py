"""
Incremental Temporal Evaluator
ارزیابی temporal با ذخیره تدریجی و method-specific window sizes

Author: Sajjad
Date: February 2025
"""

import gc
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tqdm import tqdm

from .evaluation_config import EvaluationConfig
from .incremental_storage import IncrementalStorage
from .method_configs import get_method_config, METHOD_CONFIGS
from .time_utils import create_time_helper, TimeSlotHelper
from .metrics import MetricsCalculator
from .stratification import StratificationSystem


class IncrementalTemporalEvaluator:
    """
    ارزیابی temporal با incremental saving
    
    مزایا:
    - Memory کم (<200 MB)
    - Crash-safe (ذخیره مداوم)
    - Method-specific window sizes
    - Progress tracking
    """
    
    def __init__(self,
                 config: EvaluationConfig,
                 methods: Dict,
                 data: pd.DataFrame,
                 items: np.ndarray,
                 storage_path: Path):
        """
        Args:
            config: تنظیمات evaluation
            methods: dict از methods
            data: داده اصلی
            items: لیست item_ids
            storage_path: مسیر ذخیره
        """
        self.config = config
        self.methods = methods
        self.data = data
        self.items = items
        self.storage_path = Path(storage_path)
        
        # Incremental storage
        self.storage = IncrementalStorage(
            self.storage_path,
            buffer_size=1000
        )
        
        # Stratification
        self.stratification = StratificationSystem(config)
        
        # محاسبه thresholds بر اساس کل داده
        self._initialize_stratification()
        
        # محاسبه محدوده زمانی
        # Evaluation range: روزهایی که می‌خواهیم assess کنیم
        if config.start_date:
            self.data_start = pd.to_datetime(config.start_date)
        else:
            self.data_start = self.data['timestamp'].min()
        
        if config.end_date:
            self.data_end = pd.to_datetime(config.end_date)
        else:
            self.data_end = self.data['timestamp'].max()
        
        # توجه: self.data شامل کل داده است (حتی قبل از start_date)
        # تا بتوانیم از pre-range data برای training استفاده کنیم
        
        # آمار
        self.runtime_stats = {
            'start_time': None,
            'end_time': None,
            'total_duration': 0,
            'methods_stats': {}
        }
    
    def evaluate(self):
        """
        اجرای evaluation کامل (assessment mode)
        """
        self.runtime_stats['start_time'] = time.time()
        
        print(f"\n{'='*70}")
        print("INCREMENTAL TEMPORAL EVALUATION (ASSESSMENT MODE)")
        print(f"{'='*70}")
        print(f"Total methods: {len(self.methods)}")
        print(f"Total items: {len(self.items)}")
        print(f"Evaluation range: {self.data_start.date()} to {self.data_end.date()}")
        print(f"Full data range: {self.data['timestamp'].min().date()} to {self.data['timestamp'].max().date()}")
        print(f"Use pre-range data: {self.config.use_pre_range_data}")
        print(f"Storage: {self.storage_path}")
        print(f"Buffer size: 1000 records")
        print(f"{'='*70}\n")
        
        # ارزیابی هر method
        for method_name, method in self.methods.items():
            try:
                print(f"\n{'='*70}")
                print(f"METHOD: {method_name}")
                print(f"{'='*70}")
                
                method_start = time.time()
                
                # ارزیابی این method
                self._evaluate_method_incremental(method_name, method)
                
                method_duration = time.time() - method_start
                self.runtime_stats['methods_stats'][method_name] = {
                    'duration': method_duration,
                    'status': 'completed'
                }
                
                print(f"✓ Completed in {method_duration/60:.1f} minutes")
                
            except Exception as e:
                import traceback
                print(f"✗ Error: {e}")
                print(traceback.format_exc())
                self.runtime_stats['methods_stats'][method_name] = {
                    'duration': 0,
                    'status': 'failed',
                    'error': str(e)
                }
        
        # Flush همه چیز
        print(f"\n{'='*70}")
        print("FINALIZING")
        print(f"{'='*70}")
        
        self.storage.flush_all()
        
        # ذخیره metadata
        metadata = {
            'dataset': self.config.dataset_name if hasattr(self.config, 'dataset_name') else 'unknown',
            'num_items': len(self.items),
            'date_range': {
                'start': str(self.data_start.date()),
                'end': str(self.data_end.date())
            },
            'config': {
                'window_size': self.config.window_size,
                'prediction_horizon': self.config.prediction_horizon,
                'min_observations': self.config.min_observations
            },
            'runtime_stats': self.runtime_stats,
        }
        self.storage.save_metadata(metadata)
        
        self.runtime_stats['end_time'] = time.time()
        self.runtime_stats['total_duration'] = (
            self.runtime_stats['end_time'] - self.runtime_stats['start_time']
        )
        
        print(f"\nTotal duration: {self.runtime_stats['total_duration']/60:.1f} minutes")
        print(f"Storage stats: {self.storage.get_stats()}")
        
        # ذخیره فایل‌های JSON اضافی
        # self._save_runtime_stats()
        self._save_config_json()
        self._save_thresholds_json()
        self._save_runtime_stats_json()
        
        print(f"✓ All results saved to: {self.storage_path}")
        print(f"{'='*70}\n")
    
    def _evaluate_method_incremental(self, method_name: str, method):
        """
        ارزیابی یک method با incremental saving (assessment mode)
        
        Args:
            method_name: نام method
            method: instance method
        """
        # دریافت config این method
        try:
            method_config = get_method_config(method_name)
            window_days = method_config.window_days
            min_obs = method_config.min_observations
        except KeyError:
            print(f"  Warning: No config for {method_name}, using defaults")
            window_days = self.config.window_size
            min_obs = self.config.min_observations
        
        print(f"  Window size: {window_days} days")
        print(f"  Min observations: {min_obs}")
        print(f"  Use pre-range data: {self.config.use_pre_range_data}")
        
        # محاسبه windows
        windows = self._calculate_windows(window_days)
        
        print(f"  Total windows: {len(windows)} (one per day in range)")
        
        if len(windows) == 0:
            print(f"  Warning: No valid windows for this time range")
            return
        
        # پردازش هر window با progress bar
        for window_idx, (train_start, train_end, test_start, test_end) in enumerate(
            tqdm(windows, desc=f"  {method_name}", ncols=70)
        ):
            
            # استخراج داده window
            train_data = self.data[
                (self.data['timestamp'] >= train_start) & 
                (self.data['timestamp'] < train_end)
            ]
            
            test_data = self.data[
                (self.data['timestamp'] >= test_start) & 
                (self.data['timestamp'] < test_end)
            ]
            
            if len(train_data) == 0 or len(test_data) == 0:
                continue
            
            # ارزیابی window
            results = self._evaluate_window(
                method, method_name, window_idx,
                train_data, test_data, test_start,
                min_obs
            )
            
            # اضافه به storage (append)
            if results['detailed']:
                self.storage.append_detailed(method_name, results['detailed'])
            
            if results['summary']:
                self.storage.append_summary(method_name, results['summary'])
            
            # پاک کردن memory
            del train_data, test_data, results
            
            # Garbage collection هر 50 window
            if (window_idx + 1) % 50 == 0:
                gc.collect()
        
        # Flush باقیمانده این method
        self.storage.flush_all()
    
    def _calculate_windows(self, window_days: int) -> List[tuple]:
        """
        محاسبه windows - هر روز یک window (assessment mode)
        
        Args:
            window_days: اندازه training window
        
        Returns:
            لیست (train_start, train_end, test_start, test_end)
        """
        windows = []
        
        # Absolute data range (شامل pre-range data)
        absolute_min = self.data['timestamp'].min()
        
        # Evaluation range (روزهایی که می‌خواهیم assess کنیم)
        eval_start = self.data_start
        eval_end = self.data_end
        
        total_days = (eval_end - eval_start).days + 1
        
        # برای هر روز در evaluation range
        for day_idx in range(total_days):
            target_date = eval_start + timedelta(days=day_idx)
            
            # Training: window_days روز منتهی به target (شامل target)
            train_end = target_date
            train_start = train_end - timedelta(days=window_days - 1)
            
            # اگر train_start قبل از absolute_min است
            if train_start < absolute_min:
                if self.config.use_pre_range_data:
                    # استفاده از absolute_min (داده قبلی موجود نیست)
                    train_start = absolute_min
                else:
                    # فقط از eval_start استفاده کن (implicit zero-padding)
                    train_start = max(train_start, eval_start)
            
            # Test: همان target day
            test_start = target_date
            test_end = target_date + timedelta(days=1)  # برای query
            
            windows.append((train_start, train_end, test_start, test_end))
        
        return windows
    
    def _evaluate_window(self, method, method_name: str, window_idx: int,
                        train_data: pd.DataFrame, test_data: pd.DataFrame,
                        timestamp: datetime, min_observations: int) -> Dict:
        """
        ارزیابی یک window
        
        Returns:
            Dict با کلیدهای 'detailed' و 'summary'
        """
        results = {
            'detailed': [],
            'summary': []
        }
        
        # گروه‌بندی داده بر اساس item
        train_grouped = train_data.groupby('item_id')['count'].apply(lambda x: x.values)
        test_grouped = test_data.groupby('item_id')['count'].sum()
        
        # فیلتر items با تعداد کافی
        valid_items = [
            item for item in train_grouped.index
            if len(train_grouped[item]) >= min_observations
        ]
        
        if len(valid_items) == 0:
            return results
        
        # محاسبه scores
        scores = []
        actuals = []
        train_counts = []
        
        for item in valid_items:
            time_series = train_grouped[item]
            
            try:
                # محاسبه score
                if hasattr(method, 'assess_single'):
                    score = method.assess_single(time_series)
                elif hasattr(method, 'assess'):
                    score = method.assess(time_series)
                else:
                    score = float(np.mean(time_series))
                
                score = float(score)
            except Exception as e:
                # Fallback
                score = float(np.mean(time_series))
            
            actual = int(test_grouped.get(item, 0))
            train_count = int(np.sum(time_series))
            
            scores.append(score)
            actuals.append(actual)
            train_counts.append(train_count)
        
        # تبدیل به numpy
        scores = np.array(scores)
        actuals = np.array(actuals)
        train_counts = np.array(train_counts)
        
        # محاسبه metrics
        try:
            metrics = MetricsCalculator.calculate_all_metrics(scores, actuals)
        except Exception as e:
            # اگر metrics fail شد، metrics ساده
            metrics = {
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'spearman': 0.0,
                'kendall': 0.0,
                'ndcg': 0.0,
                'coverage': 0.0,
                'rank_predicted': np.arange(len(scores)),
                'rank_actual': np.arange(len(actuals))
            }
        
        # ساخت detailed records
        timestamp_ms = int(timestamp.timestamp() * 1000)
        
        for i, item in enumerate(valid_items):
            # تعیین stratum
            stratum_label = self.stratification.get_stratum_label(train_counts[i])
            
            # محاسبه error metrics برای این item
            pred = scores[i]
            actual = actuals[i]
            error = abs(pred - actual)
            
            results['detailed'].append({
                'window_id': window_idx,
                'timestamp': timestamp_ms,
                'item_id': str(item),
                'stratum': stratum_label,
                'popularity_score': float(scores[i]),
                'actual_count': int(actuals[i]),
                'train_count': int(train_counts[i]),
                'rank_predicted': int(metrics['rank_predicted'][i]) if i < len(metrics['rank_predicted']) else 0,
                'rank_actual': int(metrics['rank_actual'][i]) if i < len(metrics['rank_actual']) else 0,
                # اضافه کردن error metrics
                'mae': float(error),
                'squared_error': float(error ** 2),
                'mape': float(error / actual * 100) if actual > 0 else 0.0,
            })
        
        # محاسبه summary per stratum
        for stratum_label in range(4):  # 0, 1, 2, 3
            stratum_mask = np.array([
                self.stratification.get_stratum_label(tc) == stratum_label
                for tc in train_counts
            ])
            
            # اگر stratum خالی است، از metrics پیش‌فرض استفاده کن
            if not stratum_mask.any():
                results['summary'].append({
                    'window_id': window_idx,
                    'timestamp': timestamp_ms,
                    'stratum': stratum_label,
                    'stratum_name': self.stratification.get_stratum_name(stratum_label),
                    'num_items': 0,
                    'mean_mae': 0.0,
                    'mean_mape': 0.0,
                    'spearman_corr': 0.0,
                    'kendall_tau': 0.0,
                    'mean_rmse': 0.0,
                    'ndcg': 0.0,
                })
                continue
            
            stratum_scores = scores[stratum_mask]
            stratum_actuals = actuals[stratum_mask]
            
            try:
                stratum_metrics = MetricsCalculator.calculate_all_metrics(
                    stratum_scores, stratum_actuals
                )
            except:
                # در صورت خطا، از metrics پیش‌فرض استفاده کن
                results['summary'].append({
                    'window_id': window_idx,
                    'timestamp': timestamp_ms,
                    'stratum': stratum_label,
                    'stratum_name': self.stratification.get_stratum_name(stratum_label),
                    'num_items': int(stratum_mask.sum()),
                    'mean_mae': 0.0,
                    'mean_mape': 0.0,
                    'spearman_corr': 0.0,
                    'kendall_tau': 0.0,
                    'mean_rmse': 0.0,
                    'ndcg': 0.0,
                })
                continue
            
            results['summary'].append({
                'window_id': window_idx,
                'timestamp': timestamp_ms,
                'stratum': stratum_label,
                'stratum_name': self.stratification.get_stratum_name(stratum_label),
                'num_items': int(stratum_mask.sum()),
                'mean_mae': float(stratum_metrics['mae']),
                'mean_mape': float(stratum_metrics['mape']),
                'spearman_corr': float(stratum_metrics['spearman']),
                'kendall_tau': float(stratum_metrics['kendall']),
                'ndcg': float(stratum_metrics['ndcg']),
                'coverage': float(stratum_metrics['coverage']),
            })
        
        return results
    
    def _initialize_stratification(self):
        """
        محاسبه thresholds برای stratification بر اساس کل داده
        """
        # محاسبه تعداد کل دسترسی هر item
        item_counts = self.data.groupby('item_id')['count'].sum()
        
        # محاسبه thresholds
        if self.config.strata_thresholds is None:
            # خودکار: Q1, Q2, Q3
            q1 = item_counts.quantile(0.25)
            q2 = item_counts.quantile(0.50)
            q3 = item_counts.quantile(0.75)
            self.stratification.thresholds = [q1, q2, q3]
        else:
            # از config استفاده کن
            self.stratification.thresholds = self.config.strata_thresholds
        
        print(f"Stratification thresholds initialized:")
        print(f"  Cold-start: < {self.stratification.thresholds[0]:.0f}")
        print(f"  Low: {self.stratification.thresholds[0]:.0f} - {self.stratification.thresholds[1]:.0f}")
        print(f"  Medium: {self.stratification.thresholds[1]:.0f} - {self.stratification.thresholds[2]:.0f}")
        print(f"  High: >= {self.stratification.thresholds[2]:.0f}")
        print()
    
    def _save_runtime_stats(self):
        """ذخیره آمار زمان اجرا"""
        
        stats = {
            **self.runtime_stats,
            'dataset_name': self.config.dataset_name if hasattr(self.config, 'dataset_name') else 'unknown',
            'window_size': self.config.window_size,
            'prediction_horizon': self.config.prediction_horizon,
            'time_granularity': self.config.time_granularity if hasattr(self.config, 'time_granularity') else 'daily',
            'num_items': len(self.items),
            'item_selection': self.config.item_selection if hasattr(self.config, 'item_selection') else 'top',
            'min_observations': self.config.min_observations,
            'start_date': str(self.data_start.date()),
            'end_date': str(self.data_end.date()),
            'use_pre_range_data': self.config.use_pre_range_data if hasattr(self.config, 'use_pre_range_data') else True,
            'methods': list(self.methods.keys()),
            'strata_names': self.config.strata_names,
        }
        
        self.storage.save_runtime_stats(stats)
        self.config.save_config(self.config.output_dir / 'metadata' / 'config.json')
        self.stratification.save_thresholds(
            self.config.output_dir / 'metadata' / 'thresholds.json'
        )


    def _save_config_json(self):
        """ذخیره config.json در root فولدر"""
        import json
        
        config_dict = {
            'dataset_name': self.config.dataset_name if hasattr(self.config, 'dataset_name') else 'unknown',
            'window_size': self.config.window_size,
            'prediction_horizon': self.config.prediction_horizon,
            'time_granularity': self.config.time_granularity if hasattr(self.config, 'time_granularity') else 'daily',
            'num_items': len(self.items),
            'item_selection': self.config.item_selection if hasattr(self.config, 'item_selection') else 'top',
            'min_observations': self.config.min_observations,
            'start_date': str(self.data_start.date()),
            'end_date': str(self.data_end.date()),
            'use_pre_range_data': self.config.use_pre_range_data if hasattr(self.config, 'use_pre_range_data') else True,
            'methods': list(self.methods.keys()),
            'strata_names': self.config.strata_names,
        }
        
        config_path = self.storage_path / 'metadata' / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        if self.config.verbose:
            print(f"  ✓ Saved config.json")
    
    def _save_thresholds_json(self):
        """ذخیره thresholds.json در root فولدر"""
        import json
        
        thresholds_dict = {
            'thresholds': self.stratification.thresholds if hasattr(self.stratification, 'thresholds') else [],
            'strata_names': self.config.strata_names,
            'total_items': len(self.items)
        }
        
        thresholds_path = self.storage_path / 'metadata' / 'thresholds.json'
        with open(thresholds_path, 'w', encoding='utf-8') as f:
            json.dump(thresholds_dict, f, indent=2)
        
        if self.config.verbose:
            print(f"  ✓ Saved thresholds.json")
    
    def _save_runtime_stats_json(self):
        """ذخیره runtime_stats.json در root فولدر"""
        import json
        
        runtime_dict = {
            'total_duration': self.runtime_stats.get('total_duration', 0),
            'total_duration_minutes': self.runtime_stats.get('total_duration', 0) / 60,
            'start_time': self.runtime_stats.get('start_time'),
            'end_time': self.runtime_stats.get('end_time'),
            'methods_stats': self.runtime_stats.get('methods_stats', {})
        }
        
        runtime_path = self.storage_path / 'metadata' / 'runtime_stats.json'
        with open(runtime_path, 'w', encoding='utf-8') as f:
            json.dump(runtime_dict, f, indent=2)
        
        if self.config.verbose:
            print(f"  ✓ Saved runtime_stats.json")


if __name__ == '__main__':
    print("\n✓ IncrementalTemporalEvaluator module loaded successfully")
    print("\nUsage:")
    print("  from evaluation.incremental_evaluator import IncrementalTemporalEvaluator")
    print("  evaluator = IncrementalTemporalEvaluator(...)")
    print("  evaluator.evaluate()")