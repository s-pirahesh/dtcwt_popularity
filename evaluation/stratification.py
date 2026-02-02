# -*- coding: utf-8 -*-
"""
Automatic Stratification System
Adaptive item categorization for all datasets
Author: Sajjad
Date: February 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class StratificationSystem:
    """
    سیستم طبقه‌بندی خودکار آیتم‌ها
    سازگار با همه دیتاست‌ها
    """
    
    def __init__(self, config):
        """
        Args:
            config: EvaluationConfig instance
        """
        self.config = config
        self.thresholds = None
        self.stratum_stats = {}
    
    def stratify_items(self, window_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        طبقه‌بندی آیتم‌ها بر اساس تعداد دسترسی
        
        Args:
            window_data: DataFrame حاوی ستون‌های item_id و count
        
        Returns:
            dict: {stratum_name: array of item_ids}
        """
        # محاسبه تعداد دسترسی هر آیتم
        item_counts = window_data.groupby('item_id')['count'].sum()
        
        # محاسبه یا استفاده از thresholds
        if self.config.strata_thresholds is None:
            self.thresholds = self._calculate_thresholds(item_counts)
        else:
            self.thresholds = self.config.strata_thresholds
        
        # طبقه‌بندی
        strata = {}
        strata['cold_start'] = item_counts[item_counts < self.thresholds[0]].index.values
        strata['low'] = item_counts[
            (item_counts >= self.thresholds[0]) & 
            (item_counts < self.thresholds[1])
        ].index.values
        strata['medium'] = item_counts[
            (item_counts >= self.thresholds[1]) & 
            (item_counts < self.thresholds[2])
        ].index.values
        strata['high'] = item_counts[item_counts >= self.thresholds[2]].index.values
        
        # ذخیره آمار
        self._save_stratum_stats(item_counts, strata)
        
        return strata
    
    def _calculate_thresholds(self, item_counts: pd.Series) -> List[int]:
        """
        محاسبه خودکار thresholds بر اساس quartiles
        
        Args:
            item_counts: Series تعداد دسترسی هر آیتم
        
        Returns:
            list: [Q1, Q2, Q3] thresholds
        """
        q1 = item_counts.quantile(0.25)
        q2 = item_counts.quantile(0.50)  # median
        q3 = item_counts.quantile(0.75)
        
        # گرد کردن به اعداد صحیح
        thresholds = [
            int(np.ceil(q1)),
            int(np.ceil(q2)),
            int(np.ceil(q3))
        ]
        
        if self.config.verbose:
            print(f"Auto-calculated thresholds: {thresholds}")
        
        return thresholds
    
    def _save_stratum_stats(self, item_counts: pd.Series, strata: Dict[str, np.ndarray]):
        """ذخیره آمار هر stratum"""
        
        self.stratum_stats = {}
        
        for stratum_name, items in strata.items():
            if len(items) == 0:
                continue
            
            counts = item_counts[items]
            
            self.stratum_stats[stratum_name] = {
                'num_items': len(items),
                'percentage': len(items) / len(item_counts) * 100,
                'min_count': counts.min(),
                'max_count': counts.max(),
                'mean_count': counts.mean(),
                'median_count': counts.median(),
                'std_count': counts.std(),
            }
        
        if self.config.verbose:
            self._print_stratum_stats()
    
    def _print_stratum_stats(self):
        """چاپ آمار strata"""
        
        print("\n" + "="*70)
        print("STRATIFICATION STATISTICS")
        print("="*70)
        print(f"{'Stratum':<15} {'Items':<8} {'%':<6} {'Min':<8} {'Max':<8} {'Mean':<8} {'Median':<8}")
        print("-"*70)
        
        for stratum_name in ['cold_start', 'low', 'medium', 'high']:
            if stratum_name not in self.stratum_stats:
                continue
            
            stats = self.stratum_stats[stratum_name]
            print(
                f"{stratum_name:<15} "
                f"{stats['num_items']:<8} "
                f"{stats['percentage']:<6.1f} "
                f"{stats['min_count']:<8.0f} "
                f"{stats['max_count']:<8.0f} "
                f"{stats['mean_count']:<8.1f} "
                f"{stats['median_count']:<8.0f}"
            )
        
        print("="*70 + "\n")
    
    def get_stratum_label(self, count: int) -> int:
        """
        برچسب stratum برای یک مقدار
        
        Args:
            count: تعداد دسترسی
        
        Returns:
            int: 0=cold_start, 1=low, 2=medium, 3=high
        """
        if self.thresholds is None:
            raise ValueError("Thresholds not calculated yet. Call stratify_items first.")
        
        if count < self.thresholds[0]:
            return 0  # cold_start
        elif count < self.thresholds[1]:
            return 1  # low
        elif count < self.thresholds[2]:
            return 2  # medium
        else:
            return 3  # high
    
    def get_stratum_name(self, label: int) -> str:
        """تبدیل label عددی به نام"""
        names = ['cold_start', 'low', 'medium', 'high']
        return names[label]
    
    def filter_by_stratum(self, items: np.ndarray, stratum_name: str, 
                         strata: Dict[str, np.ndarray]) -> np.ndarray:
        """
        فیلتر آیتم‌ها بر اساس stratum
        
        Args:
            items: لیست همه آیتم‌ها
            stratum_name: نام stratum
            strata: نتیجه stratify_items
        
        Returns:
            آیتم‌های فیلتر شده
        """
        stratum_items = strata[stratum_name]
        return np.intersect1d(items, stratum_items)
    
    def get_stratum_summary(self) -> pd.DataFrame:
        """خلاصه آمار همه strata"""
        
        if not self.stratum_stats:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.stratum_stats).T
        df.index.name = 'stratum'
        
        return df.reset_index()
    
    def save_thresholds(self, filepath):
        """ذخیره thresholds"""
        import json
        
        data = {
            'thresholds': self.thresholds,
            'stratum_stats': self.stratum_stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # تبدیل numpy types به Python types
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                return obj
            
            data = convert_types(data)
            json.dump(data, f, indent=2, ensure_ascii=False)
