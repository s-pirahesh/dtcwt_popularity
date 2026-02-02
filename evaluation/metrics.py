# -*- coding: utf-8 -*-
"""
Comprehensive Metrics Calculator V2
Metrics for scoring/assessment (not prediction)
Author: Sajjad
Date: February 2025
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import ndcg_score, mean_absolute_error, mean_squared_error
from typing import Dict, Tuple


class MetricsCalculator:
    """
    محاسبه معیارهای ارزیابی جامع
    فقط برای assessment/scoring (بدون prediction)
    """
    
    @staticmethod
    def calculate_all_metrics(scores: np.ndarray, actual_counts: np.ndarray) -> Dict:
        """
        محاسبه همه معیارها برای یک مجموعه آیتم
        
        Args:
            scores: امتیازهای محاسبه شده توسط روش (popularity scores)
            actual_counts: تعداد دسترسی‌های واقعی
        
        Returns:
            dict حاوی همه معیارها
        """
        # تبدیل به numpy arrays
        scores = np.asarray(scores, dtype=np.float64)
        actual = np.asarray(actual_counts, dtype=np.float64)
        
        # حذف NaN values
        valid_idx = ~(np.isnan(scores) | np.isnan(actual))
        scores = scores[valid_idx]
        actual = actual[valid_idx]
        
        if len(scores) == 0:
            return MetricsCalculator._empty_metrics()
        
        metrics = {}
        
        # 1. Ranking Metrics
        try:
            rank_predicted = (-scores).argsort().argsort() + 1
            rank_actual = (-actual).argsort().argsort() + 1
            
            spearman_corr, spearman_p = spearmanr(scores, actual)
            kendall_corr, kendall_p = kendalltau(scores, actual)
            
            metrics['spearman'] = spearman_corr if not np.isnan(spearman_corr) else 0.0
            metrics['spearman_pvalue'] = spearman_p if not np.isnan(spearman_p) else 1.0
            metrics['kendall'] = kendall_corr if not np.isnan(kendall_corr) else 0.0
            metrics['kendall_pvalue'] = kendall_p if not np.isnan(kendall_p) else 1.0
            
        except Exception as e:
            metrics['spearman'] = 0.0
            metrics['spearman_pvalue'] = 1.0
            metrics['kendall'] = 0.0
            metrics['kendall_pvalue'] = 1.0
            rank_predicted = np.arange(len(scores)) + 1
            rank_actual = np.arange(len(actual)) + 1
        
        # 2. Error Metrics
        mae = mean_absolute_error(actual, scores)
        rmse = np.sqrt(mean_squared_error(actual, scores))
        
        # MAPE با handling برای divide by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual - scores) / np.where(actual == 0, 1, actual))) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = 0.0
        
        metrics['mae'] = mae
        metrics['rmse'] = rmse
        metrics['mape'] = mape
        
        # 3. NDCG
        try:
            # تبدیل به فرمت مناسب برای ndcg_score
            ndcg = ndcg_score([actual], [scores])
            if np.isnan(ndcg):
                ndcg = 0.0
        except:
            ndcg = 0.0
        
        metrics['ndcg'] = ndcg
        
        # 4. Coverage
        coverage = (scores > 0).sum() / len(scores) if len(scores) > 0 else 0.0
        metrics['coverage'] = coverage
        
        # 5. Rank-based metrics
        metrics['mean_rank_error'] = np.mean(np.abs(rank_predicted - rank_actual))
        metrics['median_rank_error'] = np.median(np.abs(rank_predicted - rank_actual))
        
        # 6. Additional statistics
        metrics['mean_score'] = np.mean(scores)
        metrics['std_score'] = np.std(scores)
        metrics['mean_actual'] = np.mean(actual)
        metrics['std_actual'] = np.std(actual)
        
        # 7. Rankings (for detailed analysis)
        metrics['rank_predicted'] = rank_predicted
        metrics['rank_actual'] = rank_actual
        
        return metrics
    
    @staticmethod
    def _empty_metrics() -> Dict:
        """متریک‌های خالی برای حالت error"""
        return {
            'spearman': 0.0,
            'spearman_pvalue': 1.0,
            'kendall': 0.0,
            'kendall_pvalue': 1.0,
            'mae': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'ndcg': 0.0,
            'coverage': 0.0,
            'mean_rank_error': 0.0,
            'median_rank_error': 0.0,
            'mean_score': 0.0,
            'std_score': 0.0,
            'mean_actual': 0.0,
            'std_actual': 0.0,
            'rank_predicted': np.array([]),
            'rank_actual': np.array([]),
        }
    
    @staticmethod
    def calculate_stratum_metrics(scores: np.ndarray, actual_counts: np.ndarray, 
                                  item_ids: np.ndarray, stratum_items: np.ndarray) -> Dict:
        """
        محاسبه معیارها برای یک stratum خاص
        
        Args:
            scores: همه امتیازها
            actual_counts: همه تعداد دسترسی‌ها
            item_ids: همه شناسه‌های آیتم
            stratum_items: آیتم‌های این stratum
        
        Returns:
            dict حاوی معیارها
        """
        # فیلتر کردن فقط آیتم‌های این stratum
        mask = np.isin(item_ids, stratum_items)
        
        if not np.any(mask):
            return MetricsCalculator._empty_metrics()
        
        stratum_scores = scores[mask]
        stratum_actual = actual_counts[mask]
        
        return MetricsCalculator.calculate_all_metrics(stratum_scores, stratum_actual)
    
    @staticmethod
    def calculate_temporal_stability(scores_t: np.ndarray, scores_t_prev: np.ndarray) -> Dict:
        """
        محاسبه پایداری زمانی امتیازها
        
        Args:
            scores_t: امتیازها در زمان t
            scores_t_prev: امتیازها در زمان t-1
        
        Returns:
            dict حاوی معیارهای پایداری
        """
        if len(scores_t) != len(scores_t_prev):
            raise ValueError("Score arrays must have same length")
        
        # حذف NaN
        valid_idx = ~(np.isnan(scores_t) | np.isnan(scores_t_prev))
        scores_t = scores_t[valid_idx]
        scores_t_prev = scores_t_prev[valid_idx]
        
        if len(scores_t) == 0:
            return {'stability': 0.0, 'volatility': 0.0}
        
        # 1. Score stability (1 - normalized absolute change)
        score_changes = np.abs(scores_t - scores_t_prev)
        mean_change = np.mean(score_changes)
        max_possible_change = np.max([np.max(np.abs(scores_t)), np.max(np.abs(scores_t_prev))])
        
        if max_possible_change > 0:
            stability = 1.0 - (mean_change / max_possible_change)
        else:
            stability = 1.0
        
        # 2. Rank volatility
        rank_t = (-scores_t).argsort().argsort() + 1
        rank_t_prev = (-scores_t_prev).argsort().argsort() + 1
        rank_changes = np.abs(rank_t - rank_t_prev)
        volatility = np.mean(rank_changes)
        
        return {
            'stability': stability,
            'volatility': volatility,
            'mean_score_change': mean_change,
            'max_score_change': np.max(score_changes),
        }
    
    @staticmethod
    def compare_methods(method_scores: Dict[str, np.ndarray], 
                       actual_counts: np.ndarray) -> pd.DataFrame:
        """
        مقایسه چند روش با یکدیگر
        
        Args:
            method_scores: {method_name: scores array}
            actual_counts: تعداد دسترسی‌های واقعی
        
        Returns:
            DataFrame حاوی مقایسه
        """
        results = []
        
        for method_name, scores in method_scores.items():
            metrics = MetricsCalculator.calculate_all_metrics(scores, actual_counts)
            
            # فقط معیارهای اصلی
            result = {
                'method': method_name,
                'spearman': metrics['spearman'],
                'kendall': metrics['kendall'],
                'mae': metrics['mae'],
                'rmse': metrics['rmse'],
                'mape': metrics['mape'],
                'ndcg': metrics['ndcg'],
                'coverage': metrics['coverage'],
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # مرتب‌سازی بر اساس spearman (بالاتر بهتر)
        df = df.sort_values('spearman', ascending=False)
        
        return df
    
    @staticmethod
    def calculate_improvement(baseline_metrics: Dict, method_metrics: Dict) -> Dict:
        """
        محاسبه میزان بهبود نسبت به baseline
        
        Args:
            baseline_metrics: معیارهای baseline
            method_metrics: معیارهای روش پیشنهادی
        
        Returns:
            dict حاوی درصد بهبود
        """
        improvements = {}
        
        # معیارهایی که بالاتر بهتر است
        higher_better = ['spearman', 'kendall', 'ndcg', 'coverage']
        
        # معیارهایی که پایین‌تر بهتر است
        lower_better = ['mae', 'rmse', 'mape', 'mean_rank_error']
        
        for metric in higher_better:
            if metric in baseline_metrics and metric in method_metrics:
                baseline_val = baseline_metrics[metric]
                method_val = method_metrics[metric]
                
                if baseline_val != 0:
                    improvement = ((method_val - baseline_val) / abs(baseline_val)) * 100
                else:
                    improvement = 0.0
                
                improvements[f'{metric}_improvement'] = improvement
        
        for metric in lower_better:
            if metric in baseline_metrics and metric in method_metrics:
                baseline_val = baseline_metrics[metric]
                method_val = method_metrics[metric]
                
                if baseline_val != 0:
                    improvement = ((baseline_val - method_val) / abs(baseline_val)) * 100
                else:
                    improvement = 0.0
                
                improvements[f'{metric}_improvement'] = improvement
        
        return improvements
