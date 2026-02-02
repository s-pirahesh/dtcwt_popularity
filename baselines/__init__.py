# -*- coding: utf-8 -*-
"""
Baselines Package
روش‌های پایه (Baseline) برای مقایسه

Available baselines:
- AccessFrequency: Simple access frequency counting
- LRU: Least Recently Used
- LFU: Least Frequently Used
- EWMA: Exponentially Weighted Moving Average

Author: Sajjad
Date: February 2025
"""

from .traditional import TraditionalBaselines

# Wrapper classes برای سازگاری با interface موجود
class AccessFrequency:
    """Access Frequency baseline"""
    @staticmethod
    def assess(time_series):
        return TraditionalBaselines.access_frequency(time_series)

class LRU:
    """Least Recently Used baseline"""
    @staticmethod
    def assess(time_series):
        return TraditionalBaselines.lru_score(time_series)

class LFU:
    """Least Frequently Used baseline"""
    @staticmethod
    def assess(time_series):
        return TraditionalBaselines.lfu_score(time_series)

class EWMA:
    """Exponentially Weighted Moving Average baseline"""
    @staticmethod
    def assess(time_series, alpha=0.3):
        return TraditionalBaselines.ewma_score(time_series, alpha)

__all__ = [
    'TraditionalBaselines',
    'AccessFrequency',
    'LRU',
    'LFU',
    'EWMA',
]

__version__ = '1.0.0'
