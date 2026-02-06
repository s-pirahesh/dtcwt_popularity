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
from methods.base_method import BaseMethod

class AccessFrequency(BaseMethod):
    def __init__(self):
        super().__init__("AF")
    def assess_single(self, time_series):
        return TraditionalBaselines.access_frequency(time_series)

class LRU(BaseMethod):
    def __init__(self):
        super().__init__("LRU")
    def assess_single(self, time_series):
        return TraditionalBaselines.lru(time_series)

class LFU(BaseMethod):
    def __init__(self):
        super().__init__("LFU")
    def assess_single(self, time_series):
        return TraditionalBaselines.lfu(time_series)

class EWMA(BaseMethod):
    def __init__(self, alpha=0.3):
        super().__init__("EWMA")
        self.alpha = alpha
    def assess_single(self, time_series):
        return TraditionalBaselines.ewma(time_series, self.alpha)

__all__ = [
    'AccessFrequency',
    'LRU',
    'LFU',
    'EWMA',
    'TraditionalBaselines'
]

__version__ = '1.0.0'
