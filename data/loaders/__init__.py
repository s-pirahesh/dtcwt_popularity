# -*- coding: utf-8 -*-
"""
Data Loaders Package
بارگذاری و پردازش دیتاست‌های مختلف

Available loaders:
- MovieLensLoader: MovieLens 25M
- YouTubeLoader: YouTube (coming soon)
- YoukuLoader: Youku (coming soon)

Author: Sajjad
Date: February 2025
"""

from .base_loader import BaseLoader
from .movielens_loader import MovieLensLoader, get_movielens_loader
from .uber_loader import UberLoader

__all__ = [
    'BaseLoader',
    'MovieLensLoader',
    'get_movielens_loader',
]

__version__ = '1.0.0'
