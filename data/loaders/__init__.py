# -*- coding: utf-8 -*-
"""
Data Loaders Package
بارگذاری و پردازش دیتاست‌های مختلف

Available loaders:
- MovieLensLoader: MovieLens 32M
- YellowTaxiLoader: NYC Yellow Taxi
- YouTubeLoader: YouTube 
- YoukuLoader: Youku (coming soon)

Author: Sajjad
Date: February 2026
"""

from .base_loader import BaseLoader
from .movielens_loader import MovieLensLoader, get_movielens_loader
from .yellow_taxi_loader import YellowTaxiLoader, get_yellow_taxi_loader
from .youtube_loader import YouTubeLoader, get_youtube_loader  

__all__ = [
    'BaseLoader',
    'MovieLensLoader',
    'get_movielens_loader',
    'YellowTaxiLoader',
    'get_yellow_taxi_loader',
    'YouTubeLoader',  
    'get_youtube_loader',  
]

__version__ = '1.0.0'
