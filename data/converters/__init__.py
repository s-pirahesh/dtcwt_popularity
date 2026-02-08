"""
Data Converters Package
تبدیل دیتاست‌های خام به فرمت استاندارد

Available Converters:
- MovieLens: ratings.csv → standard CSV
- Uber/NYC Taxi: Parquet files → standard CSV

Usage:
    from data.converters import ConverterFactory
    
    converter = ConverterFactory.create('movielens', aggregate_by='day')
    df = converter.convert(input_path, output_path)
"""
from .base_converter import BaseConverter, ConverterFactory

# Import converters to auto-register them
from . import movielens_converter
from . import uber_converter

# TODO: Import future converters here
# from . import youtube_converter
# from . import youku_converter

__all__ = ['BaseConverter', 'ConverterFactory']
__version__ = '1.0.0'
