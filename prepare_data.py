#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت آماده‌سازی داده‌ها V2
تبدیل دیتاست‌های خام به فرمت استاندارد

Features:
- Dynamic argument generation (هر converter arguments خودش را می‌سازد)
- Config file support (YAML)
- Generic + Dataset-specific parameters
- Backward compatible

استفاده:
    # MovieLens - basic
    python prepare_data.py --dataset movielens \\
        --input data/raw/movielens/ratings.csv \\
        --output data/datasets/movielens.csv
    
    # MovieLens - با options
    python prepare_data.py --dataset movielens \\
        --movielens-aggregate-by day \\
        --movielens-keep-rating \\
        --movielens-min-rating 4.0
    
    # Uber - با options
    python prepare_data.py --dataset uber \\
        --input "data/raw/uber/yellow_*.parquet" \\
        --output data/datasets/uber_15min.csv \\
        --uber-granularity 15min \\
        --uber-min-trips-per-location 200 \\
        --uber-extract-features
    
    # از config file
    python prepare_data.py --config configs/uber_hourly.yaml
    
    # لیست دیتاست‌ها
    python prepare_data.py --list
"""
import argparse
import sys
from pathlib import Path
from glob import glob
import yaml
from typing import Dict, Any, Optional

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# اضافه کردن پروژه به path
sys.path.insert(0, str(Path(__file__).parent))

# Import converters (این باعث auto-registration می‌شود)
from data.converters.base_converter import ConverterFactory
from data.converters import movielens_converter
from data.converters import uber_converter

# پیکربندی دیتاست‌ها (برای --all و مسیرهای پیش‌فرض)
DATASET_CONFIGS = {
    'movielens': {
        'description': 'MovieLens ratings dataset',
        'input': 'data/raw/movielens/ratings.csv',
        'output': 'data/datasets/movielens.csv'
    },
    'uber': {
        'description': 'NYC Yellow Taxi trip data',
        'input': 'data/raw/uber/yellow_*.parquet',
        'output': 'data/datasets/uber_15min.csv'
    }
    # اضافه کردن دیتاست‌های بعدی اینجا
}


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    خواندن فایل config (YAML)
    
    Args:
        config_path: مسیر فایل config
        
    Returns:
        Dictionary تنظیمات
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def parse_args():
    """پردازش arguments از command line"""
    
    # ==========================================
    # Parser اصلی
    # ==========================================
    parser = argparse.ArgumentParser(
        description='تبدیل دیتاست‌های خام به فرمت استاندارد',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌ها:
  # MovieLens با aggregation
  python prepare_data.py --dataset movielens --movielens-aggregate-by day --movielens-keep-rating
  
  # Uber با features
  python prepare_data.py --dataset uber --uber-granularity hourly --uber-extract-features
  
  # از config file
  python prepare_data.py --config configs/movielens_daily.yaml
  
  # لیست دیتاست‌ها
  python prepare_data.py --list
        """
    )
    
    # ==========================================
    # Core Arguments
    # ==========================================
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=ConverterFactory.list_converters(),
        help='نام دیتاست'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='مسیر ورودی (می‌تواند wildcard داشته باشد)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='مسیر خروجی CSV'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='مسیر فایل config (YAML)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='تبدیل همه دیتاست‌ها (از DATASET_CONFIGS)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='نمایش لیست دیتاست‌های موجود'
    )
    
    # ==========================================
    # Generic Parameters (مشترک بین همه)
    # ==========================================
    generic_group = parser.add_argument_group('Generic Options')
    
    generic_group.add_argument(
        '--output-format', '-f',
        type=str,
        default='csv',
        choices=['csv', 'parquet', 'feather'],
        help='فرمت خروجی (پیش‌فرض: csv)'
    )
    
    generic_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='حالت ساکت (بدون نمایش پیام‌ها)'
    )
    
    generic_group.add_argument(
        '--no-validate',
        action='store_true',
        help='غیرفعال کردن اعتبارسنجی'
    )
    
    # ==========================================
    # Dataset-Specific Arguments
    # تمام converters به صورت خودکار اضافه می‌شوند
    # ==========================================
    for dataset_name in ConverterFactory.list_converters():
        converter_class = ConverterFactory.get_converter_class(dataset_name)
        converter_class.add_arguments(parser, prefix=True)
    
    return parser.parse_args()


def list_datasets():
    """نمایش لیست دیتاست‌ها"""
    print("\n" + "=" * 70)
    print("Available datasets for conversion:")
    print("=" * 70)
    
    for dataset_name in ConverterFactory.list_converters():
        converter_class = ConverterFactory.get_converter_class(dataset_name)
        
        # اطلاعات از config (اگر موجود باشد)
        config = DATASET_CONFIGS.get(dataset_name, {})
        
        print(f"\nDataset: {dataset_name.upper()}")
        print(f"   {converter_class.DESCRIPTION}")
        
        if config:
            print(f"   Input (default):  {config.get('input', 'N/A')}")
            print(f"   Output (default): {config.get('output', 'N/A')}")
        
        # نمایش parameters
        params = converter_class.get_specific_params()
        if params:
            print(f"   Parameters:")
            for param_name, param_spec in params.items():
                default = param_spec.get('default', 'None')
                print(f"     --{dataset_name}-{param_name.replace('_', '-')}: {param_spec.get('help', '')} (default: {default})")
    
    print("\n" + "=" * 70)
    print(f"Total: {len(ConverterFactory.list_converters())} datasets")
    print("=" * 70 + "\n")


def convert_single_dataset(dataset_name: str,
                           input_path: str,
                           output_path: str,
                           converter_params: Dict[str, Any],
                           verbose: bool = True) -> bool:
    """
    تبدیل یک دیتاست
    
    Args:
        dataset_name: نام دیتاست
        input_path: مسیر ورودی
        output_path: مسیر خروجی
        converter_params: پارامترهای converter
        verbose: نمایش پیام‌ها
        
    Returns:
        True اگر موفق، False اگر ناموفق
    """
    print("\n" + "=" * 70)
    print(f"Converting dataset: {dataset_name.upper()}")
    print("=" * 70 + "\n")
    
    # نمایش پارامترها
    if converter_params and verbose:
        print("Converter parameters:")
        for key, value in converter_params.items():
            if value is not None and value != False and value != 'none':
                print(f"  {key}: {value}")
        print()
    
    # بررسی wildcard در مسیر ورودی
    if '*' in input_path or '?' in input_path:
        input_files = sorted(glob(input_path))
        if not input_files:
            print(f"ERROR: No files found matching pattern '{input_path}'.")
            return False
        print(f"OK: Found {len(input_files)} files.")
        input_path = input_files
    
    try:
        # ایجاد converter
        converter = ConverterFactory.create(dataset_name, **converter_params)
        
        # تبدیل
        df = converter.convert(input_path, output_path)
        
        print(f"\nOK: Conversion completed: {len(df):,} records.")
        print(f"  Output: {output_path}\n")
        
        return True
    
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def convert_all_datasets(verbose: bool = True) -> Dict[str, bool]:
    """
    تبدیل همه دیتاست‌ها
    
    Args:
        verbose: نمایش پیام‌ها
        
    Returns:
        Dictionary نتایج {dataset_name: success}
    """
    print("\n" + "=" * 70)
    print("Starting conversion for all datasets")
    print("=" * 70 + "\n")
    
    results = {}
    
    for dataset_name, config in DATASET_CONFIGS.items():
        # استفاده از مقادیر پیش‌فرض config
        success = convert_single_dataset(
            dataset_name=dataset_name,
            input_path=config['input'],
            output_path=config['output'],
            converter_params={
                'verbose': verbose,
                'output_format': 'csv',
                'validate_output': True
            },
            verbose=verbose
        )
        results[dataset_name] = success
    
    # خلاصه نتایج
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    
    for dataset_name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {dataset_name:<20} {status}")
    
    total = len(results)
    successful = sum(results.values())
    
    print("=" * 70)
    print(f"Total: {successful}/{total} succeeded")
    print("=" * 70 + "\n")
    
    return results


def main():
    """تابع اصلی"""
    args = parse_args()
    
    # ==========================================
    # حالت: نمایش لیست
    # ==========================================
    if args.list:
        list_datasets()
        return 0
    
    # ==========================================
    # حالت: از config file
    # ==========================================
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config_file(args.config)
        
        dataset_name = config.get('dataset')
        input_path = config.get('input')
        output_path = config.get('output')
        converter_params = config.get('converter_params', {})
        
        # اضافه کردن generic params
        converter_params['verbose'] = not args.quiet
        converter_params['output_format'] = args.output_format
        converter_params['validate_output'] = not args.no_validate
        
        success = convert_single_dataset(
            dataset_name=dataset_name,
            input_path=input_path,
            output_path=output_path,
            converter_params=converter_params,
            verbose=not args.quiet
        )
        
        return 0 if success else 1
    
    # ==========================================
    # حالت: تبدیل همه
    # ==========================================
    if args.all:
        results = convert_all_datasets(verbose=not args.quiet)
        all_success = all(results.values())
        return 0 if all_success else 1
    
    # ==========================================
    # حالت: تبدیل تک دیتاست
    # ==========================================
    if not args.dataset:
        print("ERROR: You must specify --dataset, --config, or --all.")
        print("   For help: python prepare_data.py --help")
        return 1
    
    # تعیین input/output
    if not args.input or not args.output:
        # استفاده از config پیش‌فرض
        if args.dataset in DATASET_CONFIGS:
            config = DATASET_CONFIGS[args.dataset]
            input_path = args.input or config['input']
            output_path = args.output or config['output']
        else:
            print(f"ERROR: Dataset '{args.dataset}' not found in DATASET_CONFIGS.")
            print("   Please specify --input and --output.")
            return 1
    else:
        input_path = args.input
        output_path = args.output
    
    # استخراج converter parameters
    converter_class = ConverterFactory.get_converter_class(args.dataset)
    converter_params = converter_class.extract_params_from_args(args, prefix=True)
    
    # اضافه کردن generic params
    converter_params['verbose'] = not args.quiet
    converter_params['output_format'] = args.output_format
    converter_params['validate_output'] = not args.no_validate
    
    # تبدیل
    success = convert_single_dataset(
        dataset_name=args.dataset,
        input_path=input_path,
        output_path=output_path,
        converter_params=converter_params,
        verbose=not args.quiet
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
