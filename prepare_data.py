#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
اسکریپت آماده‌سازی داده‌ها
تبدیل دیتاست‌های خام به فرمت استاندارد CSV

استفاده:
    # تک دیتاست - پایه
    python prepare_data.py --dataset movielens \
                          --input data/raw/movielens/ratings.csv \
                          --output data/datasets/movielens.csv

    # با تجمیع روزانه
    python prepare_data.py --dataset movielens \
                          --aggregate day \
                          --keep-rating

    # فیلتر rating بالا با تجمیع
    python prepare_data.py --dataset movielens \
                          --aggregate day \
                          --keep-rating \
                          --min-rating 4.0

    # چند فایل (مثل Taxi)
    python prepare_data.py --dataset nyc_taxi \
                          --input "data/raw/taxi_*.csv" \
                          --output data/datasets/nyc_taxi.csv

    # همه دیتاست‌ها (از config)
    python prepare_data.py --all

    # لیست دیتاست‌های موجود
    python prepare_data.py --list
"""
import argparse
import sys
from pathlib import Path
from glob import glob
import yaml

# Fix encoding for Windows
if sys.platform == 'win32':
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# اضافه کردن پروژه به path
sys.path.insert(0, str(Path(__file__).parent))

from data.converters import ConverterFactory
from config import DATA_DIR

# پیکربندی دیتاست‌ها
DATASET_CONFIGS = {
    'youtube07': {
        'input': 'data/raw/youtube07/*.txt',
        'output': 'data/datasets/youtube07.csv',
        'description': 'YouTube-07 video access traces',
    },
    'movielens': {
        'input': 'data/raw/movielens/ratings.csv',
        'output': 'data/datasets/movielens.csv',
        'description': 'MovieLens movie ratings (ml-32m: 32M ratings, 1995-2023)',
    },
    'foursquare': {
        'input': 'data/raw/foursquare/checkins.txt',
        'output': 'data/datasets/foursquare.csv',
        'description': 'Foursquare check-in data',
    },
    'higgs_twitter': {
        'input': 'data/raw/higgs/social_network.edgelist',
        'output': 'data/datasets/higgs_twitter.csv',
        'description': 'Higgs Twitter dataset',
    },
    'nyc_taxi': {
        'input': 'data/raw/taxi/yellow_tripdata_2024-*.csv',
        'output': 'data/datasets/nyc_taxi.csv',
        'description': 'NYC Taxi trip records (multiple months)',
    },
}


def parse_args():
    """پارس آرگومان‌های خط فرمان"""
    parser = argparse.ArgumentParser(
        description='تبدیل دیتاست‌های خام به فرمت استاندارد',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='نام دیتاست (youtube07, movielens, foursquare, higgs_twitter, nyc_taxi)'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        help='مسیر فایل(ها) خام (می‌تواند شامل wildcard باشد: *.csv)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        help='مسیر خروجی CSV'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='تبدیل همه دیتاست‌ها (از config)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='نمایش لیست دیتاست‌های موجود'
    )

    parser.add_argument(
        '--format', '-f',
        type=str,
        default='csv',
        choices=['csv', 'parquet', 'feather'],
        help='فرمت خروجی (پیش‌فرض: csv)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='حالت ساکت (بدون نمایش پیام‌ها)'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='غیرفعال کردن اعتبارسنجی'
    )

    # ==========================================
    # Converter-specific arguments
    # ==========================================

    # MovieLens specific
    parser.add_argument(
        '--aggregate',
        type=str,
        choices=['hour', 'day', 'week', 'none'],
        default='none',
        help='تجمیع زمانی (hour/day/week/none) - برای MovieLens'
    )

    parser.add_argument(
        '--keep-rating',
        action='store_true',
        help='نگهداری ستون rating - برای MovieLens'
    )

    parser.add_argument(
        '--keep-user',
        action='store_true',
        help='نگهداری ستون userId - برای MovieLens'
    )

    parser.add_argument(
        '--min-rating',
        type=float,
        default=None,
        help='حداقل rating برای فیلتر - برای MovieLens'
    )

    return parser.parse_args()


def list_datasets():
    """نمایش لیست دیتاست‌ها"""
    print("\n" + "=" * 60)
    print("دیتاست‌های قابل تبدیل:")
    print("=" * 60)

    for name, config in DATASET_CONFIGS.items():
        print(f"\n📊 {name}")
        print(f"   {config['description']}")
        print(f"   Input:  {config['input']}")
        print(f"   Output: {config['output']}")

    print("\n" + "=" * 60)
    print(f"Converter های موجود: {ConverterFactory.list_converters()}")
    print("=" * 60 + "\n")


def convert_single_dataset(dataset_name: str,
                           input_path: str,
                           output_path: str,
                           output_format: str = 'csv',
                           verbose: bool = True,
                           validate: bool = True,
                           **converter_kwargs):
    """
    تبدیل یک دیتاست

    Args:
        dataset_name: نام دیتاست
        input_path: مسیر ورودی (می‌تواند wildcard داشته باشد)
        output_path: مسیر خروجی
        output_format: فرمت خروجی
        verbose: نمایش پیام‌ها
        validate: اعتبارسنجی
        **converter_kwargs: پارامترهای اختصاصی converter
    """
    print("\n" + "=" * 60)
    print(f"تبدیل دیتاست: {dataset_name}")
    print("=" * 60 + "\n")

    # نمایش پارامترهای اضافی
    if converter_kwargs:
        print("پارامترهای Converter:")
        for key, value in converter_kwargs.items():
            if value is not None and value != 'none':
                print(f"  {key}: {value}")
        print()

    # بررسی wildcard در مسیر ورودی
    if '*' in input_path or '?' in input_path:
        input_files = sorted(glob(input_path))
        if not input_files:
            print(f"❌ هیچ فایلی با الگوی '{input_path}' یافت نشد!")
            return False
        print(f"✓ {len(input_files)} فایل یافت شد")
        input_path = input_files

    try:
        # ایجاد converter با پارامترهای اختصاصی
        converter = ConverterFactory.create(
            dataset_name,
            output_format=output_format,
            verbose=verbose,
            validate_output=validate,
            **converter_kwargs  # پاس دادن پارامترهای اضافی
        )

        # تبدیل
        df = converter.convert(input_path, output_path)

        print(f"\n✓ تبدیل موفق: {len(df)} رکورد")
        print(f"  خروجی: {output_path}\n")

        return True

    except Exception as e:
        print(f"\n❌ خطا در تبدیل: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def convert_all_datasets(output_format: str = 'csv',
                         verbose: bool = True,
                         validate: bool = True,
                         **converter_kwargs):
    """تبدیل همه دیتاست‌ها"""
    print("\n" + "🚀" * 30)
    print("شروع تبدیل همه دیتاست‌ها")
    print("🚀" * 30 + "\n")

    results = {}

    for dataset_name, config in DATASET_CONFIGS.items():
        success = convert_single_dataset(
            dataset_name=dataset_name,
            input_path=config['input'],
            output_path=config['output'],
            output_format=output_format,
            verbose=verbose,
            validate=validate,
            **converter_kwargs  # پاس دادن پارامترهای اضافی
        )
        results[dataset_name] = success

    # خلاصه نتایج
    print("\n" + "=" * 60)
    print("خلاصه نتایج:")
    print("=" * 60)

    for dataset_name, success in results.items():
        status = "✓ موفق" if success else "✗ ناموفق"
        print(f"  {dataset_name:<20} {status}")

    total = len(results)
    successful = sum(results.values())

    print("=" * 60)
    print(f"جمع: {successful}/{total} موفق")
    print("=" * 60 + "\n")


def main():
    """تابع اصلی"""
    args = parse_args()

    # نمایش لیست
    if args.list:
        list_datasets()
        return

    # تبدیل همه
    if args.all:
        convert_all_datasets(
            output_format=args.format,
            verbose=not args.quiet,
            validate=not args.no_validate,
            # پارامترهای اختصاصی Converter
            aggregate_by=None if args.aggregate == 'none' else args.aggregate,
            keep_rating=args.keep_rating,
            keep_user=args.keep_user,
            min_rating=args.min_rating,
        )
        return

    # تبدیل تک دیتاست
    if not args.dataset:
        print("❌ خطا: باید --dataset یا --all را مشخص کنید")
        print("   برای راهنما: python prepare_data.py --help")
        sys.exit(1)

    # استفاده از config اگر input/output مشخص نشده
    if not args.input or not args.output:
        if args.dataset in DATASET_CONFIGS:
            config = DATASET_CONFIGS[args.dataset]
            input_path = args.input or config['input']
            output_path = args.output or config['output']
        else:
            print(f"❌ خطا: دیتاست '{args.dataset}' در config یافت نشد")
            print("   لیست دیتاست‌ها: python prepare_data.py --list")
            sys.exit(1)
    else:
        input_path = args.input
        output_path = args.output

    # تبدیل
    success = convert_single_dataset(
        dataset_name=args.dataset,
        input_path=input_path,
        output_path=output_path,
        output_format=args.format,
        verbose=not args.quiet,
        validate=not args.no_validate,
        # پارامترهای اختصاصی Converter
        aggregate_by=None if args.aggregate == 'none' else args.aggregate,
        keep_rating=args.keep_rating,
        keep_user=args.keep_user,
        min_rating=args.min_rating,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()