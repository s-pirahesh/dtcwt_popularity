# -*- coding: utf-8 -*-
"""
Popularity Assessment - تخمین و ارزیابی محبوبیت محتوا
Main Pipeline for Content Popularity Assessment and Prediction

این برنامه محاسبات اصلی تخمین محبوبیت را انجام می‌دهد:
- روش‌های مختلف را ارزیابی می‌کند (AF, DTCWT+AF, DWT+AF, etc.)
- نتایج را با timestamp منحصر به فرد ذخیره می‌کند
- خروجی برای تحلیل و نمایش آماده می‌کند

جریان کار:
  1. این برنامه → محاسبات و ذخیره
  2. analyze_results.py → تحلیل و مقایسه
  3. show_results.py → نمایش متنی و گرافیکی

Author: Sajjad
Date: February 2026
"""

import sys
import argparse
from pathlib import Path

# اضافه کردن parent directory به path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation import (
    EvaluationConfig,
    get_movielens_config,
    get_youtube_config,
    get_youku_config,
    TemporalEvaluator
)
from data.loaders import MovieLensLoader
# YouTubeLoader و YoukuLoader هنوز پیاده‌سازی نشده‌اند - فعلاً فقط MovieLens پشتیبانی می‌شود
# Import methods (برخی ممکن است به dependencies اضافی نیاز داشته باشند)
try:
    from methods import (
        DTCWTAssessment,
        DWTAssessment,
        HybridAssessment,
        StatisticalAssessment
    )
    METHODS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some methods not available: {e}")
    print("⚠️  Install dependencies: pip install pywt dtcwt")
    DTCWTAssessment = None
    DWTAssessment = None
    HybridAssessment = None
    StatisticalAssessment = None
    METHODS_AVAILABLE = False

# Import baselines
try:
    from baselines import AccessFrequency, LFU, LRU, EWMA
    BASELINES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Baselines not available: {e}")
    AccessFrequency = LFU = LRU = EWMA = None
    BASELINES_AVAILABLE = False
from config import WAVELET_CONFIG, DATASETS


def create_methods_dict(config: EvaluationConfig) -> dict:
    """
    ایجاد dictionary روش‌ها
    
    Args:
        config: EvaluationConfig instance
    
    Returns:
        dict: {method_name: method_instance}
    """
    methods = {}
    
    # Baselines
    methods['AF'] = AccessFrequency()
    methods['LFU'] = LFU()
    methods['LRU'] = LRU()
    methods['EWMA'] = EWMA(alpha=0.3)
    
    # Wavelet methods
    dwt_level = config.wavelet_config['dwt']['level']
    dtcwt_level = config.wavelet_config['dtcwt']['level']
    
    methods['DWT+AF'] = DWTAssessment(
        wavelet=config.wavelet_config['dwt']['wavelet'],
        level=dwt_level,
        mode=config.wavelet_config['dwt']['mode']
    )
    
    methods['DTCWT+AF'] = DTCWTAssessment(
        level=dtcwt_level,
        biort=config.wavelet_config['dtcwt']['biort'],
        qshift=config.wavelet_config['dtcwt']['qshift']
    )
    
    # Advanced methods
    methods['Statistical'] = StatisticalAssessment()
    methods['Hybrid V3.0'] = HybridAssessment(version='3.0')
    methods['Hybrid V3.1'] = HybridAssessment(version='3.1')
    
    # فیلتر بر اساس config.methods
    if config.methods is not None:
        methods = {k: v for k, v in methods.items() if k in config.methods}
    
    return methods


def get_data_loader(dataset_name: str, data_path: str = None):
    """
    ایجاد data loader مناسب
    
    Args:
        dataset_name: نام دیتاست
    
    Returns:
        DataLoader instance
    """
    if dataset_name == 'movielens':
        config = DATASETS['movielens'].copy()
        if data_path:
            # اگر کاربر مسیری را در خط فرمان وارد کرده باشد، جایگزین مسیر پیش‌فرض می‌شود
            config['path'] = Path(data_path)
        return MovieLensLoader(config)
    
    elif dataset_name == 'youtube07':
        raise NotImplementedError(
            "YouTubeLoader هنوز پیاده‌سازی نشده است.\n"
            "فعلاً فقط MovieLens پشتیبانی می‌شود.\n"
            "استفاده کنید: --dataset movielens"
        )
    
    elif dataset_name == 'youku':
        raise NotImplementedError(
            "YoukuLoader هنوز پیاده‌سازی نشده است.\n"
            "فعلاً فقط MovieLens پشتیبانی می‌شود.\n"
            "استفاده کنید: --dataset movielens"
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")



def run_temporal_evaluation(dataset_name: str,
                           num_items: int = None,
                           start_date: str = None,
                           end_date: str = None,
                           window_size: int = 30,
                           prediction_horizon: int = 7,
                           methods: list = None,
                           final_format: str = 'csv',
                           parallel: bool = True,
                           num_cores: int = -1,
                           data_path: str = None,
                           **kwargs):
    """
    اجرای ارزیابی زمانی کامل
    
    Args:
        dataset_name: نام دیتاست ('movielens', 'youtube07', 'youku')
        num_items: تعداد آیتم‌ها (None = همه)
        start_date: تاریخ شروع ('YYYY-MM-DD' یا None)
        end_date: تاریخ پایان ('YYYY-MM-DD' یا None)
        window_size: اندازه پنجره آموزش (days)
        prediction_horizon: افق پیش‌بینی (days)
        methods: لیست نام روش‌ها (None = همه)
        final_format: فرمت نتایج نهایی ('csv' or 'parquet')
        parallel: استفاده از پردازش موازی
        num_cores: تعداد هسته‌ها (-1 = همه)
        **kwargs: سایر پارامترها
    """
    
    print("="*70)
    print("EXPERIMENT 2: TEMPORAL EVALUATION WITH SLIDING WINDOW")
    print("="*70)
    print(f"Dataset:         {dataset_name}")
    print(f"Items:           {num_items or 'all'}")
    print(f"Time Range:      {start_date or 'start'} to {end_date or 'end'}")
    print(f"Window Size:     {window_size} days")
    print(f"Horizon:         {prediction_horizon} days")
    print(f"Final Format:    {final_format.upper()}")
    print(f"Parallel:        {parallel}")
    print("="*70 + "\n")
    
    # 1. ایجاد پیکربندی
    if dataset_name == 'movielens':
        config = get_movielens_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    
    elif dataset_name == 'youtube07':
        config = get_youtube_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    
    elif dataset_name == 'youku':
        config = get_youku_config(
            num_items=num_items,
            start_date=start_date,
            end_date=end_date,
            window_size=window_size,
            prediction_horizon=prediction_horizon,
            methods=methods,
            final_format=final_format,
            parallel=parallel,
            num_cores=num_cores,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # چاپ پیکربندی
    if config.verbose:
        print(config)
    
    # 2. بارگذاری داده
    data_loader = get_data_loader(dataset_name, data_path=data_path)
    
    # 3. ایجاد روش‌ها
    methods_dict = create_methods_dict(config)
    
    print(f"\nMethods to evaluate: {list(methods_dict.keys())}\n")
    
    # 4. ایجاد evaluator
    evaluator = TemporalEvaluator(
        data_loader=data_loader,
        methods=methods_dict,
        config=config
    )
    
    # 5. اجرای ارزیابی
    evaluator.evaluate()
    
    # 6. نتیجه
    print("\n" + "="*70)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Results saved to: {config.output_dir}")
    print("\nOutput structure:")
    print(f"  {config.output_dir}/")
    print(f"    ├── detailed/          # Detailed scores per window/item")
    print(f"    ├── summary/           # Stratum summaries")
    print(f"    ├── comparison/        # Method comparison")
    print(f"    ├── metadata/          # Config and statistics")
    print(f"    └── visualization/     # (for future plots)")
    print("="*70 + "\n")
    
    return evaluator


def main():
    """تابع اصلی با argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='تخمین و ارزیابی محبوبیت محتوا - Popularity Assessment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌های استفاده:

  # تست سریع (100 آیتم، 1 ماه)
  python run_popularity_assessment.py movielens \
      --num-items 100 \
      --start-date 2023-08-01 \
      --end-date 2023-08-31

  # آزمایش متوسط (500 آیتم، 3 ماه)
  python run_popularity_assessment.py movielens \
      --num-items 500 \
      --start-date 2023-01-01 \
      --end-date 2023-03-31

  # ارزیابی کامل (1000 آیتم، 1 سال)
  python run_popularity_assessment.py movielens \
      --num-items 1000 \
      --start-date 2023-01-01 \
      --end-date 2023-12-31

  # همه داده‌ها (بدون فیلتر تاریخ)
  python run_popularity_assessment.py movielens \
      --num-items 1000

  # دیتاست YouTube
  python run_popularity_assessment.py youtube07 \
      --num-items 200 \
      --window-size 14 \
      --horizon 3

  # فرمت نهایی Parquet
  python run_popularity_assessment.py movielens \
      --num-items 1000 \
      --start-date 2023-01-01 \
      --end-date 2023-12-31 \
      --format parquet

نکته: برای محاسبه تعداد windows:
  num_windows = (end_date - start_date) - window_size - horizon + 1
        """
    )
    
    # Positional arguments
    parser.add_argument('dataset', type=str, choices=['movielens', 'youtube07', 'youku'],
                       help='Dataset name')
    
    # Optional arguments
    parser.add_argument('--num-items', type=int, default=None,
                       help='Number of items to evaluate (default: all)')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='تاریخ شروع (فرمت: YYYY-MM-DD، پیش‌فرض: از ابتدای دیتاست)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='تاریخ پایان (فرمت: YYYY-MM-DD، پیش‌فرض: تا انتهای دیتاست)')
    
    parser.add_argument('--window-size', type=int, default=30,
                       help='Training window size in days (default: 30)')
    
    parser.add_argument('--horizon', type=int, default=7,
                       help='Prediction horizon in days (default: 7)')
    
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='Methods to evaluate (default: all)')
    
    parser.add_argument('--format', type=str, default='csv',
                       choices=['csv', 'parquet'],
                       help='Final results format (default: csv). Intermediate always Parquet.')
    
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    
    parser.add_argument('--cores', type=int, default=-1,
                       help='Number of CPU cores to use (default: all)')
    
    parser.add_argument('--item-selection', type=str, default='top',
                       choices=['top', 'random', 'stratified'],
                       help='Item selection strategy (default: top)')

    parser.add_argument('--data-path', type=str, default=None,
                       help='مسیر فایل داده (override مسیر config.py)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # اجرای ارزیابی
    run_temporal_evaluation(
        dataset_name=args.dataset,
        num_items=args.num_items,
        start_date=args.start_date,
        end_date=args.end_date,
        window_size=args.window_size,
        prediction_horizon=args.horizon,
        methods=args.methods,
        final_format=args.format,
        parallel=not args.no_parallel,
        num_cores=args.cores,
        data_path=args.data_path,
        item_selection=args.item_selection,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()
