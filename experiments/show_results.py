# -*- coding: utf-8 -*-
"""
Show Results - نمایش نتایج ارزیابی محبوبیت
Display results of popularity assessment in both textual and graphical formats

این برنامه:
- نتایج را به صورت متنی نمایش می‌دهد
- نمودارهای گرافیکی تولید می‌کند
- هر دو فرمت CSV و Parquet را می‌خواند

Author: Sajjad
Date: February 2025
"""

import sys
import argparse
from pathlib import Path

# اضافه کردن parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.results_analyzer import ResultsAnalyzer
from evaluation.visualizer import ResultsVisualizer


def show_textual(analyzer: ResultsAnalyzer, 
                 top_percent: float = None,
                 stratum: str = None):
    """
    نمایش متنی نتایج
    
    Args:
        analyzer: ResultsAnalyzer instance
        top_percent: فیلتر top-k%
        stratum: فیلتر stratum
    """
    print("\n" + "="*80)
    print("نمایش متنی نتایج")
    print("="*80)
    
    # مقایسه روش‌ها
    comparison = analyzer.compare_methods(
        filter_top_percent=top_percent,
        filter_stratum=stratum
    )
    
    if len(comparison) == 0:
        print("داده‌ای یافت نشد")
        return
    
    # فیلترها
    if top_percent or stratum:
        filters = []
        if top_percent:
            filters.append(f"Top {top_percent}%")
        if stratum:
            filters.append(stratum)
        print(f"\nفیلترها: {', '.join(filters)}")
    
    print("\n" + "-"*80)
    print("مقایسه روش‌ها:")
    print("-"*80)
    print(comparison.to_string(index=False))
    
    # بهترین روش‌ها
    print("\n" + "-"*80)
    print("بهترین روش‌ها:")
    print("-"*80)
    
    best_spearman = comparison.nlargest(1, 'spearman').iloc[0]
    print(f"  بالاترین Spearman:  {best_spearman['method']:<20} ({best_spearman['spearman']:.4f})")
    
    best_mae = comparison.nsmallest(1, 'mae').iloc[0]
    print(f"  کمترین MAE:         {best_mae['method']:<20} ({best_mae['mae']:.4f})")
    
    best_ndcg = comparison.nlargest(1, 'ndcg').iloc[0]
    print(f"  بالاترین NDCG:      {best_ndcg['method']:<20} ({best_ndcg['ndcg']:.4f})")
    
    print("="*80 + "\n")


def show_graphical(analyzer: ResultsAnalyzer,
                  top_percent: float = None,
                  stratum: str = None,
                  show_plots: bool = False):
    """
    نمایش گرافیکی نتایج
    
    Args:
        analyzer: ResultsAnalyzer instance
        top_percent: فیلتر top-k%
        stratum: فیلتر stratum
        show_plots: نمایش نمودارها (علاوه بر ذخیره)
    """
    print("\n" + "="*80)
    print("تولید نمودارهای گرافیکی")
    print("="*80 + "\n")
    
    # ایجاد visualizer
    visualizer = ResultsVisualizer(analyzer)
    
    # تولید نمودارها
    methods = analyzer.available_methods[:5]  # 5 روش اول
    
    # 1. تکامل زمانی
    print("1. تکامل زمانی MAE...")
    visualizer.plot_temporal_evolution(
        methods=methods,
        metric='mae',
        save=True,
        show=show_plots
    )
    
    # 2. مقایسه روش‌ها
    print("2. مقایسه روش‌ها...")
    visualizer.plot_method_comparison(
        filter_top_percent=top_percent,
        filter_stratum=stratum,
        save=True,
        show=show_plots
    )
    
    # 3. مقایسه strata
    print("3. مقایسه strata...")
    visualizer.plot_stratum_comparison(
        methods=methods,
        metric='spearman_corr',
        save=True,
        show=show_plots
    )
    
    print(f"\n✓ نمودارها ذخیره شدند در: {visualizer.output_dir}")
    print("="*80 + "\n")


def show_detailed_stats(analyzer: ResultsAnalyzer):
    """
    نمایش آمار تفصیلی هر روش
    
    Args:
        analyzer: ResultsAnalyzer instance
    """
    print("\n" + "="*80)
    print("آمار تفصیلی روش‌ها")
    print("="*80)
    
    for method in analyzer.available_methods:
        print(f"\n{'-'*80}")
        print(f"روش: {method}")
        print(f"{'-'*80}")
        
        try:
            # آمار کلی
            metrics = analyzer.calculate_overall_metrics(method)
            
            print(f"  Spearman:    {metrics['spearman']:>8.4f}")
            print(f"  Kendall:     {metrics['kendall']:>8.4f}")
            print(f"  MAE:         {metrics['mae']:>8.2f}")
            print(f"  RMSE:        {metrics['rmse']:>8.2f}")
            print(f"  MAPE:        {metrics['mape']:>8.2f}%")
            print(f"  NDCG:        {metrics['ndcg']:>8.4f}")
            print(f"  Coverage:    {metrics['coverage']:>8.4f}")
            print(f"  نمونه‌ها:    {metrics['num_samples']:>8,}")
            
            # آمار در strata
            print(f"\n  عملکرد در Strata:")
            for stratum_name in ['cold_start', 'low', 'medium', 'high']:
                try:
                    stratum_metrics = analyzer.calculate_overall_metrics(
                        method, filter_stratum=stratum_name
                    )
                    if stratum_metrics:
                        print(f"    {stratum_name:<12} Spearman: {stratum_metrics['spearman']:>6.3f}  "
                              f"MAE: {stratum_metrics['mae']:>6.2f}")
                except:
                    pass
        
        except Exception as e:
            print(f"  خطا: {e}")
    
    print("\n" + "="*80 + "\n")


def main():
    """تابع اصلی"""
    
    parser = argparse.ArgumentParser(
        description='نمایش نتایج ارزیابی محبوبیت (متنی و گرافیکی)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌ها:
  # نمایش متنی ساده
  python show_results.py RESULTS_PATH
  
  # نمایش با فیلتر
  python show_results.py RESULTS_PATH --top-percent 20
  
  # تولید نمودارها
  python show_results.py RESULTS_PATH --graphical
  
  # نمایش و ذخیره نمودارها
  python show_results.py RESULTS_PATH --graphical --show
  
  # هر دو (متنی + گرافیکی)
  python show_results.py RESULTS_PATH --both
  
  # آمار تفصیلی
  python show_results.py RESULTS_PATH --detailed
        """
    )
    
    # آرگومان اصلی
    parser.add_argument('results_path', type=str,
                       help='مسیر دایرکتوری نتایج')
    
    # نوع نمایش
    parser.add_argument('--textual', action='store_true',
                       help='نمایش متنی (پیش‌فرض)')
    
    parser.add_argument('--graphical', action='store_true',
                       help='نمایش گرافیکی (تولید نمودارها)')
    
    parser.add_argument('--both', action='store_true',
                       help='هر دو (متنی + گرافیکی)')
    
    parser.add_argument('--detailed', action='store_true',
                       help='آمار تفصیلی هر روش')
    
    # فیلترها
    parser.add_argument('--top-percent', type=float, default=None,
                       help='فیلتر top-k درصد')
    
    parser.add_argument('--stratum', type=str, default=None,
                       choices=['cold_start', 'low', 'medium', 'high'],
                       help='فیلتر بر اساس stratum')
    
    # گزینه‌های نمودار
    parser.add_argument('--show', action='store_true',
                       help='نمایش نمودارها (علاوه بر ذخیره)')
    
    args = parser.parse_args()
    
    # بارگذاری analyzer
    print(f"\nبارگذاری نتایج از: {args.results_path}")
    try:
        analyzer = ResultsAnalyzer(Path(args.results_path))
    except Exception as e:
        print(f"خطا در بارگذاری: {e}")
        return
    
    # چاپ خلاصه
    analyzer.print_summary()
    
    # تعیین نوع نمایش
    if args.both:
        show_textual_flag = True
        show_graphical_flag = True
    elif args.graphical:
        show_textual_flag = False
        show_graphical_flag = True
    else:
        # پیش‌فرض: متنی
        show_textual_flag = True
        show_graphical_flag = False
    
    # نمایش متنی
    if show_textual_flag:
        show_textual(analyzer, args.top_percent, args.stratum)
    
    # نمایش گرافیکی
    if show_graphical_flag:
        show_graphical(analyzer, args.top_percent, args.stratum, args.show)
    
    # آمار تفصیلی
    if args.detailed:
        show_detailed_stats(analyzer)


if __name__ == '__main__':
    main()
