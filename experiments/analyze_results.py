# -*- coding: utf-8 -*-
"""
Analyze Saved Results - Without Re-running Simulation
Quick analysis and visualization of evaluation results
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


def list_available_runs(dataset: str = 'movielens'):
    """
    لیست runهای موجود
    
    Args:
        dataset: نام دیتاست
    """
    results_dir = Path(__file__).parent.parent / 'results' / dataset
    
    if not results_dir.exists():
        print(f"No results found for dataset: {dataset}")
        return []
    
    runs = []
    for run_dir in sorted(results_dir.iterdir()):
        if run_dir.is_dir():
            # بررسی اینکه دایرکتوری معتبر است
            if (run_dir / 'metadata' / 'config.json').exists():
                runs.append(run_dir)
    
    return runs


def print_available_runs(dataset: str = 'movielens'):
    """چاپ runهای موجود"""
    
    runs = list_available_runs(dataset)
    
    if len(runs) == 0:
        print(f"\nNo runs found for dataset: {dataset}")
        print("Run an evaluation first:")
        print(f"  python exp2_temporal_evaluation.py {dataset} --num-items 100")
        return
    
    print(f"\n{'='*70}")
    print(f"AVAILABLE RUNS - {dataset}")
    print(f"{'='*70}")
    
    for i, run_dir in enumerate(runs, 1):
        # بارگذاری config
        try:
            analyzer = ResultsAnalyzer(run_dir)
            config = analyzer.config
            
            print(f"\n{i}. {run_dir.name}")
            print(f"   Window Size:   {config.get('window_size', 'N/A')} days")
            print(f"   Horizon:       {config.get('prediction_horizon', 'N/A')} days")
            print(f"   Items:         {config.get('num_items', 'all')}")
            print(f"   Methods:       {len(analyzer.available_methods)}")
            print(f"   Path:          {run_dir}")
        
        except Exception as e:
            print(f"\n{i}. {run_dir.name}")
            print(f"   Error: {e}")
    
    print(f"\n{'='*70}\n")


def analyze_run(run_path: str,
               mode: str = 'summary',
               top_percent: float = None,
               stratum: str = None,
               start_date: str = None,
               end_date: str = None,
               visualize: bool = False,
               show_plots: bool = False):
    """
    تحلیل یک run
    
    Args:
        run_path: مسیر run directory
        mode: 'summary', 'detailed', 'temporal', 'comparison'
        top_percent: فیلتر top-k%
        stratum: فیلتر stratum
        start_date: تاریخ شروع
        end_date: تاریخ پایان
        visualize: ایجاد نمودارها
        show_plots: نمایش نمودارها
    """
    # بارگذاری analyzer
    print(f"\nLoading results from: {run_path}")
    analyzer = ResultsAnalyzer(Path(run_path))
    
    # چاپ خلاصه
    analyzer.print_summary()
    
    # تحلیل بر اساس mode
    if mode == 'summary':
        print_summary_analysis(analyzer, top_percent, stratum, start_date, end_date)
    
    elif mode == 'detailed':
        print_detailed_analysis(analyzer, top_percent, stratum, start_date, end_date)
    
    elif mode == 'temporal':
        print_temporal_analysis(analyzer)
    
    elif mode == 'comparison':
        print_comparison_analysis(analyzer)
    
    # Visualization
    if visualize:
        print("\nGenerating visualizations...")
        visualizer = ResultsVisualizer(analyzer)
        visualizer.create_summary_report(
            filter_top_percent=top_percent,
            filter_stratum=stratum,
            save=True,
            show=show_plots
        )


def print_summary_analysis(analyzer, top_percent, stratum, start_date, end_date):
    """چاپ تحلیل خلاصه"""
    
    print("\n" + "="*70)
    print("SUMMARY ANALYSIS")
    print("="*70)
    
    # مقایسه روش‌ها
    comparison = analyzer.compare_methods(
        filter_top_percent=top_percent,
        filter_stratum=stratum,
        start_date=start_date,
        end_date=end_date
    )
    
    if len(comparison) == 0:
        print("No data available")
        return
    
    # فیلترها
    filters = []
    if top_percent:
        filters.append(f"Top {top_percent}%")
    if stratum:
        filters.append(stratum)
    if start_date or end_date:
        date_range = f"{start_date or 'start'} to {end_date or 'end'}"
        filters.append(date_range)
    
    if filters:
        print(f"Filters: {', '.join(filters)}")
    else:
        print("Filters: None (all data)")
    
    print()
    
    # چاپ جدول
    print(comparison.to_string(index=False))
    
    # بهترین روش‌ها
    print("\n" + "-"*70)
    print("BEST METHODS:")
    print(f"  Highest Spearman:  {comparison.iloc[0]['method']} ({comparison.iloc[0]['spearman']:.4f})")
    print(f"  Lowest MAE:        {comparison.nsmallest(1, 'mae').iloc[0]['method']} ({comparison['mae'].min():.4f})")
    print(f"  Highest NDCG:      {comparison.nlargest(1, 'ndcg').iloc[0]['method']} ({comparison['ndcg'].max():.4f})")
    print("="*70 + "\n")


def print_detailed_analysis(analyzer, top_percent, stratum, start_date, end_date):
    """چاپ تحلیل تفصیلی"""
    
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    for method in analyzer.available_methods:
        print(f"\nMethod: {method}")
        print("-"*70)
        
        metrics = analyzer.calculate_overall_metrics(
            method,
            filter_top_percent=top_percent,
            filter_stratum=stratum,
            start_date=start_date,
            end_date=end_date
        )
        
        if not metrics:
            print("  No data available")
            continue
        
        print(f"  Spearman:    {metrics['spearman']:.4f}")
        print(f"  Kendall:     {metrics['kendall']:.4f}")
        print(f"  MAE:         {metrics['mae']:.4f}")
        print(f"  RMSE:        {metrics['rmse']:.4f}")
        print(f"  MAPE:        {metrics['mape']:.4f}%")
        print(f"  NDCG:        {metrics['ndcg']:.4f}")
        print(f"  Coverage:    {metrics['coverage']:.4f}")
        print(f"  Samples:     {metrics['num_samples']:,}")
    
    print("="*70 + "\n")


def print_temporal_analysis(analyzer):
    """چاپ تحلیل زمانی"""
    
    print("\n" + "="*70)
    print("TEMPORAL ANALYSIS")
    print("="*70)
    
    for method in analyzer.available_methods:
        print(f"\nMethod: {method}")
        print("-"*70)
        
        try:
            evolution = analyzer.get_temporal_evolution(method, 'mae')
            
            print(f"  Start Date:    {evolution['date'].min()}")
            print(f"  End Date:      {evolution['date'].max()}")
            print(f"  Time Steps:    {len(evolution)}")
            print(f"  Initial MAE:   {evolution['mae'].iloc[0]:.4f}")
            print(f"  Final MAE:     {evolution['mae'].iloc[-1]:.4f}")
            print(f"  Average MAE:   {evolution['mae'].mean():.4f}")
            print(f"  Std MAE:       {evolution['mae'].std():.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
    
    print("="*70 + "\n")


def print_comparison_analysis(analyzer):
    """چاپ تحلیل مقایسه‌ای (همه strata)"""
    
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS (BY STRATUM)")
    print("="*70)
    
    strata = ['cold_start', 'low', 'medium', 'high']
    
    for stratum in strata:
        print(f"\n{stratum.upper()}")
        print("-"*70)
        
        comparison = analyzer.compare_methods(filter_stratum=stratum)
        
        if len(comparison) == 0:
            print("  No data available")
            continue
        
        # نمایش top 5
        print(comparison.head(5)[['method', 'spearman', 'mae', 'ndcg']].to_string(index=False))
    
    print("\n" + "="*70 + "\n")


def main():
    """تابع اصلی"""
    
    parser = argparse.ArgumentParser(
        description='Analyze saved evaluation results without re-running simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available runs
  python analyze_results.py --list movielens
  
  # Quick summary
  python analyze_results.py results/movielens/w30_h7_n1000_top_20250201_143052
  
  # Summary with filters
  python analyze_results.py RESULTS_PATH --top-percent 20 --stratum medium
  
  # Detailed analysis
  python analyze_results.py RESULTS_PATH --mode detailed
  
  # With visualization
  python analyze_results.py RESULTS_PATH --visualize
  
  # Show plots
  python analyze_results.py RESULTS_PATH --visualize --show
        """
    )
    
    # Main argument
    parser.add_argument('run_path', type=str, nargs='?',
                       help='Path to run directory')
    
    # List runs
    parser.add_argument('--list', type=str, nargs='?', const='movielens',
                       help='List available runs for dataset')
    
    # Analysis mode
    parser.add_argument('--mode', type=str, default='summary',
                       choices=['summary', 'detailed', 'temporal', 'comparison'],
                       help='Analysis mode (default: summary)')
    
    # Filters
    parser.add_argument('--top-percent', type=float, default=None,
                       help='Filter top-k percent (e.g., 20 for top 20%%)')
    
    parser.add_argument('--stratum', type=str, default=None,
                       choices=['cold_start', 'low', 'medium', 'high'],
                       help='Filter by stratum')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    
    # Visualization
    parser.add_argument('--visualize', action='store_true',
                       help='Generate plots')
    
    parser.add_argument('--show', action='store_true',
                       help='Show plots (in addition to saving)')
    
    args = parser.parse_args()
    
    # List runs
    if args.list is not None:
        print_available_runs(args.list)
        return
    
    # Need run_path
    if not args.run_path:
        print("Error: run_path required (or use --list)")
        parser.print_help()
        return
    
    # Analyze
    analyze_run(
        run_path=args.run_path,
        mode=args.mode,
        top_percent=args.top_percent,
        stratum=args.stratum,
        start_date=args.start_date,
        end_date=args.end_date,
        visualize=args.visualize,
        show_plots=args.show
    )


if __name__ == '__main__':
    main()
