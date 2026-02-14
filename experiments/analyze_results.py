# -*- coding: utf-8 -*-
"""
Analyze Results — تحلیل نتایج ارزیابی محبوبیت
===============================================
دو حالت کار:
  پیش‌فرض   : نمایش متریک‌های از پیش محاسبه‌شده (بدون recompute)
  --recompute: بازمحاسبه کامل 4-Layer Protocol از raw detailed scores

Author: Sajjad
Date: February 2025 (refactored)
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.results_analyzer import ResultsAnalyzer
from evaluation.visualizer import ResultsVisualizer


# =============================================================================
# Run discovery
# =============================================================================

def list_available_runs(dataset: str = 'movielens'):
    results_dir = Path(__file__).parent.parent / 'results' / dataset
    if not results_dir.exists():
        return []
    return sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / 'metadata' / 'config.json').exists()
    )


def print_available_runs(dataset: str = 'movielens'):
    runs = list_available_runs(dataset)
    if not runs:
        print(f"\nNo runs found for dataset: {dataset}")
        print(f"  Run: python run_popularity_assessment.py {dataset} --num-items 100")
        return

    print(f"\n{'='*70}")
    print(f"AVAILABLE RUNS — {dataset.upper()}")
    print(f"{'='*70}")

    for i, run_dir in enumerate(runs, 1):
        try:
            analyzer = ResultsAnalyzer(run_dir)
            cfg = analyzer.config
            proto = (run_dir / 'protocol').exists() and any(
                (run_dir / 'protocol').iterdir()
            )
            print(f"\n{i}. {run_dir.name}")
            print(f"   Window:   {cfg.get('window_size', 'N/A')} days")
            print(f"   Items:    {cfg.get('num_items', 'all')}")
            print(f"   Methods:  {', '.join(analyzer.available_methods)}")
            print(f"   Protocol: {'✓' if proto else '✗ (--recompute needed)'}")
            print(f"   Path:     {run_dir}")
        except Exception as e:
            print(f"\n{i}. {run_dir.name}  [Error: {e}]")

    print(f"\n{'='*70}\n")


# =============================================================================
# Analysis modes
# =============================================================================

def print_summary_analysis(analyzer: ResultsAnalyzer,
                           top_percent, stratum, start_date, end_date,
                           recompute: bool):
    print("\n" + "="*70)
    print("SUMMARY ANALYSIS — 4-LAYER FROZEN PROTOCOL")
    print("="*70)

    comparison = analyzer.compare_methods(
        filter_top_percent=top_percent,
        filter_stratum=stratum,
        start_date=start_date,
        end_date=end_date,
        recompute=recompute,
    )

    if len(comparison) == 0:
        print("No data available")
        return

    # Filters info
    filters = [f for f in [
        f"Top {top_percent}%" if top_percent else None,
        stratum,
        f"{start_date or 'start'} → {end_date or 'end'}" if (start_date or end_date) else None,
        "RECOMPUTED from raw scores" if recompute else "from pre-computed protocol files",
    ] if f]
    print("Filters: " + (", ".join(filters) if filters else "None (all data)"))
    print()

    # Select columns to display — multi-K Decision metrics first
    k_cols = []
    for k in analyzer.K_VALUES:
        for m in [f'ndcg@{k}', f'chr@{k}', f'rsi@{k}']:
            if m in comparison.columns:
                k_cols.append(m)

    diag_cols = [c for c in ['spearman_rho', 'kendall_tau', 'mae',
                              'robustness_distortion'] if c in comparison.columns]

    display_cols = ['method'] + k_cols + diag_cols
    display_cols = [c for c in display_cols if c in comparison.columns]

    print(comparison[display_cols].to_string(index=False))

    # Best method highlights
    print("\n" + "-"*70)
    print("HIGHLIGHTS:")

    def _best(col, largest=True, label=''):
        if col not in comparison.columns:
            return
        sub = comparison.dropna(subset=[col])
        if sub.empty:
            return
        row = sub.nlargest(1, col).iloc[0] if largest else sub.nsmallest(1, col).iloc[0]
        print(f"  {label:<30} {row['method']} ({row[col]:.4f})")

    _best('ndcg@10',               largest=True,  label='Highest NDCG@10:')
    _best('chr@10',                largest=True,  label='Highest CHR@10:')
    _best('spearman_rho',          largest=True,  label='Highest Spearman:')
    _best('rsi@10',                largest=True,  label='Highest Stability (RSI@10):')
    _best('robustness_distortion', largest=False, label='Best Robustness (min ΔRank):')
    _best('mae',                   largest=False, label='Lowest MAE:')

    print("="*70 + "\n")


def print_detailed_analysis(analyzer: ResultsAnalyzer,
                            top_percent, stratum, start_date, end_date,
                            recompute: bool):
    print("\n" + "="*70)
    print("DETAILED ANALYSIS — PER METHOD")
    print("="*70)

    for method in analyzer.available_methods:
        print(f"\nMethod: {method}")
        print("-"*50)

        m = analyzer.calculate_overall_metrics(
            method, top_percent, stratum, start_date, end_date,
            recompute=recompute
        )
        if not m:
            print("  No data available")
            continue

        # Decision Layer
        print("  [Layer 1 — Decision]")
        for k in analyzer.K_VALUES:
            ndcg_v = m.get(f'ndcg@{k}', float('nan'))
            chr_v  = m.get(f'chr@{k}',  float('nan'))
            print(f"    NDCG@{k:<4} {ndcg_v:>7.4f}    CHR@{k:<4} {chr_v:>7.4f}")

        # Diagnostic Layer
        print("  [Layer 2 — Diagnostic]")
        print(f"    Spearman ρ : {m.get('spearman_rho', float('nan')):>8.4f}")
        print(f"    Kendall τ  : {m.get('kendall_tau',  float('nan')):>8.4f}")
        print(f"    MAE        : {m.get('mae',          float('nan')):>8.2f}")

        # Stability Layer
        print("  [Layer 3 — Stability (RSI, mean over windows)]")
        for k in analyzer.K_VALUES:
            rsi_v = m.get(f'rsi@{k}', float('nan'))
            print(f"    RSI@{k:<4}  {rsi_v:>8.4f}")

        # Robustness Layer
        print("  [Layer 4 — Robustness]")
        print(f"    Avg ΔRank  : {m.get('robustness_distortion', float('nan')):>8.2f}")

        print(f"  Windows analysed: {m.get('num_samples', 'N/A')}")

    print("="*70 + "\n")


def print_temporal_analysis(analyzer: ResultsAnalyzer, recompute: bool):
    print("\n" + "="*70)
    print("TEMPORAL ANALYSIS")
    print("="*70)

    for method in analyzer.available_methods:
        print(f"\nMethod: {method}")
        print("-"*50)

        for metric in ['spearman_rho', 'ndcg@10', 'rsi@10']:
            try:
                evo = analyzer.get_temporal_evolution(method, metric,
                                                      recompute=recompute)
                col = metric
                print(f"  {metric}:")
                print(f"    Windows: {len(evo)}  "
                      f"First: {evo[col].iloc[0]:.4f}  "
                      f"Last: {evo[col].iloc[-1]:.4f}  "
                      f"Mean: {evo[col].mean():.4f}  "
                      f"Std: {evo[col].std():.4f}")
            except Exception as e:
                print(f"  {metric}: N/A ({e})")

    print("="*70 + "\n")


def print_comparison_analysis(analyzer: ResultsAnalyzer, recompute: bool):
    print("\n" + "="*70)
    print("STRATUM-LEVEL COMPARISON")
    print("="*70)

    strata = ['cold_start', 'low', 'medium', 'high']
    for stratum in strata:
        print(f"\n{stratum.upper()}")
        print("-"*50)
        comparison = analyzer.compare_methods(
            filter_stratum=stratum, recompute=recompute
        )
        if len(comparison) == 0:
            print("  No data available")
            continue

        show_cols = ['method']
        for col in ['ndcg@10', 'chr@10', 'spearman_rho', 'rsi@10',
                    'robustness_distortion']:
            if col in comparison.columns:
                show_cols.append(col)

        print(comparison[show_cols].head(10).to_string(index=False))

    print("\n" + "="*70 + "\n")


# =============================================================================
# Top-level dispatcher
# =============================================================================

def analyze_run(run_path: str,
                mode: str = 'summary',
                top_percent: float = None,
                stratum: str = None,
                start_date: str = None,
                end_date: str = None,
                recompute: bool = False,
                visualize: bool = False,
                show_plots: bool = False,
                save_recomputed: bool = False):
    """
    تحلیل یک run directory.

    Args:
        recompute:       بازمحاسبه 4-Layer از raw scores (پیش‌فرض False)
        save_recomputed: اگر recompute=True، نتایج بازمحاسبه را ذخیره کن
    """
    print(f"\nLoading results from: {run_path}")
    analyzer = ResultsAnalyzer(Path(run_path))
    analyzer.print_summary()

    if recompute:
        print("Mode: RECOMPUTE — recalculating 4-Layer metrics from raw scores")
        if save_recomputed:
            print("      Results will be saved to protocol/ directory")
    else:
        print("Mode: DISPLAY-ONLY — reading pre-computed protocol metrics")
    print()

    # Save recomputed results if requested
    if recompute and save_recomputed:
        _save_recomputed(analyzer, run_path)

    if mode == 'summary':
        print_summary_analysis(analyzer, top_percent, stratum,
                               start_date, end_date, recompute)
    elif mode == 'detailed':
        print_detailed_analysis(analyzer, top_percent, stratum,
                                start_date, end_date, recompute)
    elif mode == 'temporal':
        print_temporal_analysis(analyzer, recompute)
    elif mode == 'comparison':
        print_comparison_analysis(analyzer, recompute)

    if visualize:
        print("\nGenerating visualizations...")
        visualizer = ResultsVisualizer(analyzer)
        visualizer.create_summary_report(
            filter_top_percent=top_percent,
            filter_stratum=stratum,
            save=True,
            show=show_plots
        )


def _save_recomputed(analyzer: ResultsAnalyzer, run_path: str):
    """ذخیره نتایج بازمحاسبه به protocol/"""
    import pandas as pd
    proto_dir = Path(run_path) / 'protocol'
    proto_dir.mkdir(exist_ok=True)

    for method in analyzer.available_methods:
        try:
            df = analyzer.recompute_protocol_metrics(method)
            if len(df) == 0:
                continue
            fp = proto_dir / f"{method}_protocol.csv"
            df.to_csv(fp, index=False, encoding='utf-8')
            print(f"  ✓ Saved recomputed protocol: {fp.name}")
        except Exception as e:
            print(f"  ✗ Failed for {method}: {e}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze saved evaluation results (Frozen 4-Layer Protocol)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available runs
  python analyze_results.py --list movielens

  # Display pre-computed protocol metrics (default)
  python analyze_results.py results/movielens/run_20250201_143052

  # Re-compute all metrics from raw scores
  python analyze_results.py results/movielens/run_20250201_143052 --recompute

  # Recompute + save results to protocol/ folder
  python analyze_results.py results/movielens/run_20250201_143052 --recompute --save-recomputed

  # Detailed per-method view
  python analyze_results.py RESULTS_PATH --mode detailed

  # Stratum-level breakdown
  python analyze_results.py RESULTS_PATH --mode comparison

  # Filter: top 20% items, medium popularity stratum
  python analyze_results.py RESULTS_PATH --top-percent 20 --stratum medium

  # With visualization
  python analyze_results.py RESULTS_PATH --visualize --show
        """
    )

    parser.add_argument('run_path', type=str, nargs='?',
                        help='Path to run directory')
    parser.add_argument('--list', type=str, nargs='?', const='movielens',
                        metavar='DATASET',
                        help='List available runs for dataset (default: movielens)')

    # Analysis mode
    parser.add_argument('--mode', type=str, default='summary',
                        choices=['summary', 'detailed', 'temporal', 'comparison'],
                        help='Analysis mode (default: summary)')

    # Core new flag
    parser.add_argument('--recompute', action='store_true',
                        help='Recompute 4-Layer metrics from raw detailed scores '
                             '(default: display pre-computed protocol files)')
    parser.add_argument('--save-recomputed', action='store_true',
                        help='Save recomputed metrics to protocol/ (requires --recompute)')

    # Filters
    parser.add_argument('--top-percent', type=float, default=None,
                        help='Filter: top-k%% items by popularity (e.g. 20)')
    parser.add_argument('--stratum', type=str, default=None,
                        choices=['cold_start', 'low', 'medium', 'high'],
                        help='Filter by popularity stratum')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Filter start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='Filter end date (YYYY-MM-DD)')

    # Visualization
    parser.add_argument('--visualize', action='store_true',
                        help='Generate and save plots')
    parser.add_argument('--show', action='store_true',
                        help='Display plots interactively (in addition to saving)')

    args = parser.parse_args()

    if args.list is not None:
        print_available_runs(args.list)
        return

    if not args.run_path:
        print("Error: run_path required (or use --list)")
        parser.print_help()
        return

    if args.save_recomputed and not args.recompute:
        print("Warning: --save-recomputed requires --recompute; enabling --recompute.")
        args.recompute = True

    analyze_run(
        run_path=args.run_path,
        mode=args.mode,
        top_percent=args.top_percent,
        stratum=args.stratum,
        start_date=args.start_date,
        end_date=args.end_date,
        recompute=args.recompute,
        visualize=args.visualize,
        show_plots=args.show,
        save_recomputed=args.save_recomputed,
    )


if __name__ == '__main__':
    main()