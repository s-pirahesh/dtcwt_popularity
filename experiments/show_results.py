# -*- coding: utf-8 -*-
"""
Show Results — نمایش نتایج ارزیابی محبوبیت
============================================
این فایل **فقط** نمایش می‌کند — هیچ محاسبه‌ای انجام نمی‌دهد.
برای بازمحاسبه از analyze_results.py --recompute استفاده کنید.

نمایش:
  متنی   : جدول multi-K (NDCG, CHR, RSI) + Diagnostics + Robustness
  گرافیکی: نمودارهای مبتنی بر 4-Layer Protocol

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
# Textual display
# =============================================================================

def show_textual(analyzer: ResultsAnalyzer,
                 top_percent: float = None,
                 stratum: str = None):
    """نمایش متنی جدول multi-K Protocol"""

    print("\n" + "="*80)
    print("RESULTS — FROZEN 4-LAYER EVALUATION PROTOCOL")
    print("="*80)

    comparison = analyzer.compare_methods(
        filter_top_percent=top_percent,
        filter_stratum=stratum,
        recompute=False,        # display-only
    )

    if len(comparison) == 0:
        print("داده‌ای یافت نشد.")
        print("نکته: ممکن است protocol metrics هنوز محاسبه نشده باشد.")
        print("      اجرا کنید: python analyze_results.py RESULTS_PATH --recompute --save-recomputed")
        return

    filters = [f for f in [
        f"Top {top_percent}%" if top_percent else None,
        stratum,
    ] if f]
    if filters:
        print(f"\nفیلترها: {', '.join(filters)}")

    # ----- Layer 1 & 3: multi-K table ----------------------------------------
    print("\n[Layer 1 — Decision  &  Layer 3 — Stability]")
    print("-"*80)
    k_cols = ['method']
    for k in analyzer.K_VALUES:
        for m in [f'ndcg@{k}', f'chr@{k}', f'rsi@{k}']:
            if m in comparison.columns:
                k_cols.append(m)
    print(comparison[k_cols].to_string(index=False))

    # ----- Layer 2: Diagnostics -----------------------------------------------
    print("\n[Layer 2 — Diagnostic]")
    print("-"*80)
    diag_cols = ['method'] + [c for c in
                               ['spearman_rho', 'kendall_tau', 'mae']
                               if c in comparison.columns]
    print(comparison[diag_cols].to_string(index=False))

    # ----- Layer 4: Robustness ------------------------------------------------
    if 'robustness_distortion' in comparison.columns:
        print("\n[Layer 4 — Robustness  (lower ΔRank is better)]")
        print("-"*80)
        rob_df = comparison[['method', 'robustness_distortion']].copy()
        rob_df = rob_df.sort_values('robustness_distortion')
        print(rob_df.to_string(index=False))

    # ----- Highlights ---------------------------------------------------------
    print("\n" + "-"*80)
    print("HIGHLIGHTS:")

    def _hl(col, largest, label):
        if col not in comparison.columns:
            return
        sub = comparison.dropna(subset=[col])
        if sub.empty:
            return
        row = (sub.nlargest(1, col) if largest else sub.nsmallest(1, col)).iloc[0]
        print(f"  {label:<35} {row['method']} ({row[col]:.4f})")

    _hl('ndcg@10',               largest=True,  label='Highest NDCG@10:')
    _hl('chr@10',                largest=True,  label='Highest Cache Hit Ratio@10:')
    _hl('spearman_rho',          largest=True,  label='Highest Spearman ρ:')
    _hl('rsi@10',                largest=True,  label='Most Stable (RSI@10):')
    _hl('robustness_distortion', largest=False, label='Most Robust (min ΔRank):')
    _hl('mae',                   largest=False, label='Lowest MAE:')

    print("="*80 + "\n")


def show_detailed_stats(analyzer: ResultsAnalyzer):
    """نمایش آمار تفصیلی هر روش (از pre-computed protocol)"""

    print("\n" + "="*80)
    print("آمار تفصیلی روش‌ها — 4-Layer Protocol")
    print("="*80)

    for method in analyzer.available_methods:
        print(f"\n{'—'*80}")
        print(f"روش: {method}")
        print(f"{'—'*80}")

        m = analyzer.calculate_overall_metrics(method, recompute=False)
        if not m:
            print("  داده‌ای یافت نشد (protocol metrics موجود نیست)")
            continue

        print("  [Layer 1 — Decision]")
        for k in analyzer.K_VALUES:
            ndcg_v = m.get(f'ndcg@{k}', float('nan'))
            chr_v  = m.get(f'chr@{k}',  float('nan'))
            print(f"    NDCG@{k:<4} {ndcg_v:>7.4f}    CHR@{k:<4} {chr_v:>7.4f}")

        print("  [Layer 2 — Diagnostic]")
        print(f"    Spearman ρ : {m.get('spearman_rho', float('nan')):>8.4f}")
        print(f"    Kendall τ  : {m.get('kendall_tau',  float('nan')):>8.4f}")
        print(f"    MAE        : {m.get('mae',          float('nan')):>8.2f}")

        print("  [Layer 3 — Stability]")
        for k in analyzer.K_VALUES:
            rsi_v = m.get(f'rsi@{k}', float('nan'))
            print(f"    RSI@{k:<4}  {rsi_v:>8.4f}")

        print("  [Layer 4 — Robustness]")
        print(f"    Avg ΔRank  : {m.get('robustness_distortion', float('nan')):>8.2f}")

        # Stratum breakdown (from stratum summary)
        print("  [Stratum Breakdown — Spearman ρ]")
        for stratum_name in ['cold_start', 'low', 'medium', 'high']:
            try:
                sm = analyzer.calculate_overall_metrics(
                    method, filter_stratum=stratum_name, recompute=False
                )
                if sm:
                    spr = sm.get('spearman_rho', sm.get('spearman', float('nan')))
                    ndcg = sm.get('ndcg@10', sm.get('ndcg', float('nan')))
                    print(f"    {stratum_name:<12} ρ={spr:>6.3f}  NDCG@10={ndcg:>6.4f}")
            except Exception:
                pass

    print("\n" + "="*80 + "\n")


# =============================================================================
# Graphical display
# =============================================================================

def show_graphical(analyzer: ResultsAnalyzer,
                   top_percent: float = None,
                   stratum: str = None,
                   show_plots: bool = False):
    """نمایش گرافیکی نتایج — بر اساس متریک‌های 4-Layer Protocol"""

    print("\n" + "="*80)
    print("تولید نمودارهای گرافیکی — Frozen Evaluation Protocol")
    print("="*80 + "\n")

    visualizer = ResultsVisualizer(analyzer)
    methods = analyzer.available_methods[:6]    # حداکثر 6 روش

    # 1. تکامل زمانی Spearman
    print("1. تکامل زمانی Spearman ρ ...")
    try:
        visualizer.plot_temporal_evolution(
            methods=methods,
            metric='spearman_rho',
            save=True, show=show_plots
        )
    except Exception as e:
        print(f"   ✗ {e}")

    # 2. تکامل زمانی NDCG@10
    print("2. تکامل زمانی NDCG@10 ...")
    try:
        visualizer.plot_temporal_evolution(
            methods=methods,
            metric='ndcg@10',
            save=True, show=show_plots
        )
    except Exception as e:
        print(f"   ✗ {e}")

    # 3. مقایسه روش‌ها (bar chart)
    print("3. مقایسه روش‌ها ...")
    try:
        visualizer.plot_method_comparison(
            filter_top_percent=top_percent,
            filter_stratum=stratum,
            save=True, show=show_plots
        )
    except Exception as e:
        print(f"   ✗ {e}")

    # 4. Stratum breakdown
    print("4. مقایسه Strata ...")
    try:
        visualizer.plot_stratum_comparison(
            methods=methods,
            metric='spearman_corr',
            save=True, show=show_plots
        )
    except Exception as e:
        print(f"   ✗ {e}")

    print(f"\n✓ نمودارها در: {visualizer.output_dir}")
    print("="*80 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='نمایش نتایج ارزیابی (فقط نمایش — بدون بازمحاسبه)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
نکته مهم:
  این فایل فقط نتایج از پیش محاسبه‌شده را نمایش می‌دهد.
  برای بازمحاسبه:
      python analyze_results.py RESULTS_PATH --recompute [--save-recomputed]

مثال‌ها:
  # نمایش متنی ساده
  python show_results.py RESULTS_PATH

  # نمایش با فیلتر
  python show_results.py RESULTS_PATH --top-percent 20 --stratum medium

  # نمودارها
  python show_results.py RESULTS_PATH --graphical --show

  # هر دو
  python show_results.py RESULTS_PATH --both

  # آمار تفصیلی
  python show_results.py RESULTS_PATH --detailed
        """
    )

    parser.add_argument('results_path', type=str,
                        help='مسیر دایرکتوری نتایج')

    parser.add_argument('--textual',  action='store_true',
                        help='نمایش متنی (پیش‌فرض)')
    parser.add_argument('--graphical', action='store_true',
                        help='نمایش گرافیکی (تولید نمودارها)')
    parser.add_argument('--both',     action='store_true',
                        help='هر دو (متنی + گرافیکی)')
    parser.add_argument('--detailed', action='store_true',
                        help='آمار تفصیلی 4-Layer هر روش')

    parser.add_argument('--top-percent', type=float, default=None,
                        help='فیلتر top-k درصد')
    parser.add_argument('--stratum', type=str, default=None,
                        choices=['cold_start', 'low', 'medium', 'high'],
                        help='فیلتر بر اساس stratum')

    parser.add_argument('--show', action='store_true',
                        help='نمایش نمودارها (علاوه بر ذخیره)')

    args = parser.parse_args()

    print(f"\nبارگذاری نتایج از: {args.results_path}")
    try:
        analyzer = ResultsAnalyzer(Path(args.results_path))
    except Exception as e:
        print(f"خطا در بارگذاری: {e}")
        return

    analyzer.print_summary()

    # تعیین حالت نمایش
    do_text    = args.both or args.textual or (not args.graphical and not args.detailed)
    do_graph   = args.both or args.graphical
    do_detail  = args.detailed

    if do_text:
        show_textual(analyzer, args.top_percent, args.stratum)

    if do_detail:
        show_detailed_stats(analyzer)

    if do_graph:
        show_graphical(analyzer, args.top_percent, args.stratum, args.show)


if __name__ == '__main__':
    main()