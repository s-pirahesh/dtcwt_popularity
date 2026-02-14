# -*- coding: utf-8 -*-
"""
Show Results — Display evaluation results for a single run
============================================================
Display-only: reads pre-computed protocol metrics — no recalculation.
To recompute from raw scores, use:
    python experiments/analyze_results.py RESULTS_PATH --recompute

Displays:
  Textual  : multi-K table (NDCG, CHR, RSI) + Diagnostics + Robustness
  Graphical: all 7 publication-quality charts from visualizer.py

Author: Sajjad
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

def show_textual(analyzer, top_percent=None, stratum=None):
    """Print multi-K 4-Layer Protocol table."""
    print("\n" + "="*80)
    print("RESULTS — FROZEN 4-LAYER EVALUATION PROTOCOL")
    print("="*80)

    comparison = analyzer.compare_methods(
        filter_top_percent=top_percent,
        filter_stratum=stratum,
        recompute=False,
    )

    if len(comparison) == 0:
        print("No data found.")
        print("Tip: protocol metrics may not have been computed yet.")
        print("     Run: python experiments/analyze_results.py RESULTS_PATH --recompute --save-recomputed")
        return

    filters = [f for f in [
        f"Top {top_percent}%" if top_percent else None,
        stratum,
    ] if f]
    if filters:
        print(f"\nFilters: {', '.join(filters)}")

    # Layer 1 & 3: multi-K
    print("\n[Layer 1 — Decision  &  Layer 3 — Stability]")
    print("-"*80)
    k_cols = ['method']
    for k in analyzer.K_VALUES:
        for m in [f'ndcg@{k}', f'chr@{k}', f'rsi@{k}']:
            if m in comparison.columns:
                k_cols.append(m)
    print(comparison[k_cols].to_string(index=False))

    # Layer 2: Diagnostics
    print("\n[Layer 2 — Diagnostic]")
    print("-"*80)
    diag_cols = ['method'] + [c for c in
                               ['spearman_rho', 'kendall_tau', 'mae']
                               if c in comparison.columns]
    print(comparison[diag_cols].to_string(index=False))

    # Layer 4: Robustness
    if 'robustness_distortion' in comparison.columns:
        print("\n[Layer 4 — Robustness  (lower delta-Rank is better)]")
        print("-"*80)
        rob_df = comparison[['method', 'robustness_distortion']].copy()
        rob_df = rob_df.sort_values('robustness_distortion')
        print(rob_df.to_string(index=False))

    # Highlights
    print("\n" + "-"*80)
    print("HIGHLIGHTS:")

    def _hl(col, largest, label):
        if col not in comparison.columns:
            return
        sub = comparison.dropna(subset=[col])
        if sub.empty:
            return
        row = (sub.nlargest(1, col) if largest else sub.nsmallest(1, col)).iloc[0]
        print(f"  {label:<40} {row['method']} ({row[col]:.4f})")

    _hl('ndcg@10',               largest=True,  label='Highest NDCG@10:')
    _hl('chr@10',                largest=True,  label='Highest Cache Hit Ratio@10:')
    _hl('spearman_rho',          largest=True,  label='Highest Spearman rho:')
    _hl('rsi@10',                largest=True,  label='Most Stable (RSI@10):')
    _hl('robustness_distortion', largest=False, label='Most Robust (min delta-Rank):')
    _hl('mae',                   largest=False, label='Lowest MAE:')

    print("="*80 + "\n")


def show_detailed_stats(analyzer):
    """Print per-method detailed 4-Layer stats."""
    print("\n" + "="*80)
    print("DETAILED METHOD STATS — 4-Layer Protocol")
    print("="*80)

    for method in analyzer.available_methods:
        print(f"\n{'—'*80}")
        print(f"Method: {method}")
        print(f"{'—'*80}")

        m = analyzer.calculate_overall_metrics(method, recompute=False)
        if not m:
            print("  No data found (protocol metrics not available)")
            continue

        print("  [Layer 1 — Decision]")
        for k in analyzer.K_VALUES:
            print(f"    NDCG@{k:<4} {m.get(f'ndcg@{k}', float('nan')):>7.4f}"
                  f"    CHR@{k:<4} {m.get(f'chr@{k}', float('nan')):>7.4f}")

        print("  [Layer 2 — Diagnostic]")
        print(f"    Spearman rho : {m.get('spearman_rho', float('nan')):>8.4f}")
        print(f"    Kendall tau  : {m.get('kendall_tau',  float('nan')):>8.4f}")
        print(f"    MAE          : {m.get('mae',          float('nan')):>8.2f}")

        print("  [Layer 3 — Stability]")
        for k in analyzer.K_VALUES:
            print(f"    RSI@{k:<4}  {m.get(f'rsi@{k}', float('nan')):>8.4f}")

        print("  [Layer 4 — Robustness]")
        print(f"    Avg delta-Rank : {m.get('robustness_distortion', float('nan')):>8.2f}")

        print("  [Stratum Breakdown — Spearman rho]")
        for sname in ['cold_start', 'low', 'medium', 'high']:
            try:
                sm = analyzer.calculate_overall_metrics(
                    method, filter_stratum=sname, recompute=False)
                if sm:
                    spr  = sm.get('spearman_rho', sm.get('spearman', float('nan')))
                    ndcg = sm.get('ndcg@10', sm.get('ndcg', float('nan')))
                    print(f"    {sname:<12} rho={spr:>6.3f}  NDCG@10={ndcg:>6.4f}")
            except Exception:
                pass

    print("\n" + "="*80 + "\n")


# =============================================================================
# Graphical display — all 7 charts, ALL methods including WSPI
# =============================================================================

def show_graphical(analyzer, top_percent=None, stratum=None, show_plots=False):
    """Generate all 7 publication-quality charts."""
    print("\n" + "="*80)
    print("GENERATING CHARTS — Frozen 4-Layer Evaluation Protocol")
    print("="*80 + "\n")

    vis = ResultsVisualizer(analyzer)
    # Pass ALL available methods — visualizer highlights WSPI automatically
    methods = list(analyzer.available_methods)

    steps = [
        ("1. Protocol overview (grouped bar, all 4 layers)",
         lambda: vis.plot_protocol_overview(save=True, show=show_plots)),
        ("2. RSI stability (grouped bar K=5/10/20)",
         lambda: vis.plot_stability_rsi(save=True, show=show_plots)),
        ("3. Robustness distortion (horizontal bar)",
         lambda: vis.plot_robustness(save=True, show=show_plots)),
        ("4. Temporal RSI@10 (line chart over windows)",
         lambda: vis.plot_temporal_rsi(k=10, save=True, show=show_plots)),
        ("5. NDCG@K profile (line chart)",
         lambda: vis.plot_ndcg_profile(save=True, show=show_plots)),
        ("6. Temporal Spearman rho (line — includes WSPI)",
         lambda: vis.plot_temporal_spearman(save=True, show=show_plots)),
        ("7. Per-stratum Spearman (grouped bar — includes WSPI)",
         lambda: vis.plot_stratum_comparison(
             methods=methods, metric='spearman_corr',
             save=True, show=show_plots)),
    ]

    ok = 0
    for label, fn in steps:
        print(f"  {label} ...", end=' ', flush=True)
        try:
            fn()
            print("OK")
            ok += 1
        except Exception as e:
            print(f"SKIP — {e}")

    print(f"\n  {ok}/{len(steps)} charts generated")
    print(f"  Saved to: {vis.output_dir}")
    print("="*80 + "\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Display evaluation results (read-only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/show_results.py results/youtube/w30_h7_n100_top_20260214
  python experiments/show_results.py RESULTS_PATH --graphical --show
  python experiments/show_results.py RESULTS_PATH --both --show
  python experiments/show_results.py RESULTS_PATH --detailed
  python experiments/show_results.py RESULTS_PATH --top-percent 20 --stratum medium
        """
    )

    parser.add_argument('results_path',
                        help='Path to run results directory')
    parser.add_argument('--textual',   action='store_true',
                        help='Textual table (default)')
    parser.add_argument('--graphical', action='store_true',
                        help='Generate all 7 charts')
    parser.add_argument('--both',      action='store_true',
                        help='Textual + graphical')
    parser.add_argument('--detailed',  action='store_true',
                        help='Per-method detailed 4-Layer stats')
    parser.add_argument('--top-percent', type=float, default=None,
                        metavar='PERCENT',
                        help='Filter: top-k percent of items')
    parser.add_argument('--stratum', type=str, default=None,
                        choices=['cold_start', 'low', 'medium', 'high'],
                        help='Filter: stratum name')
    parser.add_argument('--show', action='store_true',
                        help='Show charts interactively (in addition to saving)')

    args = parser.parse_args()

    print(f"\nLoading results from: {args.results_path}")
    try:
        analyzer = ResultsAnalyzer(Path(args.results_path))
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    analyzer.print_summary()

    do_text   = args.both or args.textual or (not args.graphical and not args.detailed)
    do_graph  = args.both or args.graphical
    do_detail = args.detailed

    if do_text:
        show_textual(analyzer, args.top_percent, args.stratum)
    if do_detail:
        show_detailed_stats(analyzer)
    if do_graph:
        show_graphical(analyzer, args.top_percent, args.stratum, args.show)


if __name__ == '__main__':
    main()
