# -*- coding: utf-8 -*-
"""
Compare Experiments — Side-by-side comparison of multiple evaluation runs
==========================================================================
Loads protocol metrics from 2–N run directories and produces:
  • Console table: all methods × all 4-Layer metrics across runs
  • HTML report:   interactive comparison with charts per metric

Usage:
  python experiments/compare_experiments.py RUN1 RUN2 [RUN3 ...]
  python experiments/compare_experiments.py RUN1 RUN2 --output comparison.html
  python experiments/compare_experiments.py RUN1 RUN2 --metric ndcg@10 spearman_rho rsi@10
  python experiments/compare_experiments.py RUN1 RUN2 --show   # matplotlib charts

Examples:
  # Compare two YouTube runs (different item selections)
  python experiments/compare_experiments.py \\
      results/youtube/w30_h7_n100_top_20260214 \\
      results/youtube/w30_h7_n100_strat_20260214

  # Compare three runs, save HTML
  python experiments/compare_experiments.py \\
      results/movielens/exp1 results/movielens/exp2 results/movielens/exp3 \\
      --output comparison_report.html

Author: Sajjad
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.results_analyzer import ResultsAnalyzer

# ============================================================================
# Colour palette consistent with visualizer.py
# ============================================================================
METHOD_COLORS = {
    'AF':       '#2196F3',
    'LRU':      '#9E9E9E',
    'LFU':      '#4CAF50',
    'EWMA':     '#FF9800',
    'DWT+AF':   '#9C27B0',
    'DTCWT+AF': '#00BCD4',
    'WSPI':     '#E91E63',
}

def _mc(method: str) -> str:
    return METHOD_COLORS.get(method, '#607D8B')

METRICS_DEFAULT = [
    'ndcg@10', 'chr@10',
    'spearman_rho',
    'rsi@10',
    'robustness_distortion',
    'mae',
]

METRIC_LABELS = {
    'ndcg@10':               'NDCG@10  (↑)',
    'chr@10':                'CHR@10   (↑)',
    'spearman_rho':          'Spearman ρ (↑)',
    'rsi@10':                'RSI@10   (↑)',
    'robustness_distortion': 'ΔRank    (↓)',
    'mae':                   'MAE      (↓)',
}
HIGHER_BETTER = {'ndcg@10', 'chr@10', 'spearman_rho', 'rsi@10', 'chr@5', 'chr@20',
                 'ndcg@5', 'ndcg@20', 'rsi@5', 'rsi@20'}


# ============================================================================
# Load results from all run dirs
# ============================================================================

def _short_name(path: Path) -> str:
    """Human-readable label for a run directory."""
    return path.name


def load_all_runs(run_dirs: List[Path]) -> Dict[str, ResultsAnalyzer]:
    """Load ResultsAnalyzer for each run. Returns {label: analyzer}."""
    analyzers = {}
    for d in run_dirs:
        label = _short_name(d)
        try:
            analyzers[label] = ResultsAnalyzer(d)
            print(f"  OK  {label}  ({', '.join(analyzers[label].available_methods)})")
        except Exception as e:
            print(f"  SKIP {label}: {e}")
    return analyzers


def build_comparison_table(analyzers: Dict[str, ResultsAnalyzer],
                            metrics: List[str]) -> pd.DataFrame:
    """
    Build a wide DataFrame:
        rows  = (run_label, method)
        cols  = metrics
    """
    rows = []
    for run_label, ana in analyzers.items():
        for method in ana.available_methods:
            try:
                m = ana.calculate_overall_metrics(method, recompute=False)
                if not m:
                    m = ana.calculate_overall_metrics(method, recompute=True)
            except Exception:
                m = {}
            row = {'run': run_label, 'method': method}
            for metric in metrics:
                row[metric] = m.get(metric, float('nan'))
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Console output
# ============================================================================

def print_comparison(df: pd.DataFrame, metrics: List[str]):
    """Print comparison table grouped by method."""
    print("\n" + "="*90)
    print("CROSS-RUN COMPARISON — 4-Layer Evaluation Protocol")
    print("="*90)

    runs   = df['run'].unique().tolist()
    methods = df['method'].unique().tolist()

    for metric in metrics:
        label = METRIC_LABELS.get(metric, metric)
        higher = metric in HIGHER_BETTER
        arrow  = '↑ higher is better' if higher else '↓ lower is better'
        print(f"\n  {label}  [{arrow}]")
        print(f"  {'Method':<14}", end='')
        for r in runs:
            print(f"  {r[:20]:<22}", end='')
        print()
        print(f"  {'-'*14}", end='')
        for _ in runs:
            print(f"  {'-'*22}", end='')
        print()

        for method in methods:
            print(f"  {method:<14}", end='')
            vals = []
            for r in runs:
                sub = df[(df['run'] == r) & (df['method'] == method)]
                v = float(sub[metric].iloc[0]) if len(sub) > 0 else float('nan')
                vals.append(v)

            # Find best value
            valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
            if valid:
                best_i = max(valid, key=lambda x: x[1] if higher else -x[1])[0]
            else:
                best_i = -1

            for i, v in enumerate(vals):
                if np.isnan(v):
                    cell = f"{'—':>8}"
                else:
                    cell = f"{v:>8.4f}"
                marker = ' *' if i == best_i else '  '
                print(f"  {cell}{marker:<12}", end='')
            print()

    print("\n  (* = best value for that metric across runs)")
    print("="*90 + "\n")


# ============================================================================
# Matplotlib charts
# ============================================================================

def plot_comparison(df: pd.DataFrame, metrics: List[str],
                    save_path: Optional[Path] = None,
                    show: bool = False):
    """Bar chart per metric, grouped by method, coloured by run."""
    runs    = df['run'].unique().tolist()
    methods = df['method'].unique().tolist()
    n_m     = len(metrics)

    run_colors = plt.cm.Set2(np.linspace(0, 1, max(len(runs), 1)))

    cols = min(3, n_m)
    rows = (n_m + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 5.5, rows * 4),
                             squeeze=False)

    x = np.arange(len(methods))
    bar_w = 0.8 / max(len(runs), 1)

    for idx, metric in enumerate(metrics):
        ax = axes[idx // cols][idx % cols]
        higher = metric in HIGHER_BETTER

        for ri, (run_label, color) in enumerate(zip(runs, run_colors)):
            vals = []
            for method in methods:
                sub = df[(df['run'] == run_label) & (df['method'] == method)]
                v = float(sub[metric].iloc[0]) if len(sub) > 0 else float('nan')
                vals.append(v)
            vals = np.array(vals, dtype=np.float64)
            offset = (ri - len(runs) / 2.0 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=run_label, color=color, alpha=0.85,
                          edgecolor='white', linewidth=0.5)
            # Annotate best
            valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
            if valid:
                best_i = max(valid, key=lambda t: t[1] if higher else -t[1])[0]
                bv = vals[best_i]
                if not np.isnan(bv):
                    ax.annotate(f'{bv:.3f}',
                                xy=(x[best_i] + offset, bv),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', fontsize=7, fontweight='bold',
                                color=color)

        label = METRIC_LABELS.get(metric, metric)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=8)
        ax.legend(fontsize=7, ncol=1)
        ax.grid(True, axis='y', alpha=0.25)
        if not higher:
            ax.set_facecolor('#fff8f8')

    # Hide unused subplots
    for idx in range(n_m, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle('Cross-Run Comparison — 4-Layer Evaluation Protocol',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Chart saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# Temporal comparison (Spearman over time for each run)
# ============================================================================

def plot_temporal_comparison(analyzers: Dict[str, ResultsAnalyzer],
                              method: str = 'WSPI',
                              metric: str = 'spearman_rho',
                              save_path: Optional[Path] = None,
                              show: bool = False):
    """Line chart: metric over evaluation windows for one method across runs."""
    run_colors = plt.cm.tab10(np.linspace(0, 1, max(len(analyzers), 1)))
    fig, ax = plt.subplots(figsize=(12, 4))

    plotted = False
    for (run_label, ana), color in zip(analyzers.items(), run_colors):
        if method not in ana.available_methods:
            continue
        try:
            evo = ana.get_temporal_evolution(method, metric)
            if evo is not None and len(evo) > 0:
                x = evo.get('date', range(len(evo)))
                y = evo[metric]
                ax.plot(x, y, label=run_label, color=color, lw=1.8, alpha=0.85)
                plotted = True
        except Exception:
            pass

    if not plotted:
        plt.close(fig)
        return

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_title(f'{method} — {metric_label} over time (all runs)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Evaluation Window', fontsize=10)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Temporal chart saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ============================================================================
# HTML report
# ============================================================================

def write_html(df: pd.DataFrame,
               metrics: List[str],
               analyzers: Dict[str, ResultsAnalyzer],
               out_path: Path):
    """Write self-contained HTML comparison report."""

    runs    = df['run'].unique().tolist()
    methods = df['method'].unique().tolist()

    # Build metric tables per run
    def fmt(v, metric):
        if np.isnan(v):
            return '—'
        return f'{v:.4f}'

    def best_cell(vals, metric):
        """Return index of best value."""
        higher = metric in HIGHER_BETTER
        valid = [(i, v) for i, v in enumerate(vals) if not np.isnan(v)]
        if not valid:
            return -1
        return max(valid, key=lambda t: t[1] if higher else -t[1])[0]

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    html_parts = [f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Comparison Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 30px; color: #222; background: #fafafa; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; margin-top: 40px; border-bottom: 2px solid #3f51b5; padding-bottom: 6px; }}
  h3 {{ color: #3949ab; margin-top: 24px; }}
  table {{ border-collapse: collapse; margin: 16px 0; min-width: 600px; }}
  th {{ background: #3f51b5; color: white; padding: 8px 14px; text-align: left; font-size: 13px; }}
  td {{ padding: 7px 14px; font-size: 13px; border-bottom: 1px solid #e0e0e0; }}
  tr:hover td {{ background: #e8eaf6; }}
  .best {{ font-weight: bold; color: #c62828; }}
  .run-tag {{ display:inline-block; padding: 2px 8px; border-radius: 4px;
              background: #e8eaf6; font-size: 12px; margin: 2px; }}
  .higher {{ color: #1b5e20; font-size: 11px; }}
  .lower  {{ color: #b71c1c; font-size: 11px; }}
  .meta {{ background: #f5f5f5; padding: 12px 18px; border-radius: 6px; margin-bottom: 20px; font-size: 13px; }}
  code {{ background: #eeeeee; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}
</style>
</head>
<body>
<h1>Experiment Comparison Report</h1>
<div class="meta">
  <b>Generated:</b> {timestamp}<br>
  <b>Runs compared:</b> {len(runs)}<br>
  {"".join(f'<span class="run-tag">{r}</span>' for r in runs)}
</div>
"""]

    html_parts.append("<h2>4-Layer Protocol Metrics</h2>")

    for metric in metrics:
        label = METRIC_LABELS.get(metric, metric)
        higher = metric in HIGHER_BETTER
        direction = '<span class="higher">↑ higher is better</span>' if higher \
                    else '<span class="lower">↓ lower is better</span>'
        html_parts.append(f"<h3>{label} &nbsp; {direction}</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Method</th>" +
                          "".join(f"<th>{r}</th>" for r in runs) + "</tr>")

        for method in methods:
            vals = []
            for run in runs:
                sub = df[(df['run'] == run) & (df['method'] == method)]
                v = float(sub[metric].iloc[0]) if len(sub) > 0 else float('nan')
                vals.append(v)
            best_i = best_cell(vals, metric)
            row = f"<tr><td><b>{method}</b></td>"
            for i, v in enumerate(vals):
                cell_cls = ' class="best"' if i == best_i else ''
                row += f"<td{cell_cls}>{fmt(v, metric)}</td>"
            row += "</tr>"
            html_parts.append(row)

        html_parts.append("</table>")

    # Run metadata
    html_parts.append("<h2>Run Metadata</h2>")
    for run_label, ana in analyzers.items():
        cfg = ana.config
        html_parts.append(f"<h3>{run_label}</h3>")
        html_parts.append("<table>")
        for k, v in cfg.items():
            html_parts.append(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>")
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    out_path.write_text("\n".join(html_parts), encoding='utf-8')
    print(f"\n  HTML report saved: {out_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple experiment runs side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two runs (console output)
  python experiments/compare_experiments.py \\
      results/youtube/w30_h7_n100_top_20260214 \\
      results/youtube/w30_h7_n100_strat_20260214

  # Save HTML report
  python experiments/compare_experiments.py RUN1 RUN2 RUN3 \\
      --output comparison_report.html

  # Select specific metrics
  python experiments/compare_experiments.py RUN1 RUN2 \\
      --metric ndcg@10 spearman_rho rsi@10

  # Show matplotlib charts
  python experiments/compare_experiments.py RUN1 RUN2 --show

  # Temporal comparison for WSPI
  python experiments/compare_experiments.py RUN1 RUN2 --temporal-method WSPI
        """
    )

    parser.add_argument('run_dirs', nargs='+',
                        help='Two or more run result directories to compare')
    parser.add_argument('--output', type=str, default=None,
                        metavar='FILE.html',
                        help='Save HTML comparison report to this file')
    parser.add_argument('--metric', nargs='+', default=None,
                        metavar='METRIC',
                        help=f'Metrics to compare (default: {METRICS_DEFAULT})')
    parser.add_argument('--show', action='store_true',
                        help='Show matplotlib comparison charts interactively')
    parser.add_argument('--save-charts', action='store_true',
                        help='Save comparison bar charts as PNG')
    parser.add_argument('--temporal-method', type=str, default=None,
                        metavar='METHOD',
                        help='Also plot temporal evolution for this method (e.g. WSPI)')

    args = parser.parse_args()

    if len(args.run_dirs) < 2:
        parser.error("At least 2 run directories required for comparison.")

    run_dirs = [Path(d) for d in args.run_dirs]
    metrics  = args.metric or METRICS_DEFAULT

    # ---- Load ---------------------------------------------------------------
    print("\nLoading runs:")
    analyzers = load_all_runs(run_dirs)

    if len(analyzers) < 2:
        print("Error: need at least 2 valid run directories.")
        sys.exit(1)

    # ---- Build table --------------------------------------------------------
    df = build_comparison_table(analyzers, metrics)

    # ---- Console output -----------------------------------------------------
    print_comparison(df, metrics)

    # ---- Matplotlib charts --------------------------------------------------
    chart_dir = Path('results') / 'comparisons'
    chart_dir.mkdir(parents=True, exist_ok=True)

    if args.show or args.save_charts:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        bar_path = (chart_dir / f'comparison_bars_{ts}.png') if args.save_charts else None
        print("Generating comparison bar charts ...")
        plot_comparison(df, metrics,
                        save_path=bar_path,
                        show=args.show)

    if args.temporal_method:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        t_path = (chart_dir / f'temporal_{args.temporal_method}_{ts}.png') \
                  if args.save_charts else None
        print(f"Generating temporal comparison for {args.temporal_method} ...")
        plot_temporal_comparison(analyzers,
                                  method=args.temporal_method,
                                  metric='spearman_rho',
                                  save_path=t_path,
                                  show=args.show)

    # ---- HTML ---------------------------------------------------------------
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_html(df, metrics, analyzers, out)


if __name__ == '__main__':
    main()
