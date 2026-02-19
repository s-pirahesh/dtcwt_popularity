"""
Results Visualizer — 4-Layer Frozen Evaluation Protocol
=========================================================
Charts aligned with Chapter 3 dissertation narrative.

Charts produced:
  chart1_protocol_overview    — grouped bar: one metric per layer, all methods
  chart2_stability_rsi        — RSI@K grouped bar
  chart3_robustness           — robustness distortion bar
  chart4_temporal_rsi         — RSI@10 line over time
  chart5_ndcg_profile         — NDCG@K for K=5,10,20
  chart6_temporal_spearman    — Spearman rho over time
  chart7_stratum_performance  — per-stratum Spearman
  chart8_metric_heatmap       — heatmap: methods × metrics
  chart9_per_metric_bars      — individual bar chart per metric (one per metric)
  chart10_boxplot_windows     — box plot per method across evaluation windows

Author: Sajjad
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional

matplotlib.rcParams['font.family']        = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------------------
# Consistent colour / marker palette (updated for new baselines)
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    'AF':          '#2196F3',   # blue
    'MeanFreq':    '#4CAF50',   # green
    'EWMA':        '#FF9800',   # orange
    'RRD':         '#00BCD4',   # cyan
    'VSE':         '#795548',   # brown
    'CompoundPop': '#607D8B',   # blue-grey
    'DWT+AF':      '#9C27B0',   # purple
    'DTCWT+AF':    '#F44336',   # red
    'WSPI':        '#E91E63',   # magenta  ← proposed method
}
METHOD_MARKERS = {
    'AF': 'o', 'MeanFreq': 's', 'EWMA': '^',
    'RRD': 'D', 'VSE': 'p', 'CompoundPop': 'h',
    'DWT+AF': 'D', 'DTCWT+AF': 'v', 'WSPI': '*',
}

STYLE = 'seaborn-v0_8-darkgrid'
DPI   = 300


def _c(m):  return METHOD_COLORS.get(m, '#607D8B')
def _mk(m): return METHOD_MARKERS.get(m, 'o')
def _lw(m): return 2.8 if m == 'WSPI' else 1.6


def _finalize(fig, path, show):
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    print(f"    Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ===========================================================================
class ResultsVisualizer:

    def __init__(self, analyzer, output_dir=None):
        self.analyzer   = analyzer
        self.output_dir = (
            Path(output_dir) if output_dir
            else analyzer.run_dir / 'visualization'
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------

    # ==================================================================
    # CHART 8 — Metric Heatmap (methods × metrics)  *** NEW ***
    # ==================================================================
    def chart8_metric_heatmap(self, show=False):
        """
        Heatmap: rows = methods, columns = metrics.
        Shows relative performance across all methods and metrics at a glance.
        """
        summary = self._get_summary_df()
        if summary is None:
            return

        # Select key metrics
        key_metrics = [
            'mean_spearman_rho', 'mean_kendall_tau',
            'mean_ndcg@5', 'mean_ndcg@10', 'mean_ndcg@20',
            'mean_coverage@5', 'mean_coverage@10', 'mean_coverage@20',
            'mean_rsi@10',
        ]
        cols_present = [c for c in key_metrics if c in summary.columns]
        if not cols_present:
            print("    No suitable columns for heatmap")
            return

        df = summary[cols_present].copy()
        # Rename columns for readability
        rename = {
            'mean_spearman_rho': 'Spearman ρ',
            'mean_kendall_tau':  'Kendall τ',
            'mean_ndcg@5':       'NDCG@5',
            'mean_ndcg@10':      'NDCG@10',
            'mean_ndcg@20':      'NDCG@20',
            'mean_coverage@5':   'Coverage@5',
            'mean_coverage@10':  'Coverage@10',
            'mean_coverage@20':  'Coverage@20',
            'mean_rsi@10':       'RSI@10',
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        # Normalize each column to [0,1] for fair colour mapping
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)

        fig, (ax_heat, ax_raw) = plt.subplots(1, 2, figsize=(18, len(df) * 0.7 + 2),
                                               gridspec_kw={'width_ratios': [3, 1]})

        # -- Heatmap (normalised values, raw values as annotations) --
        sns.heatmap(df_norm, ax=ax_heat, cmap='RdYlGn', vmin=0, vmax=1,
                    linewidths=0.5, linecolor='white',
                    annot=df.round(3), fmt='.3f',
                    annot_kws={'size': 8},
                    cbar_kws={'label': 'Relative Score (normalised)'})
        ax_heat.set_title('Method × Metric Performance Heatmap', fontsize=13, pad=12)
        ax_heat.set_xlabel('')
        ax_heat.set_ylabel('Method')

        # Highlight WSPI row label
        yticklabels = ax_heat.get_yticklabels()
        for lbl in yticklabels:
            if lbl.get_text() == 'WSPI':
                lbl.set_fontweight('bold')
                lbl.set_color('#E91E63')

        # -- Rank column on the right (average normalised rank) --
        avg_rank = df_norm.mean(axis=1).sort_values(ascending=False)
        ax_raw.barh(range(len(avg_rank)), avg_rank.values,
                    color=[_c(m) for m in avg_rank.index], alpha=0.85)
        ax_raw.set_yticks(range(len(avg_rank)))
        ax_raw.set_yticklabels(avg_rank.index)
        ax_raw.set_xlabel('Avg Normalised Score')
        ax_raw.set_title('Overall Rank')
        ax_raw.set_xlim(0, 1.05)
        for i, (method, val) in enumerate(avg_rank.items()):
            ax_raw.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8)

        plt.suptitle('Comprehensive Performance Comparison', fontsize=14, y=1.01)
        _finalize(fig, self.output_dir / 'chart8_metric_heatmap.png', show)

    # ==================================================================
    # CHART 9 — Per-Metric Individual Bar Charts  *** NEW ***
    # ==================================================================
    def chart9_per_metric_bars(self, show=False):
        """
        Individual bar chart for each metric.
        Each chart saved separately so it can be used standalone in the paper.
        """
        summary = self._get_summary_df()
        if summary is None:
            return

        metric_groups = {
            'Spearman ρ':   ('mean_spearman_rho', 'Spearman Rank Correlation (ρ) — Higher is Better', True),
            'Kendall τ':    ('mean_kendall_tau',  'Kendall Rank Correlation (τ) — Higher is Better', True),
            'NDCG@10':      ('mean_ndcg@10',      'NDCG@10 — Ranking Quality (Higher is Better)', True),
            'Coverage@10':  ('mean_coverage@10',  'Coverage@10 — Top-K Interaction Coverage (Higher is Better)', True),
            'RSI@10':       ('mean_rsi@10',       'RSI@10 — Ranking Stability (Higher is Better)', True),
        }

        per_metric_dir = self.output_dir / 'per_metric'
        per_metric_dir.mkdir(exist_ok=True)

        for metric_name, (col, title, higher_better) in metric_groups.items():
            if col not in summary.columns:
                continue

            methods = list(summary.index)
            vals    = summary[col].reindex(methods).fillna(0).values
            colors  = [_c(m) for m in methods]

            with plt.style.context(STYLE):
                fig, ax = plt.subplots(figsize=(11, 5))
                bars = ax.bar(methods, vals, color=colors, alpha=0.85, zorder=3)

                # Highlight WSPI
                if 'WSPI' in methods:
                    wspi_idx = methods.index('WSPI')
                    bars[wspi_idx].set_edgecolor('#E91E63')
                    bars[wspi_idx].set_linewidth(2.5)
                    bars[wspi_idx].set_alpha(1.0)

                # Value labels
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.005, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.tick_params(axis='x', rotation=30)
                ax.set_ylim(0, min(1.15, max(vals) * 1.2 + 0.05))
                ax.grid(axis='y', alpha=0.4, zorder=0)

                # Add "★ Best" annotation for highest bar
                best_idx = int(np.argmax(vals))
                ax.annotate('★ Best', xy=(best_idx, vals[best_idx]),
                            xytext=(best_idx, vals[best_idx] + 0.04),
                            ha='center', fontsize=9, color='#1B5E20',
                            fontweight='bold')

            safe_name = metric_name.replace('@', '_at_').replace(' ', '_')
            _finalize(fig, per_metric_dir / f'metric_{safe_name}.png', show)

        print(f"    Per-metric charts saved to: {per_metric_dir}")

    # ==================================================================
    # CHART 10 — Box Plot: Score Distribution Across Evaluation Windows *** NEW ***
    # ==================================================================
    def chart10_boxplot_windows(self, show=False):
        """
        Box plot showing distribution of key metrics across evaluation windows.
        Reveals consistency of each method, not just mean performance.
        """
        records = self._get_protocol_records()
        if records is None:
            print("    No per-window records for boxplot")
            return

        metrics_to_plot = [
            ('spearman_rho', 'Spearman ρ'),
            ('ndcg@10',      'NDCG@10'),
            ('coverage@10',  'Coverage@10'),
            ('rsi@10',       'RSI@10'),
        ]
        available = [(c, label) for c, label in metrics_to_plot if c in records.columns]
        if not available:
            print("    No suitable columns for boxplot")
            return

        n_plots = len(available)
        with plt.style.context(STYLE):
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), sharey=False)
            if n_plots == 1:
                axes = [axes]

            methods = sorted(records['method'].unique())
            palette = {m: _c(m) for m in methods}

            for ax, (col, label) in zip(axes, available):
                data_list = [records[records['method'] == m][col].dropna().values
                             for m in methods]
                bp = ax.boxplot(data_list, labels=methods, patch_artist=True,
                                notch=False, vert=True,
                                medianprops=dict(color='black', linewidth=2))
                for patch, method in zip(bp['boxes'], methods):
                    patch.set_facecolor(_c(method))
                    patch.set_alpha(0.75)
                ax.set_title(label, fontsize=11)
                ax.set_ylabel(label if ax == axes[0] else '')
                ax.tick_params(axis='x', rotation=40)
                ax.grid(axis='y', alpha=0.4)

            plt.suptitle('Score Distribution Across Evaluation Windows', fontsize=13)

        _finalize(fig, self.output_dir / 'chart10_boxplot_windows.png', show)

    # ==================================================================
    # CHART 11 — Radar / Spider Chart per Method  *** NEW ***
    # ==================================================================
    def chart11_radar_charts(self, show=False):
        """
        Radar/Spider chart showing each method's multi-dimensional profile.
        One radar per method, plus one combined radar for WSPI vs best baselines.
        """
        summary = self._get_summary_df()
        if summary is None:
            return

        # Metrics for radar (all should be "higher is better"; invert robustness)
        radar_metrics = {
            'Spearman ρ':  'mean_spearman_rho',
            'Kendall τ':   'mean_kendall_tau',
            'NDCG@10':     'mean_ndcg@10',
            'Coverage@10': 'mean_coverage@10',
            'RSI@10':      'mean_rsi@10',
        }
        available = {label: col for label, col in radar_metrics.items()
                     if col in summary.columns}
        if len(available) < 3:
            print("    Not enough metrics for radar chart")
            return

        labels  = list(available.keys())
        n_axes  = len(labels)
        angles  = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
        angles += angles[:1]  # close the polygon

        methods = list(summary.index)

        # --- Normalize each metric to [0,1] across methods ---
        df = summary[[c for c in available.values()]].copy()
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)
        df_norm.columns = labels

        radar_dir = self.output_dir / 'radar'
        radar_dir.mkdir(exist_ok=True)

        # (A) Combined radar: all methods on one chart
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(8, 8),
                                   subplot_kw=dict(polar=True))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=11)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=8)

            for method in methods:
                if method not in df_norm.index:
                    continue
                vals = df_norm.loc[method].values.tolist()
                vals += vals[:1]
                lw = _lw(method)
                ax.plot(angles, vals, color=_c(method),
                        linewidth=lw, linestyle='solid',
                        label=method, marker=_mk(method), markersize=5)
                ax.fill(angles, vals, color=_c(method), alpha=0.05)

            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                      fontsize=9, framealpha=0.9)
            ax.set_title('Multi-Metric Performance Radar\n(Normalised, all methods)',
                         fontsize=12, pad=20)

        _finalize(fig, radar_dir / 'radar_all_methods.png', show)

        # (B) Individual radar per method
        n_cols = 3
        n_rows = (len(methods) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 5 * n_rows),
                                 subplot_kw=dict(polar=True))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for ax_i, method in enumerate(methods):
            ax = axes_flat[ax_i]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.5, 1.0])
            ax.set_yticklabels(['0.5', '1.0'], size=7)

            if method in df_norm.index:
                vals = df_norm.loc[method].values.tolist()
                vals += vals[:1]
                ax.plot(angles, vals, color=_c(method),
                        linewidth=2.0, linestyle='solid',
                        marker=_mk(method), markersize=5)
                ax.fill(angles, vals, color=_c(method), alpha=0.25)

            title_weight = 'bold' if method == 'WSPI' else 'normal'
            ax.set_title(method, size=11, pad=10, fontweight=title_weight,
                         color=_c(method))

        # Hide empty subplots
        for ax_i in range(len(methods), len(axes_flat)):
            axes_flat[ax_i].set_visible(False)

        plt.suptitle('Individual Performance Profiles (Radar Charts)', fontsize=13)
        _finalize(fig, radar_dir / 'radar_individual.png', show)
        print(f"    Radar charts saved to: {radar_dir}")

    def generate_all_charts(self, show: bool = False):
        print(f"\n{'='*60}")
        print("GENERATING ALL CHARTS")
        print(f"  Output: {self.output_dir}")
        print(f"{'='*60}")

        charts = [
            ('chart1_protocol_overview',   self.chart1_protocol_overview),
            ('chart2_stability_rsi',       self.chart2_stability_rsi),
            ('chart3_robustness',          self.chart3_robustness),
            ('chart4_temporal_rsi',        self.chart4_temporal_rsi),
            ('chart5_ndcg_profile',        self.chart5_ndcg_profile),
            ('chart6_temporal_spearman',   self.chart6_temporal_spearman),
            ('chart7_stratum_performance', self.chart7_stratum_performance),
            ('chart8_metric_heatmap',      self.chart8_metric_heatmap),
            ('chart9_per_metric_bars',     self.chart9_per_metric_bars),
            ('chart10_boxplot_windows',    self.chart10_boxplot_windows),
            ('chart11_radar_charts',       self.chart11_radar_charts),
        ]

        for name, fn in charts:
            print(f"\n  → {name}")
            try:
                fn(show=show)
            except Exception as e:
                print(f"    WARNING: {name} failed — {e}")

        print(f"\n{'='*60}")
        print(f"All charts saved to: {self.output_dir}")
        print(f"{'='*60}\n")
