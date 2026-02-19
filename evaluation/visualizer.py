"""
Results Visualizer — 4-Layer Frozen Evaluation Protocol
=========================================================
Charts aligned with Chapter 3 dissertation narrative.

Charts produced:
  chart1_protocol_overview    — grouped bar: representative metric per layer
  chart2_stability_rsi        — RSI@K grouped bar
  chart3_robustness           — rank distortion bar (robustness)
  chart4_temporal_rsi         — RSI@10 line over time
  chart5_ndcg_profile         — NDCG@K for K=5,10,20
  chart6_temporal_spearman    — Spearman ρ over time
  chart7_stratum_performance  — per-stratum Spearman bar
  chart8_metric_heatmap       — heatmap: methods × metrics
  chart9_per_metric_bars      — one bar chart per metric (standalone for paper)
  chart10_boxplot_windows     — box plot per method across evaluation windows
  chart11_radar_charts        — radar/spider chart per method + combined

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
# Colour / marker palette
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    'AF':          '#2196F3',
    'EWMA':        '#FF9800',
    'RRD':         '#00BCD4',
    'VSE':         '#795548',
    'CompoundPop': '#607D8B',
    'PFRF':        '#009688',
    'DWT+AF':      '#9C27B0',
    'DTCWT+AF':    '#F44336',
    'WSPI':        '#E91E63',
}
METHOD_MARKERS = {
    'AF': 'o', 'EWMA': '^', 'RRD': 'D',
    'VSE': 'p', 'CompoundPop': 'h', 'PFRF': 's',
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

    # ------------------------------------------------------------------
    # Helpers: load data from analyzer
    # ------------------------------------------------------------------
    def _get_summary_df(self) -> Optional[pd.DataFrame]:
        """Build summary DataFrame (methods as index, mean metrics as columns)."""
        try:
            rows = []
            for method in self.analyzer.available_methods:
                summary = self.analyzer.get_protocol_summary(method)
                if summary:
                    rows.append(summary)
            if not rows:
                return None
            df = pd.DataFrame(rows).set_index('method')
            # Normalize column names → mean_ prefix
            rename = {}
            for col in df.columns:
                if col == 'windows':
                    continue
                for prefix in ('ndcg@', 'chr@', 'coverage@', 'rsi@'):
                    if col.startswith(prefix) and not col.startswith('mean_'):
                        rename[col] = f'mean_{col}'
                        break
                for plain in ('kendall_tau', 'spearman_rho', 'mae', 'robustness_distortion'):
                    if col == plain:
                        rename[col] = f'mean_{plain}'
            df = df.rename(columns=rename)
            # Add coverage@ alias for chr@ columns (backward compat)
            for col in list(df.columns):
                if 'chr@' in col:
                    new = col.replace('chr@', 'coverage@')
                    if new not in df.columns:
                        df[new] = df[col]
            return df
        except Exception as e:
            print(f"    _get_summary_df failed: {e}")
            return None

    def _get_protocol_records(self) -> Optional[pd.DataFrame]:
        """Load per-window protocol records for ALL methods combined."""
        try:
            frames = []
            for method in self.analyzer.available_methods:
                df = self.analyzer.load_protocol_metrics(method)
                if df is not None and not df.empty:
                    df = df.copy()
                    df['method'] = method
                    if 'window_idx' not in df.columns:
                        df['window_idx'] = range(len(df))
                    for col in list(df.columns):
                        if col.startswith('chr@'):
                            new = col.replace('chr@', 'coverage@')
                            if new not in df.columns:
                                df[new] = df[col]
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None
        except Exception as e:
            print(f"    _get_protocol_records failed: {e}")
            return None

    def _get_stratum_summary(self) -> Optional[pd.DataFrame]:
        """Load per-stratum summary across all methods."""
        try:
            frames = []
            for method in self.analyzer.available_methods:
                df = self.analyzer.get_stratum_comparison(method)
                if df is not None and not df.empty:
                    df = df.copy()
                    df['method'] = method
                    frames.append(df)
            return pd.concat(frames, ignore_index=True) if frames else None
        except Exception:
            return None

    # ==================================================================
    # CHART 1 — Protocol Overview
    # ==================================================================
    def chart1_protocol_overview(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            print("    No summary data — skipping chart1")
            return

        metrics_shown = {
            'Decision':   'mean_ndcg@10',
            'Diagnostic': 'mean_spearman_rho',
            'Stability':  'mean_rsi@10',
        }
        available = {k: v for k, v in metrics_shown.items() if v in summary.columns}
        if not available:
            print("    Required columns not found — skipping chart1")
            return

        methods = list(summary.index)
        n_layers = len(available)
        x = np.arange(len(methods))
        width = 0.8 / n_layers

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
            for i, (layer, col) in enumerate(available.items()):
                vals = summary[col].reindex(methods).fillna(0).values
                bars = ax.bar(x + i * width - (n_layers - 1) * width / 2,
                              vals, width=width * 0.9,
                              label=f"{layer} ({col})", alpha=0.85)
                if 'WSPI' in methods:
                    bars[methods.index('WSPI')].set_edgecolor('#E91E63')
                    bars[methods.index('WSPI')].set_linewidth(2.5)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            ax.set_ylabel('Score')
            ax.set_title('Protocol Overview — Representative Metric per Layer')
            ax.legend()
            ax.set_ylim(0, 1.15)
        _finalize(fig, self.output_dir / 'chart1_protocol_overview.png', show)

    # ==================================================================
    # CHART 2 — Stability RSI@K
    # ==================================================================
    def chart2_stability_rsi(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        rsi_cols = sorted([c for c in summary.columns if 'rsi' in c.lower() and 'mean' in c.lower()])
        if not rsi_cols:
            print("    No RSI columns — skipping chart2")
            return

        methods = list(summary.index)
        x = np.arange(len(methods))
        width = 0.8 / len(rsi_cols)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
            for i, col in enumerate(rsi_cols):
                label = col.replace('mean_rsi@', 'RSI@').replace('mean_rsi_', 'RSI ')
                vals  = summary[col].reindex(methods).fillna(0).values
                ax.bar(x + i * width - (len(rsi_cols) - 1) * width / 2,
                       vals, width=width * 0.9, label=label, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            ax.set_ylabel('RSI (Jaccard Similarity)')
            ax.set_title('Ranking Stability Index (RSI) — Higher is More Stable')
            ax.legend()
            ax.set_ylim(0, 1.05)
        _finalize(fig, self.output_dir / 'chart2_stability_rsi.png', show)

    # ==================================================================
    # CHART 3 — Robustness (ΔRank)
    # ==================================================================
    def chart3_robustness(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        rob_col = next((c for c in summary.columns
                        if 'distort' in c.lower() or 'delta_rank' in c.lower()
                        or 'robustness' in c.lower()), None)
        if rob_col is None:
            print("    No robustness column — skipping chart3")
            return

        methods = list(summary.index)
        vals = summary[rob_col].reindex(methods).fillna(0).values

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
            bars = ax.bar(methods, vals, color=[_c(m) for m in methods], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=9)
            ax.set_ylabel('Mean ΔRank (lower = more robust)')
            ax.set_title('Robustness to Noise: Rank Distortion (ΔRank)')
            ax.tick_params(axis='x', rotation=30)
        _finalize(fig, self.output_dir / 'chart3_robustness.png', show)

    # ==================================================================
    # CHART 4 — Temporal RSI@10
    # ==================================================================
    def chart4_temporal_rsi(self, show=False):
        records = self._get_protocol_records()
        if records is None:
            print("    No per-window records — skipping chart4")
            return

        rsi_col = next((c for c in records.columns
                        if 'rsi@10' in c.lower() or 'rsi_10' in c.lower()), None)
        if rsi_col is None:
            print("    No RSI@10 column — skipping chart4")
            return

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(12, 5))
            for method, grp in records.groupby('method'):
                grp_s = grp.sort_values('window_idx')
                ax.plot(grp_s['window_idx'], grp_s[rsi_col],
                        label=method, color=_c(method),
                        marker=_mk(method), linewidth=_lw(method),
                        markersize=4, alpha=0.9)
            ax.set_xlabel('Window Index')
            ax.set_ylabel('RSI@10')
            ax.set_title('Ranking Stability Over Time (RSI@10)')
            ax.legend(fontsize=8, ncol=3)
            ax.set_ylim(0, 1.05)
        _finalize(fig, self.output_dir / 'chart4_temporal_rsi.png', show)

    # ==================================================================
    # CHART 5 — NDCG@K Profile
    # ==================================================================
    def chart5_ndcg_profile(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        ndcg_cols = {k: f'mean_ndcg@{k}' for k in [5, 10, 20]}
        available  = {k: c for k, c in ndcg_cols.items() if c in summary.columns}
        if not available:
            print("    No NDCG columns — skipping chart5")
            return

        methods = list(summary.index)
        x = np.arange(len(methods))
        width = 0.8 / len(available)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
            for i, (k, col) in enumerate(sorted(available.items())):
                vals = summary[col].reindex(methods).fillna(0).values
                ax.bar(x + i * width - (len(available) - 1) * width / 2,
                       vals, width=width * 0.9, label=f'NDCG@{k}', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            ax.set_ylabel('NDCG Score')
            ax.set_title('NDCG@K Profile (K = 5, 10, 20)')
            ax.legend()
            ax.set_ylim(0, 1.05)
        _finalize(fig, self.output_dir / 'chart5_ndcg_profile.png', show)

    # ==================================================================
    # CHART 6 — Temporal Spearman ρ
    # ==================================================================
    def chart6_temporal_spearman(self, show=False):
        records = self._get_protocol_records()
        if records is None:
            print("    No per-window records — skipping chart6")
            return

        rho_col = next((c for c in records.columns if 'spearman' in c.lower()), None)
        if rho_col is None:
            print("    No Spearman column — skipping chart6")
            return

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(12, 5))
            for method, grp in records.groupby('method'):
                grp_s = grp.sort_values('window_idx')
                ax.plot(grp_s['window_idx'], grp_s[rho_col],
                        label=method, color=_c(method),
                        marker=_mk(method), linewidth=_lw(method),
                        markersize=4, alpha=0.9)
            ax.set_xlabel('Window Index')
            ax.set_ylabel('Spearman ρ')
            ax.set_title('Spearman Rank Correlation Over Time')
            ax.legend(fontsize=8, ncol=3)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _finalize(fig, self.output_dir / 'chart6_temporal_spearman.png', show)

    # ==================================================================
    # CHART 7 — Per-Stratum Spearman
    # ==================================================================
    def chart7_stratum_performance(self, show=False):
        strata_data = self._get_stratum_summary()
        if strata_data is None or strata_data.empty:
            print("    No stratum data — skipping chart7")
            return

        spearman_col = next((c for c in strata_data.columns
                             if 'spearman' in c.lower()), None)
        stratum_col  = next((c for c in strata_data.columns
                             if 'stratum' in c.lower()), None)
        if spearman_col is None or stratum_col is None:
            print("    Missing stratum/spearman columns — skipping chart7")
            return

        strata  = strata_data[stratum_col].unique()
        methods = strata_data['method'].unique()
        x = np.arange(len(strata))
        width = 0.8 / len(methods)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(10, len(strata) * 2), 5))
            for i, method in enumerate(methods):
                mdata = strata_data[strata_data['method'] == method]
                vals = []
                for s in strata:
                    row = mdata[mdata[stratum_col] == s]
                    vals.append(row[spearman_col].values[0] if len(row) > 0 else 0)
                ax.bar(x + i * width - (len(methods) - 1) * width / 2,
                       vals, width=width * 0.9,
                       label=method, color=_c(method), alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(strata, rotation=0)
            ax.set_ylabel('Mean Spearman ρ')
            ax.set_title('Spearman Correlation by Popularity Stratum')
            ax.legend(fontsize=8, ncol=3)
        _finalize(fig, self.output_dir / 'chart7_stratum_performance.png', show)

    # ==================================================================
    # CHART 8 — Metric Heatmap
    # ==================================================================
    def chart8_metric_heatmap(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        key_metrics = [
            'mean_spearman_rho', 'mean_kendall_tau',
            'mean_ndcg@5', 'mean_ndcg@10', 'mean_ndcg@20',
            'mean_coverage@5', 'mean_coverage@10', 'mean_coverage@20',
            'mean_rsi@10',
        ]
        cols_present = [c for c in key_metrics if c in summary.columns]
        if not cols_present:
            print("    No suitable columns for heatmap — skipping chart8")
            return

        df = summary[cols_present].copy()
        rename = {
            'mean_spearman_rho': 'Spearman ρ', 'mean_kendall_tau': 'Kendall τ',
            'mean_ndcg@5': 'NDCG@5', 'mean_ndcg@10': 'NDCG@10', 'mean_ndcg@20': 'NDCG@20',
            'mean_coverage@5': 'Coverage@5', 'mean_coverage@10': 'Coverage@10',
            'mean_coverage@20': 'Coverage@20', 'mean_rsi@10': 'RSI@10',
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)

        n_methods = len(df)
        fig, (ax_heat, ax_raw) = plt.subplots(
            1, 2, figsize=(18, max(4, n_methods * 0.7 + 2)),
            gridspec_kw={'width_ratios': [3, 1]}
        )
        sns.heatmap(df_norm, ax=ax_heat, cmap='RdYlGn', vmin=0, vmax=1,
                    linewidths=0.5, linecolor='white',
                    annot=df.round(3), fmt='.3f', annot_kws={'size': 8},
                    cbar_kws={'label': 'Relative Score (normalised)'})
        ax_heat.set_title('Method × Metric Performance Heatmap', fontsize=13, pad=12)
        for lbl in ax_heat.get_yticklabels():
            if lbl.get_text() == 'WSPI':
                lbl.set_fontweight('bold')
                lbl.set_color('#E91E63')

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
    # CHART 9 — Per-Metric Individual Bar Charts
    # ==================================================================
    def chart9_per_metric_bars(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        metric_groups = {
            'Spearman ρ':   ('mean_spearman_rho', 'Spearman Rank Correlation (ρ) — Higher is Better'),
            'Kendall τ':    ('mean_kendall_tau',  'Kendall Rank Correlation (τ) — Higher is Better'),
            'NDCG@10':      ('mean_ndcg@10',      'NDCG@10 — Ranking Quality (Higher is Better)'),
            'Coverage@10':  ('mean_coverage@10',  'Coverage@10 — Top-K Interaction Coverage (Higher is Better)'),
            'RSI@10':       ('mean_rsi@10',       'RSI@10 — Ranking Stability (Higher is Better)'),
        }

        per_metric_dir = self.output_dir / 'per_metric'
        per_metric_dir.mkdir(exist_ok=True)

        for metric_name, (col, title) in metric_groups.items():
            if col not in summary.columns:
                continue
            methods = list(summary.index)
            vals    = summary[col].reindex(methods).fillna(0).values

            with plt.style.context(STYLE):
                fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
                bars = ax.bar(methods, vals, color=[_c(m) for m in methods], alpha=0.85, zorder=3)
                if 'WSPI' in methods:
                    bars[methods.index('WSPI')].set_edgecolor('#E91E63')
                    bars[methods.index('WSPI')].set_linewidth(2.5)
                    bars[methods.index('WSPI')].set_alpha(1.0)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(vals) * 0.01, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.tick_params(axis='x', rotation=30)
                ax.set_ylim(0, min(1.2, max(vals) * 1.25 + 0.05) if max(vals) > 0 else 1.0)
                ax.grid(axis='y', alpha=0.4, zorder=0)
                if len(vals) > 0 and max(vals) > 0:
                    best_idx = int(np.argmax(vals))
                    ax.annotate('★ Best', xy=(best_idx, vals[best_idx]),
                                xytext=(best_idx, vals[best_idx] + max(vals) * 0.06),
                                ha='center', fontsize=9, color='#1B5E20', fontweight='bold')

            safe_name = metric_name.replace('@', '_at_').replace(' ', '_')
            _finalize(fig, per_metric_dir / f'metric_{safe_name}.png', show)

        print(f"    Per-metric charts saved to: {per_metric_dir}")

    # ==================================================================
    # CHART 10 — Box Plot: Score Distribution Across Windows
    # ==================================================================
    def chart10_boxplot_windows(self, show=False):
        records = self._get_protocol_records()
        if records is None:
            print("    No per-window records — skipping chart10")
            return

        metrics_to_plot = [
            ('spearman_rho', 'Spearman ρ'),
            ('ndcg@10',      'NDCG@10'),
            ('coverage@10',  'Coverage@10'),
            ('rsi@10',       'RSI@10'),
        ]
        available = [(c, label) for c, label in metrics_to_plot if c in records.columns]
        if not available:
            print("    No suitable columns for boxplot — skipping chart10")
            return

        methods = sorted(records['method'].unique())
        n_plots = len(available)

        with plt.style.context(STYLE):
            fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6), sharey=False)
            if n_plots == 1:
                axes = [axes]
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
                ax.tick_params(axis='x', rotation=40)
                ax.grid(axis='y', alpha=0.4)
            plt.suptitle('Score Distribution Across Evaluation Windows', fontsize=13)
        _finalize(fig, self.output_dir / 'chart10_boxplot_windows.png', show)

    # ==================================================================
    # CHART 11 — Radar / Spider Chart
    # ==================================================================
    def chart11_radar_charts(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

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
            print("    Not enough metrics for radar chart — skipping chart11")
            return

        labels  = list(available.keys())
        n_axes  = len(labels)
        angles  = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
        angles += angles[:1]
        methods = list(summary.index)

        df = summary[[c for c in available.values()]].copy()
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)
        df_norm.columns = labels

        radar_dir = self.output_dir / 'radar'
        radar_dir.mkdir(exist_ok=True)

        # (A) Combined radar — all methods
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
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
                vals = df_norm.loc[method].values.tolist() + [df_norm.loc[method].values[0]]
                ax.plot(angles, vals, color=_c(method), linewidth=_lw(method),
                        label=method, marker=_mk(method), markersize=5)
                ax.fill(angles, vals, color=_c(method), alpha=0.05)
            ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
                      fontsize=9, framealpha=0.9)
            ax.set_title('Multi-Metric Performance Radar (all methods)', fontsize=12, pad=20)
        _finalize(fig, radar_dir / 'radar_all_methods.png', show)

        # (B) Individual radars grid
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
            if method in df_norm.index:
                vals = df_norm.loc[method].values.tolist() + [df_norm.loc[method].values[0]]
                ax.plot(angles, vals, color=_c(method), linewidth=2.0,
                        marker=_mk(method), markersize=5)
                ax.fill(angles, vals, color=_c(method), alpha=0.25)
            ax.set_title(method, size=11, pad=10,
                         fontweight='bold' if method == 'WSPI' else 'normal',
                         color=_c(method))
        for ax_i in range(len(methods), len(axes_flat)):
            axes_flat[ax_i].set_visible(False)
        plt.suptitle('Individual Performance Profiles (Radar Charts)', fontsize=13)
        _finalize(fig, radar_dir / 'radar_individual.png', show)
        print(f"    Radar charts saved to: {radar_dir}")

    # ==================================================================
    # Entry Points
    # ==================================================================
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

    def create_summary_report(self,
                               filter_top_percent=None,
                               filter_stratum=None,
                               save: bool = True,
                               show: bool = False):
        """
        Main entry point called by analyze_results.py.
        Generates all charts and saves them to the output directory.
        """
        print(f"  Output directory: {self.output_dir}")
        self.generate_all_charts(show=show)
        print(f"\n  ✓ Summary report complete — {self.output_dir}")
