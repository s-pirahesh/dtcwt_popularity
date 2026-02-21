"""
Results Visualizer — 4-Layer Frozen Evaluation Protocol
=========================================================
Charts produced:
  chart1_protocol_overview    — grouped bar: representative metric per layer
  chart2_stability_rsi        — RSI@K grouped bar
  chart3_robustness           — rank distortion bar (ΔRank)
  chart4_temporal_rsi         — RSI@10 rolling-mean trend (not raw points)
  chart5_ndcg_profile         — NDCG@K for K=5,10,20
  chart6_temporal_spearman    — Spearman ρ rolling-mean trend (not raw points)
  chart7_stratum_performance  — per-stratum Spearman bar
  chart8_metric_heatmap       — heatmap: RSI@5|RSI@10|RSI@20|NDCG@10|Spearman|ΔRank
  chart9_per_metric_bars      — one bar chart per metric
  chart10_boxplot_windows     — box plot per method across windows
  chart11_radar_charts        — radar: Spearman|NDCG@10|RSI@10|RSI@20|Robustness
  chart12_composite_score     — composite = 0.75·RSI + 0.25·NDCG (normalised)

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
# Colour / marker / style palette
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    'AF':          '#2196F3',   # blue      — traditional
    'EWMA':        '#FF9800',   # orange    — traditional
    'RRD':         '#00BCD4',   # cyan      — traditional
    'VSE':         '#795548',   # brown     — traditional
    'CompoundPop': '#607D8B',   # blue-grey — traditional
    'PFRF':        '#009688',   # teal      — traditional
    'DWT+AF':      '#9C27B0',   # purple    — wavelet (proposed)
    'DTCWT+AF':    '#F44336',   # red       — wavelet (proposed)
    'WSPI':        '#E91E63',   # pink/bold — proposed (best)
}
METHOD_MARKERS = {
    'AF': 'o', 'EWMA': '^', 'RRD': 'D',
    'VSE': 'p', 'CompoundPop': 'h', 'PFRF': 's',
    'DWT+AF': 'D', 'DTCWT+AF': 'v', 'WSPI': '*',
}

# Wavelet methods — proposed in this paper → rendered bold
WAVELET_METHODS = {'DWT+AF', 'DTCWT+AF', 'WSPI'}

STYLE = 'seaborn-v0_8-darkgrid'
DPI   = 300


def _c(m):   return METHOD_COLORS.get(m, '#607D8B')
def _mk(m):  return METHOD_MARKERS.get(m, 'o')
def _lw(m):  return 3.0 if m == 'WSPI' else (2.2 if m in WAVELET_METHODS else 1.5)
def _is_wavelet(m): return m in WAVELET_METHODS


def _bar_style(bars, methods):
    """Apply bold edge to wavelet methods, extra bold to WSPI."""
    for bar, method in zip(bars, methods):
        if method == 'WSPI':
            bar.set_edgecolor('#C2185B')
            bar.set_linewidth(3.0)
            bar.set_alpha(1.0)
        elif method in WAVELET_METHODS:
            bar.set_edgecolor('#4A148C')
            bar.set_linewidth(2.0)
            bar.set_alpha(0.95)


def _finalize(fig, path, show):
    plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    print(f"    Saved: {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _label_style(ax, methods):
    """Bold x-axis tick labels for wavelet methods."""
    for lbl in ax.get_xticklabels():
        txt = lbl.get_text()
        if txt in WAVELET_METHODS:
            lbl.set_fontweight('bold')
            lbl.set_color(_c(txt))


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
    # Helpers
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
    # CHART 1 — Protocol Overview (unchanged)
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
                _bar_style(bars, methods)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            _label_style(ax, methods)
            ax.set_ylabel('Score')
            ax.set_title('Protocol Overview — Representative Metric per Layer')
            ax.legend()
            ax.set_ylim(0, 1.15)
        _finalize(fig, self.output_dir / 'chart1_protocol_overview.png', show)

    # ==================================================================
    # CHART 2 — Stability RSI@K (unchanged logic, bold wavelet)
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
                bars  = ax.bar(x + i * width - (len(rsi_cols) - 1) * width / 2,
                               vals, width=width * 0.9, label=label, alpha=0.85)
                _bar_style(bars, methods)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            _label_style(ax, methods)
            ax.set_ylabel('RSI (Jaccard Similarity)')
            ax.set_title('Ranking Stability Index (RSI) — Higher is More Stable')
            ax.legend()
            ax.set_ylim(0, 1.05)
        _finalize(fig, self.output_dir / 'chart2_stability_rsi.png', show)

    # ==================================================================
    # CHART 3 — Robustness ΔRank (bold wavelet)
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
            _bar_style(bars, methods)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(vals) * 0.01, f'{v:.2f}',
                        ha='center', va='bottom', fontsize=9)
            ax.set_ylabel('Mean ΔRank (lower = more robust)')
            ax.set_title('Robustness to Noise: Rank Distortion (ΔRank)')
            ax.tick_params(axis='x', rotation=30)
            _label_style(ax, methods)
        _finalize(fig, self.output_dir / 'chart3_robustness.png', show)

    # ==================================================================
    # CHART 4 — Temporal RSI@10 — Rolling-Mean Trend
    #   Raw per-window points are too noisy when window count is large
    #   (e.g. 8000+ windows for Uber).  We compute a rolling mean with
    #   an adaptive window (~5 % of total windows, min 10) and show the
    #   smoothed trend line.  A light shaded band (±1 std, also rolling)
    #   is drawn only for WSPI to highlight its stability advantage.
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

        # Adaptive rolling window: ~5 % of total windows, at least 10
        n_total = records['window_idx'].nunique()
        roll_w  = max(10, n_total // 20)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(13, 5))
            for method, grp in records.groupby('method'):
                grp_s  = grp.sort_values('window_idx').reset_index(drop=True)
                rolled = grp_s[rsi_col].rolling(roll_w, center=True, min_periods=1)
                trend  = rolled.mean()
                x_vals = grp_s['window_idx'].values

                lw = _lw(method)
                ax.plot(x_vals, trend,
                        label=method, color=_c(method),
                        linewidth=lw, alpha=0.9)

                # Shaded ±1 std band only for wavelet methods to reduce clutter
                if method in WAVELET_METHODS:
                    std = rolled.std().fillna(0)
                    ax.fill_between(x_vals,
                                    (trend - std).clip(0), (trend + std).clip(upper=1),
                                    color=_c(method), alpha=0.10)

            ax.set_xlabel('Window Index')
            ax.set_ylabel('RSI@10')
            ax.set_title(
                f'Ranking Stability Trend Over Time (RSI@10)\n'
                f'Rolling mean, window = {roll_w} — shaded band = ±1 std (wavelet methods only)')
            ax.legend(fontsize=8, ncol=3)
            ax.set_ylim(0, 1.05)
        _finalize(fig, self.output_dir / 'chart4_temporal_rsi.png', show)

    # ==================================================================
    # CHART 5 — NDCG@K Profile (bold wavelet)
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
                bars = ax.bar(x + i * width - (len(available) - 1) * width / 2,
                              vals, width=width * 0.9, label=f'NDCG@{k}', alpha=0.85)
                _bar_style(bars, methods)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha='right')
            _label_style(ax, methods)
            ax.set_ylabel('NDCG Score')
            ax.set_title('NDCG@K Profile (K = 5, 10, 20)')
            ax.legend()
            ax.set_ylim(0, 1.05)
        _finalize(fig, self.output_dir / 'chart5_ndcg_profile.png', show)

    # ==================================================================
    # CHART 6 — Temporal Spearman ρ — Rolling-Mean Trend
    #   Same smoothing strategy as chart4: adaptive rolling mean so the
    #   trend is visible even with thousands of windows.
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

        n_total = records['window_idx'].nunique()
        roll_w  = max(10, n_total // 20)

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(13, 5))
            for method, grp in records.groupby('method'):
                grp_s  = grp.sort_values('window_idx').reset_index(drop=True)
                rolled = grp_s[rho_col].rolling(roll_w, center=True, min_periods=1)
                trend  = rolled.mean()
                x_vals = grp_s['window_idx'].values

                ax.plot(x_vals, trend,
                        label=method, color=_c(method),
                        linewidth=_lw(method), alpha=0.9)

                if method in WAVELET_METHODS:
                    std = rolled.std().fillna(0)
                    ax.fill_between(x_vals,
                                    trend - std, trend + std,
                                    color=_c(method), alpha=0.10)

            ax.set_xlabel('Window Index')
            ax.set_ylabel('Spearman ρ')
            ax.set_title(
                f'Spearman Rank Correlation Trend Over Time\n'
                f'Rolling mean, window = {roll_w} — shaded band = ±1 std (wavelet methods only)')
            ax.legend(fontsize=8, ncol=3)
            ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        _finalize(fig, self.output_dir / 'chart6_temporal_spearman.png', show)

    # ==================================================================
    # CHART 7 — Per-Stratum Spearman (unchanged)
    # ==================================================================
    def chart7_stratum_performance(self, show=False):
        strata_data = self._get_stratum_summary()
        if strata_data is None or strata_data.empty:
            print("    No stratum data — skipping chart7")
            return

        spearman_col = next((c for c in strata_data.columns if 'spearman' in c.lower()), None)
        stratum_col  = next((c for c in strata_data.columns if 'stratum' in c.lower()), None)
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
                bars = ax.bar(x + i * width - (len(methods) - 1) * width / 2,
                              vals, width=width * 0.9,
                              label=method, color=_c(method), alpha=0.85)
                _bar_style(bars, [method] * len(strata))
            ax.set_xticks(x)
            ax.set_xticklabels(strata, rotation=0)
            ax.set_ylabel('Mean Spearman ρ')
            ax.set_title('Spearman Correlation by Popularity Stratum')
            ax.legend(fontsize=8, ncol=3)
        _finalize(fig, self.output_dir / 'chart7_stratum_performance.png', show)

    # ==================================================================
    # CHART 8 — Metric Heatmap  *** REVISED ***
    #   Columns: RSI@5 | RSI@10 | RSI@20 | NDCG@10   (3 RSI + 1 NDCG)
    #   Composite score: 0.75 × mean(RSI) + 0.25 × NDCG@10
    # ==================================================================
    def chart8_metric_heatmap(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        # ---- Column selection: 3 RSI + NDCG@10 + Spearman + ΔRank (inverted) ----
        desired_cols = [
            ('mean_rsi@5',                'RSI@5'),
            ('mean_rsi@10',               'RSI@10'),
            ('mean_rsi@20',               'RSI@20'),
            ('mean_ndcg@10',              'NDCG@10'),
            ('mean_spearman_rho',         'Spearman ρ'),
            ('mean_robustness_distortion','ΔRank↓'),   # lower = better → will be inverted
        ]
        cols_present = [(c, lbl) for c, lbl in desired_cols if c in summary.columns]
        if not cols_present:
            print("    No suitable columns for heatmap — skipping chart8")
            return

        raw_cols, display_names = zip(*cols_present)
        df = summary[list(raw_cols)].copy()
        df.columns = list(display_names)

        # --- Normalise each column 0–1; ΔRank↓ is inverted (lower raw = higher norm) ---
        df_norm = pd.DataFrame(index=df.index, dtype=float)
        for col in df.columns:
            mn, mx = df[col].min(), df[col].max()
            rng = mx - mn + 1e-12
            if col == 'ΔRank↓':
                df_norm[col] = (mx - df[col]) / rng   # invert
            else:
                df_norm[col] = (df[col] - mn) / rng

        # --- Composite score: RSI columns 3×, NDCG + Spearman 1×, ΔRank (inv) 2× ---
        rsi_labels   = [c for c in df_norm.columns if c.startswith('RSI')]
        ndcg_labels  = [c for c in df_norm.columns if c.startswith('NDCG')]
        spear_labels = [c for c in df_norm.columns if 'Spearman' in c]
        rob_labels   = [c for c in df_norm.columns if 'ΔRank' in c]

        weight_sum = (3 * len(rsi_labels) + 1 * len(ndcg_labels)
                      + 1 * len(spear_labels) + 2 * len(rob_labels))
        composite = (
            df_norm[rsi_labels].sum(axis=1)  * 3
          + df_norm[ndcg_labels].sum(axis=1) * 1
          + df_norm[spear_labels].sum(axis=1)* 1
          + df_norm[rob_labels].sum(axis=1)  * 2
        ) / weight_sum
        composite = composite.sort_values(ascending=False)

        n_methods = len(df)
        fig, (ax_heat, ax_comp) = plt.subplots(
            1, 2, figsize=(18, max(4, n_methods * 0.75 + 2)),
            gridspec_kw={'width_ratios': [3, 1]}
        )

        # Reorder rows by composite score
        df_plot      = df_norm.loc[composite.index]
        df_raw_plot  = df.loc[composite.index]

        # For display annotations use raw values (ΔRank shows raw, others show raw)
        sns.heatmap(df_plot, ax=ax_heat, cmap='RdYlGn', vmin=0, vmax=1,
                    linewidths=0.5, linecolor='white',
                    annot=df_raw_plot.round(3), fmt='.3f',
                    annot_kws={'size': 8},
                    cbar_kws={'label': 'Relative Score (normalised; ΔRank inverted)'})
        ax_heat.set_title(
            'Method × Metric Heatmap\n(RSI@5, RSI@10, RSI@20 | NDCG@10 | Spearman ρ | ΔRank↓)',
            fontsize=12, pad=10)

        # Bold / colour wavelet method labels on y-axis
        for lbl in ax_heat.get_yticklabels():
            txt = lbl.get_text()
            if txt in WAVELET_METHODS:
                lbl.set_fontweight('bold')
                lbl.set_color(_c(txt))

        # Composite score bar chart
        colors  = [_c(m) for m in composite.index]
        h_bars  = ax_comp.barh(range(len(composite)), composite.values,
                               color=colors, alpha=0.85)
        # Bold border for wavelet methods
        for bar, method in zip(h_bars, composite.index):
            if method in WAVELET_METHODS:
                bar.set_edgecolor(_c(method))
                bar.set_linewidth(2.5 if method == 'WSPI' else 1.8)
        ax_comp.set_yticks(range(len(composite)))
        ax_comp.set_yticklabels(composite.index)
        # Bold y-tick labels
        for lbl in ax_comp.get_yticklabels():
            if lbl.get_text() in WAVELET_METHODS:
                lbl.set_fontweight('bold')
                lbl.set_color(_c(lbl.get_text()))
        ax_comp.set_xlabel('Composite Score\n(RSI×3 + ΔRank⁻¹×2 + NDCG + Spearman)')
        ax_comp.set_title('Weighted\nOverall Rank')
        ax_comp.set_xlim(0, 1.05)
        for i, (method, val) in enumerate(composite.items()):
            ax_comp.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=8,
                         fontweight='bold' if method in WAVELET_METHODS else 'normal')

        plt.suptitle(
            'Comprehensive Performance Comparison\n'
            'Composite = RSI×3 + ΔRank⁻¹×2 + NDCG@10 + Spearman (all normalised)',
            fontsize=12, y=1.02)
        _finalize(fig, self.output_dir / 'chart8_metric_heatmap.png', show)

    # ==================================================================
    # CHART 9 — Per-Metric Individual Bar Charts (bold wavelet)
    # ==================================================================
    def chart9_per_metric_bars(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        metric_groups = {
            'Spearman ρ':  ('mean_spearman_rho', 'Spearman Rank Correlation (ρ) — Higher is Better'),
            'NDCG@10':     ('mean_ndcg@10',      'NDCG@10 — Ranking Quality (Higher is Better)'),
            'RSI@5':       ('mean_rsi@5',         'RSI@5 — Ranking Stability (Higher is Better)'),
            'RSI@10':      ('mean_rsi@10',        'RSI@10 — Ranking Stability (Higher is Better)'),
            'RSI@20':      ('mean_rsi@20',        'RSI@20 — Ranking Stability (Higher is Better)'),
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
                _bar_style(bars, methods)
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(vals) * 0.01, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold')
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(title, fontsize=12)
                ax.tick_params(axis='x', rotation=30)
                _label_style(ax, methods)
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
    # CHART 10 — Box Plot (unchanged logic, bold wavelet)
    # ==================================================================
    def chart10_boxplot_windows(self, show=False):
        records = self._get_protocol_records()
        if records is None:
            print("    No per-window records — skipping chart10")
            return

        metrics_to_plot = [
            ('spearman_rho', 'Spearman ρ'),
            ('ndcg@10',      'NDCG@10'),
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
                    if method in WAVELET_METHODS:
                        patch.set_linewidth(2.5 if method == 'WSPI' else 1.8)
                ax.set_title(label, fontsize=11)
                ax.tick_params(axis='x', rotation=40)
                _label_style(ax, methods)
                ax.grid(axis='y', alpha=0.4)
            plt.suptitle('Score Distribution Across Evaluation Windows', fontsize=13)
        _finalize(fig, self.output_dir / 'chart10_boxplot_windows.png', show)

    # ==================================================================
    # CHART 11 — Radar / Spider Chart  *** REVISED ***
    #   Axes: Spearman ρ | NDCG@10 | RSI@10 | RSI@20 | Robustness (1/ΔRank, normalised)
    #   - Kendall τ removed (collinear with Spearman)
    #   - ΔRank inverted → higher normalised value = better robustness
    #   - Wavelet methods (DWT+AF, DTCWT+AF, WSPI) drawn BOLD / thick
    # ==================================================================
    def chart11_radar_charts(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        # ---- Revised axes ----
        radar_metrics_cfg = [
            ('Spearman ρ',   'mean_spearman_rho',          False),  # (label, col, invert)
            ('NDCG@10',      'mean_ndcg@10',                False),
            ('RSI@10',       'mean_rsi@10',                 False),
            ('RSI@20',       'mean_rsi@20',                 False),
            ('Robustness\n(1/ΔRank)', 'mean_robustness_distortion', True),  # invert
        ]

        available = [(lbl, col, inv) for lbl, col, inv in radar_metrics_cfg
                     if col in summary.columns]
        if len(available) < 3:
            print("    Not enough metrics for radar chart — skipping chart11")
            return

        labels  = [lbl for lbl, _, _ in available]
        cols    = [col for _, col, _ in available]
        inverts = [inv for _, _, inv in available]
        n_axes  = len(labels)
        angles  = [n / float(n_axes) * 2 * np.pi for n in range(n_axes)]
        angles += angles[:1]
        methods = list(summary.index)

        df = summary[cols].copy()
        df.columns = labels

        # Normalise: for inverted metrics, lower raw = higher normalised
        df_norm = pd.DataFrame(index=df.index, columns=labels, dtype=float)
        for lbl, inv in zip(labels, inverts):
            col_data = df[lbl]
            mn, mx = col_data.min(), col_data.max()
            rng = mx - mn + 1e-12
            if inv:
                df_norm[lbl] = (mx - col_data) / rng  # invert
            else:
                df_norm[lbl] = (col_data - mn) / rng

        radar_dir = self.output_dir / 'radar'
        radar_dir.mkdir(exist_ok=True)

        # Helper to classify methods
        def _lw_radar(m):
            if m == 'WSPI':      return 3.2
            if m in WAVELET_METHODS: return 2.4
            return 1.4

        def _alpha_radar(m):
            if m == 'WSPI':      return 0.18
            if m in WAVELET_METHODS: return 0.10
            return 0.04

        # ---- (A) Combined radar — all methods ----
        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=10)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=8)

            # Draw traditional methods first (behind)
            for method in methods:
                if method in WAVELET_METHODS or method not in df_norm.index:
                    continue
                vals = df_norm.loc[method].values.tolist() + [df_norm.loc[method].values[0]]
                ax.plot(angles, vals, color=_c(method), linewidth=1.4,
                        label=method, marker=_mk(method), markersize=4, alpha=0.75,
                        linestyle='--')
                ax.fill(angles, vals, color=_c(method), alpha=0.03)

            # Draw wavelet methods on top (bold)
            for method in [m for m in ['DWT+AF', 'DTCWT+AF', 'WSPI'] if m in methods]:
                if method not in df_norm.index:
                    continue
                vals = df_norm.loc[method].values.tolist() + [df_norm.loc[method].values[0]]
                lw = _lw_radar(method)
                ls = '-'
                ax.plot(angles, vals, color=_c(method), linewidth=lw,
                        label=f'★ {method}' if method == 'WSPI' else f'● {method}',
                        marker=_mk(method), markersize=7 if method == 'WSPI' else 5,
                        alpha=1.0, linestyle=ls, zorder=10)
                ax.fill(angles, vals, color=_c(method), alpha=_alpha_radar(method))

            ax.legend(loc='upper right', bbox_to_anchor=(1.45, 1.20),
                      fontsize=9, framealpha=0.9)
            ax.set_title(
                'Multi-Metric Performance Radar\n'
                '(Axes: Spearman ρ | NDCG@10 | RSI@10 | RSI@20 | Robustness)\n'
                'Bold lines = proposed wavelet methods',
                fontsize=11, pad=25)
        _finalize(fig, radar_dir / 'radar_all_methods.png', show)

        # ---- (B) Individual radars grid ----
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
            ax.set_xticklabels(labels, size=8)
            ax.set_ylim(0, 1)
            if method in df_norm.index:
                vals = df_norm.loc[method].values.tolist() + [df_norm.loc[method].values[0]]
                lw  = _lw_radar(method)
                ax.plot(angles, vals, color=_c(method), linewidth=lw,
                        marker=_mk(method), markersize=5)
                ax.fill(angles, vals, color=_c(method), alpha=0.25)
            fw    = 'bold' if method in WAVELET_METHODS else 'normal'
            label = f'★ {method}' if method == 'WSPI' else (
                    f'● {method}' if method in WAVELET_METHODS else method)
            ax.set_title(label, size=11, pad=10, fontweight=fw, color=_c(method))
        for ax_i in range(len(methods), len(axes_flat)):
            axes_flat[ax_i].set_visible(False)
        plt.suptitle(
            'Individual Performance Profiles (Radar Charts)\n'
            'Bold title + thick line = proposed wavelet method',
            fontsize=12)
        _finalize(fig, radar_dir / 'radar_individual.png', show)
        print(f"    Radar charts saved to: {radar_dir}")

    # ==================================================================
    # CHART 12 — Composite Score Bar  *** NEW ***
    #   composite = 0.75 × mean(RSI@5,10,20) + 0.25 × NDCG@10   (normalised)
    # ==================================================================
    def chart12_composite_score(self, show=False):
        summary = self._get_summary_df()
        if summary is None:
            return

        rsi_cols  = [c for c in ['mean_rsi@5', 'mean_rsi@10', 'mean_rsi@20']
                     if c in summary.columns]
        ndcg_col  = 'mean_ndcg@10'
        if not rsi_cols or ndcg_col not in summary.columns:
            print("    Missing RSI/NDCG columns — skipping chart12")
            return

        # Normalise
        def _norm(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn + 1e-12)

        rsi_norm  = summary[rsi_cols].apply(_norm).mean(axis=1)
        ndcg_norm = _norm(summary[ndcg_col])
        composite = 0.75 * rsi_norm + 0.25 * ndcg_norm
        composite = composite.sort_values(ascending=False)

        methods = list(composite.index)
        vals    = composite.values

        with plt.style.context(STYLE):
            fig, ax = plt.subplots(figsize=(max(10, len(methods) * 1.2), 5))
            bars = ax.bar(methods, vals, color=[_c(m) for m in methods], alpha=0.85, zorder=3)
            _bar_style(bars, methods)
            for bar, val, method in zip(bars, vals, methods):
                fw = 'bold' if method in WAVELET_METHODS else 'normal'
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f'{val:.3f}',
                        ha='center', va='bottom', fontsize=9, fontweight=fw)
            ax.set_ylabel('Composite Score (0–1)', fontsize=11)
            ax.set_title(
                'Composite Performance Score\n'
                '= 0.75 × norm(mean RSI@5,10,20) + 0.25 × norm(NDCG@10)\n'
                'Bold bars = proposed wavelet methods',
                fontsize=11)
            ax.tick_params(axis='x', rotation=30)
            _label_style(ax, methods)
            ax.set_ylim(0, 1.15)
            ax.grid(axis='y', alpha=0.4, zorder=0)
            # Annotate best
            ax.annotate('★ Best', xy=(0, vals[0]),
                        xytext=(0, vals[0] + 0.06),
                        ha='center', fontsize=9, color='#1B5E20', fontweight='bold')
        _finalize(fig, self.output_dir / 'chart12_composite_score.png', show)

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
            ('chart12_composite_score',    self.chart12_composite_score),
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
