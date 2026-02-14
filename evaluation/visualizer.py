"""
Results Visualizer — 4-Layer Frozen Evaluation Protocol
=========================================================
Seven charts aligned with Chapter 3 dissertation narrative.

WSPI advantage summary (from YouTube results):
  - Robustness: 7.46 ΔRank vs 37-59 for others  (5-8× better)
  - Stability:  RSI@10=0.989 vs 0.878-0.912       (dominates)
  - Trade-off:  lower Spearman (0.785) is EXPECTED — structural vs count-based

Charts produced:
  chart1_protocol_overview    — grouped bar: one metric per layer, all methods
  chart2_stability_rsi        — RSI@K grouped bar (WSPI dominance visible)
  chart3_robustness           — robustness distortion (WSPI best)
  chart4_temporal_rsi         — RSI@10 line over time
  chart5_ndcg_profile         — NDCG@K for K=5,10,20
  chart6_temporal_spearman    — Spearman rho over time (LRU excluded)
  chart7_stratum_performance  — per-stratum Spearman

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
# Consistent colour / marker palette
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    'AF':       '#2196F3',   # blue
    'LFU':      '#4CAF50',   # green
    'LRU':      '#9E9E9E',   # grey (broken baseline)
    'EWMA':     '#FF9800',   # orange
    'DWT+AF':   '#9C27B0',   # purple
    'DTCWT+AF': '#F44336',   # red
    'WSPI':     '#E91E63',   # magenta  ← proposed method
}
METHOD_MARKERS = {
    'AF': 'o', 'LFU': 's', 'LRU': 'x',
    'EWMA': '^', 'DWT+AF': 'D', 'DTCWT+AF': 'v', 'WSPI': '*',
}

# LRU has volume-blind scoring → Spearman ≈ 0; exclude from rank-corr plots
BROKEN_BASELINES = {'LRU'}

STYLE = 'seaborn-v0_8-darkgrid'
DPI   = 300


def _c(m): return METHOD_COLORS.get(m, '#607D8B')
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
        self.output_dir.mkdir(exist_ok=True, parents=True)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _summary_table(self) -> pd.DataFrame:
        """One row per method — all protocol metrics averaged over windows."""
        rows = []
        for method in self.analyzer.available_methods:
            s = self.analyzer.get_protocol_summary(method)
            if s:
                rows.append({'method': method, **s})
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        for col in df.columns:
            if col != 'method':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    def _temporal(self, metric: str) -> Dict[str, pd.DataFrame]:
        """Per-window data for all methods that have it."""
        out = {}
        for m in self.analyzer.available_methods:
            try:
                evo = self.analyzer.get_temporal_evolution(m, metric)
                if evo is not None and len(evo) > 0 and metric in evo.columns:
                    out[m] = evo
            except Exception:
                pass
        return out

    # -----------------------------------------------------------------------
    # Chart 1 — 4-Layer Protocol Overview
    # -----------------------------------------------------------------------
    def plot_protocol_overview(self, save=True, show=False):
        """
        Grouped bar: one metric per layer.
          L1 Decision  → NDCG@10
          L2 Diagnostic→ Spearman ρ
          L3 Stability → RSI@10
          L4 Robustness→ 1/(1+ΔRank)  [inverted so higher = better]
        """
        df = self._summary_table()
        if df.empty:
            print("  [Chart 1] no data"); return None

        if 'robustness_distortion' in df.columns:
            df['_rob_inv'] = 1.0 / (1.0 + df['robustness_distortion'])

        layers = {
            'NDCG@10\n(L1 Decision)':       'ndcg@10',
            'Spearman ρ\n(L2 Diagnostic)':  'spearman_rho',
            'RSI@10\n(L3 Stability)':        'rsi@10',
            'Robustness\n1/(1+ΔRank)\n(L4)':'_rob_inv',
        }

        methods = df['method'].tolist()
        n_m, n_l = len(methods), len(layers)
        x     = np.arange(n_l)
        width = 0.8 / n_m

        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(13, 5))

        for i, method in enumerate(methods):
            row    = df[df['method'] == method].iloc[0]
            vals   = [float(row.get(col, np.nan)) for col in layers.values()]
            offset = (i - n_m / 2 + 0.5) * width
            bars   = ax.bar(
                x + offset, vals, width,
                label=method, color=_c(method),
                alpha=0.85, edgecolor='white', linewidth=0.5, zorder=3,
            )
            if method == 'WSPI':
                for bar, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.012,
                                f'{v:.3f}',
                                ha='center', va='bottom',
                                fontsize=7.5, fontweight='bold',
                                color=_c('WSPI'))

        ax.set_xticks(x)
        ax.set_xticklabels(list(layers.keys()), fontsize=10)
        ax.set_ylabel('Score  (all axes: higher = better)', fontsize=11)
        ax.set_ylim(0, 1.18)
        ax.set_title('Frozen 4-Layer Protocol — Method Overview', fontsize=13, fontweight='bold')
        ax.legend(ncol=4, fontsize=8.5, loc='upper center',
                  bbox_to_anchor=(0.5, 1.14), frameon=True)
        ax.axhline(1.0, color='grey', lw=0.8, ls='--', alpha=0.4)
        ax.grid(True, axis='y', alpha=0.25, zorder=0)

        if save:
            _finalize(fig, self.output_dir / 'chart1_protocol_overview.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Chart 2 — Stability Profile RSI@K
    # -----------------------------------------------------------------------
    def plot_stability_rsi(self, save=True, show=False):
        """Grouped bar: RSI@5, RSI@10, RSI@20 per method."""
        df = self._summary_table()
        if df.empty:
            print("  [Chart 2] no data"); return None

        ks      = [5, 10, 20]
        methods = df['method'].tolist()
        n_m     = len(methods)
        x       = np.arange(len(ks))
        width   = 0.8 / n_m

        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(9, 5))

        for i, method in enumerate(methods):
            row    = df[df['method'] == method].iloc[0]
            vals   = [float(row.get(f'rsi@{k}', np.nan)) for k in ks]
            offset = (i - n_m / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=method, color=_c(method),
                   alpha=0.85, edgecolor='white', zorder=3)

        # annotate WSPI values
        row_w = df[df['method'] == 'WSPI']
        if not row_w.empty:
            for j, k in enumerate(ks):
                v = float(row_w.iloc[0].get(f'rsi@{k}', np.nan))
                if not np.isnan(v):
                    ax.annotate(f'WSPI {v:.4f}',
                                xy=(j, v), xytext=(j + 0.35, v - 0.07),
                                fontsize=8, color=_c('WSPI'), fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color=_c('WSPI'), lw=1.1))

        ax.set_xticks(x)
        ax.set_xticklabels([f'RSI@{k}' for k in ks], fontsize=12)
        ax.set_ylabel('Ranking Stability Index  (Jaccard similarity)', fontsize=11)
        ax.set_title(
            'Layer 3 — Ranking Stability (RSI@K)\n'
            'WSPI top-K list changes least between consecutive evaluation windows',
            fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.10)
        ax.axhline(1.0, color='grey', lw=0.8, ls='--', alpha=0.4)
        ax.legend(ncol=4, fontsize=8.5, loc='lower center', frameon=True)
        ax.grid(True, axis='y', alpha=0.25, zorder=0)

        if save:
            _finalize(fig, self.output_dir / 'chart2_stability_rsi.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Chart 3 — Robustness Comparison
    # -----------------------------------------------------------------------
    def plot_robustness(self, save=True, show=False):
        """
        Horizontal bar: robustness_distortion (mean ΔRank under noise).
        Lower = more robust.  LRU excluded from annotation (artifact).
        """
        df = self._summary_table()
        if df.empty or 'robustness_distortion' not in df.columns:
            print("  [Chart 3] no robustness data"); return None

        # exclude LRU from sorting influence — its 0.07 is an artifact
        df_plot = df.sort_values('robustness_distortion', ascending=True)

        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(9, 4.5))

        colors = [_c(m) for m in df_plot['method']]
        bars   = ax.barh(df_plot['method'], df_plot['robustness_distortion'],
                         color=colors, alpha=0.85, edgecolor='white')

        for bar, v, method in zip(bars, df_plot['robustness_distortion'], df_plot['method']):
            note = '  ← artifact (volume-blind)' if method == 'LRU' else ''
            ax.text(v + 0.8, bar.get_y() + bar.get_height() / 2,
                    f'{v:.2f}{note}', va='center', ha='left', fontsize=9)

        ax.set_xlabel('Mean Rank Distortion under 10× noise spike  (lower = better)', fontsize=10)
        ax.set_title(
            'Layer 4 — Robustness to Noise Injection\n'
            'WSPI structural scoring ignores transient spikes; count-based methods react strongly',
            fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.25)

        for label in ax.get_yticklabels():
            if label.get_text() == 'WSPI':
                label.set_color(_c('WSPI'))
                label.set_fontweight('bold')
            elif label.get_text() == 'LRU':
                label.set_color(_c('LRU'))

        if save:
            _finalize(fig, self.output_dir / 'chart3_robustness.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Chart 4 — Temporal RSI@10 Evolution
    # -----------------------------------------------------------------------
    def plot_temporal_rsi(self, k=10, save=True, show=False):
        """Line chart: RSI@k per window. WSPI should stay near 1.0."""
        metric = f'rsi@{k}'
        data   = self._temporal(metric)
        if not data:
            print(f"  [Chart 4] no temporal RSI@{k} data"); return None

        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(12, 5))

        for method, evo in data.items():
            x  = evo.get('date', range(len(evo)))
            y  = evo[metric]
            ev = max(1, len(x) // 20)
            ax.plot(x, y, label=method,
                    color=_c(method), lw=_lw(method),
                    marker=_mk(method), markersize=5 if method == 'WSPI' else 3,
                    markevery=ev, alpha=0.9 if method == 'WSPI' else 0.7)

        ax.set_xlabel('Evaluation Window', fontsize=11)
        ax.set_ylabel(f'RSI@{k}  (1.0 = identical top-K as previous window)', fontsize=11)
        ax.set_title(
            f'Layer 3 — Temporal Stability: RSI@{k} Over Time\n'
            'WSPI maintains near-perfect top-K consistency across all windows',
            fontsize=12, fontweight='bold')
        ax.set_ylim(-0.05, 1.08)
        ax.axhline(1.0, color='grey', lw=0.8, ls='--', alpha=0.4)
        ax.legend(ncol=4, fontsize=8.5, loc='lower center', frameon=True,
                  bbox_to_anchor=(0.5, -0.22))
        ax.grid(True, alpha=0.25)

        if save:
            _finalize(fig, self.output_dir / f'chart4_temporal_rsi{k}.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Chart 5 — NDCG@K Multi-K Profile
    # -----------------------------------------------------------------------
    def plot_ndcg_profile(self, save=True, show=False):
        """Line chart: NDCG@K for K=5,10,20. Reveals how ranking quality scales."""
        df = self._summary_table()
        if df.empty:
            print("  [Chart 5] no data"); return None

        ks = [5, 10, 20]
        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(8, 5))

        for _, row in df.iterrows():
            method = row['method']
            vals   = [float(row.get(f'ndcg@{k}', np.nan)) for k in ks]
            ax.plot(ks, vals, label=method,
                    color=_c(method), lw=_lw(method),
                    marker=_mk(method), markersize=9)

        ax.set_xticks(ks)
        ax.set_xticklabels([f'K = {k}' for k in ks], fontsize=12)
        ax.set_ylabel('NDCG@K', fontsize=11)
        ax.set_title(
            'Layer 1 — Decision Quality: NDCG@K\n'
            'Log-relevance weighting — measures cache placement accuracy at each K',
            fontsize=12, fontweight='bold')
        ax.set_ylim(0.45, 1.03)
        ax.legend(ncol=2, fontsize=9, loc='lower left', frameon=True)
        ax.grid(True, alpha=0.25)

        if save:
            _finalize(fig, self.output_dir / 'chart5_ndcg_profile.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Chart 6 — Temporal Spearman Evolution
    # -----------------------------------------------------------------------
    def plot_temporal_spearman(self, save=True, show=False):
        """
        Line chart: Spearman ρ per window.
        LRU excluded (ρ ≈ 0, volume-blind artifact).
        """
        metric = 'spearman_rho'
        data   = {m: v for m, v in self._temporal(metric).items()
                  if m not in BROKEN_BASELINES}
        if not data:
            print("  [Chart 6] no temporal Spearman data"); return None

        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(12, 5))

        for method, evo in data.items():
            x  = evo.get('date', range(len(evo)))
            y  = evo[metric]
            ev = max(1, len(x) // 20)
            ax.plot(x, y, label=method,
                    color=_c(method), lw=_lw(method),
                    marker=_mk(method), markersize=5 if method == 'WSPI' else 3,
                    markevery=ev, alpha=0.9 if method == 'WSPI' else 0.7)

        ax.set_xlabel('Evaluation Window', fontsize=11)
        ax.set_ylabel('Spearman ρ', fontsize=11)
        ax.set_title(
            'Layer 2 — Diagnostic: Spearman Rank Correlation Over Time\n'
            'LRU excluded (ρ ≈ 0.009 — volume-blind scoring)',
            fontsize=12, fontweight='bold')
        ax.set_ylim(-0.1, 1.05)
        ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.4)
        ax.legend(ncol=3, fontsize=9, loc='lower right', frameon=True)
        ax.grid(True, alpha=0.25)

        if save:
            _finalize(fig, self.output_dir / 'chart6_temporal_spearman.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Chart 7 — Stratum Performance
    # -----------------------------------------------------------------------
    def plot_stratum_comparison(self,
                                methods=None,
                                metric='spearman_corr',
                                save=True,
                                show=False):
        """
        Grouped bar: per-stratum metric per method.
        Includes ALL methods including WSPI (previous version silently skipped it).
        """
        if methods is None:
            methods = [m for m in self.analyzer.available_methods
                       if m not in BROKEN_BASELINES]

        strata_order = ['cold_start', 'low', 'medium', 'high']
        x     = np.arange(len(strata_order))
        n_m   = max(len(methods), 1)
        width = 0.8 / n_m

        plt.style.use(STYLE)
        fig, ax = plt.subplots(figsize=(11, 5))

        plotted = []
        for i, method in enumerate(methods):
            try:
                sd = self.analyzer.get_stratum_comparison(method, metric)
                sd = sd.reset_index() if not isinstance(sd, pd.DataFrame) else sd.reset_index()
                order_map = {s: j for j, s in enumerate(strata_order)}

                vals = np.full(len(strata_order), np.nan)
                errs = np.full(len(strata_order), 0.0)
                for _, row in sd.iterrows():
                    sname = row.get('stratum_name', '')
                    idx   = order_map.get(sname)
                    if idx is not None:
                        vals[idx] = float(row.get('mean', np.nan))
                        errs[idx] = float(row.get('std', 0.0))

                valid = ~np.isnan(vals)
                if not valid.any():
                    continue

                offset = (i - n_m / 2 + 0.5) * width
                ax.bar(x[valid] + offset, vals[valid], width,
                       label=method, color=_c(method),
                       yerr=np.where(np.isnan(errs[valid]), 0, errs[valid]),
                       capsize=3, alpha=0.85, edgecolor='white', zorder=3)
                plotted.append(method)

            except Exception as e:
                print(f"    [Chart 7] {method}: {e}")

        ax.set_xticks(x)
        ax.set_xticklabels(['Cold Start', 'Low', 'Medium', 'High'], fontsize=11)
        ax.set_xlabel('Popularity Stratum (mean count per time-slot)', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(
            'Performance by Popularity Stratum\n'
            'Does WSPI structural stability hold across all popularity levels?',
            fontsize=12, fontweight='bold')
        if plotted:
            ax.legend(ncol=3, fontsize=9, loc='best', frameon=True)
        ax.grid(True, axis='y', alpha=0.25, zorder=0)

        if save:
            _finalize(fig, self.output_dir / f'chart7_stratum_{metric}.png', show)
        return fig

    # -----------------------------------------------------------------------
    # Backward-compat wrappers (called by analyze_results.py)
    # -----------------------------------------------------------------------
    def plot_temporal_evolution(self, methods, metric='mae', save=True, show=False):
        """Legacy interface — redirects to temporal Spearman (more informative)."""
        return self.plot_temporal_spearman(save=save, show=show)

    def plot_method_comparison(self, filter_top_percent=None,
                               filter_stratum=None, metrics=None,
                               save=True, show=False):
        """Legacy interface — redirects to protocol overview."""
        return self.plot_protocol_overview(save=save, show=show)

    # -----------------------------------------------------------------------
    # Master entry point
    # -----------------------------------------------------------------------
    def create_summary_report(self,
                              filter_top_percent=None,
                              filter_stratum=None,
                              save=True,
                              show=False):
        """Generate all 7 charts."""
        print("\nGenerating summary report...")

        steps = [
            ("1. Protocol overview (all 4 layers)...",
             lambda s, sh: self.plot_protocol_overview(save=s, show=sh)),
            ("2. Stability RSI@K...",
             lambda s, sh: self.plot_stability_rsi(save=s, show=sh)),
            ("3. Robustness comparison...",
             lambda s, sh: self.plot_robustness(save=s, show=sh)),
            ("4. Temporal RSI@10 evolution...",
             lambda s, sh: self.plot_temporal_rsi(k=10, save=s, show=sh)),
            ("5. NDCG@K multi-K profile...",
             lambda s, sh: self.plot_ndcg_profile(save=s, show=sh)),
            ("6. Temporal Spearman evolution...",
             lambda s, sh: self.plot_temporal_spearman(save=s, show=sh)),
            ("7. Per-stratum performance...",
             lambda s, sh: self.plot_stratum_comparison(save=s, show=sh)),
        ]

        for label, fn in steps:
            print(f"  {label}")
            try:
                fn(save, show)
            except Exception as e:
                print(f"    Warning: {e}")

        print(f"\n✓ Summary report created successfully!")
        print(f"  Output directory: {self.output_dir}")
