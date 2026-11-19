r"""
Figures and tables of the popularity-prediction study, from CSV files only
=====================================================================
Reads (under ``--root``, default ``results/prediction``):
  stats/all_method_summary.csv            means and 95 % CIs (run_prediction_stats.py)
  decision/pareto_cells.csv, gain_claims.csv, domain_effect.csv, bound_check.csv
                                          (prediction_decision.py)
  prediction_numbers.csv                      (tools/prediction_numbers.py)
and writes into ``--out`` (default ``paper_figures/prediction``; never into results/):

  fig_horizon_sweep_<metric>.pdf/.png     mean and CI against the horizon, six scenarios
  fig_where_prediction_helps_<method>.pdf/.png
                                          scenario x horizon map per main metric: verdict of
                                          the method against the best traditional baseline
  fig_ablation_steps_h<h>.pdf/.png        design steps WSPI -> WSPI-F -> WSPI-F-YW -> WSPI-FT
                                          -> WSPI-F2 -> WSPI-FG, four metrics, six scenarios
  fig_gate_share_zero.pdf/.png            share of items where WSPI-FG reduces to WSPI
  tab_main_<metric>.tex                   methods x scenarios, one block per horizon
  tab_pareto.tex, tab_domain.tex, tab_bounds.tex
  figures_manifest.json                   every file with the CSV files it was made from

With ``--manuscript`` it also writes the main-text items of the paper in Markdown
(the files above are unchanged):
  fig_horizon_sweep_combined.pdf/.png     four metrics x six scenarios in one figure
  fig_where_prediction_helps_combined.pdf/.png
                                          three methods x four metrics in one figure
  tab_pareto.md, tab_summary.md, tab_domain.md
                                          Markdown tables included in the paper with
                                          <!-- include: ../figures/tab_*.md -->
  (tab_summary.md also reads stats/all_paired_tests.csv)

With ``--si`` it also writes the Markdown tables of the Supplementary Information
(the files above are unchanged):
  tab_si_selection.md                     gate selection on synthetic data
                                          (gate_selection/selection_summary.csv)
  tab_si_windows.md                       evaluation windows, block length and blocks
                                          (stats/all_paired_tests.csv)
  tab_si_bounds.md                        extremes of P_F / P_WSPI (decision/bound_check.csv)
  tab_si_gate.md                          gate profile of WSPI-F2 and WSPI-FG (prediction_numbers.csv)
  tab_si_main_<scenario>.md               mean and 95 % CI of the four main metrics
  tab_si_secondary_<scenario>.md          means of the other protocol columns
                                          (both from stats/all_method_summary.csv)

With ``--test-part`` it writes only the items of the test part of the supplementary
run C (no other file is read or written; manifest figures_manifest_test.json):
  fig_test_tradeoff_<x>_<y>_h<h>.pdf/.png  mean of two main metrics per method, six
                                          scenarios (Delta-Rank / NDCG@10 and RSI@10 / rho)
  fig_test_sensitivity.pdf/.png           range of the 18 WSPI-F2 configurations
  fig_nlms_convergence.pdf/.png           one-step error of NLMS and AR(2) along t
  tab_test_gain.md, tab_test_best_baseline.md
                                          rule 2 on the test part, both baseline sets
  tab_test_ranks.md                       median rank of every method over the 18 cells
  tab_test_gbm.md, tab_test_runtime.md    learned baseline: verdicts and time ratios
  tab_test_smoothing.md                   smoothing control WSPI-S
  tab_test_sensitivity.md                 sensitivity of WSPI-F2
  tab_test_nlms.md                        NLMS convergence, late positions
  tab_test_explore_counts.md, tab_test_explore_ma64.md
                                          exploratory test with reference WSPI-F-YW
                                          (only if decision_explore/ exists)
  (reads decision_test/*.csv, decision_explore/*.csv and run_C_nlms/*/h1/nlms_convergence.csv)

With ``--paper-test`` it writes only the items of the rewritten Results on the test part
(no other file is read or written; manifest figures_manifest_paper_test.json):
  tab_test_dominance.md                   cells where >= 1 of the 11 baselines A+X2 dominates
                                          each method, per horizon (main text)
  tab_test_positive.md                    WSPI against the moving averages of the same and of
                                          longer memory; GBM+W against GBM
  tab_test_pareto_gates.md                Pareto rule WSPI-FG against WSPI-F2
  tab_test_domain.md                      WSPI-FA against WSPI and ARIMA
  fig_test_design_path_h<h>.pdf/.png      design steps on the test part, h = 1, 8, 24 (h = 1 in
                                          the main text, 8 and 24 in the SI)
  tab_si_test_dominance_baselines.md      relation counts of every method to each baseline
  tab_si_test_pareto_fs3.md               Pareto rule WSPI-FS3 against WSPI-F2
  tab_si_test_main_<scenario>.md          mean and 95 % CI of the four main metrics on the
                                          test part, every method
  (reads decision_paper/*.csv of prediction_test_analysis.py --paper and
  decision_test/means_test.csv)

With ``--si-test`` it writes only the tables of the Supplementary Information on the test
part (no other file is read or written; manifest figures_manifest_si_test.json):
  tab_sit_data.md                         slot length, slots, items and item threshold
  tab_sit_parts.md                        tuning and test windows, block length and blocks
  tab_sit_bounds.md                       extremes of the score ratio of every clamped method
  tab_sit_gate.md                         gate profile on the test windows (window statistics)
  tab_sit_kappa.md                        selection of kappa on the tuning part
  tab_sit_main_<scenario>.md, tab_sit_secondary_<scenario>.md
                                          every method on the test part, development and
                                          confirmation scenarios
  tab_sit_dominance_baselines.md          relation of each method to each baseline
  tab_sit_confirm_pairs.md, tab_sit_confirm_parts.md
                                          descriptive pairs and both parts of the confirmation data
  tab_sit_runtime.md                      time ratios, labelled without evaluation groups

No number is typed here; every value comes from the CSV files.  Missing
scenarios are left empty and listed (``--strict`` makes that an error).

    python tools\prediction_figures.py --root results\prediction ^
        --out paper_figures\prediction
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.colors import ListedColormap                      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POINTS = False        # --points: dots with error bars and labelled log ticks (set in main)

SCENARIOS = ('youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min',
             'movielens_daily', 'movielens_weekly')
SCEN_LABEL = {'youtube_hourly': 'YouTube (1 h)', 'taxi_hourly': 'Taxi (1 h)',
              'taxi_30min': 'Taxi (30 min)', 'taxi_5min': 'Taxi (5 min)',
              'movielens_daily': 'MovieLens (1 d)', 'movielens_weekly': 'MovieLens (1 w)'}
HORIZONS = (1, 8, 24)
METRICS = ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion')
METRIC_LABEL = {'ndcg@10': 'NDCG@10', 'spearman_rho': r'Spearman $\rho$', 'rsi@10': 'RSI@10',
                'robustness_distortion': r'$\Delta$Rank (lower is better)'}
METRIC_TEX = {'ndcg@10': 'NDCG@10', 'spearman_rho': r'Spearman $\rho$', 'rsi@10': 'RSI@10',
              'robustness_distortion': r'$\Delta$Rank'}
LOWER = ('robustness_distortion',)
RUN_A = ('WSPI', 'WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
         'Persistence', 'Holt', 'ARYW', 'MA64')
SWEEP = ('WSPI-FG', 'WSPI-F2', 'WSPI', 'ARYW', 'MA64', 'Holt', 'Persistence')
STEPS = ('WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-FT', 'WSPI-F2', 'WSPI-FG')
# Okabe-Ito colours; line style and marker repeat the distinction
STYLE = {'WSPI-FG': ('#D55E00', '-', 'o'), 'WSPI-F2': ('#0072B2', '-', 's'),
         'WSPI': ('#009E73', '-', '^'), 'ARYW': ('#555555', '--', 'v'),
         'MA64': ('#999999', '--', 'D'), 'Holt': ('#CC79A7', ':', 'x'),
         'Persistence': ('#E69F00', ':', '+'), 'WSPI-F': ('#CC79A7', '-', 'x'),
         'WSPI-F-YW': ('#56B4E9', '-', 'd'), 'WSPI-FT': ('#E69F00', '-', 'P')}
VERDICT_CODE = {'ref_worse': 0, 'n.s.': 1, 'ref_better': 2}
VERDICT_CMAP = ListedColormap(['#D55E00', '#EEEEEE', '#0072B2'])
MANIFEST = []


def save(fig, out: Path, name: str, sources):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    MANIFEST.append(dict(file=name, sources=sources))


def summary_a(summ):
    d = summ[(summ['run'] == 'A') & (summ['reference_report'] == 'WSPI')]
    return d.set_index(['scenario', 'horizon', 'metric', 'method'])


def fig_sweep(sa, out, metric):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), sharex=True)
    x = np.arange(len(HORIZONS))
    missing = []
    for ax, s in zip(axes.ravel(), SCENARIOS):
        drawn = False
        for i, m in enumerate(SWEEP):
            mean, lo, hi = [], [], []
            for h in HORIZONS:
                key = (s, h, metric, m)
                if key in sa.index:
                    r = sa.loc[key]
                    mean.append(r['mean']); lo.append(r['ci_low']); hi.append(r['ci_high'])
                else:
                    mean.append(np.nan); lo.append(np.nan); hi.append(np.nan)
            mean, lo, hi = map(np.asarray, (mean, lo, hi))
            if np.isfinite(mean).any():
                c, ls, mk = STYLE[m]
                off = (i - (len(SWEEP) - 1) / 2) * 0.035
                ax.errorbar(x + off, mean, yerr=[np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)], color=c, ls=ls, marker=mk,
                            ms=4.5, lw=1.2, capsize=2, label=m)
                drawn = True
        if not drawn:
            missing.append(s)
            ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(SCEN_LABEL[s], fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in HORIZONS])
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel('horizon $h$ (slots)')
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL[metric])
    hs, ls = [], []
    for ax in axes.ravel():
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_); ls.append(l_)
    if hs:
        fig.legend(hs, ls, loc='lower center', ncol=len(ls), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, f"fig_horizon_sweep_{metric.replace('@', '').replace('_', '-')}",
         ['stats/all_method_summary.csv'])
    return missing


def fig_map(claims, out, method):
    d = claims[claims['method'] == method].set_index(['scenario', 'horizon', 'metric'])
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.6), sharey=True)
    for ax, c in zip(axes, METRICS):
        grid = np.full((len(SCENARIOS), len(HORIZONS)), np.nan)
        for i, s in enumerate(SCENARIOS):
            for j, h in enumerate(HORIZONS):
                if (s, h, c) in d.index:
                    r = d.loc[(s, h, c)]
                    grid[i, j] = VERDICT_CODE[r['verdict']]
                    ax.text(j, i, f"{r['best_baseline']}\n{r['diff_method_minus_baseline']:+.3g}",
                            ha='center', va='center', fontsize=6.5,
                            color='white' if r['verdict'] != 'n.s.' else 'black')
        ax.imshow(np.ma.masked_invalid(grid), cmap=VERDICT_CMAP, vmin=0, vmax=2, aspect='auto')
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f'$h$={h}' for h in HORIZONS])
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels([SCEN_LABEL[s] for s in SCENARIOS], fontsize=8)
        ax.set_title(METRIC_TEX[c], fontsize=10)
        ax.set_xticks(np.arange(-.5, len(HORIZONS)), minor=True)
        ax.set_yticks(np.arange(-.5, len(SCENARIOS)), minor=True)
        ax.grid(which='minor', color='white', lw=1.5)
        ax.tick_params(which='minor', length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_CMAP(k)) for k in (2, 1, 0)]
    fig.legend(handles, [f'{method} better', 'not significant', f'{method} worse'],
               loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f'{method} against the best traditional baseline '
                 '(cell: baseline, mean difference)', fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save(fig, out, f'fig_where_prediction_helps_{method}', ['decision/gain_claims.csv'])


def fig_steps(sa, out, h):
    fig, axes = plt.subplots(len(METRICS), len(SCENARIOS), figsize=(13.5, 8.0), sharex=True)
    x = np.arange(len(STEPS))
    for i, c in enumerate(METRICS):
        for j, s in enumerate(SCENARIOS):
            ax = axes[i, j]
            vals = [(sa.loc[(s, h, c, m)] if (s, h, c, m) in sa.index else None) for m in STEPS]
            if any(v is not None for v in vals):
                mean = np.array([v['mean'] if v is not None else np.nan for v in vals])
                lo = np.array([v['ci_low'] if v is not None else np.nan for v in vals])
                hi = np.array([v['ci_high'] if v is not None else np.nan for v in vals])
                ax.bar(x, mean, color=[STYLE[m][0] for m in STEPS], width=0.72)
                ax.errorbar(x, mean, yerr=[np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)], fmt='none', ecolor='black',
                            lw=0.8, capsize=2)
                if np.isfinite(mean[0]):
                    ax.axhline(mean[0], color=STYLE['WSPI'][0], lw=0.8, ls='--')
                fin = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
                if c not in LOWER and len(fin):
                    pad = 0.08 * max(fin.max() - fin.min(), 1e-6)
                    ax.set_ylim(fin.min() - pad, fin.max() + pad)
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            if i == 0:
                ax.set_title(SCEN_LABEL[s], fontsize=9)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[c], fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace('WSPI-', '') for m in STEPS], rotation=60, fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='y', alpha=0.25, lw=0.5)
    fig.suptitle(f'Design steps at horizon $h$ = {h} (dashed line: WSPI; bars: mean, 95% CI)',
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, f'fig_ablation_steps_h{h}', ['stats/all_method_summary.csv'])


def fig_gate(numbers, out):
    d = numbers[(numbers['quantity'] == 'gate_fg') & (numbers['statistic'] == 'mean_share_zero')]
    if d.empty:
        return False
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    w = 0.26
    for k, h in enumerate(HORIZONS):
        v = [d[(d['scenario'] == s) & (d['horizon'] == h)]['value'].mean() for s in SCENARIOS]
        ax.bar(np.arange(len(SCENARIOS)) + (k - 1) * w, v, w, label=f'$h$ = {h}',
               color=['#56B4E9', '#0072B2', '#004466'][k])
    ax.set_xticks(range(len(SCENARIOS)))
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCENARIOS], rotation=20, fontsize=8)
    ax.set_ylabel('share of items with $g = 0$\n(WSPI-FG equals WSPI)')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out, 'fig_gate_share_zero', ['prediction_numbers.csv'])
    return True


# ---- tables -----------------------------------------------------------------

def fmt(v, metric):
    if not np.isfinite(v):
        return '--'
    return f'{v:.2f}' if metric in LOWER else f'{v:.4f}'


def tab_main(sa, out, metric):
    lines = [r'\begin{tabular}{ll' + 'r' * len(SCENARIOS) + '}', r'\toprule',
             '$h$ & Method & ' + ' & '.join(SCEN_LABEL[s] for s in SCENARIOS) + r' \\', r'\midrule']
    for h in HORIZONS:
        best = {}
        for s in SCENARIOS:
            v = [sa.loc[(s, h, metric, m)]['mean'] for m in RUN_A if (s, h, metric, m) in sa.index]
            if v:
                best[s] = min(v) if metric in LOWER else max(v)
        for k, m in enumerate(RUN_A):
            cells = []
            for s in SCENARIOS:
                if (s, h, metric, m) in sa.index:
                    val = float(sa.loc[(s, h, metric, m)]['mean'])
                    t = fmt(val, metric)
                    cells.append(r'\textbf{' + t + '}' if val == best.get(s) else t)
                else:
                    cells.append('--')
            lines.append((str(h) if k == 0 else '') + f' & {m} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\midrule' if h != HORIZONS[-1] else r'\bottomrule')
    lines.append(r'\end{tabular}')
    name = f"tab_main_{metric.replace('@', '').replace('_', '-')}.tex"
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=['stats/all_method_summary.csv']))


SYMBOL = {'ref_better': r'$\uparrow$', 'ref_worse': r'$\downarrow$', 'n.s.': r'$\circ$',
          'missing': '--'}


def tab_pareto(cells, out):
    lines = [r'\begin{tabular}{ll' + 'c' * len(METRICS) + 'c}', r'\toprule',
             '$h$ & Scenario & ' + ' & '.join(METRIC_TEX[c] for c in METRICS) + r' & Success \\',
             r'\midrule']
    for h in HORIZONS:
        d = cells[cells['horizon'] == h].set_index('scenario')
        for k, s in enumerate(SCENARIOS):
            if s not in d.index:
                continue
            r = d.loc[s]
            lines.append((str(h) if k == 0 else '') + f' & {SCEN_LABEL[s]} & '
                         + ' & '.join(SYMBOL[r[f'verdict_{c}']] for c in METRICS)
                         + ' & ' + ('yes' if bool(r['success']) else 'no') + r' \\')
        lines.append(r'\midrule' if h != HORIZONS[-1] else r'\bottomrule')
    lines.append(r'\end{tabular}')
    (out / 'tab_pareto.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file='tab_pareto.tex', sources=['decision/pareto_cells.csv']))


def tab_domain(domain, out):
    lines = [r'\begin{tabular}{llrrrcrr}', r'\toprule',
             r'Scenario & $h$ & Metric & WSPI-FA & ARIMA & Verdict & Time ratio & Fallback share \\',
             r'\midrule']
    for _, r in domain.iterrows():
        lines.append(f"{SCEN_LABEL[r['scenario']]} & {int(r['horizon'])} & {METRIC_TEX[r['metric']]} & "
                     f"{fmt(r['mean_wspi_fa'], r['metric'])} & {fmt(r['mean_arima'], r['metric'])} & "
                     f"{SYMBOL[r['verdict']]} & {r['time_ratio_fa_to_arima']:.2f} & "
                     f"{r['fallback_share']:.4f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_domain.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file='tab_domain.tex', sources=['decision/domain_effect.csv']))


def tab_bounds(bounds, out):
    lines = [r'\begin{tabular}{rlrrrr}', r'\toprule',
             r'$h$ & Method & $\bar L$ & min ratio & max ratio & $\bar U$ \\', r'\midrule']
    g = bounds.groupby(['horizon', 'method'], sort=True).agg(
        lower=('lower_bound', 'first'), mn=('min_ratio', 'min'), mx=('max_ratio', 'max'),
        upper=('upper_bound', 'first')).reset_index()
    for _, r in g.iterrows():
        lines.append(f"{int(r['horizon'])} & {r['method']} & {r['lower']:.4f} & {r['mn']:.4f} & "
                     f"{r['mx']:.4f} & {r['upper']:.4f}" + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_bounds.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file='tab_bounds.tex', sources=['decision/bound_check.csv']))


# ---- main-text items of the manuscript (--manuscript) -----------------------------------

MAP_METHODS = ('WSPI', 'WSPI-F2', 'WSPI-FG')
BASELINES = ('Persistence', 'Holt', 'ARYW', 'MA64')
METRIC_MD = {'ndcg@10': 'NDCG@10', 'spearman_rho': 'Spearman ρ', 'rsi@10': 'RSI@10',
             'robustness_distortion': 'ΔRank'}
SYMBOL_MD = {'ref_better': '↑', 'ref_worse': '↓', 'n.s.': '○', 'missing': '–'}
MD_NOTE = ('<!-- Generated by tools/prediction_figures.py --manuscript from {src}. '
           'Do not edit by hand; rebuild instead. -->')


def _legend_below(fig, axes, ncol=None):
    hs, ls = [], []
    for ax in np.ravel(axes):
        for h_, l_ in zip(*ax.get_legend_handles_labels()):
            if l_ not in ls:
                hs.append(h_); ls.append(l_)
    if hs:
        fig.legend(hs, ls, loc='lower center', ncol=ncol or len(ls), frameon=False, fontsize=9,
                   bbox_to_anchor=(0.5, 0.0))


def fig_sweep_combined(sa, out):
    """Combined figure: the four main metrics (rows) against the horizon,
    one column per scenario; same data and style as fig_sweep."""
    fig, axes = plt.subplots(len(METRICS), len(SCENARIOS), figsize=(13.5, 9.2), sharex=True)
    x = np.arange(len(HORIZONS))
    missing = []
    for i, metric in enumerate(METRICS):
        for j, s in enumerate(SCENARIOS):
            ax = axes[i, j]
            drawn = False
            for k, m in enumerate(SWEEP):
                vals = [sa.loc[(s, h, metric, m)] if (s, h, metric, m) in sa.index else None
                        for h in HORIZONS]
                mean = np.array([v['mean'] if v is not None else np.nan for v in vals])
                lo = np.array([v['ci_low'] if v is not None else np.nan for v in vals])
                hi = np.array([v['ci_high'] if v is not None else np.nan for v in vals])
                if np.isfinite(mean).any():
                    c, ls, mk = STYLE[m]
                    off = (k - (len(SWEEP) - 1) / 2) * 0.035
                    ax.errorbar(x + off, mean, yerr=[np.clip(mean - lo, 0, None),
                                                     np.clip(hi - mean, 0, None)],
                                color=c, ls=ls, marker=mk, ms=3.5, lw=1.0, capsize=1.5, label=m)
                    drawn = True
            if not drawn:
                missing.append(s)
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            if i == 0:
                ax.set_title(SCEN_LABEL[s], fontsize=9)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[metric], fontsize=8.5)
            if metric in LOWER:
                ax.set_yscale('log')
            ax.set_xticks(x)
            ax.set_xticklabels([str(h) for h in HORIZONS])
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[-1]:
        ax.set_xlabel('horizon $h$ (slots)', fontsize=8)
    _legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    save(fig, out, 'fig_horizon_sweep_combined', ['stats/all_method_summary.csv'])
    return missing


def fig_map_combined(claims, out):
    """Combined figure: verdict of WSPI, WSPI-F2 and WSPI-FG (rows) against the best
    traditional baseline, per main metric (columns); same data and colours as fig_map."""
    fig, axes = plt.subplots(len(MAP_METHODS), len(METRICS), figsize=(12.5, 9.6),
                             sharex=True, sharey=True)
    for r, method in enumerate(MAP_METHODS):
        d = claims[claims['method'] == method].set_index(['scenario', 'horizon', 'metric'])
        for col, c in enumerate(METRICS):
            ax = axes[r, col]
            grid = np.full((len(SCENARIOS), len(HORIZONS)), np.nan)
            for i, s in enumerate(SCENARIOS):
                for j, h in enumerate(HORIZONS):
                    if (s, h, c) in d.index:
                        row = d.loc[(s, h, c)]
                        grid[i, j] = VERDICT_CODE[row['verdict']]
                        ax.text(j, i, f"{row['best_baseline']}\n{row['diff_method_minus_baseline']:+.3g}",
                                ha='center', va='center', fontsize=6,
                                color='white' if row['verdict'] != 'n.s.' else 'black')
            ax.imshow(np.ma.masked_invalid(grid), cmap=VERDICT_CMAP, vmin=0, vmax=2, aspect='auto')
            ax.set_xticks(range(len(HORIZONS)))
            ax.set_xticklabels([f'$h$={h}' for h in HORIZONS], fontsize=8)
            ax.set_yticks(range(len(SCENARIOS)))
            ax.set_yticklabels([SCEN_LABEL[s] for s in SCENARIOS], fontsize=7.5)
            if r == 0:
                ax.set_title(METRIC_TEX[c], fontsize=10)
            if col == 0:
                ax.set_ylabel(method, fontsize=10, fontweight='bold')
            ax.set_xticks(np.arange(-.5, len(HORIZONS)), minor=True)
            ax.set_yticks(np.arange(-.5, len(SCENARIOS)), minor=True)
            ax.grid(which='minor', color='white', lw=1.5)
            ax.tick_params(which='minor', length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VERDICT_CMAP(k)) for k in (2, 1, 0)]
    fig.legend(handles, ['method better', 'not significant', 'method worse'],
               loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, 'fig_where_prediction_helps_combined', ['decision/gain_claims.csv'])


def _md_table(header, rows, align):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(align) + '|']
    out += ['| ' + ' | '.join(r) + ' |' for r in rows]
    return out


def md_pareto(cells, out):
    rows = []
    for h in HORIZONS:
        d = cells[cells['horizon'] == h].set_index('scenario')
        for k, s in enumerate(SCENARIOS):
            if s not in d.index:
                continue
            r = d.loc[s]
            rows.append([str(h) if k == 0 else '', SCEN_LABEL[s]]
                        + [SYMBOL_MD[r[f'verdict_{c}']] for c in METRICS]
                        + ['yes' if bool(r['success']) else 'no'])
    lines = [MD_NOTE.format(src='decision/pareto_cells.csv'), '']
    lines += _md_table(['$h$', 'Scenario'] + [METRIC_MD[c] for c in METRICS] + ['Success'], rows,
                       [':--', ':--'] + [':-:'] * len(METRICS) + [':-:'])
    (out / 'tab_pareto.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file='tab_pareto.md', sources=['decision/pareto_cells.csv']))


def md_summary(tests, out):
    """Counts of cells (of 18: six scenarios x three horizons) in which each of our methods is
    better / not significantly different / worse than each traditional baseline (run A,
    column verdict)."""
    t = tests[(tests['run'] == 'A') & tests['metric'].isin(METRICS)
              & tests['reference'].isin(MAP_METHODS) & tests['method'].isin(BASELINES)]
    rows = []
    for m in MAP_METHODS:
        for k, b in enumerate(BASELINES):
            cells = [m if k == 0 else '', b]
            for c in METRICS:
                v = t[(t['reference'] == m) & (t['method'] == b) & (t['metric'] == c)]['verdict']
                n = len(v)
                cells.append(f"{(v == 'ref_better').sum()} / {(v == 'n.s.').sum()} / "
                             f"{(v == 'ref_worse').sum()}" + ('' if n == len(SCENARIOS) * len(HORIZONS)
                                                              else f' (n={n})'))
            rows.append(cells)
    lines = [MD_NOTE.format(src='stats/all_paired_tests.csv (run A, column verdict)'), '']
    lines += _md_table(['Method', 'Baseline'] + [METRIC_MD[c] for c in METRICS], rows,
                       [':--', ':--'] + [':-:'] * len(METRICS))
    (out / 'tab_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file='tab_summary.md', sources=['stats/all_paired_tests.csv']))


def md_domain(domain, out):
    rows = []
    for s in SCENARIOS:
        for h in HORIZONS:
            d = domain[(domain['scenario'] == s) & (domain['horizon'] == h)].set_index('metric')
            if d.empty:
                continue
            first = d.iloc[0]
            rows.append([SCEN_LABEL[s], str(h)]
                        + [SYMBOL_MD[d.loc[c, 'verdict']] if c in d.index else '–' for c in METRICS]
                        + [f"{first['time_ratio_fa_to_arima']:.2f}", f"{first['fallback_share']:.4f}"])
    lines = [MD_NOTE.format(src='decision/domain_effect.csv'), '']
    lines += _md_table(['Scenario', '$h$'] + [METRIC_MD[c] for c in METRICS]
                       + ['Time ratio', 'Fallback share'], rows,
                       [':--', ':-:'] + [':-:'] * len(METRICS) + ['--:', '--:'])
    (out / 'tab_domain.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file='tab_domain.md', sources=['decision/domain_effect.csv']))


# ---- Supplementary Information (--si) ----------------------------------------------------

SI_NOTE = ('<!-- Generated by tools/prediction_figures.py --si from {src}. '
           'Do not edit by hand; rebuild instead. -->')
SI_METHODS = ('WSPI', 'WSPI-F2', 'WSPI-FG', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT', 'WSPI-FA',
              'Persistence', 'Holt', 'ARYW', 'MA64', 'ARIMA')
RUN_B_ONLY = ('WSPI-FA', 'ARIMA')
SECONDARY = ('ndcg@5', 'ndcg@20', 'kendall_tau', 'rsi@5', 'rsi@20',
             'coverage@5', 'coverage@10', 'coverage@20', 'mae')
SECONDARY_MD = {'ndcg@5': 'NDCG@5', 'ndcg@20': 'NDCG@20', 'kendall_tau': 'Kendall τ',
                'rsi@5': 'RSI@5', 'rsi@20': 'RSI@20', 'coverage@5': 'Cov@5',
                'coverage@10': 'Cov@10', 'coverage@20': 'Cov@20', 'mae': 'MAE'}
SI_DIGITS = {'robustness_distortion': 1, 'mae': 1}   # one decimal keeps the CI cells on one line


def _si_write(out, name, src, header, rows, align):
    lines = [SI_NOTE.format(src=src), '']
    lines += _md_table(header, rows, align)
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=[s.strip() for s in src.split(';')]))


def _si_num(v, metric):
    if v is None or not np.isfinite(v):
        return '–'
    d = SI_DIGITS.get(metric, 4)
    if metric == 'mae':                      # raw-score MAE spans many orders of magnitude
        return f'{v:,.{d}f}' if abs(v) < 1e5 else f'{v:.2e}'
    return f'{v:.{d}f}'


def _range(lo, hi):
    lo, hi = int(lo), int(hi)
    return f'{lo:,}' if lo == hi else f'{lo:,}–{hi:,}'


def si_selection(sel, out):
    rows = []
    for _, r in sel.sort_values('cost_rank').iterrows():
        rows.append([r['config'], r['config'][0], r['config'][1:], f"{r['mean_spearman']:.3f}",
                     f"{r['min_cell']:.3f}", f"{r['max_cell']:.3f}", f"{r['gap_to_best']:.4f}",
                     'yes' if bool(r['chosen']) else 'no'])
    _si_write(out, 'tab_si_selection.md', 'gate_selection/selection_summary.csv',
              ['Configuration', 'Source', '$k$', 'Mean ρ', 'Lowest cell', 'Highest cell',
               'Gap to best', 'Chosen'], rows,
              [':--', ':-:', ':-:', '--:', '--:', '--:', '--:', ':-:'])


def si_windows(tests, out):
    t = tests[tests['metric'].isin(METRICS)]
    g = t.groupby(['run', 'scenario', 'horizon']).agg(
        w_lo=('n_windows', 'min'), w_hi=('n_windows', 'max'), b_lo=('block', 'min'),
        b_hi=('block', 'max'), n_lo=('n_blocks', 'min'), n_hi=('n_blocks', 'max'))
    rows = []
    for run in ('A', 'B'):
        for s in SCENARIOS:
            for h in HORIZONS:
                if (run, s, h) not in g.index:
                    continue
                r = g.loc[(run, s, h)]
                rows.append([run, SCEN_LABEL[s], str(h), _range(r['w_lo'], r['w_hi']),
                             _range(r['b_lo'], r['b_hi']), _range(r['n_lo'], r['n_hi'])])
    _si_write(out, 'tab_si_windows.md', 'stats/all_paired_tests.csv (four main metrics)',
              ['Run', 'Scenario', '$h$', 'Windows', 'Block length $b_h$', 'Blocks'], rows,
              [':-:', ':--', ':-:', '--:', '--:', '--:'])


def si_bounds(bounds, out):
    g = bounds.groupby(['horizon', 'method'], sort=False).agg(
        m=('m', 'first'), lower=('lower_bound', 'first'), mn=('min_ratio', 'min'),
        mx=('max_ratio', 'max'), upper=('upper_bound', 'first'), n=('n_ratios', 'sum'),
        out_=('n_below', 'sum'), out2=('n_above', 'sum'), nf=('n_nonfinite', 'sum'),
        nsc=('scenario', 'nunique')).reset_index()
    order = {m: i for i, m in enumerate(SI_METHODS)}
    g = g.sort_values(['horizon', 'method'], key=lambda c: c.map(order) if c.name == 'method' else c)
    rows = []
    for _, r in g.iterrows():
        rows.append([str(int(r['horizon'])), str(int(r['m'])), r['method'], str(int(r['nsc'])),
                     f"{int(r['n']):,}", f"{r['lower']:.6f}", f"{r['mn']:.6f}", f"{r['mx']:.4f}",
                     f"{r['upper']:.4f}", str(int(r['out_'] + r['out2'] + r['nf']))])
    _si_write(out, 'tab_si_bounds.md', 'decision/bound_check.csv',
              ['$h$', '$m$', 'Method', 'Scenarios', 'Ratios', '$\\bar L$', 'Min ratio', 'Max ratio',
               '$\\bar U$', 'Outside'], rows,
              [':-:', ':-:', ':--', ':-:', '--:', '--:', '--:', '--:', '--:', '--:'])


def si_gate(numbers, out):
    def val(q, st, s, h):
        d = numbers[(numbers['quantity'] == q) & (numbers['statistic'] == st)
                    & (numbers['scenario'] == s) & (numbers['horizon'] == h)]['value']
        return float(d.iloc[0]) if len(d) else np.nan
    rows = []
    for s in SCENARIOS:
        for h in HORIZONS:
            rows.append([SCEN_LABEL[s], str(h)]
                        + [f'{v:.3f}' if np.isfinite(v) else '–' for v in (
                            val('gate_f2', 'pooled_median', s, h),
                            val('gate_fg', 'pooled_median', s, h),
                            val('gate_fg', 'median_of_window_median', s, h),
                            val('gate_fg', 'mean_share_zero', s, h))])
    _si_write(out, 'tab_si_gate.md', 'prediction_numbers.csv (gate_f2, gate_fg)',
              ['Scenario', '$h$', 'F2: median $g$', 'FG: median $g$',
               'FG: median of window medians', 'FG: share $g=0$'], rows,
              [':--', ':-:', '--:', '--:', '--:', '--:'])


def _si_summary(summ):
    a = summ[(summ['run'] == 'A') & (summ['reference_report'] == 'WSPI')]
    b = summ[(summ['run'] == 'B') & (summ['reference_report'] == 'WSPI-FA')
             & summ['method'].isin(RUN_B_ONLY)]
    return pd.concat([a, b]).set_index(['scenario', 'horizon', 'metric', 'method'])


def si_main(summ, out):
    d = _si_summary(summ)
    for s in SCENARIOS:
        rows = []
        for h in HORIZONS:
            ms = [m for m in SI_METHODS if (s, h, METRICS[0], m) in d.index]
            best = {}
            for c in METRICS:
                v = [float(d.loc[(s, h, c, m)]['mean']) for m in ms if (s, h, c, m) in d.index]
                if v:
                    best[c] = min(v) if c in LOWER else max(v)
            for k, m in enumerate(ms):
                cells = [str(h) if k == 0 else '', m]
                for c in METRICS:
                    if (s, h, c, m) not in d.index:
                        cells.append('–')
                        continue
                    r = d.loc[(s, h, c, m)]
                    t = _si_num(float(r['mean']), c)
                    t = f'**{t}**' if float(r['mean']) == best.get(c) else t
                    cells.append(f"{t} ({_si_num(float(r['ci_low']), c)}–{_si_num(float(r['ci_high']), c)})")
                rows.append(cells)
        _si_write(out, f'tab_si_main_{s}.md',
                  'stats/all_method_summary.csv (run A, reference report WSPI; WSPI-FA and ARIMA '
                  'from run B, reference report WSPI-FA)',
                  ['$h$', 'Method'] + [METRIC_MD[c] for c in METRICS], rows,
                  [':-:', ':--'] + ['--:'] * len(METRICS))


def si_secondary(summ, out):
    d = _si_summary(summ)
    for s in SCENARIOS:
        rows = []
        for h in HORIZONS:
            ms = [m for m in SI_METHODS if (s, h, SECONDARY[0], m) in d.index]
            for k, m in enumerate(ms):
                rows.append([str(h) if k == 0 else '', m]
                            + [_si_num(float(d.loc[(s, h, c, m)]['mean']), c)
                               if (s, h, c, m) in d.index else '–' for c in SECONDARY])
        _si_write(out, f'tab_si_secondary_{s}.md',
                  'stats/all_method_summary.csv (run A, reference report WSPI; WSPI-FA and ARIMA '
                  'from run B, reference report WSPI-FA)',
                  ['$h$', 'Method'] + [SECONDARY_MD[c] for c in SECONDARY], rows,
                  [':-:', ':--'] + ['--:'] * len(SECONDARY))


# ---- test part of the supplementary run C (--test-part) ------------------------

TEST_NOTE = ('<!-- Generated by tools/prediction_figures.py --test-part from {src}. '
             'Do not edit by hand; rebuild instead. -->')
TEST_REFS = ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W')
SMA_LINE = ('SMA8', 'SMA16', 'SMA32', 'MA64')            # MA64 is SMA64
EWMA_LINE = ('EWMA8', 'EWMA16', 'EWMA32', 'EWMA64')
# (colour, marker, filled); Okabe-Ito colours as in STYLE
TRADE_POINTS = {'WSPI': ('#009E73', '^', True), 'WSPI-F2': ('#0072B2', 's', True),
                'WSPI-FS3': ('#0072B2', 's', False), 'WSPI-F-YW': ('#56B4E9', 'd', True),
                'WSPI-S': ('#CC79A7', 'o', False), 'ARYW': ('#555555', 'v', True),
                'Holt': ('#CC79A7', 'x', True), 'Persistence': ('#E69F00', '+', True),
                'GBM': ('#000000', 'X', True), 'GBM+W': ('#000000', 'X', False)}
TRADE_PAIRS = (('robustness_distortion', 'ndcg@10'), ('rsi@10', 'spearman_rho'))
RANK_ORDER = SMA_LINE[:3] + ('MA64',) + EWMA_LINE + ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'WSPI-F-YW',
                                                     'WSPI-S', 'WSPI-F', 'WSPI-FT', 'WSPI-FG',
                                                     'ARYW', 'Holt', 'Persistence', 'GBM', 'GBM+W')


def _test_write(out, name, src, header, rows, align, pre=()):
    lines = [TEST_NOTE.format(src=src), ''] + list(pre)
    lines += _md_table(header, rows, align)
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=[s.strip() for s in src.split(';')]))


def _counts(vc, ref, meth, metric):
    d = vc[(vc['reference'] == ref) & (vc['method'] == meth) & (vc['metric'] == metric)]
    if len(d) != 1:
        return '–'
    r = d.iloc[0]
    return f"{int(r['ref_better'])} / {int(r['n.s.'])} / {int(r['ref_worse'])}"


from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402


def fig_test_tradeoff(means, out, h, xm, ym):
    """Mean of two main metrics on the test part, one panel per scenario."""
    m = means[means['horizon'] == h].set_index(['scenario', 'metric', 'method'])['mean']
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6))
    for ax, s in zip(axes.ravel(), SCENARIOS):
        def xy(meth):
            k1, k2 = (s, xm, meth), (s, ym, meth)
            return (m[k1], m[k2]) if k1 in m.index and k2 in m.index else (np.nan, np.nan)
        for line, colour, ls, lab in ((SMA_LINE, '#999999', '--', 'SMA (8, 16, 32, 64 = MA64)'),
                                      (EWMA_LINE, '#7F4F24', ':', 'EWMA (8, 16, 32, 64)')):
            pts = np.array([xy(n) for n in line])
            ax.plot(pts[:, 0], pts[:, 1], color=colour, ls=ls, marker='.', ms=7, lw=1.1,
                    label=lab)
            for n, (x_, y_) in zip(line, pts):
                if np.isfinite(x_):
                    ax.annotate(n.replace('SMA', '').replace('EWMA', '').replace('MA', ''),
                                (x_, y_), textcoords='offset points', xytext=(3, 3),
                                fontsize=6.5, color=colour)
        for meth, (c, mk, filled) in TRADE_POINTS.items():
            x_, y_ = xy(meth)
            if not np.isfinite(x_):
                continue
            ax.scatter([x_], [y_], marker=mk, s=42 if filled else 95, label=meth,
                       zorder=3 if filled else 4,
                       color=c if (filled or mk in 'x+') else 'none',
                       edgecolors=c if mk not in 'x+' else None, linewidths=1.2)
        if xm in LOWER:
            ax.set_xscale('log')
            if POINTS:
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=20))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(SCEN_LABEL[s], fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(METRIC_LABEL[xm] + (' (log scale)' if xm in LOWER else ''), fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL[ym], fontsize=9)
    _legend_below(fig, axes, ncol=6)
    fig.suptitle(f'Test part, horizon $h$ = {h}: mean over the test windows '
                 '(better: ' + ('left' if xm in LOWER else 'right') + ' and up)', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    name = (f"fig_test_tradeoff_{xm.replace('@', '').replace('_', '-')}_"
            f"{ym.replace('@', '').replace('_', '-')}_h{h}")
    save(fig, out, name, ['decision_test/means_test.csv'])


def md_test_gain(gsum, claims, out):
    rows = []
    for bset, lab in (('A', 'Classic (4)'), ('A+X2', 'Full (11)')):
        for k, meth in enumerate(TEST_REFS):
            cells = [lab if k == 0 else '', meth]
            tot = 0
            for c in METRICS:
                d = gsum[(gsum['baseline_set'] == bset) & (gsum['method'] == meth)
                         & (gsum['metric'] == c)]
                if len(d) == 1:
                    r = d.iloc[0]
                    tot += int(r['ref_better'])
                    cells.append(f"{int(r['ref_better'])} / {int(r['n.s.'])} / "
                                 f"{int(r['ref_worse'])}")
                else:
                    cells.append('–')
            cells.append(f'{tot} of {len(SCENARIOS) * len(HORIZONS) * len(METRICS)}')
            rows.append(cells)
    _test_write(out, 'tab_test_gain.md', 'decision_test/gain_summary_test.csv',
                ['Baselines', 'Method'] + [METRIC_MD[c] for c in METRICS] + ['Gains'], rows,
                [':--', ':--'] + [':-:'] * len(METRICS) + [':-:'])
    # best traditional baseline of each cell (set A+X2; one row per cell and metric)
    d = claims[(claims['baseline_set'] == 'A+X2') & (claims['method'] == TEST_REFS[0])]
    rows = []
    for c in METRICS:
        v = d[d['metric'] == c]['best_baseline'].value_counts()
        rows.append([METRIC_MD[c], ', '.join(f'{n} ({int(k)})' for n, k in v.items())])
    _test_write(out, 'tab_test_best_baseline.md', 'decision_test/gain_claims_test.csv',
                ['Metric', 'Best traditional baseline (cells of 18)'], rows, [':--', ':--'])


def md_test_ranks(means, out):
    """Median rank of every method over the 18 cells (1 = best)."""
    rk = means.copy()
    rk['rank'] = np.nan
    for (s, h, c), d in rk.groupby(['scenario', 'horizon', 'metric']):
        rk.loc[d.index, 'rank'] = d['mean'].rank(ascending=c in LOWER, method='average')
    med = rk.groupby(['method', 'metric'])['rank'].median()
    rows = []
    for meth in RANK_ORDER:
        cells = [meth + (' (= SMA64)' if meth == 'MA64' else '')]
        for c in METRICS:
            v = med.get((meth, c), np.nan)
            cells.append('–' if not np.isfinite(v) else (f'{v:.0f}' if v == int(v) else f'{v:.1f}'))
        rows.append(cells)
    n = int(rk['method'].nunique())
    _test_write(out, 'tab_test_ranks.md', 'decision_test/means_test.csv',
                ['Method'] + [METRIC_MD[c] for c in METRICS], rows,
                [':--'] + [':-:'] * len(METRICS),
                pre=[f'<!-- Median over 18 cells of the rank among {n} methods; 1 = best. -->', ''])


def md_test_pairs(vc, out, name, pairs, src='decision_test/verdict_counts_test.csv'):
    rows = [[ref, meth] + [_counts(vc, ref, meth, c) for c in METRICS] for ref, meth in pairs]
    _test_write(out, name, src, ['Reference', 'Method'] + [METRIC_MD[c] for c in METRICS], rows,
                [':--', ':--'] + [':-:'] * len(METRICS))


def md_test_runtime(rr, out):
    q = (('WSPI-F2 / WSPI (run A)', 'wspi_f2_over_wspi_runA', 2),
         ('WSPI-FS3 / WSPI (run C)', 'wspi_fs3_over_wspi_runC', 2),
         ('GBM training (s)', 'GBM_train_seconds', 2),
         ('GBM scoring (s)', 'GBM_score_seconds', 2),
         ('GBM total / WSPI (run C)', 'GBM_total_over_wspi_runC', 2),
         ('GBM total / WSPI-F2', 'GBM_total_over_wspi_f2', 2),
         ('GBM+W total / WSPI-F2', 'GBM+W_total_over_wspi_f2', 2))
    rows = []
    for lab, col, d in q:
        x = rr[col].to_numpy(float)
        rows.append([lab] + [f'{v:,.{d}f}' for v in (np.min(x), np.median(x), np.max(x))])
    _test_write(out, 'tab_test_runtime.md', 'decision_test/runtime_ratio.csv',
                [f'Quantity ({len(rr)} cells)', 'Min', 'Median', 'Max'], rows,
                [':--', '--:', '--:', '--:'])


def md_test_sensitivity(sr, out):
    rows = []
    for c in METRICS:
        d = sr[sr['metric'] == c]
        top = d['best_config'].value_counts()
        rows.append([METRIC_MD[c], f"{d['range_rel_to_default'].median():.4f}",
                     f"{d['range_rel_to_default'].max():.4f}",
                     f'{top.index[0]} ({int(top.iloc[0])} of {len(d)})'])
    _test_write(out, 'tab_test_sensitivity.md', 'decision_test/sensitivity_range.csv',
                ['Metric', 'Median range / default', 'Max range / default',
                 'Most frequent best configuration'], rows, [':--', '--:', '--:', ':--'])


def fig_test_sensitivity(sr, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    cols = ['#56B4E9', '#0072B2', '#004466']
    for i, c in enumerate(METRICS):
        for k, h in enumerate(HORIZONS):
            v = sr[(sr['metric'] == c) & (sr['horizon'] == h)]['range_rel_to_default']
            ax.scatter(np.full(len(v), i + (k - 1) * 0.16), v, s=18, color=cols[k],
                       label=f'$h$ = {h}' if i == 0 else None, zorder=3)
    ax.set_yscale('log')
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([METRIC_TEX[c] for c in METRICS])
    ax.set_ylabel('(max - min) / default\nover 18 configurations')
    ax.grid(axis='y', alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, out, 'fig_test_sensitivity', ['decision_test/sensitivity_range.csv'])


def fig_nlms(conv, out):
    fig, axes = plt.subplots(1, len(conv), figsize=(9.0, 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    style = {'NLMS': ('#CC79A7', '-', 'o'), 'AR2-YW': ('#0072B2', '--', 's')}
    for ax, (s, d) in zip(axes, conv.items()):
        for meth, (c, ls, mk) in style.items():
            e = d[d['method'] == meth].sort_values('t')
            if e.empty:
                continue
            ax.fill_between(e['t'], e['q0.25'], e['q0.75'], color=c, alpha=0.15, lw=0)
            ax.plot(e['t'], e['q0.5'], color=c, ls=ls, marker=mk, ms=3.5, lw=1.2,
                    label=f'{meth} (median, IQR)')
        ax.set_title(SCEN_LABEL[s], fontsize=10)
        ax.set_xlabel('position $t$ in the coefficient sequence')
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r'$|\hat a_t - a_t| / \mu_L$')
    _legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, out, 'fig_nlms_convergence',
         [f'run_C_nlms/{s}/h1/nlms_convergence.csv' for s in conv])


def md_test_nlms(late, out):
    rows = []
    for _, r in late.iterrows():
        share = r['share_nlms_better']
        rows.append([SCEN_LABEL.get(r['scenario'], r['scenario']), r['method'],
                     f"{int(r['t_from'])}–{int(r['t_to'])}", f"{r['q0.5']:.3f}",
                     f"{r['q0.25']:.3f}", f"{r['q0.75']:.3f}", f"{r['mean']:.3f}",
                     '' if r['method'] != 'NLMS' or not np.isfinite(share) else f'{share:.3f}'])
    _test_write(out, 'tab_test_nlms.md', 'decision_test/nlms_late.csv',
                ['Scenario', 'Method', '$t$', 'Median', 'Q1', 'Q3', 'Mean', 'Share NLMS better'],
                rows, [':--', ':--', ':-:', '--:', '--:', '--:', '--:', '--:'])


EXPLORE_REF = 'WSPI-F-YW'
EXPLORE_PAIRS = ('MA64', 'SMA32', 'EWMA64', 'EWMA32', 'ARYW', 'WSPI', 'WSPI-F2', 'GBM')


def md_test_explore(vce, pairs, out):
    md_test_pairs(vce, out, 'tab_test_explore_counts.md',
                  [(EXPLORE_REF, m) for m in EXPLORE_PAIRS],
                  src='decision_explore/verdict_counts_explore.csv')
    p = pairs[pairs['method'] == 'MA64'].set_index(['scenario', 'horizon', 'metric'])
    rows = []
    for h in HORIZONS:
        for k, s in enumerate(SCENARIOS):
            if (s, h, 'robustness_distortion') not in p.index:
                continue
            r = p.loc[(s, h, 'robustness_distortion')]
            cells = [str(h) if k == 0 else '', SCEN_LABEL[s], f"{r['mean_reference']:.2f}",
                     f"{r['mean_method']:.2f}",
                     f"{r['diff_ref_minus_method']:+.2f} [{r['diff_ci_low']:+.2f}, "
                     f"{r['diff_ci_high']:+.2f}]",
                     SYMBOL_MD.get(r['verdict'], '–')]
            for c in ('rsi@10', 'ndcg@10', 'spearman_rho'):
                cells.append(SYMBOL_MD.get(p.loc[(s, h, c)]['verdict'], '–')
                             if (s, h, c) in p.index else '–')
            rows.append(cells)
    _test_write(out, 'tab_test_explore_ma64.md', 'decision_explore/pairs_explore.csv',
                ['$h$', 'Scenario', 'ΔRank WSPI-F-YW', 'ΔRank MA64',
                 'Difference [95 % CI]', 'ΔRank', 'RSI@10', 'NDCG@10', 'Spearman ρ'], rows,
                [':--', ':--', '--:', '--:', '--:', ':-:', ':-:', ':-:', ':-:'],
                pre=['<!-- Exploratory: reference chosen after seeing the test-part means; '
                     'outside the gain rule. ↑ WSPI-F-YW better, ↓ worse, ○ n.s. -->', ''])


def test_part(root: Path, out: Path) -> int:
    """Figures and Markdown tables of the test part (run C); only CSV input."""
    def rd(rel, need=True):
        f = root / rel
        if not f.exists():
            if need:
                sys.exit(f'missing {f}')
            return pd.DataFrame()
        return pd.read_csv(f, float_precision='round_trip')
    means = rd('decision_test/means_test.csv')
    gsum = rd('decision_test/gain_summary_test.csv')
    claims = rd('decision_test/gain_claims_test.csv')
    vc = rd('decision_test/verdict_counts_test.csv')
    rr = rd('decision_test/runtime_ratio.csv')
    sr = rd('decision_test/sensitivity_range.csv')
    late = rd('decision_test/nlms_late.csv')
    conv = {s: rd(f'run_C_nlms/{s}/h1/nlms_convergence.csv') for s in ('youtube_hourly',
                                                                       'taxi_hourly')}
    for h in HORIZONS:
        for xm, ym in TRADE_PAIRS:
            fig_test_tradeoff(means, out, h, xm, ym)
    if POINTS:
        (out / 'figures_manifest_test_points.json').write_text(json.dumps(dict(
            root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), files=MANIFEST),
            indent=2), encoding='utf-8')
        print(f'{len(MANIFEST)} trade-off figures (--points) -> {out}')
        return 0
    md_test_gain(gsum, claims, out)
    md_test_ranks(means, out)
    md_test_pairs(vc, out, 'tab_test_gbm.md',
                  [('WSPI-F2', 'GBM'), ('WSPI-F2', 'GBM+W'), ('WSPI', 'GBM'), ('WSPI-FS3', 'GBM'),
                   ('GBM', 'GBM+W'), ('GBM', 'MA64')])
    md_test_runtime(rr, out)
    md_test_pairs(vc, out, 'tab_test_smoothing.md',
                  [('WSPI-F2', 'WSPI-S'), ('WSPI', 'WSPI-S'), ('WSPI-F2', 'WSPI')])
    md_test_sensitivity(sr, out)
    fig_test_sensitivity(sr, out)
    fig_nlms(conv, out)
    md_test_nlms(late, out)
    vce = rd('decision_explore/verdict_counts_explore.csv', need=False)
    explored = len(vce) > 0
    if explored:
        md_test_explore(vce, rd('decision_explore/pairs_explore.csv'), out)
    (out / 'figures_manifest_test.json').write_text(json.dumps(dict(
        root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), exploratory=explored,
        files=MANIFEST), indent=2), encoding='utf-8')
    print(f'{len(MANIFEST)} test-part figures and tables -> {out}'
          + ('' if explored else ' (no decision_explore/: exploratory tables left out)'))
    return 0


# ---- tables and figures of the rewritten Results on the test part (--paper-test) ------

PAPER_NOTE = ('<!-- Generated by tools/prediction_figures.py --paper-test from {src}. '
              'Do not edit by hand; rebuild instead. -->')
DOM_MAIN = ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W')
DOM_BASELINES = ('Persistence', 'Holt', 'ARYW', 'MA64', 'SMA8', 'SMA16', 'SMA32',
                 'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64')
PATH_STEPS = STEPS + ('WSPI-S', 'WSPI-FS3')
PATH_COLOR = {'WSPI-S': '#999999', 'WSPI-FS3': '#003F5C'}
SI_TEST_METHODS = ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'WSPI-FG', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
                   'WSPI-S', 'Persistence', 'Holt', 'ARYW', 'MA64', 'SMA8', 'SMA16', 'SMA32',
                   'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64', 'GBM', 'GBM+W')


def _paper_write(out, name, src, header, rows, align, pre=()):
    lines = [PAPER_NOTE.format(src=src), ''] + list(pre)
    lines += _md_table(header, rows, align)
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=[s.strip() for s in src.split(';')]))


def md_paper_dominance(summ, out):
    d = summ.set_index('method')
    rows = []
    for m in DOM_MAIN:
        if m not in d.index:
            continue
        r = d.loc[m]
        rows.append([m, f"{int(r['D'])} of {int(r['n_cells'])}"]
                    + [str(int(r[f'D_h{h}'])) for h in HORIZONS]
                    + [str(int(r['dominates_any']))])
    _paper_write(out, 'tab_test_dominance.md', 'decision_paper/dominance_summary.csv',
                 ['Method', 'Dominated cells'] + [f'$h$ = {h}' for h in HORIZONS]
                 + ['Cells where it dominates a baseline'], rows,
                 [':--', ':-:'] + [':-:'] * len(HORIZONS) + [':-:'],
                 pre=['<!-- A cell is dominated if at least one of the 11 baselines is better in '
                      'at least one main metric and worse in none (verdicts of the test part). '
                      'Six cells per horizon. -->', ''])


def md_paper_dominance_baselines(by_b, out):
    rows = []
    for m in DOM_MAIN + ('WSPI-F-YW',):
        d = by_b[by_b['method'] == m].set_index('baseline')
        if d.empty:
            continue
        cells = [m + (' (exploratory)' if m == 'WSPI-F-YW' else '')]
        for b in DOM_BASELINES:
            if b not in d.index:
                cells.append('–')
                continue
            r = d.loc[b]
            cells.append(f"{int(r['dominated'])} / {int(r['dominates'])} / "
                         f"{int(r['trade-off']) + int(r['equal'])}")
        rows.append(cells)
    _paper_write(out, 'tab_si_test_dominance_baselines.md', 'decision_paper/dominance_by_baseline.csv',
                 ['Method'] + list(DOM_BASELINES), rows, [':--'] + [':-:'] * len(DOM_BASELINES),
                 pre=['<!-- Cells of 18: dominated by the baseline / dominates it / neither. -->', ''])


def md_paper_positive(pf, out):
    rows = []
    for _, r in pf.iterrows():
        rows.append([r['method'], r['baseline']]
                    + [f"{int(r[c + '_better'])} / {int(r[c + '_ns'])} / {int(r[c + '_worse'])}"
                       for c in METRICS]
                    + [f"{int(r['dominates'])} / {int(r['dominated'])}"])
    _paper_write(out, 'tab_test_positive.md', 'decision_paper/positive_facts.csv',
                 ['Method', 'Baseline'] + [METRIC_MD[c] for c in METRICS]
                 + ['Dominates / dominated'], rows,
                 [':--', ':--'] + [':-:'] * len(METRICS) + [':-:'],
                 pre=['<!-- Cells of 18: method better / n.s. / worse; last column: cells where '
                      'the method dominates the baseline / is dominated by it. -->', ''])


def _md_gate(pg, out, method, name):
    d = pg[pg['method'] == method]
    rows = []
    for h in HORIZONS:
        e = d[d['horizon'] == h].set_index('scenario')
        for k, s in enumerate(SCENARIOS):
            if s not in e.index:
                continue
            r = e.loc[s]
            rows.append([str(h) if k == 0 else '', SCEN_LABEL[s]]
                        + [SYMBOL_MD[r[f'verdict_{c}']] for c in METRICS]
                        + ['yes' if bool(r['success']) else 'no'])
    against = d['against'].iloc[0]
    _paper_write(out, name, 'decision_paper/pareto_gates_test.csv',
                 ['$h$', 'Scenario'] + [METRIC_MD[c] for c in METRICS] + ['Success'], rows,
                 [':--', ':--'] + [':-:'] * len(METRICS) + [':-:'],
                 pre=[f'<!-- {method} against {against} on the test part. ↑ {method} better, '
                      f'↓ worse, ○ n.s. Success: better in at least one metric and worse in '
                      'none. -->', ''])


def md_paper_domain(dt, out):
    rows = []
    for s in SCENARIOS:
        for h in HORIZONS:
            d = dt[(dt['scenario'] == s) & (dt['horizon'] == h)]
            if d.empty:
                continue
            cells = [SCEN_LABEL[s], str(h)]
            for vs in ('WSPI', 'ARIMA'):
                e = d[d['against'] == vs].set_index('metric')
                cells += [SYMBOL_MD.get(e.loc[c, 'verdict'], '–') if c in e.index else '–'
                          for c in METRICS]
            cells += [f"{d['time_ratio_fa_to_arima'].iloc[0]:.2f}",
                      f"{d['fallback_share'].iloc[0]:.4f}"]
            rows.append(cells)
    head = (['Scenario', '$h$'] + [f'{METRIC_MD[c]} vs WSPI' for c in METRICS]
            + [f'{METRIC_MD[c]} vs ARIMA' for c in METRICS] + ['Time ratio', 'Fallback share'])
    _paper_write(out, 'tab_test_domain.md', 'decision_paper/domain_effect_test.csv', head, rows,
                 [':--', ':-:'] + [':-:'] * (2 * len(METRICS)) + ['--:', '--:'],
                 pre=['<!-- WSPI-FA against WSPI and ARIMA on the test part. ↑ WSPI-FA better, '
                      '↓ worse, ○ n.s. Time ratio: scorer time of WSPI-FA over ARIMA (run B, '
                      'every window). -->', ''])


def fig_paper_design_path(means, out, h):
    m = means[means['horizon'] == h].set_index(['scenario', 'metric', 'method'])
    fig, axes = plt.subplots(len(METRICS), len(SCENARIOS), figsize=(13.5, 8.0), sharex=True)
    x = np.arange(len(PATH_STEPS))
    colors = [STYLE[s_][0] if s_ in STYLE else PATH_COLOR[s_] for s_ in PATH_STEPS]
    for i, c in enumerate(METRICS):
        for j, s in enumerate(SCENARIOS):
            ax = axes[i, j]
            keys = [(s, c, st) for st in PATH_STEPS]
            if any(k in m.index for k in keys):
                mean = np.array([m.loc[k, 'mean'] if k in m.index else np.nan for k in keys])
                lo = np.array([m.loc[k, 'ci_low'] if k in m.index else np.nan for k in keys])
                hi = np.array([m.loc[k, 'ci_high'] if k in m.index else np.nan for k in keys])
                if not POINTS:
                    ax.bar(x, mean, color=colors, width=0.72)
                ax.errorbar(x, mean, yerr=[np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)],
                            fmt='none', ecolor='black', lw=0.8, capsize=2)
                if POINTS:
                    ax.scatter(x, mean, c=colors, s=22, zorder=3, edgecolors='black', linewidths=0.4)
                if np.isfinite(mean[0]):
                    ax.axhline(mean[0], color=STYLE['WSPI'][0], lw=0.8, ls='--')
                fin = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]])
                if (POINTS or c not in LOWER) and len(fin):
                    pad = 0.08 * max(fin.max() - fin.min(), 1e-6)
                    ax.set_ylim(fin.min() - pad, fin.max() + pad)
            else:
                ax.text(0.5, 0.5, 'no data', ha='center', va='center', transform=ax.transAxes)
            if i == 0:
                ax.set_title(SCEN_LABEL[s], fontsize=9)
            if j == 0:
                ax.set_ylabel(METRIC_LABEL[c], fontsize=8)
            ax.set_xticks(x)
            ax.set_xticklabels([st.replace('WSPI-', '') for st in PATH_STEPS], rotation=60, fontsize=7)
            ax.tick_params(axis='y', labelsize=7)
            ax.grid(axis='y', alpha=0.25, lw=0.5)
    fig.suptitle(f'Design steps on the test part, horizon $h$ = {h} '
                 f'(dashed line: WSPI; {"points" if POINTS else "bars"}: mean, 95% CI)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out, f'fig_test_design_path_h{h}', ['decision_test/means_test.csv'])


def md_paper_si_main(means, out):
    d = means.set_index(['scenario', 'horizon', 'metric', 'method'])
    for s in SCENARIOS:
        rows = []
        for h in HORIZONS:
            ms = [m for m in SI_TEST_METHODS if (s, h, METRICS[0], m) in d.index]
            best = {}
            for c in METRICS:
                v = [float(d.loc[(s, h, c, m), 'mean']) for m in ms if (s, h, c, m) in d.index]
                if v:
                    best[c] = min(v) if c in LOWER else max(v)
            for k, m in enumerate(ms):
                cells = [str(h) if k == 0 else '', m]
                for c in METRICS:
                    if (s, h, c, m) not in d.index:
                        cells.append('–')
                        continue
                    r = d.loc[(s, h, c, m)]
                    t = _si_num(float(r['mean']), c)
                    t = f'**{t}**' if float(r['mean']) == best.get(c) else t
                    cells.append(f"{t} ({_si_num(float(r['ci_low']), c)}–"
                                 f"{_si_num(float(r['ci_high']), c)})")
                rows.append(cells)
        _paper_write(out, f'tab_si_test_main_{s}.md',
                     'decision_test/means_test.csv (test part, reference report WSPI)',
                     ['$h$', 'Method'] + [METRIC_MD[c] for c in METRICS], rows,
                     [':-:', ':--'] + ['--:'] * len(METRICS))


def paper_test(root: Path, out: Path) -> int:
    """Items of the rewritten Results on the test part; only CSV input."""
    def rd(rel):
        f = root / rel
        if not f.exists():
            sys.exit(f'missing {f} (run prediction_test_analysis.py --paper first)')
        return pd.read_csv(f, float_precision='round_trip')
    means = rd('decision_test/means_test.csv')
    if POINTS:
        for h in HORIZONS:
            fig_paper_design_path(means, out, h)
        (out / 'figures_manifest_paper_test_points.json').write_text(json.dumps(dict(
            root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), files=MANIFEST),
            indent=2), encoding='utf-8')
        print(f'{len(MANIFEST)} design-path figures (--points) -> {out}')
        return 0
    md_paper_dominance(rd('decision_paper/dominance_summary.csv'), out)
    md_paper_dominance_baselines(rd('decision_paper/dominance_by_baseline.csv'), out)
    md_paper_positive(rd('decision_paper/positive_facts.csv'), out)
    pg = rd('decision_paper/pareto_gates_test.csv')
    _md_gate(pg, out, 'WSPI-FG', 'tab_test_pareto_gates.md')
    _md_gate(pg, out, 'WSPI-FS3', 'tab_si_test_pareto_fs3.md')
    md_paper_domain(rd('decision_paper/domain_effect_test.csv'), out)
    for h in HORIZONS:
        fig_paper_design_path(means, out, h)
    md_paper_si_main(means, out)
    (out / 'figures_manifest_paper_test.json').write_text(json.dumps(dict(
        root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), files=MANIFEST), indent=2),
        encoding='utf-8')
    print(f'{len(MANIFEST)} test-part items of the paper -> {out}')
    return 0


CONFIRM_NOTE = ('<!-- Generated by tools/prediction_figures.py --confirm-tables from {src}. '
                'Do not edit by hand; rebuild instead. -->')
DOM_DEV_LEVEL = ('WSPI-M', 'WSPI-H')
DOM_CONFIRM = ('WSPI', 'WSPI-F2', 'WSPI-M', 'WSPI-H')


def md_confirm_dominance(dev, dev_level, conf, out):
    """Dominance table: development data (test part, 18 cells) with the
    level methods of run D, and the confirmation data (test part, 12 cells)."""
    def row(data, r):
        return [data, r['method'], f"{int(r['D'])} of {int(r['n_cells'])}"] \
            + [str(int(r[f'D_h{h}'])) for h in HORIZONS] + [str(int(r['dominates_any']))]
    rows = []
    d = dev.set_index('method', drop=False)
    for m in DOM_MAIN:
        if m in d.index:
            rows.append(row('Development' if not rows else '', d.loc[m]))
    dl = dev_level.set_index('method', drop=False)
    for m in DOM_DEV_LEVEL:
        if m in dl.index:
            rows.append(row('', dl.loc[m]))
    c = conf.set_index('method', drop=False)
    first = True
    for m in DOM_CONFIRM:
        if m in c.index:
            rows.append(row('Confirmation' if first else '', c.loc[m]))
            first = False
    src = ('decision_paper/dominance_summary.csv; decision_D/test_dominance_summary.csv; '
           'confirm/decision/test_dominance_summary.csv')
    lines = [CONFIRM_NOTE.format(src=src), '',
             '<!-- Test part of each data set. A cell is dominated if at least one of the 11 '
             'baselines is better in at least one main metric and worse in none. Development: '
             '6 scenarios, 18 cells; confirmation: 4 scenarios, 12 cells. -->', '']
    lines += _md_table(['Data', 'Method', 'Dominated cells'] + [f'$h$ = {h}' for h in HORIZONS]
                       + ['Cells where it dominates a baseline'], rows,
                       [':--', ':--', ':-:'] + [':-:'] * len(HORIZONS) + [':-:'])
    name = 'tab_test_dominance_full.md'
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=[x.strip() for x in src.split(';')]))


def confirm_tables(root: Path, out: Path) -> int:
    """Tables that combine the development data with the confirmation data; only CSV input."""
    def rd(rel):
        f = root / rel
        if not f.exists():
            sys.exit(f'missing {f}')
        return pd.read_csv(f, float_precision='round_trip')
    md_confirm_dominance(rd('decision_paper/dominance_summary.csv'),
                         rd('decision_D/test_dominance_summary.csv'),
                         rd('confirm/decision/test_dominance_summary.csv'), out)
    (out / 'figures_manifest_confirm.json').write_text(json.dumps(dict(
        root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), files=MANIFEST), indent=2),
        encoding='utf-8')
    print(f'{len(MANIFEST)} confirmation tables of the paper -> {out}')
    return 0



# ---- Supplementary Information on the test part (--si-test) ------------------------

SIT_NOTE = ('<!-- Generated by tools/prediction_figures.py --si-test from {src}. '
            'Do not edit by hand; rebuild instead. -->')
CONF_SCENARIOS = ('taxi2024_hourly', 'taxi2024_30min', 'taxi2024_5min', 'wikipedia_daily')
CONF_LABEL = {'taxi2024_hourly': 'Taxi 2024 (1 h)', 'taxi2024_30min': 'Taxi 2024 (30 min)',
              'taxi2024_5min': 'Taxi 2024 (5 min)', 'wikipedia_daily': 'Wikipedia (1 d)'}
ALL_LABEL = {**SCEN_LABEL, **CONF_LABEL}
SIT_METHODS = ('WSPI', 'WSPI-F2', 'WSPI-H', 'WSPI-M', 'WSPI-FS3', 'WSPI-FG', 'WSPI-F-YW',
               'WSPI-F', 'WSPI-FT', 'WSPI-S', 'WSPI-FA', 'Persistence', 'Holt', 'ARYW', 'ARIMA',
               'MA64', 'SMA8', 'SMA16', 'SMA32', 'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64',
               'GBM', 'GBM+W')
SIT_BOUND_ORDER = ('WSPI-F2', 'WSPI-FG', 'WSPI-FT', 'WSPI-FA', 'WSPI-S', 'WSPI-FS1', 'WSPI-FS1.5',
                   'WSPI-FS2', 'WSPI-FS3', 'WSPI-H')
SIT_DOM_DEV = ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W', 'WSPI-M', 'WSPI-H')
SIT_DOM_CONF = ('WSPI', 'WSPI-F2', 'WSPI-M', 'WSPI-H')
SIT_PAIRS = ('WSPI-H vs WSPI-F2', 'WSPI-H vs WSPI', 'WSPI-M vs WSPI', 'WSPI-H vs WSPI-M',
             'WSPI-F-YW vs WSPI-M')
SLOT_LABEL = {5.0: '5 min', 30.0: '30 min', 60.0: '1 h', 1440.0: '1 d', 10080.0: '1 w'}


def _sit_write(out, name, src, header, rows, align, pre=()):
    lines = [SIT_NOTE.format(src=src), ''] + list(pre)
    lines += _md_table(header, rows, align)
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=[s.strip() for s in src.split(';')]))


def sit_data(root, out):
    """Shape of the data: slot length, slots, items and item threshold of every scenario."""
    rows = []
    for data, base, scen in (('Development', '', SCENARIOS), ('Confirmation', 'confirm/', CONF_SCENARIOS)):
        for k, s in enumerate(scen):
            f = root / f'{base}run_A/{s}/h1/metadata/protocol_v5_run.json'
            d = json.loads(f.read_text(encoding='utf-8'))
            rows.append([data if k == 0 else '', ALL_LABEL[s], SLOT_LABEL.get(float(d['slot_minutes']), '–'),
                         f"{int(d['num_slots']):,}", f"{int(d['num_items']):,}", str(int(d['dataset_min_obs']))])
    _sit_write(out, 'tab_sit_data.md', 'run_A/*/h1/metadata/protocol_v5_run.json; '
               'confirm/run_A/*/h1/metadata/protocol_v5_run.json',
               ['Data', 'Scenario', 'Slot', 'Slots', 'Items', 'Item threshold'], rows,
               [':--', ':--', ':-:', '--:', '--:', '--:'])


def sit_parts(root, out):
    """Windows and blocks of the tuning and the test part, per scenario and horizon."""
    def rd(rel):
        return pd.read_csv(root / rel, float_precision='round_trip')
    rows = []
    for data, base, scen in (('Development', '', SCENARIOS), ('Confirmation', 'confirm/', CONF_SCENARIOS)):
        pw = rd(f'{base}stats_test/part_windows.csv').set_index(['scenario', 'horizon'])
        blk = {}
        for part in ('tune', 'test'):
            t = rd(f'{base}stats_{part}/all_paired_tests.csv')
            t = t[t['metric'].isin(METRICS)]
            blk[part] = t.groupby(['scenario', 'horizon']).agg(
                b_lo=('block', 'min'), b_hi=('block', 'max'),
                n_lo=('n_blocks', 'min'), n_hi=('n_blocks', 'max'))
        first = True
        for s in scen:
            for h in HORIZONS:
                if (s, h) not in pw.index:
                    continue
                p = pw.loc[(s, h)]
                bt, bs = blk['tune'].loc[(s, h)], blk['test'].loc[(s, h)]
                rows.append([data if first else '', ALL_LABEL[s] if h == HORIZONS[0] else '', str(h),
                             f"{int(p['n_tune']):,}", f"{int(p['n_test']):,}",
                             _range(bs['b_lo'], bs['b_hi']), _range(bt['n_lo'], bt['n_hi']),
                             _range(bs['n_lo'], bs['n_hi'])])
                first = False
    _sit_write(out, 'tab_sit_parts.md', 'stats_test/part_windows.csv; stats_tune/all_paired_tests.csv; '
               'stats_test/all_paired_tests.csv; confirm/stats_test/part_windows.csv; '
               'confirm/stats_tune/all_paired_tests.csv; confirm/stats_test/all_paired_tests.csv',
               ['Data', 'Scenario', '$h$', 'Tuning windows', 'Test windows', 'Block length $b_h$',
                'Tuning blocks', 'Test blocks'], rows,
               [':--', ':--', ':-:', '--:', '--:', '--:', '--:', '--:'])


def sit_bounds(root, out):
    """Extremes of the score ratio of every clamped method, pooled over the scenarios."""
    def load(pattern):
        fs = sorted(root.glob(pattern))
        return pd.concat([pd.read_csv(f, float_precision='round_trip').assign(scenario=f.parent.parent.name)
                          for f in fs]) if fs else pd.DataFrame()

    def rows_of(data, d, order=None):
        g = d.groupby(['horizon', 'method'], sort=False).agg(
            m=('m', 'first'), lower=('lower_bound', 'first'), mn=('min_ratio', 'min'),
            mx=('max_ratio', 'max'), upper=('upper_bound', 'first'), n=('n_ratios', 'sum'),
            o1=('n_below', 'sum'), o2=('n_above', 'sum'), nf=('n_nonfinite', 'sum'),
            nsc=('scenario', 'nunique'), nlo=('lower_bound', 'nunique'),
            nup=('upper_bound', 'nunique')).reset_index()
        if (g['nlo'] > 1).any() or (g['nup'] > 1).any():
            sys.exit('bound table: one method and horizon with more than one bound')
        meths = [m for m in (order or list(dict.fromkeys(g['method']))) if m in set(g['method'])]
        res = []
        for meth in meths:
            for h in HORIZONS:
                r = g[(g['method'] == meth) & (g['horizon'] == h)]
                if r.empty:
                    continue
                r = r.iloc[0]
                res.append([data if not res else '', meth if h == HORIZONS[0] else '',
                            str(int(h)), str(int(r['m'])), str(int(r['nsc'])), f"{int(r['n']):,}",
                            f"{r['lower']:.4f}", f"{r['mn']:.4f}", f"{r['mx']:.4f}", f"{r['upper']:.4f}",
                            str(int(r['o1'] + r['o2'] + r['nf']))])
        return res

    dev = pd.concat([load(f'run_{r}/*/h*/bound_check.csv') for r in 'ABCD'])
    rows = rows_of('Development', dev, order=SIT_BOUND_ORDER)
    sens = load('run_C_sens/*/h*/bound_check.csv')
    if len(sens):
        sens = sens.sort_values('clamp', kind='stable').copy()
        sens['method'] = 'WSPI-F2, $c$ = ' + sens['clamp'].map(lambda c: f'{c:g}')
        rows += [[''] + r[1:] for r in rows_of('x', sens)]
    conf = pd.concat([load(f'confirm/run_{r}/*/h*/bound_check.csv') for r in 'ACD'])
    rows += rows_of('Confirmation', conf, order=SIT_BOUND_ORDER)
    _sit_write(out, 'tab_sit_bounds.md', 'run_A, run_B, run_C, run_C_sens, run_D, confirm/run_A, '
               'confirm/run_C, confirm/run_D: */h*/bound_check.csv',
               ['Data', 'Method', '$h$', '$m$', 'Scenarios', 'Ratios', 'Lower bound', 'Min ratio',
                'Max ratio', 'Upper bound', 'Outside'], rows,
               [':--', ':--', ':-:', ':-:', ':-:', '--:', '--:', '--:', '--:', '--:', '--:'])


def sit_gate(root, out):
    """Gate profile of WSPI-F2 and WSPI-FG on the test windows (window statistics only)."""
    rows = []
    for data, base, scen in (('Development', '', SCENARIOS), ('Confirmation', 'confirm/', CONF_SCENARIOS)):
        pw = pd.read_csv(root / f'{base}stats_test/part_windows.csv').set_index(['scenario', 'horizon'])
        first = True
        for s in scen:
            for h in HORIZONS:
                f = root / f'{base}run_A/{s}/h{h}/gate_profile.csv'
                if not f.exists() or (s, h) not in pw.index:
                    continue
                gp = pd.read_csv(f, float_precision='round_trip')
                p = pw.loc[(s, h)]
                gp = gp[(gp['window_id'] >= p['first_test']) & (gp['window_id'] <= p['last_test'])]
                if len(gp) != int(p['n_test']):
                    sys.exit(f'gate profile {s} h{h}: {len(gp)} test windows, expected {int(p["n_test"])}')
                rows.append([data if first else '', ALL_LABEL[s] if h == HORIZONS[0] else '', str(h),
                             f"{gp['g_median'].median():.3f}", f"{gp['fg_g_median'].median():.3f}",
                             f"{gp['fg_share_zero'].mean():.3f}"])
                first = False
    _sit_write(out, 'tab_sit_gate.md', 'run_A/*/h*/gate_profile.csv; confirm/run_A/*/h*/gate_profile.csv; '
               'stats_test/part_windows.csv; confirm/stats_test/part_windows.csv',
               ['Data', 'Scenario', '$h$', 'WSPI-F2: median $g$', 'WSPI-FG: median $g$',
                'WSPI-FG: share $g=0$'], rows, [':--', ':--', ':-:', '--:', '--:', '--:'])


def sit_kappa(root, out):
    k = pd.read_csv(root / 'stats_tune/kappa_selection.csv')
    rows = [[f"{r['kappa']:g}", r['reference'], f"{int(r['n_complete'])} of {int(r['n_cells'])}",
             str(int(r['n_success'])), 'yes' if str(r['selected']) == 'True' else 'no']
            for _, r in k.iterrows()]
    _sit_write(out, 'tab_sit_kappa.md', 'stats_tune/kappa_selection.csv',
               ['$\\kappa$', 'Method', 'Cells', 'Cells where it dominates WSPI-F2', 'Selected'], rows,
               [':-:', ':--', ':-:', ':-:', ':-:'])


def _sit_means(root):
    """Test-part means of every method: development (23 methods + WSPI-FA and ARIMA on two
    scenarios) and confirmation (23 methods)."""
    dev = pd.read_csv(root / 'decision_D/test_means.csv', float_precision='round_trip')
    b = pd.read_csv(root / 'stats_test_B/all_method_summary.csv', float_precision='round_trip')
    b = b[(b['reference_report'] == 'WSPI-FA') & b['method'].isin(RUN_B_ONLY) & b['metric'].isin(METRICS)]
    dev = pd.concat([dev, b[dev.columns]])
    conf = pd.read_csv(root / 'confirm/decision/test_means.csv', float_precision='round_trip')
    return pd.concat([dev, conf]).set_index(['scenario', 'horizon', 'metric', 'method'])


def _sit_secondary(root):
    d = pd.read_csv(root / 'stats_test_D/all_method_summary.csv', float_precision='round_trip')
    d = d[d['reference_report'] == 'WSPI-H']
    b = pd.read_csv(root / 'stats_test_B/all_method_summary.csv', float_precision='round_trip')
    b = b[(b['reference_report'] == 'WSPI-FA') & b['method'].isin(RUN_B_ONLY)]
    c = pd.read_csv(root / 'confirm/stats_test/all_method_summary.csv', float_precision='round_trip')
    c = c[c['reference_report'] == 'WSPI']
    cols = ['scenario', 'horizon', 'metric', 'method', 'mean']
    return pd.concat([d[cols], b[cols], c[cols]]).set_index(['scenario', 'horizon', 'metric', 'method'])


def sit_full(root, out):
    d = _sit_means(root)
    sec = _sit_secondary(root)
    for s in SCENARIOS + CONF_SCENARIOS:
        rows, rows2 = [], []
        for h in HORIZONS:
            ms = [m for m in SIT_METHODS if (s, h, METRICS[0], m) in d.index]
            best = {}
            for c in METRICS:
                v = [float(d.loc[(s, h, c, m), 'mean']) for m in ms if (s, h, c, m) in d.index]
                if v:
                    best[c] = min(v) if c in LOWER else max(v)
            for k, m in enumerate(ms):
                cells = [str(h) if k == 0 else '', m]
                for c in METRICS:
                    if (s, h, c, m) not in d.index:
                        cells.append('–')
                        continue
                    r = d.loc[(s, h, c, m)]
                    t = _si_num(float(r['mean']), c)
                    t = f'**{t}**' if float(r['mean']) == best.get(c) else t
                    cells.append(f"{t} ({_si_num(float(r['ci_low']), c)}–{_si_num(float(r['ci_high']), c)})")
                rows.append(cells)
                rows2.append([str(h) if k == 0 else '', m]
                             + [_si_num(float(sec.loc[(s, h, c, m), 'mean']), c)
                                if (s, h, c, m) in sec.index else '–' for c in SECONDARY])
        src = ('confirm/decision/test_means.csv' if s in CONF_SCENARIOS else
               'decision_D/test_means.csv; stats_test_B/all_method_summary.csv (WSPI-FA, ARIMA)')
        _sit_write(out, f'tab_sit_main_{s}.md', src,
                   ['$h$', 'Method'] + [METRIC_MD[c] for c in METRICS], rows,
                   [':-:', ':--'] + ['--:'] * len(METRICS))
        src2 = ('confirm/stats_test/all_method_summary.csv (reference report WSPI)' if s in CONF_SCENARIOS else
                'stats_test_D/all_method_summary.csv (reference report WSPI-H); '
                'stats_test_B/all_method_summary.csv (WSPI-FA, ARIMA)')
        _sit_write(out, f'tab_sit_secondary_{s}.md', src2,
                   ['$h$', 'Method'] + [SECONDARY_MD[c] for c in SECONDARY], rows2,
                   [':-:', ':--'] + ['--:'] * len(SECONDARY))


def sit_dominance_baselines(root, out):
    """Relation of each method to each baseline, development and confirmation data (test part)."""
    parts = (('Development', pd.concat([pd.read_csv(root / 'decision_paper/dominance_by_baseline.csv'),
                                        pd.read_csv(root / 'decision_D/test_dominance_by_baseline.csv')]),
              SIT_DOM_DEV),
             ('Confirmation', pd.read_csv(root / 'confirm/decision/test_dominance_by_baseline.csv'), SIT_DOM_CONF))
    rows = []
    for data, by_b, meths in parts:
        first = True
        for m in meths:
            d = by_b[by_b['method'] == m].drop_duplicates('baseline').set_index('baseline')
            if d.empty:
                continue
            cells = [data if first else '', m]
            for b in DOM_BASELINES:
                if b not in d.index:
                    cells.append('–')
                    continue
                r = d.loc[b]
                cells.append(f"{int(r['dominated'])} / {int(r['dominates'])} / "
                             f"{int(r['trade-off']) + int(r['equal'])}")
            rows.append(cells)
            first = False
    _sit_write(out, 'tab_sit_dominance_baselines.md', 'decision_paper/dominance_by_baseline.csv; '
               'decision_D/test_dominance_by_baseline.csv; confirm/decision/test_dominance_by_baseline.csv',
               ['Data', 'Method'] + list(DOM_BASELINES), rows, [':--', ':--'] + [':-:'] * len(DOM_BASELINES))


def sit_confirm(root, out):
    """Descriptive pairs and the tuning part of the confirmation data."""
    pc = pd.read_csv(root / 'confirm/decision/test_pair_counts.csv')
    pr = pd.read_csv(root / 'confirm/decision/test_pairs.csv')
    rows = []
    for pair in SIT_PAIRS:
        first_m, second = pair.split(' vs ')
        # the files store "WSPI-F2 vs WSPI-H" (verdict from the view of WSPI-F2); turn it around
        stored, flip = (pair, False) if (pc['pair'] == pair).any() else (f'{second} vs {first_m}', True)
        cells = [pair]
        for c in METRICS:
            r = pc[(pc['pair'] == stored) & (pc['metric'] == c)]
            if len(r) != 1:
                cells.append('–')
                continue
            r = r.iloc[0]
            b, w = (int(r['worse']), int(r['better'])) if flip else (int(r['better']), int(r['worse']))
            cells.append(f"{b} / {int(r['ns'])} / {w}")
        rel = pr[pr['pair'] == stored]['relation'].value_counts()
        dom, sub = int(rel.get('dominates', 0)), int(rel.get('dominated', 0))
        if flip:
            dom, sub = sub, dom
        cells += [str(dom), str(sub), str(int(len(pr[pr['pair'] == stored]) - dom - sub))]
        rows.append(cells)
    _sit_write(out, 'tab_sit_confirm_pairs.md', 'confirm/decision/test_pair_counts.csv; '
               'confirm/decision/test_pairs.csv',
               ['Pair (first method)'] + [METRIC_MD[c] for c in METRICS] + ['Dominates', 'Dominated', 'Neither'],
               rows, [':--'] + [':-:'] * len(METRICS) + [':-:'] * 3,
               pre=['<!-- Test part of the confirmation data, 12 cells. Metric cells: first method better / '
                    'n.s. / worse. -->', ''])
    rows = []
    for part, f in (('Tuning', 'confirm/decision/tune_dominance_summary.csv'),
                    ('Test', 'confirm/decision/test_dominance_summary.csv')):
        d = pd.read_csv(root / f).set_index('method', drop=False)
        for k, m in enumerate(SIT_DOM_CONF):
            if m not in d.index:
                continue
            r = d.loc[m]
            rows.append([part if k == 0 else '', m, f"{int(r['D'])} of {int(r['n_cells'])}"]
                        + [str(int(r[f'D_h{h}'])) for h in HORIZONS] + [str(int(r['dominates_any']))])
    _sit_write(out, 'tab_sit_confirm_parts.md', 'confirm/decision/tune_dominance_summary.csv; '
               'confirm/decision/test_dominance_summary.csv',
               ['Part', 'Method', 'Dominated cells'] + [f'$h$ = {h}' for h in HORIZONS]
               + ['Cells where it dominates a baseline'], rows,
               [':--', ':--', ':-:'] + [':-:'] * len(HORIZONS) + [':-:'])


def sit_runtime(root, out):
    """Time ratios of md_test_runtime, labelled without the names of the evaluation groups."""
    rr = pd.read_csv(root / 'decision_test/runtime_ratio.csv', float_precision='round_trip')
    spec = (('WSPI-F2 / WSPI', 'wspi_f2_over_wspi_runA', 2), ('WSPI-FS3 / WSPI', 'wspi_fs3_over_wspi_runC', 2),
            ('GBM training (s)', 'GBM_train_seconds', 2), ('GBM scoring (s)', 'GBM_score_seconds', 2),
            ('GBM total / WSPI', 'GBM_total_over_wspi_runC', 2), ('GBM total / WSPI-F2', 'GBM_total_over_wspi_f2', 2),
            ('GBM+W total / WSPI-F2', 'GBM+W_total_over_wspi_f2', 2))
    rows = []
    for lab, col, dg in spec:
        v = rr[col].astype(float)
        rows.append([lab, f'{v.min():,.{dg}f}', f'{v.median():,.{dg}f}', f'{v.max():,.{dg}f}'])
    _sit_write(out, 'tab_sit_runtime.md', 'decision_test/runtime_ratio.csv',
               [f'Quantity ({len(rr)} cells)', 'Min', 'Median', 'Max'], rows, [':--', '--:', '--:', '--:'])


def si_test(root: Path, out: Path) -> int:
    """Tables of the Supplementary Information on the test part; only CSV and JSON input."""
    sit_data(root, out)
    sit_parts(root, out)
    sit_bounds(root, out)
    sit_gate(root, out)
    sit_kappa(root, out)
    sit_full(root, out)
    sit_dominance_baselines(root, out)
    sit_confirm(root, out)
    sit_runtime(root, out)
    (out / 'figures_manifest_si_test.json').write_text(json.dumps(dict(
        root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), files=MANIFEST), indent=2),
        encoding='utf-8')
    print(f'{len(MANIFEST)} SI tables on the test part -> {out}')
    return 0


# ---- tables and figures of the aggregate frame of the paper (--frame) ------------------

FRAME_NOTE = ('<!-- Generated by tools/prediction_figures.py --frame from {src}. '
              'Do not edit by hand; rebuild instead. -->')
# Methods that left the main text for the SI; ranks still use all methods.
FRAME_SI_ONLY = ('WSPI-FT', 'WSPI-S', 'WSPI-FS3')
FRAME_RANK_ORDER = SMA_LINE[:3] + ('MA64',) + EWMA_LINE + ('WSPI', 'WSPI-F2', 'WSPI-H', 'WSPI-M',
                                                           'WSPI-F-YW', 'WSPI-F', 'WSPI-FG',
                                                           'ARYW', 'Holt', 'Persistence',
                                                           'GBM', 'GBM+W')
FRAME_GAIN_METHODS = ('WSPI', 'WSPI-F2', 'WSPI-H', 'GBM', 'GBM+W')
FRAME_DOM_DEV = ('WSPI', 'WSPI-F2', 'GBM', 'GBM+W', 'WSPI-M', 'WSPI-H')
FRAME_DOM_CONF = ('WSPI', 'WSPI-F2', 'WSPI-M', 'WSPI-H')
FRAME_TRADE_POINTS = {'WSPI': ('#009E73', '^', True), 'WSPI-F2': ('#0072B2', 's', True),
                      'WSPI-H': ('#D55E00', 'o', True), 'WSPI-M': ('#D55E00', 'o', False),
                      'WSPI-F-YW': ('#56B4E9', 'd', True), 'ARYW': ('#555555', 'v', True),
                      'Holt': ('#CC79A7', 'x', True), 'Persistence': ('#E69F00', '+', True),
                      'GBM': ('#000000', 'X', True), 'GBM+W': ('#000000', 'X', False)}
# (data, source folder, coefficient-domain method, raw-domain method, sign):
# sign +1: the coefficient-domain method is the reference of the report;
# sign -1: the raw-domain method is the reference (GBM, method GBM+W).
FRAME_DOMAIN_PAIRS = (
    ('Development', 'stats_explore', 'WSPI-F-YW', 'ARYW', 'AR(2), Yule–Walker', 'WSPI-F-YW', +1),
    ('Development', 'stats_test_D', 'WSPI-M', 'MA64', 'Window mean', 'WSPI-M', +1),
    ('Confirmation', 'confirm/stats_test', 'WSPI-M', 'MA64', 'Window mean', 'WSPI-M', +1),
    ('Development', 'stats_test_B', 'WSPI-FA', 'ARIMA', 'ARMA(1,1)', 'WSPI-FA', +1),
    ('Development', 'stats_test', 'GBM+W', 'GBM', 'Gradient boosting', 'GBM', -1),
)


def _frame_write(out, name, src, header, rows, align, pre=()):
    lines = [FRAME_NOTE.format(src=src), ''] + list(pre)
    lines += _md_table(header, rows, align)
    (out / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    MANIFEST.append(dict(file=name, sources=[s.strip() for s in src.split(';')]))


def frame_ranks(means, out):
    """Median rank over the 18 development cells; ranks among all methods of the test part."""
    rk = means.copy()
    rk['rank'] = np.nan
    for (s, h, c), d in rk.groupby(['scenario', 'horizon', 'metric']):
        rk.loc[d.index, 'rank'] = d['mean'].rank(ascending=c in LOWER, method='average')
    med = rk.groupby(['method', 'metric'])['rank'].median()
    rows = []
    for meth in FRAME_RANK_ORDER:
        cells = [meth + (' (= SMA64)' if meth == 'MA64' else '')]
        for c in METRICS:
            v = med.get((meth, c), np.nan)
            cells.append('–' if not np.isfinite(v) else f'{v:g}')
        rows.append(cells)
    n = int(rk['method'].nunique())
    _frame_write(out, 'tab_frame_ranks.md', 'decision_D/test_means.csv',
                 ['Method'] + [METRIC_MD[c] for c in METRICS], rows,
                 [':--'] + [':-:'] * len(METRICS),
                 pre=[f'<!-- Median over 18 cells of the rank among {n} methods; 1 = best. '
                      'Rows of ' + ', '.join(FRAME_SI_ONLY) + ' are in the SI. -->', ''])


def frame_gain(gsum, claims_h, out):
    """Gain rule against the best baseline; WSPI-H from its own claims file (run D)."""
    def counts(bset, meth, c):
        if meth == 'WSPI-H':
            d = claims_h[(claims_h['baseline_set'] == bset) & (claims_h['method'] == meth)
                         & (claims_h['metric'] == c)]
            if len(d) == 0:
                return None
            v = d['verdict'].value_counts()
            return int(v.get('ref_better', 0)), int(v.get('n.s.', 0)), int(v.get('ref_worse', 0))
        d = gsum[(gsum['baseline_set'] == bset) & (gsum['method'] == meth) & (gsum['metric'] == c)]
        if len(d) != 1:
            return None
        r = d.iloc[0]
        return int(r['ref_better']), int(r['n.s.']), int(r['ref_worse'])
    rows = []
    for bset, lab in (('A', 'Classic (4)'), ('A+X2', 'Full (11)')):
        for k, meth in enumerate(FRAME_GAIN_METHODS):
            cells = [lab if k == 0 else '', meth]
            tot = 0
            for c in METRICS:
                v = counts(bset, meth, c)
                if v is None:
                    cells.append('–')
                else:
                    tot += v[0]
                    cells.append(f'{v[0]} / {v[1]} / {v[2]}')
            cells.append(f'{tot} of {len(SCENARIOS) * len(HORIZONS) * len(METRICS)}')
            rows.append(cells)
    _frame_write(out, 'tab_frame_gain.md',
                 'decision_test/gain_summary_test.csv; decision_D/test_gain_claims.csv',
                 ['Baselines', 'Method'] + [METRIC_MD[c] for c in METRICS] + ['Gains'], rows,
                 [':--', ':--'] + [':-:'] * len(METRICS) + [':-:'])


def frame_dominance(dev, dev_level, conf, out):
    """Dominance table of the main text, without the SI-only methods."""
    def row(data, r):
        return [data, r['method'], f"{int(r['D'])} of {int(r['n_cells'])}"] \
            + [str(int(r[f'D_h{h}'])) for h in HORIZONS] + [str(int(r['dominates_any']))]
    rows = []
    d = dev.set_index('method', drop=False)
    dl = dev_level.set_index('method', drop=False)
    for m in FRAME_DOM_DEV:
        src = dl if m in dl.index else d
        if m in src.index:
            rows.append(row('Development' if not rows else '', src.loc[m]))
    c = conf.set_index('method', drop=False)
    first = True
    for m in FRAME_DOM_CONF:
        if m in c.index:
            rows.append(row('Confirmation' if first else '', c.loc[m]))
            first = False
    _frame_write(out, 'tab_frame_dominance.md',
                 'decision_paper/dominance_summary.csv; decision_D/test_dominance_summary.csv; '
                 'confirm/decision/test_dominance_summary.csv',
                 ['Data', 'Method', 'Dominated cells'] + [f'$h$ = {h}' for h in HORIZONS]
                 + ['Cells where it dominates a baseline'], rows,
                 [':--', ':--', ':-:'] + [':-:'] * len(HORIZONS) + [':-:'],
                 pre=['<!-- Test part of each data set. A cell is dominated if at least one of '
                      'the 11 baselines is better in at least one main metric and worse in none. '
                      'Development: 6 scenarios, 18 cells; confirmation: 4 scenarios, 12 cells. -->',
                      ''])


def frame_domain(tests, out):
    """Same model in two domains: cells in which the coefficient-domain version (or GBM+W)
    is significantly better / not different / significantly worse than the raw version."""
    rows, rec = [], []
    for data, src, coef, raw, model, ref, sign in FRAME_DOMAIN_PAIRS:
        t = tests[src]
        meth = raw if ref == coef else coef
        cells = [data, model, f'{coef} vs {raw}']
        n_cells = None
        for c in METRICS:
            d = t[(t['reference'] == ref) & (t['method'] == meth) & (t['metric'] == c)]
            v = d['verdict'].value_counts()
            rb, ns, rw = int(v.get('ref_better', 0)), int(v.get('n.s.', 0)), int(v.get('ref_worse', 0))
            better, worse = (rb, rw) if sign > 0 else (rw, rb)
            n_cells = len(d)
            cells.append(f'{better} / {ns} / {worse}')
            rec.append(dict(data=data, source=src, coef=coef, raw=raw, metric=c, cells=len(d),
                            better=better, ns=ns, worse=worse))
        cells.insert(3, str(n_cells))
        rows.append(cells)
    _frame_write(out, 'tab_frame_domain4.md',
                 'stats_explore/all_paired_tests.csv; stats_test_D/all_paired_tests.csv; '
                 'confirm/stats_test/all_paired_tests.csv; stats_test_B/all_paired_tests.csv; '
                 'stats_test/all_paired_tests.csv',
                 ['Data', 'Model', 'Pair', 'Cells'] + [METRIC_MD[c] for c in METRICS], rows,
                 [':--', ':--', ':--', ':-:'] + [':-:'] * len(METRICS))
    return pd.DataFrame(rec)


def frame_mechanism(tests, out, name, src):
    """Mechanism control: WSPI-H and MA64 against WSPI-HL (reference), four main metrics.
    Entries: cells in which the row method is significantly better / not different / worse."""
    rows = []
    for meth in ('WSPI-H', 'MA64'):
        cells = [meth]
        for c in METRICS:
            d = tests[(tests['reference'] == 'WSPI-HL') & (tests['method'] == meth)
                      & (tests['metric'] == c)]
            v = d['verdict'].value_counts()
            # ref_worse: WSPI-HL significantly worse, i.e. the row method better
            cells.append(f"{int(v.get('ref_worse', 0))} / {int(v.get('n.s.', 0))} / "
                         f"{int(v.get('ref_better', 0))}")
        rows.append(cells)
    _frame_write(out, name, src, ['Method'] + [METRIC_MD[c] for c in METRICS], rows,
                 [':--'] + [':-:'] * len(METRICS))


def fig_frame_tradeoff(means, out, h, xm, ym):
    """As fig_test_tradeoff, with WSPI-H and WSPI-M and without the SI-only methods."""
    m = means[means['horizon'] == h].set_index(['scenario', 'metric', 'method'])['mean']
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.6))
    for ax, s in zip(axes.ravel(), SCENARIOS):
        def xy(meth):
            k1, k2 = (s, xm, meth), (s, ym, meth)
            return (m[k1], m[k2]) if k1 in m.index and k2 in m.index else (np.nan, np.nan)
        for line, colour, ls, lab in ((SMA_LINE, '#999999', '--', 'SMA (8, 16, 32, 64 = MA64)'),
                                      (EWMA_LINE, '#7F4F24', ':', 'EWMA (8, 16, 32, 64)')):
            pts = np.array([xy(n) for n in line])
            ax.plot(pts[:, 0], pts[:, 1], color=colour, ls=ls, marker='.', ms=7, lw=1.1,
                    label=lab)
            for n, (x_, y_) in zip(line, pts):
                if np.isfinite(x_):
                    ax.annotate(n.replace('SMA', '').replace('EWMA', '').replace('MA', ''),
                                (x_, y_), textcoords='offset points', xytext=(3, 3),
                                fontsize=6.5, color=colour)
        for meth, (c, mk, filled) in FRAME_TRADE_POINTS.items():
            x_, y_ = xy(meth)
            if not np.isfinite(x_):
                continue
            ax.scatter([x_], [y_], marker=mk, s=42 if filled else 95, label=meth,
                       zorder=3 if filled else 4,
                       color=c if (filled or mk in 'x+') else 'none',
                       edgecolors=c if mk not in 'x+' else None, linewidths=1.2)
        if xm in LOWER:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=20))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(SCEN_LABEL[s], fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(METRIC_LABEL[xm] + (' (log scale)' if xm in LOWER else ''), fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRIC_LABEL[ym], fontsize=9)
    _legend_below(fig, axes, ncol=6)
    fig.suptitle(f'Test part, horizon $h$ = {h}: mean over the test windows '
                 '(better: ' + ('left' if xm in LOWER else 'right') + ' and up)', fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))
    name = (f"fig_frame_tradeoff_{xm.replace('@', '').replace('_', '-')}_"
            f"{ym.replace('@', '').replace('_', '-')}_h{h}")
    save(fig, out, name, ['decision_D/test_means.csv'])


def frame(root: Path, out: Path) -> int:
    """Items of the aggregate frame; only CSV input; new file names only."""
    def rd(rel):
        f = root / rel
        if not f.exists():
            sys.exit(f'missing {f}')
        return pd.read_csv(f, float_precision='round_trip')
    means = rd('decision_D/test_means.csv')
    frame_ranks(means, out)
    frame_gain(rd('decision_test/gain_summary_test.csv'), rd('decision_D/test_gain_claims.csv'),
               out)
    frame_dominance(rd('decision_paper/dominance_summary.csv'),
                    rd('decision_D/test_dominance_summary.csv'),
                    rd('confirm/decision/test_dominance_summary.csv'), out)
    tests = {src: rd(f'{src}/all_paired_tests.csv')
             for src in sorted({p[1] for p in FRAME_DOMAIN_PAIRS})}
    for src in tests:
        tests[src] = tests[src][tests[src]['part'] == 'test'] if 'part' in tests[src] else tests[src]
    dom = frame_domain(tests, out)
    frame_mechanism(rd('stats_test_E/all_paired_tests.csv'), out, 'tab_frame_mechanism_dev.md',
                    'stats_test_E/all_paired_tests.csv')
    frame_mechanism(rd('confirm/stats_test_E/all_paired_tests.csv'), out,
                    'tab_frame_mechanism_conf.md', 'confirm/stats_test_E/all_paired_tests.csv')
    for h in HORIZONS:
        for xm, ym in TRADE_PAIRS:
            fig_frame_tradeoff(means, out, h, xm, ym)
    (out / 'figures_manifest_frame.json').write_text(json.dumps(dict(
        root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'), files=MANIFEST), indent=2),
        encoding='utf-8')
    dom.to_csv(out / 'frame_domain_counts_check.csv', index=False)
    print(f'{len(MANIFEST)} frame figures and tables -> {out}')
    return 0



def main():
    ap = argparse.ArgumentParser(description='Figures and tables of the popularity-prediction study')
    ap.add_argument('--root', default='results/prediction')
    ap.add_argument('--out', default='paper_figures/prediction')
    ap.add_argument('--strict', action='store_true', help='fail if a scenario is missing')
    ap.add_argument('--manuscript', action='store_true',
                    help='also write the main-text items of the paper in Markdown '
                         '(two combined figures and three Markdown tables)')
    ap.add_argument('--si', action='store_true',
                    help='also write the Markdown tables of the Supplementary Information')
    ap.add_argument('--test-part', action='store_true',
                    help='only the figures and tables of the test part of run C')
    ap.add_argument('--paper-test', action='store_true',
                    help='only the items of the rewritten Results on the test part '
                         '(dominance, named baselines, gates, domain effect, design path, SI)')
    ap.add_argument('--confirm-tables', action='store_true',
                    help='only the tables that combine the development data with the '
                         'confirmation data (dominance)')
    ap.add_argument('--si-test', action='store_true',
                    help='only the tables of the Supplementary Information on the test part '
                         '(data, parts, bounds, gate, kappa, full tables, dominance, confirmation)')
    ap.add_argument('--points', action='store_true',
                    help='with --paper-test: only the design-path figures as points with error '
                         'bars; with --test-part: only the trade-off figures, with labelled ticks '
                         'on the logarithmic axis')
    ap.add_argument('--frame', action='store_true',
                    help='only the items of the aggregate frame: rank, gain and dominance tables without the SI-only methods, four same-model pairs, mechanism tables, trade-off figures')
    a = ap.parse_args()
    global POINTS
    POINTS = a.points
    if a.points and not (a.test_part or a.paper_test):
        sys.exit('--points needs --test-part or --paper-test')
    root = Path(a.root)
    root = root if root.is_absolute() else ROOT / root
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT / out
    if (ROOT / 'results').resolve() in [out.resolve(), *out.resolve().parents]:
        sys.exit('--out must not be inside results/ (result folders hold only CSV and JSON)')
    out.mkdir(parents=True, exist_ok=True)
    if a.test_part:
        return test_part(root, out)
    if a.paper_test:
        return paper_test(root, out)
    if a.confirm_tables:
        return confirm_tables(root, out)
    if a.si_test:
        return si_test(root, out)
    if a.frame:
        return frame(root, out)

    def rd(rel, need=True):
        f = root / rel
        if not f.exists():
            if need:
                sys.exit(f'missing {f}')
            return pd.DataFrame()
        return pd.read_csv(f, float_precision='round_trip')

    summ = rd('stats/all_method_summary.csv')
    cells = rd('decision/pareto_cells.csv')
    claims = rd('decision/gain_claims.csv')
    domain = rd('decision/domain_effect.csv', need=False)
    bounds = rd('decision/bound_check.csv', need=False)
    numbers = rd('prediction_numbers.csv', need=False)
    sa = summary_a(summ)

    missing = set()
    for c in METRICS:
        missing |= set(fig_sweep(sa, out, c))
        tab_main(sa, out, c)
    for m in ('WSPI-FG', 'WSPI-F2', 'WSPI'):
        fig_map(claims, out, m)
    for h in HORIZONS:
        fig_steps(sa, out, h)
    if len(numbers):
        fig_gate(numbers, out)
    tab_pareto(cells, out)
    if len(domain):
        tab_domain(domain, out)
    if len(bounds):
        tab_bounds(bounds, out)
    if a.manuscript:
        missing |= set(fig_sweep_combined(sa, out))
        fig_map_combined(claims, out)
        md_pareto(cells, out)
        md_summary(rd('stats/all_paired_tests.csv'), out)
        if len(domain):
            md_domain(domain, out)
    if a.si:
        si_selection(rd('gate_selection/selection_summary.csv'), out)
        si_windows(rd('stats/all_paired_tests.csv'), out)
        if len(bounds):
            si_bounds(bounds, out)
        if len(numbers):
            si_gate(numbers, out)
        si_main(summ, out)
        si_secondary(summ, out)
    (out / 'figures_manifest.json').write_text(json.dumps(dict(
        root=str(root), created=time.strftime('%Y-%m-%d %H:%M:%S'),
        missing_scenarios=sorted(missing), files=MANIFEST), indent=2), encoding='utf-8')
    print(f'{len(MANIFEST)} figures and tables -> {out}')
    if missing:
        print('scenarios without data: ' + ', '.join(sorted(missing)))
        if a.strict:
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
