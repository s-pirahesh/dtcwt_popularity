r"""
Generate the figures of the PhD thesis (Chapter 4) from the protocol-V5 results
===============================================================================
One command builds every thesis figure that has V5 data, so the thesis never
uses a V4 picture again.  Figures shared with the paper are drawn by the same
functions as the paper (scripts/generate_revision_figures.py), so the thesis
and the paper show the same numbers and the same look (paper V4 style).
Labels are English (decision of 27 Sep 2026); captions are Persian in the
thesis text.  Added in task T3.10 (chat 17, 27 Sep 2026).

Parameters
----------
  --results  folder of the revision results (normally results\revision_v5)
  --out      folder for the figures (PDF + PNG), e.g. Thesis Document\Figures\V5
  --only     optional list of figure ids
  --list     print the registry (thesis figure -> id -> status) and stop

Registry
--------
THESIS_FIGURES below maps every Chapter-4 figure of thesis draft V1.5 (file
05_Ch4_Arzyabi.md) and the new MovieLens figures to a figure id and a status:
  ready    built by this program from V5 CSVs
  paper    the same file as a paper figure (built here under the thesis name)
  planned  not built yet; the V4 picture must not be used; to be added when
           Chapter 4 is rewritten (task T3.12 of 07_Task_Tracker.md)
  other    belongs to another work stream (prediction method) with its own scripts

Figures built here
------------------
  thesis_fig4_03_youtube_rsi_time   = paper Figure 6
  thesis_fig4_08_taxi_rsi_time      = paper Figure 7
  thesis_fig4_13_taxi30_rsi_time    RSI@10 over time, taxi 30 min, rolling 336 (1 week)
  thesis_fig4_16_taxi5_rsi_time     RSI@10 over time, taxi 5 min, rolling 2016 (1 week)
  thesis_fig4_18..21                = paper Figures 2-5 (NDCG@10, rho, RSI@10, Delta Rank)
  thesis_fig4_26_granularity        = paper Figure 8
  thesis_ml_default                 = paper fig:movielens
  thesis_ml_equal64                 MovieLens, equal window 64 (nine methods, 95 % CI)
  thesis_ml_rsi_time                = SI figure of MovieLens RSI@10 over time
  thesis_ml_profile                 MovieLens data profile per year 1998-2023:
                                    ratings, active movies, share of ratings on the
                                    user's first day (input data_prep/year_profile.csv)
  thesis_ml_periods                 mean of the four metrics in 1998-2014 against
                                    2015-2023 (default configuration, window time
                                    stamp; daily and weekly)

Usage (Windows, from the project root)
--------------------------------------
  python scripts\generate_thesis_figures.py --results results\revision_v5 ^
         --out "D:\Research\Thesis Research\Thesis Document\Figures\V5"
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_revision_figures as grf  # noqa: E402
from generate_revision_figures import plt  # noqa: E402  (Agg backend already set)

ROOT = HERE.parent
PERIOD_SPLIT = pd.Timestamp('2015-01-01', tz='UTC')   # decision of chat 16 (R15 section 6.7)
TAXI_ROLL = {'taxi_30min': 336, 'taxi_5min': 2016}     # one week, as the T1.6 blocks


def _paper(fn, name):
    """Run a paper figure function and rename its output to the thesis name."""
    def run(results, out):
        files, msg = fn(results, out)
        if files is None:
            return None, msg
        new = []
        for f in files:
            src = out / f
            dst = out / (name + src.suffix)
            src.replace(dst)
            new.append(dst.name)
        return new, msg + f' (same figure as {files[0].rsplit(".", 1)[0]})'
    return run


def taxi_time(scenario, name, title):
    def run(results, out):
        try:
            df, n = grf.v4_window_series(results, 'default', scenario)
        except (FileNotFoundError, KeyError, ValueError) as e:
            return None, f'input missing or inconsistent: {e}'
        roll = TAXI_ROLL[scenario]
        fig, ax = plt.subplots(figsize=(12, 4.6))
        grf._v4_time_panel(ax, df, roll, 'slots')
        ax.set_title(title, fontsize=10.5)
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False, fontsize=9)
        for t in leg.get_texts():
            if t.get_text() in grf.V4_WAVELET:
                t.set_color(grf.V4_PURPLE)
                t.set_fontweight('bold')
        return grf._save_v4(fig, out, name), f'{n} common windows, rolling mean {roll} (one week)'
    return run


def ml_equal64(results, out):
    try:
        return grf.ml_bars_figure(results, out, 'equal64', 'thesis_ml_equal64')
    except (FileNotFoundError, KeyError) as e:
        return None, f'input missing: {e}'


def ml_profile(results, out):
    p = results / 'T3.9_movielens' / 'data_prep' / 'year_profile.csv'
    if not p.is_file():
        return None, f'input missing: {p}'
    d = pd.read_csv(p)
    d = d[d['year'].astype(str).str.fullmatch(r'\d{4}')].copy()   # drop the 'all' row
    d['year'] = d['year'].astype(int)
    d = d[d['year'] >= 1998]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [('ratings', '(a) Ratings per year (millions)', 1e-6, '#C71585'),
             ('active_movies', '(b) Movies rated in the year (thousands)', 1e-3, '#4682B4'),
             ('share_on_user_first_day', "(c) Share of ratings on the user's first day", 1.0, '#FF8C00')]
    for ax, (col, title, k, c) in zip(axes, specs):
        grf._v4_axes(ax)
        ax.bar(d['year'], d[col] * k, color=c, edgecolor=grf.V4_EDGE, linewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis='x', labelsize=8)
    axes[2].set_ylim(0, 1)
    fig.suptitle('MovieLens 32M, 1998-2023 (last year up to 12 Oct 2023)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return grf._save_v4(fig, out, 'thesis_ml_profile'), f'{len(d)} years from {p.name}'


def ml_periods(results, out):
    """Mean of the four metrics before and after 1 Jan 2015 (window time stamp),
    default configuration, common windows; rows daily and weekly."""
    rows = []
    try:
        for sc, lab in grf.ML_SCENARIOS:
            for met in grf.MAIN_METRICS:
                df, _ = grf.v4_window_series(results, 'default', sc, metric=met)
                early = df['time'] < PERIOD_SPLIT
                for per, mask in (('1998-2014', early), ('2015-2023', ~early)):
                    for m in grf.V4_ORDER:
                        rows.append(dict(scenario=sc, period=per, metric=met, method=m,
                                         mean=float(df.loc[mask, m].mean()),
                                         n=int(df.loc[mask, m].notna().sum())))
    except (FileNotFoundError, KeyError, ValueError) as e:
        return None, f'input missing or inconsistent: {e}'
    v = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    periods = ['1998-2014', '2015-2023']
    width = 0.8 / len(grf.V4_ORDER)
    offsets = (np.arange(len(grf.V4_ORDER)) - (len(grf.V4_ORDER) - 1) / 2.0) * width
    for i, (sc, lab) in enumerate(grf.ML_SCENARIOS):
        for j, met in enumerate(grf.MAIN_METRICS):
            ax = axes[i, j]
            grf._v4_axes(ax)
            for k, m in enumerate(grf.V4_ORDER):
                ys = [v[(v.scenario == sc) & (v.period == p) & (v.metric == met) & (v.method == m)]['mean'].iloc[0]
                      for p in periods]
                kw = dict(width=width, color=grf.V4_COLORS[m], zorder=3)
                if m in grf.V4_WAVELET:
                    kw.update(edgecolor=grf.V4_EDGE, linewidth=grf.V4_EDGE_LW, hatch=grf.V4_HATCH)
                else:
                    kw.update(edgecolor='white', linewidth=0.4)
                ax.bar(np.arange(2) + offsets[k], ys, **kw)
            ax.set_xticks(np.arange(2))
            ax.set_xticklabels(periods, fontsize=9)
            if met == 'robustness_distortion':
                ax.set_yscale('log')
            else:
                ax.set_ylim(0, 1.05)
            ax.set_title(f'{lab} \u2014 {grf.V4_SHORT[met]}', fontsize=10)
    n = v[(v.metric == 'ndcg@10') & (v.method == 'WSPI')].set_index(['scenario', 'period'])['n']
    grf._v4_legend(fig, grf.V4_ORDER, anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return grf._save_v4(fig, out, 'thesis_ml_periods'), \
        'windows per period: ' + ', '.join(f'{s}/{p}={c}' for (s, p), c in n.items())


# thesis figure (V1.5 numbering) -> (id, status, builder or note)
THESIS_FIGURES = [
    ('4-1 overview, YouTube', 'yt_overview', 'planned', None),
    ('4-2 RSI at depths 5/10/20, YouTube', 'yt_rsi_depths', 'planned', None),
    ('4-3 RSI@10 over time, YouTube', 'thesis_fig4_03_youtube_rsi_time', 'paper',
     _paper(grf.fig_t310_fig6, 'thesis_fig4_03_youtube_rsi_time')),
    ('4-4 robustness, YouTube', 'yt_robustness', 'planned', None),
    ('4-5 metric heatmap, YouTube', 'yt_heatmap', 'planned', None),
    ('4-6 overview, taxi hourly', 'th_overview', 'planned', None),
    ('4-7 RSI at depths, taxi hourly', 'th_rsi_depths', 'planned', None),
    ('4-8 RSI@10 over time, taxi hourly', 'thesis_fig4_08_taxi_rsi_time', 'paper',
     _paper(grf.fig_t310_fig7, 'thesis_fig4_08_taxi_rsi_time')),
    ('4-9 Spearman over time, taxi hourly', 'th_spearman_time', 'planned', None),
    ('4-10 robustness, taxi hourly', 'th_robustness', 'planned', None),
    ('4-11 metric heatmap, taxi hourly', 'th_heatmap', 'planned', None),
    ('4-12 overview, taxi 30 min', 't30_overview', 'planned', None),
    ('4-13 RSI@10 over time, taxi 30 min', 'thesis_fig4_13_taxi30_rsi_time', 'ready',
     taxi_time('taxi_30min', 'thesis_fig4_13_taxi30_rsi_time',
               'NYC Yellow Taxi 30m \u2014 RSI@10 over the common evaluation windows')),
    ('4-14 metric heatmap, taxi 30 min', 't30_heatmap', 'planned', None),
    ('4-15 overview, taxi 5 min', 't5_overview', 'planned', None),
    ('4-16 RSI@10 over time, taxi 5 min', 'thesis_fig4_16_taxi5_rsi_time', 'ready',
     taxi_time('taxi_5min', 'thesis_fig4_16_taxi5_rsi_time',
               'NYC Yellow Taxi 5m \u2014 RSI@10 over the common evaluation windows')),
    ('4-17 metric heatmap, taxi 5 min', 't5_heatmap', 'planned', None),
    ('4-18 NDCG@10, grouped by scenario', 'thesis_fig4_18_ndcg10', 'paper',
     _paper(grf.fig_t310_fig2, 'thesis_fig4_18_ndcg10')),
    ('4-19 Spearman, grouped by scenario', 'thesis_fig4_19_spearman', 'paper',
     _paper(grf.fig_t310_fig3, 'thesis_fig4_19_spearman')),
    ('4-20 RSI@10, grouped by scenario', 'thesis_fig4_20_rsi10', 'paper',
     _paper(grf.fig_t310_fig4, 'thesis_fig4_20_rsi10')),
    ('4-21 Delta Rank, grouped by scenario', 'thesis_fig4_21_deltarank', 'paper',
     _paper(grf.fig_t310_fig5, 'thesis_fig4_21_deltarank')),
    ('4-22..4-25 the four metrics grouped by method', 'method_view', 'planned', None),
    ('4-26 effect of granularity', 'thesis_fig4_26_granularity', 'paper',
     _paper(grf.fig_t310_fig8, 'thesis_fig4_26_granularity')),
    ('4-27 sparsity against granularity', 'sparsity', 'planned', None),
    ('4-28..4-31 prediction method (WSPI-F2)', 'prediction', 'other', None),
    ('4-32 ablation', 'ablation', 'planned', None),
    ('4-33 sensitivity of alpha and beta', 'sensitivity', 'planned', None),
    ('new: MovieLens, default configuration', 'thesis_ml_default', 'paper',
     _paper(grf.fig_t310_movielens, 'thesis_ml_default')),
    ('new: MovieLens, equal window 64', 'thesis_ml_equal64', 'ready', ml_equal64),
    ('new: MovieLens, RSI@10 over time', 'thesis_ml_rsi_time', 'paper',
     _paper(grf.fig_t310_ml_rsi_time, 'thesis_ml_rsi_time')),
    ('new: MovieLens data profile', 'thesis_ml_profile', 'ready', ml_profile),
    ('new: MovieLens 1998-2014 against 2015-2023', 'thesis_ml_periods', 'ready', ml_periods),
]
BUILDERS = {fid: fn for _, fid, st, fn in THESIS_FIGURES if fn is not None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results', required=False)
    ap.add_argument('--out', required=False)
    ap.add_argument('--only', nargs='*', default=None, choices=list(BUILDERS))
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for tf, fid, st, _ in THESIS_FIGURES:
            print(f'{st:<8} {fid:<36} {tf}')
        return
    if not (a.results and a.out):
        ap.error('--results and --out are required (or use --list)')
    res = Path(a.results) if Path(a.results).is_absolute() else ROOT / a.results
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if not res.is_dir():
        sys.exit(f'--results folder not found: {res}')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    ids = a.only or list(BUILDERS)
    ok = 0
    for fid in ids:
        files, msg = BUILDERS[fid](res, out)
        if files is None:
            print(f'[skip] {fid}: {msg}')
        else:
            ok += 1
            print(f'[ok]   {fid}: {", ".join(files)}\n       {msg}')
    planned = sum(st == 'planned' for _, _, st, _ in THESIS_FIGURES)
    print(f'{ok} of {len(ids)} figures written to {out}; {planned} thesis figures still planned (--list)')


if __name__ == '__main__':
    main()
