r"""
Generate all figures of the Scientific Reports revision (branch revision-srep-v5)
================================================================================
Every figure is rebuilt from the real result CSVs under ``results/revision_v5``.
A figure whose input is missing is skipped with a message; the others still run.
New figures of later tasks (T2.4, T3.x, ...) are added to FIGURES below.

The older script ``scripts/generate_paper_figures.py`` (figures of paper V4)
is not touched.

Parameters
----------
  --results  folder of the revision results (normally results\revision_v5)
  --out      folder for the figures (PDF + PNG)
  --only     optional list of figure ids (default: all)

Figures
-------
  T2.2_window_curves  (task T2.2, SI figure) input: <results>/T2.2_window_sweep/sweep_summary.csv
      NDCG@10 (row 1) and RSI@10 (row 2) against the window length N for every
      method except PFRF and Holt, 4 scenarios.  Output T2.2_window_curves.pdf/.png.
      Decision of Sajjad, 25 Sep 2026: this replaces the Pareto plot in the paper.

  T2.3_tradeoff  (task T2.3) input: <results>/T2.2_window_sweep/sweep_summary.csv
      Accuracy-stability trade-off over window length N = 7, 16, 32, 64, 128.
      One curve per method; 2 rows (x = RSI@10; y = NDCG@10, Spearman rho) x
      4 scenarios.  Dashed step line = 2-D Pareto frontier of the panel
      (higher is better on both axes), computed over the methods shown.
      Kept in the program only; not used in the paper (decision of 25 Sep 2026).
      Two variants:
        main -> T2.3_tradeoff_main.pdf/.png  (without PFRF and Holt; paper text)
        full -> T2.3_tradeoff_full.pdf/.png  (all 12 methods; SI)
      Source data of the figure (CSV, with 2-D and 4-criteria Pareto flags):
        <results>/T2.3_tradeoff/pareto_points.csv
        <results>/T2.3_tradeoff/metadata/generate_revision_figures_run.json
      The 4-criteria flag (NDCG@10, rho, RSI@10 up; robustness down) is kept in
      the CSV only: it marks 13-29 of 47 points per scenario and does not
      discriminate, so it is not drawn.

  T2.4_surge_examples  (task T2.4 / E11, main-text figure)
      input: <results>/T2.4_responsiveness/{youtube_hourly,taxi_hourly}/examples/
             example_events.csv and example_traces.csv
      Two automatically chosen entries per scenario (typical, worst case for
      WSPI; rule in evaluation/responsiveness.select_examples).  Row 1: real
      count; row 2: rank of the item for WSPI, AF(7), DTCWT+AF(64), RRD(64).

  T2.4_delay_ecdf  (task T2.4 / E11, SI figure)
      input: <results>/T2.4_responsiveness/<scenario>/<config>/delays.csv
      Share of entries in the method's Top-10 within d slots; rows = default
      and equal-64 configurations; 4 scenarios.

  T3.2_param_heatmap  (task T3.2 / E3, SI figure)
      input: <results>/T3.2_param_grid/grid_summary.csv (part = test) and
             <results>/T3.2_param_grid/selection/selection.csv
      WSPI (N=64, J=3) over the alpha x beta grid on the test part (last 70 %
      of the common windows).  Rows: NDCG@10, RSI@10, robustness (Delta Rank);
      one column per scenario.  One blue scale per panel, darker = better
      (reversed for Delta Rank).  Square = default (1, 1); circle = the
      configuration chosen on the tuning part by the pre-registered 30/70 rule.

  T3.5_spike_size  (task T3.5 / E4, SI figure)
      input: <results>/T3.5_robustness/robustness_summary.csv
      Rank displacement (Delta Rank, log scale) against the spike size
      (2, 5, 10, 20, 50 x the 64-slot mean; last slot, one slot) under the
      common perturbation of E4 (same targets and spike for every method,
      5 seeds).  Rows: default configuration (baselines N=7, wavelet-based
      N=64) and equal window N=64; one column per scenario.  WSPI, DTCWT+AF,
      RRD and AF highlighted, the other methods grey with their name.

  T3.6_shift_invariance  (task T3.6 / E5, SI figure)
      input: <results>/T3.6_shift_invariance/shift_summary.csv and
             <results>/T3.6_shift_invariance/synthetic/synthetic_{scores,summary}.csv
      Row 1: real data, every 64-slot window of every eligible item shifted
      circularly by s = 0..7 (content fixed, periodic boundary): mean coefficient
      of variation of the band energies, DTCWT against DWT, 4 scenarios.
      Row 2: synthetic event on a fixed background moved towards the newest
      slot, settings of each method: (e) example curves (repetition 0, 3-slot
      burst, recent part); (f) share of steps in which the score falls although
      the event got newer, WSPI, DWT-WSPI, DTCWT+AF, DWT+AF, SMA, EWMA-eq.

  T3.7_feature_relation  (task T3.7 / E7, SI figure)
      input: <results>/T3.7_feature_relation/<scenario>/density.csv and pooled_summary.csv
      Density (log colour, 100 x 100 cells) of R against WE over all WSPI
      item-windows of each scenario, with the bounds that the identity
      WE log2(J+1) = h(R) + (1-R) H(q) puts on WE for a given R: h(R)/2 (all
      detail energy in one band) and (h(R) + (1-R) log2 3)/2 (detail energy spread
      evenly over the three bands).  Spearman correlation and eta^2(WE | R) in
      each panel.

  T3.8_runtime  (task T3.8 / E9, SI figure)
      input: <results>/T3.8_runtime/bench/runtime_grid.csv and
             <results>/T3.8_runtime/memory/tracemalloc_grid.csv
      Measured cost of the nine protocol-V5 scorers on one CPU core (median of
      10 repeats, band = IQR): (a) time to score all M items against M, default
      windows (baselines 7, wavelet-based 64); (b) microseconds per item-window
      against N (M = 1e4); (c) peak traced working memory per item against N
      (batch 1e4), with the exact size of the DTCWT coefficients.  Hardware and
      versions from bench/metadata/bench_run.json in the title.

  T3.10_fig2 ... T3.10_fig8  (task T3.10, main-text Figures 2-8 of the paper)
      Rebuilt from the causal protocol-V5 runs with the look of paper V4
      (colours, grey axes, hatched wavelet-based bars of
      evaluation/cross_dataset_visualizer.py).  Default configuration
      (baselines 7, wavelet-based 64), common windows of all nine methods.
      input: <results>/T1.5_causal_universe/T1.4_protocol_v5/<scenario>/
             comparison/summary_common_windows.csv and protocol/*_protocol.csv;
             95 % block-bootstrap CI from <results>/T1.6_stats/causal/all_method_summary.csv
             (the mean there is checked against the summary; mismatch = stop).
        fig2 NDCG@10, fig3 Spearman rho, fig4 RSI@10, fig5 Delta Rank (log scale):
             grouped bars, 4 scenarios x 9 methods, whiskers = 95 % CI.
        fig6 YouTube, fig7 NYC Taxi hourly: window-by-window RSI@10, centred
             rolling mean of one period (24 h, 168 h); WSPI, DTCWT+AF, DWT+AF,
             AF, RRD in colour, the other baselines grey; x axis = date (UTC).
        fig8 taxi granularity: (a) RSI@10, (b) Delta Rank, (c) NDCG@10 for WSPI,
             DTCWT+AF and the best baseline of each point (PFRF excluded).

  T3.10_fig_movielens  (task T3.10, paper Subsection 4.10)
      input: <results>/T3.9_movielens/default/movielens_{daily,weekly}/comparison/
             summary_common_windows.csv + <results>/T1.6_stats/T3.9_movielens/
      2 x 2 bars (NDCG@10, rho, RSI@10, Delta Rank log), groups daily/weekly.

  T3.10_si_movielens_rsi_time  (task T3.10, SI and thesis)
      Window-by-window RSI@10 on MovieLens, rolling 28 days / 13 weeks.

  Tables of T3.10 (LaTeX) come from scripts/generate_revision_tables.py and the
  thesis-only figures from scripts/generate_thesis_figures.py; both import the
  loaders v4_values / v4_tests / v4_window_series of this file.

Usage (Windows, from the project root)
--------------------------------------
  python scripts\generate_revision_figures.py --results results\revision_v5 ^
         --out "D:\Research\Thesis Research\Articles\Popularity With Wavelet\03 - Popularity with Wavelets\Submit Paper\Scientific Reports\Revisions\V4\Response\Figures"

Commands: Revisions/V4/Response/Runbooks/RUN_T2.3_T2.5.md
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- common style
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
GREY = '#9a9994'
SCENARIOS = [('youtube_hourly', 'YouTube (hourly)'),
             ('taxi_hourly', 'NYC Taxi (hourly)'),
             ('taxi_30min', 'NYC Taxi (30 min)'),
             ('taxi_5min', 'NYC Taxi (5 min)')]


def style_axes(ax):
    ax.set_facecolor(SURF)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color('#b5b4ae')
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [f'{name}.pdf', f'{name}.png']


# ------------------------------------------------------------ Pareto helpers
def pareto_flags(df, up, down=()):
    """True where no other row is >= on all criteria and > on one."""
    v = df[list(up) + list(down)].to_numpy(dtype=float).copy()
    if down:
        v[:, len(up):] *= -1.0
    flags = []
    for i in range(len(v)):
        o = np.delete(v, i, axis=0)
        flags.append(not np.any(np.all(o >= v[i], axis=1) & np.any(o > v[i], axis=1)))
    return np.array(flags)


def staircase(df, x, y):
    f = df[pareto_flags(df, [x, y])].sort_values(x)
    xs, ys = f[x].to_numpy(), f[y].to_numpy()
    sx, sy = [], []
    for i in range(len(xs)):
        if i:
            sx.append(xs[i]); sy.append(ys[i - 1])
        sx.append(xs[i]); sy.append(ys[i])
    return sx, sy


# ----------------------------------------------------------- figure T2.3
T23_X = 'rsi@10_mean'
T23_ROWS = [('ndcg@10_mean', 'NDCG@10'), ('spearman_rho_mean', r'Spearman $\rho$')]
T23_ROB = 'robustness_distortion_mean'
T23_HIGHLIGHT = {                  # colour + marker, so colour is never alone
    'WSPI':        ('#2a78d6', 'o', 2.4, 9),
    'SMA':         ('#eb6834', 's', 1.6, 7),
    'DTCWT+AF':    ('#1baf7a', '^', 1.6, 7),
    'CompoundPop': ('#4a3aa7', 'D', 1.6, 6),
}
T23_VARIANTS = {'main': lambda m: m not in ('PFRF', 'Holt'),
                'full': lambda m: True}


def _t23_draw(d, variant, out):
    keep = T23_VARIANTS[variant]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    rows = []
    for c, (sc, title) in enumerate(SCENARIOS):
        s = d[(d.scenario == sc) & d.method.map(keep)].copy()
        if s.empty:
            raise ValueError(f'scenario {sc} missing in sweep_summary.csv')
        s['pareto4'] = pareto_flags(s, ['ndcg@10_mean', 'spearman_rho_mean', T23_X], [T23_ROB])
        for r, (y, ylab) in enumerate(T23_ROWS):
            s[f'pareto2_{y}'] = pareto_flags(s, [T23_X, y])
            ax = axes[r, c]
            style_axes(ax)
            fx, fy = staircase(s, T23_X, y)
            ax.plot(fx, fy, ls='--', lw=1.1, color=INK2, zorder=1)
            order = [m for m in s.method.unique() if m not in T23_HIGHLIGHT] + \
                    [m for m in T23_HIGHLIGHT if m in set(s.method)]
            for m in order:
                g = s[s.method == m].sort_values('window')
                if m in T23_HIGHLIGHT:
                    col, mk, lw, ms = T23_HIGHLIGHT[m]
                    z = 4 if m == 'WSPI' else 3
                else:
                    col, mk, lw, ms, z = GREY, '.', 0.9, 6, 2
                ax.plot(g[T23_X], g[y], color=col, lw=lw, marker=mk, ms=ms,
                        mec=SURF, mew=1.0, zorder=z)
                if m in T23_HIGHLIGHT:
                    for _, p in g.iterrows():
                        ax.annotate(str(int(p.window)), (p[T23_X], p[y]), xytext=(4, 4),
                                    textcoords='offset points', fontsize=6.5, color=INK2, zorder=6)
                else:
                    last = g.iloc[-1]
                    ax.annotate(m, (last[T23_X], last[y]), xytext=(3, -9),
                                textcoords='offset points', fontsize=7, color=INK2)
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK)
            else:
                ax.set_xlabel('RSI@10 (higher = more stable)', color=INK2)
            if c == 0:
                ax.set_ylabel(ylab + ' (higher = more accurate)', color=INK2)
        s['variant'] = variant
        rows.append(s[['variant', 'scenario', 'method', 'window', 'ndcg@10_mean',
                       'spearman_rho_mean', T23_X, T23_ROB, 'n_windows',
                       'pareto2_ndcg@10_mean', 'pareto2_spearman_rho_mean', 'pareto4']])
    h = [plt.Line2D([], [], color=v[0], marker=v[1], lw=v[2], ms=v[3] * 0.85, label=m)
         for m, v in T23_HIGHLIGHT.items()]
    h += [plt.Line2D([], [], color=GREY, marker='.', lw=0.9, label='other methods (name at N=128)'),
          plt.Line2D([], [], color=INK2, ls='--', lw=1.1, label='2-D Pareto frontier of the panel')]
    fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    shown = 'all 12 methods' if variant == 'full' else 'PFRF and Holt not shown'
    fig.suptitle('Accuracy-stability trade-off over window length N (small numbers = N); '
                 f'mean over common windows >= 32; {shown}', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    files = save(fig, out, f'T2.3_tradeoff_{variant}')
    return pd.concat(rows), files


def fig_t23_tradeoff(results, out):
    src = results / 'T2.2_window_sweep' / 'sweep_summary.csv'
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    parts, files = [], []
    for v in T23_VARIANTS:
        p, f = _t23_draw(d, v, out)
        parts.append(p); files += f
    csv_dir = results / 'T2.3_tradeoff'
    (csv_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    pts = pd.concat(parts)
    pts.to_csv(csv_dir / 'pareto_points.csv', index=False)
    meta = {'task': 'T2.3', 'date_utc': datetime.now(timezone.utc).isoformat(),
            'input': str(src), 'figures_dir': str(out), 'figures': files,
            'variants': {'main': 'without PFRF and Holt', 'full': 'all methods'},
            'pareto_2d': 'per panel: RSI@10 and the y metric, higher is better, over methods shown',
            'pareto_4': 'CSV only: NDCG@10, spearman_rho, RSI@10 up; robustness_distortion down',
            'python': platform.python_version(), 'pandas': pd.__version__,
            'matplotlib': matplotlib.__version__}
    (csv_dir / 'metadata' / 'generate_revision_figures_run.json').write_text(json.dumps(meta, indent=2))
    w = pts[(pts.method == 'WSPI') & (pts.window == 64)]
    note = '; '.join(f"{r.variant}/{r.scenario}: NDCG-RSI {'on' if r['pareto2_ndcg@10_mean'] else 'off'}, "
                     f"rho-RSI {'on' if r['pareto2_spearman_rho_mean'] else 'off'}" for _, r in w.iterrows())
    return files, 'WSPI(64) frontier -> ' + note



# ----------------------------------------------------------- figure T2.2 (SI)
T22_ROWS = [('ndcg@10_mean', 'NDCG@10 (higher = more accurate)'),
            ('rsi@10_mean', 'RSI@10 (higher = more stable)')]
T22_N = [7, 16, 32, 64, 128]


def fig_t22_window_curves(results, out):
    """NDCG@10 and RSI@10 against the window length N for every method."""
    src = results / 'T2.2_window_sweep' / 'sweep_summary.csv'
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    d = d[~d.method.isin(['PFRF', 'Holt'])]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for c, (sc, title) in enumerate(SCENARIOS):
        s = d[d.scenario == sc]
        if s.empty:
            raise ValueError(f'scenario {sc} missing in sweep_summary.csv')
        for r, (y, ylab) in enumerate(T22_ROWS):
            ax = axes[r, c]
            style_axes(ax)
            order = [m for m in s.method.unique() if m not in T23_HIGHLIGHT] + \
                    [m for m in T23_HIGHLIGHT if m in set(s.method)]
            for m in order:
                g = s[s.method == m].sort_values('window')
                if m in T23_HIGHLIGHT:
                    col, mk, lw, ms = T23_HIGHLIGHT[m]
                    z = 4 if m == 'WSPI' else 3
                else:
                    col, mk, lw, ms, z = GREY, '.', 0.9, 6, 2
                ax.plot(g.window, g[y], color=col, lw=lw, marker=mk, ms=ms,
                        mec=SURF, mew=1.0, zorder=z)
                if m not in T23_HIGHLIGHT:
                    last = g.iloc[-1]
                    ax.annotate(m, (last.window, last[y]), xytext=(4, -3),
                                textcoords='offset points', fontsize=7, color=INK2)
            ax.set_xscale('log', base=2)
            ax.set_xticks(T22_N)
            ax.set_xticklabels([str(n) for n in T22_N])
            ax.set_xlim(6, 200)
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK)
            else:
                ax.set_xlabel('window length N (slots)', color=INK2)
            if c == 0:
                ax.set_ylabel(ylab, color=INK2)
    h = [plt.Line2D([], [], color=v[0], marker=v[1], lw=v[2], ms=v[3] * 0.85, label=m)
         for m, v in T23_HIGHLIGHT.items()]
    h += [plt.Line2D([], [], color=GREY, marker='.', lw=0.9, label='other methods (name at N=128)')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Accuracy and stability against window length N; wavelet-based methods from N=16; '
                 'mean over common windows >= 32; PFRF and Holt not shown', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    files = save(fig, out, 'T2.2_window_curves')
    w = s = d[(d.method.isin(['WSPI', 'SMA']))]
    rng = '; '.join(f"{sc}: WSPI NDCG {g[g.method=='WSPI']['ndcg@10_mean'].min():.4f}-"
                    f"{g[g.method=='WSPI']['ndcg@10_mean'].max():.4f}, SMA NDCG "
                    f"{g[g.method=='SMA']['ndcg@10_mean'].min():.4f}-{g[g.method=='SMA']['ndcg@10_mean'].max():.4f}"
                    for sc, g in w.groupby('scenario'))
    return files, rng



# ----------------------------------------------------------- figures T2.4 (E11)
T24_ROOT = 'T2.4_responsiveness'
T24_EXAMPLE_SCEN = [('youtube_hourly', 'YouTube (hourly)'), ('taxi_hourly', 'NYC Taxi (hourly)')]
T24_EXAMPLE_KIND = [('typical', 'typical entry'), ('worst_for_reference', 'worst case for WSPI')]
T24_LINES = [                      # (config, method, label, colour, marker, lw)
    ('default', 'WSPI', 'WSPI (N=64)', '#2a78d6', 'o', 2.2),
    ('default', 'AF', 'AF (N=7)', '#1baf7a', 's', 1.5),
    ('default', 'DTCWT+AF', 'DTCWT+AF (N=64)', '#4a3aa7', '^', 1.5),
    ('equal64', 'RRD', 'RRD (N=64)', '#eb6834', 'D', 1.5),
]
T24_RANK_CAP = 100


def fig_t24_surge_examples(results, out):
    """Main-text figure of T2.4: the two automatically chosen entries of
    YouTube and taxi hourly (rule in evaluation/responsiveness.select_examples).
    Row 1: real count of the item (band = slots in the true Top-10).
    Row 2: rank of the item for four methods (log axis, 1 at the top); gaps =
    item not eligible for the method; dashed line = rank 10."""
    root = results / T24_ROOT
    cols = []
    for sc, title in T24_EXAMPLE_SCEN:
        tf = root / sc / 'examples' / 'example_traces.csv'
        ef = root / sc / 'examples' / 'example_events.csv'
        if not (tf.exists() and ef.exists()):
            return None, f'input missing: {tf}'
        tr, ev = pd.read_csv(tf), pd.read_csv(ef)
        for kind, klab in T24_EXAMPLE_KIND:
            e = ev[ev.example == kind]
            if e.empty:
                return None, f'example {kind} missing in {ef}'
            cols.append((title, klab, e.iloc[0], tr[tr.example == kind]))
    fig, axes = plt.subplots(2, len(cols), figsize=(4.3 * len(cols), 6.6),
                             gridspec_kw={'height_ratios': [1, 1.25]}, sharex='col')
    notes = []
    for c, (title, klab, e, t) in enumerate(cols):
        t0, R = int(e.t0), int(e.run_len)
        base = t[(t.config == 'default') & (t.method == 'WSPI')].sort_values('window_id')
        x = base.window_id.to_numpy() - t0
        ax = axes[0, c]
        style_axes(ax)
        ax.axvspan(-0.5, R - 0.5, color='#e9e6f7', zorder=0, lw=0)
        ax.plot(x, base['count'], color=INK, lw=1.3, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8, ls=':')
        ax.set_title(f'{title}: {klab}\n'
                     f'item {e.item_id}; delay WSPI {int(e.delay_WSPI)}, AF {int(e.delay_AF)} slots',
                     fontsize=9.5, color=INK)
        if c == 0:
            ax.set_ylabel('real count per slot', color=INK2)
        ax2 = axes[1, c]
        style_axes(ax2)
        ax2.axvspan(-0.5, R - 0.5, color='#e9e6f7', zorder=0, lw=0)
        ax2.axhline(10, color=INK2, lw=0.9, ls='--', zorder=1)
        ax2.axvline(0, color=INK2, lw=0.8, ls=':')
        tr_rank = base['truth_rank'].clip(upper=T24_RANK_CAP)
        ax2.plot(x, tr_rank, color=GREY, lw=1.0, ls='-', zorder=2)
        for cfg, m, lab, col, mk, lw in T24_LINES:
            g = t[(t.config == cfg) & (t.method == m)].sort_values('window_id')
            ax2.plot(g.window_id - t0, g['rank'].clip(upper=T24_RANK_CAP), color=col, lw=lw,
                     marker=mk, ms=3.2, mec=SURF, mew=0.5, zorder=4 if m == 'WSPI' else 3)
        ax2.set_yscale('log')
        ax2.set_ylim(T24_RANK_CAP * 1.15, 0.85)
        ax2.set_yticks([1, 3, 10, 30, 100])
        ax2.set_yticklabels(['1', '3', '10', '30', '100+'])
        ax2.set_xlabel('slots from entry (t0 = 0)', color=INK2)
        if c == 0:
            ax2.set_ylabel('rank of the item (log)', color=INK2)
        notes.append(f"{title}/{klab}: item {e.item_id}, t0={t0}, run={R}, "
                     f"delay WSPI {int(e.delay_WSPI)}, AF {int(e.delay_AF)}")
    h = [plt.Line2D([], [], color=v[3], marker=v[4], lw=v[5], ms=4, label=v[2]) for v in T24_LINES]
    h += [plt.Line2D([], [], color=GREY, lw=1.0, label='true rank'),
          plt.Line2D([], [], color=INK2, ls='--', lw=0.9, label='Top-10 boundary'),
          Patch(color='#e9e6f7', label='item in the true Top-10')]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    files = save(fig, out, 'T2.4_surge_examples')
    return files, '; '.join(notes)


def fig_t24_delay_ecdf(results, out):
    """SI figure of T2.4: share of entries that each method has brought into its
    Top-10 within d slots (main variant).  The plateau is 1 - miss rate.
    Row 1: default configuration (baselines 7, wavelet-based 64); row 2: all
    methods with a 64-slot window."""
    root = results / T24_ROOT
    cfgs = [('default', 'default windows (baselines 7, wavelet-based 64)'),
            ('equal64', 'equal window N = 64')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=True)
    dmax = 24
    for c, (sc, title) in enumerate(SCENARIOS):
        for r, (cfg, clab) in enumerate(cfgs):
            f = root / sc / cfg / 'delays.csv'
            if not f.exists():
                return None, f'input missing: {f}'
            d = pd.read_csv(f)
            ax = axes[r, c]
            style_axes(ax)
            n = d.event_id.nunique()
            hl = {l[1]: (l[3], l[4], l[5]) for l in T24_LINES}
            ms = [m for m in d.method.unique() if m not in hl] + [m for m in hl if m in set(d.method)]
            for m in ms:
                g = d[d.method == m]
                xs = np.arange(0, dmax + 1)
                ys = [(g.detected & (g.delay <= k)).sum() / n for k in xs]
                if m in hl:
                    col, mk, lw = hl[m]
                    ax.step(xs, ys, where='post', color=col, lw=lw, zorder=4 if m == 'WSPI' else 3)
                else:
                    ax.step(xs, ys, where='post', color=GREY, lw=0.9, zorder=2)
                    ax.annotate(m, (dmax, ys[-1]), xytext=(2, -3), textcoords='offset points',
                                fontsize=6.5, color=INK2)
            ax.set_xlim(0, dmax + 4)
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(f'{title}  ({n} entries)', fontsize=10.5, color=INK)
            else:
                ax.set_xlabel('delay d (slots after entry)', color=INK2)
            if c == 0:
                ax.set_ylabel(f'share of entries in Top-10 by d\n{clab}', color=INK2, fontsize=9)
    h = [plt.Line2D([], [], color=v[3], lw=v[5], label=v[1]) for v in T24_LINES]
    h += [plt.Line2D([], [], color=GREY, lw=0.9, label='other methods')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Delay to bring a genuine entry into the Top-10 (entry: >= 6 slots in the true '
                 'Top-10 after >= 6 slots outside); plateau = 1 - miss rate', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, out, 'T2.4_delay_ecdf'), 'ok'


# ------------------------------------------------------------ figure T3.2 (E3)
T32_ROOT = 'T3.2_param_grid'
T32_ROWS = [('ndcg@10_mean', 'NDCG@10', False, '{:.4f}'),
            ('rsi@10_mean', 'RSI@10', False, '{:.4f}'),
            ('robustness_distortion_mean', 'robustness $\\Delta$Rank', True, '{:.1f}')]
T32_LIGHT, T32_DARK = '#eef4fb', '#1c4f8f'


def fig_t32_param_heatmap(results, out):
    """alpha x beta heat maps of WSPI on the test part (SI figure)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    src = results / T32_ROOT / 'grid_summary.csv'
    sel = results / T32_ROOT / 'selection' / 'selection.csv'
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    d = d[d.part == 'test']
    chosen = {}
    if sel.exists():
        sv = pd.read_csv(sel)
        sv = sv[sv.param == 'alpha_beta']
        chosen = {r.scenario: (r.selected_alpha, r.selected_beta) for r in sv.itertuples()}
    scen = [(k, t) for k, t in SCENARIOS if k in set(d.scenario)]
    if not scen:
        return None, 'no scenario in grid_summary.csv'
    cmap = LinearSegmentedColormap.from_list('t32', [T32_LIGHT, T32_DARK])
    fig, axes = plt.subplots(len(T32_ROWS), len(scen), figsize=(4.3 * len(scen), 11.2),
                             squeeze=False)
    notes = []
    for c, (sc, title) in enumerate(scen):
        s = d[d.scenario == sc]
        al = sorted(s.alpha.unique())
        be = sorted(s.beta.unique())
        for r, (col, lab, low_better, fmt) in enumerate(T32_ROWS):
            ax = axes[r, c]
            M = s.pivot(index='beta', columns='alpha', values=col).reindex(index=be, columns=al)
            v = M.to_numpy(dtype=float)
            vmin, vmax = np.nanmin(v), np.nanmax(v)
            z = (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)
            if low_better:
                z = 1.0 - z
            ax.imshow(z, cmap=cmap, vmin=0, vmax=1, origin='lower', aspect='equal')
            for i in range(len(be)):
                for j in range(len(al)):
                    ax.text(j, i, fmt.format(v[i, j]), ha='center', va='center', fontsize=5.6,
                            color=SURF if z[i, j] > 0.6 else INK)
            ax.set_xticks(range(len(al)))
            ax.set_xticklabels([f'{x:g}' for x in al], fontsize=7.5, color=INK2)
            ax.set_yticks(range(len(be)))
            ax.set_yticklabels([f'{x:g}' for x in be], fontsize=7.5, color=INK2)
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.tick_params(length=0)
            j0, i0 = al.index(1.0), be.index(1.0)
            ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), 1, 1, fill=False, ec=INK, lw=1.8))
            if sc in chosen:
                ja, ib = al.index(chosen[sc][0]), be.index(chosen[sc][1])
                ax.plot(ja, ib, marker='o', ms=19, mfc='none', mec=INK, mew=1.4)
            ax.set_title((f'{title}\n' if r == 0 else '') +
                         f'{lab}: {fmt.format(vmin)} to {fmt.format(vmax)}',
                         fontsize=9.5 if r else 10.5, color=INK)
            if r == len(T32_ROWS) - 1:
                ax.set_xlabel(r'$\alpha$ (weight of R)', color=INK2)
            if c == 0:
                ax.set_ylabel(r'$\beta$ (weight of $W_E$)', color=INK2)
        n = int(s.n_windows.iloc[0])
        notes.append(f'{sc}: {n} test windows')
    h = [plt.Line2D([], [], marker='s', ms=10, mfc='none', mec=INK, lw=0, mew=1.8,
                    label=r'default $\alpha=\beta=1$'),
         plt.Line2D([], [], marker='o', ms=10, mfc='none', mec=INK, lw=0, mew=1.4,
                    label='chosen on the first 30 % (tuning part)')]
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(r'WSPI ($N=64$, $J=3$) over the $\alpha\times\beta$ grid, mean over the test part '
                 '(last 70 % of the common windows); darker = better within each panel',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return save(fig, out, 'T3.2_param_heatmap'), '; '.join(notes)


# ----------------------------------------------------------- figure T3.5 (E4)
T35_SIZES = [2, 5, 10, 20, 50]
T35_ROWS = [('default', 'default: baselines N=7, wavelet-based N=64'),
            ('eq64', 'equal window: all methods N=64')]
T35_HIGHLIGHT = {                 # method: (colour, marker, lw, ms)
    'WSPI':     ('#2a78d6', 'o', 2.4, 8),
    'DTCWT+AF': ('#1baf7a', '^', 1.6, 7),
    'RRD':      ('#eb6834', 'D', 1.6, 6),
    'AF':       ('#4a3aa7', 's', 1.6, 6),
}


def fig_t35_spike_size(results, out):
    """Delta Rank against spike size, common perturbation (E4)."""
    src = results / 'T3.5_robustness' / 'robustness_summary.csv'
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    d = d[d.condition.isin([f'size{z}' for z in T35_SIZES])].copy()
    d['size'] = d.condition.str[4:].astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharex=True)
    missing, notes = [], []
    for c, (sc, title) in enumerate(SCENARIOS):
        for r, (cfg, rlab) in enumerate(T35_ROWS):
            ax = axes[r, c]
            style_axes(ax)
            g = d[(d.scenario == sc) & (d.config == cfg)]
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK)
            if c == 0:
                ax.set_ylabel(f'{rlab}\n\u0394Rank (log scale)', color=INK2, fontsize=9)
            if r == 1:
                ax.set_xlabel('spike size (x mean of the last 64 slots)', color=INK2)
            if g.empty:
                missing.append(f'{sc}/{cfg}')
                ax.text(0.5, 0.5, 'no data yet', transform=ax.transAxes, ha='center',
                        color=INK2)
                continue
            order = [m for m in g.method.unique() if m not in T35_HIGHLIGHT] + \
                    [m for m in T35_HIGHLIGHT if m in set(g.method)]
            for m in order:
                h = g[g.method == m].sort_values('size')
                y = h['dr_mean'].clip(lower=0.1)
                if m in T35_HIGHLIGHT:
                    col, mk, lw, ms = T35_HIGHLIGHT[m]
                    z = 4 if m == 'WSPI' else 3
                else:
                    col, mk, lw, ms, z = GREY, '.', 0.9, 6, 2
                ax.plot(h['size'], y, color=col, lw=lw, marker=mk, ms=ms,
                        mec=SURF, mew=1.0, zorder=z)
                if m not in T35_HIGHLIGHT:
                    ax.annotate(m, (h['size'].iloc[-1], y.iloc[-1]), xytext=(4, -3),
                                textcoords='offset points', fontsize=7, color=INK2)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(T35_SIZES)
            ax.set_xticklabels([str(z) for z in T35_SIZES])
            ax.set_xlim(1.7, 75)
            w = g[(g.method == 'WSPI') & (g['size'] == 10)]
            if len(w):
                notes.append(f"{sc}/{cfg}: WSPI 10x {float(w['dr_mean'].iloc[0]):.2f}")
    h = [plt.Line2D([], [], color=v[0], marker=v[1], lw=v[2], ms=v[3] * 0.85, label=m)
         for m, v in T35_HIGHLIGHT.items()]
    h += [plt.Line2D([], [], color=GREY, marker='.', lw=0.9, label='other methods (name at 50x)')]
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Rank displacement of 50 low-activity items against spike size (spike in the last '
                 'slot, one slot); same items and spike for every method; mean of 5 seeds',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    files = save(fig, out, 'T3.5_spike_size')
    msg = '; '.join(notes) + (f'; MISSING: {", ".join(missing)}' if missing else '')
    return files, msg


# ------------------------------------------------------------------ T3.6 (E5)
T36_BANDS = [('cv_E_L', 'lowpass'), ('cv_E_1', 'level 1'), ('cv_E_2', 'level 2'),
             ('cv_E_3', 'level 3')]
T36_TR = {'DTCWT': '#2a78d6', 'DWT': '#b5b4ae'}
T36_METHODS = {                    # method: (colour, marker, line style)
    'WSPI':     ('#2a78d6', 'o', '-'),
    'DWT-WSPI': ('#2a78d6', 'o', ':'),
    'DTCWT+AF': ('#1baf7a', '^', '-'),
    'DWT+AF':   ('#1baf7a', '^', ':'),
    'SMA':      ('#eb6834', 's', '-'),
    'EWMA-eq':  ('#4a3aa7', 'D', '-'),
}
T36_SETTINGS = [('spike', 'middle'), ('burst3', 'middle'), ('spike', 'recent'), ('burst3', 'recent')]


def fig_t36_shift_invariance(results, out):
    """Controlled shift-invariance test (E5): real data (circular shift) and a
    synthetic moving event (settings of the index)."""
    root = results / 'T3.6_shift_invariance'
    src_a = root / 'shift_summary.csv'
    syn = root / 'synthetic'
    if not src_a.exists() and not (syn / 'synthetic_summary.csv').exists():
        return None, f'input missing: {src_a} and {syn}'
    fig = plt.figure(figsize=(17, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.28)
    missing, notes = [], []
    # ---- row 1: real data, circular shift
    a = pd.read_csv(src_a) if src_a.exists() else pd.DataFrame()
    for c, (sc, title) in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[0, c])
        style_axes(ax)
        ax.set_title(f'({chr(97 + c)}) {title}', fontsize=10.5, color=INK)
        g = a[a['scenario'] == sc] if len(a) else a
        if g.empty:
            missing.append(sc)
            ax.text(0.5, 0.5, 'no data yet', transform=ax.transAxes, ha='center', color=INK2)
            continue
        x = np.arange(len(T36_BANDS))
        for j, (tr, col) in enumerate(T36_TR.items()):
            y = [float(g[(g['transform'] == tr) & (g['metric'] == m)]['mean'].iloc[0]) for m, _ in T36_BANDS]
            ax.bar(x + (j - 0.5) * 0.38, y, width=0.36, color=col, label=tr, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([b for _, b in T36_BANDS], fontsize=8)
        if c == 0:
            ax.set_ylabel('CV of band energy over\ncircular shifts s = 0..7', color=INK2, fontsize=9)
        r3 = g[(g['metric'] == 'cv_E_3')]
        if len(r3) == 2:
            notes.append(f"{sc}: cv_E_3 DTCWT {float(r3[r3['transform'] == 'DTCWT']['mean'].iloc[0]):.3f}, "
                         f"DWT {float(r3[r3['transform'] == 'DWT']['mean'].iloc[0]):.3f}")
    # ---- row 2 left: example (repetition 0, burst3, recent)
    ax = fig.add_subplot(gs[1, 0:2])
    style_axes(ax)
    ax.set_title('(e) Synthetic 3-slot burst moving towards the present (recent part of the '
                 'window; repetition 0)', fontsize=10, color=INK)
    f = syn / 'synthetic_scores.csv'
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d['repetition'] == 0) & (d['shape'] == 'burst3') & (d['age'] == 'recent')]
        for m, (col, mk, ls) in T36_METHODS.items():
            h = d[d['method'] == m].sort_values('shift')
            if h.empty:
                continue
            y = np.log(h['score'].to_numpy() / h['score'].iloc[0])
            ax.plot(h['shift'], y, color=col, marker=mk, ls=ls, lw=1.6, ms=5, label=m, zorder=3)
        ax.axhline(0, color='#b5b4ae', lw=0.8, zorder=1)
        ax.set_xlabel('shift s (slots towards the newest slot)', color=INK2)
        ax.set_ylabel('log(score(s) / score(0))', color=INK2, fontsize=9)
        ax.legend(ncol=3, frameon=False, fontsize=8, loc='upper left')
    else:
        missing.append('synthetic_scores')
    # ---- row 2 right: share of wrong-direction steps
    ax = fig.add_subplot(gs[1, 2:4])
    style_axes(ax)
    ax.set_title('(f) Share of steps in which the score falls although the event got newer '
                 '(mean, 95% CI, 500 repetitions)', fontsize=10, color=INK)
    f = syn / 'synthetic_summary.csv'
    if f.exists():
        s = pd.read_csv(f)
        s = s[(s['kind'] == 'score') & (s['metric'] == 'share_wrong')]
        x = np.arange(len(T36_SETTINGS))
        w = 0.8 / len(T36_METHODS)
        for j, (m, (col, mk, ls)) in enumerate(T36_METHODS.items()):
            y, lo, hi = [], [], []
            for shp, age in T36_SETTINGS:
                r = s[(s['shape'] == shp) & (s['age'] == age) & (s['name'] == m)]
                y.append(float(r['mean'].iloc[0]) if len(r) else np.nan)
                lo.append(float(r['ci_low'].iloc[0]) if len(r) else np.nan)
                hi.append(float(r['ci_high'].iloc[0]) if len(r) else np.nan)
            y, lo, hi = map(np.asarray, (y, lo, hi))
            xx = x + (j - (len(T36_METHODS) - 1) / 2) * w
            ax.bar(xx, y, width=w * 0.92, color=col, alpha=1.0 if ls == '-' else 0.45,
                   hatch=None if ls == '-' else '//', edgecolor=SURF, label=m, zorder=3)
            ax.errorbar(xx, y, yerr=[y - lo, hi - y], fmt='none', ecolor=INK2, lw=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{"spike" if a_ == "spike" else "3-slot burst"}\n{b_} part'
                            for a_, b_ in T36_SETTINGS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('share of the 8 steps', color=INK2, fontsize=9)
        ax.legend(ncol=3, frameon=False, fontsize=8, loc='upper left')
        r = s[(s['shape'] == 'burst3') & (s['age'] == 'recent')]
        notes.append('burst3/recent share_wrong: ' + ', '.join(
            f"{n} {float(v):.3f}" for n, v in zip(r['name'], r['mean'])))
    else:
        missing.append('synthetic_summary')
    h = [Patch(color=c, label=t) for t, c in T36_TR.items()]
    fig.legend(handles=h, loc='upper right', ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle('Controlled shift test: (a-d) real 64-slot windows shifted circularly '
                 '(content fixed); (e-f) synthetic event with the settings of each method',
                 fontsize=11, color=INK, x=0.45)
    files = save(fig, out, 'T3.6_shift_invariance')
    msg = '; '.join(notes) + (f'; MISSING: {", ".join(missing)}' if missing else '')
    return files, msg


def fig_t37_feature_relation(results, out):
    """R against WE over all WSPI item-windows (E7): density of the item-windows
    and the bounds that the identity WE log2(J+1) = h(R) + (1-R) H(q) puts on WE
    for a given R (J = 3)."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    root = results / 'T3.7_feature_relation'
    have = [(sc, t) for sc, t in SCENARIOS if (root / sc / 'density.csv').exists()]
    if not have:
        return None, f'input missing: {root}/<scenario>/density.csv'
    cmap = LinearSegmentedColormap.from_list('t37', ['#eaf2fc', '#2a78d6', '#0d2f5c'])
    r = np.linspace(1e-6, 1 - 1e-6, 400)
    h = -(r * np.log2(r) + (1 - r) * np.log2(1 - r))
    lower, upper = h / 2.0, (h + (1 - r) * np.log2(3.0)) / 2.0
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4), sharey=True)
    notes, missing = [], []
    for c, (sc, title) in enumerate(SCENARIOS):
        ax = axes[c]
        style_axes(ax)
        ax.set_title(f'({chr(97 + c)}) {title}', fontsize=10.5, color=INK)
        f = root / sc / 'density.csv'
        if not f.exists():
            missing.append(sc)
            ax.text(0.5, 0.5, 'no data yet', transform=ax.transAxes, ha='center', color=INK2)
            continue
        d = pd.read_csv(f)
        nb = int(round(1.0 / float((d['r_hi'] - d['r_lo']).iloc[0])))
        H = np.full((nb, nb), np.nan)
        i = np.clip(np.round(d['r_lo'].to_numpy() * nb).astype(int), 0, nb - 1)
        j = np.clip(np.round(d['we_lo'].to_numpy() * nb).astype(int), 0, nb - 1)
        H[j, i] = d['count'].to_numpy()
        edges = np.linspace(0, 1, nb + 1)
        pm = ax.pcolormesh(edges, edges, H, cmap=cmap, shading='flat',
                           norm=LogNorm(vmin=1, vmax=np.nanmax(H)), zorder=2)
        ax.plot(r, lower, color=INK, lw=1.2, zorder=3,
                label='lower bound: detail energy in one band')
        ax.plot(r, upper, color=INK, lw=1.2, ls='--', zorder=3,
                label='upper bound: detail energy spread evenly')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('R (share of energy in the trend band)', color=INK2)
        if c == 0:
            ax.set_ylabel('WE (normalised wavelet entropy)', color=INK2)
        ps = root / sc / 'pooled_summary.csv'
        if ps.exists():
            q = pd.read_csv(ps).iloc[0]
            ax.text(0.97, 0.97, f"Spearman {q['sp_R_WE']:.3f}\n$\\eta^2$(WE | R) {q['eta2_WE_given_R']:.3f}\n"
                    f"n = {int(q['n_item_windows']):,}", transform=ax.transAxes, ha='right', va='top',
                    fontsize=8.5, color=INK, bbox=dict(fc=SURF, ec=GRID, lw=0.6))
            notes.append(f"{sc}: sp {q['sp_R_WE']:.4f}, eta2 {q['eta2_WE_given_R']:.4f}")
        cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=INK2)
        if c == 3:
            cb.set_label('item-windows per cell (log)', color=INK2, fontsize=8)
    axes[0].legend(loc='lower left', frameon=False, fontsize=8)
    fig.suptitle('R against WE over all WSPI item-windows (N = 64, J = 3); lines: the range of WE '
                 'for a given R, from WE log2(J+1) = h(R) + (1-R) H(q)', fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    files = save(fig, out, 'T3.7_feature_relation')
    msg = '; '.join(notes) + (f'; MISSING: {", ".join(missing)}' if missing else '')
    return files, msg


T38_STYLE = {   # method: colour, marker, line style, line width
    'WSPI':        ('#2a78d6', 'o', '-', 2.4),
    'DTCWT+AF':    ('#1baf7a', '^', '-', 1.6),
    'DWT+AF':      ('#1baf7a', 'v', ':', 1.6),
    'AF':          ('#6f6e69', 's', '-', 1.1),
    'EWMA':        ('#6f6e69', 'D', '--', 1.1),
    'RRD':         ('#9a9994', 'P', '-', 1.1),
    'VSE':         ('#9a9994', 'X', '--', 1.1),
    'CompoundPop': ('#b9b8b2', '*', '-', 1.1),
    'PFRF':        ('#b9b8b2', 'h', '--', 1.1),
}
T38_BASE = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF']


def fig_t38_runtime(results, out):
    """Measured cost of the nine protocol-V5 scorers (E9), one CPU core.
    (a) time to score all M items against M, default windows (baselines 7,
    wavelet-based 64); (b) microseconds per item-window against N, M = 1e4;
    (c) traced working memory per item against N, batch 1e4."""
    root = results / 'T3.8_runtime'
    gf, tf = root / 'bench' / 'runtime_grid.csv', root / 'memory' / 'tracemalloc_grid.csv'
    if not gf.exists():
        return None, f'input missing: {gf}'
    g = pd.read_csv(gf)
    t = pd.read_csv(tf) if tf.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    notes = []
    ax = axes[0]
    style_axes(ax)
    for m, (col, mk, ls, lw) in T38_STYLE.items():
        nd = 7 if m in T38_BASE else 64
        d = g[(g['method'] == m) & (g['N'] == nd)].sort_values('M')
        if d.empty:
            continue
        ax.plot(d['M'], d['median_s'], color=col, marker=mk, ls=ls, lw=lw, ms=5,
                label=f'{m} (N={nd})', zorder=3 if m == 'WSPI' else 2)
        ax.fill_between(d['M'], d['q1_s'], d['q3_s'], color=col, alpha=0.15, lw=0)
        big = d[d['M'] == d['M'].max()]
        if len(big):
            notes.append(f"{m}: {float(big['median_s'].iloc[0]):.3g} s for M={int(big['M'].iloc[0]):,}")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of items M (log)', color=INK2)
    ax.set_ylabel('time to score all items, one window (s, log)', color=INK2)
    ax.set_title('(a) Time against the number of items', fontsize=10.5, color=INK)
    ax.legend(loc='upper left', frameon=False, fontsize=7.5)
    ax = axes[1]
    style_axes(ax)
    Mb = 10_000 if (g['M'] == 10_000).any() else int(g['M'].max())
    for m, (col, mk, ls, lw) in T38_STYLE.items():
        d = g[(g['method'] == m) & (g['M'] == Mb)].sort_values('N')
        if d.empty:
            continue
        ax.plot(d['N'], d['us_per_item_median'], color=col, marker=mk, ls=ls, lw=lw, ms=5, label=m,
                zorder=3 if m == 'WSPI' else 2)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('window length N (log2)', color=INK2)
    ax.set_ylabel('microseconds per item-window (log)', color=INK2)
    ax.set_title(f'(b) Cost per item against N (M = {Mb:,})', fontsize=10.5, color=INK)
    ax = axes[2]
    style_axes(ax)
    if t is not None:
        Bb = 10_000 if (t['batch'] == 10_000).any() else int(t['batch'].max())
        for m, (col, mk, ls, lw) in T38_STYLE.items():
            d = t[(t['method'] == m) & (t['batch'] == Bb)].sort_values('N')
            if d.empty:
                continue
            ax.plot(d['N'], d['traced_peak_bytes_per_item'], color=col, marker=mk, ls=ls, lw=lw, ms=5,
                    label=m, zorder=3 if m == 'WSPI' else 2)
        c = t[(t['method'] == 'WSPI') & (t['batch'] == Bb)].sort_values('N')
        if len(c):
            ax.plot(c['N'], c['coef_bytes_per_item'], color=INK, ls='-.', lw=1.0,
                    label='DTCWT coefficients (exact)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_title(f'(c) Working memory per item (batch {Bb:,})', fontsize=10.5, color=INK)
        ax.legend(loc='upper left', frameon=False, fontsize=7.5)
    else:
        ax.text(0.5, 0.5, 'no memory data yet', transform=ax.transAxes, ha='center', color=INK2)
        ax.set_title('(c) Working memory per item', fontsize=10.5, color=INK)
    ax.set_xlabel('window length N (log2)', color=INK2)
    ax.set_ylabel('peak traced bytes per item (log)', color=INK2)
    hw = root / 'bench' / 'metadata' / 'bench_run.json'
    sub = ''
    if hw.exists():
        h = json.loads(hw.read_text(encoding='utf-8')).get('hardware', {})
        v = h.get('versions', {})
        sub = (f"{h.get('cpu_name', '')}, one thread; Python {v.get('python', '')}, "
               f"NumPy {v.get('numpy', '')}, dtcwt {v.get('dtcwt', '')}")
    fig.suptitle('Measured cost of the nine methods (median of 10 repeats, band = IQR)'
                 + (f'\n{sub}' if sub else ''), fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    files = save(fig, out, 'T3.8_runtime')
    return files, '; '.join(notes)


# =================================================================== T3.10
# Main-text figures of the paper rebuilt from the causal protocol-V5 runs,
# with the visual style of paper V4 (evaluation/cross_dataset_visualizer.py:
# Paired colours, light-grey axes, hatched wavelet-based bars).  The style
# constants are copied here, not imported, because importing that module
# changes the global matplotlib style of every other figure of this program.
# Added in task T3.10 (chat 17, 27 Sep 2026); decisions in 07_Task_Tracker.md
# section "e".  The same loaders are used by scripts/generate_revision_tables.py
# and scripts/generate_thesis_figures.py, so figures and tables share numbers.

V4_BASELINES = ['AF', 'CompoundPop', 'EWMA', 'PFRF', 'RRD', 'VSE']
V4_WAVELET = ['DWT+AF', 'DTCWT+AF', 'WSPI']
V4_ORDER = V4_BASELINES + V4_WAVELET
V4_COLORS = {'AF': '#A6CEE3', 'CompoundPop': '#B2DF8A', 'EWMA': '#FDBF6F',
             'PFRF': '#CAB2D6', 'RRD': '#FB9A99', 'VSE': '#FFFF99',
             'DWT+AF': '#FF8C00', 'DTCWT+AF': '#4682B4', 'WSPI': '#C71585',
             # extra methods of the equal-window runs (SI and thesis only)
             'SMA': '#6A3D9A', 'EWMA-eq': '#B15928', 'Holt': '#8C8C8C'}
V4_BG, V4_HATCH, V4_EDGE, V4_EDGE_LW = '#EAEAF2', '//', '#222222', 1.2
V4_PURPLE, V4_DPI = '#7E1C9F', 300
V4_SCENARIOS = [('youtube_hourly', 'YouTube Hourly'),
                ('taxi_hourly', 'NYC Yellow Taxi Hourly'),
                ('taxi_30min', 'NYC Yellow Taxi 30m'),
                ('taxi_5min', 'NYC Yellow Taxi 5m')]
ML_SCENARIOS = [('movielens_daily', 'MovieLens Daily'),
                ('movielens_weekly', 'MovieLens Weekly')]
MAIN_METRICS = ['ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion']
V4_YLABEL = {'ndcg@10': 'NDCG@10  (higher is better ↑)',
             'spearman_rho': 'Spearman ρ  (higher is better ↑)',
             'rsi@10': 'RSI@10  (higher is better ↑)',
             'robustness_distortion': 'ΔRank  (lower is better ↓, log scale)'}
V4_SHORT = {'ndcg@10': 'NDCG@10', 'spearman_rho': 'Spearman ρ',
            'rsi@10': 'RSI@10', 'robustness_distortion': 'ΔRank'}
# rolling-mean length of the window-by-window figures = one natural period
# (YouTube 1 day, taxi 1 week = the bootstrap blocks of T1.6; MovieLens 4 weeks
# and 13 weeks = the sensitivity blocks of T3.9)
TIME_ROLL = {'youtube_hourly': 24, 'taxi_hourly': 168,
             'movielens_daily': 28, 'movielens_weekly': 13}
TIME_UNIT = {'youtube_hourly': 'hours', 'taxi_hourly': 'hours',
             'movielens_daily': 'days', 'movielens_weekly': 'weeks'}
# window-by-window figures: these are drawn in colour, other baselines grey
TIME_HILITE = {'WSPI': ('#C71585', 2.4), 'DTCWT+AF': ('#4682B4', 1.6),
               'DWT+AF': ('#FF8C00', 1.3), 'AF': ('#33A02C', 1.1),
               'RRD': ('#E31A1C', 1.1)}


def v4_summary_path(results, config, scenario):
    """comparison/summary_common_windows.csv of one run (paper configuration)."""
    if scenario.startswith('movielens'):
        base = results / 'T3.9_movielens' / config / scenario
        if config == 'equal64':
            base = base / 'W064'
    elif config == 'default':
        base = results / 'T1.5_causal_universe' / 'T1.4_protocol_v5' / scenario
    elif config == 'equal64':
        base = results / 'T2.2_window_sweep' / scenario / 'W064'
    else:
        raise ValueError(config)
    return base / 'comparison' / 'summary_common_windows.csv'


def v4_protocol_dir(results, config, scenario):
    return v4_summary_path(results, config, scenario).parent.parent / 'protocol'


def _stats_rows(results, config, scenario, kind):
    """Rows of the T1.6 statistics (kind = method_summary | paired_tests)."""
    name = f'all_{kind}.csv'
    if scenario.startswith('movielens'):
        d = pd.read_csv(results / 'T1.6_stats' / 'T3.9_movielens' / name)
        return d[(d['run_group'] == config) & (d['scenario'] == scenario)]
    if config == 'default':
        d = pd.read_csv(results / 'T1.6_stats' / 'causal' / name)
        return d[(d['run_group'] == 'T1.4_protocol_v5') & (d['scenario'] == scenario)]
    # T2.2 sweep statistics: run_group = scenario, scenario = window folder
    d = pd.read_csv(results / 'T1.6_stats' / 'T2.2_window_sweep' / name)
    return d[(d['run_group'] == scenario) & (d['scenario'] == 'W064')]


def v4_values(results, config, scenarios, metrics=MAIN_METRICS):
    """Mean, 95 % block-bootstrap CI (T1.6) and ties share for every method.

    The mean of the statistics file is checked against the run's own
    summary_common_windows.csv (relative difference <= 1e-9); a mismatch stops
    the program.  Returns a long DataFrame (config, scenario, method, metric,
    mean, ci_low, ci_high, n_windows, ties_top21_share, block, summary_mean,
    rel_diff)."""
    rows = []
    for sc in scenarios:
        summ = pd.read_csv(v4_summary_path(results, config, sc)).set_index('method')
        st = _stats_rows(results, config, sc, 'method_summary')
        if st.empty:
            raise FileNotFoundError(f'no T1.6 statistics for {config}/{sc}')
        for _, r in st[st['metric'].isin(metrics)].iterrows():
            m, met = r['method'], r['metric']
            sm = float(summ.loc[m, f'{met}_mean'])
            rel = abs(sm - r['mean']) / max(abs(sm), 1e-12)
            if rel > 1e-9:
                raise ValueError(f'{config}/{sc}/{m}/{met}: statistics mean {r["mean"]} '
                                 f'!= summary mean {sm}')
            rows.append(dict(config=config, scenario=sc, method=m, metric=met,
                             mean=r['mean'], ci_low=r['ci_low'], ci_high=r['ci_high'],
                             n_windows=int(r['n_windows']),
                             ties_top21_share=float(summ.loc[m, 'ties_top21_share']),
                             block=r['block'], summary_mean=sm, rel_diff=rel))
    return pd.DataFrame(rows)


def v4_tests(results, config, scenarios, metrics=MAIN_METRICS):
    """Paired tests against WSPI (T1.6): verdict, Holm block p, Cliff's delta."""
    out = []
    for sc in scenarios:
        t = _stats_rows(results, config, sc, 'paired_tests')
        t = t[t['metric'].isin(metrics)].assign(config=config)
        if config != 'default' and not sc.startswith('movielens'):
            # T2.2 sweep statistics keep the scenario in run_group and the
            # window folder (W064) in scenario; align them with the other rows.
            t = t.assign(run_group=t['scenario'], scenario=sc)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def _v4_axes(ax):
    ax.set_facecolor(V4_BG)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.9, linewidth=1.0, color='white')
    ax.xaxis.grid(False)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', length=0)


def _v4_legend(fig_or_ax, methods, anchor=(0.5, -0.10), ncol=None, **kw):
    handles = []
    for m in methods:
        pk = dict(facecolor=V4_COLORS[m], label=m)
        if m in V4_WAVELET:
            pk.update(edgecolor=V4_EDGE, linewidth=V4_EDGE_LW, hatch=V4_HATCH)
        handles.append(Patch(**pk))
    leg = fig_or_ax.legend(handles=handles, loc='upper center', bbox_to_anchor=anchor,
                           fontsize=9, frameon=False, ncol=ncol or len(methods),
                           title='Method', title_fontsize=9, handlelength=1.4,
                           columnspacing=1.0, **kw)
    for text, m in zip(leg.get_texts(), methods):
        if m in V4_WAVELET:
            text.set_color(V4_PURPLE)
            text.set_fontweight('bold')
    return leg


def _v4_bar_panel(ax, vals, groups, metric, methods=V4_ORDER, width=0.085):
    """One V4-style grouped bar panel with 95 % CI whiskers.
    groups = [(scenario id, label)]; vals = v4_values() rows of one config."""
    v = vals[vals['metric'] == metric].set_index(['scenario', 'method'])
    centers = np.arange(len(groups), dtype=float)
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    _v4_axes(ax)
    for i, m in enumerate(methods):
        xs = centers + offsets[i]
        ys = np.array([v.loc[(g, m), 'mean'] for g, _ in groups])
        lo = np.array([v.loc[(g, m), 'ci_low'] for g, _ in groups])
        hi = np.array([v.loc[(g, m), 'ci_high'] for g, _ in groups])
        kw = dict(width=width, color=V4_COLORS[m], zorder=3)
        if m in V4_WAVELET:
            kw.update(edgecolor=V4_EDGE, linewidth=V4_EDGE_LW, hatch=V4_HATCH)
        else:
            kw.update(edgecolor='white', linewidth=0.4)
        ax.bar(xs, ys, **kw)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt='none', ecolor='#333333',
                    elinewidth=0.8, capsize=1.6, zorder=4)
    ax.set_xticks(centers)
    ax.set_xticklabels([lab for _, lab in groups], fontsize=10)
    ax.set_xlim(centers[0] - 0.55, centers[-1] + 0.55)
    if metric == 'robustness_distortion':
        ax.set_yscale('log')
        lo_all = v['ci_low'].min()
        ax.set_ylim(10 ** np.floor(np.log10(max(lo_all, 1e-3) * 0.8)),
                    v['ci_high'].max() * 1.6)
        ax.yaxis.grid(True, which='major', alpha=0.9, linewidth=1.0, color='white')
    else:
        ax.set_ylim(0, 1.10)


def _save_v4(fig, out, name):
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'{name}.{ext}', dpi=V4_DPI, bbox_inches='tight')
    plt.close(fig)
    return [f'{name}.pdf', f'{name}.png']


def _fig_t310_bars(results, out, metric, name):
    try:
        vals = v4_values(results, 'default', [s for s, _ in V4_SCENARIOS])
    except (FileNotFoundError, KeyError) as e:
        return None, f'input missing: {e}'
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _v4_bar_panel(ax, vals, V4_SCENARIOS, metric)
    ax.set_ylabel(V4_YLABEL[metric], fontsize=10)
    _v4_legend(ax, V4_ORDER)
    w = vals[(vals['metric'] == metric) & (vals['method'] == 'WSPI')]
    nwin = ', '.join(f'{s}={n}' for s, n in zip(w['scenario'], w['n_windows']))
    return _save_v4(fig, out, name), (f'default configuration (baselines 7, wavelet-based 64), '
                                      f'common windows {nwin}; whiskers = 95% block-bootstrap CI (T1.6)')


def fig_t310_fig2(results, out):
    """Paper Figure 2: NDCG@10, four scenarios, nine methods."""
    return _fig_t310_bars(results, out, 'ndcg@10', 'T3.10_fig2_ndcg10')


def fig_t310_fig3(results, out):
    """Paper Figure 3: Spearman rho."""
    return _fig_t310_bars(results, out, 'spearman_rho', 'T3.10_fig3_spearman')


def fig_t310_fig4(results, out):
    """Paper Figure 4: RSI@10."""
    return _fig_t310_bars(results, out, 'rsi@10', 'T3.10_fig4_rsi10')


def fig_t310_fig5(results, out):
    """Paper Figure 5: Delta Rank (log scale; values span 4 to about 215)."""
    return _fig_t310_bars(results, out, 'robustness_distortion', 'T3.10_fig5_deltarank')


def v4_window_series(results, config, scenario, metric='rsi@10', methods=V4_ORDER):
    """Window-by-window values on the common windows of all methods.
    Returns (DataFrame index=window_id with a 'time' column + one column per
    method, n_common).  The number of common windows is checked against
    summary_common_windows.csv."""
    pdir = v4_protocol_dir(results, config, scenario)
    frames = {}
    for m in methods:
        d = pd.read_csv(pdir / f'{m}_protocol.csv', usecols=['window_id', 'timestamp', metric])
        frames[m] = d.set_index('window_id')
    common = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    summ = pd.read_csv(v4_summary_path(results, config, scenario))
    n_expected = int(summ['n_windows'].iloc[0])
    if len(common) != n_expected:
        raise ValueError(f'{scenario}: {len(common)} common windows, summary says {n_expected}')
    df = pd.DataFrame({m: frames[m].loc[common, metric].to_numpy() for m in methods},
                      index=pd.Index(common, name='window_id'))
    ts = frames['WSPI'].loc[common, 'timestamp']
    df.insert(0, 'time', pd.to_datetime(ts.to_numpy(), unit='ms', utc=True))
    return df, len(common)


def _v4_time_panel(ax, df, roll, unit, methods=V4_ORDER):
    _v4_axes(ax)
    ax.xaxis.grid(True, alpha=0.9, linewidth=1.0, color='white')
    others = [m for m in methods if m not in TIME_HILITE]
    for m in others:
        ax.plot(df['time'], df[m].rolling(roll, center=True, min_periods=roll // 2).mean(),
                color='#9A9A9A', lw=0.8, alpha=0.8, zorder=2)
    for m in [m for m in ['AF', 'RRD', 'DWT+AF', 'DTCWT+AF', 'WSPI'] if m in methods]:
        c, lw = TIME_HILITE[m]
        ax.plot(df['time'], df[m].rolling(roll, center=True, min_periods=roll // 2).mean(),
                color=c, lw=lw, zorder=5 if m == 'WSPI' else 3, label=m)
    ax.plot([], [], color='#9A9A9A', lw=0.8, label='Other baselines (' + ', '.join(others) + ')')
    ax.set_ylabel(f'RSI@10  (rolling mean, {roll} {unit})', fontsize=10)
    ax.set_ylim(None, 1.01)


def _fig_t310_time(results, out, scenario, name, title):
    try:
        df, n = v4_window_series(results, 'default', scenario)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return None, f'input missing or inconsistent: {e}'
    roll = TIME_ROLL[scenario]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    _v4_time_panel(ax, df, roll, TIME_UNIT[scenario])
    ax.set_title(title, fontsize=10.5)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6,
                    frameon=False, fontsize=9)
    for t in leg.get_texts():
        if t.get_text() in V4_WAVELET:
            t.set_color(V4_PURPLE)
            t.set_fontweight('bold')
    return _save_v4(fig, out, name), f'{n} common windows, rolling mean {roll} (centred)'


def fig_t310_fig6(results, out):
    """Paper Figure 6: window-by-window RSI@10, YouTube (rolling mean 24 h)."""
    return _fig_t310_time(results, out, 'youtube_hourly', 'T3.10_fig6_youtube_rsi_time',
                          'YouTube Hourly — RSI@10 over the common evaluation windows')


def fig_t310_fig7(results, out):
    """Paper Figure 7: window-by-window RSI@10, NYC Taxi hourly (rolling mean 168 h)."""
    return _fig_t310_time(results, out, 'taxi_hourly', 'T3.10_fig7_taxi_rsi_time',
                          'NYC Yellow Taxi Hourly — RSI@10 over the common evaluation windows')


def v4_best_traditional(vals, scenario, metric, exclude=('PFRF',)):
    """Best baseline (PFRF excluded, as in V4) for one scenario and metric."""
    v = vals[(vals['scenario'] == scenario) & (vals['metric'] == metric)
             & vals['method'].isin([m for m in V4_BASELINES if m not in exclude])]
    r = v.loc[v['mean'].idxmin()] if metric == 'robustness_distortion' else v.loc[v['mean'].idxmax()]
    return r


def fig_t310_fig8(results, out):
    """Paper Figure 8: effect of temporal granularity (taxi hourly, 30 min, 5 min).
    Panels (a) RSI@10, (b) Delta Rank, (c) NDCG@10 (added in T3.10 for the
    accuracy-stability frame).  Lines: WSPI, DTCWT+AF and the best baseline of
    each granularity (PFRF excluded), whose name is written at the point."""
    gran = V4_SCENARIOS[1:]
    try:
        vals = v4_values(results, 'default', [s for s, _ in gran])
    except (FileNotFoundError, KeyError) as e:
        return None, f'input missing: {e}'
    x = np.arange(len(gran))
    styles = {'WSPI': dict(color=V4_COLORS['WSPI'], ls='-', lw=2.6, marker='o', ms=7.5,
                           markeredgecolor=V4_EDGE, markeredgewidth=0.9, zorder=6),
              'DTCWT+AF': dict(color=V4_COLORS['DTCWT+AF'], ls='--', lw=2.0, marker='s', ms=6.5,
                               markeredgecolor=V4_EDGE, markeredgewidth=0.7, zorder=5),
              'Best traditional': dict(color='#888888', ls=':', lw=1.9, marker='^', ms=6.5, zorder=4)}
    panels = [('rsi@10', '(a) Temporal stability', '{:.3f}', False),
              ('robustness_distortion', '(b) Noise robustness', '{:.2f}', True),
              ('ndcg@10', '(c) Ranking quality', '{:.3f}', True)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (met, title, fmt, below) in zip(axes, panels):
        _v4_axes(ax)
        v = vals[vals['metric'] == met].set_index(['scenario', 'method'])
        series = {}
        for m in ('WSPI', 'DTCWT+AF'):
            series[m] = [v.loc[(s, m)] for s, _ in gran]
        best = [v4_best_traditional(vals, s, met) for s, _ in gran]
        series['Best traditional'] = best
        for name, rows in series.items():
            ys = np.array([r['mean'] for r in rows])
            lo = np.array([r['ci_low'] for r in rows])
            hi = np.array([r['ci_high'] for r in rows])
            ax.plot(x, ys, label=name, **styles[name])
            ax.errorbar(x, ys, yerr=[ys - lo, hi - ys], fmt='none',
                        ecolor=styles[name]['color'], elinewidth=1.0, capsize=3, zorder=3)
        for xi, r in zip(x, series['WSPI']):
            ax.annotate(fmt.format(r['mean']), (xi, r['mean']), textcoords='offset points',
                        xytext=(0, -16 if below else 9), ha='center', fontsize=8.5,
                        color=V4_COLORS['WSPI'], fontweight='bold')
        for xi, r in zip(x, best):
            ax.annotate(r['method'], (xi, r['mean']), textcoords='offset points',
                        xytext=(8, 4), ha='left', fontsize=7.5, color='#555555')
        ax.set_xticks(x)
        ax.set_xticklabels(['Hourly', '30-min', '5-min'], fontsize=10)
        ax.set_xlim(x[0] - 0.35, x[-1] + 0.45)
        ylab = V4_YLABEL[met].replace(', log scale', '')
        ax.set_ylabel(ylab, fontsize=10)
        ax.set_title(title, fontsize=10.5)
        if met == 'robustness_distortion':
            ax.set_ylim(0, max(max(r['ci_high'] for r in rows) for rows in series.values()) * 1.15)
        else:
            lo_all = min(min(r['ci_low'] for r in rows) for rows in series.values())
            hi_all = max(max(r['ci_high'] for r in rows) for rows in series.values())
            pad = (hi_all - lo_all) * 0.25
            ax.set_ylim(lo_all - pad, hi_all + pad)
    h, lab = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3,
                     frameon=False, fontsize=9, handlelength=2.2, columnspacing=2.0)
    for t in leg.get_texts():
        if t.get_text() in ('WSPI', 'DTCWT+AF'):
            t.set_color(V4_PURPLE)
            t.set_fontweight('bold')
    fig.suptitle('NYC Yellow Taxi — effect of temporal granularity', fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return _save_v4(fig, out, 'T3.10_fig8_granularity'), \
        'default configuration; best traditional = best of the six baselines except PFRF at each point; whiskers = 95% CI'


def ml_bars_figure(results, out, config, name, methods=V4_ORDER):
    """2 x 2 bar figure on MovieLens (daily and weekly) for one configuration."""
    vals = v4_values(results, config, [s for s, _ in ML_SCENARIOS])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    groups = [(s, lab.replace('MovieLens ', '')) for s, lab in ML_SCENARIOS]
    for ax, met, tag in zip(axes.ravel(), MAIN_METRICS, 'abcd'):
        _v4_bar_panel(ax, vals, groups, met, methods=methods, width=0.8 / len(methods))
        ax.set_ylabel(V4_YLABEL[met], fontsize=9.5)
        ax.set_title(f'({tag}) {V4_SHORT[met]}', fontsize=10.5)
    _v4_legend(fig, methods, anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    w = vals[(vals['metric'] == 'ndcg@10') & (vals['method'] == 'WSPI')]
    return _save_v4(fig, out, name), \
        f'{config}; common windows ' + ', '.join(f'{s}={n}' for s, n in zip(w['scenario'], w['n_windows']))


def fig_t310_movielens(results, out):
    """Paper figure fig:movielens (Subsection 4.10): default configuration,
    NDCG@10, rho, RSI@10, Delta Rank (log), daily and weekly, 95 % CI."""
    try:
        return ml_bars_figure(results, out, 'default', 'T3.10_fig_movielens')
    except (FileNotFoundError, KeyError) as e:
        return None, f'input missing: {e}'


def fig_t310_ml_rsi_time(results, out):
    """SI (and thesis) figure: window-by-window RSI@10 on MovieLens, default
    configuration; rows daily (rolling 28 days) and weekly (rolling 13 weeks)."""
    try:
        data = [(s, lab) + v4_window_series(results, 'default', s) for s, lab in ML_SCENARIOS]
    except (FileNotFoundError, KeyError, ValueError) as e:
        return None, f'input missing or inconsistent: {e}'
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.2))
    msg = []
    for ax, (s, lab, df, n) in zip(axes, data):
        _v4_time_panel(ax, df, TIME_ROLL[s], TIME_UNIT[s])
        ax.set_title(f'{lab} — RSI@10 over the common evaluation windows', fontsize=10.5)
        msg.append(f'{s}: {n} windows, rolling {TIME_ROLL[s]}')
    h, lab = axes[0].get_legend_handles_labels()
    leg = fig.legend(h, lab, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=6,
                     frameon=False, fontsize=9)
    for t in leg.get_texts():
        if t.get_text() in V4_WAVELET:
            t.set_color(V4_PURPLE)
            t.set_fontweight('bold')
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save_v4(fig, out, 'T3.10_si_movielens_rsi_time'), '; '.join(msg)



FIGURES = {
    'T2.2_window_curves': fig_t22_window_curves,   # SI figure of the paper
    'T2.3_tradeoff': fig_t23_tradeoff,             # kept in the program only, not in the paper
    'T2.4_surge_examples': fig_t24_surge_examples,  # main text (E11 examples)
    'T2.4_delay_ecdf': fig_t24_delay_ecdf,          # SI (E11 delay distribution)
    'T3.2_param_heatmap': fig_t32_param_heatmap,    # SI (E3 alpha x beta grid)
    'T3.5_spike_size': fig_t35_spike_size,          # SI (E4 spike size)
    'T3.6_shift_invariance': fig_t36_shift_invariance,  # SI (E5 shift test)
    'T3.7_feature_relation': fig_t37_feature_relation,  # SI (E7 R and WE)
    'T3.8_runtime': fig_t38_runtime,                    # SI (E9 runtime and memory)
    'T3.10_fig2': fig_t310_fig2,                        # main text, Figure 2 (NDCG@10)
    'T3.10_fig3': fig_t310_fig3,                        # main text, Figure 3 (Spearman rho)
    'T3.10_fig4': fig_t310_fig4,                        # main text, Figure 4 (RSI@10)
    'T3.10_fig5': fig_t310_fig5,                        # main text, Figure 5 (Delta Rank)
    'T3.10_fig6': fig_t310_fig6,                        # main text, Figure 6 (YouTube over time)
    'T3.10_fig7': fig_t310_fig7,                        # main text, Figure 7 (taxi over time)
    'T3.10_fig8': fig_t310_fig8,                        # main text, Figure 8 (granularity)
    'T3.10_fig_movielens': fig_t310_movielens,          # main text, Subsection 4.10
    'T3.10_si_movielens_rsi_time': fig_t310_ml_rsi_time,  # SI + thesis
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results', required=True, help='results\\revision_v5 folder')
    ap.add_argument('--out', required=True, help='output folder for the figures')
    ap.add_argument('--only', nargs='*', default=None, choices=list(FIGURES))
    a = ap.parse_args()
    res = Path(a.results) if Path(a.results).is_absolute() else ROOT / a.results
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if not res.is_dir():
        sys.exit(f'--results folder not found: {res}')
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10})
    ok = 0
    for fid in (a.only or FIGURES):
        files, msg = FIGURES[fid](res, out)
        if files is None:
            print(f'[skip] {fid}: {msg}')
        else:
            ok += 1
            print(f'[ok]   {fid}: {", ".join(files)}\n       {msg}')
    print(f'{ok} of {len(a.only or FIGURES)} figures written to {out}')


if __name__ == '__main__':
    main()
