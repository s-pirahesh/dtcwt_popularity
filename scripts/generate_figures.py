r"""
Generate every figure of the paper, its Supplementary Information and the thesis
================================================================================
All figures are drawn from the result CSVs under --results (normally
results\revision_v5).  Nothing is typed by hand: the loaders check the
block-bootstrap statistics against summary_common_windows.csv of every run and
stop on a mismatch.  The result folders are read from result_paths.json (next
to this file).

Targets (--target)
------------------
  paper   main-text figures at their printed size (text 7-9 pt, TrueType fonts,
          PDF + PNG 600 dpi), file names = the names used in the LaTeX source:
            fig_ndcg10, fig_spearman, fig_rsi10, fig_deltarank   grouped bars, four
                YouTube and taxi scenarios, nine methods, 95 % CI whiskers
            fig_movielens          the same four metrics on MovieLens (2 x 2)
            fig_youtube_rsi_time, fig_taxi_rsi_time   window-by-window RSI@10,
                rolling mean of one period, every method in its V4 colour
            fig_granularity        taxi at three granularities
            fig_surge              genuine entries into the true Top-10
          and figures_paper_run.json (sizes, font range, Matplotlib version, md5).
          The pipeline figure is drawn in TikZ in the LaTeX source.
  si      figures of the Supplementary Information at printed size:
            si_window_curves, si_feature_relation, si_spike_size, si_runtime,
            si_shift_invariance, si_param_heatmap, si_movielens_rsi_time,
            si_delay_ecdf; and figures_si_run.json
  thesis  figures of Chapter 4 of the thesis (names = thesis figure numbers);
          --list prints the registry (ready / paper / planned / other)
  draft   the same figures at working size (12-17 inches wide), plus the
          accuracy-stability trade-off plot, which is kept in the program only;
          the trade-off also writes pareto_points.csv (CSV and JSON only) to
          its result folder

Usage (from the project root)
-----------------------------
  python scripts/generate_figures.py --target paper --results results/revision_v5 --out <folder>
  optional: --only <figure names>
Full command list: REPRODUCE.md
"""
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
from matplotlib import font_manager, legend as mlegend  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PATHS = json.loads((Path(__file__).resolve().parent / 'result_paths.json').read_text(encoding='utf-8'))


def rp(results, key, *parts):
    """Path of a result folder (or a file in it) named in result_paths.json."""
    return Path(results).joinpath(*PATHS[key].split('/'), *parts)


# ---------------------------------------------------------------- common style
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e0', '#fcfcfb'
GREY = '#9a9994'

# Every method in its own colour, in every figure (no grey group of 'other
# methods').  The nine methods use the V4 colours,
# markers and line widths of evaluation/visualizer.py (METHOD_COLORS,
# METHOD_MARKERS, _lw), read inside an rc_context so that its global rcParams do
# not leak into these figures (the same source as the RSI@10-over-time figures
# of the paper target).  SMA, EWMA-eq and Holt are not in V4 and get
# colours of their own.
METHOD_EXTRA = {'SMA': ('#3F51B5', 'P', 1.5), 'EWMA-eq': ('#8BC34A', 'X', 1.5),
                'Holt': ('#FFC107', '<', 1.5)}
_V4_STYLE = {}


def method_style(m):
    """(colour, marker, line width) of a method."""
    if not _V4_STYLE:
        import importlib.util
        with matplotlib.rc_context():
            spec = importlib.util.spec_from_file_location(
                'v4_visualizer', ROOT / 'evaluation' / 'visualizer.py')
            vis = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(vis)
            for k, c in vis.METHOD_COLORS.items():
                _V4_STYLE[k] = (c, vis.METHOD_MARKERS[k], vis._lw(k))
        _V4_STYLE.update(METHOD_EXTRA)
    if m not in _V4_STYLE:
        raise KeyError(f'no colour defined for method {m}')
    return _V4_STYLE[m]


def method_order(methods):
    """Legend and drawing order: V4 baselines, the added smoothers, then the three
    wavelet-based methods with WSPI last (drawn on top)."""
    ref = ['AF', 'CompoundPop', 'EWMA', 'PFRF', 'RRD', 'VSE', 'SMA', 'EWMA-eq', 'Holt',
           'DWT+AF', 'DTCWT+AF', 'WSPI']
    ms = list(methods)
    extra = [m for m in ms if m not in ref]
    if extra:
        raise KeyError(f'unknown methods: {extra}')
    return [m for m in ref if m in ms]


def all_method_handles(methods, marker=True):
    out = []
    for m in method_order(methods):
        c, mk, lw = method_style(m)
        out.append(plt.Line2D([], [], color=c, marker=mk if marker else None, lw=lw * 0.6,
                              ms=5, label=m))
    return out
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


# ----------------------------------------------------------- accuracy-stability trade-off (draft only)
TRADEOFF_X = 'rsi@10_mean'
TRADEOFF_ROWS = [('ndcg@10_mean', 'NDCG@10'), ('spearman_rho_mean', r'Spearman $\rho$')]
TRADEOFF_ROB = 'robustness_distortion_mean'
TRADEOFF_HIGHLIGHT = {                  # colour + marker, so colour is never alone
    'WSPI':        ('#2a78d6', 'o', 2.4, 9),
    'SMA':         ('#eb6834', 's', 1.6, 7),
    'DTCWT+AF':    ('#1baf7a', '^', 1.6, 7),
    'CompoundPop': ('#4a3aa7', 'D', 1.6, 6),
}
TRADEOFF_VARIANTS = {'main': lambda m: m not in ('PFRF', 'Holt'),
                'full': lambda m: True}


def _tradeoff_draw(d, variant, out):
    keep = TRADEOFF_VARIANTS[variant]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6))
    rows = []
    for c, (sc, title) in enumerate(SCENARIOS):
        s = d[(d.scenario == sc) & d.method.map(keep)].copy()
        if s.empty:
            raise ValueError(f'scenario {sc} missing in sweep_summary.csv')
        s['pareto4'] = pareto_flags(s, ['ndcg@10_mean', 'spearman_rho_mean', TRADEOFF_X], [TRADEOFF_ROB])
        for r, (y, ylab) in enumerate(TRADEOFF_ROWS):
            s[f'pareto2_{y}'] = pareto_flags(s, [TRADEOFF_X, y])
            ax = axes[r, c]
            style_axes(ax)
            fx, fy = staircase(s, TRADEOFF_X, y)
            ax.plot(fx, fy, ls='--', lw=1.1, color=INK2, zorder=1)
            order = [m for m in s.method.unique() if m not in TRADEOFF_HIGHLIGHT] + \
                    [m for m in TRADEOFF_HIGHLIGHT if m in set(s.method)]
            for m in order:
                g = s[s.method == m].sort_values('window')
                if m in TRADEOFF_HIGHLIGHT:
                    col, mk, lw, ms = TRADEOFF_HIGHLIGHT[m]
                    z = 4 if m == 'WSPI' else 3
                else:
                    col, mk, lw, ms, z = GREY, '.', 0.9, 6, 2
                ax.plot(g[TRADEOFF_X], g[y], color=col, lw=lw, marker=mk, ms=ms,
                        mec=SURF, mew=1.0, zorder=z)
                if m in TRADEOFF_HIGHLIGHT:
                    for _, p in g.iterrows():
                        ax.annotate(str(int(p.window)), (p[TRADEOFF_X], p[y]), xytext=(4, 4),
                                    textcoords='offset points', fontsize=6.5, color=INK2, zorder=6)
                else:
                    last = g.iloc[-1]
                    ax.annotate(m, (last[TRADEOFF_X], last[y]), xytext=(3, -9),
                                textcoords='offset points', fontsize=7, color=INK2)
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK)
            else:
                ax.set_xlabel('RSI@10 (higher = more stable)', color=INK2)
            if c == 0:
                ax.set_ylabel(ylab + ' (higher = more accurate)', color=INK2)
        s['variant'] = variant
        rows.append(s[['variant', 'scenario', 'method', 'window', 'ndcg@10_mean',
                       'spearman_rho_mean', TRADEOFF_X, TRADEOFF_ROB, 'n_windows',
                       'pareto2_ndcg@10_mean', 'pareto2_spearman_rho_mean', 'pareto4']])
    h = [plt.Line2D([], [], color=v[0], marker=v[1], lw=v[2], ms=v[3] * 0.85, label=m)
         for m, v in TRADEOFF_HIGHLIGHT.items()]
    h += [plt.Line2D([], [], color=GREY, marker='.', lw=0.9, label='other methods (name at N=128)'),
          plt.Line2D([], [], color=INK2, ls='--', lw=1.1, label='2-D Pareto frontier of the panel')]
    fig.legend(handles=h, loc='lower center', ncol=6, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    shown = 'all 12 methods' if variant == 'full' else 'PFRF and Holt not shown'
    fig.suptitle('Accuracy-stability trade-off over window length N (small numbers = N); '
                 f'mean over common windows >= 32; {shown}', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    files = save(fig, out, f'tradeoff_{variant}')
    return pd.concat(rows), files


def fig_tradeoff(results, out):
    src = rp(results, 'window_sweep', 'sweep_summary.csv')
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    parts, files = [], []
    for v in TRADEOFF_VARIANTS:
        p, f = _tradeoff_draw(d, v, out)
        parts.append(p); files += f
    csv_dir = rp(results, 'tradeoff')
    (csv_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    pts = pd.concat(parts)
    pts.to_csv(csv_dir / 'pareto_points.csv', index=False)
    meta = {'date_utc': datetime.now(timezone.utc).isoformat(),
            'input': str(src), 'figures_dir': str(out), 'figures': files,
            'variants': {'main': 'without PFRF and Holt', 'full': 'all methods'},
            'pareto_2d': 'per panel: RSI@10 and the y metric, higher is better, over methods shown',
            'pareto_4': 'CSV only: NDCG@10, spearman_rho, RSI@10 up; robustness_distortion down',
            'python': platform.python_version(), 'pandas': pd.__version__,
            'matplotlib': matplotlib.__version__}
    (csv_dir / 'metadata' / 'tradeoff_run.json').write_text(json.dumps(meta, indent=2))
    w = pts[(pts.method == 'WSPI') & (pts.window == 64)]
    note = '; '.join(f"{r.variant}/{r.scenario}: NDCG-RSI {'on' if r['pareto2_ndcg@10_mean'] else 'off'}, "
                     f"rho-RSI {'on' if r['pareto2_spearman_rho_mean'] else 'off'}" for _, r in w.iterrows())
    return files, 'WSPI(64) frontier -> ' + note



# ----------------------------------------------------------- window-length curves (SI)
CURVES_ROWS = [('ndcg@10_mean', 'NDCG@10 (higher = more accurate)'),
            ('rsi@10_mean', 'RSI@10 (higher = more stable)')]
CURVES_N = [7, 16, 32, 64, 128]


def fig_window_curves(results, out):
    """NDCG@10 and RSI@10 against the window length N for every method."""
    src = rp(results, 'window_sweep', 'sweep_summary.csv')
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    d = d[~d.method.isin(['PFRF', 'Holt'])]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for c, (sc, title) in enumerate(SCENARIOS):
        s = d[d.scenario == sc]
        if s.empty:
            raise ValueError(f'scenario {sc} missing in sweep_summary.csv')
        for r, (y, ylab) in enumerate(CURVES_ROWS):
            ax = axes[r, c]
            style_axes(ax)
            for k, m in enumerate(method_order(s.method.unique())):
                g = s[s.method == m].sort_values('window')
                col, mk, lw = method_style(m)
                ax.plot(g.window, g[y], color=col, lw=lw * 0.6, marker=mk, ms=5,
                        mec=SURF, mew=0.6, zorder=3 + k)
            ax.set_xscale('log', base=2)
            ax.set_xticks(CURVES_N)
            ax.set_xticklabels([str(n) for n in CURVES_N])
            ax.set_xlim(6, 200)
            if r == 0:
                ax.set_title(title, fontsize=11, color=INK)
            else:
                ax.set_xlabel('window length N (slots)', color=INK2)
            if c == 0:
                ax.set_ylabel(ylab, color=INK2)
    h = all_method_handles(d.method.unique())
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Accuracy and stability against window length N; wavelet-based methods from N=16; '
                 'mean over common windows >= 32; PFRF and Holt not shown', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    files = save(fig, out, 'window_curves')
    w = s = d[(d.method.isin(['WSPI', 'SMA']))]
    rng = '; '.join(f"{sc}: WSPI NDCG {g[g.method=='WSPI']['ndcg@10_mean'].min():.4f}-"
                    f"{g[g.method=='WSPI']['ndcg@10_mean'].max():.4f}, SMA NDCG "
                    f"{g[g.method=='SMA']['ndcg@10_mean'].min():.4f}-{g[g.method=='SMA']['ndcg@10_mean'].max():.4f}"
                    for sc, g in w.groupby('scenario'))
    return files, rng



# ----------------------------------------------------------- responsiveness to genuine entries
SURGE_EXAMPLE_SCEN = [('youtube_hourly', 'YouTube (hourly)'), ('taxi_hourly', 'NYC Taxi (hourly)')]
SURGE_EXAMPLE_KIND = [('typical', 'typical entry'), ('worst_for_reference', 'worst case for WSPI')]
SURGE_LINES = [                      # (config, method, label, colour, marker, lw)
    ('default', 'WSPI', 'WSPI (N=64)', '#2a78d6', 'o', 2.2),
    ('default', 'AF', 'AF (N=7)', '#1baf7a', 's', 1.5),
    ('default', 'DTCWT+AF', 'DTCWT+AF (N=64)', '#4a3aa7', '^', 1.5),
    ('equal64', 'RRD', 'RRD (N=64)', '#eb6834', 'D', 1.5),
]
SURGE_RANK_CAP = 100


def fig_surge_examples(results, out):
    """Four-column version: the two automatically chosen entries of
    YouTube and taxi hourly (rule in evaluation/responsiveness.select_examples).
    Row 1: real count of the item (band = slots in the true Top-10).
    Row 2: rank of the item for four methods (log axis, 1 at the top); gaps =
    item not eligible for the method; dashed line = rank 10."""
    root = rp(results, 'responsiveness')
    cols = []
    for sc, title in SURGE_EXAMPLE_SCEN:
        tf = root / sc / 'examples' / 'example_traces.csv'
        ef = root / sc / 'examples' / 'example_events.csv'
        if not (tf.exists() and ef.exists()):
            return None, f'input missing: {tf}'
        tr, ev = pd.read_csv(tf), pd.read_csv(ef)
        for kind, klab in SURGE_EXAMPLE_KIND:
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
        tr_rank = base['truth_rank'].clip(upper=SURGE_RANK_CAP)
        ax2.plot(x, tr_rank, color=GREY, lw=1.0, ls='-', zorder=2)
        for cfg, m, lab, col, mk, lw in SURGE_LINES:
            g = t[(t.config == cfg) & (t.method == m)].sort_values('window_id')
            ax2.plot(g.window_id - t0, g['rank'].clip(upper=SURGE_RANK_CAP), color=col, lw=lw,
                     marker=mk, ms=3.2, mec=SURF, mew=0.5, zorder=4 if m == 'WSPI' else 3)
        ax2.set_yscale('log')
        ax2.set_ylim(SURGE_RANK_CAP * 1.15, 0.85)
        ax2.set_yticks([1, 3, 10, 30, 100])
        ax2.set_yticklabels(['1', '3', '10', '30', '100+'])
        ax2.set_xlabel('slots from entry (t0 = 0)', color=INK2)
        if c == 0:
            ax2.set_ylabel('rank of the item (log)', color=INK2)
        notes.append(f"{title}/{klab}: item {e.item_id}, t0={t0}, run={R}, "
                     f"delay WSPI {int(e.delay_WSPI)}, AF {int(e.delay_AF)}")
    h = [plt.Line2D([], [], color=v[3], marker=v[4], lw=v[5], ms=4, label=v[2]) for v in SURGE_LINES]
    h += [plt.Line2D([], [], color=GREY, lw=1.0, label='true rank'),
          plt.Line2D([], [], color=INK2, ls='--', lw=0.9, label='Top-10 boundary'),
          Patch(color='#e9e6f7', label='item in the true Top-10')]
    fig.legend(handles=h, loc='lower center', ncol=7, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    files = save(fig, out, 'surge_examples')
    return files, '; '.join(notes)


def fig_surge_examples_2col(results, out):
    """Paper figure of genuine entries: the four examples of
    fig_surge_examples in a 2-column layout. Rows: YouTube count, YouTube
    rank, taxi count, taxi rank; columns: typical entry, worst case for WSPI.
    Same data, same selection rule and same drawing as the 4-column figure."""
    root = rp(results, 'responsiveness')
    blocks = []
    for sc, title in SURGE_EXAMPLE_SCEN:
        tf = root / sc / 'examples' / 'example_traces.csv'
        ef = root / sc / 'examples' / 'example_events.csv'
        if not (tf.exists() and ef.exists()):
            return None, f'input missing: {tf}'
        tr, ev = pd.read_csv(tf), pd.read_csv(ef)
        row = []
        for kind, klab in SURGE_EXAMPLE_KIND:
            e = ev[ev.example == kind]
            if e.empty:
                return None, f'example {kind} missing in {ef}'
            row.append((title, klab, e.iloc[0], tr[tr.example == kind]))
        blocks.append(row)
    fig = plt.figure(figsize=(7.4, 9.0), layout='constrained')
    subs = fig.subfigures(2, 1, hspace=0.03)
    axes = [sf.subplots(2, 2, sharex='col', gridspec_kw={'height_ratios': [1, 1.25]})
            for sf in subs]
    notes = []
    for b, row in enumerate(blocks):
        for c, (title, klab, e, t) in enumerate(row):
            t0, R = int(e.t0), int(e.run_len)
            base = t[(t.config == 'default') & (t.method == 'WSPI')].sort_values('window_id')
            x = base.window_id.to_numpy() - t0
            ax = axes[b][0, c]
            ax2 = axes[b][1, c]
            style_axes(ax)
            ax.axvspan(-0.5, R - 0.5, color='#e9e6f7', zorder=0, lw=0)
            ax.plot(x, base['count'], color=INK, lw=1.2, zorder=3)
            ax.axvline(0, color=INK2, lw=0.8, ls=':')
            ax.set_title(f'{title}: {klab}\n'
                         f'item {e.item_id}; delay WSPI {int(e.delay_WSPI)}, AF {int(e.delay_AF)} slots',
                         fontsize=8.5, color=INK)
            ax.tick_params(labelbottom=False)
            if base['count'].max() >= 1e4:     # YouTube: 400k instead of an offset label
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '0' if v == 0 else f'{v / 1e3:.0f}k'))
            if c == 0:
                ax.set_ylabel('real count\nper slot', color=INK2, fontsize=8.5)
            style_axes(ax2)
            ax2.axvspan(-0.5, R - 0.5, color='#e9e6f7', zorder=0, lw=0)
            ax2.axhline(10, color=INK2, lw=0.9, ls='--', zorder=1)
            ax2.axvline(0, color=INK2, lw=0.8, ls=':')
            ax2.plot(x, base['truth_rank'].clip(upper=SURGE_RANK_CAP), color=GREY, lw=1.0, zorder=2)
            for cfg, m, lab, col, mk, lw in SURGE_LINES:
                g = t[(t.config == cfg) & (t.method == m)].sort_values('window_id')
                ax2.plot(g.window_id - t0, g['rank'].clip(upper=SURGE_RANK_CAP), color=col,
                         lw=lw * 0.85, marker=mk, ms=2.6, mec=SURF, mew=0.4,
                         zorder=4 if m == 'WSPI' else 3)
            ax2.set_yscale('log')
            ax2.set_ylim(SURGE_RANK_CAP * 1.15, 0.85)
            ax2.set_yticks([1, 3, 10, 30, 100])
            ax2.set_yticklabels(['1', '3', '10', '30', '100+'])
            ax2.set_xlabel('slots from entry (t0 = 0)', color=INK2, fontsize=8.5)
            if c == 0:
                ax2.set_ylabel('rank of the\nitem (log)', color=INK2, fontsize=8.5)
            notes.append(f"{title}/{klab}: item {e.item_id}, t0={t0}, run={R}, "
                         f"delay WSPI {int(e.delay_WSPI)}, AF {int(e.delay_AF)}")
    h = [plt.Line2D([], [], color=v[3], marker=v[4], lw=v[5] * 0.85, ms=4, label=v[2]) for v in SURGE_LINES]
    h += [plt.Line2D([], [], color=GREY, lw=1.0, label='true rank'),
          plt.Line2D([], [], color=INK2, ls='--', lw=0.9, label='Top-10 boundary'),
          Patch(color='#e9e6f7', label='item in the true Top-10')]
    fig.legend(handles=h, loc='outside lower center', ncol=4, frameon=False, fontsize=8)
    files = save(fig, out, 'surge_examples_2col')
    return files, '; '.join(notes)


def fig_delay_ecdf(results, out):
    """SI figure: share of entries that each method has brought into its
    Top-10 within d slots (main variant).  The plateau is 1 - miss rate.
    Row 1: default configuration (baselines 7, wavelet-based 64); row 2: all
    methods with a 64-slot window."""
    root = rp(results, 'responsiveness')
    cfgs = [('default', 'default windows (baselines 7, wavelet-based 64)'),
            ('equal64', 'equal window N = 64')]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=True)
    dmax = 24
    ecdf_methods = set()
    for c, (sc, title) in enumerate(SCENARIOS):
        for r, (cfg, clab) in enumerate(cfgs):
            f = root / sc / cfg / 'delays.csv'
            if not f.exists():
                return None, f'input missing: {f}'
            d = pd.read_csv(f)
            ax = axes[r, c]
            style_axes(ax)
            n = d.event_id.nunique()
            ecdf_methods.update(d.method.unique())
            for k, m in enumerate(method_order(d.method.unique())):
                g = d[d.method == m]
                xs = np.arange(0, dmax + 1)
                ys = [(g.detected & (g.delay <= k2)).sum() / n for k2 in xs]
                col, mk, lw = method_style(m)
                ax.step(xs, ys, where='post', color=col, lw=lw * 0.6, zorder=3 + k)
            ax.set_xlim(0, dmax + 1)
            ax.set_ylim(0, 1.02)
            if r == 0:
                ax.set_title(f'{title}  ({n} entries)', fontsize=10.5, color=INK)
            else:
                ax.set_xlabel('delay d (slots after entry)', color=INK2)
            if c == 0:
                ax.set_ylabel(f'share of entries in Top-10 by d\n{clab}', color=INK2, fontsize=9)
    h = all_method_handles(ecdf_methods, marker=False)
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Delay to bring a genuine entry into the Top-10 (entry: >= 6 slots in the true '
                 'Top-10 after >= 6 slots outside); plateau = 1 - miss rate', fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, out, 'delay_ecdf'), 'ok'


# ------------------------------------------------------------ alpha x beta grid (SI)
HEAT_ROWS = [('ndcg@10_mean', 'NDCG@10', False, '{:.4f}'),
            ('rsi@10_mean', 'RSI@10', False, '{:.4f}'),
            ('robustness_distortion_mean', 'robustness $\\Delta$Rank', True, '{:.1f}')]
HEAT_LIGHT, HEAT_DARK = '#eef4fb', '#1c4f8f'


def fig_param_heatmap(results, out):
    """alpha x beta heat maps of WSPI on the test part (SI figure)."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    src = rp(results, 'param_grid') / 'grid_summary.csv'
    sel = rp(results, 'param_grid') / 'selection' / 'selection.csv'
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
    cmap = LinearSegmentedColormap.from_list('heat', [HEAT_LIGHT, HEAT_DARK])
    fig, axes = plt.subplots(len(HEAT_ROWS), len(scen), figsize=(4.3 * len(scen), 11.2),
                             squeeze=False)
    notes = []
    for c, (sc, title) in enumerate(scen):
        s = d[d.scenario == sc]
        al = sorted(s.alpha.unique())
        be = sorted(s.beta.unique())
        for r, (col, lab, low_better, fmt) in enumerate(HEAT_ROWS):
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
            if r == len(HEAT_ROWS) - 1:
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
    return save(fig, out, 'param_heatmap'), '; '.join(notes)


# ----------------------------------------------------------- spike size (SI)
SPIKE_SIZES = [2, 5, 10, 20, 50]
SPIKE_ROWS = [('default', 'default: baselines N=7, wavelet-based N=64'),
            ('eq64', 'equal window: all methods N=64')]


def fig_spike_size(results, out):
    """Delta Rank against spike size, common perturbation."""
    src = rp(results, 'robustness', 'robustness_summary.csv')
    if not src.exists():
        return None, f'input missing: {src}'
    d = pd.read_csv(src)
    d = d[d.condition.isin([f'size{z}' for z in SPIKE_SIZES])].copy()
    d['size'] = d.condition.str[4:].astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharex=True)
    missing, notes = [], []
    spike_methods = set()
    for c, (sc, title) in enumerate(SCENARIOS):
        for r, (cfg, rlab) in enumerate(SPIKE_ROWS):
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
            spike_methods.update(g.method.unique())
            for k, m in enumerate(method_order(g.method.unique())):
                h = g[g.method == m].sort_values('size')
                y = h['dr_mean'].clip(lower=0.1)
                col, mk, lw = method_style(m)
                ax.plot(h['size'], y, color=col, lw=lw * 0.6, marker=mk, ms=5,
                        mec=SURF, mew=0.6, zorder=3 + k)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(SPIKE_SIZES)
            ax.set_xticklabels([str(z) for z in SPIKE_SIZES])
            ax.set_xlim(1.7, 75)
            w = g[(g.method == 'WSPI') & (g['size'] == 10)]
            if len(w):
                notes.append(f"{sc}/{cfg}: WSPI 10x {float(w['dr_mean'].iloc[0]):.2f}")
    h = all_method_handles(spike_methods)
    fig.legend(handles=h, loc='lower center', ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Rank displacement of 50 low-activity items against spike size (spike in the last '
                 'slot, one slot); same items and spike for every method; mean of 5 seeds',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    files = save(fig, out, 'spike_size')
    msg = '; '.join(notes) + (f'; MISSING: {", ".join(missing)}' if missing else '')
    return files, msg


# ------------------------------------------------------------------ controlled shift test (SI)
SHIFT_BANDS = [('cv_E_L', 'lowpass'), ('cv_E_1', 'level 1'), ('cv_E_2', 'level 2'),
             ('cv_E_3', 'level 3')]
SHIFT_TR = {'DTCWT': '#2a78d6', 'DWT': '#b5b4ae'}
SHIFT_METHODS = {                    # method: (colour, marker, line style)
    'WSPI':     ('#2a78d6', 'o', '-'),
    'DWT-WSPI': ('#2a78d6', 'o', ':'),
    'DTCWT+AF': ('#1baf7a', '^', '-'),
    'DWT+AF':   ('#1baf7a', '^', ':'),
    'SMA':      ('#eb6834', 's', '-'),
    'EWMA-eq':  ('#4a3aa7', 'D', '-'),
}
SHIFT_SETTINGS = [('spike', 'middle'), ('burst3', 'middle'), ('spike', 'recent'), ('burst3', 'recent')]


def fig_shift_invariance(results, out):
    """Controlled shift-invariance test: real data (circular shift) and a
    synthetic moving event (settings of the index)."""
    root = rp(results, 'shift_invariance')
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
        x = np.arange(len(SHIFT_BANDS))
        for j, (tr, col) in enumerate(SHIFT_TR.items()):
            y = [float(g[(g['transform'] == tr) & (g['metric'] == m)]['mean'].iloc[0]) for m, _ in SHIFT_BANDS]
            ax.bar(x + (j - 0.5) * 0.38, y, width=0.36, color=col, label=tr, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([b for _, b in SHIFT_BANDS], fontsize=8)
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
        for m, (col, mk, ls) in SHIFT_METHODS.items():
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
        x = np.arange(len(SHIFT_SETTINGS))
        w = 0.8 / len(SHIFT_METHODS)
        for j, (m, (col, mk, ls)) in enumerate(SHIFT_METHODS.items()):
            y, lo, hi = [], [], []
            for shp, age in SHIFT_SETTINGS:
                r = s[(s['shape'] == shp) & (s['age'] == age) & (s['name'] == m)]
                y.append(float(r['mean'].iloc[0]) if len(r) else np.nan)
                lo.append(float(r['ci_low'].iloc[0]) if len(r) else np.nan)
                hi.append(float(r['ci_high'].iloc[0]) if len(r) else np.nan)
            y, lo, hi = map(np.asarray, (y, lo, hi))
            xx = x + (j - (len(SHIFT_METHODS) - 1) / 2) * w
            ax.bar(xx, y, width=w * 0.92, color=col, alpha=1.0 if ls == '-' else 0.45,
                   hatch=None if ls == '-' else '//', edgecolor=SURF, label=m, zorder=3)
            ax.errorbar(xx, y, yerr=[y - lo, hi - y], fmt='none', ecolor=INK2, lw=0.8, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{"spike" if a_ == "spike" else "3-slot burst"}\n{b_} part'
                            for a_, b_ in SHIFT_SETTINGS], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_ylabel('share of the 8 steps', color=INK2, fontsize=9)
        ax.legend(ncol=3, frameon=False, fontsize=8, loc='upper left')
        r = s[(s['shape'] == 'burst3') & (s['age'] == 'recent')]
        notes.append('burst3/recent share_wrong: ' + ', '.join(
            f"{n} {float(v):.3f}" for n, v in zip(r['name'], r['mean'])))
    else:
        missing.append('synthetic_summary')
    h = [Patch(color=c, label=t) for t, c in SHIFT_TR.items()]
    fig.legend(handles=h, loc='upper right', ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.99, 1.0))
    fig.suptitle('Controlled shift test: (a-d) real 64-slot windows shifted circularly '
                 '(content fixed); (e-f) synthetic event with the settings of each method',
                 fontsize=11, color=INK, x=0.45)
    files = save(fig, out, 'shift_invariance')
    msg = '; '.join(notes) + (f'; MISSING: {", ".join(missing)}' if missing else '')
    return files, msg


def fig_feature_relation(results, out):
    """R against WE over all WSPI item-windows: density of the item-windows
    and the bounds that the identity WE log2(J+1) = h(R) + (1-R) H(q) puts on WE
    for a given R (J = 3)."""
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    root = rp(results, 'feature_relation')
    have = [(sc, t) for sc, t in SCENARIOS if (root / sc / 'density.csv').exists()]
    if not have:
        return None, f'input missing: {root}/<scenario>/density.csv'
    cmap = LinearSegmentedColormap.from_list('density', ['#eaf2fc', '#2a78d6', '#0d2f5c'])
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
    files = save(fig, out, 'feature_relation')
    msg = '; '.join(notes) + (f'; MISSING: {", ".join(missing)}' if missing else '')
    return files, msg


COST_STYLE = {   # method: colour, marker, line style, line width
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
COST_BASE = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF']


def fig_runtime(results, out):
    """Measured cost of the nine protocol-V5 scorers, one CPU core.
    (a) time to score all M items against M, default windows (baselines 7,
    wavelet-based 64); (b) microseconds per item-window against N, M = 1e4;
    (c) traced working memory per item against N, batch 1e4."""
    root = rp(results, 'runtime')
    gf, tf = root / 'bench' / 'runtime_grid.csv', root / 'memory' / 'tracemalloc_grid.csv'
    if not gf.exists():
        return None, f'input missing: {gf}'
    g = pd.read_csv(gf)
    t = pd.read_csv(tf) if tf.exists() else None
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    notes = []
    ax = axes[0]
    style_axes(ax)
    for m, (col, mk, ls, lw) in COST_STYLE.items():
        nd = 7 if m in COST_BASE else 64
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
    for m, (col, mk, ls, lw) in COST_STYLE.items():
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
        for m, (col, mk, ls, lw) in COST_STYLE.items():
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
    files = save(fig, out, 'runtime')
    return files, '; '.join(notes)


# =================================================================== main results
# Main-text figures of the paper rebuilt from the causal protocol-V5 runs,
# with the visual style of paper V4 (evaluation/cross_dataset_visualizer.py:
# Paired colours, light-grey axes, hatched wavelet-based bars).  The style
# constants are copied here, not imported, because importing that module
# changes the global matplotlib style of every other figure of this program.
# The same loaders are used by scripts/generate_tables.py, so figures and
# tables share numbers.

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
# (YouTube 1 day, taxi 1 week = the bootstrap blocks; MovieLens 4 weeks
# and 13 weeks = the sensitivity blocks of the MovieLens statistics)
TIME_ROLL = {'youtube_hourly': 24, 'taxi_hourly': 168,
             'movielens_daily': 28, 'movielens_weekly': 13}
TIME_UNIT = {'youtube_hourly': 'hours', 'taxi_hourly': 'hours',
             'movielens_daily': 'days', 'movielens_weekly': 'weeks'}


def v4_summary_path(results, config, scenario):
    """comparison/summary_common_windows.csv of one run (paper configuration)."""
    if scenario.startswith('movielens'):
        base = rp(results, 'movielens', config, scenario)
        if config == 'equal64':
            base = base / 'W064'
    elif config == 'default':
        base = rp(results, 'main_default', scenario)
    elif config == 'equal64':
        base = rp(results, 'window_sweep', scenario, 'W064')
    else:
        raise ValueError(config)
    return base / 'comparison' / 'summary_common_windows.csv'


def v4_protocol_dir(results, config, scenario):
    return v4_summary_path(results, config, scenario).parent.parent / 'protocol'


def _stats_rows(results, config, scenario, kind):
    """Rows of the block-bootstrap statistics (kind = method_summary | paired_tests)."""
    name = f'all_{kind}.csv'
    if scenario.startswith('movielens'):
        d = pd.read_csv(rp(results, 'stats_movielens', name))
        return d[(d['run_group'] == config) & (d['scenario'] == scenario)]
    if config == 'default':
        d = pd.read_csv(rp(results, 'stats_default', name))
        return d[(d['run_group'] == PATHS['default_run_group']) & (d['scenario'] == scenario)]
    # window-sweep statistics: run_group = scenario, scenario = window folder
    d = pd.read_csv(rp(results, 'stats_window_sweep', name))
    return d[(d['run_group'] == scenario) & (d['scenario'] == 'W064')]


def v4_values(results, config, scenarios, metrics=MAIN_METRICS):
    """Mean, 95 % block-bootstrap CI and ties share for every method.

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
            raise FileNotFoundError(f'no block-bootstrap statistics for {config}/{sc}')
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
    """Paired tests against WSPI: verdict, Holm block p, Cliff's delta."""
    out = []
    for sc in scenarios:
        t = _stats_rows(results, config, sc, 'paired_tests')
        t = t[t['metric'].isin(metrics)].assign(config=config)
        if config != 'default' and not sc.startswith('movielens'):
            # window-sweep statistics keep the scenario in run_group and the
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


def _fig_bars(results, out, metric, name):
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
                                      f'common windows {nwin}; whiskers = 95% block-bootstrap CI')


def fig_bars_ndcg10(results, out):
    """NDCG@10, four scenarios, nine methods."""
    return _fig_bars(results, out, 'ndcg@10', 'bars_ndcg10')


def fig_bars_spearman(results, out):
    """Spearman rho."""
    return _fig_bars(results, out, 'spearman_rho', 'bars_spearman')


def fig_bars_rsi10(results, out):
    """RSI@10."""
    return _fig_bars(results, out, 'rsi@10', 'bars_rsi10')


def fig_bars_deltarank(results, out):
    """Delta Rank (log scale; values span 4 to about 215)."""
    return _fig_bars(results, out, 'robustness_distortion', 'bars_deltarank')


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
    for k, m in enumerate(method_order(methods)):
        c, mk, lw = method_style(m)
        ax.plot(df['time'], df[m].rolling(roll, center=True, min_periods=roll // 2).mean(),
                color=c, lw=lw * 0.6, zorder=3 + k, label=m)
    ax.set_ylabel(f'RSI@10  (rolling mean, {roll} {unit})', fontsize=10)
    ax.set_ylim(None, 1.01)


def _fig_time(results, out, scenario, name, title):
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


def fig_time_youtube(results, out):
    """Window-by-window RSI@10, YouTube (rolling mean 24 h)."""
    return _fig_time(results, out, 'youtube_hourly', 'time_youtube_rsi',
                          'YouTube Hourly — RSI@10 over the common evaluation windows')


def fig_time_taxi(results, out):
    """Window-by-window RSI@10, NYC Taxi hourly (rolling mean 168 h)."""
    return _fig_time(results, out, 'taxi_hourly', 'time_taxi_rsi',
                          'NYC Yellow Taxi Hourly — RSI@10 over the common evaluation windows')


def v4_best_traditional(vals, scenario, metric, exclude=('PFRF',)):
    """Best baseline (PFRF excluded, as in V4) for one scenario and metric."""
    v = vals[(vals['scenario'] == scenario) & (vals['metric'] == metric)
             & vals['method'].isin([m for m in V4_BASELINES if m not in exclude])]
    r = v.loc[v['mean'].idxmin()] if metric == 'robustness_distortion' else v.loc[v['mean'].idxmax()]
    return r


def fig_granularity(results, out):
    """Effect of temporal granularity (taxi hourly, 30 min, 5 min).
    Panels (a) RSI@10, (b) Delta Rank, (c) NDCG@10 (the accuracy side
    of the accuracy-stability trade-off).  Lines: WSPI, DTCWT+AF and the best baseline of
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
    return _save_v4(fig, out, 'granularity'), \
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


def fig_bars_movielens(results, out):
    """MovieLens bars: default configuration,
    NDCG@10, rho, RSI@10, Delta Rank (log), daily and weekly, 95 % CI."""
    try:
        return ml_bars_figure(results, out, 'default', 'bars_movielens')
    except (FileNotFoundError, KeyError) as e:
        return None, f'input missing: {e}'


def fig_time_movielens(results, out):
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
    return _save_v4(fig, out, 'time_movielens_rsi'), '; '.join(msg)


# =================================================================== printed size: paper figures
TEXT_WIDTH_IN = 6.5          # letter paper, 1 in margins (WSPI_ScientificReports.tex)
FONT_MIN, FONT_MAX = 7.0, 9.0
PNG_DPI = 600

# paper name: (function, printed width as share of \textwidth, height in, legend columns)
PAPER_FIGURES = {
    'fig_ndcg10': (fig_bars_ndcg10, 0.92, 2.9, 5),
    'fig_spearman': (fig_bars_spearman, 0.92, 2.9, 5),
    'fig_rsi10': (fig_bars_rsi10, 0.92, 2.9, 5),
    'fig_deltarank': (fig_bars_deltarank, 0.92, 2.9, 5),
    'fig_youtube_rsi_time': (lambda r, o: fig_time_all_colours(
        r, o, 'youtube_hourly', 'time_youtube_rsi',
        'YouTube Hourly — RSI@10 over the common evaluation windows'), 0.92, 2.6, 5),
    'fig_taxi_rsi_time': (lambda r, o: fig_time_all_colours(
        r, o, 'taxi_hourly', 'time_taxi_rsi',
        'NYC Yellow Taxi Hourly — RSI@10 over the common evaluation windows'), 0.92, 2.6, 5),
    'fig_granularity': (fig_granularity, 1.00, 2.7, 3),
    'fig_movielens': (fig_bars_movielens, 0.85, 4.1, 5),
    'fig_surge': (fig_surge_examples_2col, 0.95, 7.6, 4),
}


REPO = ROOT


def _v4_line_palette():
    """Colours and line widths of the V4 line charts (evaluation/visualizer.py:
    METHOD_COLORS and _lw), read from that file inside an rc_context so that its
    global rcParams do not leak into the other figures."""
    with matplotlib.rc_context():
        spec = importlib.util.spec_from_file_location(
            'v4_visualizer', REPO / 'evaluation' / 'visualizer.py')
        vis = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(vis)
        return dict(vis.METHOD_COLORS), {m: vis._lw(m) for m in vis.METHOD_COLORS}


def fig_time_all_colours(results, out, scenario, name, title):
    """RSI@10 over time (common windows, date axis, rolling mean
    of one period, same axes and title), but every one of the nine methods is
    drawn in its own V4 colour (no grey group).  Only the colours change."""
    try:
        df, n = v4_window_series(results, 'default', scenario)
    except (FileNotFoundError, KeyError, ValueError) as e:
        return None, f'input missing or inconsistent: {e}'
    colours, widths = _v4_line_palette()
    roll = TIME_ROLL[scenario]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    _v4_axes(ax)
    ax.xaxis.grid(True, alpha=0.9, linewidth=1.0, color='white')
    order = V4_BASELINES + ['DWT+AF', 'DTCWT+AF', 'WSPI']   # WSPI drawn last (on top)
    for k, m in enumerate(order):
        ax.plot(df['time'], df[m].rolling(roll, center=True, min_periods=roll // 2).mean(),
                color=colours[m], lw=widths[m] * 0.6, zorder=3 + k, label=m)
    ax.set_ylabel(f'RSI@10  (rolling mean, {roll} {TIME_UNIT[scenario]})', fontsize=10)
    ax.set_ylim(None, 1.01)
    ax.set_title(title, fontsize=10.5)
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=5,
                    frameon=False, fontsize=9)
    for t in leg.get_texts():
        if t.get_text() in V4_WAVELET:
            t.set_color(V4_PURPLE)
            t.set_fontweight('bold')
    return _save_v4(fig, out, name), f'{n} common windows, rolling mean {roll} (centred); V4 colours'


def _clip(size):
    return min(max(size, FONT_MIN), FONT_MAX)


def tidy(fig):
    """Label layout for the printed size; no data is touched.
    - y labels 'Metric  (note)' are broken into two lines;
    - scenario tick labels are broken into two lines ('NYC Yellow Taxi' / 'Hourly');
    - date axes: ticks every Monday ('07 May') for a period under 90 days,
      otherwise at most 7 ticks with a concise date format;
    - the legend under two-line tick labels moves down to clear them;
    - point annotations use FONT_MIN."""
    for ax in fig.axes:
        yl = ax.get_ylabel()
        if '  (' in yl:
            ax.set_ylabel(yl.replace('  (', '\n(', 1))
        labs = [t.get_text() for t in ax.get_xticklabels()]
        if any(l.startswith(('NYC Yellow Taxi ', 'YouTube ')) for l in labs):
            new = [l.replace('NYC Yellow Taxi ', 'NYC Yellow Taxi\n')
                    .replace('YouTube ', 'YouTube\n') for l in labs]
            ax.set_xticks(ax.get_xticks(), new)
            leg = ax.get_legend()
            if leg is not None:        # legend below two-line tick labels
                leg.set_bbox_to_anchor((0.5, -0.27), transform=ax.transAxes)
        if isinstance(ax.xaxis.get_major_formatter(), mdates.AutoDateFormatter):
            lo, hi = ax.get_xlim()
            if hi - lo < 90:           # short period (YouTube): ticks every Monday, '07 May'
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
            else:
                loc = mdates.AutoDateLocator(maxticks=7)
                ax.xaxis.set_major_locator(loc)
                ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        for ch in ax.get_children():
            if isinstance(ch, Annotation):
                ch.set_fontsize(FONT_MIN)


@contextmanager
def paper_mode(size, ncols, sink, paper_name):
    """Wrap figure size, font sizes, legend columns and saving for one figure."""
    orig_subplots, orig_figure = plt.subplots, plt.figure
    orig_set_size = font_manager.FontProperties.set_size
    orig_legend_init = mlegend.Legend.__init__
    orig_save, orig_save_v4 = globals()['save'], globals()['_save_v4']

    def subplots(*a, **kw):
        kw['figsize'] = size
        kw.setdefault('layout', 'constrained')
        return orig_subplots(*a, **kw)

    def figure(*a, **kw):
        kw['figsize'] = size
        kw.setdefault('layout', 'constrained')
        return orig_figure(*a, **kw)

    def set_size(self, s):
        orig_set_size(self, s)
        orig_set_size(self, _clip(self.get_size_in_points()))

    def legend_init(self, *a, **kw):
        if 'ncol' in kw or 'ncols' in kw:
            kw.pop('ncol', None)
            kw['ncols'] = ncols
        orig_legend_init(self, *a, **kw)

    def save_any(fig, out, name):
        tidy(fig)
        if not fig.get_layout_engine() or 'Constrained' not in type(fig.get_layout_engine()).__name__:
            fig.set_layout_engine('constrained')
        files = []
        for ext in ('pdf', 'png'):
            f = Path(out) / f'{paper_name}.{ext}'
            fig.savefig(f, dpi=PNG_DPI, bbox_inches='tight', pad_inches=0.02,
                        metadata={'CreationDate': None} if ext == 'pdf' else None)
            files.append(f.name)
        sink['source_function_name'] = name
        plt.close(fig)
        return files

    plt.subplots, plt.figure = subplots, figure
    font_manager.FontProperties.set_size = set_size
    mlegend.Legend.__init__ = legend_init
    globals()['save'] = globals()['_save_v4'] = save_any
    try:
        yield
    finally:
        plt.subplots, plt.figure = orig_subplots, orig_figure
        font_manager.FontProperties.set_size = orig_set_size
        mlegend.Legend.__init__ = orig_legend_init
        globals()['save'], globals()['_save_v4'] = orig_save, orig_save_v4


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# =================================================================== printed size: SI figures
LANDSCAPE_WIDTH_IN = 9.0      # text height of the letter page, used for the landscape SI figures

# SI name: (function, printed width in, height in, legend columns, landscape page)
SI_FIGURES = {
    'si_window_curves': (fig_window_curves, LANDSCAPE_WIDTH_IN, 5.6, 5, True),
    'si_feature_relation': (fig_feature_relation, 6.5, 2.4, 2, False),
    'si_spike_size': (fig_spike_size, LANDSCAPE_WIDTH_IN, 5.6, 5, True),
    'si_runtime': (fig_runtime, 6.5, 2.7, 5, False),
    'si_shift_invariance': (fig_shift_invariance, LANDSCAPE_WIDTH_IN, 5.2, 3, True),
    'si_param_heatmap': (fig_param_heatmap, LANDSCAPE_WIDTH_IN, 6.2, 2, True),
    'si_movielens_rsi_time': (fig_time_movielens, 6.5, 4.6, 6, False),
    'si_delay_ecdf': (fig_delay_ecdf, LANDSCAPE_WIDTH_IN, 5.6, 5, True),
}

# long panel titles that the caption repeats: shortened to the panel letter
SHORT_TITLES = {'si_runtime', 'si_shift_invariance'}
# per-panel text boxes that Supplementary Table S2 repeats
DROP_AXES_TEXT = {'si_feature_relation'}
# cell labels of the heat maps: kept below the 7 pt floor, so that they fit the cells
CELL_LABEL_PT = 5.0


def si_fix(name, fig, ncols):
    """Layout of one SI figure at printed size; no data is touched.
    - the suptitle is removed (the caption carries it);
    - legends drawn below the panels become one legend outside the panels
      (constrained layout then keeps it clear of the axis labels);
    - listed figures: long panel titles shortened to '(a)', text boxes removed."""
    if fig._suptitle is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    handles, labels = [], []
    for leg in list(fig.legends):
        for h, t in zip(leg.legend_handles, leg.get_texts()):
            if t.get_text() not in labels:
                handles.append(h)
                labels.append(t.get_text())
        leg.remove()
    if name in ('si_feature_relation', 'si_runtime'):
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            for h, t in zip(leg.legend_handles, leg.get_texts()):
                lab = t.get_text()
                # the same method may be labelled 'WSPI (N=64)' in one panel and 'WSPI' in another
                if lab.split(' (')[0] not in [x.split(' (')[0] for x in labels]:
                    handles.append(h)
                    labels.append(lab)
            leg.remove()
    if handles:
        fig.legend(handles, labels, loc='outside lower center', ncols=ncols, frameon=False,
                   fontsize=FONT_MIN + 0.5)
    for ax in fig.axes:
        yl, xl = ax.get_ylabel(), ax.get_xlabel()
        if name in ('si_spike_size', 'si_delay_ecdf') and yl:
            first = 'Default' if 'default' in yl.lower() else 'Equal window of 64' if 'equal' in yl.lower() else ''
            second = '$\\Delta$Rank (log scale)' if name == 'si_spike_size' else 'share of entries in Top-10'
            ax.set_ylabel(f'{first}\n{second}' if first else second)
        if name == 'si_spike_size' and xl:
            ax.set_xlabel('spike size (x mean of 64 slots)')
        if name == 'si_feature_relation':
            if xl:
                ax.set_xlabel('$R$')
            if yl.startswith('WE'):
                ax.set_ylabel('$W_E$')
        if name == 'si_runtime' and yl:
            ax.set_ylabel({'(a)': 'seconds per window', '(b)': 'microseconds per item-window',
                           '(c)': 'traced bytes per item'}.get(ax.get_title()[:3], yl))
        if name == 'si_shift_invariance' and len(ax.get_xticklabels()) and any(
                t.get_text().startswith('level') for t in ax.get_xticklabels()):
            ax.set_xticks(ax.get_xticks(), [t.get_text().replace('lowpass', 'low').replace('level ', '')
                                            for t in ax.get_xticklabels()])
            ax.set_xlabel('band')
        if name == 'si_param_heatmap':
            for axis in (ax.xaxis, ax.yaxis):
                labs = [t.get_text() for t in axis.get_ticklabels()]
                if labs:
                    axis.set_ticks(axis.get_ticklocs(), [l.replace('0.', '.') if l.startswith('0.') else l
                                                          for l in labs])
        if name in SHORT_TITLES:
            t = ax.get_title()
            if t.startswith('(') and ')' in t:
                ax.set_title(t[:t.index(')') + 1])
        if name in DROP_AXES_TEXT:
            for tx in list(ax.texts):
                tx.remove()
        if name == 'si_param_heatmap':
            for tx in ax.texts:     # bypasses the 7-9 pt clip of paper_mode on purpose
                tx.get_fontproperties()._size = CELL_LABEL_PT
                if tx.get_text().startswith('0.'):    # '0.9667' -> '.9667', so that it fits the cell
                    tx.set_text(tx.get_text()[1:])


class si_tidy:
    """Run si_fix after the tidy of the paper target, inside its save step."""

    def __init__(self, name, ncols):
        self.name, self.ncols = name, ncols

    def __enter__(self):
        self.orig = globals()['tidy']

        def tidy(fig):
            self.orig(fig)
            si_fix(self.name, fig, self.ncols)
        globals()['tidy'] = tidy

    def __exit__(self, *exc):
        globals()['tidy'] = self.orig


class no_tight_layout:
    """The figures are drawn with the constrained layout of paper_mode; a call of
    Figure.tight_layout inside a figure function (which cannot switch engines once a
    colorbar exists) is skipped while an SI figure is drawn."""

    def __enter__(self):
        self.orig = Figure.tight_layout
        Figure.tight_layout = lambda self, *a, **kw: None

    def __exit__(self, *exc):
        Figure.tight_layout = self.orig


# =================================================================== thesis figures
PERIOD_SPLIT = pd.Timestamp('2015-01-01', tz='UTC')
TAXI_ROLL = {'taxi_30min': 336, 'taxi_5min': 2016}     # one week, as the bootstrap blocks


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
            df, n = v4_window_series(results, 'default', scenario)
        except (FileNotFoundError, KeyError, ValueError) as e:
            return None, f'input missing or inconsistent: {e}'
        roll = TAXI_ROLL[scenario]
        fig, ax = plt.subplots(figsize=(12, 4.6))
        _v4_time_panel(ax, df, roll, 'slots')
        ax.set_title(title, fontsize=10.5)
        leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=6, frameon=False, fontsize=9)
        for t in leg.get_texts():
            if t.get_text() in V4_WAVELET:
                t.set_color(V4_PURPLE)
                t.set_fontweight('bold')
        return _save_v4(fig, out, name), f'{n} common windows, rolling mean {roll} (one week)'
    return run


def ml_equal64(results, out):
    try:
        return ml_bars_figure(results, out, 'equal64', 'thesis_ml_equal64')
    except (FileNotFoundError, KeyError) as e:
        return None, f'input missing: {e}'


def ml_profile(results, out):
    p = rp(results, 'movielens', 'data_prep', 'year_profile.csv')
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
        _v4_axes(ax)
        ax.bar(d['year'], d[col] * k, color=c, edgecolor=V4_EDGE, linewidth=0.5, zorder=3)
        ax.set_title(title, fontsize=10.5)
        ax.tick_params(axis='x', labelsize=8)
    axes[2].set_ylim(0, 1)
    fig.suptitle('MovieLens 32M, 1998-2023 (last year up to 12 Oct 2023)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save_v4(fig, out, 'thesis_ml_profile'), f'{len(d)} years from {p.name}'


def ml_periods(results, out):
    """Mean of the four metrics before and after 1 Jan 2015 (window time stamp),
    default configuration, common windows; rows daily and weekly."""
    rows = []
    try:
        for sc, lab in ML_SCENARIOS:
            for met in MAIN_METRICS:
                df, _ = v4_window_series(results, 'default', sc, metric=met)
                early = df['time'] < PERIOD_SPLIT
                for per, mask in (('1998-2014', early), ('2015-2023', ~early)):
                    for m in V4_ORDER:
                        rows.append(dict(scenario=sc, period=per, metric=met, method=m,
                                         mean=float(df.loc[mask, m].mean()),
                                         n=int(df.loc[mask, m].notna().sum())))
    except (FileNotFoundError, KeyError, ValueError) as e:
        return None, f'input missing or inconsistent: {e}'
    v = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6))
    periods = ['1998-2014', '2015-2023']
    width = 0.8 / len(V4_ORDER)
    offsets = (np.arange(len(V4_ORDER)) - (len(V4_ORDER) - 1) / 2.0) * width
    for i, (sc, lab) in enumerate(ML_SCENARIOS):
        for j, met in enumerate(MAIN_METRICS):
            ax = axes[i, j]
            _v4_axes(ax)
            for k, m in enumerate(V4_ORDER):
                ys = [v[(v.scenario == sc) & (v.period == p) & (v.metric == met) & (v.method == m)]['mean'].iloc[0]
                      for p in periods]
                kw = dict(width=width, color=V4_COLORS[m], zorder=3)
                if m in V4_WAVELET:
                    kw.update(edgecolor=V4_EDGE, linewidth=V4_EDGE_LW, hatch=V4_HATCH)
                else:
                    kw.update(edgecolor='white', linewidth=0.4)
                ax.bar(np.arange(2) + offsets[k], ys, **kw)
            ax.set_xticks(np.arange(2))
            ax.set_xticklabels(periods, fontsize=9)
            if met == 'robustness_distortion':
                ax.set_yscale('log')
            else:
                ax.set_ylim(0, 1.05)
            ax.set_title(f'{lab} \u2014 {V4_SHORT[met]}', fontsize=10)
    n = v[(v.metric == 'ndcg@10') & (v.method == 'WSPI')].set_index(['scenario', 'period'])['n']
    _v4_legend(fig, V4_ORDER, anchor=(0.5, 0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save_v4(fig, out, 'thesis_ml_periods'), \
        'windows per period: ' + ', '.join(f'{s}/{p}={c}' for (s, p), c in n.items())


# thesis figure (V1.5 numbering) -> (id, status, builder or note)
THESIS_FIGURES = [
    ('4-1 overview, YouTube', 'yt_overview', 'planned', None),
    ('4-2 RSI at depths 5/10/20, YouTube', 'yt_rsi_depths', 'planned', None),
    ('4-3 RSI@10 over time, YouTube', 'thesis_fig4_03_youtube_rsi_time', 'paper',
     _paper(fig_time_youtube, 'thesis_fig4_03_youtube_rsi_time')),
    ('4-4 robustness, YouTube', 'yt_robustness', 'planned', None),
    ('4-5 metric heatmap, YouTube', 'yt_heatmap', 'planned', None),
    ('4-6 overview, taxi hourly', 'th_overview', 'planned', None),
    ('4-7 RSI at depths, taxi hourly', 'th_rsi_depths', 'planned', None),
    ('4-8 RSI@10 over time, taxi hourly', 'thesis_fig4_08_taxi_rsi_time', 'paper',
     _paper(fig_time_taxi, 'thesis_fig4_08_taxi_rsi_time')),
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
     _paper(fig_bars_ndcg10, 'thesis_fig4_18_ndcg10')),
    ('4-19 Spearman, grouped by scenario', 'thesis_fig4_19_spearman', 'paper',
     _paper(fig_bars_spearman, 'thesis_fig4_19_spearman')),
    ('4-20 RSI@10, grouped by scenario', 'thesis_fig4_20_rsi10', 'paper',
     _paper(fig_bars_rsi10, 'thesis_fig4_20_rsi10')),
    ('4-21 Delta Rank, grouped by scenario', 'thesis_fig4_21_deltarank', 'paper',
     _paper(fig_bars_deltarank, 'thesis_fig4_21_deltarank')),
    ('4-22..4-25 the four metrics grouped by method', 'method_view', 'planned', None),
    ('4-26 effect of granularity', 'thesis_fig4_26_granularity', 'paper',
     _paper(fig_granularity, 'thesis_fig4_26_granularity')),
    ('4-27 sparsity against granularity', 'sparsity', 'planned', None),
    ('4-28..4-31 prediction method (WSPI-F2)', 'prediction', 'other', None),
    ('4-32 ablation', 'ablation', 'planned', None),
    ('4-33 sensitivity of alpha and beta', 'sensitivity', 'planned', None),
    ('new: MovieLens, default configuration', 'thesis_ml_default', 'paper',
     _paper(fig_bars_movielens, 'thesis_ml_default')),
    ('new: MovieLens, equal window 64', 'thesis_ml_equal64', 'ready', ml_equal64),
    ('new: MovieLens, RSI@10 over time', 'thesis_ml_rsi_time', 'paper',
     _paper(fig_time_movielens, 'thesis_ml_rsi_time')),
    ('new: MovieLens data profile', 'thesis_ml_profile', 'ready', ml_profile),
    ('new: MovieLens 1998-2014 against 2015-2023', 'thesis_ml_periods', 'ready', ml_periods),
]
BUILDERS = {fid: fn for _, fid, st, fn in THESIS_FIGURES if fn is not None}


# =================================================================== main
DRAFT_FIGURES = {
    'window_curves': fig_window_curves,
    'tradeoff': fig_tradeoff,                  # kept in the program only
    'surge_examples': fig_surge_examples,
    'surge_examples_2col': fig_surge_examples_2col,
    'delay_ecdf': fig_delay_ecdf,
    'param_heatmap': fig_param_heatmap,
    'spike_size': fig_spike_size,
    'shift_invariance': fig_shift_invariance,
    'feature_relation': fig_feature_relation,
    'runtime': fig_runtime,
    'bars_ndcg10': fig_bars_ndcg10,
    'bars_spearman': fig_bars_spearman,
    'bars_rsi10': fig_bars_rsi10,
    'bars_deltarank': fig_bars_deltarank,
    'time_youtube_rsi': fig_time_youtube,
    'time_taxi_rsi': fig_time_taxi,
    'granularity': fig_granularity,
    'bars_movielens': fig_bars_movielens,
    'time_movielens_rsi': fig_time_movielens,
}


def run_draft(res, out, only):
    plt.rcParams.update({'font.size': 10})
    ok = 0
    for fid in (only or DRAFT_FIGURES):
        files, msg = DRAFT_FIGURES[fid](res, out)
        if files is None:
            print(f'[skip] {fid}: {msg}')
        else:
            ok += 1
            print(f'[ok]   {fid}: {", ".join(files)}\n       {msg}')
    print(f'{ok} of {len(only or DRAFT_FIGURES)} figures written to {out}')


def run_thesis(res, out, only):
    plt.rcParams.update({'font.size': 10})
    ids = only or list(BUILDERS)
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


def run_printed(target, res, out, only):
    """Paper and SI figures at printed size."""
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['font.size'] = FONT_MIN + 1
    table = PAPER_FIGURES if target == 'paper' else SI_FIGURES
    meta = dict(program=f'scripts/generate_figures.py --target {target}',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                matplotlib=matplotlib.__version__, font_range_pt=[FONT_MIN, FONT_MAX],
                text_width_in=TEXT_WIDTH_IN, png_dpi=PNG_DPI, fonttype=42, figures={})
    ok = 0
    names = only or list(table)
    for name in names:
        sink = {}
        if target == 'paper':
            func, share, height, ncols = table[name]
            size = (round(share * TEXT_WIDTH_IN, 3), height)
            with paper_mode(size, ncols, sink, name):
                res_f = func(res, out)
            extra = {}
        else:
            func, width, height, ncols, landscape = table[name]
            size = (width, height)
            with paper_mode(size, ncols, sink, name), no_tight_layout(), si_tidy(name, ncols):
                res_f = func(res, out)
            extra = dict(landscape=landscape)
        files, note = res_f if isinstance(res_f, tuple) else (res_f, '')
        if not files:
            print(f'[skip] {name}: {note}')
            continue
        ok += 1
        meta['figures'][name] = dict(function=func.__name__, figsize_in=size, legend_ncols=ncols, **extra,
                                     note=note, md5={f: md5(out / f) for f in files})
        print(f'[ok]   {name}: {", ".join(files)}')
    (out / f'figures_{target}_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'{ok} of {len(names)} figures written to {out}')


def main():
    ap = argparse.ArgumentParser(description='Figures of the paper, the SI and the thesis')
    ap.add_argument('--target', required=True, choices=['paper', 'si', 'thesis', 'draft'])
    ap.add_argument('--results')
    ap.add_argument('--out')
    ap.add_argument('--only', nargs='*', default=None)
    ap.add_argument('--list', action='store_true', help='thesis target: print the registry and stop')
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
    known = {'paper': PAPER_FIGURES, 'si': SI_FIGURES, 'thesis': BUILDERS, 'draft': DRAFT_FIGURES}[a.target]
    bad = [n for n in (a.only or []) if n not in known]
    if bad:
        ap.error(f'unknown figure names for --target {a.target}: {bad}; known: {list(known)}')
    out.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context():
        if a.target == 'draft':
            run_draft(res, out, a.only)
        elif a.target == 'thesis':
            run_thesis(res, out, a.only)
        else:
            run_printed(a.target, res, out, a.only)


if __name__ == '__main__':
    main()
