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


FIGURES = {
    'T2.2_window_curves': fig_t22_window_curves,   # SI figure of the paper
    'T2.3_tradeoff': fig_t23_tradeoff,             # kept in the program only, not in the paper
    'T2.4_surge_examples': fig_t24_surge_examples,  # main text (E11 examples)
    'T2.4_delay_ecdf': fig_t24_delay_ecdf,          # SI (E11 delay distribution)
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
