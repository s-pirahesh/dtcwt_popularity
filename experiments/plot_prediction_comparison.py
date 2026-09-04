# -*- coding: utf-8 -*-
"""
Prediction-Comparison Figures (Level 1 + Level 2)
=================================================
Reads a run folder (e.g. results/<ds>/predcmp_<ts>) that contains the
cached thesis methods AND the newly computed prediction methods
(Persistence, Holt, ARYW, ARIMA, WSPI-F, WSPI-F-YW), and draws the
comparison figures for the new thesis subsections:

  fig_pred_ndcg10.png      NDCG@10        (rank-prediction accuracy, Top-K)
  fig_pred_spearman.png    Spearman rho   (rank-prediction accuracy, full list)
  fig_pred_rsi10.png       RSI@10         (temporal stability)
  fig_pred_deltarank.png   Robustness distortion (lower = better)
  fig_pred_overview.png    all four panels in one figure

Color coding (fixed, entity-based):
  blue   = wavelet-structural methods (WSPI, WSPI-F, WSPI-F-YW, WSPI-F2, WSPI-FT)
  orange = explicit value forecasters (Persistence, Holt, ARYW, ARIMA)
  gray   = classical popularity baselines (AF, EWMA, ...)

METHOD SELECTION (added Aug 2026)
---------------------------------
With 17 methods a single chart is unreadable. Four ways to narrow it, in
precedence order --methods > --preset > (all), then --exclude, then --top-n:

    --methods A B C     plot exactly these, in the order given
    --preset core       a named set (see PRESETS below); --list-presets to see
    --exclude X Y       drop these
    --top-n 8           keep only the 8 best on each chart's OWN metric
                        (proposed methods are always kept — see ALWAYS_KEEP)

`--top-n` is applied per-figure, so each panel shows the leaders for its own
metric. Every figure prints, and records in its caption file, how many methods
were dropped — a chart that silently hides half the field is a chart that
lies.

Usage:
    python experiments\\plot_prediction_comparison.py results\\youtube\\predcmp_<ts>
    python experiments\\plot_prediction_comparison.py <folder> --tag youtube --preset core
    python experiments\\plot_prediction_comparison.py <folder> --methods WSPI WSPI-FT ARIMA
    python experiments\\plot_prediction_comparison.py <folder> --preset forecast --top-n 8
    :: re-plot from an existing summary CSV, no protocol folder needed:
    python experiments\\plot_prediction_comparison.py --from-csv prediction_summary_youtube.csv --tag youtube

Author: Sajjad (with Claude)
Date: August 2026
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# NOTE: imported lazily inside load_summary() so that --from-csv works even
# when the protocol-reading dependencies are unavailable.

# ---- validated palette ------------------------------------------------------
C_BLUE = '#2a78d6'      # wavelet-structural family (proposed)
C_ORANGE = '#eb6834'    # explicit forecasters (new baselines)
C_GRAY = '#a8a7a0'      # classical baselines
C_TEXT = '#0b0b0b'
C_TEXT2 = '#52514e'
C_GRID = '#e4e3df'
C_SURFACE = '#fcfcfb'

WAVELET_FAMILY = {'WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT',
                  'DWT+AF', 'DTCWT+AF'}
STRUCTURAL = {'WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT'}
FORECASTERS = {'Persistence', 'Holt', 'ARYW', 'ARIMA'}
CLASSICAL = {'AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF'}

# Never dropped by --top-n: the thesis's own methods must stay visible even
# when they lose on a given metric. Hiding them would be cherry-picking.
ALWAYS_KEEP = STRUCTURAL

# filename-safe stems -> display names (protocol files may use either form)
DISPLAY_NAMES = {'DWTPLUSAF': 'DWT+AF', 'DTCWTPLUSAF': 'DTCWT+AF'}

# ---- named method sets ------------------------------------------------------
# Order matters: it is preserved in --methods and used as the drawing order
# hint. Unknown / not-yet-computed names are skipped with a warning.
PRESETS = {
    # main text: proposed family + the strongest classical + the AR baseline
    'core': ['WSPI', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT', 'DTCWT+AF', 'ARYW'],

    # subsection 4-5-4: everything that forecasts
    'forecast': ['WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT',
                 'Persistence', 'Holt', 'ARYW', 'ARIMA'],

    # component-contribution analysis: wavelet family only
    'wavelet': ['WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT',
                'DWT+AF', 'DTCWT+AF'],

    # ablation of the Level-2 module: predictor choice, architecture fixed
    'ablation': ['WSPI', 'WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT'],

    # classical baselines vs the proposed index
    'classic': ['WSPI', 'AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF'],

    # appendix: current behaviour, everything present
    'all': None,
}

METRICS = [
    ('ndcg@10',               'NDCG@10 — rank-prediction accuracy (Top-10)', True),
    ('spearman_rho',          'Spearman ρ — full-list rank prediction',      True),
    ('rsi@10',                'RSI@10 — temporal stability',                 True),
    ('robustness_distortion', 'ΔRank — noise-injection distortion',          False),
]


def color_for(method: str) -> str:
    if method in STRUCTURAL:
        return C_BLUE
    if method in FORECASTERS:
        return C_ORANGE
    return C_GRAY


def select_methods(df: pd.DataFrame, methods=None, preset=None,
                   exclude=None) -> pd.DataFrame:
    """
    Apply --methods / --preset / --exclude to the summary frame.

    Precedence: explicit --methods wins over --preset. Ordering follows the
    requested list so the caller controls the legend/axis order. Names that
    were requested but are not present (e.g. ARIMA still running) are reported
    rather than silently dropped.
    """
    present = list(df['method'])
    wanted = None

    if methods:
        wanted = list(methods)
    elif preset:
        if preset not in PRESETS:
            raise SystemExit(
                f"Unknown preset '{preset}'. Available: {', '.join(PRESETS)}")
        wanted = PRESETS[preset]          # None for 'all'

    if wanted is not None:
        missing = [m for m in wanted if m not in present]
        if missing:
            print(f'  ! requested but not available: {", ".join(missing)}')
        keep = [m for m in wanted if m in present]
        if not keep:
            raise SystemExit('Selection left no methods to plot.')
        df = df.set_index('method').loc[keep].reset_index()

    if exclude:
        before = len(df)
        df = df[~df['method'].isin(set(exclude))].reset_index(drop=True)
        if len(df) < before:
            print(f'  - excluded {before - len(df)} method(s)')
        if df.empty:
            raise SystemExit('Exclusion left no methods to plot.')

    return df


def apply_top_n(sub: pd.DataFrame, col: str, higher_better: bool,
                top_n: int) -> tuple:
    """
    Keep the top-N rows on `col`, plus every ALWAYS_KEEP method.

    Returns (frame, n_dropped). Sorting is the caller's job.
    """
    if not top_n or top_n >= len(sub):
        return sub, 0
    ranked = sub.sort_values(col, ascending=not higher_better)
    keep = set(ranked.head(top_n)['method'])
    keep |= (set(sub['method']) & ALWAYS_KEEP)
    out = sub[sub['method'].isin(keep)]
    return out, len(sub) - len(out)


def load_summary(run_dir: Path) -> pd.DataFrame:
    from experiments._auto_extract import (_discover_methods, _load_protocol,
                                           _summarise_method)
    rows = []
    for m in _discover_methods(run_dir):
        df = _load_protocol(run_dir, m)
        if df is None or df.empty:
            continue
        row = {'method': DISPLAY_NAMES.get(m, m)}
        row.update(_summarise_method(df))
        rows.append(row)
    if not rows:
        raise SystemExit(f'No protocol files found in {run_dir}/protocol')
    return pd.DataFrame(rows)


def draw_metric(ax, df, col, title, higher_better, top_n: int = 0):
    sub = df[df[col].notna()].copy()
    sub, n_dropped = apply_top_n(sub, col, higher_better, top_n)
    sub = sub.sort_values(col, ascending=not higher_better)
    y = np.arange(len(sub))[::-1]
    vals = sub[col].to_numpy()
    colors = [color_for(m) for m in sub['method']]

    ax.set_facecolor(C_SURFACE)
    ax.grid(True, axis='x', color=C_GRID, linewidth=0.8, zorder=0)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT2, labelsize=9)
    ax.barh(y, vals, height=0.62, color=colors, zorder=3)

    # value labels at bar ends (text in ink, not series color)
    span = float(vals.max() - min(vals.min(), 0)) or 1.0
    for yi, v, m in zip(y, vals, sub['method']):
        ax.text(v + 0.012 * span, yi, f'{v:.3g}', va='center',
                fontsize=8.2, color=C_TEXT2, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub['method'], fontsize=9,
                       color=C_TEXT)
    # bold the proposed methods' tick labels
    for tick, m in zip(ax.get_yticklabels(), sub['method']):
        if m in STRUCTURAL:
            tick.set_fontweight('bold')
    ax.set_xlim(0, vals.max() * 1.14)
    arrow = '↑ higher is better' if higher_better else '↓ lower is better'
    subtitle = arrow
    if n_dropped:
        # never hide a truncation: state it on the figure itself
        subtitle += f'   ·   top-{top_n} shown, {n_dropped} method(s) not shown'
    ax.set_title(f'{title}\n{subtitle}', color=C_TEXT, fontsize=10, loc='left')
    return n_dropped


def add_legend(fig):
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(color=C_BLUE, label='Wavelet-structural (WSPI / WSPI-F)'),
        mpatches.Patch(color=C_ORANGE, label='Explicit value forecasters'),
        mpatches.Patch(color=C_GRAY, label='Classical baselines'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False,
               fontsize=9, labelcolor=C_TEXT2)


def main():
    ap = argparse.ArgumentParser(
        description='Draw prediction-comparison figures from a run folder.')
    ap.add_argument('run_dir', type=str, nargs='?', default=None,
                    help='run folder (e.g. results/youtube/predcmp_<ts>)')
    ap.add_argument('--tag', type=str, default=None,
                    help='filename tag (default: run folder name)')
    ap.add_argument('--from-csv', type=str, default=None,
                    help='re-plot from an existing prediction_summary_*.csv '
                         'instead of reading the protocol folder')
    ap.add_argument('--methods', type=str, nargs='+', default=None,
                    help='plot exactly these methods, in this order')
    ap.add_argument('--exclude', type=str, nargs='+', default=None,
                    help='drop these methods')
    ap.add_argument('--preset', type=str, default=None,
                    help='named method set: ' + ', '.join(PRESETS))
    ap.add_argument('--top-n', type=int, default=0,
                    help="keep only the N best on each chart's own metric "
                         '(proposed methods are always kept)')
    ap.add_argument('--list-presets', action='store_true',
                    help='print the preset definitions and exit')
    ap.add_argument('--suffix', type=str, default=None,
                    help='extra filename suffix (default: preset name, so '
                         'different selections do not overwrite each other)')
    args = ap.parse_args()

    if args.list_presets:
        print('\nAvailable presets:\n')
        for k, v in PRESETS.items():
            print(f'  {k:<10} {"(everything present)" if v is None else ", ".join(v)}')
        print()
        return

    # ---- load ---------------------------------------------------------------
    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.is_absolute():
            csv_path = ROOT / csv_path
        df = pd.read_csv(csv_path)
        tag = args.tag or csv_path.stem.replace('prediction_summary_', '')
        out_dir = csv_path.parent
    else:
        if not args.run_dir:
            raise SystemExit('Provide a run folder, or use --from-csv.')
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        tag = args.tag or run_dir.name
        out_dir = run_dir / 'figures_prediction'
        out_dir.mkdir(parents=True, exist_ok=True)
        df = load_summary(run_dir)

    print('Methods found:', ', '.join(df['method']))

    # ---- select -------------------------------------------------------------
    n_all = len(df)
    df = select_methods(df, methods=args.methods, preset=args.preset,
                        exclude=args.exclude)
    if len(df) != n_all:
        print(f'Plotting {len(df)} of {n_all}:', ', '.join(df['method']))

    # keep selections in separate files rather than overwriting each other
    sel = args.suffix if args.suffix is not None else (
        args.preset if args.preset else ('sel' if (args.methods or args.exclude)
                                         else ''))
    if sel:
        tag = f'{tag}_{sel}'

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # individual figures
    for col, title, hb in METRICS:
        if col not in df.columns or df[col].isna().all():
            print(f'  (skip {col}: not present)')
            continue
        fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(df) + 1.6), dpi=200)
        fig.patch.set_facecolor(C_SURFACE)
        draw_metric(ax, df, col, title, hb, top_n=args.top_n)
        add_legend(fig)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        p = out_dir / f'fig_pred_{col.replace("@", "").replace("_rho", "")}_{tag}.png'
        fig.savefig(p, facecolor=C_SURFACE, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {p.name}')

    # 2x2 overview
    avail = [(c, t, h) for c, t, h in METRICS
             if c in df.columns and not df[c].isna().all()]
    if len(avail) >= 2:
        rows = int(np.ceil(len(avail) / 2))
        fig, axes = plt.subplots(rows, 2,
                                 figsize=(13.5, rows * (0.4 * len(df) + 1.5)),
                                 dpi=200)
        fig.patch.set_facecolor(C_SURFACE)
        axes = np.atleast_1d(axes).ravel()
        for ax, (c, t, h) in zip(axes, avail):
            draw_metric(ax, df, c, t, h, top_n=args.top_n)
        for ax in axes[len(avail):]:
            ax.set_visible(False)
        add_legend(fig)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        p = out_dir / f'fig_pred_overview_{tag}.png'
        fig.savefig(p, facecolor=C_SURFACE, bbox_inches='tight')
        plt.close(fig)
        print(f'  -> {p.name}')

    # tidy summary CSV next to the figures
    df.to_csv(out_dir / f'prediction_summary_{tag}.csv',
              index=False, encoding='utf-8-sig')
    print(f'  -> prediction_summary_{tag}.csv')


if __name__ == '__main__':
    main()
