# -*- coding: utf-8 -*-
"""
WSPI-F Convergence & Coefficient-Forecast Experiment (Hypothesis 4 evidence)
============================================================================
Produces the two empirical results of the Level-2 (WSPI-F) module:

  Panel A — ADAPTATION: the NLMS predictor's normalised squared error along
            the DTCWT trend-band coefficient sequence, averaged over items.
            A decreasing curve = "prediction error shrinks over time as the
            model adapts" — the literal claim of Hypothesis 4 in the proposal.

  Panel B — COEFFICIENT FORECAST QUALITY: one-step-ahead relative error of
            the NLMS coefficient forecast vs the persistence forecast
            (next coeff = last coeff), averaged over items. Shows the
            adaptive predictor beats the naive one in the coefficient domain.

Input: any binned CSV with (time, item, count) columns — e.g.
    data\\datasets\\yellow_taxi_2025_all_hourly.csv

Usage:
    python experiments\\run_wspiF_convergence.py data\\datasets\\yellow_taxi_2025_all_hourly.csv
    python experiments\\run_wspiF_convergence.py <csv> --time-col timestamp --item-col item_id --count-col count
    python experiments\\run_wspiF_convergence.py <csv> --num-items 200 --tag taxi_hourly

Outputs (results/wspiF_convergence/):
    <tag>_adaptation.csv        per-step mean/std error curve
    <tag>_forecast_quality.csv  per-item NLMS vs persistence relative errors
    <tag>_convergence.png       the two-panel figure

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

from methods.wspi_forecast import nlms_walk  # noqa: E402
import dtcwt  # noqa: E402
from config import WAVELET_CONFIG  # noqa: E402

# ---- figure style (validated palette; see SELECTIVE_RUN_GUIDE) -------------
C_BLUE = '#2a78d6'      # WSPI-F / NLMS
C_ORANGE = '#eb6834'    # persistence / reference
C_TEXT = '#0b0b0b'
C_TEXT2 = '#52514e'
C_GRID = '#e4e3df'
C_SURFACE = '#fcfcfb'


def load_series(csv_path, time_col, item_col, count_col, num_items):
    """Pivot the CSV into {item: 1-D count series} for the top-N items."""
    df = pd.read_csv(csv_path)
    for col in (time_col, item_col, count_col):
        if col not in df.columns:
            raise SystemExit(
                f"Column '{col}' not in CSV. Available: {list(df.columns)}\n"
                f"Use --time-col/--item-col/--count-col to override.")
    top = (df.groupby(item_col)[count_col].sum()
             .sort_values(ascending=False).head(num_items).index)
    df = df[df[item_col].isin(top)]
    pivot = (df.pivot_table(index=time_col, columns=item_col,
                            values=count_col, aggfunc='sum')
               .sort_index().fillna(0.0))
    return {c: pivot[c].to_numpy(dtype=float) for c in pivot.columns}


def lowpass_sequence(ts, transform, level):
    """Full-history DTCWT trend-band magnitude sequence of one item."""
    n = len(ts)
    target = 2 ** int(np.ceil(np.log2(max(n, 2 ** (level + 1)))))
    if n < target:
        ts = np.pad(ts, (target - n, 0), mode='reflect')
    pyr = transform.forward(ts, nlevels=level)
    return np.abs(np.asarray(pyr.lowpass).ravel())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('csv', type=str, help='binned dataset CSV')
    ap.add_argument('--time-col', default='timestamp')
    ap.add_argument('--item-col', default='item_id')
    ap.add_argument('--count-col', default='count')
    ap.add_argument('--num-items', type=int, default=200,
                    help='top-N items by volume (default 200)')
    ap.add_argument('--order', type=int, default=2, help='NLMS order')
    ap.add_argument('--mu', type=float, default=0.2, help='NLMS step size')
    ap.add_argument('--max-steps', type=int, default=200,
                    help='truncate curves to this many steps (default 200)')
    ap.add_argument('--tag', type=str, default=None,
                    help='output filename tag (default: csv stem)')
    args = ap.parse_args()

    tag = args.tag or Path(args.csv).stem
    out_dir = ROOT / 'results' / 'wspiF_convergence'
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = dtcwt.Transform1d(biort=WAVELET_CONFIG['dtcwt_biort'],
                                  qshift=WAVELET_CONFIG['dtcwt_qshift'])
    level = WAVELET_CONFIG['decomposition_level']

    print(f'Loading {args.csv} (top {args.num_items} items)...')
    series = load_series(args.csv, args.time_col, args.item_col,
                         args.count_col, args.num_items)
    print(f'  {len(series)} items, series length {len(next(iter(series.values())))}')

    curves, rel_nlms, rel_pers = [], [], []
    for item, ts in series.items():
        a = lowpass_sequence(ts, transform, level)
        scale = float(np.mean(a ** 2))
        if scale <= 0 or len(a) < args.order + 4:
            continue
        _, errors, preds = nlms_walk(a, order=args.order, mu=args.mu)
        curves.append((errors ** 2) / scale)

        # one-step relative errors: NLMS vs persistence, same targets
        # (nlms_walk predicts a[k] for k = order+1 .. n-1)
        targets = a[args.order + 1:]
        pers_preds = a[args.order:-1]              # previous coefficient
        denom = np.maximum(np.abs(targets), 1e-9)
        rel_nlms.append(float(np.mean(np.abs(targets - preds) / denom)))
        rel_pers.append(float(np.mean(np.abs(targets - pers_preds) / denom)))

    if not curves:
        raise SystemExit('No usable items (series too short?).')

    # ---- Panel A data: per-step median/IQR error (robust, log-safe) --------
    L = min(args.max_steps, min(len(c) for c in curves))
    mat = np.vstack([c[:L] for c in curves])
    q1_curve, med_curve, q3_curve = np.percentile(mat, [25, 50, 75], axis=0)
    mean_curve = mat.mean(axis=0)

    def _smooth(v, k=9):
        if len(v) < k:
            return v
        kernel = np.ones(k) / k
        pad = k // 2
        vp = np.pad(v, pad, mode='edge')
        return np.convolve(vp, kernel, mode='valid')

    med_s, q1_s, q3_s = _smooth(med_curve), _smooth(q1_curve), _smooth(q3_curve)
    pd.DataFrame({'step': np.arange(1, L + 1),
                  'median_nmse': med_curve, 'q1_nmse': q1_curve,
                  'q3_nmse': q3_curve, 'mean_nmse': mean_curve,
                  'n_items': len(curves)}
                 ).to_csv(out_dir / f'{tag}_adaptation.csv', index=False)

    # ---- Panel B data ------------------------------------------------------
    fq = pd.DataFrame({'nlms_rel_err': rel_nlms, 'persistence_rel_err': rel_pers})
    fq.to_csv(out_dir / f'{tag}_forecast_quality.csv', index=False)

    first_half = med_curve[:L // 2].mean()
    second_half = med_curve[L // 2:].mean()
    improve = 100 * (1 - np.mean(rel_nlms) / np.mean(rel_pers))
    print(f'  Adaptation: mean nMSE first half {first_half:.4g} -> '
          f'second half {second_half:.4g} '
          f'({100 * (1 - second_half / max(first_half, 1e-12)):.0f}% lower)')
    print(f'  Forecast:   NLMS rel.err {np.mean(rel_nlms):.3f} vs '
          f'persistence {np.mean(rel_pers):.3f} ({improve:.0f}% better)')

    # ---- Figure ------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200)
    fig.patch.set_facecolor(C_SURFACE)

    for ax in (ax1, ax2):
        ax.set_facecolor(C_SURFACE)
        ax.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(C_GRID)
        ax.tick_params(colors=C_TEXT2, labelsize=9)

    # Panel A — adaptation curve: smoothed median with IQR band
    steps = np.arange(1, L + 1)
    n_it = len(curves)
    ax1.fill_between(steps, np.maximum(q1_s, 1e-12), q3_s,
                     color=C_BLUE, alpha=0.15, linewidth=0, zorder=2)
    ax1.plot(steps, med_curve, color=C_BLUE, linewidth=1,
             alpha=0.30, zorder=3)
    ax1.plot(steps, med_s, color=C_BLUE, linewidth=2, zorder=4)
    ax1.set_yscale('log')
    ax1.set_xlabel('Adaptation step (coefficient index)', color=C_TEXT2, fontsize=10)
    ax1.set_ylabel('Normalised squared prediction error', color=C_TEXT2, fontsize=10)
    ax1.set_title('(a) NLMS adaptation on DTCWT trend-band coefficients\n'
                  f'median over {n_it} items, IQR band (smoothed)',
                  color=C_TEXT, fontsize=10.5, loc='left')

    # Panel B — relative forecast error distribution: NLMS vs persistence
    data = [rel_nlms, rel_pers]
    labels = ['WSPI-F (NLMS)', 'Persistence']
    colors = [C_BLUE, C_ORANGE]
    means = [np.mean(d) for d in data]
    pos = [0, 1]
    for p, d, c in zip(pos, data, colors):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax2.vlines(p, q1, q3, color=c, linewidth=6, alpha=0.35, zorder=2)
        ax2.scatter([p], [med], color=c, s=48, zorder=4,
                    edgecolor=C_SURFACE, linewidth=2)
        ax2.text(p + 0.08, med, f'{med:.3f}', color=C_TEXT2,
                 fontsize=9, va='center')
    ax2.set_xticks(pos)
    ax2.set_xticklabels(labels, color=C_TEXT, fontsize=10)
    ax2.set_xlim(-0.5, 1.7)
    ax2.set_ylabel('One-step relative error (coefficient domain)',
                   color=C_TEXT2, fontsize=10)
    ax2.set_title('(b) Coefficient forecast error — median & IQR\n'
                  f'NLMS is {improve:.0f}% better on average',
                  color=C_TEXT, fontsize=10.5, loc='left')

    fig.tight_layout()
    fig_path = out_dir / f'{tag}_convergence.png'
    fig.savefig(fig_path, facecolor=C_SURFACE, bbox_inches='tight')
    print(f'  Figure  -> {fig_path}')
    print(f'  CSVs    -> {out_dir}')


if __name__ == '__main__':
    main()
