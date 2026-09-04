# -*- coding: utf-8 -*-
"""
Paired per-window significance test between methods of one run folder.
===================================================================
The summary CSV reports MEANS over windows. A mean difference of 0.5% is
meaningless unless it survives a paired test across the windows themselves —
this script supplies that test, which is what a defence committee will ask for.

Wilcoxon signed-rank on the per-window metric, paired by window_id over the
INTERSECTION of windows both methods produced (so unequal window counts can
never bias the comparison).

Direction is decided by WIN/LOSS COUNTS, not by the median: with many tied
windows the median is exactly 0.0 while the distribution is clearly one-sided,
and reading direction off a zero median silently inverts the conclusion.

Usage:
    python paired_significance_test.py results\youtube\predcmp_20260815_232251
    python paired_significance_test.py <run_dir> --methods WSPI WSPI-F2 WSPI-F-YW
    python paired_significance_test.py <run_dir> --baseline WSPI --alpha 0.05
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    sys.exit("scipy is required:  pip install scipy")

# (column, higher_is_better)
METRICS = [
    ('ndcg@10', True),
    ('spearman_rho', True),
    ('rsi@10', True),
    ('robustness_distortion', False),
]


def load(run_dir: Path, name: str) -> pd.DataFrame:
    """Protocol files are CSV (not parquet, despite older docs)."""
    p = run_dir / 'protocol' / f'{name}_protocol.csv'
    if not p.exists():
        raise SystemExit(f'not found: {p}')
    return pd.read_csv(p).set_index('window_id')


def compare(a_df, b_df, a, b, common, alpha, n_tests):
    print(f'\n{a}  vs  {b}    (paired over {len(common)} common windows)')
    print(f'  {"metric":<22}{"mean diff":>12}{"win":>6}{"loss":>6}{"tie":>6}'
          f'{"p":>12}   verdict')
    for col, higher_better in METRICS:
        x = a_df.loc[common, col]
        y = b_df.loc[common, col]
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        diff = (x - y).to_numpy()

        # direction from win/loss counts, NOT the median
        if higher_better:
            win, loss = int((diff > 0).sum()), int((diff < 0).sum())
        else:
            win, loss = int((diff < 0).sum()), int((diff > 0).sum())
        tie = int((diff == 0).sum())

        if win + loss < 10:
            print(f'  {col:<22}{diff.mean():>12.5f}{win:>6}{loss:>6}{tie:>6}'
                  f'{"—":>12}   too few non-tied windows')
            continue

        p = float(wilcoxon(x, y, zero_method='wilcox').pvalue)
        thr = alpha / n_tests                      # Bonferroni
        if p >= thr:
            verdict = 'no significant difference'
        else:
            verdict = 'BETTER' if win > loss else 'worse'
        star = '***' if p < thr / 100 else '**' if p < thr / 10 else \
               '*' if p < thr else 'ns'
        print(f'  {col:<22}{diff.mean():>12.5f}{win:>6}{loss:>6}{tie:>6}'
              f'{p:>12.2e} {star:<4}{verdict}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--baseline', default='WSPI')
    ap.add_argument('--methods', nargs='+',
                    default=['WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT'])
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    names = [args.baseline] + list(args.methods)
    dfs = {n: load(run_dir, n) for n in names}

    common = sorted(set.intersection(*[set(d.index) for d in dfs.values()]))
    counts = {n: len(d) for n, d in dfs.items()}
    print(f'windows per method: {counts}')
    print(f'common windows    : {len(common)}')
    if len(set(counts.values())) > 1:
        print('  ! window counts differ — the test uses the intersection only')

    # Bonferroni over every test this run prints
    n_tests = len(args.methods) * len(METRICS)
    print(f'\nBonferroni: {n_tests} tests, alpha={args.alpha} '
          f'-> significance threshold p < {args.alpha / n_tests:.2e}')

    for m in args.methods:
        compare(dfs[m], dfs[args.baseline], m, args.baseline,
                common, args.alpha, n_tests)


if __name__ == '__main__':
    main()
