r"""
Slope diagnostic — does a trend/slope term earn its place? (fast, real data)
============================================================================
Settles, empirically and on the ACTUAL dataset, whether a slope/trend feature
adds ranking information BEYOND what mu_L already captures.

For many (window, future-horizon) samples drawn from the real series it
computes mu_L (exactly as WSPI does) and SEVERAL slope formulations, then the
target = mean popularity over the next horizon. It reports, per slope variant:

    Spearman(slope, target)        raw predictive corr (can be inflated by mu_L)
    Spearman(slope, mu_L)          redundancy with mu_L
    PARTIAL(slope, target | mu_L)  the decisive number: info BEYOND mu_L

Interpretation:
    partial >  ~+0.10  -> slope adds real, usable signal; worth a slope term
    partial ~   0      -> slope is redundant with mu_L; skip it
    partial <  ~-0.10  -> trends mean-revert; slope MISLEADS (would hurt)

It also reports lag-1 autocorrelation of item series (a persistence proxy):
high -> trends persist (slope likely helps); low/negative -> bursty/reverting.

This does NOT run the full evaluator; it's a feature-level probe and finishes
in a couple of minutes.

Usage (Windows):
    python experiments\diagnose_slope.py youtube
    python experiments\diagnose_slope.py yellow_taxi --data-path data\datasets\yellow_taxi_2025_all_hourly.csv
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.stats import rankdata

from methods.hybrid_assessment import HybridAssessment
from methods.wspi_candidates import _CandidateBase
import experiments.run_popularity_assessment as runner


def _spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan
    return float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1])


def _partial(s, t, u):
    rst, rsu, rtu = _spear(s, t), _spear(s, u), _spear(t, u)
    d = np.sqrt(max(0.0, (1 - rsu ** 2) * (1 - rtu ** 2)))
    return (rst - rsu * rtu) / d if d > 1e-9 else np.nan


def _lag1(x):
    x = np.asarray(x, float)
    if x.size < 3 or np.std(x[:-1]) < 1e-12 or np.std(x[1:]) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


# Slope formulations on a raw window -----------------------------------------
def slope_variants(win):
    n = len(win)
    t = np.arange(n)
    mean = win.mean() + 1e-8
    raw = np.polyfit(t, win, 1)[0]                       # OLS slope, raw units
    norm = raw / mean                                    # relative growth rate
    # last-third minus first-third, normalised (robust direction)
    k = max(1, n // 3)
    block = (win[-k:].mean() - win[:k].mean()) / mean
    # sign-consistency (efficiency): net change / total path
    diffs = np.diff(win)
    path = np.sum(np.abs(diffs)) + 1e-12
    eff = float(np.sum(diffs)) / path                    # in [-1, 1]
    return {'slope_raw_z': raw, 'slope_norm(S_L-like)': norm,
            'block_3rds': block, 'efficiency_signed': eff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset')
    ap.add_argument('--data-path', default=None)
    ap.add_argument('--window', type=int, default=64)
    ap.add_argument('--horizon', type=int, default=24)
    ap.add_argument('--max-items', type=int, default=400)
    ap.add_argument('--samples-per-item', type=int, default=8)
    args = ap.parse_args()

    loader = runner.get_data_loader(args.dataset, data_path=args.data_path)
    df = loader.load_data()
    ic = 'item_id'  if 'item_id'  in df.columns else getattr(loader, 'item_col', 'item_id')
    cc = 'count'    if 'count'    in df.columns else getattr(loader, 'count_col', 'count')
    tc = 'timestamp' if 'timestamp' in df.columns else getattr(loader, 'time_col', None)

    totals = df.groupby(ic)[cc].sum().sort_values(ascending=False)
    hyb = HybridAssessment()
    base = _CandidateBase()

    rng = np.random.RandomState(0)
    W, H = args.window, args.horizon
    mu_list, tgt_list, lag1_list = [], [], []
    var_lists = {k: [] for k in
                 ['slope_raw_z', 'slope_norm(S_L-like)', 'block_3rds', 'efficiency_signed']}

    for item in totals.index[:args.max_items]:
        sub = df[df[ic] == item]
        if tc and tc in sub.columns:
            sub = sub.sort_values(tc)
        s = sub[cc].to_numpy(dtype=np.float64)
        if len(s) < W + H + 1:
            continue
        lag1_list.append(_lag1(s))
        starts = rng.randint(0, len(s) - W - H, size=args.samples_per_item)
        for st in starts:
            win = s[st:st + W]
            fut = s[st + W:st + W + H]
            if win.std() < 1e-9:
                continue
            try:
                mu_L, _Sm, _R, _WE = base._extract_features(win)
            except Exception:
                continue
            mu_list.append(mu_L)
            tgt_list.append(float(fut.mean()))
            sv = slope_variants(win)
            for k in var_lists:
                var_lists[k].append(sv[k])

    mu = np.array(mu_list); tgt = np.array(tgt_list)
    print('=' * 74)
    print(f'SLOPE DIAGNOSTIC  dataset={args.dataset}  samples={len(mu)}  '
          f'window={W} horizon={H}')
    print('=' * 74)
    med_lag1 = np.nanmedian(lag1_list)
    print(f'median lag-1 autocorrelation of series = {med_lag1:.3f}  '
          f'({"persistent -> slope may help" if med_lag1 > 0.2 else "weak/bursty -> slope likely wont help"})')
    print(f'Spearman(mu_L , target)               = {_spear(mu, tgt):.4f}')
    print('-' * 74)
    print(f'{"slope variant":24s}{"Sp(slope,tgt)":>15s}{"Sp(slope,mu)":>14s}{"PARTIAL|mu":>13s}')
    for k, v in var_lists.items():
        v = np.array(v)
        print(f'{k:24s}{_spear(v, tgt):15.4f}{_spear(v, mu):14.4f}{_partial(v, tgt, mu):13.4f}')
    print('=' * 74)
    print('Decision rule: PARTIAL|mu  > +0.10 => add a slope term;'
          '  ~0 => redundant;  < -0.10 => slope misleads.')


if __name__ == '__main__':
    main()
