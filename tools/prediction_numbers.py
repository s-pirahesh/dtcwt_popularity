r"""
Numbers of the prediction runs, collected into one CSV file
===========================================================
Reads the result folders of runs A and B under ``--root`` (default
``results/prediction``): the paired-test reports ``stats/``, the run summaries
and the bound checks.  Nothing is computed that is not already in a result
file; the means are cross-checked against ``summary_common_windows.csv``.
Writes ``<root>/prediction_numbers.csv`` with one row per quantity (run,
scenario, horizon, quantity, statistic, value, source file).  The figure and
table program ``tools/prediction_figures.py`` reads that file.

The program stops with exit code 1 and lists the problems if a folder is
missing, if a mean differs from the run summary, if the window check has a row
that is not OK, or if a score bound was violated.

    python tools/prediction_numbers.py
    python tools/prediction_numbers.py --root results/prediction
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENARIOS = ('youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min',
             'movielens_daily', 'movielens_weekly')


def hist_quantile(lo: np.ndarray, hi: np.ndarray, cnt: np.ndarray, q: float) -> float:
    """Quantile of binned data (linear inside the bin)."""
    c = np.cumsum(cnt)
    tot = c[-1]
    if tot == 0:
        return float('nan')
    target = q * tot
    i = int(np.searchsorted(c, target))
    prev = c[i - 1] if i > 0 else 0
    frac = (target - prev) / cnt[i] if cnt[i] else 0.0
    return float(lo[i] + frac * (hi[i] - lo[i]))


RUNS = {
    'A': (SCENARIOS, ('WSPI', 'WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
                      'Persistence', 'Holt', 'ARYW', 'MA64'), 'WSPI-FG'),
    'B': (('youtube_hourly', 'taxi_hourly'), ('WSPI', 'WSPI-FA', 'ARIMA'), 'WSPI-FA'),
}
HORIZONS = (1, 8, 24)
METRICS = ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion')


def run_summary(a):
    from methods.wspi_forecast import (BACKTEST_K, CLAMP, GATE_GAMMA,
                                       n_forecast_coefficients, ratio_bounds)
    root = Path(a.root)
    if not root.is_absolute():
        root = ROOT / root
    rows, problems = [], []

    def add(run, scen, h, qty, stat, val, src):
        rows.append({'run': run, 'scenario': scen, 'horizon': h, 'quantity': qty,
                     'statistic': stat, 'value': val, 'source': src})

    def rd(f):
        return pd.read_csv(f, float_precision='round_trip')

    for run, (scenarios, methods, ref) in RUNS.items():
        for s in scenarios:
            for h in HORIZONS:
                rel = f'run_{run}/{s}/h{h}'
                d = root / rel
                st = root / 'stats' / rel / f'ref_{ref}'
                if not (d / 'metadata' / 'protocol_v5_run.json').exists():
                    problems.append(f'missing run folder {rel}')
                    continue
                if not (st / 'method_summary.csv').exists():
                    problems.append(f'missing stats/{rel}/ref_{ref}')
                    continue
                sm = json.loads((st / 'metadata' / 'stats_run.json').read_text(encoding='utf-8'))
                add(run, s, h, 'windows', 'n_common', float(sm['n_common_windows']),
                    f'stats/{rel}/ref_{ref}/metadata/stats_run.json')
                add(run, s, h, 'block', 'windows', float(sm['block']),
                    f'stats/{rel}/ref_{ref}/metadata/stats_run.json')
                if set(sm['methods']) != set(methods):
                    problems.append(f'{rel}: methods {sorted(sm["methods"])}')
                ms = rd(st / 'method_summary.csv')
                sc = rd(d / 'comparison' / 'summary_common_windows.csv').set_index('method')
                for c in METRICS:
                    for m in methods:
                        r = ms[(ms['metric'] == c) & (ms['method'] == m)]
                        if len(r) != 1:
                            problems.append(f'{rel}: no summary row for {m} {c}')
                            continue
                        r = r.iloc[0]
                        for stat in ('mean', 'ci_low', 'ci_high'):
                            add(run, s, h, f'{c}:{stat}', m, float(r[stat]),
                                f'stats/{rel}/ref_{ref}/method_summary.csv')
                        if not abs(float(sc.loc[m, f'{c}_mean']) - float(r['mean'])) < 1e-9:
                            problems.append(f'{rel}: mean of {m} {c} differs from '
                                            f'summary_common_windows.csv')
                meta = json.loads((d / 'metadata' / 'protocol_v5_run.json').read_text(encoding='utf-8'))
                rt = meta.get('runtime_seconds', {})
                tm = rd(d / 'scorer_timing.csv').set_index('method')
                for m in methods:
                    add(run, s, h, 'runtime_s', m, float(rt[m]), f'{rel}/metadata/protocol_v5_run.json')
                    add(run, s, h, 'scorer_s', m, float(tm.loc[m, 'scorer_seconds']),
                        f'{rel}/scorer_timing.csv')
                if run == 'A':
                    for m in ('WSPI-FG', 'WSPI-F2'):
                        add(run, s, h, 'runtime_ratio', f'{m}/WSPI', rt[m] / rt['WSPI'],
                            f'{rel}/metadata/protocol_v5_run.json')
                    gp = rd(d / 'gate_profile.csv')
                    gh = rd(d / 'gate_hist.csv')
                    add(run, s, h, 'gate_fg', 'mean_share_zero', float(gp['fg_share_zero'].mean()),
                        f'{rel}/gate_profile.csv')
                    add(run, s, h, 'gate_fg', 'median_of_window_median',
                        float(gp['fg_g_median'].median()), f'{rel}/gate_profile.csv')
                    lo, hi = gh['bin_lo'].to_numpy(), gh['bin_hi'].to_numpy()
                    add(run, s, h, 'gate_f2', 'pooled_median',
                        hist_quantile(lo, hi, gh['count_g'].to_numpy(), 0.5), f'{rel}/gate_hist.csv')
                    add(run, s, h, 'gate_fg', 'pooled_median',
                        hist_quantile(lo, hi, gh['count_g_fg'].to_numpy(), 0.5), f'{rel}/gate_hist.csv')
                    add(run, s, h, 'gate', 'n_item_windows', float(gh['count_g_fg'].sum()),
                        f'{rel}/gate_hist.csv')
                else:
                    add(run, s, h, 'runtime_ratio', 'WSPI-FA/ARIMA (scorer)',
                        float(tm.loc['WSPI-FA', 'scorer_seconds'] / tm.loc['ARIMA', 'scorer_seconds']),
                        f'{rel}/scorer_timing.csv')
                    add(run, s, h, 'fits', 'WSPI-FA', float(tm.loc['WSPI-FA', 'n_fits']),
                        f'{rel}/scorer_timing.csv')
                    add(run, s, h, 'fallback_share', 'WSPI-FA',
                        float(tm.loc['WSPI-FA', 'fallback_share']), f'{rel}/scorer_timing.csv')
                bf = d / 'bound_check.csv'
                if bf.exists():
                    for _, r in rd(bf).iterrows():
                        add(run, s, h, 'ratio_min', r['method'], float(r['min_ratio']), f'{rel}/bound_check.csv')
                        add(run, s, h, 'ratio_max', r['method'], float(r['max_ratio']), f'{rel}/bound_check.csv')
                        if r['status'] != 'OK':
                            problems.append(f'{rel}: bound violated by {r["method"]}')
                else:
                    problems.append(f'missing {rel}/bound_check.csv')

    wf = root / 'stats' / 'window_check.csv'
    if wf.exists():
        w = rd(wf)
        bad = w[w['status'] != 'OK']
        add('all', 'all', 0, 'window_check', 'rows_not_ok', float(len(bad)), 'stats/window_check.csv')
        if len(bad):
            problems.append(f'window check: {len(bad)} rows not OK')
    else:
        problems.append('missing stats/window_check.csv')

    nl = 16
    add('all', 'all', 0, 'backtest_k', 'value', float(BACKTEST_K), 'methods/wspi_forecast.py')
    add('all', 'all', 0, 'clamp_c', 'value', float(CLAMP), 'methods/wspi_forecast.py')
    add('all', 'all', 0, 'gate_gamma', 'value', float(GATE_GAMMA), 'methods/wspi_forecast.py')
    for h in HORIZONS:
        m = n_forecast_coefficients(h)
        lo, up = ratio_bounds(nl, m, CLAMP)
        add('all', 'all', h, 'forecast_coefficients', 'm', float(m), 'ceil(h / 2^(J-1)), J = 3')
        add('all', 'all', h, 'bound', 'lower', lo, f'ratio_bounds(n = {nl}, m, c)')
        add('all', 'all', h, 'bound', 'upper', up, f'ratio_bounds(n = {nl}, m, c)')
        add('all', 'all', h, 'bound', 'rank_threshold', up / lo, 'upper / lower')

    df = pd.DataFrame(rows)
    out = root / 'prediction_numbers.csv'
    df.to_csv(out, index=False, encoding='utf-8', float_format='%.10g')
    print(f'{len(df)} rows -> {out}')
    if problems:
        print('PROBLEMS:')
        for p in problems:
            print('  ' + p)
        sys.exit(1)
    print('cross-check OK: all folders present, means equal the run summaries, '
          'window check clean, no bound violated')


def main():
    ap = argparse.ArgumentParser(description='Collect the numbers of the prediction runs into one CSV file')
    ap.add_argument('--root', default='results/prediction', help='result folder (default: results/prediction)')
    run_summary(ap.parse_args())


if __name__ == '__main__':
    main()
