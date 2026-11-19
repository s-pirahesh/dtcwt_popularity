r"""
Estimate the running time of the supplementary runs (C, C_sens, C_nlms)
=======================================================================
The estimate comes from measured times, never from a guess:

  run C      a preview run of run C on one scenario and horizon (``--preview``:
             the h<h> folder of ``run_prediction.py`` with all run C methods,
             normally YouTube hourly, h = 1).  For every method the preview
             ratio  run_seconds(method) / run_seconds(WSPI)  is multiplied by the
             measured WSPI time of run A in the same scenario and horizon
             (``<run-a>/<scenario>/h<h>/scorer_timing.csv``).  GBM and GBM+W add
             their training time: the preview training seconds per training row
             times the training rows of the scenario (first 30 % of the item
             rows of the run A WSPI file, at most 200 000).
  run C_sens the 18 configurations cost what WSPI-F2 costs in run A (same steps),
             plus WSPI.
  run C_nlms two forecasters on the trend-band coefficients of every item-window:
             taken as twice the run A time of WSPI-F (NLMS) at h = 1.

Assumption: the ratio to WSPI carries over from the preview scenario to the
other scenarios (the methods work row by row on the same matrices).

    python tools\estimate_plan_runtime.py --preview results\prediction\smoke_C\youtube_hourly\h1 ^
           --run-a results\prediction\run_A --out results\prediction\smoke_C\runtime_estimate_C.csv

Prints the core hours of every run and the longest job; writes one row per job.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = ('youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min',
             'movielens_daily', 'movielens_weekly')
NLMS_SCENARIOS = ('youtube_hourly', 'taxi_hourly')
HORIZONS = (1, 8, 24)
LEARNED = ('GBM', 'GBM+W')
N_SENS = 18
MAX_ROWS = 200_000
TUNE_SHARE = 0.30


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def timing(folder: Path) -> dict:
    t = pd.read_csv(folder / 'scorer_timing.csv')
    return dict(zip(t['method'], t['run_seconds'].astype(float)))


def train_rows(folder: Path) -> int:
    w = pd.read_csv(folder / 'protocol' / 'WSPI_protocol.csv', usecols=['window_id', 'num_items'])
    n_t = int(np.floor(TUNE_SHARE * len(w)))
    return int(min(MAX_ROWS, w['num_items'].iloc[:n_t].sum()))


def learned_cost(preview: Path) -> dict:
    """Training seconds per training row of the preview."""
    out = {}
    for n in LEARNED:
        f = preview / 'metadata' / f"learned_{n.replace('+', '_')}.json"
        if f.exists():
            i = json.loads(f.read_text(encoding='utf-8'))
            out[n] = (i['rows_seconds'] + i['fit_seconds']) / max(i['n_rows_used'], 1)
    return out


def estimate(preview: Path, run_a: Path):
    prev = timing(preview)
    if 'WSPI' not in prev:
        sys.exit(f'{preview}: no WSPI row in scorer_timing.csv')
    ratio = {m: s / prev['WSPI'] for m, s in prev.items()}
    per_row = learned_cost(preview)
    rows = []
    for s in SCENARIOS:
        for h in HORIZONS:
            fa = run_a / s / f'h{h}'
            ta = timing(fa)
            n_train = train_rows(fa) if per_row else 0
            sec_c = sum(r * ta['WSPI'] for r in ratio.values())
            sec_c += sum(c * n_train for c in per_row.values())
            rows.append(dict(run='C', scenario=s, horizon=h, seconds=sec_c, train_rows=n_train))
            rows.append(dict(run='C_sens', scenario=s, horizon=h,
                             seconds=ta['WSPI'] + N_SENS * ta['WSPI-F2'], train_rows=0))
            if h == 1 and s in NLMS_SCENARIOS:
                rows.append(dict(run='C_nlms', scenario=s, horizon=h,
                                 seconds=2.0 * ta['WSPI-F'], train_rows=0))
    return pd.DataFrame(rows), ratio


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--preview', required=True, help='h<h> folder of a run C preview')
    ap.add_argument('--run-a', default='results/prediction/run_A')
    ap.add_argument('--out', default=None, help='CSV with one row per job')
    ap.add_argument('--jobs', type=int, default=6, help='parallel jobs (for the wall time)')
    a = ap.parse_args(argv)
    est, ratio = estimate(_abs(a.preview), _abs(a.run_a))
    est['hours'] = est['seconds'] / 3600.0
    if a.out:
        o = _abs(a.out)
        o.parent.mkdir(parents=True, exist_ok=True)
        est.to_csv(o, index=False, encoding='utf-8')
    print('preview ratio to WSPI: ' + ', '.join(f'{m} {r:.2f}' for m, r in ratio.items()))
    for run, d in est.groupby('run', sort=False):
        tot = d['hours'].sum()
        lng = d.loc[d['hours'].idxmax()]
        wall = max(tot / a.jobs, lng['hours'])
        print(f'run {run:<7} {tot:6.2f} core hours, {len(d)} jobs; longest '
              f"{lng['scenario']} h{int(lng['horizon'])} {lng['hours']:.2f} h; "
              f'about {wall:.1f} h with {a.jobs} jobs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
