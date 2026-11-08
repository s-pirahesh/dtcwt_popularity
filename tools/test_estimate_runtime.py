r"""
Unit checks for tools/estimate_runtime.py
=========================================
Synthetic preview, reference and timing-sample folders with round numbers, so
that every estimate can be computed by hand:

  1. Run A, no base: the time of a method at horizon h is the reference time
     times preview(h) / preview(1); WSPI-FG is taken as WSPI-F2 times the
     preview ratio FG / F2, MA64 as Persistence times MA64 / Persistence.
  2. Run A with a base run: the base times replace the reference times.
  3. Run B: fits = sum of the items + 50 robustness rows per window, scaled by
     (windows - (h - 1)) / windows; seconds per fit from the timing sample;
     ARIMA falls back to arima_timing.csv; a missing WSPI-FA timing is NaN.

    python tools/test_estimate_runtime.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import estimate_runtime as E                                   # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def timing_csv(path, h, run, rows=None, windows=100, scorer_seconds=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame({'horizon': h, 'method': list(run), 'run_seconds': list(run.values())})
    t['windows'] = windows
    t['rows'] = rows if rows is not None else 1000
    t['scorer_seconds'] = scorer_seconds if scorer_seconds is not None else 1.0
    t.to_csv(path, index=False)


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        # preview: every method 10 s at h = 1, 20 s at h = 8, 40 s at h = 24; FG is 2x F2
        prev = tmp / 'prev' / 'youtube_hourly'
        for h, f in ((1, 1.0), (8, 2.0), (24, 4.0)):
            run = {m: 10.0 * f for m in E.RUN_A_METHODS}
            run['WSPI-FG'] = 20.0 * f
            run['MA64'] = 5.0 * f                       # half of Persistence
            timing_csv(prev / f'h{h}' / 'scorer_timing.csv', h, run)
        # reference: run time 100 s for the eight old methods; 100 windows with 10 items
        ref = tmp / 'ref'
        for s in E.SCENARIOS:
            (ref / s / 'metadata').mkdir(parents=True)
            (ref / s / 'protocol').mkdir()
            old = [m for m in E.RUN_A_METHODS if m not in ('WSPI-FG', 'MA64')]
            (ref / s / 'metadata' / 'protocol_v5_run.json').write_text(
                json.dumps({'runtime_seconds': {m: 100.0 for m in old}}), encoding='utf-8')
            pd.DataFrame({'window_id': range(32, 132), 'num_items': 10}).to_csv(
                ref / s / 'protocol' / 'WSPI_protocol.csv', index=False)
        arima = tmp / 'arima_timing.csv'
        pd.DataFrame({'n_fits': [200], 'window': [64], 'seconds_per_fit': [0.01]}).to_csv(arima, index=False)

        out = tmp / 'est.csv'
        E.main(['--preview', str(prev.parent / 'youtube_hourly'), '--reference', str(ref),
                '--arima-timing', str(arima), '--out', str(out)])
        est = pd.read_csv(out)
        A = est[est['run'] == 'A'].set_index(['scenario', 'horizon', 'method'])['est_seconds']
        g = lambda s, h, m: A[(s, h, m)]                                       # noqa: E731
        check('A: WSPI at h = 8 is 100 s x 2', np.isclose(g('taxi_5min', 8, 'WSPI'), 200.0))
        check('A: WSPI at h = 24 is 100 s x 4', np.isclose(g('taxi_5min', 24, 'WSPI'), 400.0))
        check('A: WSPI-FG proxy = F2 100 s x (20/10) x horizon ratio',
              np.isclose(g('movielens_daily', 1, 'WSPI-FG'), 200.0) and
              np.isclose(g('movielens_daily', 8, 'WSPI-FG'), 400.0))
        check('A: MA64 proxy = Persistence 100 s x (5/10)',
              np.isclose(g('movielens_daily', 1, 'MA64'), 50.0))
        check('A: ten methods x three horizons x six scenarios',
              len(est[est['run'] == 'A']) == 10 * 3 * 6)

        # base run for one scenario replaces the reference times
        base = tmp / 'base'
        run = {m: 30.0 for m in E.RUN_A_METHODS}
        timing_csv(base / 'taxi_hourly' / 'h1' / 'scorer_timing.csv', 1, run, rows=5000, windows=200)
        E.main(['--preview', str(prev), '--reference', str(ref), '--base', str(base),
                '--arima-timing', str(arima), '--out', str(out)])
        est = pd.read_csv(out)
        A = est[est['run'] == 'A'].set_index(['scenario', 'horizon', 'method'])['est_seconds']
        check('A with base: WSPI-FG uses the base time (30 s x 2)',
              np.isclose(A[('taxi_hourly', 8, 'WSPI-FG')], 60.0))
        check('A with base: other scenarios keep the reference times',
              np.isclose(A[('taxi_30min', 8, 'WSPI')], 200.0))

        # run B
        B = est[est['run'] == 'B'].set_index(['scenario', 'horizon', 'method'])['est_seconds']
        fits_yt = 100 * 10 + 50 * 100                      # 6000 rows, 100 windows
        check('B: ARIMA youtube h = 1 = 6000 fits x 0.01 s',
              np.isclose(B[('youtube_hourly', 1, 'ARIMA')], fits_yt * 0.01))
        check('B: windows shrink with the horizon (h = 8: 93 of 100 windows, h = 24: 77)',
              np.isclose(B[('youtube_hourly', 8, 'ARIMA')], fits_yt * 0.93 * 0.01) and
              np.isclose(B[('youtube_hourly', 24, 'ARIMA')], fits_yt * 0.77 * 0.01))
        check('B: taxi_hourly with base: rows from the base run (5000 x 0.01 s)',
              np.isclose(B[('taxi_hourly', 1, 'ARIMA')], 5000 * 0.01))
        check('B: WSPI-FA without a timing sample is NaN',
              np.isnan(B[('youtube_hourly', 1, 'WSPI-FA')]))

        ft = tmp / 'ft'
        timing_csv(ft / 'youtube_hourly' / 'h1' / 'scorer_timing.csv', 1,
                   {'WSPI-FA': 1.0, 'ARIMA': 1.0}, rows=1000, scorer_seconds=30.0)
        E.main(['--preview', str(prev), '--reference', str(ref), '--arima-timing', str(arima),
                '--fit-timing', str(ft), '--out', str(out)])
        est = pd.read_csv(out)
        B = est[est['run'] == 'B'].set_index(['scenario', 'horizon', 'method'])['est_seconds']
        check('B: seconds per fit from the sample (30 s / 1000 rows)',
              np.isclose(B[('youtube_hourly', 1, 'WSPI-FA')], fits_yt * 0.03) and
              np.isclose(B[('youtube_hourly', 1, 'ARIMA')], fits_yt * 0.03))
        check('B: the other scenario still uses arima_timing.csv for ARIMA, NaN for WSPI-FA',
              np.isclose(B[('taxi_hourly', 1, 'ARIMA')], fits_yt * 0.01) and
              np.isnan(B[('taxi_hourly', 1, 'WSPI-FA')]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
