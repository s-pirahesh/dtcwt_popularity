r"""
Unit test of tools/audit_youtube_provenance.py.

Builds a small synthetic raw file in the Kaggle format (videos in blocks,
cumulative viewCount, viewCount_diff = consecutive difference) with every case
the audit must count: first snapshot, a collection gap (all videos NaN),
a single video missing, a negative difference, a low-view video.  Checks:
  1. profile counts (NaN categories, negatives, identity of the diff column)
  2. replica of the converter == the real YouTubeConverter (unchanged module)
  3. threshold scan finds the threshold and the interval
  4. gap hours are found from the processed file
  5. gap windows and sensitivity on a synthetic V5 run folder
Runs in a temporary folder; writes nothing into the project.
Usage:  python tools/test_audit_youtube_provenance.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import audit_youtube_provenance as A   # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name)


def synthetic_raw(H=12, gap=(5,), missing=('v2', 8), neg=('v1', 10)):
    t0 = pd.Timestamp('2018-05-07 18:00')
    rng = np.random.RandomState(0)
    rows = []
    for v, rate in [('v1', 100), ('v2', 40), ('v3', 1)]:
        cum = 1000.0
        for h in range(H):
            inc = float(rng.poisson(rate))
            cum += inc
            vc = cum
            if h in gap or (v, h) == missing:
                vc = np.nan
            if (v, h) == neg:
                vc = cum - 500.0          # correction below the previous value
                cum = vc
            rows.append(dict(videoId=v, Time=str(t0 + pd.Timedelta(hours=h)), viewCount=vc))
    df = pd.DataFrame(rows)
    df['viewCount_diff'] = df.groupby('videoId', sort=False)['viewCount'].diff()
    df.insert(0, 'index', df.groupby('videoId', sort=False).cumcount())
    for c in ['commentCount', 'dislikeCount', 'favoriteCount', 'likeCount',
              'commentCount_diff', 'dislikeCount_diff', 'favoriteCount_diff', 'likeCount_diff']:
        df[c] = 0.0
    return df


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        raw_df = synthetic_raw()
        rp = td / 'raw.csv'
        raw_df.to_csv(rp)                       # with the unnamed index column, as on Kaggle
        raw = A.load_raw(rp)
        rows, by_hour = A.profile_raw(raw)
        P = dict(rows)
        check('raw rows / videos / hours', P['raw_rows'] == 36 and P['raw_videos'] == 3 and P['raw_hours'] == 12)
        check('diff column = consecutive difference', P['diff_equals_consecutive_difference_share'] == 1.0)
        check('NaN diff: first snapshot = 3', P['diff_nan_first_snapshot'] == 3)
        check('NaN viewCount = 3 (gap) + 1 (missing)', P['viewcount_nan_rows'] == 4)
        check('NaN diff after NaN viewCount = 4', P['diff_nan_after_viewcount_nan'] == 4)
        check('negative rows = 1', P['diff_negative_rows'] == 1 and P['diff_negative_videos'] == 1)
        check('one hour with all viewCount NaN', P['hours_all_viewcount_nan'] == 1)

        clean, steps = A.replica_steps(raw)
        n_nan = P['diff_nan_rows']
        check('replica drops NaN and negative rows', len(clean) == 36 - n_nan - 1)

        # real converter (unchanged module) vs replica, threshold 50
        tot = clean.groupby('item_id')['count'].sum()
        th = 50
        rep = clean[clean['item_id'].isin(tot[tot >= th].index)]
        real = A.rerun_converter(rp, th)
        cmp_ = A.compare_frames(rep, real)
        check('replica == real converter (rows, ids, times, counts)', cmp_['equal'])
        check('low-view video v3 removed', 'v3' not in set(real['item_id']))
        iv = A.threshold_interval(tot, 2)
        check('threshold interval contains 50', iv and iv[0] <= 50 <= iv[1])
        check('threshold interval edges', iv[1] == int(np.floor(np.sort(tot.values)[::-1][1]))
              and iv[0] == int(np.floor(np.sort(tot.values)[::-1][2])) + 1)

        gaps = A.gap_hours(real)
        check('gap hours from processed file = hour 5 and the hour after it',
              [str(g) for g in gaps] == ['2018-05-07 23:00:00', '2018-05-08 00:00:00'])

        # synthetic V5 run: windows 3..10 for all methods, 2 of them contain a gap
        run = td / 'run'
        (run / 'protocol').mkdir(parents=True)
        (run / 'comparison').mkdir()
        start = pd.to_datetime(real['timestamp']).min()
        summ = []
        for j, m in enumerate(A.METHODS):
            ks = [k for k in range(3, 11) if k not in (4, 5)]      # gap slots 4, 5 never tested
            d = pd.DataFrame({'window_id': ks,
                              'ndcg@10': np.linspace(0.5, 0.9, len(ks)) + j * 0.001,
                              'spearman_rho': 0.5, 'rsi@10': np.linspace(0.9, 0.5, len(ks))})
            d.to_csv(run / 'protocol' / f'{m}_protocol.csv', index=False)
            summ.append({'method': m, **{f'{c}_mean': d[c].mean() for c in A.METRICS}})
        pd.DataFrame(summ).to_csv(run / 'comparison' / 'summary_common_windows.csv', index=False)
        gw, sens, info = A.gap_windows_and_sensitivity(run, gaps, start)
        check('gap slot ids = 4, 5', info['gap_slot_ids'] == [4, 5])
        check('gap hours never a test slot', info['gap_hours_used_as_test'] == 0)
        check('64-slot input: windows 6..10 contain the gap (window 3 is before it)',
              info['n_common_input64_gap'] == 5)
        check('7-slot input: windows 6..10 contain the gap (5 windows)',
              info['n_common_input7_gap'] == 5)
        check('control against summary passes', bool(sens['control_pass'].all()))

    n_ok = sum(OK)
    print(f'\n{n_ok} of {len(OK)} checks passed')
    print('ALL OK' if all(OK) else 'SOME CHECKS FAILED')
    return 0 if all(OK) else 1


if __name__ == '__main__':
    sys.exit(main())
