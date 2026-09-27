r"""
Unit tests for tools/prepare_movielens_v5.py (task T3.9)
========================================================
Synthetic ratings.csv with known answers.  Checks:
  1. daily aggregation on the UTC day (a rating at 23:59:59 and 00:00:00);
  2. period cut [start, end] inclusive;
  3. weeks: Monday..Sunday, labelled by Monday; partial first/last weeks dropped;
  4. total count is preserved inside the period;
  5. item pre-filter (total >= min-obs) and slot-grid check;
  6. rating value is ignored (same output for any rating values);
  7. output layout (timestamp,item_id,count; '\n' line ends) and fixed md5
     for a repeated run;
  8. result neutrality: ProtocolV5Evaluator with --causal-universe gives the
     same per-window results with and without the pre-filter (needs dtcwt);
  9. first-day share in year_profile.csv;
 10. streaming profile() equals a dense reference implementation.

Run from the project root:  python tools\test_prepare_movielens_v5.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import prepare_movielens_v5 as P  # noqa: E402

OK = []


def check(name, cond):
    OK.append(bool(cond))
    print(('OK   ' if cond else 'FAIL ') + name)


def ts(date, sec=0):
    return int(pd.Timestamp(date, tz='UTC').value // 10 ** 9) + sec


def synthetic(path: Path, rating_value=None, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    # hand-made rows
    rows += [(1, 10, 4.0, ts('2015-01-05', 86399)),   # Monday 23:59:59 -> 2015-01-05
             (2, 10, 3.0, ts('2015-01-06', 0)),       # Tuesday 00:00:00 -> 2015-01-06
             (3, 10, 5.0, ts('2014-12-31', 100)),     # before period
             (4, 11, 2.0, ts('2015-01-02', 0)),       # in period, first partial week
             (5, 11, 2.0, ts('2015-03-12', 0))]       # last day, partial last week
    # random background: 40 movies, 60 users, days 2015-01-01..2015-03-12
    d0, d1 = P.day_of('2015-01-01'), P.day_of('2015-03-12')
    for _ in range(3000):
        m = 100 + int(rng.zipf(1.6)) % 40
        u = 10 + rng.randint(60)
        d = rng.randint(d0, d1 + 1)
        rows.append((u, m, float(rng.randint(1, 11)) / 2, d * 86400 + rng.randint(86400)))
    rows.append((99, 999, 4.0, ts('2015-02-01', 5)))  # rare movie (1 rating)
    df = pd.DataFrame(rows, columns=['userId', 'movieId', 'rating', 'timestamp'])
    if rating_value is not None:
        df['rating'] = rating_value
    df.to_csv(path, index=False)
    return df


def dense_profile(t, gran, min_obs):
    """Reference: the first (dense matrix) version of P.profile, for check 10."""
    step = 1 if gran == 'daily' else 7
    slots = np.arange(t['slot'].min(), t['slot'].max() + 1, step)
    sidx = ((t['slot'].values - slots[0]) // step).astype(np.int64)
    items, iidx = np.unique(t['item'].values, return_inverse=True)
    n, S = len(items), len(slots)
    V = np.zeros((n, S), dtype=np.int32); V[iidx, sidx] = t['count'].values
    Pm = V > 0
    C = np.zeros((n, S + 1), dtype=np.int32); np.cumsum(Pm, axis=1, out=C[:, 1:])
    cum = np.zeros((n, S + 1), dtype=np.int64); np.cumsum(V, axis=1, out=cum[:, 1:])
    row = dict(rows=int(Pm.sum()), ratings=int(V.sum(dtype=np.int64)),
               zero_share_item_slot=float(1.0 - Pm.mean()))
    for name, (W, need) in P.RULES.items():
        n_el, zs = [], []
        for k in range(32, S):
            lo = max(k - W, 0); obs = C[:, k] - C[:, lo]
            ok = (obs >= need) & (cum[:, k] >= min_obs); m = int(ok.sum()); n_el.append(m)
            if m:
                zs.append(1.0 - obs[ok].mean() / (k - lo))
        row[f'{name}_eligible_median'] = float(np.median(n_el))
        row[f'{name}_zero_share_in_window_mean'] = float(np.mean(zs))
    row['max_count_per_slot_median'] = float(np.median(V.max(axis=0)))
    row['count_rank10_per_slot_median'] = float(np.median(np.partition(V, n - 10, axis=0)[n - 10]))
    return row


def run(raw, out, min_obs=5, grans=('daily', 'weekly')):
    P.main(['--raw', str(raw), '--start', '2015-01-01', '--end', '2015-03-12',
            '--min-obs', str(min_obs), '--data-out', str(out / 'data'),
            '--meta-out', str(out / 'meta'), '--grans', *grans])
    return (pd.read_csv(out / 'data' / 'movielens_v5_daily.csv', dtype={'item_id': str}),
            pd.read_csv(out / 'data' / 'movielens_v5_weekly.csv', dtype={'item_id': str}),
            json.load(open(out / 'meta' / 'prep_summary.json')))


def main():
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / 'ratings.csv'
    df = synthetic(raw)
    d, w, s = run(raw, tmp / 'a')

    # 1. UTC day
    dd = df.assign(day=df.timestamp // 86400)
    dd = dd[(dd.day >= P.day_of('2015-01-01')) & (dd.day <= P.day_of('2015-03-12'))]
    tot = dd.groupby('movieId').size()
    dd = dd[dd.movieId.isin(tot.index[tot >= 5])]
    exp = dd.groupby(['day', 'movieId']).size().reset_index(name='count')
    exp = pd.DataFrame({'timestamp': P.day_label(exp.day.values), 'item_id': exp.movieId.astype(str),
                        'count': exp['count']}).sort_values(['timestamp', 'item_id']).reset_index(drop=True)
    check('daily file equals direct UTC-day count (23:59:59 and 00:00:00 rows included)',
          exp.equals(d.reset_index(drop=True)))
    got = d[(d.item_id == '10')]
    check('movie 10 (2 ratings) is below min-obs and absent', got.empty)
    # 2. period
    check('day before period dropped', d.timestamp.min() >= '2015-01-01')
    check('last day kept', d.timestamp.max() == '2015-03-12')
    # 3. weeks
    wd = pd.to_datetime(w.timestamp).dt.dayofweek.unique()
    check('weeks labelled by Monday', list(wd) == [0])
    check('first partial week dropped (first label 2015-01-05)', w.timestamp.min() == '2015-01-05')
    check('last partial week dropped (last label 2015-03-02)', w.timestamp.max() == '2015-03-02')
    # 4. totals
    day = df.timestamp // 86400
    inp = df[(day >= P.day_of('2015-01-01')) & (day <= P.day_of('2015-03-12'))]
    tot_in = inp.groupby('movieId').size()
    kept = tot_in[tot_in >= 5]
    check('daily total = ratings of kept movies in period', int(d['count'].sum()) == int(kept.sum()))
    wk_lo, wk_hi = P.day_of('2015-01-05'), P.day_of('2015-03-08')
    inw = df[(day >= wk_lo) & (day <= wk_hi)]
    totw = inw.groupby('movieId').size()
    check('weekly total = ratings of kept movies in full weeks',
          int(w['count'].sum()) == int(totw[totw >= 5].sum()))
    # 5. item filter
    check('rare movie removed', '999' not in set(d.item_id))
    check('filter keeps exactly the movies with total >= min-obs',
          set(d.item_id) == set(kept.index.astype(str)))
    check('slot grid unchanged flag', s['files']['daily']['slots_with_data_unchanged_by_filter'])
    # 6. rating ignored
    raw2 = tmp / 'ratings2.csv'
    synthetic(raw2, rating_value=0.5)
    _, _, s2 = run(raw2, tmp / 'b')
    check('rating value does not change the daily file',
          s2['files']['daily']['md5'] == s['files']['daily']['md5'])
    check('rating value does not change the weekly file',
          s2['files']['weekly']['md5'] == s['files']['weekly']['md5'])
    # 7. layout
    b = (tmp / 'a' / 'data' / 'movielens_v5_daily.csv').read_bytes()
    check('header timestamp,item_id,count', b.startswith(b'timestamp,item_id,count\n'))
    check('no CR line ends', b'\r' not in b)
    _, _, s3 = run(raw, tmp / 'c')
    check('repeated run gives the same md5', s3['files']['daily']['md5'] == s['files']['daily']['md5'])
    # 9. first-day share
    y = pd.read_csv(tmp / 'a' / 'meta' / 'year_profile.csv')
    first = df.groupby('userId').timestamp.min() // 86400
    fl = (day.values == first.reindex(df.userId).values)
    inside = ((day >= P.day_of('2015-01-01')) & (day <= P.day_of('2015-03-12'))).values
    exp = fl[inside].mean()
    got = float(y.loc[y.year.astype(str) == 'all', 'share_on_user_first_day'].iloc[0])
    check('first-day share matches direct computation', abs(got - exp) < 1e-12)
    g = pd.read_csv(tmp / 'a' / 'meta' / 'granularity_profile.csv')
    check('profile has both files', list(g.file) == ['daily', 'weekly'])

    # 10. streaming profile equals the dense reference (random data, >= 33 slots)
    rng = np.random.RandomState(3)
    for gran, step in [('daily', 1), ('weekly', 7)]:
        rr = []
        for it in range(200):
            act = rng.rand() * 0.6
            for sl in range(300):
                if rng.rand() < act:
                    rr.append((sl * step + 10003, it, rng.poisson(3) + 1))
        t = pd.DataFrame(rr, columns=['slot', 'item', 'count'])
        a, b = P.profile(t, gran, 24), dense_profile(t, gran, 24)
        check('streaming profile equals dense reference (%s)' % gran, all(a[k] == b[k] for k in b))

    # 8. result neutrality with the V5 evaluator (skipped if dtcwt is missing)
    try:
        from evaluation.protocol_v5 import ProtocolV5Evaluator, build_v5_methods
    except Exception as e:                                 # pragma: no cover
        print('SKIP result neutrality (import failed: %s)' % e)
    else:
        full = P.slot_table(P.read_raw(raw)[0], 'daily',
                            P.day_of('2015-01-01'), P.day_of('2015-03-12'))
        unf = pd.DataFrame({'timestamp': P.day_label(full['slot'].values),
                            'item_id': full['item'].astype(str).values,
                            'count': full['count'].values})
        res = {}
        for tag, data in [('unfiltered', unf), ('filtered', d)]:
            ev = ProtocolV5Evaluator(data.copy(), dataset_min_obs=5, causal_universe=True,
                                     seed=42, verbose=False)
            methods = build_v5_methods(names=['AF', 'RRD', 'WSPI', 'DTCWT+AF']).values()
            res[tag] = pd.concat([ev.run_method(m) for m in methods], ignore_index=True)
        a_, b_ = res['unfiltered'], res['filtered']
        num = a_.select_dtypes('number').columns
        same = (len(a_) == len(b_) and len(a_) > 0 and
                np.allclose(a_[num].values, b_[num].values, equal_nan=True, rtol=0, atol=0))
        check('pre-filter is result-neutral (AF, RRD, WSPI, DTCWT+AF; %d rows)' % len(a_), same)

    print('\n%d / %d OK' % (sum(OK), len(OK)))
    return 0 if all(OK) else 1


if __name__ == '__main__':
    sys.exit(main())
