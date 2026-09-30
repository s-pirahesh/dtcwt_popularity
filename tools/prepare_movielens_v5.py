r"""
MovieLens execution datasets for protocol V5 (revision-srep-v5, task T3.9 / E10)
===============================================================================
Builds the two MovieLens files used in the paper from the raw GroupLens file
``ratings.csv`` (ML-32M: userId, movieId, rating, timestamp):

  <data-out>/movielens_v5_daily.csv    one row per (UTC day, movie) with >= 1 rating
  <data-out>/movielens_v5_weekly.csv   one row per (Monday-Sunday UTC week, movie)

Columns: ``timestamp,item_id,count`` (same layout as the YouTube and taxi files).

Rules (27 Sep 2026):
  * signal = number of ratings of the movie in the slot; the rating VALUE is
    not used (one rating = one interaction), no rating filter;
  * all ratings are kept, including those made on the user's first day;
  * slot = UTC calendar day (Unix timestamp // 86400); week = Monday..Sunday UTC,
    labelled by its Monday (same convention as data/datasets/movielens_weekly.csv);
  * period: days in [--start, --end]; weeks only if all seven days are inside
    [--start, --end] (partial first/last weeks are dropped);
  * item pre-filter: keep a movie only if its total count IN THAT FILE is
    >= --min-obs.  Under ``--causal-universe`` an item enters window k only if
    its count before k is >= min-obs, so a movie below the threshold over the
    whole file can never enter any window: the filter does not change any
    result.  It only keeps the dense item x slot matrix small.  The script
    checks that the set of slots with data is the same before and after the
    filter (the evaluator derives its slot grid from it).

Metadata (CSV/JSON only) goes to --meta-out:
  prep_summary.json        inputs (md5, size), rules, per-file counts, output md5
  granularity_profile.csv  per file: items, slots, rows, ratings, zero share,
                           eligible items per window for both entry rules,
                           count scale (per-slot maximum and 10th-largest count)
  year_profile.csv         per calendar year of the period: ratings, active
                           movies, share of ratings made on the user's first
                           rating day (first day over the whole raw file)

Example (Windows, from the project root):

  python tools\prepare_movielens_v5.py --raw data\raw\movielens\ratings.csv ^
         --start 2015-01-01 --end 2023-10-12 --min-obs 24 ^
         --data-out data\datasets --meta-out results\revision_v5\T3.9_movielens\data_prep

Full command list: REPRODUCE.md
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DAY_S = 86400
CHUNK = 4_000_000
# entry rules of the V5 methods (evaluation/method_configs.py)
RULES = {'wavelet_W64_min32': (64, 32), 'baseline_W7_min3': (7, 3)}


# -----------------------------------------------------------------------------
def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def day_of(date_str: str) -> int:
    """UTC day index (days since 1970-01-01) of a YYYY-MM-DD date."""
    return int(pd.Timestamp(date_str, tz='UTC').value // (DAY_S * 10 ** 9))


def week_of_day(day: np.ndarray) -> np.ndarray:
    """Monday-based week index: 1970-01-01 was a Thursday, so day + 3 puts
    every Monday on a multiple of 7."""
    return (day + 3) // 7


def week_start_day(week: np.ndarray) -> np.ndarray:
    return week * 7 - 3


def day_label(day: np.ndarray) -> np.ndarray:
    return pd.to_datetime(day, unit='D').strftime('%Y-%m-%d').values


# -----------------------------------------------------------------------------
def read_raw(raw: Path):
    """One pass over ratings.csv.  Returns
    daily   DataFrame (day, item, count) over the whole file,
    first   first rating day of every user (index = userId),
    ud      DataFrame (day, is_first_day) rating-level flags aggregated per day,
    n_rows  number of ratings."""
    parts, n = [], 0
    first = None
    user_day_parts = []
    for ch in pd.read_csv(raw, usecols=['userId', 'movieId', 'timestamp'],
                          dtype={'userId': 'int64', 'movieId': 'int64', 'timestamp': 'int64'},
                          chunksize=CHUNK):
        n += len(ch)
        day = (ch['timestamp'].values // DAY_S).astype(np.int64)
        g = pd.DataFrame({'day': day, 'item': ch['movieId'].values}) \
            .groupby(['day', 'item'], sort=False).size()
        parts.append(g)
        # first day per user (running minimum)
        f = pd.Series(day).groupby(ch['userId'].values).min()
        first = f if first is None else pd.concat([first, f]).groupby(level=0).min()
        user_day_parts.append(pd.DataFrame({'user': ch['userId'].values, 'day': day})
                              .groupby(['user', 'day'], sort=False).size())
    daily = pd.concat(parts).groupby(level=[0, 1]).sum().rename('count').reset_index()
    ud = pd.concat(user_day_parts).groupby(level=[0, 1]).sum().rename('n').reset_index()
    return daily, first.astype(np.int64), ud, n


def slot_table(daily: pd.DataFrame, gran: str, d0: int, d1: int) -> pd.DataFrame:
    """(slot, item, count) for one granularity inside the period [d0, d1]."""
    if gran == 'daily':
        x = daily[(daily['day'] >= d0) & (daily['day'] <= d1)]
        return x.rename(columns={'day': 'slot'})[['slot', 'item', 'count']].copy()
    wk = week_of_day(daily['day'].values)
    ws = week_start_day(wk)
    keep = (ws >= d0) & (ws + 6 <= d1)                    # full weeks only
    x = pd.DataFrame({'slot': ws[keep], 'item': daily['item'].values[keep],
                      'count': daily['count'].values[keep]})
    return x.groupby(['slot', 'item'], as_index=False)['count'].sum()


def apply_item_filter(t: pd.DataFrame, min_obs: int):
    tot = t.groupby('item')['count'].sum()
    keep_items = tot.index[tot >= min_obs]
    f = t[t['item'].isin(keep_items)]
    same_slots = bool(np.array_equal(np.unique(t['slot'].values), np.unique(f['slot'].values)))
    info = dict(items_in_period=int(len(tot)), items_kept=int(len(keep_items)),
                ratings_in_period=int(tot.sum()), ratings_kept=int(tot[tot >= min_obs].sum()),
                slots_with_data_unchanged_by_filter=same_slots)
    return f, info


def write_csv(t: pd.DataFrame, path: Path, chunk: int = 2_000_000):
    """Rows sorted by (timestamp, item_id as text), written in chunks so that
    memory stays low.  '\n' on every OS, so that the md5 is the same on
    Windows and Linux."""
    slot_u, slot_i = np.unique(t['slot'].values, return_inverse=True)
    item_u, item_i = np.unique(t['item'].values, return_inverse=True)
    lab = day_label(slot_u)
    txt = item_u.astype(np.int64).astype(str)
    rank = np.empty(len(txt), dtype=np.int64)
    rank[np.argsort(txt, kind='mergesort')] = np.arange(len(txt))
    order = np.argsort(slot_i.astype(np.int64) * len(txt) + rank[item_i], kind='mergesort')
    cnt = t['count'].values.astype(np.int64)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        for a in range(0, len(order), chunk):
            o = order[a:a + chunk]
            pd.DataFrame({'timestamp': lab[slot_i[o]], 'item_id': txt[item_i[o]],
                          'count': cnt[o]}).to_csv(fh, index=False, header=(a == 0),
                                                   lineterminator='\n')


def profile(t: pd.DataFrame, gran: str, min_obs: int) -> dict:
    """Pre-run statistics of one execution file (E10: items, span, zeros).
    One pass over the slots with running window counts, so memory stays at
    one slot x item count matrix (about 0.8 GB for 1998-2023 daily)."""
    step = 1 if gran == 'daily' else 7
    slots = np.arange(t['slot'].min(), t['slot'].max() + 1, step)
    sidx = ((t['slot'].values - slots[0]) // step).astype(np.int64)
    items, iidx = np.unique(t['item'].values, return_inverse=True)
    n, S = len(items), len(slots)
    VT = np.zeros((S, n), dtype=np.int32)                 # slot x item, counts < 2**31
    VT[sidx, iidx] = t['count'].values
    nnz = int(np.count_nonzero(VT))
    row = dict(file=gran, items=int(n), slots=int(S),
               first_slot=str(day_label(slots[:1])[0]), last_slot=str(day_label(slots[-1:])[0]),
               rows=nnz, ratings=int(VT.sum(dtype=np.int64)),
               zero_share_item_slot=float(1.0 - nnz / (n * S)))
    cum = np.zeros(n, dtype=np.int64)                     # causal catalogue: count in [0, k)
    obs = {name: np.zeros(n, dtype=np.int32) for name in RULES}   # rows in [k-W, k)
    n_el = {name: [] for name in RULES}
    zero_sh = {name: [] for name in RULES}
    mx, r10 = np.empty(S), np.empty(S)
    for k in range(S):
        if k > 0:
            cum += VT[k - 1]
            for name, (W, _) in RULES.items():
                obs[name] += (VT[k - 1] > 0)
                if k - 1 - W >= 0:
                    obs[name] -= (VT[k - 1 - W] > 0)
        mx[k] = VT[k].max()
        r10[k] = np.partition(VT[k], n - 10)[n - 10] if n >= 10 else np.nan
        if k < 32:                                        # evaluation windows start at 32
            continue
        for name, (W, need) in RULES.items():
            lo = max(k - W, 0)
            o = obs[name]
            ok = (o >= need) & (cum >= min_obs)
            m = int(ok.sum())
            n_el[name].append(m)
            if m:
                zero_sh[name].append(1.0 - o[ok].mean() / (k - lo))
    for name in RULES:
        e = n_el[name] or [np.nan]                        # fewer than 33 slots
        row[f'{name}_eligible_median'] = float(np.median(e))
        row[f'{name}_eligible_min'] = float(np.min(e))
        row[f'{name}_eligible_max'] = float(np.max(e))
        row[f'{name}_zero_share_in_window_mean'] = (float(np.mean(zero_sh[name]))
                                                    if zero_sh[name] else float('nan'))
    row['max_count_per_slot_median'] = float(np.median(mx))
    row['count_rank10_per_slot_median'] = float(np.median(r10)) if n >= 10 else float('nan')
    return row


def year_profile(daily: pd.DataFrame, first: pd.Series, ud: pd.DataFrame,
                 d0: int, d1: int) -> pd.DataFrame:
    x = daily[(daily['day'] >= d0) & (daily['day'] <= d1)].copy()
    x['year'] = pd.to_datetime(x['day'], unit='D').dt.year
    y = x.groupby('year').agg(ratings=('count', 'sum'), active_movies=('item', 'nunique'),
                              days_with_data=('day', 'nunique'))
    u = ud[(ud['day'] >= d0) & (ud['day'] <= d1)].copy()
    u['is_first'] = u['day'].values == first.reindex(u['user'].values).values
    u['year'] = pd.to_datetime(u['day'], unit='D').dt.year
    fy = u.groupby('year').apply(lambda g: g.loc[g['is_first'], 'n'].sum() / g['n'].sum(),
                                 include_groups=False)
    y['share_on_user_first_day'] = fy
    y.loc['all'] = [y['ratings'].sum(), int(x['item'].nunique()), int(x['day'].nunique()),
                    float(u.loc[u['is_first'], 'n'].sum() / u['n'].sum())]
    return y.reset_index()


# -----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True, help='GroupLens ratings.csv')
    ap.add_argument('--start', required=True, help='first day, YYYY-MM-DD (UTC)')
    ap.add_argument('--end', required=True, help='last day, YYYY-MM-DD (UTC, inclusive)')
    ap.add_argument('--min-obs', type=int, required=True,
                    help='item pre-filter; must equal the --min-obs of the runs')
    ap.add_argument('--data-out', required=True)
    ap.add_argument('--meta-out', required=True)
    ap.add_argument('--grans', nargs='*', default=['daily', 'weekly'])
    a = ap.parse_args(argv)

    t0 = time.time()
    raw = Path(a.raw)
    dout, mout = Path(a.data_out), Path(a.meta_out)
    dout.mkdir(parents=True, exist_ok=True)
    mout.mkdir(parents=True, exist_ok=True)
    d0, d1 = day_of(a.start), day_of(a.end)

    print(f'[prep] reading {raw}', flush=True)
    daily, first, ud, n_rows = read_raw(raw)
    print(f'[prep] {n_rows:,} ratings, {daily["item"].nunique():,} movies, '
          f'{time.time() - t0:.0f}s', flush=True)

    summary = dict(task='T3.9', created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                   raw=dict(path=str(raw), md5=md5_of(raw), bytes=raw.stat().st_size,
                            ratings=int(n_rows), movies=int(daily['item'].nunique()),
                            users=int(len(first)),
                            first_day=str(day_label(np.array([daily['day'].min()]))[0]),
                            last_day=str(day_label(np.array([daily['day'].max()]))[0])),
                   rules=dict(signal='number of ratings per slot; rating value not used',
                              first_day_ratings='kept', slot_daily='UTC calendar day',
                              slot_weekly='Monday..Sunday UTC, labelled by Monday; full weeks only',
                              start=a.start, end=a.end, min_obs=a.min_obs,
                              item_filter='total count in the file >= min_obs (result-neutral '
                                          'under --causal-universe)'),
                   files={},
                   env=dict(python=sys.version.split()[0], numpy=np.__version__,
                            pandas=pd.__version__, platform=platform.platform()))
    prof = []
    for gran in a.grans:
        t = slot_table(daily, gran, d0, d1)
        f, info = apply_item_filter(t, a.min_obs)
        if not info['slots_with_data_unchanged_by_filter']:
            raise RuntimeError(f'{gran}: item filter changed the slot grid')
        path = dout / f'movielens_v5_{gran}.csv'
        write_csv(f, path)
        info.update(path=str(path), md5=md5_of(path), bytes=path.stat().st_size, rows=int(len(f)))
        summary['files'][gran] = info
        prof.append(profile(f, gran, a.min_obs))
        print(f'[prep] {gran}: {info["items_kept"]:,} movies, {len(f):,} rows -> {path} '
              f'({time.time() - t0:.0f}s)', flush=True)

    pd.DataFrame(prof).to_csv(mout / 'granularity_profile.csv', index=False, lineterminator='\n')
    year_profile(daily, first, ud, d0, d1).to_csv(mout / 'year_profile.csv', index=False,
                                                  lineterminator='\n')
    summary['runtime_seconds'] = round(time.time() - t0, 1)
    with open(mout / 'prep_summary.json', 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print(f'[prep] done in {summary["runtime_seconds"]}s; metadata -> {mout}', flush=True)


if __name__ == '__main__':
    main()
