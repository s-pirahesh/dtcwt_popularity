r"""
Provenance audit of the YouTube dataset (revision-srep-v5, task T3.11, R4.10)
==============================================================================
Reviewer 4 (R4.10) asks for the source, collection period, number of videos,
inclusion / exclusion criteria, preprocessing, missing-value treatment, and
whether the data allow the experiments to be reproduced.  This script only
READS the raw file, the processed file, the V5 protocol CSVs and the other
dataset files, and writes CSV / JSON.  No existing module is changed.

Facts it establishes (27 Sep 2026)
  * Raw file = Kaggle "YouTube videos viewCount every hour"
    (nnqkfdjq/statistics-observation-of-random-youtube-video, CC0, version 2),
    file count_observation_upload.csv: one row per video and hourly snapshot.
  * viewCount_diff (the column the converter uses) equals the difference of two
    consecutive cumulative viewCount values of the same video.
  * data/converters/youtube_converter.py, as used for data/datasets/
    youtube_hourly.csv: no aggregation (granularity 'none'), no date filter,
    rows with a missing or negative difference are DROPPED (count >= 0 is False
    for NaN), and videos with a total of >= min_views new views are kept.
    The threshold that reproduces the processed file is found by scanning, and
    the real converter is re-run (into a temporary folder, outside the project)
    and compared row by row with the processed file.
  * Missing hours: hours of the regular hourly grid of the processed file with no
    row at all (a collection gap).  Protocol V5 zero-fills them; they are never a
    test slot.  The sensitivity table repeats the common-window means of the V5
    causal run without the windows whose 64-slot input contains a gap hour.
  * dataset_summary.csv gives, for every data file of the paper, the number of
    items in the file, the number of items that the causal catalogue rule
    (total count before the test slot >= min_obs) admits at least once, and the
    period.  Used for the two new rows of tab:data (T4.4).

Usage (from the project root)
  python tools/audit_youtube_provenance.py ^
      --raw data/raw/youtube/count_observation_upload.csv ^
      --processed data/datasets/youtube_hourly.csv ^
      --v5-run results/revision_v5/T1.5_causal_universe/T1.4_protocol_v5/youtube_hourly ^
      --out results/revision_v5/T3.11_youtube_provenance
  python tools/audit_youtube_provenance.py --summary ^
      --out results/revision_v5/T3.11_youtube_provenance

Outputs (<out>)
  raw_profile.csv          key, value: size of the raw file and every count cited
  missing_by_hour.csv      per raw hour: share of NaN viewCount / NaN diff, in processed?
  cleaning_steps.csv       rows and videos after each converter step
  threshold_check.csv      min_views -> videos, rows, equal to processed?
  converter_check.csv      re-run of the real converter vs the processed file
  gap_windows.csv          V5 windows whose 7- or 64-slot input contains a gap hour
  gap_sensitivity.csv      common-window means with and without those windows
  dataset_summary.csv      (--summary) items and period of every data file
  metadata/provenance_run.json

Author: Sajjad Pirahesh, September 2026
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

KAGGLE = dict(
    ref='nnqkfdjq/statistics-observation-of-random-youtube-video',
    url='https://www.kaggle.com/datasets/nnqkfdjq/statistics-observation-of-random-youtube-video',
    title='YouTube videos viewCount every hour',
    license='CC0: Public Domain',
    version=2,
    last_updated='2018-06-15',
    file='count_observation_upload.csv',
    description_summary=('hourly counts (views, comments, likes, dislikes) during May 2018 of '
                         'about 1500 videos released in April 2018; video list retrieved on '
                         '7 May 2018; retrieved by the YouTube API; the creator notes some NaN '
                         'values caused by a collection mistake'),
    read_on='2026-09-27 (Kaggle dataset metadata page)',
)

METHODS = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF', 'DWT+AF', 'DTCWT+AF', 'WSPI']
METRICS = ['ndcg@10', 'spearman_rho', 'rsi@10']

# data files of the paper: name -> (path, min_obs of the causal catalogue rule)
SUMMARY_FILES = {
    'youtube_hourly': ('data/datasets/youtube_hourly.csv', 50),
    'taxi_hourly': ('data/datasets/yellow_taxi_2025_all_hourly.csv', 24),
    'taxi_30min': ('data/datasets/yellow_taxi_2025_all_30min.csv', 24),
    'taxi_5min': ('data/datasets/yellow_taxi_2025_all_5min.csv', 24),
    'movielens_daily': ('data/datasets/movielens_v5_daily.csv', 24),
    'movielens_weekly': ('data/datasets/movielens_v5_weekly.csv', 24),
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__)


# ----------------------------------------------------------------------------
# raw file
# ----------------------------------------------------------------------------
def load_raw(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=['videoId', 'Time', 'viewCount', 'viewCount_diff'])
    raw['Time'] = pd.to_datetime(raw['Time'], errors='coerce')
    return raw          # file order kept (videos in blocks, time increasing)


def profile_raw(raw: pd.DataFrame) -> tuple[list, pd.DataFrame]:
    """Returns (key/value rows, missing_by_hour)."""
    rows = []

    def put(k, v):
        rows.append((k, v))

    g = raw.groupby('videoId', sort=False)
    per_video = g.size()
    put('raw_rows', len(raw))
    put('raw_videos', raw['videoId'].nunique())
    put('raw_hours', raw['Time'].nunique())
    put('raw_first_time', str(raw['Time'].min()))
    put('raw_last_time', str(raw['Time'].max()))
    put('raw_time_nat', int(raw['Time'].isna().sum()))
    put('raw_times_not_on_hour', int(((raw['Time'].dt.minute != 0) | (raw['Time'].dt.second != 0)).sum()))
    put('raw_duplicate_video_time', int(raw.duplicated(['videoId', 'Time']).sum()))
    put('raw_snapshots_per_video_min', int(per_video.min()))
    put('raw_snapshots_per_video_max', int(per_video.max()))
    grid = pd.date_range(raw['Time'].min(), raw['Time'].max(), freq='h')
    put('raw_hours_in_span', len(grid))
    put('raw_time_monotone_within_video', bool(g['Time'].apply(lambda t: t.is_monotonic_increasing).all()))

    # viewCount_diff == difference of consecutive cumulative counts (same video)
    prev = g['viewCount'].shift()
    calc = raw['viewCount'] - prev
    d = raw['viewCount_diff']
    both = d.notna() & calc.notna()
    put('diff_rows_compared', int(both.sum()))
    put('diff_equals_consecutive_difference_share',
        float(np.isclose(d[both].values, calc[both].values, rtol=0, atol=1e-6).mean()))
    put('diff_nan_but_both_counts_present', int((d.isna() & calc.notna()).sum()))

    first = g.cumcount() == 0
    vc_nan = raw['viewCount'].isna()
    d_nan = d.isna()
    put('viewcount_nan_rows', int(vc_nan.sum()))
    put('viewcount_nan_videos', int(raw.loc[vc_nan, 'videoId'].nunique()))
    put('diff_nan_rows', int(d_nan.sum()))
    put('diff_nan_first_snapshot', int((d_nan & first).sum()))
    put('diff_nan_viewcount_nan', int((d_nan & vc_nan & ~first).sum()))
    put('diff_nan_after_viewcount_nan', int((d_nan & ~vc_nan & ~first).sum()))
    put('diff_negative_rows', int((d < 0).sum()))
    put('diff_negative_videos', int(raw.loc[d < 0, 'videoId'].nunique()))
    put('diff_negative_share_of_rows', float((d < 0).mean()))
    put('diff_zero_rows', int((d == 0).sum()))
    put('diff_positive_rows', int((d > 0).sum()))
    last = g.tail(1)
    put('videos_viewcount_nan_at_last_snapshot', int(last['viewCount'].isna().sum()))

    by_hour = raw.groupby('Time').agg(
        n_videos=('videoId', 'size'),
        share_viewcount_nan=('viewCount', lambda s: float(s.isna().mean())),
        share_diff_nan=('viewCount_diff', lambda s: float(s.isna().mean())),
        share_diff_negative=('viewCount_diff', lambda s: float((s < 0).mean())),
    ).reset_index().rename(columns={'Time': 'hour'})
    put('hours_all_viewcount_nan', int((by_hour['share_viewcount_nan'] == 1.0).sum()))
    put('hours_all_diff_nan', int((by_hour['share_diff_nan'] == 1.0).sum()))
    return rows, by_hour


# ----------------------------------------------------------------------------
# converter replica (steps of youtube_converter.YouTubeConverter)
# ----------------------------------------------------------------------------
def replica_steps(raw: pd.DataFrame, min_valid_date='2018-05-01'):
    df = raw[['Time', 'videoId', 'viewCount_diff']].rename(
        columns={'Time': 'timestamp', 'videoId': 'item_id', 'viewCount_diff': 'count'})
    steps = [('raw', len(df), df['item_id'].nunique())]
    df = df.dropna(subset=['timestamp'])
    df = df[df['item_id'].notna()]
    df = df[df['timestamp'] >= pd.Timestamp(min_valid_date)]
    steps.append(('valid timestamp and id', len(df), df['item_id'].nunique()))
    n0 = len(df)
    nan_rows = int(df['count'].isna().sum())
    neg_rows = int((df['count'] < 0).sum())
    df = df[df['count'] >= 0]
    assert n0 - len(df) == nan_rows + neg_rows
    steps.append(('drop missing difference', n0 - nan_rows, None))
    steps.append(('drop negative difference', len(df), df['item_id'].nunique()))
    return df, steps


def threshold_interval(tot: pd.Series, n: int) -> list:
    """Integer thresholds t (as [low, high]) for which exactly n videos have total >= t."""
    s = np.sort(tot.values)[::-1]
    if n < 1 or n > len(s):
        return []
    hi = int(np.floor(s[n - 1]))                       # t <= s[n-1]
    lo = int(np.floor(s[n])) + 1 if n < len(s) else int(np.floor(s.min()))  # t > s[n]
    return [lo, hi] if lo <= hi else []


def compare_frames(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    out = dict(rows_a=len(a), rows_b=len(b), same_rows=len(a) == len(b))
    if out['same_rows']:
        out['item_equal'] = bool((a['item_id'].astype(str).values == b['item_id'].astype(str).values).all())
        out['time_equal'] = bool((pd.to_datetime(a['timestamp']).values
                                  == pd.to_datetime(b['timestamp']).values).all())
        out['count_max_abs_diff'] = float(np.max(np.abs(a['count'].values.astype(float)
                                                        - b['count'].values.astype(float))))
        out['equal'] = out['item_equal'] and out['time_equal'] and out['count_max_abs_diff'] == 0.0
    else:
        out.update(item_equal=False, time_equal=False, count_max_abs_diff=np.nan, equal=False)
    return out


def rerun_converter(raw_path: Path, min_views: int) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from data.converters.youtube_converter import YouTubeConverter   # unchanged module
    with tempfile.TemporaryDirectory() as td:
        conv = YouTubeConverter(granularity='none', min_views_per_video=min_views,
                                verbose=False, validate_output=True)
        outp = Path(td) / 'youtube_hourly_rerun.csv'
        conv.convert(str(raw_path), str(outp))
        return pd.read_csv(outp)


# ----------------------------------------------------------------------------
# V5 windows and gap sensitivity
# ----------------------------------------------------------------------------
def gap_hours(proc: pd.DataFrame) -> list:
    t = pd.to_datetime(proc['timestamp'])
    grid = pd.date_range(t.min(), t.max(), freq='h')
    present = set(t.unique())
    return [h for h in grid if h not in present]


def gap_windows_and_sensitivity(run_dir: Path, gaps: list, start: pd.Timestamp):
    proto = {m: pd.read_csv(run_dir / 'protocol' / f'{m}_protocol.csv') for m in METHODS}
    gap_ids = sorted(int(round((h - start) / pd.Timedelta(hours=1))) for h in gaps)
    sets = [set(d.dropna(subset=['ndcg@10'])['window_id']) for d in proto.values()]
    common = sorted(set.intersection(*sets))
    all_ids = sorted(set.union(*[set(d['window_id']) for d in proto.values()]))

    def contains(k, W):
        return any(k - W <= g < k for g in gap_ids)

    gw = pd.DataFrame({'window_id': all_ids})
    gw['test_time'] = [str(start + pd.Timedelta(hours=int(k))) for k in gw['window_id']]
    gw['common'] = gw['window_id'].isin(common)
    gw['input7_contains_gap'] = [contains(k, 7) for k in gw['window_id']]
    gw['input64_contains_gap'] = [contains(k, 64) for k in gw['window_id']]
    for m in METHODS:
        gw[f'has_{m}'] = gw['window_id'].isin(set(proto[m]['window_id']))
    # every gap hour: is it ever a test slot of any method?
    tested_gap = sorted(set(gap_ids) & set(all_ids))

    summ = pd.read_csv(run_dir / 'comparison' / 'summary_common_windows.csv').set_index('method')
    excl = set(gw.loc[gw['common'] & gw['input64_contains_gap'], 'window_id'])
    keep = [k for k in common if k not in excl]
    rows = []
    for m in METHODS:
        d = proto[m].set_index('window_id')
        for met in METRICS:
            a = float(d.loc[common, met].mean())
            b = float(d.loc[keep, met].mean())
            ref = float(summ.loc[m, f'{met}_mean']) if f'{met}_mean' in summ.columns else np.nan
            rows.append(dict(method=m, metric=met, n_all=len(common), mean_all=a,
                             n_excl=len(keep), mean_excl=b, diff=b - a,
                             control_summary_mean=ref,
                             control_pass=bool(np.isfinite(ref) and abs(ref - a) <= 1e-12 * max(1.0, abs(ref)))))
    sens = pd.DataFrame(rows)
    for met in METRICS:
        s = sens['metric'] == met
        sens.loc[s, 'rank_all'] = sens.loc[s, 'mean_all'].rank(ascending=False, method='min')
        sens.loc[s, 'rank_excl'] = sens.loc[s, 'mean_excl'].rank(ascending=False, method='min')
    return gw, sens, dict(gap_slot_ids=gap_ids, gap_hours_used_as_test=len(tested_gap),
                          n_common=len(common), n_common_input64_gap=len(excl),
                          n_common_input7_gap=int((gw['common'] & gw['input7_contains_gap']).sum()))


# ----------------------------------------------------------------------------
# dataset summary (tab:data)
# ----------------------------------------------------------------------------
def dataset_summary(files: dict) -> pd.DataFrame:
    rows = []
    for name, (rel, min_obs) in files.items():
        p = ROOT / rel
        df = pd.read_csv(p, usecols=['timestamp', 'item_id', 'count'])
        t = pd.to_datetime(df['timestamp'])
        last = t.max()
        before_last = df.loc[t < last].groupby('item_id')['count'].sum()
        ts = pd.Series(t.unique()).sort_values()
        step_min = float(ts.diff().dropna().dt.total_seconds().median() / 60.0)
        n_grid = int(round((last - t.min()).total_seconds() / 60.0 / step_min)) + 1
        rows.append(dict(dataset=name, file=rel, md5=md5(p), rows=len(df),
                         items_in_file=int(df['item_id'].nunique()),
                         items_admitted_causal=int((before_last >= min_obs).sum()),
                         min_obs=min_obs, first_time=str(t.min()), last_time=str(last),
                         slot_minutes=step_min, slots_in_grid=n_grid,
                         slots_with_rows=int(t.nunique()), total_count=float(df['count'].sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--raw', type=Path)
    ap.add_argument('--processed', type=Path)
    ap.add_argument('--v5-run', type=Path, help='V5 causal run of youtube_hourly (protocol/ + comparison/)')
    ap.add_argument('--thresholds', default='0,1,10,25,50,100,200')
    ap.add_argument('--summary', action='store_true', help='only write dataset_summary.csv')
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args(argv)
    out = a.out
    (out / 'metadata').mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.summary:
        ds = dataset_summary(SUMMARY_FILES)
        ds.to_csv(out / 'dataset_summary.csv', index=False)
        print(ds[['dataset', 'items_in_file', 'items_admitted_causal', 'first_time', 'last_time',
                  'slots_in_grid']].to_string(index=False))
        meta = dict(task='T3.11', part='dataset_summary', date=time.strftime('%Y-%m-%d %H:%M:%S'),
                    files={k: v[0] for k, v in SUMMARY_FILES.items()},
                    min_obs={k: v[1] for k, v in SUMMARY_FILES.items()},
                    items_admitted_rule='total count over rows with timestamp < last timestamp >= min_obs',
                    runtime_seconds=round(time.time() - t0, 2),
                    versions=_versions(), host=platform.platform())
        (out / 'metadata' / 'dataset_summary_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        return 0

    if not (a.raw and a.processed and a.v5_run):
        ap.error('--raw, --processed and --v5-run are required (or use --summary)')

    raw = load_raw(a.raw)
    prof_rows, by_hour = profile_raw(raw)
    proc = pd.read_csv(a.processed)
    proc_t = pd.to_datetime(proc['timestamp'])
    by_hour['in_processed'] = by_hour['hour'].isin(set(proc_t.unique()))
    by_hour.to_csv(out / 'missing_by_hour.csv', index=False)

    # converter replica and threshold scan
    clean, steps = replica_steps(raw)
    tot = clean.groupby('item_id')['count'].sum()
    th_rows, match = [], []
    for th in [int(x) for x in a.thresholds.split(',')]:
        keep = tot[tot >= th].index
        x = clean[clean['item_id'].isin(keep)]
        cmp_ = compare_frames(x, proc)
        th_rows.append(dict(min_views=th, videos=len(keep), rows=len(x), **cmp_))
        if cmp_['equal']:
            match.append(th)
    exact = threshold_interval(tot, proc['item_id'].nunique())
    pd.DataFrame(th_rows).to_csv(out / 'threshold_check.csv', index=False)
    th_used = match[0] if match else None
    if th_used is not None:
        kept = clean[clean['item_id'].isin(tot[tot >= th_used].index)]
        steps.append((f'keep videos with >= {th_used} new views', len(kept), kept['item_id'].nunique()))
    pd.DataFrame(steps, columns=['step', 'rows', 'videos']).to_csv(out / 'cleaning_steps.csv', index=False)

    # re-run of the real converter
    if th_used is not None:
        rer = rerun_converter(a.raw, th_used)
        cc = compare_frames(rer, proc)
        cc = dict(min_views=th_used, granularity='none', **cc)
    else:
        cc = dict(min_views=None, equal=False, note='no threshold reproduced the processed file')
    pd.DataFrame([cc]).to_csv(out / 'converter_check.csv', index=False)

    # processed file
    pv = proc.groupby('item_id')['count'].sum()
    prof_rows += [
        ('processed_rows', len(proc)), ('processed_videos', int(proc['item_id'].nunique())),
        ('processed_hours', int(proc_t.nunique())),
        ('processed_first_time', str(proc_t.min())), ('processed_last_time', str(proc_t.max())),
        ('processed_zero_rows', int((proc['count'] == 0).sum())),
        ('processed_zero_share', float((proc['count'] == 0).mean())),
        ('processed_min_video_total', float(pv.min())),
        ('processed_total_views', float(proc['count'].sum())),
        ('videos_removed_by_filter', int(raw['videoId'].nunique() - proc['item_id'].nunique())),
        ('filter_threshold_reproducing_processed', th_used),
        ('filter_thresholds_giving_same_video_count', ';'.join(map(str, exact))),
        ('converter_rerun_equal', cc.get('equal')),
    ]

    # gap hours and V5 windows
    gaps = gap_hours(proc)
    gw, sens, ginfo = gap_windows_and_sensitivity(a.v5_run, gaps, proc_t.min())
    gw.to_csv(out / 'gap_windows.csv', index=False)
    sens.to_csv(out / 'gap_sensitivity.csv', index=False)
    prof_rows += [('gap_hours', len(gaps)), ('gap_hours_list', ';'.join(str(h) for h in gaps)),
                  ('gap_hours_used_as_test_slot', ginfo['gap_hours_used_as_test']),
                  ('common_windows', ginfo['n_common']),
                  ('common_windows_input64_contains_gap', ginfo['n_common_input64_gap']),
                  ('common_windows_input7_contains_gap', ginfo['n_common_input7_gap']),
                  ('gap_sensitivity_max_abs_diff', float(sens['diff'].abs().max())),
                  ('gap_sensitivity_rank_changes', int((sens['rank_all'] != sens['rank_excl']).sum())),
                  ('gap_sensitivity_control_all_pass', bool(sens['control_pass'].all()))]
    pd.DataFrame(prof_rows, columns=['key', 'value']).to_csv(out / 'raw_profile.csv', index=False)

    meta = dict(task='T3.11', review_point='R4.10', date=time.strftime('%Y-%m-%d %H:%M:%S'),
                kaggle=KAGGLE,
                files=dict(raw=str(a.raw), raw_md5=md5(a.raw), raw_bytes=a.raw.stat().st_size,
                           processed=str(a.processed), processed_md5=md5(a.processed),
                           v5_run=str(a.v5_run)),
                converter=dict(module='data/converters/youtube_converter.py',
                               module_md5=md5(ROOT / 'data' / 'converters' / 'youtube_converter.py'),
                               granularity='none', min_views_per_video=th_used,
                               rule='rows with missing or negative viewCount_diff dropped; '
                                    'videos with total new views >= min_views kept'),
                v5_rule='missing (item, hour) = 0; gap hours never a test slot',
                gap_info=ginfo, runtime_seconds=round(time.time() - t0, 2),
                versions=_versions(), host=platform.platform())
    (out / 'metadata' / 'provenance_run.json').write_text(json.dumps(meta, indent=2, default=str),
                                                          encoding='utf-8')
    print(pd.DataFrame(prof_rows, columns=['key', 'value']).to_string(index=False))
    print('\nconverter re-run equal to processed file:', cc.get('equal'))
    print('gap sensitivity: max |diff| = %.4f, rank changes = %d, control pass = %s'
          % (sens['diff'].abs().max(), (sens['rank_all'] != sens['rank_excl']).sum(),
             sens['control_pass'].all()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
