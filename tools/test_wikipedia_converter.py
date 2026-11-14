r"""
Unit checks for the data of the confirmation step (synthetic files only)
========================================================================
Wikipedia converter (``data/converters/wikipedia_converter.py``):

  1. The stride rule: smallest k with ceil(n / k) * days <= max_cells.
  2. A synthetic wide file (four access/agent rows per article, page names with
     commas and non-Latin letters, empty cells, zeros): only the
     ``_all-access_all-agents`` rows are items; of these the rows 0, k, 2k, ...
     in file order; item_id is the raw row number; every count equals the raw
     value; empty cells and zeros have no row; the timestamps are
     ``YYYY-MM-DD 00:00:00``; pages file and metadata (k, counts) are right.
  3. The kept items do not depend on the values: the same file with the values
     of the rows shuffled keeps the same item ids.
  4. A zip file gives the same output.
  5. Shape checks: wrong number of rows, a missing day, days out of order, a
     negative or a fractional value, no matching page.  Each one stops the
     converter before anything is written.
  6. The evaluator reads the output: daily slots, the expected item count.
  7. prepare_data.py lists the converter and refuses the synthetic file, which
     does not have the 803 days of train_2.csv, without writing a file.

  9. The Monash .tsf form of the same synthetic data (? for a missing value)
     gives the same prepared file and the same page list as the CSV form; a
     .tsf file is refused when the series names are not page names, a series
     is one day short, a series starts on another day, or the attributes or
     the frequency differ.

Yellow Taxi converter (``data/converters/yellow_taxi_converter.py``, unchanged):

  8. With --yellow_taxi-start-date 2024-01-01 and --yellow_taxi-end-date
     "2024-12-31 23:59:59", trips dated 2023 and 2025 are dropped and the last
     hourly slot is 2024-12-31 23:00.

    python tools/test_wikipedia_converter.py
"""
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.converters import ConverterFactory                              # noqa: E402
from data.converters.wikipedia_converter import (WikipediaConverter,       # noqa: E402
                                                 stride_for)

FAILED = []
AGENTS = ('all-access_all-agents', 'desktop_all-agents', 'mobile-web_all-agents',
          'all-access_spider')
TITLES = ('Main_Page', 'Tabriz', 'Riga,_Latvia', 'Тебриз', '大不里士', 'Wavelet',
          'Popularity', 'Ranking_(statistics)', 'Café', 'Data,_big', 'Zeta', 'Omega',
          'Alpha', 'Beta')
FIRST, LAST = '2016-01-01', '2016-03-10'           # 70 days


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def synthetic(rng):
    dates = pd.date_range(FIRST, LAST, freq='D')
    pages = [f'{t}_{p}.wikipedia.org_{a}' for i, t in enumerate(TITLES)
             for p in (('en', 'ru')[i % 2],) for a in AGENTS]
    vals = rng.poisson(30, size=(len(pages), len(dates))).astype(float)
    vals[rng.random(vals.shape) < 0.15] = 0.0
    vals[rng.random(vals.shape) < 0.10] = np.nan
    vals[12, :] = np.nan                          # one kept item never viewed (row 12)
    raw = pd.DataFrame(vals, columns=[d.strftime('%Y-%m-%d') for d in dates])
    raw.insert(0, 'Page', pages)
    return raw


def write_tsf(raw, path, names=None, start=FIRST, attrs=('series_name', 'start_timestamp'),
              freq='daily', drop_last_of=None):
    lines = ['# synthetic copy in the Monash format', '@relation web_traffic']
    lines += [f'@attribute {a} {"string" if a == "series_name" else "date"}' for a in attrs]
    lines += [f'@frequency {freq}', '@horizon 59', '@missing true', '@equallength true', '@data']
    vals = raw.iloc[:, 1:].to_numpy(float)
    for i, page in enumerate(raw['Page'] if names is None else names):
        v = ['?' if np.isnan(x) else str(int(x)) for x in vals[i]]
        if drop_last_of == i:
            v = v[:-1]
        lines.append(f'{page}:{start} 00-00-00:' + ','.join(v))
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def convert(raw_file, out, n_rows, max_cells, quiet=True):
    c = WikipediaConverter(max_cells=max_cells, expected_rows=n_rows, first_date=FIRST,
                           last_date=LAST, verbose=not quiet)
    return c, c.convert(raw_file, out)


def run_tests():
    # ---- 1 stride rule
    check('1 stride 39402 x 803 -> 4', stride_for(39402, 803, 9256350) == 4)
    check('1 stride 145063 x 803 -> 13', stride_for(145063, 803, 9256350) == 13)
    check('1 stride fits -> 1', stride_for(100, 803, 9256350) == 1)
    ok = True
    for n in (1, 7, 56, 57, 1000, 39402):
        for d, mx in ((70, 1000), (803, 9256350), (70, 70)):
            k = stride_for(n, d, mx)
            ok &= math.ceil(n / k) * d <= mx and (k == 1 or math.ceil(n / (k - 1)) * d > mx)
    check('1 stride is the smallest k (grid)', ok)
    try:
        stride_for(5, 100, 50)
        check('1 stride refuses days > max_cells', False)
    except ValueError:
        check('1 stride refuses days > max_cells', True)

    rng = np.random.default_rng(7)
    raw = synthetic(rng)
    td = Path(tempfile.mkdtemp())
    try:
        f = td / 'train_2.csv'
        raw.to_csv(f, index=False)
        days = raw.shape[1] - 1
        sel = np.flatnonzero(raw['Page'].str.endswith('_all-access_all-agents').to_numpy())
        max_cells = 5 * days                     # 14 selected -> k = 3 -> 5 items
        k_exp = stride_for(len(sel), days, max_cells)
        keep = sel[::k_exp]

        # ---- 2 conversion
        out = td / 'out' / 'wikipedia_daily.csv'
        c, df = convert(f, out, len(raw), max_cells)
        meta = json.loads((td / 'out' / 'wikipedia_daily_meta.json').read_text(encoding='utf-8'))
        pages = pd.read_csv(td / 'out' / 'wikipedia_daily_pages.csv', encoding='utf-8')
        got = pd.read_csv(out)
        check('2 k from the rule', meta['stride_k'] == k_exp == 3, f"k={meta['stride_k']}")
        check('2 counts in metadata', meta['n_rows'] == len(raw) and meta['n_days'] == days
              and meta['n_selected'] == len(sel) and meta['n_items'] == len(keep))
        check('2 items = rows 0, k, 2k of the all-access_all-agents rows',
              pages['item_id'].tolist() == keep.tolist(), str(keep.tolist()))
        check('2 page names round trip (commas, non-Latin)',
              pages['page'].tolist() == raw['Page'].iloc[keep].tolist())
        check('2 only all-access_all-agents pages',
              pages['page'].str.endswith('_all-access_all-agents').all())
        check('2 header', list(got.columns) == ['timestamp', 'item_id', 'count'])
        check('2 timestamp format', got['timestamp'].str.fullmatch(r'\d{4}-\d{2}-\d{2} 00:00:00').all())
        vals = raw.iloc[:, 1:].to_numpy(float)
        exp = []
        for j, d in enumerate(raw.columns[1:]):
            for i in keep:
                v = vals[i, j]
                if np.isfinite(v) and v > 0:
                    exp.append((f'{d} 00:00:00', int(i), int(v)))
        rows = list(got.itertuples(index=False, name=None))
        check('2 every row equals the raw value; no zero or empty day',
              rows == exp, f'{len(rows)} rows')
        check('2 count is integer and positive', (got['count'] > 0).all()
              and pd.api.types.is_integer_dtype(got['count']))
        nm = int(np.isnan(vals[keep]).sum())
        check('2 missing cells counted', meta['n_missing_cells_set_to_zero'] == nm)
        check('2 item without views counted', meta['n_items_without_views'] == 1
              and 12 not in set(got["item_id"]) and 12 in set(pages["item_id"]))
        check('2 file hash in metadata', len(meta['source_sha256']) == 64)

        # ---- 3 values do not choose the items
        sh = raw.copy()
        sh.iloc[:, 1:] = vals[rng.permutation(len(raw))]
        f3 = td / 'shuffled.csv'
        sh.to_csv(f3, index=False)
        c3, _ = convert(f3, td / 'o3' / 'w.csv', len(raw), max_cells)
        check('3 same items with shuffled values', c3.pages['item_id'].tolist() == keep.tolist())

        # ---- 4 zip
        z = td / 'train_2.csv.zip'
        with zipfile.ZipFile(z, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(f, 'train_2.csv')
        convert(z, td / 'oz' / 'wikipedia_daily.csv', len(raw), max_cells)
        check('4 zip gives the same file',
              (td / 'oz' / 'wikipedia_daily.csv').read_bytes() == out.read_bytes())

        # ---- 5 shape checks
        bad = {
            'wrong row count': (raw, len(raw) + 1),
            'missing day': (raw.drop(columns=[raw.columns[10]]), len(raw)),
            'days out of order': (raw[[raw.columns[0]] + list(raw.columns[1:][::-1])], len(raw)),
            'negative value': (raw.assign(**{raw.columns[5]: -1.0}), len(raw)),
            'fractional value': (raw.assign(**{raw.columns[5]: 1.5}), len(raw)),
            'no matching page': (raw.assign(Page=raw['Page'].str.replace('all-access_all-agents',
                                                                          'x', regex=False)), len(raw)),
        }
        for name, (d, n) in bad.items():
            fb = td / 'bad.csv'
            d.to_csv(fb, index=False)
            ob = td / 'ob' / 'w.csv'
            if ob.parent.exists():
                shutil.rmtree(ob.parent)
            try:
                convert(fb, ob, n, max_cells)
                check(f'5 refuses: {name}', False)
            except ValueError:
                check(f'5 refuses: {name}', not ob.exists())

        # ---- 6 evaluator reads the output
        from evaluation.protocol_v5 import ProtocolV5Evaluator
        ev = ProtocolV5Evaluator.from_csv(out, 50, causal_universe=True, verbose=False)
        check('6 evaluator: daily slots', ev.slot == pd.Timedelta(days=1))
        check('6 evaluator: items with views', len(ev.items) == got['item_id'].nunique())

        # ---- 7 prepare_data.py
        r = subprocess.run([sys.executable, str(ROOT / 'prepare_data.py'), '--list'],
                           capture_output=True, text=True, cwd=ROOT)
        check('7 prepare_data lists wikipedia', r.returncode == 0 and 'wikipedia' in r.stdout)
        o7 = td / 'o7' / 'w.csv'
        r = subprocess.run([sys.executable, str(ROOT / 'prepare_data.py'), '--dataset', 'wikipedia',
                            '--input', str(f), '--output', str(o7)],
                           capture_output=True, text=True, cwd=ROOT)
        msg = r.stdout + r.stderr
        check('7 prepare_data refuses a file that is not shaped like train_2.csv',
              r.returncode != 0 and 'expected 803 consecutive days' in msg and not o7.exists(),
              f'rc={r.returncode}')

        # ---- 9 Monash .tsf form
        ft = td / 'kaggle_web_traffic_dataset_with_missing_values.tsf'
        write_tsf(raw, ft)
        ct, _ = convert(ft, td / 'o9' / 'wikipedia_daily.csv', len(raw), max_cells)
        check('9 tsf: same prepared file as the CSV form',
              (td / 'o9' / 'wikipedia_daily.csv').read_bytes() == out.read_bytes())
        check('9 tsf: same page list',
              (td / 'o9' / 'wikipedia_daily_pages.csv').read_bytes()
              == (td / 'out' / 'wikipedia_daily_pages.csv').read_bytes())
        mt = json.loads((td / 'o9' / 'wikipedia_daily_meta.json').read_text(encoding='utf-8'))
        check('9 tsf: metadata (k, missing cells, source file)',
              mt['stride_k'] == meta['stride_k'] and mt['source_file'] == ft.name
              and mt['n_missing_cells_set_to_zero'] == meta['n_missing_cells_set_to_zero'])
        bad_tsf = {
            'series names are not page names': dict(names=[f'T{i + 1}' for i in range(len(raw))]),
            'a series one day short': dict(drop_last_of=7),
            'series start on another day': dict(start='2016-01-02'),
            'attributes differ': dict(attrs=('series_name',)),
            'frequency differs': dict(freq='weekly'),
        }
        for name, kw in bad_tsf.items():
            fb = td / 'bad.tsf'
            write_tsf(raw, fb, **kw)
            ob = td / 'ob9' / 'w.csv'
            if ob.parent.exists():
                shutil.rmtree(ob.parent)
            try:
                convert(fb, ob, len(raw), max_cells)
                check(f'9 tsf refuses: {name}', False)
            except ValueError:
                check(f'9 tsf refuses: {name}', not ob.exists())

        # ---- 8 yellow taxi 2024 date range
        ts = pd.to_datetime(['2023-12-31 23:50:00', '2024-01-01 00:00:00', '2024-01-01 00:20:00',
                             '2024-06-30 12:00:00', '2024-12-31 23:30:00', '2024-12-31 23:59:59',
                             '2025-01-01 00:00:00', '2025-01-01 00:10:00'])
        trips = pd.DataFrame({'tpep_pickup_datetime': ts, 'PULocationID': [1, 1, 2, 2, 1, 2, 1, 2]})
        (td / 'taxi').mkdir()
        trips.iloc[:4].to_parquet(td / 'taxi' / 'yellow_tripdata_2024-01.parquet', index=False)
        trips.iloc[4:].to_parquet(td / 'taxi' / 'yellow_tripdata_2024-12.parquet', index=False)
        o8 = td / 'o8' / 'taxi.csv'
        r = subprocess.run([sys.executable, str(ROOT / 'prepare_data.py'), '--dataset', 'yellow_taxi',
                            '--input', str(td / 'taxi' / '*.parquet'), '--output', str(o8),
                            '--yellow_taxi-granularity', 'hourly',
                            '--yellow_taxi-min-trips-per-location', '1',
                            '--yellow_taxi-start-date', '2024-01-01',
                            '--yellow_taxi-end-date', '2024-12-31 23:59:59'],
                           capture_output=True, text=True, cwd=ROOT)
        ok = r.returncode == 0 and o8.exists()
        if ok:
            t8 = pd.read_csv(o8)
            t8['timestamp'] = pd.to_datetime(t8['timestamp'])
            ok = (t8['timestamp'].min() == pd.Timestamp('2024-01-01 00:00:00')
                  and t8['timestamp'].max() == pd.Timestamp('2024-12-31 23:00:00')
                  and int(t8['count'].sum()) == 5)
        check('8 taxi 2024: trips of 2023 and 2025 dropped, last slot 23:00', ok,
              '' if ok else (r.stdout + r.stderr)[-400:])
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    run_tests()
    print('\nALL OK' if not FAILED else f'\nFAILED: {FAILED}')
    return 1 if FAILED else 0


if __name__ == '__main__':
    sys.exit(main())
