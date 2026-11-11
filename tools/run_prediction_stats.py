r"""
Paired tests for the prediction runs (every scenario and horizon)
=================================================================
Calls ``tools/stats_report.py`` (unchanged) on every ``<root>/run_<X>/<scenario>/
h<h>`` folder, with the settings fixed before the runs:

  references   run A: WSPI-FG, WSPI-F2, WSPI          run B: WSPI-FA
  block        b_h = max(b, h) windows, with the base block b of the scenario
               (BASE_BLOCK: the block of the stored horizon-1 tests);
               at horizon h the ground truth of h consecutive windows overlaps
  test         two-sided Wilcoxon on block means, Holm within one
               scenario x horizon x metric, alpha 0.05 (column ``verdict``)
  bootstrap    circular block bootstrap, B = 10000, seed 42

Before the tests the script checks that the paired tests are really paired:

  window check   every method of a run folder has the same window ids and the
                 same number of items in every window as WSPI
  cross-run      the WSPI file of run B equals the WSPI file of run A
                 (same scenario and horizon), cell by cell

Outputs (``--out``, default ``<root>/stats``):
  run_<X>/<scenario>/h<h>/ref_<reference>/   method_summary.csv, paired_tests.csv,
                                             metadata/stats_run.json
  window_check.csv         one row per run folder and method
  all_paired_tests.csv     every paired_tests.csv with the columns run, scenario, horizon
  all_method_summary.csv   every method_summary.csv, same extra columns
  stats_plan.json          settings and the list of reports

A report that exists is not computed again (``--force`` to recompute).  Run
folders that are missing are listed and make the exit code 1, unless
``--allow-missing`` is given (the tables then hold what exists).

    python tools\run_prediction_stats.py --root results\paper2
    python tools\run_prediction_stats.py --root results\paper2 --runs A --allow-missing
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import stats_report                                   # noqa: E402

BASE_BLOCK = {'youtube_hourly': 24, 'taxi_hourly': 168, 'taxi_30min': 336,
              'taxi_5min': 2016, 'movielens_daily': 7, 'movielens_weekly': 4}
HORIZONS = (1, 8, 24)
RUNS = {
    'A': dict(scenarios=tuple(BASE_BLOCK), references=('WSPI-FG', 'WSPI-F2', 'WSPI'),
              methods=('WSPI', 'WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT',
                       'Persistence', 'Holt', 'ARYW', 'MA64')),
    'B': dict(scenarios=('youtube_hourly', 'taxi_hourly'), references=('WSPI-FA',),
              methods=('WSPI', 'WSPI-FA', 'ARIMA')),
}
N_BOOT = 10000
SEED = 42


def block_length(scenario: str, horizon: int) -> int:
    return max(BASE_BLOCK[scenario], int(horizon))


def read_protocol(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def window_check(run: str, scenario: str, horizon: int, folder: Path, methods):
    """Rows of the window check of one run folder, and whether it passed."""
    rows, ok = [], True
    pdir = folder / 'protocol'
    found = sorted(f.name[:-len('_protocol.csv')] for f in pdir.glob('*_protocol.csv'))
    ref = read_protocol(pdir / 'WSPI_protocol.csv') if 'WSPI' in found else None
    for m in sorted(set(methods) | set(found)):
        row = dict(run=run, scenario=scenario, horizon=horizon, method=m, expected=m in methods,
                   present=m in found, n_windows=0, first_window=np.nan, last_window=np.nan,
                   same_windows=False, same_num_items=False, n_nonfinite=0, status='FAIL')
        if m in found and ref is not None:
            d = read_protocol(pdir / f'{m}_protocol.csv')
            same_w = np.array_equal(d['window_id'].to_numpy(), ref['window_id'].to_numpy())
            same_n = same_w and np.array_equal(d['num_items'].to_numpy(),
                                               ref['num_items'].to_numpy())
            row.update(n_windows=len(d),
                       first_window=int(d['window_id'].min()) if len(d) else np.nan,
                       last_window=int(d['window_id'].max()) if len(d) else np.nan,
                       same_windows=bool(same_w), same_num_items=bool(same_n),
                       n_nonfinite=int(d['n_nonfinite'].sum()))
            if same_w and same_n and m in methods:
                row['status'] = 'OK'
            elif same_w and same_n:
                row['status'] = 'EXTRA'           # a method that is not part of the run
        ok &= row['status'] == 'OK'
        rows.append(row)
    return rows, ok


def cross_run_check(scenario: str, horizon: int, folder_a: Path, folder_b: Path):
    fa, fb = folder_a / 'protocol' / 'WSPI_protocol.csv', folder_b / 'protocol' / 'WSPI_protocol.csv'
    row = dict(run='A=B', scenario=scenario, horizon=horizon, method='WSPI', expected=True,
               present=fa.exists() and fb.exists(), n_windows=0, first_window=np.nan,
               last_window=np.nan, same_windows=False, same_num_items=False,
               n_nonfinite=0, status='FAIL')
    if row['present']:
        a, b = read_protocol(fa), read_protocol(fb)
        same = (a.shape == b.shape and list(a.columns) == list(b.columns))
        if same:
            for c in a.columns:
                x, y = a[c], b[c]
                if pd.api.types.is_float_dtype(x):
                    xv, yv = x.to_numpy(float), y.to_numpy(float)
                    same &= bool(((xv == yv) | (np.isnan(xv) & np.isnan(yv))).all())
                else:
                    same &= bool((x.astype(str) == y.astype(str)).all())
        row.update(n_windows=len(a), same_windows=bool(same), same_num_items=bool(same),
                   status='OK' if same else 'FAIL')
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='results/paper2')
    ap.add_argument('--out', default=None, help='default: <root>/stats')
    ap.add_argument('--runs', nargs='+', default=['A', 'B'], choices=sorted(RUNS))
    ap.add_argument('--scenarios', nargs='+', default=None, choices=sorted(BASE_BLOCK))
    ap.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    ap.add_argument('--B', type=int, default=N_BOOT, help='bootstrap resamples (fixed: 10000)')
    ap.add_argument('--force', action='store_true', help='recompute existing reports')
    ap.add_argument('--allow-missing', action='store_true')
    a = ap.parse_args(argv)

    root = Path(a.root)
    root = root if root.is_absolute() else ROOT / root
    out = Path(a.out) if a.out else root / 'stats'
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wrows, missing, failed, reports = [], [], [], []
    for run in a.runs:
        spec = RUNS[run]
        for s in spec['scenarios']:
            if a.scenarios and s not in a.scenarios:
                continue
            for h in a.horizons:
                folder = root / f'run_{run}' / s / f'h{h}'
                tag = f'run_{run}/{s}/h{h}'
                if not (folder / 'metadata' / 'protocol_v5_run.json').exists():
                    missing.append(tag)
                    continue
                rows, ok = window_check(run, s, h, folder, spec['methods'])
                wrows += rows
                if run == 'B' and (root / 'run_A' / s / f'h{h}' / 'protocol').is_dir():
                    r = cross_run_check(s, h, root / 'run_A' / s / f'h{h}', folder)
                    wrows.append(r)
                    ok &= r['status'] == 'OK'
                if not ok:
                    failed.append(tag)
                    print(f'WINDOW CHECK FAILED  {tag}  (no tests for this folder)')
                    continue
                b = block_length(s, h)
                for ref in spec['references']:
                    o = out / f'run_{run}' / s / f'h{h}' / f'ref_{ref}'
                    done = (o / 'paired_tests.csv').exists() and \
                        (o / 'metadata' / 'stats_run.json').exists()
                    if done and not a.force:
                        state = 'kept'
                    else:
                        t1 = time.time()
                        stats_report.analyse(folder, o, b, None, ref, a.B, SEED, None)
                        state = f'{time.time() - t1:.1f}s'
                    reports.append(dict(run=run, scenario=s, horizon=h, reference=ref,
                                        block=b, folder=str(o.relative_to(out).as_posix())))
                    print(f'{tag:<34} ref {ref:<8} block {b:<5} {state}', flush=True)

    wc = pd.DataFrame(wrows)
    wc.to_csv(out / 'window_check.csv', index=False, encoding='utf-8')
    for name in ('paired_tests', 'method_summary'):
        parts = []
        for r in reports:
            f = out / r['folder'] / f'{name}.csv'
            d = pd.read_csv(f, float_precision='round_trip')
            if name == 'method_summary':
                d.insert(0, 'reference_report', r['reference'])
            d.insert(0, 'horizon', r['horizon'])
            d.insert(0, 'scenario', r['scenario'])
            d.insert(0, 'run', r['run'])
            parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / f'all_{name}.csv', index=False,
                                                       encoding='utf-8')
    plan = dict(root=str(root), runs=a.runs, horizons=a.horizons, base_block=BASE_BLOCK,
                block_rule='max(base block, horizon)', B=a.B, seed=SEED,
                references={r: list(RUNS[r]['references']) for r in a.runs},
                reports=reports, missing=missing, window_check_failed=failed,
                created=time.strftime('%Y-%m-%d %H:%M:%S'),
                runtime_seconds=round(time.time() - t0, 1))
    (out / 'stats_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')

    print(f'\nreports: {len(reports)}; run folders missing: {len(missing)}; '
          f'window check failed: {len(failed)}; {time.time() - t0:.0f}s')
    for m in missing:
        print(f'  missing: {m}')
    print(f'Saved to {out}')
    bad = bool(failed) or (bool(missing) and not a.allow_missing) or a.B != N_BOOT
    if a.B != N_BOOT:
        print(f'NOTE: B = {a.B} is not the registered value {N_BOOT}; test run only')
    print('STATS FAILED' if bad else 'STATS OK')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
