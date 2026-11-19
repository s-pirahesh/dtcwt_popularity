r"""
Check that a horizon-1 run reproduces the stored results exactly
================================================================
Horizon 1 of ``tools/run_prediction.py`` is protocol V5.  For the methods that
were run before (WSPI, WSPI-F, WSPI-F-YW, WSPI-F2, WSPI-FT, Persistence, Holt,
ARYW) the window-by-window results must therefore equal the stored reference
run (``results/thesis_v16/prediction``) with a difference of exactly zero.
Methods that exist only in the new run (WSPI-FG, MA64, ...) are not compared.

For every scenario the script compares

  protocol      every ``protocol/<method>_protocol.csv`` of the reference:
                same windows, same columns, every cell equal (NaN equals NaN)
  gate_profile  the columns the reference has (``gate_profile.csv``), if present
  gate_hist     the columns the reference has (``gate_hist.csv``), if present
  metadata      data size, slots, item rule, seed, window and minimum
                observations of every shared method

CSV files are read with ``float_precision='round_trip'``; the default parser
can change the last digit of a float and would show a difference that is not
in the file.  The tolerance is zero.  A scenario or a reference method that is
missing from the candidate is a failure, not a skip.

Layout of the candidate: ``<candidate>/<scenario>/h1/...``, which is what
``run_prediction.py --horizons 1 --out <candidate>/<scenario>`` writes.

    python tools/check_reproduction.py --candidate results/prediction/repro
    python tools/check_reproduction.py --candidate results/prediction/smoke ^
           --scenarios youtube_hourly

Exit code 0 and a last line ``REPRODUCTION OK`` if every check passes,
otherwise 1 and ``REPRODUCTION FAILED``.  The table of all checks is written
to ``<candidate>/reproduction_report.csv`` (``--report`` to change it).
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
DEFAULT_REFERENCE = 'results/thesis_v16/prediction'
META_KEYS = ('protocol', 'mode', 'seed', 'rsi_by_item', 'causal_universe',
             'universe_rule', 'tie_rule', 'nonfinite_rule', 'robustness',
             'slot_minutes', 'num_items', 'num_slots', 'dataset_min_obs',
             'window', 'min_obs', 'horizon')
METHOD_KEYS = ('window_slots', 'min_obs')


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def compare_frames(cand: pd.DataFrame, ref: pd.DataFrame, columns=None):
    """Cell-by-cell comparison of two frames with the same rows in the same
    order.  Returns (cells_compared, cells_different, max_abs_diff, note)."""
    columns = list(ref.columns) if columns is None else list(columns)
    missing = [c for c in columns if c not in cand.columns]
    if missing:
        return 0, 0, float('nan'), f'columns missing in candidate: {missing}'
    if len(cand) != len(ref):
        return 0, 0, float('nan'), f'row count {len(cand)} != reference {len(ref)}'
    n_diff, n_cells, dmax = 0, 0, 0.0
    for c in columns:
        x, y = cand[c], ref[c]
        n_cells += len(x)
        numeric = (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)
                   and not pd.api.types.is_bool_dtype(x) and not pd.api.types.is_bool_dtype(y))
        if numeric:
            a, b = x.to_numpy(float), y.to_numpy(float)
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            with np.errstate(invalid='ignore'):
                d = np.abs(a - b)
            d = d[~same]
            d = d[np.isfinite(d)]
            if len(d):
                dmax = max(dmax, float(d.max()))
            if (~same).any() and not len(d):
                dmax = float('inf')           # differs only by inf / nan
        else:
            same = (x.astype(str) == y.astype(str)).to_numpy()
        n_diff += int((~same).sum())
    return n_cells, n_diff, dmax, ''


def _row(scenario, kind, item, cells, diff, dmax, status, note=''):
    return dict(scenario=scenario, kind=kind, item=item, cells_compared=cells,
                cells_different=diff, max_abs_diff=dmax, status=status, note=note)


def check_protocol(scenario, cand_dir, ref_dir):
    rows = []
    for ref_file in sorted((ref_dir / 'protocol').glob('*_protocol.csv')):
        method = ref_file.name[:-len('_protocol.csv')]
        cand_file = cand_dir / 'protocol' / ref_file.name
        if not cand_file.exists():
            rows.append(_row(scenario, 'protocol', method, 0, 0, float('nan'),
                             'FAIL', 'file missing in candidate'))
            continue
        cand, ref = read_csv(cand_file), read_csv(ref_file)
        if list(cand.columns) != list(ref.columns):
            rows.append(_row(scenario, 'protocol', method, 0, 0, float('nan'), 'FAIL',
                             f'columns differ: {list(cand.columns)} vs {list(ref.columns)}'))
            continue
        if not np.array_equal(cand['window_id'].to_numpy(), ref['window_id'].to_numpy()):
            only_c = sorted(set(cand['window_id']) - set(ref['window_id']))
            only_r = sorted(set(ref['window_id']) - set(cand['window_id']))
            rows.append(_row(scenario, 'protocol', method, 0, 0, float('nan'), 'FAIL',
                             f'windows differ: {len(only_c)} only in candidate, '
                             f'{len(only_r)} only in reference'))
            continue
        cells, diff, dmax, note = compare_frames(cand, ref)
        rows.append(_row(scenario, 'protocol', method, cells, diff, dmax,
                         'OK' if (diff == 0 and not note) else 'FAIL', note))
    return rows


def check_side_file(scenario, kind, name, cand_dir, ref_dir):
    ref_file, cand_file = ref_dir / name, cand_dir / name
    if not ref_file.exists():
        return _row(scenario, kind, name, 0, 0, 0.0, 'SKIP', 'no reference file')
    if not cand_file.exists():
        return _row(scenario, kind, name, 0, 0, 0.0, 'SKIP',
                    'not in candidate (run without --gate-profile)')
    cand, ref = read_csv(cand_file), read_csv(ref_file)
    extra = [c for c in cand.columns if c not in ref.columns]
    cells, diff, dmax, note = compare_frames(cand, ref)
    if extra and not note:
        note = f'extra columns in candidate (not compared): {extra}'
    return _row(scenario, kind, name, cells, diff, dmax,
                'OK' if (diff == 0 and not (note and note.startswith(('columns', 'row')))) else 'FAIL',
                note)


def check_metadata(scenario, cand_dir, ref_dir, methods):
    cf, rf = cand_dir / 'metadata' / 'protocol_v5_run.json', ref_dir / 'metadata' / 'protocol_v5_run.json'
    if not cf.exists() or not rf.exists():
        return _row(scenario, 'metadata', 'protocol_v5_run.json', 0, 0, float('nan'),
                    'FAIL', 'metadata file missing')
    c, r = json.loads(cf.read_text(encoding='utf-8')), json.loads(rf.read_text(encoding='utf-8'))
    bad = [f'{k}: {c.get(k)!r} != {r.get(k)!r}' for k in META_KEYS if c.get(k) != r.get(k)]
    for m in methods:
        for k in METHOD_KEYS:
            cv = c.get('methods', {}).get(m, {}).get(k)
            rv = r.get('methods', {}).get(m, {}).get(k)
            if cv != rv:
                bad.append(f'{m}.{k}: {cv!r} != {rv!r}')
    n = len(META_KEYS) + len(METHOD_KEYS) * len(methods)
    return _row(scenario, 'metadata', 'protocol_v5_run.json', n, len(bad), 0.0,
                'OK' if not bad else 'FAIL', '; '.join(bad))


def check_scenario(scenario, candidate: Path, reference: Path):
    cand_dir, ref_dir = candidate / scenario / 'h1', reference / scenario
    if not cand_dir.is_dir():
        return [_row(scenario, 'scenario', str(cand_dir), 0, 0, float('nan'),
                     'FAIL', 'candidate folder missing')]
    if not ref_dir.is_dir():
        return [_row(scenario, 'scenario', str(ref_dir), 0, 0, float('nan'),
                     'FAIL', 'reference folder missing')]
    methods = [f.name[:-len('_protocol.csv')] for f in sorted((ref_dir / 'protocol').glob('*_protocol.csv'))]
    rows = check_protocol(scenario, cand_dir, ref_dir)
    rows.append(check_side_file(scenario, 'gate_profile', 'gate_profile.csv', cand_dir, ref_dir))
    rows.append(check_side_file(scenario, 'gate_hist', 'gate_hist.csv', cand_dir, ref_dir))
    rows.append(check_metadata(scenario, cand_dir, ref_dir, methods))
    return rows


def run(candidate: Path, reference: Path, scenarios=SCENARIOS):
    rows = []
    for s in scenarios:
        rows += check_scenario(s, candidate, reference)
    report = pd.DataFrame(rows)
    return report, bool((report['status'] != 'FAIL').all())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--candidate', required=True,
                    help='folder with <scenario>/h1/ from tools/run_prediction.py')
    ap.add_argument('--reference', default=DEFAULT_REFERENCE)
    ap.add_argument('--scenarios', nargs='+', default=list(SCENARIOS), choices=SCENARIOS)
    ap.add_argument('--report', default=None,
                    help='CSV of all checks (default: <candidate>/reproduction_report.csv)')
    a = ap.parse_args(argv)

    cand = Path(a.candidate)
    cand = cand if cand.is_absolute() else ROOT / cand
    ref = Path(a.reference)
    ref = ref if ref.is_absolute() else ROOT / ref
    report, ok = run(cand, ref, a.scenarios)

    out = Path(a.report) if a.report else cand / 'reproduction_report.csv'
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False, encoding='utf-8')

    with pd.option_context('display.width', 200, 'display.max_colwidth', 70,
                           'display.max_rows', 500):
        show = report[report['kind'].isin(['protocol']) | (report['status'] != 'OK')]
        print(show[['scenario', 'kind', 'item', 'cells_compared', 'cells_different',
                    'max_abs_diff', 'status']].to_string(index=False))
    for _, r in report[(report['note'] != '') & report['note'].notna()].iterrows():
        print(f"note [{r['scenario']} {r['kind']} {r['item']}]: {r['note']}")
    n_fail = int((report['status'] == 'FAIL').sum())
    p = report[report['kind'] == 'protocol']
    print(f"\nscenarios: {len(a.scenarios)}; protocol files compared: {len(p)}; "
          f"cells compared: {int(p['cells_compared'].sum())}; "
          f"cells different: {int(p['cells_different'].sum())}; "
          f"max abs difference: {p['max_abs_diff'].max() if len(p) else float('nan')}")
    print(f'report: {out}')
    print('REPRODUCTION OK' if ok else f'REPRODUCTION FAILED ({n_fail} failing check(s))')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
