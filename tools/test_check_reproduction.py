r"""
Unit checks for tools/check_reproduction.py
===========================================
Builds a small reference tree and candidate trees in a temporary folder and
checks that the comparison

  1. passes on an identical copy (and ignores methods and columns that only the
     candidate has),
  2. fails on a difference of one unit in the last place of one cell, and
     reports it,
  3. fails on a missing method, a missing window, a missing scenario, a changed
     column, a changed boolean and a changed metadata value,
  4. treats NaN as equal to NaN and skips a gate file the candidate lacks,
  5. reads floats with float_precision='round_trip' (the exact float comes
     back),
  6. returns exit code 0 / 1 and writes the report from main().

    python tools/test_check_reproduction.py
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

from tools import check_reproduction as C                                 # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def protocol_frame(method, n=12, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'window_id': np.arange(32, 32 + n),
                       'timestamp': 1525834800000 + 3600000 * np.arange(n),
                       'method': method, 'num_items': rng.integers(200, 900, n),
                       'ndcg@10': rng.random(n), 'spearman_rho': rng.random(n),
                       'rsi@10': rng.random(n), 'robustness_distortion': rng.random(n) * 30,
                       'ties_top21': False, 'padded': [True] * 5 + [False] * (n - 5),
                       'n_nonfinite': 0})
    df.loc[0, 'rsi@10'] = np.nan            # the first window has no previous ranking
    return df


def write_scenario(root: Path, scenario, methods, with_h1, gate=True, extra=()):
    base = root / scenario / ('h1' if with_h1 else '')
    (base / 'protocol').mkdir(parents=True)
    (base / 'metadata').mkdir()
    for i, m in enumerate(list(methods) + list(extra)):
        protocol_frame(m, seed=i).to_csv(base / 'protocol' / f'{m}_protocol.csv', index=False)
    meta = {'protocol': 'v5', 'mode': 'dense', 'seed': 42, 'rsi_by_item': True,
            'causal_universe': True, 'universe_rule': 'r', 'tie_rule': 't',
            'nonfinite_rule': 'n', 'robustness': True, 'slot_minutes': 60.0,
            'num_items': 100, 'num_slots': 44, 'dataset_min_obs': 50, 'window': 64,
            'min_obs': 32, 'horizon': 1,
            'methods': {m: {'window_slots': 64, 'min_obs': 32} for m in methods}}
    (base / 'metadata' / 'protocol_v5_run.json').write_text(json.dumps(meta), encoding='utf-8')
    if gate:
        gp = pd.DataFrame({'window_id': np.arange(32, 44), 'g_median': np.linspace(0, 1, 12)})
        gh = pd.DataFrame({'bin_lo': [0.0, 0.5], 'bin_hi': [0.5, 1.0], 'count_p': [3, 4]})
        if with_h1:
            gp['fg_g_median'] = 0.1            # a column only the candidate has
        gp.to_csv(base / 'gate_profile.csv', index=False)
        gh.to_csv(base / 'gate_hist.csv', index=False)


def make_trees(tmp: Path, scenarios=('youtube_hourly', 'taxi_hourly')):
    ref, cand = tmp / 'ref', tmp / 'cand'
    methods = ['WSPI', 'WSPI-F2', 'Holt']
    for s in scenarios:
        write_scenario(ref, s, methods, with_h1=False)
        write_scenario(cand, s, methods, with_h1=True, extra=['WSPI-FG', 'MA64'])
    return ref, cand, methods


def status_of(report, scenario, kind, item=None):
    r = report[(report['scenario'] == scenario) & (report['kind'] == kind)]
    if item is not None:
        r = r[r['item'] == item]
    return list(r['status'])


def main():
    tmp = Path(tempfile.mkdtemp())
    try:
        scen = ('youtube_hourly', 'taxi_hourly')

        # 1. identical copy
        ref, cand, _ = make_trees(tmp / 'a')
        rep, ok = C.run(cand, ref, scen)
        p = rep[rep['kind'] == 'protocol']
        check('identical copy passes', ok and (rep['status'] != 'FAIL').all())
        check('only the reference methods are compared (3 per scenario)',
              len(p) == 6 and set(p['item']) == {'WSPI', 'WSPI-F2', 'Holt'})
        check('zero cells different, zero max difference',
              int(p['cells_different'].sum()) == 0 and float(p['max_abs_diff'].max()) == 0.0)
        check('NaN equals NaN (rsi@10 of the first window)', p['cells_compared'].min() == 12 * 11)
        check('gate files compared, extra candidate columns ignored',
              status_of(rep, 'youtube_hourly', 'gate_profile') == ['OK'] and
              status_of(rep, 'youtube_hourly', 'gate_hist') == ['OK'])
        check('metadata compared', status_of(rep, 'taxi_hourly', 'metadata') == ['OK'])

        # 2. one ulp in one cell
        ref, cand, _ = make_trees(tmp / 'b')
        f = cand / 'youtube_hourly' / 'h1' / 'protocol' / 'WSPI-F2_protocol.csv'
        d = pd.read_csv(f, float_precision='round_trip')
        old = d.loc[4, 'ndcg@10']
        d.loc[4, 'ndcg@10'] = np.nextafter(old, np.inf)
        d.to_csv(f, index=False)
        rep, ok = C.run(cand, ref, scen)
        r = rep[(rep['scenario'] == 'youtube_hourly') & (rep['item'] == 'WSPI-F2')].iloc[0]
        check('one-ulp difference in one cell fails', (not ok) and r['status'] == 'FAIL')
        check('...and is reported (1 cell, max diff = 1 ulp)',
              r['cells_different'] == 1 and r['max_abs_diff'] == float(np.nextafter(old, np.inf) - old))
        check('...other methods and the other scenario stay OK',
              status_of(rep, 'youtube_hourly', 'protocol', 'WSPI') == ['OK'] and
              status_of(rep, 'taxi_hourly', 'protocol', 'WSPI-F2') == ['OK'])

        # 3. structural failures
        ref, cand, _ = make_trees(tmp / 'c')
        (cand / 'youtube_hourly' / 'h1' / 'protocol' / 'Holt_protocol.csv').unlink()
        rep, ok = C.run(cand, ref, scen)
        check('missing method fails', (not ok) and
              status_of(rep, 'youtube_hourly', 'protocol', 'Holt') == ['FAIL'])

        ref, cand, _ = make_trees(tmp / 'd')
        f = cand / 'taxi_hourly' / 'h1' / 'protocol' / 'WSPI_protocol.csv'
        pd.read_csv(f, float_precision='round_trip').iloc[:-1].to_csv(f, index=False)
        rep, ok = C.run(cand, ref, scen)
        check('missing window fails', (not ok) and
              status_of(rep, 'taxi_hourly', 'protocol', 'WSPI') == ['FAIL'])

        ref, cand, _ = make_trees(tmp / 'e')
        rep, ok = C.run(cand, ref, scen + ('taxi_30min',))
        check('missing scenario fails', (not ok) and
              status_of(rep, 'taxi_30min', 'scenario') == ['FAIL'])

        ref, cand, _ = make_trees(tmp / 'f')
        f = cand / 'youtube_hourly' / 'h1' / 'protocol' / 'WSPI_protocol.csv'
        pd.read_csv(f, float_precision='round_trip').rename(
            columns={'ndcg@10': 'ndcg10'}).to_csv(f, index=False)
        rep, ok = C.run(cand, ref, scen)
        check('changed column name fails', (not ok) and
              status_of(rep, 'youtube_hourly', 'protocol', 'WSPI') == ['FAIL'])

        ref, cand, _ = make_trees(tmp / 'g')
        f = cand / 'taxi_hourly' / 'h1' / 'protocol' / 'WSPI-F2_protocol.csv'
        d = pd.read_csv(f, float_precision='round_trip')
        d.loc[2, 'ties_top21'] = True
        d.to_csv(f, index=False)
        rep, ok = C.run(cand, ref, scen)
        check('changed boolean cell fails', (not ok) and
              status_of(rep, 'taxi_hourly', 'protocol', 'WSPI-F2') == ['FAIL'])

        ref, cand, _ = make_trees(tmp / 'h')
        f = cand / 'youtube_hourly' / 'h1' / 'metadata' / 'protocol_v5_run.json'
        m = json.loads(f.read_text(encoding='utf-8'))
        m['num_items'] = 101
        f.write_text(json.dumps(m), encoding='utf-8')
        rep, ok = C.run(cand, ref, scen)
        check('changed metadata (num_items) fails', (not ok) and
              status_of(rep, 'youtube_hourly', 'metadata') == ['FAIL'])

        # 4. gate file absent in the candidate is a skip
        ref, cand, _ = make_trees(tmp / 'i')
        (cand / 'youtube_hourly' / 'h1' / 'gate_profile.csv').unlink()
        rep, ok = C.run(cand, ref, scen)
        check('gate file missing in the candidate is skipped, not failed', ok and
              status_of(rep, 'youtube_hourly', 'gate_profile') == ['SKIP'])

        # 5. round-trip reading
        x = 0.1 + 0.2                                    # 0.30000000000000004
        f = tmp / 'rt.csv'
        f.write_text(f'a\n{x!r}\n', encoding='utf-8')
        check('round_trip parser returns the exact float',
              C.read_csv(f)['a'][0] == x)

        # 6. main(): exit codes and report
        ref, cand, _ = make_trees(tmp / 'j')
        argv = ['--candidate', str(cand), '--reference', str(ref),
                '--scenarios', 'youtube_hourly', 'taxi_hourly']
        code = C.main(argv)
        check('main() returns 0 on identical trees', code == 0)
        check('report written', (cand / 'reproduction_report.csv').exists())
        f = cand / 'taxi_hourly' / 'h1' / 'protocol' / 'Holt_protocol.csv'
        d = pd.read_csv(f, float_precision='round_trip')
        d.loc[0, 'spearman_rho'] += 1e-3
        d.to_csv(f, index=False)
        check('main() returns 1 when a cell differs', C.main(argv) == 1)
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
