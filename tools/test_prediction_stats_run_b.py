r"""
Unit checks for --part test --runs B (tools/run_prediction_stats.py)
====================================================================
Uses the synthetic run A folders of ``tools/test_prediction_stats_part.py`` and
adds run B folders (WSPI, WSPI-FA, ARIMA) for YouTube and the hourly taxi data.

  1. Reports: one per scenario and horizon of run B, reference WSPI-FA, the two
     other run B methods; exit code 1 only because B is not 10000.
  2. Only test windows: the windows of split_ids (first 30 % tuning) of the
     run A WSPI windows; the means of the report equal the means of those
     windows in the protocol files.
  3. A planted effect (WSPI-FA higher RSI@10 than ARIMA in three cells) is
     found in exactly those cells.
  4. A run B WSPI file that differs from run A stops the tests of that cell.
  5. Refusals: --part tune --runs B, --runs B C, --references, --out stats_test.
  6. The test reports of run C (stats_test/) are not touched.

    python tools/test_prediction_stats_run_b.py
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import run_prediction_stats as S                               # noqa: E402
from tools import test_prediction_stats_part as T                         # noqa: E402
from tools.run_param_grid import split_ids                                # noqa: E402

FAILED = []
B_TEST = 200
SCEN_B = S.RUNS['B']['scenarios']
PLANTED = 3                      # first cells (scenario-major) of run B with a planted effect


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def digest(folder: Path) -> dict:
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file() and f.name != 'stats_plan.json'}


def build_b(root: Path):
    """Run B folders: WSPI copied byte for byte from run A; WSPI-FA and ARIMA from WSPI-F2 of
    run A with small noise; WSPI-FA gets a higher RSI@10 in the first cells."""
    k = 0
    for i, s in enumerate(SCEN_B):
        for h in S.HORIZONS:
            fa = root / 'run_A' / s / f'h{h}' / 'protocol'
            rng = np.random.default_rng([100 + i, h])
            base = pd.read_csv(fa / 'WSPI-F2_protocol.csv')
            frames = {}
            for m in ('WSPI-FA', 'ARIMA'):
                d = base.copy()
                d['method'] = m
                for c in S.MAIN_METRICS:
                    d[c] = d[c] + rng.normal(0, 0.001, len(d))
                if m == 'WSPI-FA' and k < PLANTED:
                    d['rsi@10'] = d['rsi@10'] + 0.05
                frames[m] = d
            T.write(root / 'run_B' / s / f'h{h}', frames)
            shutil.copyfile(fa / 'WSPI_protocol.csv',             # same bytes as run A
                            root / 'run_B' / s / f'h{h}' / 'protocol' / 'WSPI_protocol.csv')
            k += 1


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'p2'
        T.build(root)
        build_b(root)
        S.main(['--root', str(root), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST),
                '--allow-missing'])
        S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST)])
        test_dir = root / 'stats_test'
        before = digest(test_dir)

        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'B', '--B', str(B_TEST)])
        out = root / 'stats_test_B'
        check('1 exit code 1 only because B is not 10000', rc == 1)
        p = pd.read_csv(out / 'all_paired_tests.csv')
        check('1 one reference WSPI-FA', sorted(p['reference'].unique()) == ['WSPI-FA'])
        n_rep = p.groupby(['scenario', 'horizon']).ngroups
        check('1 one report per run B cell', n_rep == len(SCEN_B) * len(S.HORIZONS), str(n_rep))
        check('1 the other two methods', sorted(p['method'].unique()) == ['ARIMA', 'WSPI'])
        check('1 columns part and run', (p['part'] == 'test').all() and (p['run'] == 'B').all())
        plan = json.loads((out / 'stats_plan.json').read_text())
        check('1 plan', plan.get('runs') == ['B'] and plan.get('references') == ['WSPI-FA']
              and len(plan.get('reports', [])) == n_rep and not plan['window_check_failed'])
        wc = pd.read_csv(out / 'window_check.csv')
        check('1 window check OK (3 methods + A=B per cell)',
              len(wc) == 4 * n_rep and (wc['status'] == 'OK').all(), str(len(wc)))

        pw = pd.read_csv(out / 'part_windows.csv')
        ms = pd.read_csv(out / 'all_method_summary.csv')
        ok_ids, ok_mean = True, True
        for s in SCEN_B:
            for h in S.HORIZONS:
                fb = root / 'run_B' / s / f'h{h}' / 'protocol'
                ids = pd.read_csv(root / 'run_A' / s / f'h{h}' / 'protocol' /
                                  'WSPI_protocol.csv')['window_id'].tolist()
                _, test_ids = split_ids(ids)
                r = pw[(pw.scenario == s) & (pw.horizon == h)].iloc[0]
                ok_ids &= int(r.n_test) == len(test_ids) and int(r.first_test) == test_ids[0] \
                    and int(r.last_test) == test_ids[-1]
                for m in S.RUNS['B']['methods']:
                    d = pd.read_csv(fb / f'{m}_protocol.csv')
                    d = d[d['window_id'].isin(set(test_ids))]
                    for c in S.MAIN_METRICS:
                        row = ms[(ms.scenario == s) & (ms.horizon == h) & (ms.method == m)
                                 & (ms.metric == c)]
                        ok_mean &= len(row) == 1 and int(row['n_windows'].iloc[0]) == len(d) \
                            and np.isclose(row['mean'].iloc[0], d[c].mean(), rtol=0, atol=1e-12)
        check('2 only test windows (split_ids of run A)', bool(ok_ids))
        check('2 means equal the means of the test windows', bool(ok_mean))

        d = p[(p.method == 'ARIMA') & (p.metric == 'rsi@10')].copy()
        order = {(s, h): i * 3 + k for i, s in enumerate(SCEN_B) for k, h in enumerate(S.HORIZONS)}
        d['cell'] = [order[(s, int(h))] for s, h in zip(d.scenario, d.horizon)]
        better = sorted(d[d.verdict == 'ref_better']['cell'])
        check('3 planted effect found in exactly those cells', better == list(range(PLANTED)),
              str(better))

        check('6 stats_test unchanged', digest(test_dir) == before)

        bad = root / 'bad'
        shutil.copytree(root, bad, ignore=shutil.ignore_patterns('stats_*'))
        f = bad / 'run_B' / SCEN_B[0] / 'h8' / 'protocol' / 'WSPI_protocol.csv'
        w = pd.read_csv(f)
        w.loc[5, 'ndcg@10'] = w.loc[5, 'ndcg@10'] + 0.01
        w.to_csv(f, index=False)
        rc = S.main(['--root', str(bad), '--part', 'test', '--runs', 'B', '--B', str(B_TEST)])
        pb = json.loads((bad / 'stats_test_B' / 'stats_plan.json').read_text())
        check('4 differing WSPI file stops that cell',
              rc == 1 and pb['window_check_failed'] == [f'{SCEN_B[0]}/h8']
              and len(pb['reports']) == n_rep - 1, str(pb['window_check_failed']))

        rc = S.main(['--root', str(root), '--part', 'tune', '--runs', 'B', '--B', str(B_TEST)])
        check('5 refused with --part tune', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'B', 'C', '--B', str(B_TEST)])
        check('5 refused with --runs B C', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'B', '--B', str(B_TEST),
                     '--references', 'WSPI', '--out', str(root / 'x')])
        check('5 refused with --references', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'B', '--B', str(B_TEST),
                     '--out', str(test_dir)])
        check('5 refused with --out stats_test', rc == 2)
        check('6 stats_test unchanged after the refusals', digest(test_dir) == before)


def main():
    reg = dict(S.BASE_BLOCK)
    for s in S.BASE_BLOCK:
        S.BASE_BLOCK[s] = 4                       # short blocks for the small test series
    try:
        run_tests()
    finally:
        S.BASE_BLOCK.update(reg)
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)}')
        for f in FAILED:
            print('  ' + f)
        return 1
    print('ALL OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
