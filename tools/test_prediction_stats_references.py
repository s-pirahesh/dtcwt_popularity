r"""
Unit checks for --references of the test part (tools/run_prediction_stats.py)
=============================================================================
Uses the synthetic run folders of ``tools/test_prediction_stats_part.py``.

  1. Default: without --references the test part writes the five fixed
     references, as before.
  2. Extra reference: --references WSPI-F-YW --out <other folder> writes one
     report per cell with WSPI-F-YW as the reference and the other 20 methods
     of the test report; only test windows; the plan lists the reference.
  3. The extra report agrees with the fixed ones: against WSPI-F2 the mean
     difference has the opposite sign and the same size, and the raw block p
     value is the same (two-sided test).
  4. A planted effect (WSPI-F-YW lower Delta-Rank than MA64 in six cells) is
     found in exactly those cells.
  5. stats_test/ is not touched by the extra run (same bytes before and after).
  6. Refusals: --references with --out stats_test, with --part tune, or with a
     method that is not in the test report.

    python tools/test_prediction_stats_references.py
"""
import hashlib
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
EXTRA = 'WSPI-F-YW'
PLANTED = 6                      # first cells (scenario-major) with a planted effect


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def digest(folder: Path) -> dict:
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file() and f.name != 'stats_plan.json'}


def plant(root: Path):
    """WSPI-F-YW of run A gets a clearly lower Delta-Rank in the first cells."""
    k = 0
    for s in T.SCEN:
        for h in S.HORIZONS:
            if k < PLANTED:
                f = root / 'run_A' / s / f'h{h}' / 'protocol' / f'{EXTRA}_protocol.csv'
                d = pd.read_csv(f)
                d['robustness_distortion'] = d['robustness_distortion'] - 2.0
                d.to_csv(f, index=False)
            k += 1


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'p2'
        T.build(root)
        plant(root)
        rc = S.main(['--root', str(root), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST),
                     '--allow-missing'])
        check('tuning part runs', rc in (0, 1) and (root / 'stats_tune' / 'kappa_selection.json').exists())
        S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST)])
        test_dir = root / 'stats_test'
        paired = pd.read_csv(test_dir / 'all_paired_tests.csv')
        fixed = sorted(paired['reference'].unique())
        sel = 'WSPI-FS' + format(float(__import__('json').loads(
            (root / 'stats_tune' / 'kappa_selection.json').read_text())['selected_kappa']), 'g')
        check('1 default: five fixed references',
              fixed == sorted([sel, 'WSPI-F2', 'WSPI', 'GBM', 'GBM+W']), str(fixed))
        before = digest(test_dir)

        ex = root / 'stats_explore'
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST),
                     '--references', EXTRA, '--out', str(ex)])
        check('2 extra run exit code 1 only because B is not 10000', rc == 1)
        p = pd.read_csv(ex / 'all_paired_tests.csv')
        check('2 one reference', sorted(p['reference'].unique()) == [EXTRA])
        n_rep = p.groupby(['scenario', 'horizon']).ngroups
        check('2 one report per cell', n_rep == len(T.SCEN) * len(S.HORIZONS), str(n_rep))
        meth = set(p['method'])
        expect = (set(S.RUNS['A']['methods']) | set(S.TEST_FROM_C) | {sel}) - {EXTRA}
        check('2 the other 20 methods', meth == expect and len(meth) == 20, str(len(meth)))
        plan = __import__('json').loads((ex / 'stats_plan.json').read_text())
        check('2 plan lists the reference', plan.get('references') == [EXTRA])
        pw = pd.read_csv(ex / 'part_windows.csv')
        f = root / 'run_A' / T.SCEN[0] / 'h1' / 'protocol' / 'WSPI_protocol.csv'
        _, test_ids = split_ids(pd.read_csv(f)['window_id'].tolist())
        r0 = pw[(pw.scenario == T.SCEN[0]) & (pw.horizon == 1)].iloc[0]
        check('2 only test windows', int(r0.n_test) == len(test_ids)
              and int(r0.first_test) == test_ids[0])

        a = p[p.method == 'WSPI-F2'].set_index(['scenario', 'horizon', 'metric'])
        b = paired[(paired.reference == 'WSPI-F2') & (paired.method == EXTRA)] \
            .set_index(['scenario', 'horizon', 'metric'])
        j = a.join(b, lsuffix='_x', rsuffix='_f', how='inner')
        check('3 same rows', len(j) == len(a) == len(b), f'{len(j)} {len(a)} {len(b)}')
        check('3 mean difference: opposite sign, same size',
              np.allclose(j['diff_ref_minus_method_x'], -j['diff_ref_minus_method_f'],
                          rtol=0, atol=1e-12))
        check('3 raw block p value equal', np.allclose(j['p_block_x'], j['p_block_f'],
                                                       rtol=1e-12, atol=0, equal_nan=True))

        d = p[(p.method == 'MA64') & (p.metric == 'robustness_distortion')].copy()
        order = {(s, h): i * 3 + k for i, s in enumerate(T.SCEN) for k, h in enumerate(S.HORIZONS)}
        d['cell'] = [order[(s, int(h))] for s, h in zip(d.scenario, d.horizon)]
        better = sorted(d[d.verdict == 'ref_better']['cell'])
        check('4 planted effect found in exactly those cells', better == list(range(PLANTED)),
              str(better))

        check('5 stats_test unchanged', digest(test_dir) == before)

        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST),
                     '--references', EXTRA])
        check('6 refused without another --out', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST),
                     '--references', EXTRA, '--out', str(test_dir)])
        check('6 refused with --out stats_test', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST),
                     '--references', EXTRA, '--out', str(root / 'x')])
        check('6 refused with --part tune', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST),
                     '--references', 'ARIMA', '--out', str(root / 'y')])
        check('6 refused for a method outside the test report', rc == 2)
        check('5 stats_test unchanged after the refusals', digest(test_dir) == before)


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
