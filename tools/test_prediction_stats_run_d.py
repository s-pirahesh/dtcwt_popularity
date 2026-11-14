r"""
Unit checks for --part tune|test --runs D (tools/run_prediction_stats.py)
========================================================================
Uses the synthetic run A and run C folders of ``tools/test_prediction_stats_part.py``
and adds run D folders (WSPI, WSPI-M, WSPI-H) for every scenario and horizon.

  1. Reports, both parts: one per cell and reference; references
     tune: WSPI-H, WSPI-M, WSPI-F2, WSPI, WSPI-F-YW and test: WSPI-H, WSPI-M;
     the method family (test: the 21 methods of stats_test plus WSPI-M and
     WSPI-H; tune: the same without GBM and GBM+W); exit code 1 only because
     B is not 10000.
  2. Only the windows of the part (split_ids of the run A WSPI windows); the
     means of the report equal the means of those windows in the protocol files.
  3. A planted effect (WSPI-H higher RSI@10 than MA64 in the first cells, test
     part only) is found in exactly those cells of the test part and in none of
     the tuning part.
  4. A run D WSPI file that differs from run A stops the tests of that cell.
  5. Refusals: --runs D C, --references, --out stats_tune or stats_test,
     a missing kappa selection.
  6. The reports of stats_tune/ and stats_test/ are not touched.

    python tools/test_prediction_stats_run_d.py
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
SCEN = tuple(S.BASE_BLOCK)
PLANTED = 4                      # first cells (scenario-major) with a planted effect


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def digest(folder: Path) -> dict:
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file() and f.name != 'stats_plan.json'}


def build_d(root: Path):
    """Run D folders: WSPI copied byte for byte from run A; WSPI-M and WSPI-H from
    WSPI-F2 of run A with small noise; WSPI-H gets a higher RSI@10 on the test
    windows of the first cells (the tuning windows are left as they are)."""
    k = 0
    for i, s in enumerate(SCEN):
        for h in S.HORIZONS:
            fa = root / 'run_A' / s / f'h{h}' / 'protocol'
            rng = np.random.default_rng([200 + i, h])
            base = pd.read_csv(fa / 'WSPI-F2_protocol.csv')
            _, test_ids = split_ids(pd.read_csv(fa / 'WSPI_protocol.csv')['window_id'].tolist())
            is_test = base['window_id'].isin(set(test_ids)).to_numpy()
            frames = {}
            for m in ('WSPI-M', 'WSPI-H'):
                d = base.copy()
                d['method'] = m
                for c in S.MAIN_METRICS:
                    d[c] = d[c] + rng.normal(0, 0.001, len(d))
                if m == 'WSPI-H' and k < PLANTED:
                    d.loc[is_test, 'rsi@10'] = d.loc[is_test, 'rsi@10'] + 0.08
                frames[m] = d
            T.write(root / 'run_D' / s / f'h{h}', frames)
            shutil.copyfile(fa / 'WSPI_protocol.csv',
                            root / 'run_D' / s / f'h{h}' / 'protocol' / 'WSPI_protocol.csv')
            k += 1


def cell_order():
    return {(s, h): i * 3 + k for i, s in enumerate(SCEN) for k, h in enumerate(S.HORIZONS)}


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'p2'
        T.build(root)
        build_d(root)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'D', '--B', str(B_TEST)])
        check('5 refused without a kappa selection', rc == 2)
        S.main(['--root', str(root), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST),
                '--allow-missing'])
        S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST)])
        kappa = json.loads((root / 'stats_tune' / 'kappa_selection.json').read_text())['selected_kappa']
        fs_sel = f'WSPI-FS{float(kappa):g}'
        before = {p: digest(root / p) for p in ('stats_tune', 'stats_test')}
        n_cells = len(SCEN) * len(S.HORIZONS)

        for part in ('tune', 'test'):
            rc = S.main(['--root', str(root), '--part', part, '--runs', 'D', '--B', str(B_TEST)])
            out = root / f'stats_{part}_D'
            check(f'1 {part}: exit code 1 only because B is not 10000', rc == 1)
            p = pd.read_csv(out / 'all_paired_tests.csv')
            refs = list(S.D_REFERENCES[part])
            check(f'1 {part}: references', sorted(p['reference'].unique()) == sorted(refs),
                  str(sorted(p['reference'].unique())))
            n_rep = p.groupby(['scenario', 'horizon', 'reference']).ngroups
            check(f'1 {part}: one report per cell and reference', n_rep == n_cells * len(refs),
                  str(n_rep))
            family = set(S.RUNS['A']['methods']) | set(S.TEST_FROM_C) | {fs_sel} | set(S.LEVEL_NAMES)
            if part == 'tune':
                family -= {'GBM', 'GBM+W'}
            got = set(p[p.reference == 'WSPI-H']['method']) | {'WSPI-H'}
            check(f'1 {part}: method family ({len(family)} methods)', got == family,
                  str(sorted(got ^ family)))
            check(f'1 {part}: columns part and run', (p['part'] == part).all() and
                  (p['run'] == 'D').all())
            plan = json.loads((out / 'stats_plan.json').read_text())
            check(f'1 {part}: plan', plan.get('runs') == ['D'] and plan.get('references') == refs
                  and plan.get('selected_kappa') == kappa and not plan['window_check_failed']
                  and len(plan['reports']) == n_cells * len(refs))
            wc = pd.read_csv(out / 'window_check.csv')
            n_rows = len(S.RUNS['A']['methods']) + len(S.SUPP_RUNS['C']['methods']) + 3 + 3
            check(f'1 {part}: window check OK (A, C, D, A=C, A=D, MA64=SMA64 per cell)',
                  len(wc) == n_rows * n_cells and (wc['status'] == 'OK').all(), str(len(wc)))

            pw = pd.read_csv(out / 'part_windows.csv')
            ms = pd.read_csv(out / 'all_method_summary.csv')
            ok_ids, ok_mean = True, True
            for s in SCEN:
                for h in S.HORIZONS:
                    fa = root / 'run_A' / s / f'h{h}' / 'protocol'
                    ids = pd.read_csv(fa / 'WSPI_protocol.csv')['window_id'].tolist()
                    tune_ids, test_ids = split_ids(ids)
                    keep = tune_ids if part == 'tune' else test_ids
                    r = pw[(pw.scenario == s) & (pw.horizon == h)].iloc[0]
                    n_col, f_col, l_col = (('n_tune', 'first_tune', 'last_tune') if part == 'tune'
                                           else ('n_test', 'first_test', 'last_test'))
                    ok_ids &= int(r[n_col]) == len(keep) and int(r[f_col]) == keep[0] \
                        and int(r[l_col]) == keep[-1]
                    for m, f in (('WSPI-H', root / 'run_D' / s / f'h{h}' / 'protocol'),
                                 ('MA64', fa)):
                        d = pd.read_csv(f / f'{m}_protocol.csv')
                        d = d[d['window_id'].isin(set(keep))]
                        for c in S.MAIN_METRICS:
                            row = ms[(ms.scenario == s) & (ms.horizon == h) & (ms.method == m)
                                     & (ms.metric == c) & (ms.reference_report == 'WSPI-H')]
                            ok_mean &= len(row) == 1 and int(row['n_windows'].iloc[0]) == len(d) \
                                and np.isclose(row['mean'].iloc[0], d[c].mean(), rtol=0, atol=1e-12)
            check(f'2 {part}: only the windows of the part (split_ids of run A)', bool(ok_ids))
            check(f'2 {part}: means equal the means of those windows', bool(ok_mean))

            d = p[(p.reference == 'WSPI-H') & (p.method == 'MA64') & (p.metric == 'rsi@10')].copy()
            order = cell_order()
            d['cell'] = [order[(s, int(h))] for s, h in zip(d.scenario, d.horizon)]
            better = sorted(d[d.verdict == 'ref_better']['cell'])
            want = list(range(PLANTED)) if part == 'test' else []
            check(f'3 {part}: planted effect only where planted', better == want, str(better))

        for p_ in ('stats_tune', 'stats_test'):
            check(f'6 {p_} unchanged', digest(root / p_) == before[p_])

        bad = root / 'bad'
        shutil.copytree(root, bad, ignore=shutil.ignore_patterns('stats_*_D'))
        f = bad / 'run_D' / SCEN[1] / 'h24' / 'protocol' / 'WSPI_protocol.csv'
        w = pd.read_csv(f)
        w.loc[3, 'rsi@10'] = w.loc[3, 'rsi@10'] + 0.01
        w.to_csv(f, index=False)
        rc = S.main(['--root', str(bad), '--part', 'test', '--runs', 'D', '--B', str(B_TEST)])
        pb = json.loads((bad / 'stats_test_D' / 'stats_plan.json').read_text())
        check('4 differing run D WSPI file stops that cell',
              rc == 1 and pb['window_check_failed'] == [f'{SCEN[1]}/h24']
              and len(pb['reports']) == (n_cells - 1) * 2, str(pb['window_check_failed']))

        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'D', 'C', '--B', str(B_TEST)])
        check('5 refused with --runs D C', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'D', '--B', str(B_TEST),
                     '--references', 'WSPI', '--out', str(root / 'x')])
        check('5 refused with --references', rc == 2)
        for p_ in ('stats_tune', 'stats_test'):
            part = p_.split('_')[1]
            rc = S.main(['--root', str(root), '--part', part, '--runs', 'D', '--B', str(B_TEST),
                         '--out', str(root / p_)])
            check(f'5 refused with --out {p_}', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'all', '--runs', 'D', '--B', str(B_TEST)])
        check('5 refused with --part all', rc == 2)
        for p_ in ('stats_tune', 'stats_test'):
            check(f'6 {p_} unchanged after the refusals', digest(root / p_) == before[p_])


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
