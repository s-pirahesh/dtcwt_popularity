r"""
Unit checks for the confirmation data (--confirm)
=================================================
``tools/run_prediction_plan.py --confirm`` and
``tools/run_prediction_stats.py --confirm --part tune|test`` on synthetic run
folders (no data file is read).

  1. Plan: 12 jobs per plan (four held-out scenarios x three horizons) under
     results/prediction/confirm; the data files and item thresholds of the
     held-out scenarios; run C with WSPI-FS3 only among the gates; the
     development plans are unchanged; refusals (--confirm with plan B, C_sens,
     C_nlms; a development scenario with --confirm).
  2. Stats, both parts: one report per cell and reference (WSPI-H, WSPI-F2,
     WSPI-M, WSPI); method family 21 (tune) and 23 (test); the kappa of the
     development tuning part; block max(b, h); exit code 1 only because B is
     not 10000; window check OK.
  3. Only the windows of the part (split_ids of the confirmation run A WSPI
     windows); the means equal the means of those windows.
  4. A planted effect (WSPI-H higher RSI@10 than MA64 on the test windows of the
     first cells) is found in exactly those cells of the test part and in none
     of the tuning part.
  5. A confirmation run C WSPI file that differs from run A, or a run C without
     WSPI-FS3, stops the tests of that cell.
  6. Refusals: --part all, --runs, --references, --out on a development folder,
     a missing kappa selection, a development scenario.
  7. The development reports (stats_tune/, stats_test/) are not touched.

    python tools/test_prediction_confirm.py
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

from tools import run_prediction_plan as P                                # noqa: E402
from tools import run_prediction_stats as S                               # noqa: E402
from tools import test_prediction_stats_part as T                         # noqa: E402
from tools.run_param_grid import split_ids                                # noqa: E402

FAILED = []
B_TEST = 200
SCEN = tuple(S.CONFIRM_BLOCK)
PLANTED = 3
C_METHODS = ('WSPI', 'WSPI-S', 'WSPI-FS3', 'SMA8', 'SMA16', 'SMA32', 'SMA64', 'EWMA8',
             'EWMA16', 'EWMA32', 'EWMA64', 'GBM', 'GBM+W')


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def digest(folder: Path) -> dict:
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file()}


def build(root: Path):
    """Development root with a kappa selection and two report folders, and the
    confirmation folders confirm/run_A, run_C, run_D for every held-out cell."""
    (root / 'stats_tune').mkdir(parents=True)
    (root / 'stats_tune' / 'kappa_selection.json').write_text(json.dumps({'selected_kappa': 3.0}))
    (root / 'stats_test').mkdir()
    (root / 'stats_test' / 'all_paired_tests.csv').write_text('x\n1\n')
    croot = root / 'confirm'
    k = 0
    for i, s in enumerate(SCEN):
        for h in S.HORIZONS:
            rng = np.random.default_rng([300 + i, h])
            base = {c: rng.normal(0.8 if c != 'robustness_distortion' else 20.0, 0.02, T.N_WIN)
                    for c in T.COLUMNS}
            a = {m: T.frame(base, m, rng) for m in S.RUNS['A']['methods']}
            a['MA64'] = T.frame(base, 'MA64', rng, {'ndcg@10': 0.01})
            T.write(croot / 'run_A' / s / f'h{h}', a)
            c = {}
            for m in C_METHODS:
                if m == 'WSPI':
                    c[m] = a['WSPI'].copy()
                elif m == 'SMA64':
                    c[m] = a['MA64'].copy()
                    c[m]['method'] = 'SMA64'
                else:
                    c[m] = T.frame(base, m, rng)
            T.write(croot / 'run_C' / s / f'h{h}', c)
            _, test_ids = split_ids(a['WSPI']['window_id'].tolist())
            is_test = a['WSPI']['window_id'].isin(set(test_ids)).to_numpy()
            d = {'WSPI': a['WSPI'].copy()}
            for m in ('WSPI-M', 'WSPI-H'):
                x = a['WSPI-F2'].copy()
                x['method'] = m
                for col in S.MAIN_METRICS:
                    x[col] = x[col] + rng.normal(0, 0.001, len(x))
                if m == 'WSPI-H' and k < PLANTED:
                    x.loc[is_test, 'rsi@10'] = x.loc[is_test, 'rsi@10'] + 0.08
                d[m] = x
            T.write(croot / 'run_D' / s / f'h{h}', d)
            for run in ('run_C', 'run_D'):
                shutil.copyfile(croot / 'run_A' / s / f'h{h}' / 'protocol' / 'WSPI_protocol.csv',
                                croot / run / s / f'h{h}' / 'protocol' / 'WSPI_protocol.csv')
            k += 1


def plan_tests():
    n = len(P.CONFIRM_SCENARIOS) * len(P.HORIZONS)
    for plan in ('A', 'C', 'D'):
        jobs = P.build_jobs(plan, P.CONFIRM_ROOT, confirm=True)
        ok = len(jobs) == n and all(j['out'].startswith(f'{P.CONFIRM_ROOT}/run_{plan}/')
                                    for j in jobs)
        for j in jobs:
            data, mo = P.CONFIRM_SCENARIOS[j['scenario']]
            c = j['cmd']
            ok &= c[c.index('--data') + 1] == data and c[c.index('--min-obs') + 1] == str(mo)
        check(f'1 plan {plan}: {n} jobs on the held-out scenarios', ok)
    cmd = P.build_jobs('C', P.CONFIRM_ROOT, confirm=True)[0]['cmd']
    meth = cmd[cmd.index('--methods') + 1:]
    check('1 plan C: WSPI-FS3 only among the gates', tuple(meth) == C_METHODS, ' '.join(meth))
    cmd = P.build_jobs('D', P.CONFIRM_ROOT, confirm=True)[0]['cmd']
    check('1 plan D: WSPI, WSPI-M, WSPI-H with the bound check',
          cmd[cmd.index('--methods') + 1:] == ['WSPI', 'WSPI-M', 'WSPI-H'] and '--bound-check' in cmd)
    cmd = P.build_jobs('A', P.CONFIRM_ROOT, confirm=True)[0]['cmd']
    check('1 plan A: all methods, gate profile and bound check',
          '--methods' not in cmd and '--gate-profile' in cmd and '--bound-check' in cmd)
    check('1 wikipedia: threshold 50, taxi 2024: 24',
          P.CONFIRM_SCENARIOS['wikipedia_daily'][1] == 50
          and all(P.CONFIRM_SCENARIOS[s][1] == 24 for s in SCEN if s.startswith('taxi')))
    dev = P.build_jobs('A', 'results/prediction')
    check('1 development plan A unchanged', len(dev) == 18
          and {j['scenario'] for j in dev} == set(P.SCENARIOS))
    for plan in ('B', 'C_sens', 'C_nlms'):
        try:
            P.build_jobs(plan, P.CONFIRM_ROOT, confirm=True)
            check(f'1 refused: --confirm with plan {plan}', False)
        except ValueError:
            check(f'1 refused: --confirm with plan {plan}', True)
    for argv in (['--confirm', '--plan', 'B', '--dry-run'],
                 ['--confirm', '--plan', 'A', '--scenarios', 'taxi_5min', '--dry-run'],
                 ['--plan', 'A', '--scenarios', 'wikipedia_daily', '--dry-run']):
        try:
            P.main(argv)
            check('1 refused: ' + ' '.join(argv), False)
        except SystemExit as e:
            check('1 refused: ' + ' '.join(argv), e.code == 2)


def cell_order():
    return {(s, h): i * 3 + k for i, s in enumerate(SCEN) for k, h in enumerate(S.HORIZONS)}


def stats_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'p2'
        build(root)
        before = {p: digest(root / p) for p in ('stats_tune', 'stats_test')}
        n_cells = len(SCEN) * len(S.HORIZONS)
        refs = list(S.CONFIRM_REFERENCES)
        for part in ('tune', 'test'):
            rc = S.main(['--root', str(root), '--confirm', '--part', part, '--B', str(B_TEST)])
            out = root / 'confirm' / f'stats_{part}'
            check(f'2 {part}: exit code 1 only because B is not 10000', rc == 1)
            p = pd.read_csv(out / 'all_paired_tests.csv')
            check(f'2 {part}: references', sorted(p['reference'].unique()) == sorted(refs))
            n_rep = p.groupby(['scenario', 'horizon', 'reference']).ngroups
            check(f'2 {part}: one report per cell and reference', n_rep == n_cells * len(refs),
                  str(n_rep))
            family = set(S.RUNS['A']['methods']) | set(S.TEST_FROM_C) | {'WSPI-FS3'} \
                | set(S.LEVEL_NAMES)
            if part == 'tune':
                family -= {'GBM', 'GBM+W'}
            got = set(p[p.reference == 'WSPI-H']['method']) | {'WSPI-H'}
            check(f'2 {part}: method family ({len(family)} methods)',
                  got == family and len(family) == (21 if part == 'tune' else 23),
                  str(sorted(got ^ family)))
            plan = json.loads((out / 'stats_plan.json').read_text())
            ok_b = all(r['block'] == max(S.CONFIRM_BLOCK[r['scenario']], r['horizon'])
                       for r in plan['reports'])
            check(f'2 {part}: block max(b, h)', ok_b)
            check(f'2 {part}: plan', plan.get('references') == refs and plan.get('selected_kappa') == 3.0
                  and not plan['window_check_failed'] and len(plan['reports']) == n_cells * len(refs)
                  and plan['part'] == part)
            wc = pd.read_csv(out / 'window_check.csv')
            n_rows = len(S.RUNS['A']['methods']) + len(C_METHODS) + 3 + 3
            check(f'2 {part}: window check OK', len(wc) == n_rows * n_cells
                  and (wc['status'] == 'OK').all(), str(len(wc)))

            pw = pd.read_csv(out / 'part_windows.csv')
            ms = pd.read_csv(out / 'all_method_summary.csv')
            ok_ids, ok_mean = True, True
            for s in SCEN:
                for h in S.HORIZONS:
                    fa = root / 'confirm' / 'run_A' / s / f'h{h}' / 'protocol'
                    tune_ids, test_ids = split_ids(
                        pd.read_csv(fa / 'WSPI_protocol.csv')['window_id'].tolist())
                    keep = tune_ids if part == 'tune' else test_ids
                    r = pw[(pw.scenario == s) & (pw.horizon == h)].iloc[0]
                    col = 'n_tune' if part == 'tune' else 'n_test'
                    ok_ids &= int(r[col]) == len(keep)
                    for m, f in (('WSPI-H', root / 'confirm' / 'run_D' / s / f'h{h}' / 'protocol'),
                                 ('MA64', fa)):
                        d = pd.read_csv(f / f'{m}_protocol.csv')
                        d = d[d['window_id'].isin(set(keep))]
                        for c in S.MAIN_METRICS:
                            row = ms[(ms.scenario == s) & (ms.horizon == h) & (ms.method == m)
                                     & (ms.metric == c) & (ms.reference_report == 'WSPI-H')]
                            ok_mean &= len(row) == 1 and int(row['n_windows'].iloc[0]) == len(d) \
                                and np.isclose(row['mean'].iloc[0], d[c].mean(), rtol=0, atol=1e-12)
            check(f'3 {part}: only the windows of the part', bool(ok_ids))
            check(f'3 {part}: means equal the means of those windows', bool(ok_mean))

            d = p[(p.reference == 'WSPI-H') & (p.method == 'MA64') & (p.metric == 'rsi@10')].copy()
            order = cell_order()
            d['cell'] = [order[(s, int(h))] for s, h in zip(d.scenario, d.horizon)]
            better = sorted(d[d.verdict == 'ref_better']['cell'])
            want = list(range(PLANTED)) if part == 'test' else []
            check(f'4 {part}: planted effect only where planted', better == want, str(better))

        # 5. a differing WSPI file, a missing WSPI-FS3
        bad = Path(td) / 'bad'
        shutil.copytree(root, bad, ignore=shutil.ignore_patterns('stats_*'))
        (bad / 'stats_tune').mkdir()
        shutil.copyfile(root / 'stats_tune' / 'kappa_selection.json',
                        bad / 'stats_tune' / 'kappa_selection.json')
        f = bad / 'confirm' / 'run_C' / SCEN[1] / 'h8' / 'protocol' / 'WSPI_protocol.csv'
        w = pd.read_csv(f)
        w.loc[3, 'rsi@10'] = w.loc[3, 'rsi@10'] + 0.01
        w.to_csv(f, index=False)
        (bad / 'confirm' / 'run_C' / SCEN[2] / 'h1' / 'protocol' / 'WSPI-FS3_protocol.csv').unlink()
        rc = S.main(['--root', str(bad), '--confirm', '--part', 'test', '--B', str(B_TEST)])
        pb = json.loads((bad / 'confirm' / 'stats_test' / 'stats_plan.json').read_text())
        check('5 differing WSPI file and missing WSPI-FS3 stop those cells',
              rc == 1 and sorted(pb['window_check_failed']) == sorted([f'{SCEN[1]}/h8', f'{SCEN[2]}/h1'])
              and len(pb['reports']) == (n_cells - 2) * len(refs), str(pb['window_check_failed']))

        # 6. refusals
        for name, argv in (
                ('--part all', ['--confirm']),
                ('--runs D', ['--confirm', '--part', 'test', '--runs', 'D']),
                ('--references', ['--confirm', '--part', 'test', '--references', 'WSPI']),
                ('--out stats_test', ['--confirm', '--part', 'test', '--out', str(root / 'stats_test')]),
                ('--out stats_test_D', ['--confirm', '--part', 'test',
                                        '--out', str(root / 'stats_test_D')])):
            rc = S.main(['--root', str(root), '--B', str(B_TEST)] + argv)
            check(f'6 refused: {name}', rc == 2)
        nok = Path(td) / 'nokappa'
        (nok / 'confirm').mkdir(parents=True)
        rc = S.main(['--root', str(nok), '--confirm', '--part', 'tune', '--B', str(B_TEST)])
        check('6 refused: no kappa selection', rc == 2)
        for argv in (['--confirm', '--part', 'test', '--scenarios', 'taxi_5min'],
                     ['--part', 'test', '--runs', 'D', '--scenarios', 'wikipedia_daily']):
            try:
                S.main(['--root', str(root), '--B', str(B_TEST)] + argv)
                check('6 refused: ' + ' '.join(argv[-2:]), False)
            except SystemExit as e:
                check('6 refused: ' + ' '.join(argv[-2:]), e.code == 2)
        for p_ in ('stats_tune', 'stats_test'):
            check(f'7 {p_} unchanged', digest(root / p_) == before[p_])


def main():
    reg = dict(S.CONFIRM_BLOCK)
    for s in S.CONFIRM_BLOCK:
        S.CONFIRM_BLOCK[s] = 4                    # short blocks for the small test series
    try:
        plan_tests()
        stats_tests()
    finally:
        S.CONFIRM_BLOCK.update(reg)
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
