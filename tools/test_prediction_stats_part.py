r"""
Unit checks for the tuning and test parts of the paired tests (--part)
=====================================================================
Covers ``--part tune`` and ``--part test`` of ``tools/run_prediction_stats.py``
on synthetic run folders (run A, run C, run C_sens) with known differences.

  1. Split and copy: the part folders hold exactly the windows of split_ids
     (first 30 % tuning) of the run A WSPI windows, with the lines unchanged.
  2. Tuning reports: one per kappa, WSPI-FS<kappa> against the ten run A
     methods; the kappa rule (most Pareto cells against WSPI-F2; a tie goes to
     the larger kappa) on constructed cases.
  3. Test reports: need the selection; five references; the 21 methods
     (SMA64 left out); every report has only test windows.  The sensitivity
     summary equals the means of the test windows.
  4. Checks: a run C WSPI file that differs from run A, or an SMA64 file that
     differs from MA64, or a C_sens default that differs from WSPI-F2, stops
     the tests of that cell; --part all refuses run C; the plan of runs A and B
     is not changed by the new option.

    python tools/test_prediction_stats_part.py
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

from evaluation.fast_evaluator import PROTOCOL_COLUMNS                    # noqa: E402
from evaluation.protocol_v5 import V5_EXTRA_COLUMNS                       # noqa: E402
from tools import run_prediction_stats as S                               # noqa: E402
from tools.run_param_grid import split_ids                                # noqa: E402

FAILED = []
N_WIN = 960
B_TEST = 200
COLUMNS = list(PROTOCOL_COLUMNS) + list(V5_EXTRA_COLUMNS)
SCEN = tuple(S.BASE_BLOCK)
# cells (index 0..17 = scenario-major) where WSPI-FS<kappa> beats WSPI-F2 on RSI
N_GOOD = {'WSPI-FS1': 5, 'WSPI-FS1.5': 6, 'WSPI-FS2': 8, 'WSPI-FS3': 8}


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def frame(base, m, rng, off=None):
    df = pd.DataFrame({c: base[c].copy() for c in COLUMNS})
    df['window_id'] = np.arange(64, 64 + N_WIN)
    df['timestamp'] = df['window_id'] * 3600000
    df['method'] = m
    df['num_items'] = 50
    df['ties_top21'] = False
    df['padded'] = False
    df['n_nonfinite'] = 0
    for c, v in (off or {}).items():
        df[c] = df[c] + v + rng.normal(0, 0.001, N_WIN)
    return df[COLUMNS]


def write(folder: Path, frames):
    (folder / 'protocol').mkdir(parents=True, exist_ok=True)
    (folder / 'metadata').mkdir(exist_ok=True)
    for m, df in frames.items():
        df.to_csv(folder / 'protocol' / f'{m}_protocol.csv', index=False)
    (folder / 'metadata' / 'protocol_v5_run.json').write_text('{}')


def build(root: Path):
    for i, s in enumerate(SCEN):
        for j, h in enumerate(S.HORIZONS):
            cell = i * 3 + j
            rng = np.random.default_rng([i, h])
            base = {c: rng.normal(0.8 if c != 'robustness_distortion' else 20.0, 0.02, N_WIN)
                    for c in COLUMNS}
            a = {m: frame(base, m, rng) for m in S.RUNS['A']['methods']}
            a['MA64'] = frame(base, 'MA64', rng, {'ndcg@10': 0.01})
            write(root / 'run_A' / s / f'h{h}', a)
            c = {}
            for m in S.SUPP_RUNS['C']['methods']:
                if m == 'WSPI':
                    c[m] = a['WSPI'].copy()
                elif m == 'SMA64':
                    c[m] = a['MA64'].copy()
                    c[m]['method'] = 'SMA64'
                elif m in N_GOOD:
                    c[m] = frame(base, m, rng, {'rsi@10': 0.05} if cell < N_GOOD[m] else None)
                elif m == 'GBM':
                    c[m] = frame(base, m, rng, {'spearman_rho': 0.03})
                else:
                    c[m] = frame(base, m, rng)
            write(root / 'run_C' / s / f'h{h}', c)
            sens = {'WSPI': a['WSPI'].copy()}
            for k, m in enumerate(S.SENS_NAMES):
                if m == S.SENS_DEFAULT:
                    sens[m] = a['WSPI-F2'].copy()
                    sens[m]['method'] = m
                else:
                    sens[m] = frame(base, m, rng, {'ndcg@10': 0.001 * k})
            write(root / 'run_C_sens' / s / f'h{h}', sens)


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
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


def run_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'prediction'
        build(root)
        ids = list(range(64, 64 + N_WIN))
        tune, test = split_ids(ids)

        # 1. copy of the part
        src = root / 'run_A' / SCEN[0] / 'h1' / 'protocol' / 'WSPI_protocol.csv'
        part = S.write_part(Path(td) / 'part', {'WSPI': src}, tune)
        lines = (part / 'protocol' / 'WSPI_protocol.csv').read_text().splitlines()
        orig = src.read_text().splitlines()
        check('part copy: header and tuning lines unchanged',
              lines[0] == orig[0] and lines[1:] == orig[1:1 + len(tune)])
        check('split: first 30 % tuning', len(tune) == 288 and tune[-1] + 1 == test[0])

        # test part before tuning: refused
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', '--B', str(B_TEST)])
        check('--part test without a selection is refused', rc == 2)
        rc = S.main(['--root', str(root), '--part', 'all', '--runs', 'C'])
        check('--part all refuses run C', rc == 2)

        # 2. tuning
        S.main(['--root', str(root), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST)])
        tdir = root / 'stats_tune'
        sel = json.loads((tdir / 'kappa_selection.json').read_text())
        tab = pd.read_csv(tdir / 'kappa_selection.csv')
        counts = dict(zip(tab['reference'], tab['n_success']))
        check('tuning: Pareto cells per kappa', counts == N_GOOD, str(counts))
        check('tuning: tie 8 = 8 goes to the larger kappa (3)', sel['selected_kappa'] == 3.0
              and sel['selected_method'] == 'WSPI-FS3')
        pt = pd.read_csv(tdir / 'all_paired_tests.csv')
        n_methods = pt.groupby(['scenario', 'horizon', 'reference'])['method'].nunique()
        check('tuning: 72 reports with the ten run A methods each',
              len(n_methods) == 72 and (n_methods == 10).all())
        ms = pd.read_csv(tdir / 'all_method_summary.csv')
        check('tuning: only tuning windows', (ms['n_windows'] == len(tune)).all())
        meta = json.loads((tdir / SCEN[0] / 'h8' / 'ref_WSPI-FS2' / 'metadata' /
                           'stats_run.json').read_text())
        check('tuning: report metadata', meta['part'] == 'tune' and
              meta['n_part_windows'] == len(tune) and meta['first_common_window'] == tune[0]
              and meta['block'] == 8)
        pw = pd.read_csv(tdir / 'part_windows.csv')
        check('part_windows.csv', len(pw) == 18 and (pw['n_tune'] == len(tune)).all()
              and (pw['first_test'] == test[0]).all())
        # rule on constructed cases
        cells = []
        for kappa, ref in zip(S.FS_KAPPAS, S.FS_NAMES):
            for s in SCEN:
                for h in S.HORIZONS:
                    for c in S.MAIN_METRICS:
                        v = 'n.s.'
                        if ref == 'WSPI-FS1' and c == 'rsi@10':
                            v = 'ref_better'
                        if ref == 'WSPI-FS1' and c == 'ndcg@10' and s == SCEN[0]:
                            v = 'ref_worse'
                        cells.append(dict(scenario=s, horizon=h, reference=ref,
                                          method='WSPI-F2', metric=c, verdict=v))
        _, st, best = S.select_kappa(pd.DataFrame(cells))
        check('rule: one worse metric removes the cell; best kappa 1',
              best == 1.0 and int(st.loc[st['kappa'] == 1.0, 'n_success'].iloc[0]) == 15)
        _, st, best = S.select_kappa(pd.DataFrame(cells)[lambda d: d['reference'] != 'WSPI-FS1']
                                     .assign(verdict='n.s.'))
        check('rule: all zero -> tie -> largest kappa; missing kappa -> no selection',
              best is None)
        allns = pd.DataFrame(cells).assign(verdict='n.s.')
        _, st, best = S.select_kappa(allns)
        check('rule: all zero -> largest kappa (3)', best == 3.0)

        # 3. test part
        rc = S.main(['--root', str(root), '--part', 'test', '--runs', 'C', 'C_sens',
                     '--B', str(B_TEST)])
        edir = root / 'stats_test'
        pe = pd.read_csv(edir / 'all_paired_tests.csv')
        refs = sorted(pe['reference'].unique())
        check('test: five references', refs == sorted(['WSPI-FS3', 'WSPI-F2', 'WSPI', 'GBM',
                                                       'GBM+W']), str(refs))
        nm = pe.groupby(['scenario', 'horizon', 'reference'])['method'].nunique()
        check('test: 90 reports, 20 other methods each (21 in the family)',
              len(nm) == 90 and (nm == 20).all())
        allm = set(pe['method']) | set(pe['reference'])
        check('test: SMA64 and the other kappa left out',
              'SMA64' not in allm and not ({'WSPI-FS1', 'WSPI-FS2', 'WSPI-FS1.5'} & allm))
        me = pd.read_csv(edir / 'all_method_summary.csv')
        check('test: only test windows', (me['n_windows'] == len(test)).all())
        g = pe[(pe['reference'] == 'WSPI') & (pe['method'] == 'GBM') &
               (pe['metric'] == 'spearman_rho')]
        check('test: constructed GBM gain is found', (g['verdict'] == 'ref_worse').all())
        sens = pd.read_csv(edir / 'sensitivity_summary.csv')
        d = pd.read_csv(root / 'run_C_sens' / SCEN[1] / 'h8' / 'protocol' /
                        'WSPI-F2_c8_g0.5_p0.8_protocol.csv')
        mean = d[d['window_id'].isin(test)]['ndcg@10'].mean()
        r = sens[(sens['scenario'] == SCEN[1]) & (sens['horizon'] == 8) &
                 (sens['method'] == 'WSPI-F2_c8_g0.5_p0.8')]
        check('test: sensitivity summary = mean of the test windows',
              len(sens) == 18 * 18 and np.isclose(r['ndcg@10_mean'].iloc[0], mean, rtol=1e-14)
              and int(sens['default'].sum()) == 18)
        wc = pd.read_csv(edir / 'window_check.csv')
        check('test: checks recorded and passed',
              set(['A=C', 'MA64=SMA64', 'A=C_sens', 'F2=C_sens']) <= set(wc['run']) and
              (wc[wc['expected']]['status'] == 'OK').all())
        check('test: exit code 1 only because B is not 10000', rc == 1)

        # 4. failed checks
        root2 = Path(td) / 'p2'
        shutil.copytree(root, root2, ignore=shutil.ignore_patterns('stats_*'))
        f = root2 / 'run_C' / SCEN[2] / 'h24' / 'protocol' / 'WSPI_protocol.csv'
        d = pd.read_csv(f)
        d.loc[5, 'ndcg@10'] += 1e-12
        d.to_csv(f, index=False)
        f = root2 / 'run_C' / SCEN[3] / 'h1' / 'protocol' / 'SMA64_protocol.csv'
        d = pd.read_csv(f)
        d.loc[3, 'rsi@10'] += 0.1
        d.to_csv(f, index=False)
        rc = S.main(['--root', str(root2), '--part', 'tune', '--runs', 'C', '--B', str(B_TEST)])
        plan = json.loads((root2 / 'stats_tune' / 'stats_plan.json').read_text())
        check('check failures: WSPI A != C and SMA64 != MA64 stop their cells',
              sorted(plan['window_check_failed']) == sorted([f'{SCEN[2]}/h24', f'{SCEN[3]}/h1'])
              and rc == 1)
        check('check failures: no selection with missing cells',
              plan['selected_kappa'] is None)
        f = root2 / 'run_C_sens' / SCEN[0] / 'h8' / 'protocol' / f'{S.SENS_DEFAULT}_protocol.csv'
        d = pd.read_csv(f)
        d.loc[0, 'ndcg@10'] += 0.5
        d.to_csv(f, index=False)
        shutil.copytree(root / 'stats_tune', root2 / 'stats_tune', dirs_exist_ok=True)
        S.main(['--root', str(root2), '--part', 'test', '--runs', 'C', 'C_sens',
                '--B', str(B_TEST), '--scenarios', SCEN[0], '--horizons', '8'])
        plan = json.loads((root2 / 'stats_test' / 'stats_plan.json').read_text())
        check('check failures: C_sens default != WSPI-F2 stops the cell',
              plan['window_check_failed'] == [f'{SCEN[0]}/h8'])


if __name__ == '__main__':
    main()
