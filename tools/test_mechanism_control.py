r"""
Unit checks for the mechanism control WSPI-HL (run E)
====================================================
WSPI-HL is WSPI-H with alpha = beta = 0: the score is the clamped window-mean
level mu_hat without the structural factor exp(alpha R - beta W_E).  Covers the
additions to ``methods/wspi_forecast.py``, ``tools/prediction_protocol.py``,
``tools/run_prediction_plan.py`` and ``tools/run_prediction_stats.py``.

  1. Closed form: P_HL = (1 - s_m) mu_L + s_m mu_L clip(a_bar / mu_L, 1/c, c),
     and P_H = P_HL * exp(R - W_E) for the same window.
  2. Bound: P_HL / mu_L lies in [1 - s_m (1 - 1/c), 1 + s_m (c - 1)] for
     m = 1..6; both limits are attained.
  3. Batch scorer == reference class on full 64-slot windows, h = 1, 8, 24;
     the WSPI-H scorer times exp(R - W_E) of the window equals WSPI-H.
  4. Wiring: method lists, plan E (18 jobs, 12 with --confirm), no bound
     recorder for WSPI-HL (its bound is relative to mu_L, not to WSPI).
  5. Stats --runs E on synthetic folders, development and confirmation data:
     one report per cell, reference WSPI-HL, family WSPI-H and MA64, only the
     test windows, a planted effect found exactly where planted, a run E WSPI-H
     file that differs from run D stops that cell, refusals, earlier report
     folders untouched.

    python tools/test_mechanism_control.py
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

from methods.wspi_forecast import (CLAMP, get_level_methods,                  # noqa: E402
                                   get_mechanism_methods, level_anchor_bounds,
                                   mean_share, n_forecast_coefficients, recency_mean)
from tools import prediction_protocol as P                                # noqa: E402
from tools import run_prediction as RP                                    # noqa: E402
from tools import run_prediction_plan as PL                               # noqa: E402
from tools import run_prediction_stats as S                               # noqa: E402
from tools import test_prediction_stats_part as T                         # noqa: E402
from tools.run_param_grid import split_ids                                # noqa: E402
from tools.test_level_methods import random_rows                          # noqa: E402
from tools.test_prediction import signals                                 # noqa: E402

HORIZONS = (1, 8, 24)
FAILED = []
B_TEST = 200
PLANTED = 4                      # first cells (scenario-major) with a planted effect


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def scorer(name, h):
    return P.build_prediction_methods(horizon=h, names=[name])[name].scorer


# =============================================================================
# 1 and 2. Closed form and bound of the reference class
# =============================================================================

def test_class(rng):
    for h in HORIZONS:
        m = n_forecast_coefficients(h)
        hl = get_mechanism_methods(horizon=h)['WSPI-HL']
        wh = get_level_methods(horizon=h)['WSPI-H']
        worst_cf, worst_h = 0.0, 0.0
        for x in signals():
            a, _, pyr = hl.forecast_lowpass(x)
            R, WE = hl._structure(a, pyr)
            mu_L = recency_mean(a)
            if mu_L <= 1e-12:
                check(f'zero level gives 0  h={h}', hl.assess_single(x) == 0.0)
                continue
            s = mean_share(len(a), m)
            want = (1 - s) * mu_L + s * mu_L * np.clip(np.mean(a) / mu_L, 1 / CLAMP, CLAMP)
            got = hl.assess_single(x)
            worst_cf = max(worst_cf, abs(got / want - 1.0))
            worst_h = max(worst_h, abs(got * np.exp(R - WE) / wh.assess_single(x) - 1.0))
        check(f'WSPI-HL == closed form  h={h:<2}', worst_cf < 1e-12, f'(max rel. diff {worst_cf:.1e})')
        check(f'WSPI-HL * exp(R - W_E) == WSPI-H  h={h:<2}', worst_h < 1e-12,
              f'(max rel. diff {worst_h:.1e})')
    worst, hit_lo, hit_hi = 0.0, 0, 0
    for m_h in (1, 5, 9, 13, 17, 21):                      # m = 1..6
        m = n_forecast_coefficients(m_h)
        lo, hi = level_anchor_bounds(16, m, CLAMP)
        hl = get_mechanism_methods(horizon=m_h)['WSPI-HL']
        for x in random_rows(rng):
            a, _, _ = hl.forecast_lowpass(x)
            mu_L = recency_mean(a)
            if mu_L <= 1e-12:
                continue
            r = hl.assess_single(x) / mu_L
            worst = max(worst, lo / r - 1.0, r / hi - 1.0)
            hit_lo += int(abs(r / lo - 1.0) < 1e-12)
            hit_hi += int(abs(r / hi - 1.0) < 1e-12)
    check('P_HL / mu_L inside [1 - s_m (1 - 1/c), 1 + s_m (c - 1)], m = 1..6', worst <= 1e-9,
          f'(worst excess {worst:.1e})')
    check('both limits attained', hit_lo > 0 and hit_hi > 0, f'(lower {hit_lo}, upper {hit_hi})')
    hl = get_mechanism_methods(horizon=8)['WSPI-HL']
    check('class settings: mean, level anchor, c = 4, alpha = beta = 0',
          hl.predictor == 'mean' and hl.clamp_anchor == 'level' and hl.clamp == CLAMP
          and hl.alpha == 0.0 and hl.beta == 0.0 and hl.name == 'WSPI-HL')


# =============================================================================
# 3. Batch scorer
# =============================================================================

def test_batch():
    X = signals()
    for h in HORIZONS:
        obj = get_mechanism_methods(horizon=h)['WSPI-HL']
        fast = scorer('WSPI-HL', h)(X)
        slow = np.array([obj.assess_single(x) for x in X])
        rel = np.abs(fast - slow) / np.maximum(np.abs(slow), 1e-9)
        check(f'scorer == class  h={h:<2} WSPI-HL', rel.max() < 1e-10, f'max rel. diff {rel.max():.2e}')
        tr = P.dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
        F = P.CoefFeatures(P._pad_left_reflect(np.asarray(X, dtype=np.float64), 64), tr, 3)
        h_fast = scorer('WSPI-H', h)(X)
        check(f'WSPI-HL scorer * exp(R - W_E) == WSPI-H scorer  h={h:<2}',
              np.allclose(fast * np.exp(F.R - F.WE), h_fast, rtol=1e-12, atol=0))
        check(f'scorer name  h={h:<2}', scorer('WSPI-HL', h).__name__ == f'wspi_hl_c4_h{h}_W64')


# =============================================================================
# 4. Wiring
# =============================================================================

def test_wiring():
    check('method lists', P.MECHANISM_METHODS == ('WSPI-HL',) and
          P.RUN_E_METHODS == ('WSPI', 'WSPI-H', 'WSPI-HL') and
          PL.RUN_E_METHODS == P.RUN_E_METHODS and S.SUPP_RUNS['E']['methods'] == P.RUN_E_METHODS)
    check('earlier method lists unchanged', P.LEVEL_METHODS == ('WSPI-M', 'WSPI-H') and
          P.RUN_D_METHODS == ('WSPI', 'WSPI-M', 'WSPI-H') and PL.RUN_D_METHODS == P.RUN_D_METHODS)
    jobs = PL.build_jobs('E', 'results/prediction', python='python')
    ok = len(jobs) == 18 and all(j['cmd'][j['cmd'].index('--methods') + 1:] == list(P.RUN_E_METHODS)
                                 and '--bound-check' in j['cmd'] and '--resume' in j['cmd']
                                 and j['out'].startswith('results/prediction/run_E/') for j in jobs)
    check('plan E: 18 jobs, methods, bound check, resume, output run_E', ok, f'({len(jobs)} jobs)')
    cj = PL.build_jobs('E', 'results/prediction/confirm', python='python', confirm=True)
    check('plan E --confirm: 12 jobs on the confirmation scenarios',
          len(cj) == 12 and {j['scenario'] for j in cj} == set(PL.CONFIRM_SCENARIOS)
          and all(j['out'].startswith('results/prediction/confirm/run_E/') for j in cj))
    check('plan D unchanged', [j['cmd'][j['cmd'].index('--methods') + 1:] for j in
                               PL.build_jobs('D', 'r', python='python')][0] == list(P.RUN_D_METHODS))
    check('no clamp entry for the bound check of WSPI-HL (its bound is relative to mu_L)',
          P.control_clamp('WSPI-HL') is None and P.control_bounds('WSPI-HL', 8) is None)
    meths = P.build_prediction_methods(horizon=8, names=list(P.RUN_E_METHODS))
    meths, rec = RP.add_bound_recorders(meths, 8)
    meths, rec = RP.add_control_bound_recorders(meths, rec, 8)
    check('bound recorder for WSPI-H only', sorted(rec) == ['WSPI-H'], str(sorted(rec)))


# =============================================================================
# 5. Stats --runs E on synthetic folders
# =============================================================================

def digest(folder: Path) -> dict:
    if not folder.exists():
        return {}
    return {f.relative_to(folder).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(folder.rglob('*')) if f.is_file()}


def build(base_root: Path, scenarios, seed0: int):
    """Run A, D and E folders.  Run E: WSPI copied from run A, WSPI-H copied from
    run D, WSPI-HL from WSPI-H with small noise; in the first PLANTED cells
    WSPI-HL gets a higher Delta Rank on the test windows only."""
    k = 0
    for i, s in enumerate(scenarios):
        for h in S.HORIZONS:
            rng = np.random.default_rng([seed0 + i, h])
            base = {c: rng.normal(0.8 if c != 'robustness_distortion' else 20.0, 0.02, T.N_WIN)
                    for c in T.COLUMNS}
            a = {m: T.frame(base, m, rng) for m in S.RUNS['A']['methods']}
            T.write(base_root / 'run_A' / s / f'h{h}', a)
            d = {'WSPI': a['WSPI'].copy(), 'WSPI-M': T.frame(base, 'WSPI-M', rng),
                 'WSPI-H': T.frame(base, 'WSPI-H', rng)}
            T.write(base_root / 'run_D' / s / f'h{h}', d)
            _, test_ids = split_ids(a['WSPI']['window_id'].tolist())
            hl = d['WSPI-H'].copy()
            hl['method'] = 'WSPI-HL'
            for c in S.MAIN_METRICS:
                hl[c] = hl[c] + rng.normal(0, 0.001, len(hl))
            if k < PLANTED:
                is_test = hl['window_id'].isin(set(test_ids)).to_numpy()
                hl.loc[is_test, 'robustness_distortion'] += 2.0
            T.write(base_root / 'run_E' / s / f'h{h}', {'WSPI': a['WSPI'].copy(),
                                                        'WSPI-H': d['WSPI-H'].copy(),
                                                        'WSPI-HL': hl})
            k += 1


def check_reports(root: Path, base_root: Path, scenarios, label: str, extra=()):
    rc = S.main(['--root', str(root), *extra, '--part', 'test', '--runs', 'E', '--B', str(B_TEST)])
    out = base_root / 'stats_test_E'
    n_cells = len(scenarios) * len(S.HORIZONS)
    check(f'5 {label}: exit code 1 only because B is not 10000', rc == 1)
    p = pd.read_csv(out / 'all_paired_tests.csv')
    check(f'5 {label}: reference WSPI-HL, family WSPI-H and MA64',
          set(p['reference']) == {'WSPI-HL'} and set(p['method']) == set(S.E_FAMILY))
    check(f'5 {label}: one report per cell',
          p.groupby(['scenario', 'horizon']).ngroups == n_cells and
          len(json.loads((out / 'stats_plan.json').read_text())['reports']) == n_cells)
    wc = pd.read_csv(out / 'window_check.csv')
    n_rows = len(S.RUNS['A']['methods']) + 3 + 3 + 2
    check(f'5 {label}: window check OK (A, D, E, A=E, D=E per cell)',
          len(wc) == n_rows * n_cells and (wc['status'] == 'OK').all(), str(len(wc)))
    ms = pd.read_csv(out / 'all_method_summary.csv')
    ok = True
    for s in scenarios:
        for h in S.HORIZONS:
            fa = base_root / 'run_A' / s / f'h{h}' / 'protocol'
            _, test_ids = split_ids(pd.read_csv(fa / 'WSPI_protocol.csv')['window_id'].tolist())
            for m, f in (('WSPI-HL', base_root / 'run_E' / s / f'h{h}' / 'protocol'),
                         ('WSPI-H', base_root / 'run_D' / s / f'h{h}' / 'protocol'), ('MA64', fa)):
                dd = pd.read_csv(f / f'{m}_protocol.csv')
                dd = dd[dd['window_id'].isin(set(test_ids))]
                for c in S.MAIN_METRICS:
                    row = ms[(ms.scenario == s) & (ms.horizon == h) & (ms.method == m) & (ms.metric == c)]
                    ok &= len(row) == 1 and int(row['n_windows'].iloc[0]) == len(dd) and \
                        np.isclose(row['mean'].iloc[0], dd[c].mean(), rtol=0, atol=1e-12)
    check(f'5 {label}: only the test windows; means equal the protocol files', bool(ok))
    order = {(s, h): i * 3 + j for i, s in enumerate(scenarios) for j, h in enumerate(S.HORIZONS)}
    d = p[(p.method == 'WSPI-H') & (p.metric == 'robustness_distortion')]
    worse = sorted(order[(s, int(h))] for s, h, v in zip(d.scenario, d.horizon, d.verdict)
                   if v == 'ref_worse')
    check(f'5 {label}: planted effect (WSPI-HL higher Delta Rank) only where planted',
          worse == list(range(PLANTED)), str(worse))
    return out


def test_stats():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / 'p2'
        build(root, tuple(S.BASE_BLOCK), 300)
        build(root / S.CONFIRM_DIR, tuple(S.CONFIRM_BLOCK), 400)
        (root / 'stats_test_D').mkdir(parents=True)
        (root / 'stats_test_D' / 'keep.csv').write_text('x\n1\n')
        before = digest(root / 'stats_test_D')
        check_reports(root, root, tuple(S.BASE_BLOCK), 'development')
        check_reports(root, root / S.CONFIRM_DIR, tuple(S.CONFIRM_BLOCK), 'confirmation',
                      extra=('--confirm',))
        check('5 earlier report folder untouched', digest(root / 'stats_test_D') == before)

        bad = Path(td) / 'bad'
        shutil.copytree(root, bad, ignore=shutil.ignore_patterns('stats_test_E'))
        s1 = tuple(S.BASE_BLOCK)[1]
        f = bad / 'run_E' / s1 / 'h24' / 'protocol' / 'WSPI-H_protocol.csv'
        w = pd.read_csv(f)
        w.loc[3, 'rsi@10'] += 0.01
        w.to_csv(f, index=False)
        rc = S.main(['--root', str(bad), '--part', 'test', '--runs', 'E', '--B', str(B_TEST)])
        pb = json.loads((bad / 'stats_test_E' / 'stats_plan.json').read_text())
        check('5 run E WSPI-H differing from run D stops that cell',
              rc == 1 and pb['window_check_failed'] == [f'{s1}/h24']
              and len(pb['reports']) == len(S.BASE_BLOCK) * 3 - 1, str(pb['window_check_failed']))

        for args, what in ((['--part', 'tune'], '--part tune'),
                           (['--part', 'test', '--references', 'WSPI', '--out', str(root / 'x')],
                            '--references'),
                           (['--part', 'test', '--out', str(root / 'stats_test_D')], '--out stats_test_D'),
                           (['--part', 'test', '--out', str(root / 'stats_test')], '--out stats_test'),
                           (['--confirm', '--part', 'test', '--out',
                             str(root / S.CONFIRM_DIR / 'stats_test')], '--confirm --out confirm/stats_test'),
                           (['--confirm', '--part', 'tune'], '--confirm --part tune'),
                           (['--part', 'all'], '--part all')):
            rc = S.main(['--root', str(root), *args, '--runs', 'E', '--B', str(B_TEST)])
            check(f'5 refused with {what}', rc == 2)
        check('5 earlier report folder untouched after the refusals',
              digest(root / 'stats_test_D') == before)


def main():
    reg, creg = dict(S.BASE_BLOCK), dict(S.CONFIRM_BLOCK)
    for s in S.BASE_BLOCK:
        S.BASE_BLOCK[s] = 4                       # short blocks for the small test series
    for s in S.CONFIRM_BLOCK:
        S.CONFIRM_BLOCK[s] = 4
    try:
        test_class(np.random.default_rng(11))
        test_batch()
        test_wiring()
        test_stats()
    finally:
        S.BASE_BLOCK.update(reg)
        S.CONFIRM_BLOCK.update(creg)
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
