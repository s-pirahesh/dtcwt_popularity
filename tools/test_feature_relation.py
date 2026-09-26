r"""
Unit test for tools/run_feature_relation.py (revision-srep-v5, task T3.7 / E7)
=============================================================================
Run from the project root:   python tools\test_feature_relation.py
Ends with "ALL OK" when every check passes.  Uses synthetic data only (no file of
data/ is needed) and writes only to a temporary folder.
"""
import json
import shutil
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import run_feature_relation as fr  # noqa: E402
from evaluation.fast_evaluator import FastMethod  # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator, make_v5_wspi  # noqa: E402
from run_param_grid import WSPIFeatureCache  # noqa: E402

FAILS = []


def check(name, ok, info=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {info}")
    if not ok:
        FAILS.append(name)


def synthetic(n_items=60, n_slots=150, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp('2025-01-01')
    for i in range(n_items):
        lam = rng.gamma(1.0, 3.0)
        p_obs = rng.uniform(0.5, 1.0)
        for t in range(n_slots):
            if rng.random() < p_obs:
                rows.append((t0 + pd.Timedelta(hours=t), f'i{i:03d}', int(rng.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=['timestamp', 'item_id', 'count'])


def rank_ref(x):
    """Average ranks by brute force (reference for fr.ranks)."""
    x = np.asarray(x, dtype=float)
    return np.array([(x < v).sum() + ((x == v).sum() + 1) / 2 for v in x])


def main():
    rng = np.random.default_rng(1)
    X = rng.poisson(4.0, size=(40, 64)).astype(float)
    X[:5, 50] += 60.0
    X[5] = 0.0                                   # one all-zero series
    Xs = X[:, 20:]                               # 44 slots -> reflect padding to 64

    # 1. features bit-identical to the WSPI feature code (with and without padding)
    fm = fr.FeatureMaker()
    for nm, M in (('64 slots', X), ('44 slots, padded', Xs)):
        f = fm(M)
        mu, R, WE = WSPIFeatureCache(use_cache=False).features(M)
        check(f'1 mu, R, WE == WSPIFeatureCache ({nm})',
              np.array_equal(f['mu'], mu) and np.array_equal(f['R'], R) and np.array_equal(f['WE'], WE))
        s = make_v5_wspi(64)(M)
        check(f'1 mu exp(R - WE) == make_v5_wspi ({nm})',
              np.array_equal(f['mu'] * np.exp(f['R'] - f['WE']), s))

    # 2. identity WE log2(J+1) = h(R) + (1-R) H(q), and D in [0, 1]
    f = fm(X)
    v = f['e_tot'] > 0
    res = np.abs(fr.identity_residual(f))[v]
    check('2 identity holds', res.max() < 1e-12, f'max {res.max():.1e}')
    check('2 zero series: e_tot 0, R = WE = D = 0',
          f['e_tot'][5] == 0 and f['R'][5] == 0 and f['WE'][5] == 0 and f['D'][5] == 0)
    check('2 D in [0, 1]', f['D'].min() >= 0 and f['D'].max() <= 1 + 1e-12)
    # by hand: energies (4, 2, 1, 1) -> R = 0.5, q = (1/2, 1/4, 1/4)
    E = np.array([[4.0, 2.0, 1.0, 1.0]])
    WE_h = fr._entropy_bits(E)[0] / 2
    Hq_h = fr._entropy_bits(E[:, 1:])[0]
    check('2 by hand', abs(Hq_h - 1.5) < 1e-12 and
          abs(WE_h * 2 - (fr.binary_entropy(np.array([0.5]))[0] + 0.5 * 1.5)) < 1e-12)
    check('2 binary entropy ends', fr.binary_entropy(np.array([0.0, 1.0, 0.5])).tolist() == [0.0, 0.0, 1.0])

    # 3. rank statistics
    x = np.array([3.0, 1.0, 2.0, 2.0, 5.0, 1.0])
    check('3 ranks with ties', np.array_equal(fr.ranks(x), rank_ref(x)))
    try:
        from scipy.stats import rankdata, spearmanr
        z = rng.normal(size=200)
        w = z + rng.normal(size=200)
        check('3 ranks == scipy rankdata', np.array_equal(fr.ranks(z), rankdata(z)))
        check('3 Spearman == scipy', abs(fr.corr(fr.ranks(z), fr.ranks(w)) - spearmanr(z, w)[0]) < 1e-12)
    except (ImportError, RuntimeError):
        print('     (scipy not available: comparison with scipy skipped)')
    a = rng.normal(size=500)
    c = rng.normal(size=500)
    b = a + c
    y = c + 0.01 * rng.normal(size=500)
    check('3 partial corr removes the control', abs(fr.partial_corr(b, y, [a]) - 1) < 1e-3
          and fr.corr(b, y) < 0.8)
    check('3 partial corr of independent = ~0', abs(fr.partial_corr(a, rng.normal(size=500), [c])) < 0.15)

    # 4. pooled statistics
    n = 4000
    R = rng.uniform(0.05, 0.95, n)
    q = rng.dirichlet([1, 1, 1], n)
    Hq = -(q * np.log2(q)).sum(axis=1)
    WE = (fr.binary_entropy(R) + (1 - R) * Hq) / 2
    lmu = rng.normal(size=n)
    ps = fr.pooled_stats(R, WE, Hq / np.log2(3), lmu)
    check('4 share_var_B + r2 sensible', 0 < ps['share_var_B'] < 1 and 0 < ps['r2_WE_A'] < 1)
    check('4 eta2 <= 1, >= r2 of A', ps['eta2_WE_given_R'] <= 1 and ps['eta2_WE_given_R'] >= ps['r2_WE_A'] - 0.02)
    check('4 VIF of independent lmu ~ 1', abs(ps['vif_logmu'] - 1) < 0.05, f"{ps['vif_logmu']:.3f}")
    u = rng.uniform(size=n)
    check('4 MI: identical > independent', fr.mutual_info_ranks(u, u) > 4.0
          and fr.mutual_info_ranks(u, rng.uniform(size=n)) < 0.1)
    check('4 MI of identical = log2(bins)', abs(fr.mutual_info_ranks(u, u) - np.log2(fr.MI_BINS)) < 1e-9)
    check('4 eta2 of a function = ~1', fr.eta_squared(np.sin(3 * u), u) > 0.99)
    dn = fr.density(R, WE)
    check('4 density counts add up', int(dn['count'].sum()) == n and (dn['count'] > 0).all())

    # 5. window statistics: zero rows left out, NaN below MIN_VALID
    f = fm(X)
    y = rng.poisson(4.0, size=40).astype(float)
    ws = fr.window_stats(f, y)
    check('5 n_valid / n_zero', ws['n_valid'] == 39 and ws['n_zero'] == 1)
    check('5 all statistics finite', all(np.isfinite(ws[c]) for c in fr.WINDOW_STATS))
    small = {k: v[:8] for k, v in f.items()}
    ws2 = fr.window_stats(small, y[:8])
    check('5 < MIN_VALID -> NaN', all(np.isnan(ws2[c]) for c in fr.WINDOW_STATS))
    ws3 = fr.window_stats(f, np.full(40, 3.0))
    check('5 constant next slot -> y statistics NaN, others kept',
          all(np.isnan(ws3[c]) for c in fr.Y_STATS) and
          all(np.isfinite(ws3[c]) for c in fr.WINDOW_STATS if c not in fr.Y_STATS))
    xx = rng.normal(size=50)
    check('5 partial corr NaN when the control explains y', np.isnan(fr.partial_corr(xx, 2 * xx + 1, [xx])))

    # 6. full run on synthetic data: windows and NDCG@10 equal to the protocol run of WSPI
    tmp = Path(tempfile.mkdtemp(prefix='t37_'))
    try:
        df = synthetic()
        data = tmp / 'syn.csv'
        df.to_csv(data, index=False)
        out = tmp / 'root' / 'youtube_hourly'
        fr.run(Namespace(data=str(data), min_obs=50, causal_universe=True, out=str(out)))
        ev = ProtocolV5Evaluator.from_csv(data, dataset_min_obs=50, seed=42, robustness=False,
                                          causal_universe=True)
        ev.verbose = False
        ref = ev.run_method(FastMethod('WSPI', 64, 32, make_v5_wspi(64)))
        p = pd.read_csv(out / 'protocol' / 'feature_relation_protocol.csv')
        ref.to_csv(tmp / 'ref.csv', index=False)          # CSV against CSV, as in --collect
        ref = pd.read_csv(tmp / 'ref.csv')
        m = p.merge(ref, on='window_id', suffixes=('', '_ref'))
        check('6 same windows as protocol WSPI', set(p.window_id) == set(ref.window_id) and len(p) > 50)
        check('6 NDCG@10 and num_items equal', (m['ndcg@10'] - m['ndcg@10_ref']).abs().max() == 0
              and (m['num_items'] == m['num_items_ref']).all())
        meta = json.loads((out / 'metadata' / 'feature_relation_run.json').read_text())
        check('6 identity in the run', meta['identity_max_abs'] < 1e-12)
        # 7. collect with a reference folder built from the same protocol run
        refdir = tmp / 'ref' / 'youtube_hourly' / 'protocol'
        refdir.mkdir(parents=True)
        ref.to_csv(refdir / 'WSPI_protocol.csv', index=False)
        grid = tmp / 'grid'
        grid.mkdir()
        gr = [dict(scenario='youtube_hourly', alpha=a_, beta=b_, part=pt, n_windows=10,
                   **{'ndcg@10_mean': 0.9, 'spearman_rho_mean': 0.8, 'rsi@10_mean': 0.9,
                      'robustness_distortion_mean': 10 + a_ + 2 * b_})
              for a_ in fr_grid() for b_ in fr_grid() for pt in ('all', 'test')]
        pd.DataFrame(gr).to_csv(grid / 'grid_summary.csv', index=False)
        fr.collect(tmp / 'root', tmp / 'ref', grid)
        c = pd.read_csv(tmp / 'root' / 'feature_relation_control.csv')
        check('7 control passes', bool(c['pass'].all()))
        s = pd.read_csv(tmp / 'root' / 'feature_relation_summary.csv')
        ww = s[s.kind == 'within_window']
        check('7 summary: every window statistic with a CI', set(ww.statistic) == set(fr.WINDOW_STATS)
              and (ww.ci_low <= ww.value).all() and (ww.value <= ww.ci_high).all())
        ab = pd.read_csv(tmp / 'root' / 'alpha_beta_total.csv')
        check('7 alpha_beta_total rows', len(ab) == 1 + 5 + 5 and set(ab.total) == {0.0, 1.0, 2.0},
              str(sorted(zip(ab.alpha, ab.beta))))
        files = sorted(q.name for q in (tmp / 'root').rglob('*') if q.is_file())
        check('8 outputs CSV/JSON only', all(q.endswith(('.csv', '.json')) for q in files), str(files))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


def fr_grid():
    return (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


if __name__ == '__main__':
    main()
