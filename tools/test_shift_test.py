r"""
Unit test for tools/run_shift_test.py
=====================================
Run from the project root:   python tools\test_shift_test.py
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

import run_shift_test as st  # noqa: E402
from evaluation.fast_evaluator import FastMethod  # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator, make_v5_wspi  # noqa: E402

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


def main():
    rng = np.random.default_rng(1)
    X = rng.poisson(4.0, size=(25, 64)).astype(float)
    X[:, 30] += 40.0

    # 1. period: a circular shift by 2^J = 8 gives the s = 0 energies back
    for t in st.TRANSFORMS:
        E = st.circular_energies(X, t, [0, 8, 16])
        dev = np.max(np.abs(E[1] - E[0]) / E[0])
        dev2 = np.max(np.abs(E[2] - E[0]) / E[0])
        check(f'1 period 8 ({t})', dev < 1e-12 and dev2 < 1e-12, f'{dev:.1e}')

    # 2. band energies add up to the signal energy (both transforms, circular)
    for t, tol in (('DWT', 1e-10), ('DTCWT', 0.35)):
        E = st.circular_energies(X, t, [0])[0]
        ratio = E.sum(axis=1) / (X ** 2).sum(axis=1)
        if t == 'DWT':
            ok = np.max(np.abs(ratio - 1)) < tol          # orthogonal: exact
        else:
            ok = np.all(ratio > 0) and np.max(np.abs(ratio - ratio.mean())) < tol * ratio.mean()
        check(f'2 energy bookkeeping ({t})', ok, f'ratio {ratio.min():.3f}..{ratio.max():.3f}')

    # 3. DTCWT level 1 is not decimated -> its energy is exactly shift invariant
    E = st.circular_energies(X, 'DTCWT')
    M = st.shift_metrics(E)
    check('3 DTCWT level-1 cv = 0', np.max(M[:, 1]) < 1e-12, f'{np.max(M[:, 1]):.1e}')
    check('3 DWT level-1 cv > 0', np.max(st.shift_metrics(st.circular_energies(X, 'DWT'))[:, 1]) > 1e-3)

    # 4. shift_metrics by hand
    Eh = np.array([[[1.0, 1.0, 1.0, 1.0]], [[3.0, 1.0, 1.0, 1.0]]])     # (S=2, n=1, 4)
    Mh = st.shift_metrics(Eh)[0]
    R, WE = st.r_we(Eh[:, 0, :])
    check('4 cv by hand', abs(Mh[0] - 0.5) < 1e-12 and Mh[1] == 0 and Mh[2] == 0)
    check('4 sd R and WE by hand', abs(Mh[4] - np.std(R)) < 1e-12 and abs(Mh[5] - np.std(WE)) < 1e-12)
    check('4 R, WE of equal bands', abs(R[0] - 0.25) < 1e-12 and abs(WE[0] - 1.0) < 1e-12)

    # 5. R and WE from the band energies == the WSPI features (DTCWT and DWT)
    from tools.run_param_grid import WSPIFeatureCache
    mu, R1, WE1 = WSPIFeatureCache(use_cache=False).features(X)
    F = st.syn_features(X, 'DTCWT')
    check('5 DTCWT features == WSPI', np.allclose(F.R, R1, rtol=0, atol=1e-13) and
          np.allclose(F.WE, WE1, rtol=0, atol=1e-13) and np.allclose(F.mu_L, mu, rtol=1e-13, atol=0))
    mu2, R2, WE2 = st.DWTFeatureCache(use_cache=False).features(X)
    F2 = st.syn_features(X, 'DWT')
    check('5 DWT features == T3.3 DWT', np.allclose(F2.R, R2, atol=1e-13) and
          np.allclose(F2.WE, WE2, atol=1e-13) and np.allclose(F2.mu_L, mu2, rtol=1e-13, atol=0))
    s = st.syn_scorers()['WSPI'](X)
    check('5 WSPI scorer == mu exp(R - WE)', np.allclose(s, F.mu_L * np.exp(F.R - F.WE), rtol=1e-12))

    # 6. synthetic matrix and step statistics
    bg = np.ones(64)
    Xs = st.syn_matrix(bg, st.SHAPES['burst3'], 48)
    amp = st.AMP_X * (st.LAM0 + st.LAM1) / 2
    check('6 event moves by one slot', all(Xs[j, 49 + j] == 1 + amp for j in range(st.SYN_SHIFTS)) and
          np.allclose(Xs.sum(axis=1), 64 + 2 * amp))
    check('6 event stays inside', Xs.shape == (9, 64) and 48 + 8 + 2 <= 63)
    sc = st.syn_scorers()
    check('6 SMA invariant', st.step_stats(sc['SMA'](Xs))['mean_abs_dlog'] < 1e-12)
    e = st.step_stats(sc['EWMA-eq'](Xs))
    check('6 EWMA never goes down', e['share_wrong'] == 0 and e['mean_dlog'] > 0)
    ss = st.step_stats(np.array([1.0, 2.0, 1.0, 1.0, 4.0]))
    check('6 step stats by hand', ss['share_wrong'] == 0.25 and
          abs(ss['mean_wrong_dlog'] - np.log(2) / 4) < 1e-12 and abs(ss['mean_dlog'] - np.log(4) / 4) < 1e-12)

    # 7. part A: eligibility and window set == protocol V5 run_method (windows >= 64)
    df = synthetic()
    tmp = Path(tempfile.mkdtemp(prefix='t36test_'))
    try:
        data = tmp / 'toy.csv'
        df.to_csv(data, index=False)
        for causal in (True, False):
            a = Namespace(data=str(data), min_obs=30, causal_universe=causal, out=str(tmp / f'A{causal}'),
                          first_window=64)
            st.run_real(a)
            got = pd.read_csv(tmp / f'A{causal}' / 'protocol' / 'DTCWT_protocol.csv')
            ev = ProtocolV5Evaluator.from_csv(data, 30, causal_universe=causal, robustness=False)
            ev.verbose = False
            ref = ev.run_method(FastMethod('WSPI', 64, 32, make_v5_wspi(64, 3)))
            ref = ref[ref.window_id >= 64]
            check(f'7 windows and item counts (causal={causal})',
                  got.window_id.tolist() == ref.window_id.tolist() and
                  got.num_items.tolist() == ref.num_items.tolist(), f'{len(got)} windows')
            meta = json.loads((tmp / f'A{causal}' / 'metadata' / 'shift_run.json').read_text())
            check(f'7 period check in run (causal={causal})', max(meta['period_max_rel_dev'].values()) < 1e-12)
        w = pd.read_csv(tmp / 'ATrue' / 'protocol' / 'DWT_protocol.csv')
        check('7 same windows for DWT', w.window_id.tolist() == got.window_id.tolist())
        # per-window mean by hand, one window
        k = int(got.window_id.iloc[3])
        ev = ProtocolV5Evaluator.from_csv(data, 30, causal_universe=True, robustness=False)
        cum = ev.V[:, :k].sum(axis=1)
        lo, hi = ev.window_bounds(k, 64)
        idx = np.where(((ev.C[:, hi] - ev.C[:, lo]) >= 32) & (cum >= 30))[0]
        Mk = st.shift_metrics(st.circular_energies(ev.V[idx, lo:hi], 'DWT'))
        check('7 per-window mean by hand', np.allclose(w.set_index('window_id').loc[k, st.METRICS].to_numpy(float),
                                                      Mk.mean(axis=0), rtol=1e-12, atol=1e-15))
        # collect and stats run without error
        root = tmp / 'root'
        shutil.copytree(tmp / 'ATrue', root / 'youtube_hourly')
        st.collect(root, ref_t15=tmp / 'none')
        s = pd.read_csv(root / 'shift_summary.csv')
        check('7 collect rows', len(s) == 12 and set(s['transform']) == set(st.TRANSFORMS))
        st.stats(root, tmp / 'stats', B=200)
        p = pd.read_csv(tmp / 'stats' / 'youtube_hourly' / 'paired_tests.csv')
        check('7 stats: lower is better, reference DTCWT',
              set(p.metric) == set(st.METRICS) and p.lower_is_better.all() and (p.reference == 'DTCWT').all())
        sub = p[p.metric == 'cv_E_3'].iloc[0]
        check('7 stats orientation', (sub.diff_ref_minus_method < 0) == (sub.cliffs_delta_favours_ref > 0))
        # part B (small) runs and writes CSV/JSON only
        b = Namespace(out=str(tmp / 'syn'), reps=6, seed=42, B=200)
        st.run_synthetic(b)
        files = sorted(p.name for p in (tmp / 'syn').rglob('*') if p.is_file())
        check('8 synthetic outputs CSV/JSON only', all(f.endswith(('.csv', '.json')) for f in files)
              and len(files) == 5, str(files))
        sm = pd.read_csv(tmp / 'syn' / 'synthetic_summary.csv')
        check('8 summary covers all', len(sm) == 2 * 2 * (6 * 4 + 2 * 6))
        sc1 = pd.read_csv(tmp / 'syn' / 'synthetic_scores.csv')
        st2 = Namespace(out=str(tmp / 'syn2'), reps=6, seed=42, B=200)
        st.run_synthetic(st2)
        sc2 = pd.read_csv(tmp / 'syn2' / 'synthetic_scores.csv')
        check('8 synthetic reproducible (seed)', sc1.equals(sc2))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
