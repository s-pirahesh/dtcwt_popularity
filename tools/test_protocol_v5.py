r"""
T1.4 unit checks for evaluation/protocol_v5.py
==============================================
1. Tie-stable metrics == metrics.py on tie-free score vectors
   (NDCG@K, Coverage@K, rank distortion).
2. With ties, the stable order puts the lower item index first, and the
   result does not change when the input is shuffled and mapped back.
3. V5 wavelet scorers on full 64-slot series == the ORIGINAL classes with
   J=3 (DWTAssessment, DTCWTAssessment, WSPIAssessment): no padding.
4. V5 wavelet scorers on shorter series (32..63) == original class applied
   to np.pad(x, (64-L, 0), 'reflect')  (padding policy of protocol V5).
5. Baseline scorers are unchanged (they are the fast_evaluator ones).

    python tools/test_protocol_v5.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import calculate_ndcg, calculate_coverage, calculate_rank_distortion
from evaluation.protocol_v5 import (stable_order, ndcg_stable, coverage_stable,
                                    rank_distortion_stable, build_v5_methods)
from methods.dwt_assessment import DWTAssessment
from methods.dtcwt_assessment import DTCWTAssessment
from methods.wspi_assessment import WSPIAssessment


def samples(rng, n_rows, L):
    X = np.vstack([rng.poisson(30, size=(n_rows, L)).astype(float),
                   (rng.poisson(2, size=(n_rows, L)) * (rng.rand(n_rows, L) < 0.3)).astype(float),
                   rng.lognormal(5, 2, size=(n_rows, L)).round(),
                   np.zeros((2, L))])
    X[0, -1] += 5000
    return X


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    return ok


def main():
    warnings.filterwarnings('ignore')
    rng = np.random.RandomState(7)
    ok_all = True

    # 1. tie-free equality with metrics.py
    worst = 0.0
    for _ in range(300):
        n = rng.randint(2, 300)
        sc = rng.rand(n) * 100          # continuous -> no ties
        act = rng.poisson(5, n).astype(float) * (rng.rand(n) < 0.7)
        o = stable_order(sc)
        for K in (5, 10, 20):
            worst = max(worst, abs(ndcg_stable(o, act, K) - calculate_ndcg(sc, act, K)))
            worst = max(worst, abs(coverage_stable(o, act, K) - calculate_coverage(sc, act, K)))
        t = rng.randint(n)
        noisy = sc.copy()
        noisy[t] += rng.rand() * 100
        worst = max(worst, abs(rank_distortion_stable(sc, noisy, t)
                               - calculate_rank_distortion(sc, noisy, t)))
    ok_all &= check('1 tie-free metrics == metrics.py', worst < 1e-12, f'max|diff|={worst:.2e}')

    # 2. tie rule
    sc = np.array([3.0, 5.0, 5.0, 1.0, 5.0, np.nan])
    o = stable_order(sc)
    ok_all &= check('2a ties -> lower index first, NaN last', o.tolist() == [1, 2, 4, 0, 3, 5],
                    str(o.tolist()))
    same = True
    for _ in range(50):
        n = 200
        sc = rng.randint(0, 5, n).astype(float)          # many ties
        o1 = stable_order(sc)
        o2 = stable_order(sc.copy())
        same &= np.array_equal(o1, o2)
        # order is fully determined: score desc, then index asc
        ref = sorted(range(n), key=lambda i: (-sc[i], i))
        same &= o1.tolist() == ref
    ok_all &= check('2b tie order deterministic (score desc, item index asc)', same)

    # 3/4. wavelet scorers
    m = build_v5_methods(level=3)
    orig = {'DWT+AF': DWTAssessment(wavelet='db4', level=3, mode='symmetric'),
            'DTCWT+AF': DTCWTAssessment(level=3, biort='near_sym_a', qshift='qshift_a'),
            'WSPI': WSPIAssessment(alpha=1.0, beta=1.0)}
    for name, o in orig.items():
        worst64, worstpad = 0.0, 0.0
        for _ in range(5):
            X = samples(rng, 8, 64)
            ref = np.array([float(o.assess_single(x.copy())) for x in X])
            got = m[name].scorer(X.copy())
            worst64 = max(worst64, float((np.abs(ref - got) / np.maximum(np.abs(ref), 1e-12)).max()))
        for L in range(32, 64):
            X = samples(rng, 4, L)
            ref = np.array([float(o.assess_single(np.pad(x, (64 - L, 0), mode='reflect')))
                            for x in X])
            got = m[name].scorer(X.copy())
            worstpad = max(worstpad, float((np.abs(ref - got) / np.maximum(np.abs(ref), 1e-12)).max()))
        ok_all &= check(f'3 {name:<9} W=64 no pad == original J=3', worst64 < 1e-9, f'rel={worst64:.2e}')
        ok_all &= check(f'4 {name:<9} L<64 reflect pad == original(pad)', worstpad < 1e-9,
                        f'rel={worstpad:.2e}')

    # 5. baselines identical objects
    from evaluation.fast_evaluator import build_default_methods
    fm = build_default_methods()
    same = all(m[n].scorer is fm[n].scorer for n in ('AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF'))
    same &= all(m[n].window_slots == 7 and m[n].min_obs == 3
                for n in ('AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF'))
    same &= all(m[n].window_slots == 64 and m[n].min_obs == 32 for n in orig)
    ok_all &= check('5 baseline scorers unchanged; W and min_obs from method_configs', same)

    print('\nALL OK' if ok_all else '\nSOME FAILED')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
