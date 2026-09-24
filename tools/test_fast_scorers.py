r"""
T1.2 unit check — batched scorers vs. the original assess_single()
==================================================================
For every method, random count series (Poisson, bursty, sparse with many
zeros, all-zero) of every length that occurs in the protocol are scored
twice: row by row with the ORIGINAL class, and in one batch with
evaluation/fast_evaluator.py.  Reports the max absolute and relative
difference.  Pass criterion: max relative difference < 1e-9.

    python tools/test_fast_scorers.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baselines.popularity_baselines import (AFMethod, EWMAMethod, RRDMethod,
                                            VSEMethod, CompoundPopMethod, PFRFMethod)
from methods.dwt_assessment import DWTAssessment
from methods.dtcwt_assessment import DTCWTAssessment
from methods.wspi_assessment import WSPIAssessment
from evaluation.fast_evaluator import build_default_methods


def samples(rng, n_rows, L):
    blocks = [rng.poisson(30, size=(n_rows, L)).astype(float),
              (rng.poisson(2, size=(n_rows, L)) * (rng.rand(n_rows, L) < 0.3)).astype(float),
              rng.lognormal(5, 2, size=(n_rows, L)).round(),
              np.zeros((2, L))]
    X = np.vstack(blocks)
    X[0, -1] += 5000          # a spike
    return X


def main():
    warnings.filterwarnings('ignore')
    rng = np.random.RandomState(1)
    orig = {
        'AF': AFMethod(), 'EWMA': EWMAMethod(alpha=0.2), 'RRD': RRDMethod(),
        'VSE': VSEMethod(), 'CompoundPop': CompoundPopMethod(), 'PFRF': PFRFMethod(),
        'DWT+AF': DWTAssessment(wavelet='db4', level=3, mode='symmetric'),
        'DTCWT+AF': DTCWTAssessment(level=2, biort='near_sym_a', qshift='qshift_a'),
        'WSPI': WSPIAssessment(alpha=1.0, beta=1.0),
    }
    fast = build_default_methods(dtcwt_level=2)
    lengths = {'AF': range(3, 8), 'EWMA': range(3, 8), 'RRD': range(3, 8),
               'VSE': range(3, 8), 'CompoundPop': range(3, 8), 'PFRF': range(3, 8),
               'DWT+AF': range(32, 65), 'DTCWT+AF': range(32, 65), 'WSPI': range(32, 65)}
    ok_all = True
    for name, m in orig.items():
        worst_abs, worst_rel = 0.0, 0.0
        for L in lengths[name]:
            X = samples(rng, 8, L)
            ref = np.array([float(m.assess_single(x.copy())) for x in X])
            got = fast[name].scorer(X.copy())
            a = np.abs(ref - got)
            r = a / np.maximum(np.abs(ref), 1e-12)
            worst_abs = max(worst_abs, float(a.max()))
            worst_rel = max(worst_rel, float(r.max()))
        ok = worst_rel < 1e-9
        ok_all &= ok
        print(f"{name:<12} max|diff|={worst_abs:.3e}  max rel={worst_rel:.3e}  "
              f"{'OK' if ok else 'FAIL'}")
    print('\nALL OK' if ok_all else '\nSOME FAILED')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
