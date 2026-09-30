r"""
Unit tests for evaluation/sweep_methods.py (T2.2).

    python tools\test_sweep_methods.py

Prints one line per test and ends with ALL OK.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.sweep_methods import (SWEEP_WINDOWS, batch_sma, build_sweep_methods,  # noqa: E402
                                      make_batch_ewma_eq, make_batch_holt, wavelet_min_obs)
from methods.forecasting_baselines import HoltForecast  # noqa: E402

FAILED = []


def check(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILED.append(name)


rng = np.random.RandomState(0)
X = rng.poisson(5, size=(50, 16)).astype(float)
X[3] = 0
X[4, :8] = 0

check('SMA equals row mean', np.allclose(batch_sma(X), X.mean(1)))

f = make_batch_ewma_eq(9)
ref = []
for row in X:
    e = row[0]
    for v in row[1:]:
        e = 0.2 * v + 0.8 * e
    ref.append(e)
check('EWMA-eq alpha = 2/(N+1) (N=9 gives 0.2)', abs(f.alpha - 0.2) < 1e-12 and np.allclose(f(X), ref))
check('EWMA-eq alpha for N=64', abs(make_batch_ewma_eq(64).alpha - 2 / 65) < 1e-12)

h = make_batch_holt()
hf = HoltForecast(alpha=0.3, gamma=0.1, horizon=7)
for W in (3, 7, 16, 64):
    Xw = rng.poisson(3, size=(40, W)).astype(float)
    Xw[0] = 0
    Xw[1, -1] = 50
    ref = np.array([hf.assess_single(r) for r in Xw])
    check(f'Holt batch == HoltForecast.assess_single (W={W})',
          np.max(np.abs(h(Xw) - ref)) < 1e-9 * max(1, np.abs(ref).max()))

check('wavelet min_obs = min(32, N/2)',
      [wavelet_min_obs(n) for n in (16, 32, 64, 128)] == [8, 16, 32, 32])

m7 = build_sweep_methods(7)
check('N=7: 9 methods, no wavelet-based', len(m7) == 9 and 'WSPI' not in m7)
ok = True
for N in SWEEP_WINDOWS[1:]:
    m = build_sweep_methods(N)
    ok &= len(m) == 12 and all(fm.window_slots == N for fm in m.values())
    ok &= all(m[n].min_obs == 3 for n in ('AF', 'PFRF', 'SMA', 'EWMA-eq', 'Holt'))
    ok &= m['WSPI'].min_obs == wavelet_min_obs(N)
check('N>=16: 12 methods, window N, min_obs rule', ok)

for N in (16, 32, 128):
    m = build_sweep_methods(N, names=['DWT+AF', 'DTCWT+AF', 'WSPI'])
    Xn = rng.poisson(4, size=(30, N)).astype(float) + 0.1
    s = {n: fm.scorer(Xn) for n, fm in m.items()}
    check(f'wavelet-based scorers run at N={N} (finite, one score per item)',
          all(v.shape == (30,) and np.isfinite(v).all() for v in s.values()))

print('ALL OK' if not FAILED else f'FAILED: {FAILED}')
sys.exit(1 if FAILED else 0)
