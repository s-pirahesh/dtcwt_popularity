r"""
Unit tests for tools/run_level_sweep.py (T3.1 / E2).

    python tools\test_level_sweep.py

Prints one line per test and ends with ALL OK.
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from evaluation.sweep_methods import build_sweep_methods  # noqa: E402
from methods.dtcwt_assessment import DTCWTAssessment  # noqa: E402
from methods.wspi_assessment import WSPIAssessment  # noqa: E402
from run_level_sweep import LEVEL_GRID, grid_pairs  # noqa: E402

warnings.filterwarnings('ignore')
FAILED = []


def check(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILED.append(name)


check('grid: N=64 -> J 2..5, N=32 -> J 2..4 (7 pairs)',
      grid_pairs([64, 32]) == [(64, 2), (64, 3), (64, 4), (64, 5), (32, 2), (32, 3), (32, 4)])
check('grid rule N >= 2^(J+1) drops (32, 5)', (32, 5) not in grid_pairs([32], [2, 3, 4, 5]))
check('default level grid', LEVEL_GRID == {64: (2, 3, 4, 5), 32: (2, 3, 4)})

rng = np.random.RandomState(1)
for N, J in grid_pairs([64, 32]):
    X = rng.poisson(4, size=(25, N)).astype(float)
    X[0, :N // 2] = 0
    X[1, -1] = 80
    m = build_sweep_methods(N, level=J, names=['WSPI', 'DTCWT+AF'])
    ok_shape = set(m) == {'WSPI', 'DTCWT+AF'} and all(fm.window_slots == N for fm in m.values())
    w = WSPIAssessment()
    w.level = J
    ref_w = np.array([w.assess_single(r) for r in X])
    d = DTCWTAssessment(level=J)
    ref_d = np.array([d.assess_single(r) for r in X])
    sw = m['WSPI'].scorer(X)
    sd = m['DTCWT+AF'].scorer(X)
    tol = lambda r: 1e-9 * max(1.0, np.abs(r).max())  # noqa: E731
    check(f'N={N} J={J}: WSPI batch == WSPIAssessment(level={J})',
          ok_shape and np.max(np.abs(sw - ref_w)) < tol(ref_w))
    check(f'N={N} J={J}: DTCWT+AF batch == DTCWTAssessment(level={J})',
          np.max(np.abs(sd - ref_d)) < tol(ref_d))

Y = rng.poisson(3, size=(25, 64)).astype(float)
a = build_sweep_methods(64, level=2, names=['WSPI'])['WSPI'].scorer(Y)
b = build_sweep_methods(64, level=5, names=['WSPI'])['WSPI'].scorer(Y)
check('level changes the WSPI score', not np.allclose(a, b))

print('ALL OK' if not FAILED else f'FAILED: {FAILED}')
sys.exit(1 if FAILED else 0)
