r"""
Unit tests for tools/run_param_grid.py (task T3.2 / E3)
=======================================================
  1. Cached scorer == protocol_v5.make_v5_wspi, bit for bit, for every
     (alpha, beta) of the grid, on matrices of length 32..64 (with and
     without padding), with zero rows and repeated calls (cache hits).
  2. Cached scorer == methods.wspi_assessment.WSPIAssessment on a padded
     64-sample series (to 1e-12, as in tools/test_level_sweep.py).
  3. Time split: floor(0.30 n) tuning windows, order kept, no overlap.
  4. Selection rule on toy tables: tolerance, max RSI, tie-breaks.
  5. A short protocol run (synthetic data): the cached default scorer gives
     the same protocol table as make_v5_wspi.

Run from the project root:  python tools\test_param_grid.py   (ends with ALL OK)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

from run_param_grid import (GRID, WSPIFeatureCache, grid_order, select_config,  # noqa: E402
                            split_ids, tag)
from evaluation.protocol_v5 import ProtocolV5Evaluator, make_v5_wspi  # noqa: E402
from evaluation.fast_evaluator import FastMethod  # noqa: E402

FAILS = []


def check(name, ok, info=''):
    print(f'{"OK  " if ok else "FAIL"} {name} {info}')
    if not ok:
        FAILS.append(name)


rng = np.random.RandomState(0)

# 1 --------------------------------------------------------------------------
cache = WSPIFeatureCache()
mats = []
for L in (32, 40, 63, 64):
    X = rng.poisson(rng.gamma(1.0, 20.0, size=(30, 1)), size=(30, L)).astype(float)
    X[3] = 0.0
    X[5, :-1] = 0.0
    mats.append(X)
worst_eq = True
for rep in range(2):                                   # 2nd pass = cache hits
    for X in mats:
        for al in GRID:
            for be in GRID:
                a = cache.scorer(al, be)(X)
                b = make_v5_wspi(window=64, level=3, alpha=al, beta=be)(X)
                if not np.array_equal(a, b):
                    worst_eq = False
check('cached scorer == make_v5_wspi (bitwise, 4 lengths x 49 configs x 2 passes)', worst_eq)
check('cache hits used', cache.hits > 0 and cache.misses == len(mats),
      f'(hits {cache.hits}, misses {cache.misses})')

# 2 --------------------------------------------------------------------------
try:
    from methods.wspi_assessment import WSPIAssessment
    x = rng.poisson(30.0, size=64).astype(float)
    ok = True
    for al, be in ((1.0, 1.0), (0.0, 2.0), (1.5, 0.25)):
        ref = WSPIAssessment(alpha=al, beta=be).assess_single(x)
        got = cache.scorer(al, be)(x[None, :])[0]
        ok &= abs(ref - got) <= 1e-12 * max(1.0, abs(ref))
    check('cached scorer == WSPIAssessment.assess_single (64 samples, 3 configs)', ok)
except Exception as e:  # pragma: no cover
    check('cached scorer == WSPIAssessment.assess_single', False, repr(e))

# 3 --------------------------------------------------------------------------
ids = list(range(32, 32 + 652))
tu, te = split_ids(ids[::-1])
check('split sizes 195 / 457 for n = 652', (len(tu), len(te)) == (195, 457))
check('split keeps time order, no overlap', tu == ids[:195] and te == ids[195:] and
      not set(tu) & set(te))
tu, te = split_ids(list(range(10)))
check('split floor(0.3 * 10) = 3', len(tu) == 3 and te[0] == 3)

# 4 --------------------------------------------------------------------------
t = pd.DataFrame({'config': ['a', 'b', 'c', 'd'],
                  'tune_ndcg': [0.900, 0.895, 0.890, 0.880],
                  'tune_rsi': [0.80, 0.85, 0.90, 0.99],
                  'dist': [2, 1, 0, 0]})
s = select_config(t, 'dist')
check('tolerance 1 %: d infeasible (0.880 < 0.891)', not bool(s.loc[3, 'feasible']))
check('max RSI among feasible -> b (c = 0.890 < 0.891 infeasible)',
      s.loc[s['selected'], 'config'].tolist() == ['b'])
t2 = pd.DataFrame({'config': ['x', 'y', 'z'], 'tune_ndcg': [0.90, 0.90, 0.905],
                   'tune_rsi': [0.95001, 0.95004, 0.94], 'dist': [0.5, 0.25, 0]})
s2 = select_config(t2, 'dist')
check('tie on RSI (4 d.) and NDCG -> closest to default (y)',
      s2.loc[s2['selected'], 'config'].tolist() == ['y'])
t3 = pd.DataFrame({'config': ['x', 'y'], 'tune_ndcg': [0.901, 0.903],
                   'tune_rsi': [0.95001, 0.95002], 'dist': [0, 1]})
s3 = select_config(t3, 'dist')
check('tie on RSI -> higher NDCG (y)', s3.loc[s3['selected'], 'config'].tolist() == ['y'])
check('exactly one selected', int(s['selected'].sum()) == 1 and int(s2['selected'].sum()) == 1)
check('grid order starts with default, 49 pairs',
      grid_order()[0] == (1.0, 1.0) and len(grid_order()) == 49 and len(set(grid_order())) == 49)
check('tag format', tag(0.25, 1.0) == 'a0.25_b1.00')

# 5 --------------------------------------------------------------------------
rows = []
start = pd.Timestamp('2025-01-01')
for it in range(40):
    lam = rng.gamma(1.0, 10.0)
    for k in range(140):
        v = rng.poisson(lam * (1 + 0.5 * np.sin(k / 6.0)))
        if v > 0:
            rows.append((str(it), start + pd.Timedelta(hours=k), int(v)))
df = pd.DataFrame(rows, columns=['item_id', 'timestamp', 'count'])
try:
    ev = ProtocolV5Evaluator(df, dataset_min_obs=5, seed=42, robustness=True,
                             causal_universe=True)
    ev.verbose = False
    c2 = WSPIFeatureCache()
    a = ev.run_method(FastMethod('WSPI', 64, 32, c2.scorer(1.0, 1.0)))
    b = ev.run_method(FastMethod('WSPI', 64, 32, make_v5_wspi(window=64, level=3)))
    num = [c for c in a.columns if c not in ('method',) and np.issubdtype(a[c].dtype, np.number)]
    same = len(a) == len(b) and len(a) > 20 and np.allclose(
        a[num].to_numpy(float), b[num].to_numpy(float), rtol=0, atol=0, equal_nan=True)
    check('protocol run: cached default == make_v5_wspi (synthetic, exact)', same,
          f'({len(a)} windows)')
    a2 = ev.run_method(FastMethod('WSPI', 64, 32, c2.scorer(0.5, 1.5)))
    b2 = ev.run_method(FastMethod('WSPI', 64, 32, make_v5_wspi(window=64, level=3,
                                                                alpha=0.5, beta=1.5)))
    same2 = np.allclose(a2[num].to_numpy(float), b2[num].to_numpy(float), rtol=0, atol=0,
                        equal_nan=True)
    check('protocol run: cached (0.5, 1.5) after cache fill == make_v5_wspi', same2,
          f'(hits {c2.hits})')
except Exception as e:  # pragma: no cover
    check('protocol run on synthetic data', False, repr(e))

print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
sys.exit(1 if FAILS else 0)
