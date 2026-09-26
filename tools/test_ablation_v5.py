r"""
Unit tests for tools/run_ablation_v5.py (task T3.3 / E6)
========================================================
  1. DTCWT variants == protocol_v5.make_v5_wspi (bit for bit): WSPI = (1,1),
     Trend = use_R=False,use_WE=False, Trend+R = use_WE=False, Trend+WE = use_R=False;
     and == the T3.2 cached grid scorer at (1,1), (0,0), (1,0), (0,1).
  2. DWT decomposition == the one of DWT+AF in V5: the DWT+AF score rebuilt
     from our coefficients equals protocol_v5.make_v5_dwt bit for bit.
  3. DWT features: batch == row-by-row 1-D pywt + scalar loops (1e-12), and
     DWT-WSPI == the V4 class methods.wspi_ablation.WSPIAblation with DWT,
     no slope, no clip, R and WE weights 1 (1e-12, 64-sample series).
  4. Fusion formulas == direct formulas on the features.
  5. Per-item scores: the score of a row does not depend on the other rows
     (needed for the robustness test, which scores 50 targets alone).
  6. End-to-end on synthetic data: run() + collect() with reference folders
     built from make_v5_wspi and the T3.2 scorer -> every control 'equal';
     a perturbed reference -> 'DIFFERENT'; the fusion WSPI file equals the
     ablation WSPI file; summary has 8 + 4 rows.

Run from the project root:  python tools\test_ablation_v5.py   (ends with ALL OK)
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pywt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import run_ablation_v5 as ab  # noqa: E402
from run_param_grid import WSPIFeatureCache  # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator, make_v5_wspi, make_v5_dwt  # noqa: E402
from evaluation.fast_evaluator import FastMethod, _waf_rows  # noqa: E402

FAILS = []


def check(name, ok, info=''):
    print(f'{"OK  " if ok else "FAIL"} {name} {info}')
    if not ok:
        FAILS.append(name)


rng = np.random.RandomState(0)
mats = []
for L in (32, 40, 63, 64):
    X = rng.poisson(rng.gamma(1.0, 20.0, size=(30, 1)), size=(30, L)).astype(float)
    X[3] = 0.0
    X[5, :-1] = 0.0
    mats.append(X)

# 1 --------------------------------------------------------------------------
dc = WSPIFeatureCache()
g = WSPIFeatureCache()
pairs = {'full': dict(), 'trend': dict(use_R=False, use_WE=False),
         'R': dict(use_WE=False), 'WE': dict(use_R=False)}
grid = {'full': (1.0, 1.0), 'trend': (0.0, 0.0), 'R': (1.0, 0.0), 'WE': (0.0, 1.0)}
ok1, ok1b = True, True
for rep in range(2):
    for X in mats:
        for kind, kw in pairs.items():
            a = ab.make_scorer(dc, kind, kind)(X)
            ok1 &= np.array_equal(a, make_v5_wspi(window=64, level=3, **kw)(X))
            ok1b &= np.array_equal(a, g.scorer(*grid[kind])(X))
check('DTCWT variants == make_v5_wspi (bitwise, 4 lengths x 4 variants x 2 passes)', ok1)
check('DTCWT variants == T3.2 grid scorer (bitwise)', ok1b)

# 2 --------------------------------------------------------------------------
ok2 = True
for X in mats:
    low, highs = ab.dwt_coeffs(X)
    rebuilt = _waf_rows(low) + 0.1 * _waf_rows(highs[0])
    ok2 &= np.array_equal(rebuilt, make_v5_dwt(window=64, level=3)(X))
check('DWT coefficients == DWT+AF (make_v5_dwt), bitwise', ok2)
lens = ab.coeff_lengths()
check('coefficient lengths recorded', lens['dwt']['lowpass'] > 0 and lens['dtcwt']['lowpass'] > 0,
      f'({lens})')

# 3 --------------------------------------------------------------------------
wc = ab.DWTFeatureCache()


def row_features(x):
    xp = x if len(x) >= 64 else np.pad(x, (64 - len(x), 0), mode='reflect')
    c = pywt.wavedec(xp, 'db4', level=3, mode='symmetric')
    lm = np.abs(c[0])
    n = len(lm)
    w = [2.0 ** -(n - 1 - k) for k in range(n)]
    mu = sum(w[k] * lm[k] for k in range(n)) / sum(w)
    en = [float(np.sum(lm ** 2))] + [float(np.sum(np.abs(h) ** 2)) for h in c[1:]]
    tot = sum(en)
    R = en[0] / tot if tot > 0 else 0.0
    p = [e / tot for e in en if e > 0] if tot > 0 else []
    WE = (-sum(q * np.log2(q) for q in p) / np.log2(len(en))) if p else 0.0
    return mu, R, WE


worst = 0.0
for X in mats:
    mu, R, WE = wc.features(X)
    for i in range(X.shape[0]):
        m2, r2, w2 = row_features(X[i])
        worst = max(worst, abs(mu[i] - m2) / max(1.0, abs(m2)), abs(R[i] - r2), abs(WE[i] - w2))
check('DWT features batch == row-by-row (1e-12)', worst <= 1e-12, f'(worst {worst:.2e})')
try:
    from methods.wspi_ablation import WSPIAblation
    old = WSPIAblation(alpha_slope=0.0, beta_ratio=1.0, gamma_entropy=1.0,
                       use_dtcwt=False, use_clip=False)
    ok3 = old.level == 3
    for _ in range(5):
        x = rng.poisson(rng.gamma(1.0, 30.0), size=64).astype(float) + 1.0
        ref = old.assess_single(x)
        got = ab.make_scorer(wc, 'full', 'DWT-WSPI')(x[None, :])[0]
        ok3 &= abs(ref - got) <= 1e-12 * max(1.0, abs(ref))
    check('DWT-WSPI == WSPIAblation(DWT, no slope, no clip) (1e-12, 5 series)', ok3)
except Exception as e:  # pragma: no cover
    check('DWT-WSPI == WSPIAblation', False, repr(e))

# 4 --------------------------------------------------------------------------
X = mats[-1]
mu, R, WE = dc.features(X)
direct = {'full': mu * np.exp(R - WE), 'trend': mu, 'R': mu * np.exp(R), 'WE': mu * np.exp(-WE),
          'linear': mu * (1 + R - WE), 'product': mu * R * (1 - WE), 'additive': mu + R - WE}
ok4 = all(np.allclose(ab.fuse(k, mu, R, WE, len(mu)), v, rtol=1e-15, atol=0)
          for k, v in direct.items())
check('fusion formulas == direct formulas (7 kinds)', ok4)
check('every variant has a formula', all(k in ab.FORMULA for fam in ab.FAMILIES.values()
                                         for _, k in fam.values()))

# 5 --------------------------------------------------------------------------
ok5 = True
sub = [7, 2, 19, 11]
for fam in ab.FAMILIES.values():
    for name, (trn, kind) in fam.items():
        cache = WSPIFeatureCache(use_cache=False) if trn == 'dtcwt' else ab.DWTFeatureCache(use_cache=False)
        f = ab.make_scorer(cache, kind, name)
        for X in mats:
            ok5 &= np.allclose(f(X)[sub], f(X[sub]), rtol=1e-12, atol=0)
            ok5 &= np.allclose(f(X)[[4]], f(X[[4]]), rtol=1e-12, atol=0)
check('per-item scores (subset of rows gives the same scores), 11 variants', ok5)

# 6 --------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix='t33_'))
try:
    rows = []
    start = pd.Timestamp('2025-01-01')
    for it in range(40):
        lam = rng.gamma(1.0, 10.0)
        for k in range(140):
            v = rng.poisson(lam * (1 + 0.5 * np.sin(k / 6.0)))
            if v > 0:
                rows.append((str(it), start + pd.Timedelta(hours=k), int(v)))
    df = pd.DataFrame(rows, columns=['item_id', 'timestamp', 'count'])
    csv = tmp / 'syn.csv'
    df.to_csv(csv, index=False)
    root = tmp / 'T3.3'
    args = argparse.Namespace(data=str(csv), min_obs=5, out=str(root / 'syn'), first_window=32,
                              seed=42, causal_universe=True, resume=False, no_cache=False)
    ab.run(args)
    # reference folders
    ev = ProtocolV5Evaluator.from_csv(csv, dataset_min_obs=5, seed=42, robustness=True,
                                      causal_universe=True)
    ev.verbose = False
    r15 = tmp / 'T15' / 'syn' / 'protocol'
    r15.mkdir(parents=True)
    ev.run_method(FastMethod('WSPI', 64, 32, make_v5_wspi(window=64, level=3))).to_csv(
        r15 / 'WSPI_protocol.csv', index=False)
    r32 = tmp / 'T32' / 'syn' / 'protocol'
    r32.mkdir(parents=True)
    gc = WSPIFeatureCache()
    for tg, (al, be) in (('a0.00_b0.00', (0, 0)), ('a1.00_b0.00', (1, 0)), ('a0.00_b1.00', (0, 1))):
        d = ev.run_method(FastMethod('WSPI', 64, 32, gc.scorer(al, be)))
        d = d[d['window_id'] >= 32]
        d['method'] = tg
        d.to_csv(r32 / f'{tg}_protocol.csv', index=False)
    ab.collect(root, tmp / 'T15', tmp / 'T32')
    ctrl = pd.read_csv(root / 'ablation_control.csv')
    check('collect: 4 controls, all equal', len(ctrl) == 4 and (ctrl['status'] == 'equal').all(),
          f'({ctrl["status"].tolist()})')
    sm = pd.read_csv(root / 'ablation_summary.csv')
    check('collect: summary 8 ablation + 4 fusion rows',
          (sm['family'] == 'ablation').sum() == 8 and (sm['family'] == 'fusion').sum() == 4)
    check('all variants share the same windows', bool(sm['same_windows_as_family'].all()),
          f'({int(sm["n_windows"].iloc[0])} windows)')
    fa = pd.read_csv(root / 'syn' / 'ablation' / 'protocol' / 'WSPI_protocol.csv')
    ff = pd.read_csv(root / 'syn' / 'fusion' / 'protocol' / 'WSPI_protocol.csv')
    check('fusion WSPI == ablation WSPI', fa.equals(ff))
    bad = pd.read_csv(r15 / 'WSPI_protocol.csv')
    bad.loc[bad.index[-1], 'rsi@10'] += 0.01
    bad.to_csv(r15 / 'WSPI_protocol.csv', index=False)
    ab.collect(root, tmp / 'T15', tmp / 'T32')
    ctrl = pd.read_csv(root / 'ablation_control.csv')
    check('perturbed reference is detected',
          ctrl.loc[ctrl['variant'] == 'WSPI', 'status'].iloc[0] == 'DIFFERENT')
    only = sorted(p.suffix for p in root.rglob('*') if p.is_file())
    check('only CSV and JSON written', set(only) <= {'.csv', '.json'}, f'({set(only)})')
except Exception as e:  # pragma: no cover
    import traceback
    traceback.print_exc()
    check('end-to-end run on synthetic data', False, repr(e))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
