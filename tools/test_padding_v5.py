r"""
Unit tests for tools/run_padding_v5.py (task T3.4 / E8)
======================================================
  1. dtcwt_forward_ext(mode='symmetric') == dtcwt.Transform1d.forward, bit for
     bit (lowpass and the three highpasses), N in {16, 32, 64, 128}, J in {2, 3}.
  2. Default scorers == protocol V5 scorers, bit for bit:
     WSPI == make_v5_wspi, DTCWT+AF == make_v5_dtcwt_af, DWT+AF == make_v5_dwt
     (series of length 32, 40, 63, 64).
  3. Every extension mode: (a) matches a direct construction of the extended
     signal for the first level (np.pad of the column + 'valid' filtering);
     (b) DWT+AF modes pass the right name to pywt (score == direct pywt call).
  4. Left-padding modes change only series shorter than 64; with 64 samples
     every pad mode gives the default score.
  5. Per-item scores: a subset of rows gives the same scores (robustness test
     scores targets alone).
  6. boundary_weight: interior coefficients (6..11) have a share < 0.001, shares are
     symmetric, the mu_L weights sum to 1.
  7. End-to-end on synthetic data: run(both) + collect() with a reference T1.5
     folder built from the V5 scorers -> 6 controls 'equal'; the pad run covers
     windows 32..64 and equals the ext default there except in the padded
     windows for non-default modes; windows >= 65 of the combined 'all' subset
     equal the default run; a perturbed reference is detected; only CSV/JSON.

Run from the project root:  python tools\test_padding_v5.py   (ends with ALL OK)
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import dtcwt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import run_padding_v5 as pv  # noqa: E402
from evaluation.protocol_v5 import (ProtocolV5Evaluator, make_v5_wspi,  # noqa: E402
                                    make_v5_dtcwt_af, make_v5_dwt)
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
ok1 = True
tr = dtcwt.Transform1d(biort=pv.BIORT, qshift=pv.QSHIFT)
for N in (16, 32, 64, 128):
    for J in (2, 3):
        X = rng.poisson(5.0, size=(N, 12)).astype(float)
        p = tr.forward(X, nlevels=J)
        lo, hs = pv.dtcwt_forward_ext(X, J, 'symmetric')
        ok1 &= np.array_equal(lo, p.lowpass)
        ok1 &= all(np.array_equal(a, b) for a, b in zip(hs, p.highpasses))
check('symmetric extension == dtcwt library (bit for bit, 8 N x J)', ok1)

# 2 --------------------------------------------------------------------------
refs = {'WSPI': make_v5_wspi(window=64, level=3), 'DTCWT+AF': make_v5_dtcwt_af(window=64, level=3),
        'DWT+AF': make_v5_dwt(window=64, level=3)}
for m, rf in refs.items():
    f = pv.make_scorer(m)
    check(f'default {m} == protocol V5 scorer (bit for bit, 4 lengths)',
          all(np.array_equal(f(X), rf(X)) for X in mats))

# 3 --------------------------------------------------------------------------
NPMODE = {'symmetric': 'symmetric', 'reflect': 'reflect', 'edge': 'edge',
          'zero': 'constant', 'periodic': 'wrap'}
ok3 = True
h0o = dtcwt.coeffs.biort(pv.BIORT)[0].ravel()
X = rng.poisson(5.0, size=(64, 7)).astype(float)
m2 = len(h0o) // 2
for mode in pv.MODES:
    got = pv.colfilter_ext(X, h0o, mode)
    Xe = np.pad(X, ((m2, m2), (0, 0)), mode=NPMODE[mode])
    direct = np.stack([np.convolve(Xe[:, j], h0o, mode='valid') for j in range(X.shape[1])], axis=1)
    ok3 &= np.allclose(got, direct, rtol=1e-12, atol=1e-12)
check('level-1 extension == np.pad construction (5 modes)', ok3)
ok3b = True
for mode in pv.MODES:
    Xp = mats[-1]
    c = pywt.wavedec(Xp, 'db4', level=3, mode=pv.PYWT_MODE[mode], axis=-1)
    ok3b &= np.array_equal(pv.make_scorer('DWT+AF', ext=mode)(Xp),
                           _waf_rows(c[0]) + 0.1 * _waf_rows(c[-1]))
check('DWT+AF extension modes == direct pywt calls (5 modes)', ok3b)
diff = [not np.allclose(pv.make_scorer('WSPI', ext=m)(mats[-1]), pv.make_scorer('WSPI')(mats[-1]))
        for m in pv.MODES if m != 'symmetric']
check('every non-default extension changes the WSPI score', all(diff))

# 4 --------------------------------------------------------------------------
ok4, chg = True, True
for m in pv.METHODS:
    base = pv.make_scorer(m)
    for pm in pv.PAD_MODES:
        f = pv.make_scorer(m, pad=pm)
        ok4 &= np.array_equal(f(mats[-1]), base(mats[-1]))
        if pm != 'reflect':
            chg &= not np.allclose(f(mats[0]), base(mats[0]))
check('pad modes do not change 64-sample series (3 methods x 5 modes)', ok4)
check('pad modes change 32-sample series (3 methods x 4 modes)', chg)
Xs = pv.pad_left(mats[0], 'zero')
check('zero pad puts zeros on the left, data on the right',
      Xs.shape[1] == 64 and not Xs[:, :32].any() and np.array_equal(Xs[:, 32:], mats[0]))

# 5 --------------------------------------------------------------------------
ok5 = True
sub = [7, 2, 19, 11]
for m in pv.METHODS:
    for mode in pv.MODES:
        f = pv.make_scorer(m, ext=mode)
        for X in mats:
            ok5 &= np.allclose(f(X)[sub], f(X[sub]), rtol=1e-12, atol=0)
            ok5 &= np.allclose(f(X)[[4]], f(X[[4]]), rtol=1e-12, atol=0)
check('per-item scores (3 methods x 5 modes)', ok5)

# 6 --------------------------------------------------------------------------
bw = pv.boundary_weight()
s = bw['extended_weight_share'].to_numpy()
check('boundary_weight: 16 coefficients, interior < 0.001, symmetric, weights sum 1',
      len(bw) == 16 and np.all(s[5:11] < 1e-3) and np.allclose(s, s[::-1])
      and abs(bw['mu_L_weight'].sum() - 1) < 1e-12 and s[-1] > s[-2] > s[-3] > 0,
      f'(last three {np.round(s[-3:], 3).tolist()})')

# 7 --------------------------------------------------------------------------
tmp = Path(tempfile.mkdtemp(prefix='t34_'))
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
    root = tmp / 'T3.4'
    args = argparse.Namespace(data=str(csv), min_obs=5, out=str(root / 'syn'), layer='both',
                              methods=None, seed=42, causal_universe=True, resume=False)
    pv.run(args)
    ev = ProtocolV5Evaluator.from_csv(csv, dataset_min_obs=5, seed=42, robustness=True,
                                      causal_universe=True)
    ev.verbose = False
    r15 = tmp / 'T15' / 'syn' / 'protocol'
    r15.mkdir(parents=True)
    for m, rf in refs.items():
        ev.run_method(FastMethod(m, 64, 32, rf)).to_csv(r15 / f'{m}_protocol.csv', index=False)
    pv.collect(root, tmp / 'T15')
    ctrl = pd.read_csv(root / 'padding_control.csv')
    check('collect: 6 controls, all equal', len(ctrl) == 6 and (ctrl['status'] == 'equal').all(),
          f'({ctrl["status"].tolist()})')
    padf = pd.read_csv(root / 'syn' / 'pad' / 'WSPI' / 'protocol' / 'zero_protocol.csv')
    check('pad run covers windows 32..64',
          padf['window_id'].min() == 32 and padf['window_id'].max() == 64
          and padf['padded'].sum() == 32)
    sm = pd.read_csv(root / 'padding_summary.csv')
    n_exp = 3 * (5 + 5 * 2 + 1)
    check('summary rows: 3 methods x (5 ext + 5 pad x 2 subsets + none)', len(sm) == n_exp,
          f'({len(sm)} rows)')
    base = pd.read_csv(root / 'syn' / 'ext' / 'WSPI' / 'protocol' / 'symmetric_protocol.csv')
    a = sm[(sm.method == 'WSPI') & (sm.layer == 'ext') & (sm['mode'] == 'symmetric')].iloc[0]
    b = sm[(sm.method == 'WSPI') & (sm.layer == 'pad') & (sm['mode'] == 'reflect') & (sm.subset == 'all')].iloc[0]
    check('pad/reflect "all" == ext/symmetric (same windows and means)',
          a['n_windows'] == b['n_windows'] == len(base)
          and all(np.isclose(a[c], b[c], equal_nan=True) for c in sm.columns if c.endswith('_mean')))
    nn = sm[(sm['mode'] == 'none') & (sm.method == 'WSPI')].iloc[0]
    check('"none" = default run from window 64',
          nn['first_window'] == 64 and nn['n_windows'] == (base['window_id'] >= 64).sum()
          and nn['padded_share'] == 0)
    ps = pd.read_csv(root / 'padded_share.csv')
    check('padded_share: 3 methods, 32 padded windows each',
          len(ps) == 3 and (ps['padded_windows'] == 32).all())
    bad = pd.read_csv(r15 / 'WSPI_protocol.csv')
    bad.loc[bad.index[-1], 'rsi@10'] += 0.01
    bad.to_csv(r15 / 'WSPI_protocol.csv', index=False)
    pv.collect(root, tmp / 'T15')
    ctrl = pd.read_csv(root / 'padding_control.csv')
    st = ctrl.set_index(['layer', 'method'])['status']
    check('perturbed reference is detected (ext) and pad range not affected',
          st[('ext', 'WSPI')] == 'DIFFERENT' and st[('pad', 'WSPI')] == 'equal')
    only = sorted(p.suffix for p in root.rglob('*') if p.is_file())
    check('only CSV and JSON written', set(only) <= {'.csv', '.json'}, f'({set(only)})')
except Exception as e:  # pragma: no cover
    import traceback
    traceback.print_exc()
    check('end-to-end run on synthetic data', False, repr(e))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
