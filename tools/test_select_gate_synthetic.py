r"""
Unit checks for experiments/select_gate_synthetic.py
====================================================
  1. Identity with the pre-registered run: the script is run once into a
     temporary folder and compared with results/prediction/gate_selection/
     (selection_summary.csv, selection_cells.csv, diagnostics.csv).  The
     configurations, the chosen configuration and the group sizes must be
     equal, the numbers equal to 1e-9.  If those files do not exist (a clone
     of the repository has no results/), this check is skipped and says so.
  2. Alignment of candidate C.  For a linear signal, the trend-band magnitudes
     of the window cut by 8s slots (and reflect-padded on the left) are the
     magnitudes of the full window moved by 2s coefficients:
     b[j + 2s] == a[j] in the interior (not within 2 coefficients of the left
     end and 3 of the right end, where the boundary of the transform acts).
     A shift of 2s + 1 does not align (wrong tree): its error is more than ten
     times larger.
  3. Rules of candidate C: the number of origins, the true coefficients lie
     inside the window (2s >= m), the cut window keeps half real data.
  4. Gates: g in [0, 1]; a constant window gives g = 0 for both sources.

    python tools/test_select_gate_synthetic.py
"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import dtcwt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import experiments.select_gate_synthetic as S                         # noqa: E402
from tools.prediction_protocol import CoefFeatures                    # noqa: E402

REFERENCE = ROOT / 'results' / 'prediction' / 'gate_selection'
FAILED = []


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)


def test_identity():
    files = ('selection_summary.csv', 'selection_cells.csv', 'diagnostics.csv')
    if not all((REFERENCE / f).exists() for f in files):
        print(f'SKIP identity with the pre-registered run: {REFERENCE} not found')
        return
    with tempfile.TemporaryDirectory() as td:
        summ = S.run(Path(td))
        for f in files:
            a = pd.read_csv(Path(td) / f, float_precision='round_trip')
            b = pd.read_csv(REFERENCE / f, float_precision='round_trip')
            num = a.select_dtypes('number').columns
            same_shape = a.shape == b.shape and list(a.columns) == list(b.columns)
            same_text = a.drop(columns=num).equals(b.drop(columns=num))
            d = float(np.nanmax(np.abs(a[num].to_numpy() - b[num].to_numpy()))) if same_shape else np.inf
            check(f'identity with the pre-registered run: {f}', same_shape and same_text and d < 1e-9,
                  f'max abs. diff {d:.1e}')
        chosen = summ.loc[summ['chosen'], 'config'].tolist()
        check('chosen configuration is W2', chosen == ['W2'], str(chosen))


def test_alignment():
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    t = np.arange(S.WINDOW, dtype=float)
    for label, x in (('rising', 3 + 0.5 * t), ('falling', 100 - 0.7 * t)):
        X = x[None, :]
        a = S.magnitudes(tr, X)[0]
        n = len(a)
        for s in (1, 2, 3, 4):
            b = S.magnitudes(tr, X[:, :S.WINDOW - 8 * s])[0]
            js = np.arange(2, n - 2 * s - 3)                  # interior of the shifted pair
            rel = np.abs(b[js + 2 * s] - a[js]) / a[js]
            rel_odd = np.abs(b[js + 2 * s + 1] - a[js]) / a[js]
            check(f'candidate C alignment, {label} ramp, s={s}: b[j+2s] == a[j]',
                  rel.max() < 0.01 and rel_odd.min() > 10 * rel.max(),
                  f'(interior j={js[0]}..{js[-1]}: max rel. diff {rel.max():.1e}; '
                  f'min rel. diff of the shift 2s+1: {rel_odd.min():.1e})')


def test_rules_of_C():
    n = 16
    for m in (1, 2, 6):
        for k in S.KS:
            s0 = S.first_shift(m)
            ke = S.k_eff_C(m, k)
            ss = range(s0, s0 + ke)
            ok = (ke >= 1 and all(2 * s >= m for s in ss)                 # truth inside the window
                  and all(2 * s + m <= n for s in ss)                     # a[n-2s : n-2s+m] exists
                  and all(S.WINDOW - 8 * s >= S.WINDOW // 2 for s in ss))   # half real data
            check(f'rules of candidate C  m={m} k={k}', ok, f'(s0={s0}, origins {ke})')
    check('origins of candidate C: m=6 has at most 2, m=1 has k',
          [S.k_eff_C(6, k) for k in S.KS] == [2, 2, 2] and
          [S.k_eff_C(1, k) for k in S.KS] == [2, 3, 4])


def test_gates():
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    x = S.make_group('burst', 200, 11)[:, :S.WINDOW]
    a = CoefFeatures(x, tr, S.LEVEL).a
    const = np.full((3, S.WINDOW), 7.0)
    ac = CoefFeatures(const, tr, S.LEVEL).a
    for m in (1, 2, 6):
        for k in S.KS:
            gw = S.gate_W(a, m, k)
            gc = S.gate_C(x, a, m, k, tr)
            check(f'gates in [0, 1]  m={m} k={k}',
                  gw.min() >= 0 and gw.max() <= 1 and gc.min() >= 0 and gc.max() <= 1)
            check(f'constant window gives g = 0  m={m} k={k}',
                  np.all(S.gate_W(ac, m, k) == 0) and np.all(S.gate_C(const, ac, m, k, tr) == 0))
    try:
        S.gate_W(a, 6, 8)
        check('source W rejects k > n - m - 3', False)
    except ValueError:
        check('source W rejects k > n - m - 3', True)


def main():
    warnings.filterwarnings('ignore')
    test_identity()
    test_alignment()
    test_rules_of_C()
    test_gates()
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
