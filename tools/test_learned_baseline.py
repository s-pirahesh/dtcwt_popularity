r"""
Unit checks for the learned baselines (GBM, GBM+W) and the NLMS convergence script
==================================================================================
Covers ``tools/learned_baseline.py``, the learned methods in
``tools/run_prediction.py``, ``iter_eligible_windows`` of
``tools/prediction_protocol.py`` and ``experiments/nlms_convergence.py``.

  1. Features: sums, slope and zero share against direct computations; the
     wavelet features equal mu_L, R and W_E of WSPI (WSPI = mu_L exp(R - W_E)).
  2. iter_eligible_windows gives exactly the windows and item counts of the
     evaluator (h = 1, 8).
  3. Split: first 30 % tuning (split_ids); training windows end h - 1 windows
     before the first test window.
  4. No leakage: changing every count from the first test window on does not
     change the trained model (same predictions on any input).
  5. Sampling: at most max_rows rows, the same rows for GBM and GBM+W, the
     same model on a second training (deterministic).
  6. tools/run_prediction.py with GBM and GBM+W: protocol files, the training
     summary in metadata/, the scores of the run equal the scorer.
  7. NLMS convergence: the row-wise a-priori predictions equal nlms_walk; the
     AR(2) one-step forecast equals aryw_forecast; output files and columns.

    python tools/test_learned_baseline.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fast_evaluator import FastMethod                          # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator                    # noqa: E402
from experiments import nlms_convergence as NC                            # noqa: E402
from methods.wspi_forecast import aryw_forecast, nlms_walk                 # noqa: E402
from tools import learned_baseline as L                                   # noqa: E402
from tools import prediction_protocol as P                                # noqa: E402
from tools import run_prediction                                          # noqa: E402
from tools.run_param_grid import split_ids                                # noqa: E402
from tools.test_prediction import make_dataset, signals                   # noqa: E402

FAILED = []


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def dataset(n_items=30, n_slots=260, seed=5):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(2.0, 8.0, n_items)
    t = np.arange(n_slots)
    trend = 1 + 0.004 * rng.normal(size=n_items)[:, None] * t
    pres = rng.random((n_items, n_slots)) < 0.9
    pres[0] = True
    counts = rng.poisson(np.clip(lam[:, None] * trend, 0.1, None)) * pres
    ts = pd.date_range('2025-01-01', periods=n_slots, freq='h')
    ii, ss = np.nonzero(pres)
    df = pd.DataFrame({'timestamp': ts[ss], 'item_id': [f'i{j:02d}' for j in ii],
                       'count': counts[ii, ss].astype(float)})
    return df


def evaluator(df, h):
    return ProtocolV5Evaluator(df, dataset_min_obs=24, seed=42, robustness=False,
                               causal_universe=True, horizon=h, verbose=False)


def test_features():
    X = signals(n_rows=50)
    F = L.value_features(X)
    ok = all(np.allclose(F[:, j], np.log1p(X[:, -n:].sum(axis=1)))
             for j, n in enumerate(L.SUM_LENGTHS))
    check('features: log1p of the last-n sums', ok)
    sl = np.array([np.polyfit(np.arange(16), x[-16:], 1)[0] for x in X])
    check('features: slope of the last 16 slots', np.allclose(F[:, 7], sl))
    check('features: zero share', np.allclose(F[:, 8], (X == 0).mean(axis=1)))
    W = L.WaveletFeatures()(X)
    wspi = P.make_v5_wspi(window=64, level=3)(X)
    check('wavelet features: mu_L exp(R - W_E) == WSPI',
          np.allclose(W[:, 0] * np.exp(W[:, 1] - W[:, 2]), wspi, rtol=1e-12))
    short = signals(n_rows=5, W=20)
    Fs = L.value_features(short)
    check('features: short window uses all slots',
          np.allclose(Fs[:, 6], np.log1p(short.sum(axis=1))) and Fs.shape == (5, 9))
    check('feature names', len(L.feature_names('GBM')) == 9 and len(L.feature_names('GBM+W')) == 12)


def ref_windows(fe):
    def f(X):
        return X.sum(axis=1)
    f.__name__ = 'sum'
    res = fe.run_method(FastMethod('SUM', 64, 32, f))
    return list(res['window_id']), list(res['num_items'])


def test_windows_and_split(df):
    for h in (1, 8):
        fe = evaluator(df, h)
        ids, n = ref_windows(fe)
        got = [(k, len(idx)) for k, idx in P.iter_eligible_windows(fe)]
        check(f'eligible windows == evaluator windows  h={h}',
              [k for k, _ in got] == ids and [c for _, c in got] == n, f'({len(ids)} windows)')
        tune, test, train = L.split_windows(ids, h)
        t2, s2 = split_ids(ids)
        check(f'split == split_ids (30 %)               h={h}',
              tune == t2 and test == s2 and len(tune) == int(np.floor(0.3 * len(ids))))
        check(f'training windows end before the test part  h={h}',
              all(k + h - 1 < test[0] for k in train) and
              len(tune) - len(train) == sum(1 for k in tune if k + h - 1 >= test[0]))


def test_no_leakage(df):
    h = 8
    fe = evaluator(df, h)
    ids = [k for k, _ in P.iter_eligible_windows(fe)]
    _, test, _ = L.split_windows(ids, h)
    sc1, info1 = L.train_learned(fe, ['GBM', 'GBM+W'], h)
    # change every count from the first test slot on
    ts = pd.to_datetime(df['timestamp'])
    first = fe.start + test[0] * fe.slot
    df2 = df.copy()
    late = ts >= first
    df2.loc[late, 'count'] = df2.loc[late, 'count'] * 7 + 3
    fe2 = evaluator(df2, h)
    sc2, info2 = L.train_learned(fe2, ['GBM', 'GBM+W'], h)
    X = signals(n_rows=80)
    ok = all(np.array_equal(sc1[n](X), sc2[n](X)) for n in ('GBM', 'GBM+W'))
    check('no leakage: counts of the test part do not change the model', ok,
          f"({info1['GBM']['n_rows_used']} training rows)")
    check('training summary', info1['GBM']['first_test_window'] == test[0] and
          info1['GBM']['last_train_window'] + h - 1 < test[0] and not info1['GBM']['sampled'])
    sc3, _ = L.train_learned(fe, ['GBM'], h)
    check('training is deterministic', np.array_equal(sc3['GBM'](X), sc1['GBM'](X)))
    check('scores are finite and >= 0', all(np.isfinite(sc1[n](X)).all() and
                                            (sc1[n](X) >= 0).all() for n in sc1))


def test_sampling(df):
    fe = evaluator(df, 1)
    ids = [k for k, _ in P.iter_eligible_windows(fe)]
    _, _, train = L.split_windows(ids, 1)
    feats, y, info = L.training_rows(fe, train, {'GBM': False, 'GBM+W': True}, max_rows=300)
    check('sampling: at most max_rows rows', len(y) == 300 and info['sampled'] and
          info['n_rows_available'] > 300)
    check('sampling: same rows for GBM and GBM+W',
          np.array_equal(feats['GBM'], feats['GBM+W'][:, :9]))
    feats2, y2, _ = L.training_rows(fe, train, {'GBM': False}, max_rows=300)
    check('sampling: deterministic (seed 42)', np.array_equal(feats2['GBM'], feats['GBM']) and
          np.array_equal(y2, y))
    full, yf, inf2 = L.training_rows(fe, train, {'GBM': False})
    check('no sampling below max_rows', not inf2['sampled'] and len(yf) == inf2['n_rows_available'])


def test_runner(df):
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / 'd.csv'
        df.to_csv(data, index=False)
        out = Path(td) / 'out'
        run_prediction.main(['--data', str(data), '--min-obs', '24', '--horizons', '8',
                             '--methods', 'WSPI', 'GBM', 'GBM+W', 'SMA16', 'WSPI-FS2',
                             '--bound-check', '--out', str(out)])
        h8 = out / 'h8'
        files = all((h8 / 'protocol' / f'{m}_protocol.csv').exists()
                    for m in ('WSPI', 'GBM', 'GBM+W', 'SMA16', 'WSPI-FS2'))
        check('runner: protocol files of the learned and control methods', files)
        info = json.loads((h8 / 'metadata' / 'learned_GBM_W.json').read_text(encoding='utf-8'))
        check('runner: training summary in metadata', info['method'] == 'GBM+W' and
              info['horizon'] == 8 and len(info['features']) == 12)
        bc = pd.read_csv(h8 / 'bound_check.csv')
        check('runner: bound check of WSPI-FS2', list(bc['method']) == ['WSPI-FS2'] and
              (bc['status'] == 'OK').all())
        g = pd.read_csv(h8 / 'protocol' / 'GBM_protocol.csv')
        w = pd.read_csv(h8 / 'protocol' / 'WSPI_protocol.csv')
        check('runner: GBM has the windows of WSPI', list(g['window_id']) == list(w['window_id'])
              and list(g['num_items']) == list(w['num_items']))
        st = pd.read_csv(h8 / 'scorer_timing.csv')
        check('runner: GBM is timed', set(st['method']) == {'WSPI', 'GBM', 'GBM+W', 'SMA16',
                                                           'WSPI-FS2'})
        # the run's scores equal a scorer trained on the same evaluator
        fe = evaluator(df, 8)
        sc, _ = L.train_learned(fe, ['GBM'], 8)
        res = fe.run_method(FastMethod('GBM', 64, 32, sc['GBM']))
        a = res[['ndcg@10', 'spearman_rho', 'mae']].to_numpy(float)
        b = g[['ndcg@10', 'spearman_rho', 'mae']].to_numpy(float)
        check('runner: GBM metrics == direct evaluator run', np.allclose(a, b, rtol=1e-12))


def test_nlms_convergence():
    X = signals(n_rows=60)
    tr = P.dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    F = P.CoefFeatures(P._pad_left_reflect(X, 64), tr)
    a = F.a
    rows = NC.nlms_apriori_rows(a)
    ref = np.vstack([nlms_walk(x)[2] for x in a])
    check('NLMS a-priori rows == nlms_walk', rows.shape == ref.shape and np.allclose(rows, ref))
    ts, en, ea = NC.window_errors(X, tr, 64)
    check('positions t = 3..15', list(ts) == list(range(3, 16)))
    keep = P.recency_mean_rows(a) > 1e-12
    mu = P.recency_mean_rows(a)[keep]
    ar = np.array([[aryw_forecast(x[:t], 1)[0] for t in ts] for x in a[keep]])
    check('AR(2) error == aryw_forecast', np.allclose(ea, np.abs(ar - a[keep][:, ts]) / mu[:, None]))
    df, _, _ = make_dataset(n_items=14, n_slots=150)
    with tempfile.TemporaryDirectory() as td:
        data = Path(td) / 'd.csv'
        df.to_csv(data, index=False)
        out = Path(td) / 'n'
        rc = NC.main(['--data', str(data), '--min-obs', '1', '--out', str(out)])
        c = pd.read_csv(out / 'h1' / 'nlms_convergence.csv')
        lt = pd.read_csv(out / 'h1' / 'nlms_convergence_late.csv')
        check('NLMS script: outputs', rc == 0 and set(c['method']) == {'NLMS', 'AR2-YW'} and
              len(c) == 26 and len(lt) == 2 and
              (out / 'h1' / 'metadata' / 'nlms_convergence_run.json').exists())
        check('NLMS script: quantiles ordered',
              bool((c['q0.1'] <= c['q0.5']).all() and (c['q0.5'] <= c['q0.9']).all()))
        rc2 = NC.main(['--data', str(data), '--min-obs', '1', '--out', str(out), '--resume'])
        check('NLMS script: --resume skips a complete output', rc2 == 0)
        check('NLMS script: only horizon 1',
              NC.main(['--data', str(data), '--min-obs', '1', '--out', str(out),
                       '--horizons', '8']) == 2)


def main():
    warnings.filterwarnings('ignore')
    df = dataset()
    test_features()
    test_windows_and_split(df)
    test_no_leakage(df)
    test_sampling(df)
    test_runner(df)
    test_nlms_convergence()
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
