r"""
Unit checks for the prediction protocol
=======================================
Covers ``tools/prediction_protocol.py``, ``tools/run_prediction.py`` and the
``horizon`` parameter of ``evaluation/protocol_v5.py``.

  1. Every batch scorer reproduces its per-item reference class
     (methods/wspi_forecast.py, methods/forecasting_baselines.py) on full
     64-slot windows, for the horizons 1, 8 and 24 (m = 1, 2, 6 coefficients,
     step 2^(J-1)).  The forecast length really follows the rule: the scores
     at h = 8 and h = 24 differ from the scores at h = 1.
  2. WSPI-FG gate: batch gate == class gate; where g = 0 the score is exactly
     WSPI, where g > 0 it differs.
  3. TimedScorer: same output, counts calls and rows, keeps the name; the
     WSPI-FA fallback counters are reported.
  4. Multi-slot ground truth of the evaluator: for h = 1, 8, 24 the window set
     and the metrics of every window equal an independent computation from
     the raw counts (truth = sum of the h slots k..k+h-1, missing rows are
     zeros).  h = 1 is identical to the call without the parameter.
  5. tools/run_prediction.py: one folder per horizon, scorer_timing.csv,
     horizon in the metadata; the h = 1 results equal a direct evaluator run.

    python tools/test_prediction.py
"""
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.metrics import calculate_diagnostics                      # noqa: E402
from evaluation.protocol_v5 import (ProtocolV5Evaluator, coverage_stable,  # noqa: E402
                                    ndcg_stable, stable_order)
from evaluation.fast_evaluator import FastMethod                          # noqa: E402
from methods.forecasting_baselines import (ARYWForecast, HoltForecast,    # noqa: E402
                                           PersistenceForecast)
from methods.wspi_forecast import (MovingAverageForecast,                  # noqa: E402
                                   get_coefficient_domain_methods,
                                   n_forecast_coefficients)
from tools import prediction_protocol as P                                # noqa: E402
from tools import run_prediction                                          # noqa: E402

HORIZONS = (1, 8, 24)
FAILED = []


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        FAILED.append(name)
    return ok


def signals(n_rows=360, W=64, seed=7):
    """Poisson demand with trend, daily cycle, spikes, sparse and all-zero rows."""
    rng = np.random.default_rng(seed)
    t = np.arange(W)
    rows = []
    for i in range(n_rows):
        kind = i % 6
        base = rng.gamma(2.0, 20.0)
        lam = base * (1 + 0.01 * rng.normal() * t)
        if kind == 1:
            lam = lam * (1 + 0.5 * np.sin(2 * np.pi * t / 24))
        lam = np.clip(lam, 0.05, None)
        x = rng.poisson(lam).astype(float)
        if kind == 2:
            x[-rng.integers(1, 4):] += base * rng.uniform(5, 30)
        if kind == 3:
            x[rng.random(W) < 0.7] = 0
        if kind == 4 and i % 12 == 4:
            x[:] = 0
        if kind == 5:
            x[rng.integers(0, W)] += base * 20
        rows.append(x)
    return np.vstack(rows)


# =============================================================================
# 1-3. Scorers against the reference classes
# =============================================================================

def test_scorers_against_classes(X):
    for h in HORIZONS:
        refs = get_coefficient_domain_methods(horizon=h, include_arma=True)
        refs.update({
            'Persistence': PersistenceForecast(horizon=h),
            'Holt': HoltForecast(horizon=h),
            'ARYW': ARYWForecast(order=2, horizon=h),
            'MA64': MovingAverageForecast(window=64, horizon=h),
        })
        for name, obj in refs.items():
            rows = X[:48] if name == 'WSPI-FA' else X          # ARMA fits are slow
            fast = P.build_prediction_methods(horizon=h, names=[name])[name].scorer(rows)
            slow = np.array([obj.assess_single(x) for x in rows])
            rel = np.abs(fast - slow) / np.maximum(np.abs(slow), 1e-9)
            check(f'scorer == class  h={h:<2} {name:<12}', rel.max() < 1e-8,
                  f'max rel. diff {rel.max():.2e}')


def test_rule_m_is_followed(X):
    check('m = ceil(h / 2^(J-1)) gives 1, 2, 6',
          [n_forecast_coefficients(h) for h in HORIZONS] == [1, 2, 6])
    for name in ('WSPI-F-YW', 'WSPI-F2', 'WSPI-FG'):
        s = {h: P.build_prediction_methods(horizon=h, names=[name])[name].scorer(X)
             for h in HORIZONS}
        check(f'{name}: scores at h=8 and h=24 differ from h=1',
              not np.allclose(s[8], s[1]) and not np.allclose(s[24], s[1]) and
              not np.allclose(s[24], s[8]))


def test_gate(X):
    for h in HORIZONS:
        obj = get_coefficient_domain_methods(horizon=h, include_arma=False)['WSPI-FG']
        slow = np.array([obj.gate_value(x) for x in X])
        _, _, _, fast = P.gate_features(X, horizon=h)
        d = np.abs(fast - slow).max()
        check(f'FG gate == class gate  h={h:<2}', d < 1e-10, f'max abs. diff {d:.2e}')
        check(f'FG gate in [0, 1]       h={h:<2}', fast.min() >= 0 and fast.max() <= 1)
        wspi = P.build_prediction_methods(horizon=h, names=['WSPI'])['WSPI'].scorer(X)
        fg = P.build_prediction_methods(horizon=h, names=['WSPI-FG'])['WSPI-FG'].scorer(X)
        zero = fast == 0
        check(f'FG == WSPI where g = 0  h={h:<2}',
              zero.any() and np.array_equal(fg[zero], wspi[zero]),
              f'({int(zero.sum())} of {len(X)} rows)')
        check(f'FG != WSPI where g > 0  h={h:<2}',
              (~zero).any() and bool(np.any(fg[~zero] != wspi[~zero])))


def test_timed_scorer(X):
    inner = P.make_coef_scorer('WSPI-F2', 8)
    t = P.TimedScorer(inner)
    out1 = t(X[:50])
    out2 = t(X[50:80])
    check('TimedScorer: same output', np.array_equal(np.concatenate([out1, out2]),
                                                      inner(X[:80])))
    st = t.stats()
    check('TimedScorer: calls and rows', st['calls'] == 2 and st['rows'] == 80)
    check('TimedScorer: seconds >= 0 and name kept',
          st['seconds'] > 0 and t.__name__ == inner.__name__)
    fa = P.TimedScorer(P.make_coef_scorer('WSPI-FA', 1))
    Z = np.vstack([np.full((1, 64), 5.0), X[:11]])           # constant row: fallback
    fa(Z)
    st = fa.stats()
    check('TimedScorer: WSPI-FA fits and fallbacks',
          st['n_fits'] == 12 and 1 <= st['n_fallbacks'] <= 12 and
          0 < st['fallback_share'] <= 1, f"({st['n_fallbacks']} of {st['n_fits']})")
    check('TimedScorer: no counters for other methods',
          'n_fits' not in P.TimedScorer(inner).stats())


# =============================================================================
# 4. Multi-slot ground truth of the evaluator
# =============================================================================

def make_dataset(n_items=14, n_slots=150, seed=3):
    """Dense hourly counts with missing rows (= zeros); item 0 is always present.
    Returns the long table and the dense count / presence matrices."""
    rng = np.random.default_rng(seed)
    pres_prob = np.array([1.0] + [0.9, 0.85, 0.8, 0.7, 0.6] + [0.95] * 4 + [0.4, 0.3, 0.5, 0.45])[:n_items]
    lam = rng.gamma(2.0, 10.0, n_items)
    pres = rng.random((n_items, n_slots)) < pres_prob[:, None]
    pres[0, :] = True
    counts = (rng.poisson(lam[:, None], (n_items, n_slots)) + 1) * pres
    ts = pd.date_range('2025-03-01', periods=n_slots, freq='h')
    ii, ss = np.nonzero(pres)
    df = pd.DataFrame({'timestamp': ts[ss], 'item_id': [f'i{j:02d}' for j in ii],
                       'count': counts[ii, ss].astype(float)})
    return df, counts.astype(float), pres


def window_sum_scorer(X):
    return X[:, -8:].sum(axis=1)


def independent_truth_check(h, df, counts, pres, W=64, min_obs=32, universe_min=1):
    n_items, n_slots = counts.shape
    fe = ProtocolV5Evaluator(df, dataset_min_obs=universe_min, seed=42, robustness=False,
                             causal_universe=True, horizon=h, verbose=False)
    window_sum_scorer.__name__ = 'window_sum'
    fm = FastMethod('SUM', W, min_obs, window_sum_scorer)
    res = fe.run_method(fm)
    # the evaluator orders items by id; ids are i00.. so the order is the row order
    expected_ids = list(range(min_obs, n_slots - h + 1))   # k + h - 1 <= last slot (n_slots - 1)
    check(f'ground truth h={h:<2}: window set', list(res['window_id']) == expected_ids,
          f'{len(res)} windows, last {int(res["window_id"].max())}, expected last {expected_ids[-1]}')
    ok_all = True
    for _, r in res.iterrows():
        k = int(r['window_id'])
        lo = max(k - W, 0)
        elig = pres[:, lo:k].sum(axis=1) >= min_obs
        elig &= counts[:, :k].sum(axis=1) >= universe_min
        idx = np.where(elig)[0]
        sc = window_sum_scorer(counts[idx, lo:k])
        truth = counts[idx, k:k + h].sum(axis=1)
        order = stable_order(sc)
        dg = calculate_diagnostics(sc, truth)
        ok = (int(r['num_items']) == len(idx)
              and np.isclose(r['ndcg@10'], ndcg_stable(order, truth, 10), rtol=0, atol=1e-12)
              and np.isclose(r['coverage@10'], coverage_stable(order, truth, 10), rtol=0, atol=1e-12)
              and np.isclose(r['spearman_rho'], dg['spearman_rho'], rtol=0, atol=1e-12,
                             equal_nan=True))
        ok_all &= bool(ok)
    check(f'ground truth h={h:<2}: metrics of every window equal the raw-count computation', ok_all)
    return res


def test_ground_truth():
    df, counts, pres = make_dataset()
    res = {h: independent_truth_check(h, df, counts, pres) for h in HORIZONS}
    base = ProtocolV5Evaluator(df, dataset_min_obs=1, seed=42, robustness=False,
                               causal_universe=True, verbose=False)
    r0 = base.run_method(FastMethod('SUM', 64, 32, window_sum_scorer))
    check('h=1 explicit == without the parameter', r0.equals(res[1]))
    j = res[8].merge(res[1], on='window_id', suffixes=('_8', '_1'))
    check('h=8 truth differs from h=1 truth',
          bool(np.any(~np.isclose(j['ndcg@10_8'], j['ndcg@10_1']))))
    for bad in (0, -3):
        try:
            ProtocolV5Evaluator(df, dataset_min_obs=1, horizon=bad, verbose=False)
            check(f'horizon={bad} is rejected', False)
        except ValueError:
            check(f'horizon={bad} is rejected', True)


# =============================================================================
# 5. Runner
# =============================================================================

def test_runner():
    df, _, _ = make_dataset()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        data = td / 'toy.csv'
        df.to_csv(data, index=False)
        out = td / 'out'
        run_prediction.main(['--data', str(data), '--min-obs', '1', '--out', str(out),
                             '--horizons', '1', '8', '--methods', 'WSPI', 'WSPI-FG', 'MA64',
                             '--gate-profile'])
        for h in (1, 8):
            d = out / f'h{h}'
            check(f'runner h={h}: protocol files',
                  all((d / 'protocol' / f'{m}_protocol.csv').exists()
                      for m in ('WSPI', 'WSPI-FG', 'MA64')))
            tm = pd.read_csv(d / 'scorer_timing.csv')
            check(f'runner h={h}: scorer_timing.csv',
                  list(tm['method']) == ['WSPI', 'WSPI-FG', 'MA64'] and
                  (tm['rows'] > 0).all() and (tm['scorer_seconds'] > 0).all() and
                  (tm['horizon'] == h).all())
            meta = pd.read_json(d / 'metadata' / 'protocol_v5_run.json', typ='series')
            check(f'runner h={h}: horizon in the metadata',
                  int(meta['horizon']) == h and
                  int(meta['n_forecast_coefficients']) == n_forecast_coefficients(h))
            gp = pd.read_csv(d / 'gate_profile.csv')
            check(f'runner h={h}: gate profile windows follow the horizon',
                  gp['window_id'].max() <= 150 - h and 'fg_share_zero' in gp.columns)
        direct = ProtocolV5Evaluator(df, dataset_min_obs=1, seed=42, robustness=True,
                                     causal_universe=True, verbose=False)
        ref = direct.run_method(P.build_prediction_methods(1, ['WSPI-FG'])['WSPI-FG'])
        # the default C parser of read_csv is not exact to the last digit
        got = pd.read_csv(out / 'h1' / 'protocol' / 'WSPI-FG_protocol.csv',
                          float_precision='round_trip')
        check('runner h=1 == direct evaluator run (WSPI-FG, all columns)',
              np.allclose(ref.drop(columns=['method', 'padded', 'ties_top21']).to_numpy(float),
                          got.drop(columns=['method', 'padded', 'ties_top21']).to_numpy(float),
                          rtol=0, atol=0, equal_nan=True) and
              list(ref['padded']) == list(got['padded']))


def main():
    warnings.filterwarnings('ignore')
    X = signals()
    test_scorers_against_classes(X)
    test_rule_m_is_followed(X)
    test_gate(X)
    test_timed_scorer(X)
    test_ground_truth()
    test_runner()
    print()
    if FAILED:
        print(f'FAILED: {len(FAILED)} check(s)')
        for f in FAILED:
            print('  -', f)
        sys.exit(1)
    print('ALL OK')


if __name__ == '__main__':
    main()
