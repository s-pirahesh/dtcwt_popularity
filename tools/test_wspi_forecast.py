r"""
Unit checks for methods/wspi_forecast.py
========================================
 1. Reduction by switch: with phi_d = 0 every method gives exactly the WSPI
    score (equal floats), including windows of zeros, spikes and sparse rows.
 2. Reduction by gate: WSPI-FG gives exactly WSPI on every window where its
    backtest gate is 0, and differs from WSPI where the gate is positive.
 3. Gate by hand: the backtest origins, the targets and the weights of
    backtest_gate (spy forecaster, hand-computed sums).  No target or prefix
    reaches past the observed coefficients.
 4. Gate range: g and lam stay in [0, 1].
 5. Lemma 1: appending m copies of mu_L leaves mu_L unchanged.
 6. Two-sided bound for c = 4, m = 1..6: random search and extreme cases; both
    limits are attained; the clamped methods keep P_F / WSPI inside the limits.
 7. Coefficient step: the centre of the trend-band response to an impulse moves
    2^(J-1) = 4 slots per coefficient (not 2^J = 8).  Rule m = ceil(h / 2^(J-1)).
 8. WSPI-FA fallbacks (short, constant sequence, failed fit, non-finite forecast)
    and counters.
 9. MA64.
10. Argument checks and the set of fixed settings.

    python tools/test_wspi_forecast.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import dtcwt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from methods.wspi_assessment import WSPIAssessment                    # noqa: E402
from methods import wspi_forecast as wf                               # noqa: E402
from methods.wspi_forecast import (                                    # noqa: E402
    MovingAverageForecast, WSPIForecast, arma11_forecast, backtest_gate,
    bound_factor, clamp_forecast, coefficient_step, get_coefficient_domain_methods,
    n_forecast_coefficients, ratio_bounds, recency_mean, recency_weights)


def check(name, ok, detail=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")
    return bool(ok)


def signals(n_rows=60, W=64, seed=7):
    """Poisson demand with trend, daily cycle, end spike, sparse and zero rows."""
    rng = np.random.default_rng(seed)
    t = np.arange(W)
    rows = []
    for i in range(n_rows):
        kind = i % 6
        base = rng.gamma(2.0, 20.0)
        lam = np.clip(base * (1 + 0.01 * rng.normal() * t), 0.05, None)
        if kind == 1:
            lam = lam * (1 + 0.5 * np.sin(2 * np.pi * t / 24))
        x = rng.poisson(np.clip(lam, 0.05, None)).astype(float)
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


def main():
    warnings.filterwarnings('ignore')
    ok_all = True
    X = signals()
    wspi = WSPIAssessment()
    ref = np.array([wspi.assess_single(x) for x in X])

    # 1. reduction by switch (phi_d = 0), all six methods, horizons 1, 8, 24
    methods0 = {}
    for h in (1, 8, 24):
        for name, m in get_coefficient_domain_methods(h).items():
            m.phi_d = 0.0
            methods0[(name, h)] = m
    for (name, h), m in methods0.items():
        got = np.array([m.assess_single(x) for x in X])
        ok_all &= check(f'1 phi_d = 0 -> WSPI exactly   {name:<10} h={h:<2}', np.array_equal(got, ref),
                        f'max|diff|={np.abs(got - ref).max():.1e}')

    # 2. reduction by gate (WSPI-FG)
    pool = np.vstack([X, signals(120, seed=23)])
    for h in (1, 8, 24):
        fg = WSPIForecast(predictor='aryw', gate='backtest', horizon=h, clamp=4.0)
        g = np.array([fg.gate_value(x) for x in pool])
        sc = np.array([fg.assess_single(x) for x in pool])
        rf = np.array([wspi.assess_single(x) for x in pool])
        closed = g == 0.0
        ok_all &= check(f'2a gate closed -> WSPI exactly  h={h:<2}',
                        closed.sum() >= 5 and np.array_equal(sc[closed], rf[closed]),
                        f'{int(closed.sum())} of {len(g)} windows have g = 0')
        opened = g > 0.0
        ok_all &= check(f'2b gate open -> differs from WSPI h={h:<2}',
                        opened.sum() >= 5 and bool(np.all(sc[opened] != rf[opened])),
                        f'{int(opened.sum())} windows have g > 0')
        ok_all &= check(f'4  g in [0, 1]                  h={h:<2}', bool(np.all((g >= 0) & (g <= 1))))

    # 3. gate by hand with a spy forecaster
    a = np.array([5., 7, 6, 9, 11, 10, 12, 15, 14, 13, 17, 19, 18, 22, 21, 24])    # n = 16
    for m, origins in ((1, [14, 15]), (2, [13, 14]), (6, [9, 10])):
        seen = []

        def spy(prefix, steps, seen=seen):
            seen.append(len(prefix))
            return np.full(steps, prefix[-1] + 1.0)       # last value + 1, every step

        g = backtest_gate(a, m, spy, k=2, c=4.0)
        w = recency_weights(m)
        sf = smu = 0.0
        for t in origins:
            f = np.full(m, a[t - 1] + 1.0)                # clamp is inactive here
            ref_mu = recency_mean(a[:t])
            y = a[t:t + m]
            sf += float(np.sum(w * np.abs(f - y)))
            smu += float(np.sum(w * np.abs(ref_mu - y)))
        want = max(0.0, 1.0 - sf / smu)
        ok_all &= check(f'3 origins and sums by hand       m={m}',
                        seen == origins and abs(g - want) < 1e-12 and max(origins) + m <= len(a),
                        f'prefixes {seen}, g={g:.6f} vs {want:.6f}')
    short = backtest_gate(a[:5], 2, lambda p, s: np.full(s, p[-1]), k=2, c=4.0)
    ok_all &= check('3 short window -> no backtest -> g = 0', short == 0.0)
    ok_all &= check('3 constant window -> g = 0',
                    backtest_gate(np.full(16, 3.0), 2, wf.aryw_forecast, k=2, c=4.0) == 0.0)

    # 5. Lemma 1
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(2000):
        n = int(rng.integers(2, 40))
        s = rng.gamma(1.5, 100.0, n)
        mu = recency_mean(s)
        for m in range(1, 7):
            worst = max(worst, abs(recency_mean(np.concatenate([s, np.full(m, mu)])) - mu) / mu)
    ok_all &= check('5 Lemma 1: appending mu_L keeps mu_L', worst < 1e-12, f'max rel.diff={worst:.1e}')

    # 6. two-sided bound
    c = 4.0
    rng = np.random.default_rng(5)
    ok_bound, hit_up, hit_lo = True, 0.0, 1e9
    for _ in range(20000):
        n = int(rng.choice([8, 16, 16, 16, 32]))
        m = int(rng.integers(1, 7))
        s = rng.gamma(1.5, 100.0, n)
        if rng.random() < 0.3:
            s[:-1] = 0.0
        if rng.random() < 0.1:
            s[-1] = 0.0
        raw = s[-1] * np.exp(rng.uniform(-6, 6, m))
        f = clamp_forecast(raw, s[-1], c)
        if recency_mean(s) == 0.0:
            continue
        r = recency_mean(np.concatenate([s, f])) / recency_mean(s)
        lo, up = ratio_bounds(n, m, c)
        ok_bound &= (lo - 1e-12 <= r <= up + 1e-12) and up <= bound_factor(m, c) + 1e-12
    ok_all &= check('6a bound holds (20000 random cases)', ok_bound)
    for m in range(1, 7):
        n = 16
        up_seq = np.zeros(n)
        up_seq[-1] = 7.0
        r_up = recency_mean(np.concatenate([up_seq, clamp_forecast(np.full(m, 1e9), 7.0, c)])) / recency_mean(up_seq)
        lo_seq = np.concatenate([np.full(n - 1, 5.0), [0.0]])
        r_lo = recency_mean(np.concatenate([lo_seq, clamp_forecast(np.full(m, 3.0), 0.0, c)])) / recency_mean(lo_seq)
        lo, up = ratio_bounds(n, m, c)
        ok_all &= check(f'6b both limits attained          m={m}',
                        abs(r_up - up) < 1e-12 and abs(r_lo - lo) < 1e-12,
                        f'upper {r_up:.6f}/{up:.6f}  lower {r_lo:.6f}/{lo:.6f}')
    ok_all &= check('6c B(m, 4) = 4.5, 6.25, 7.890625 (m = 1, 2, 6)',
                    [bound_factor(m, 4.0) for m in (1, 2, 6)] == [4.5, 6.25, 7.890625])
    sig = signals(120, seed=19)
    for h in (1, 8, 24):
        for name in ('WSPI-F2', 'WSPI-FT', 'WSPI-FG', 'WSPI-FA'):
            mth = get_coefficient_domain_methods(h)[name]
            keep = slice(0, 120, 4) if name == 'WSPI-FA' else slice(0, 120)
            rr = np.array([mth.assess_single(x) / wspi.assess_single(x)
                           for x in sig[keep] if wspi.assess_single(x) > 0])
            lo, up = mth.ratio_bounds(16)
            ok_all &= check(f'6d P_F/WSPI inside the limits   {name:<8} h={h:<2}',
                            bool(np.all((rr >= lo - 1e-9) & (rr <= up + 1e-9))),
                            f'range [{rr.min():.3f}, {rr.max():.3f}] in [{lo:.3f}, {up:.3f}]')

    # 7. coefficient step and the m rule
    tr = dtcwt.Transform1d(biort='near_sym_a', qshift='qshift_a')
    pos = np.arange(16, 49)
    cen = []
    for s in pos:
        x = np.zeros(64)
        x[s] = 1.0
        a_ = np.abs(tr.forward(x, nlevels=3).lowpass.ravel())
        e = a_ ** 2
        cen.append((np.arange(len(a_)) * e).sum() / e.sum())
    step = 1.0 / np.polyfit(pos, cen, 1)[0]
    ok_all &= check('7a impulse response: step = 4 slots', 3.9 < step < 4.1 and coefficient_step(3) == 4,
                    f'estimated {step:.3f}')
    ms = [n_forecast_coefficients(h) for h in (1, 8, 24)]
    old = [int(np.ceil(h / 8)) for h in (1, 8, 24)]
    ok_all &= check('7b m = 1, 2, 6 for h = 1, 8, 24 (old rule 1, 1, 3)', ms == [1, 2, 6] and old == [1, 1, 3],
                    f'new {ms} old {old}')

    # 8. WSPI-FA fallbacks
    from unittest import mock
    rng = np.random.default_rng(2)

    class _Raises:
        def __init__(self, *a, **k):
            pass

        def fit(self, *a, **k):
            raise RuntimeError('no convergence')

    class _Nan(_Raises):
        def fit(self, *a, **k):
            return mock.Mock(forecast=lambda steps: np.full(steps, np.nan))

    seq16 = rng.gamma(3.0, 5.0, 16)
    with mock.patch('statsmodels.tsa.arima.model.ARIMA', _Raises):
        fit_error = arma11_forecast(seq16, 2)
    with mock.patch('statsmodels.tsa.arima.model.ARIMA', _Nan):
        nan_fc = arma11_forecast(seq16, 2)
    fb = [arma11_forecast(np.arange(5.0), 1)[1], arma11_forecast(np.full(16, 4.0), 2)[1],
          fit_error[1], nan_fc[1]]
    fc, _ = arma11_forecast(np.arange(5.0), 2)
    ok_all &= check('8a fallback: short, constant, failed fit, non-finite forecast',
                    all(fb) and np.all(fc == 4.0) and np.all(fit_error[0] == seq16[-1])
                    and np.all(nan_fc[0] == seq16[-1]), str(fb))
    fc, fell = arma11_forecast(seq16, 6)
    ok_all &= check('8b regular fit: 6 finite values', fc.shape == (6,) and np.isfinite(fc).all(), f'fell_back={fell}')
    fa = get_coefficient_domain_methods(1)['WSPI-FA']
    for x in X[:12]:
        fa.assess_single(x)
    ok_all &= check('8c counters', fa.n_forecasts == 12 and 0 <= fa.n_fallbacks <= 12,
                    f'{fa.n_fallbacks} fallbacks in {fa.n_forecasts} forecasts')
    fa.reset_counters()
    ok_all &= check('8d reset', fa.n_forecasts == 0 and fa.n_fallbacks == 0)

    # 9. MA64
    ma = MovingAverageForecast(window=64, horizon=8)
    x = np.arange(100.0)
    ok_all &= check('9a MA64 = mean of the last 64 slots times h',
                    abs(ma.assess_single(x) - np.mean(x[-64:]) * 8) < 1e-12 and ma.name == 'MA64')
    ok_all &= check('9b MA64 short and empty series',
                    ma.assess_single(np.array([2.0, 4.0])) == 24.0 and ma.assess_single(np.array([])) == 0.0)

    # 10. arguments and fixed settings
    bad = []
    for kw in (dict(predictor='x'), dict(gate='x'), dict(clamp=0.5), dict(phi_d=1.5),
               dict(gate='structural', gate_gamma=0.0), dict(gate='backtest', backtest_k=0)):
        try:
            WSPIForecast(**kw)
            bad.append(kw)
        except ValueError:
            pass
    ok_all &= check('10a invalid arguments rejected', not bad, str(bad))
    sets = get_coefficient_domain_methods(1)
    fixed = (list(sets) == ['WSPI-F', 'WSPI-F-YW', 'WSPI-F2', 'WSPI-FT', 'WSPI-FG', 'WSPI-FA']
             and sets['WSPI-FG'].backtest_k == 2 and sets['WSPI-FG'].gate == 'backtest'
             and sets['WSPI-F2'].gate_gamma == 0.3 and sets['WSPI-F'].clamp is None
             and sets['WSPI-F-YW'].clamp is None
             and all(sets[n].clamp == 4.0 for n in ('WSPI-F2', 'WSPI-FT', 'WSPI-FG', 'WSPI-FA'))
             and all(m.phi_d == 1.0 for m in sets.values())
             and len(get_coefficient_domain_methods(1, include_arma=False)) == 5)
    ok_all &= check('10b fixed settings (k = 2, c = 4, gamma = 0.3, phi_d = 1)', fixed)

    print('\nALL OK' if ok_all else '\nSOME FAILED')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
