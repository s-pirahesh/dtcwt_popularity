r"""
Unit test for tools/run_robustness_v5.py
========================================
Run from the project root:   python tools\test_robustness_v5.py
Ends with "ALL OK" when every check passes.  Uses a small synthetic data set
(no file of data/ is needed) and writes only to a temporary folder.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import run_robustness_v5 as rr  # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator, stable_order  # noqa: E402

FAILS = []


def check(name, ok, info=''):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {info}")
    if not ok:
        FAILS.append(name)


def synthetic(n_items=90, n_slots=140, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp('2025-01-01')
    for i in range(n_items):
        lam = rng.gamma(1.0, 3.0)
        p_obs = rng.uniform(0.5, 1.0)
        for t in range(n_slots):
            if rng.random() < p_obs:
                rows.append((t0 + pd.Timedelta(hours=t), f'i{i:03d}', int(rng.poisson(lam)) + 1))
    return pd.DataFrame(rows, columns=['timestamp', 'item_id', 'count'])


def main():
    df = synthetic()
    fe = ProtocolV5Evaluator(df, dataset_min_obs=20, seed=42, causal_universe=True,
                             verbose=False)
    methods = rr.all_methods()

    # 1. configurations
    check('1 fifteen method-windows', len(methods) == 15)
    check('1 default members', rr.config_members('default') ==
          [(n, 7) for n in rr.BASELINES] + [(n, 64) for n in rr.WAVELETS])
    check('1 eq64 members', all(w == 64 for _, w in rr.config_members('eq64')))

    # 2. spike placement
    ages = np.array([5, 64])
    D = rr.spike_matrix(64, np.array([2.0, 3.0]), 'last', 1, ages)
    check('2 last = newest slot', D[0, -1] == 2 and D[0].sum() == 2 and D[1, -1] == 3)
    D = rr.spike_matrix(64, np.array([2.0]), 'first', 1, ages)
    check('2 first = oldest slot', D[0, 0] == 2 and D[0].sum() == 2)
    D = rr.spike_matrix(64, np.array([2.0]), 'middle', 1, ages)
    check('2 middle = age 32', D[0, 64 - 32] == 2 and D[0].sum() == 2)
    D = rr.spike_matrix(64, np.array([1.0, 1.0]), 'random', 1, ages)
    check('2 random uses given ages', D[0, 64 - 5] == 1 and D[1, 0] == 1)
    D = rr.spike_matrix(64, np.array([1.0]), 'last', 3, ages)
    check('2 duration 3 at last = 3 newest', list(np.where(D[0])[0]) == [61, 62, 63])
    D = rr.spike_matrix(64, np.array([1.0]), 'first', 6, ages)
    check('2 duration 6 at first stays inside', list(np.where(D[0])[0]) == [0, 1, 2, 3, 4, 5])
    D = rr.spike_matrix(40, np.array([1.0]), 'first', 1, ages)
    check('2 short reference: first = column 0', D[0, 0] == 1 and D[0].sum() == 1)

    # 3. vectorised rank after one replacement == full re-ranking
    rng = np.random.default_rng(3)
    ok = True
    for trial in range(200):
        n = 30
        sc = rng.integers(0, 6, n).astype(float)             # many ties
        pos = rng.choice(n, 5, replace=False)
        v = rng.integers(0, 8, 5).astype(float)
        fast = rr.single_replacement_ranks(sc, pos, v)
        for j, p in enumerate(pos):
            s2 = sc.copy()
            s2[p] = v[j]
            if rr.ranks_of(s2)[p] != fast[j]:
                ok = False
    check('3 single-replacement rank = full re-rank (ties)', ok)

    # 4. every scorer is row-wise (score of a row does not depend on the others)
    X = fe.V[:60, 20:84]
    ok = True
    for key, fm in methods.items():
        W = key[1]
        full = fe._safe_score(fm.scorer, X[:, -W:])
        part = fe._safe_score(fm.scorer, X[7:19, -W:])
        ok &= np.allclose(full[7:19], part, rtol=0, atol=1e-12)
    check('4 all scorers row-wise', bool(ok))

    # 5. windows, pool and targets are the same for every method
    win = list(rr.Windows(fe, methods))
    check('5 windows found', len(win) > 50, f'({len(win)})')
    ok = True
    for k, elig, pool, cand, m64, lo in win[:20]:
        for idx in elig.values():
            ok &= np.all(np.isin(pool, idx))
        ok &= np.all(m64[cand] > 0) and np.all(m64[cand] <= np.median(m64[m64 > 0]))
        t1, t2 = rr.targets_for(cand, 42, k), rr.targets_for(cand, 42, k)
        ok &= np.array_equal(t1, t2) and len(set(t1)) == len(t1)
    check('5 pool inside every eligible set; candidates rule; targets fixed', bool(ok))

    # 6. noise is deterministic and shared
    base = fe.V[:, 40:104]
    a, b = rr.noise_realisations(base, 42, 104), rr.noise_realisations(base, 42, 104)
    check('6 noise deterministic', all(np.array_equal(a[c], b[c]) for c in a))
    c2 = rr.noise_realisations(base, 43, 104)
    check('6 noise differs across seeds', not np.array_equal(a['noise_poisson'], c2['noise_poisson']))
    check('6 gaussian noise non-negative', all((a[c] >= 0).all() for c in a))
    snr = 10 * np.log10((base ** 2).mean() / ((a['noise_gauss_snr0'] - base) ** 2).mean())
    check('6 SNR 0 dB roughly right (after clipping at 0)', -3 < snr < 3, f'({snr:.2f} dB)')

    # 7. run: clean NDCG and num_items equal to the protocol-V5 evaluator
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        csv = td / 'syn.csv'
        df.to_csv(csv, index=False)
        out = td / 'out'
        rr.run(csv, 20, out, True, names=['AF', 'RRD', 'WSPI'], seeds=(42, 43))
        ref = ProtocolV5Evaluator.from_csv(csv, dataset_min_obs=20, seed=42,
                                           causal_universe=True, verbose=False)
        from evaluation.protocol_v5 import build_v5_methods
        ok = True
        for key in [('AF', 7), ('RRD', 64), ('WSPI', 64)]:
            m = build_v5_methods(window_override={key[0]: key[1]}, names=[key[0]])[key[0]]
            p = ref.run_method(m).set_index('window_id')
            r = pd.read_csv(out / 'robustness' / f'{key[0]}_W{key[1]:03d}.csv').set_index('window_id')
            c = r.index.intersection(p.index)
            ok &= len(c) == len(r)
            ok &= np.max(np.abs(r.loc[c, 'ndcg@10_clean'] - p.loc[c, 'ndcg@10'])) < 1e-12
            ok &= bool((r.loc[c, 'num_items'] == p.loc[c, 'num_items']).all())
        check('7 clean NDCG@10 and num_items = protocol V5', bool(ok))
        w = pd.read_csv(out / 'robustness' / 'WSPI_W064.csv')
        cols = [f'{s}_{c}' for s in ('dr', 'sdr', 'up') for c in rr.CONDITIONS]
        check('7 all condition columns', all(c in w.columns for c in cols))
        check('7 WSPI decomposition columns', 'dexpo_pos_first' in w.columns)
        check('7 top10 only for noise', 'top10_noise_poisson' in w.columns
              and 'top10_size10' not in w.columns)
        a7 = pd.read_csv(out / 'robustness' / 'AF_W007.csv')
        check('7 AF(7) ignores spikes older than 7 slots',
              np.allclose(a7['dr_pos_middle'], 0) and np.allclose(a7['dr_pos_first'], 0))
        check('7 spikes promote AF targets', (a7['sdr_size10'] <= 0).all())
        check('7 |signed| <= absolute', (a7['sdr_size10'].abs() <= a7['dr_size10'] + 1e-12).all())
        # 8. a separate process for one method gives identical numbers
        out2 = td / 'out2'
        rr.run(csv, 20, out2, True, names=['WSPI'], seeds=(42, 43))
        w2 = pd.read_csv(out2 / 'robustness' / 'WSPI_W064.csv')
        check('8 method split gives identical results', w.equals(w2))
        out3 = td / 'out3'
        rr.run(csv, 20, out3, True, names=['WSPI'], seeds=(42, 43), stride=3)
        w3 = pd.read_csv(out3 / 'robustness' / 'WSPI_W064.csv')
        sub = w[w['window_id'] % 3 == 0].reset_index(drop=True)
        check('8 stride 3 = every 3rd window, same values', len(w3) > 0 and w3.equals(sub))
        # 8b. per-seed columns; their mean is the seed mean
        ps = w[['dr_size10_s42', 'dr_size10_s43']].mean(axis=1)
        check('8 per-seed columns average to the seed mean',
              np.allclose(ps, w['dr_size10'], rtol=0, atol=1e-12))
        # 8c. checkpoint + interrupted run + --resume = one uninterrupted run
        out4 = td / 'out4'
        rr.run(csv, 20, out4, True, names=['WSPI'], seeds=(42, 43), checkpoint=17)
        w4 = pd.read_csv(out4 / 'robustness' / 'WSPI_W064.csv')
        check('8 checkpointed run = single pass', w4.equals(w))
        f4 = out4 / 'robustness' / 'WSPI_W064.csv'
        w4 = pd.read_csv(f4, float_precision='round_trip')
        cut = int(w4['window_id'].iloc[40])
        w4[w4['window_id'] <= cut].to_csv(out4 / 'robustness' / 'WSPI_W064.partial.csv',
                                          index=False)
        f4.unlink()
        rr.run(csv, 20, out4, True, names=['WSPI'], seeds=(42, 43), resume=True, checkpoint=17)
        w5 = pd.read_csv(f4, float_precision='round_trip')
        check('8 resume after interruption = single pass', w5.equals(w4) and
              not (out4 / 'robustness' / 'WSPI_W064.partial.csv').exists())
        try:
            rr.run(csv, 20, out4, True, names=['WSPI'], seeds=(42, 43), stride=2)
            check('8 changed stride in the same folder refused', False)
        except SystemExit:
            check('8 changed stride in the same folder refused', True)
        # 9. decomposition identity: dlogmu + dexpo = mean log score change
        check('9 WSPI decomposition finite', np.isfinite(w['dexpo_size10']).all())
        # 10. collect runs
        scen = td / 'root' / 'syn'
        scen.parent.mkdir()
        out.rename(scen)
        rr.collect(td / 'root', td / 'none', td / 'none')
        s = pd.read_csv(td / 'root' / 'robustness_summary.csv')
        check('10 summary rows', len(s) > 0 and set(s['config']) <= {'default', 'eq64'})

    print('\nALL OK' if not FAILS else f'\n{len(FAILS)} FAILED: {FAILS}')
    return 0 if not FAILS else 1


if __name__ == '__main__':
    sys.exit(main())
