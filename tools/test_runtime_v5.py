r"""
Unit test for tools/run_runtime_v5.py (task T3.8 / E9).

Run from the project root:   python tools\test_runtime_v5.py
The last line must be ALL OK.  Writes only to a temporary folder.
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_runtime_v5 as rr  # noqa: E402
from evaluation.protocol_v5 import ProtocolV5Evaluator, build_v5_methods  # noqa: E402

N_OK = 0


def check(cond, msg):
    global N_OK
    if not cond:
        print(f'FAIL: {msg}')
        sys.exit(1)
    N_OK += 1
    print(f'ok  {N_OK:2d}  {msg}')


def synth_dataset(n_items=30, n_slots=140, seed=7):
    rng = np.random.default_rng(seed)
    t = pd.date_range('2025-01-01', periods=n_slots, freq='h')
    rows = []
    for i in range(n_items):
        lam = rng.lognormal(0.5, 1.0)
        c = rng.poisson(lam, n_slots)
        keep = rng.random(n_slots) > 0.15               # sparse file: missing rows
        keep[0] = keep[-1] = True
        for k in np.where(keep)[0]:
            rows.append((t[k], f'item{i:03d}', int(c[k])))
    return pd.DataFrame(rows, columns=['timestamp', 'item_id', 'count'])


def main():
    # 1-3 threads and scorers
    check(rr.THREADS == '1' and all(os.environ.get(v) == '1' for v in rr.THREAD_VARS),
          'one thread set for NumPy / BLAS before import')
    X = rr.synth_chunk(np.random.default_rng(1), 500, 64)
    same = True
    for m in rr.METHODS:
        a = rr.make_scorer(m, 64)(X)
        b = build_v5_methods(level=3, window_override={m: 64}, names=[m])[m].scorer(X)
        same &= np.array_equal(a, b)
    check(same, 'make_scorer = build_v5_methods scorer (all nine, N=64)')
    dflt = build_v5_methods(level=3)
    same = all(np.array_equal(rr.make_scorer(m, rr.DEFAULT_N[m])(X[:, -rr.DEFAULT_N[m]:]),
                              dflt[m].scorer(X[:, -rr.DEFAULT_N[m]:])) for m in rr.METHODS)
    check(same, 'default windows 7 / 64 equal the paper configuration')

    # 4-6 pool and batching
    p1, p2 = rr.make_pool(64, 25_000, chunk=10_000), rr.make_pool(64, 25_000, chunk=10_000)
    check(len(p1) == 3 and all(np.array_equal(a, b) for a, b in zip(p1, p2))
          and not np.array_equal(p1[0], p1[1]), 'pool: 3 distinct deterministic batches')
    pool = rr.make_pool(64, 2_500, chunk=1_000)
    full = np.concatenate(pool)
    ok = True
    for m in rr.METHODS:
        sc = rr.make_scorer(m, 64)
        ok &= np.array_equal(rr.score_all(sc, pool, 2_500), sc(full)[:2_500])
    check(ok, 'batched scoring = one call (bit for bit, incl. last partial batch)')
    small = rr.make_pool(32, 1)
    check(len(small) == 1 and small[0].shape == (1, 32), 'M = 1: one single-item batch')

    # 7-10 memory
    check(rr.coef_bytes('WSPI', 64) == (1024, 128) and rr.coef_bytes('DTCWT+AF', 32) == (512, 64),
          'DTCWT coefficients: 1024 B (128 reals) for N=64, 512 B for N=32')
    import pywt
    ref = sum(c.nbytes for c in pywt.wavedec(np.zeros(64), 'db4', level=3, mode='symmetric'))
    check(rr.coef_bytes('DWT+AF', 64)[0] == ref and rr.coef_bytes('AF', 64) == (0, 0),
          'DWT coefficient bytes from pywt; baselines have none')
    Xb = rr.synth_chunk(np.random.default_rng(2), 2_000, 64)
    pw, ob = rr.traced_peak(rr.make_scorer('WSPI', 64), Xb)
    pa, _ = rr.traced_peak(rr.make_scorer('AF', 64), Xb)
    check(pw > 2_000 * 1024 and ob == 2_000 * 8 and 0 < pa < pw,
          'traced peak: WSPI above its coefficients, AF smaller, output 8 B/item')
    st = rr.state_memory_table()
    w = st[(st['method'] == 'WSPI') & (st['N'] == 64)].iloc[0]
    af = st[(st['method'] == 'AF') & (st['N'] == 7)].iloc[0]
    check(w['window_buffer_bytes'] == 512 and w['transient_coef_bytes'] == 1024
          and af['recursive_unwindowed_state_bytes'] == 8 and len(st) == 6 * 5 + 3 * 4,
          'state memory table (42 rows)')

    # 11-12 probes
    cur, pk = rr.current_rss(), rr.peak_rss()
    check(cur and pk and pk >= cur * 0.5, 'current and peak RSS readable')
    hw = rr.hardware_info()
    check(all(k in hw for k in ('cpu_name', 'ram_bytes', 'logical_cpus', 'versions', 'thread_env'))
          and hw['versions']['dtcwt'] and hw['ram_bytes'], 'hardware info has CPU, RAM, versions')

    # 13-14 bench summary
    raw = pd.DataFrame(dict(method='AF', N=7, M=1000, chunk=1000, n_pool=1, inner=1,
                            repeat=range(4), seconds=[1.0, 2.0, 3.0, 4.0]))
    s = rr.summarize_bench(raw).iloc[0]
    check(s['median_s'] == 2.5 and s['q1_s'] == 1.75 and s['us_per_item_median'] == 2500.0,
          'bench summary: median, quartiles, us per item')
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        g = rr.run_bench(td / 'bench', items=(1, 300), windows=(7, 32), repeats=2, verbose=False)
        check(len(g) == 6 * 2 * 2 + 3 * 1 * 2 and (g['repeats'] == 2).all()
              and (td / 'bench' / 'metadata' / 'bench_run.json').exists(),
              'bench run: grid rows, repeats, metadata')
        raw1 = pd.read_csv(td / 'bench' / 'runtime_raw.csv')
        rr.run_bench(td / 'bench', items=(1, 300), windows=(7, 32), repeats=2, resume=True,
                     verbose=False)
        raw2 = pd.read_csv(td / 'bench' / 'runtime_raw.csv')
        check(raw1.equals(raw2), 'resume keeps finished rows unchanged')

        # 16-17 memory run with a child process
        rr.run_memory(td / 'memory', batches=(200,), windows=(7, 32), rss_windows=(32,),
                      rss_cases=(('chunked', 3_000, 1_000),), methods=['AF', 'WSPI'],
                      verbose=False)
        tg = pd.read_csv(td / 'memory' / 'tracemalloc_grid.csv')
        rs = pd.read_csv(td / 'memory' / 'rss_grid.csv')
        check(len(tg) == 3 and len(rs) == 2 and (rs['n_samples'] > 0).all()
              and (rs['rss_base_bytes'] > 0).all(), 'memory run: traced grid and RSS child process')
        check((td / 'memory' / 'state_memory.csv').exists()
              and (td / 'memory' / 'metadata' / 'memory_run.json').exists(),
              'memory run: state table and metadata')

        # 18-21 timed evaluator on a synthetic file
        df = synth_dataset()
        ev0 = ProtocolV5Evaluator(df, 5, seed=42, robustness=True, causal_universe=True,
                                  verbose=False, robustness_sample_size=5)
        ev1 = rr.TimedEvaluator(df, 5, seed=42, robustness=True, causal_universe=True,
                                verbose=False, robustness_sample_size=5)
        methods = build_v5_methods(level=3, window_override={m: 64 for m in rr.WAVELETS})
        methods = {k: rr.FastMethod(v.name, v.window_slots, 3 if k in rr.BASELINES else 32,
                                    v.scorer) for k, v in methods.items()}
        eq, rec_ok, rob_ok = True, True, True
        for n, fm in methods.items():
            a0 = ev0.run_method(fm, out_dir=None)
            a1, sr, rob, tot = ev1.run_timed(fm)
            eq &= a0.equals(a1)
            rec_ok &= (len(sr) == len(a1) and np.array_equal(sr['window_id'], a1['window_id'])
                       and np.array_equal(sr['n_scored'], a1['num_items']))
            rob_ok &= (set(rob) == set(a1['window_id']) and sr['score_s'].sum() + sum(rob.values()) <= tot)
        check(eq, 'timers do not change any output (all nine methods, robustness on)')
        check(rec_ok, 'one timed scorer call per evaluated window, with its item count')
        check(rob_ok, 'robustness time recorded per window; parts <= total')
        ref = ev0.run_method(methods['WSPI'], out_dir=None)
        c_ok = rr.control_frame(ref.copy(), ref, 'WSPI')
        bad = ref.copy()
        bad.loc[bad.index[3], 'ndcg@10'] += 1e-9
        c_bad = rr.control_frame(bad, ref, 'WSPI')
        c_miss = rr.control_frame(ref.iloc[1:].copy(), ref, 'WSPI')
        check(c_ok['pass_'] and not c_bad['pass_'] and not c_miss['pass_']
              and c_ok['columns_checked'] >= 17, 'control: equal passes, changed value / window fails')

        # 22 real run end-to-end + collect
        dpath = td / 'synth.csv'
        df.to_csv(dpath, index=False)
        ns = type('A', (), dict(out=str(td / 'real' / 'youtube_hourly'), data=str(dpath), min_obs=5,
                                causal_universe=True, methods=None, ref=None))
        rr.run_real(ns)
        rs = pd.read_csv(td / 'real' / 'youtube_hourly' / 'real_summary.csv')
        wt = pd.read_csv(td / 'real' / 'youtube_hourly' / 'protocol' / 'window_timing.csv')
        check(len(rs) == 9 and (rs['other_s'] > 0).all() and len(wt) == rs['n_windows'].sum(),
              'real run: summary for nine methods, window timing rows')
        rr.run_bench(td / 'bench', items=(10_000,), windows=(7, 64), repeats=1, verbose=False)
        rr.run_memory(td / 'memory', batches=(10_000,), windows=(7, 64), rss_windows=(), methods=rr.METHODS,
                      verbose=False)
        rr.collect(td)
        pt = pd.read_csv(td / 'runtime_paper_table.csv')
        check(len(pt) == 9 and pt['us_per_item_default_M1e4'].notna().all()
              and pt.loc[pt['method'] == 'WSPI', 'coef_bytes_per_item_N64'].iloc[0] == 1024,
              'collect: paper table with nine methods')
    print(f'ALL OK ({N_OK} checks)')


if __name__ == '__main__':
    main()
