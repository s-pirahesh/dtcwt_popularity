r"""
Unit tests for tools/stats_report.py (revision-srep-v5, T1.6).

    python tools\test_stats_report.py

Prints one line per test and ends with ALL OK.
"""
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_report as sr  # noqa: E402

FAILED = []


def check(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILED.append(name)


def t_cbb():
    rng = np.random.default_rng(0)
    x = rng.normal(size=101)
    # one block covering the whole series: every resample is a rotation
    bm = sr.cbb_means(x, 101, sr.cbb_starts(101, 101, 50, 1))
    check('cbb: block = n gives the exact mean', np.allclose(bm, x.mean()))
    # every resample has exactly n values: constant series stays constant
    bm = sr.cbb_means(np.full(97, 3.5), 10, sr.cbb_starts(97, 10, 200, 1))
    check('cbb: constant series', np.allclose(bm, 3.5))
    # brute force for one resample
    st = sr.cbb_starts(23, 5, 1, 7)
    xx = np.concatenate([x[:23], x[:23]])
    k = st.shape[1]
    vals = np.concatenate([xx[s:s + 5] for s in st[0, :k - 1]] +
                          [xx[st[0, k - 1]:st[0, k - 1] + 23 - (k - 1) * 5]])
    check('cbb: matches brute-force resample',
          len(vals) == 23 and np.isclose(vals.mean(), sr.cbb_means(x[:23], 5, st)[0]))
    # reproducible
    check('cbb: starts reproducible', np.array_equal(sr.cbb_starts(50, 7, 30, 42),
                                                     sr.cbb_starts(50, 7, 30, 42)))


def t_cliff():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 40).astype(float)
    b = rng.integers(0, 5, 35).astype(float)
    brute = np.mean([np.sign(i - j) for i, j in product(a, b)])
    check('cliffs delta = brute force (with ties)', np.isclose(sr.cliffs_delta(a, b), brute))


def t_rb():
    rng = np.random.default_rng(2)
    d = rng.integers(-3, 5, 60).astype(float)
    dn = d[d != 0]
    r = sr.rankdata(np.abs(dn))
    brute = (r[dn > 0].sum() - r[dn < 0].sum()) / r.sum()
    check('rank-biserial = brute force', np.isclose(sr.rank_biserial(d), brute))
    check('rank-biserial all positive = 1', sr.rank_biserial(np.array([1., 2., 0., 3.])) == 1.0)


def t_holm():
    got = sr.holm([0.01, 0.04, 0.03, 0.005])
    check('holm hand example', np.allclose(got, [0.03, 0.06, 0.06, 0.02]))
    got = sr.holm([0.2, np.nan, 0.01])
    check('holm keeps NaN', np.isnan(got[1]) and np.allclose(got[[0, 2]], [0.2, 0.02]))


def t_blocks():
    d = np.arange(10, dtype=float)
    check('block means drop tail', np.allclose(sr.block_means(d, 4), [1.5, 5.5]))


def t_run():
    """Synthetic run folder: reference better in ndcg@10, worse in robustness."""
    rng = np.random.default_rng(3)
    n = 480
    ids = np.arange(32, 32 + n)
    base = rng.normal(0.8, 0.05, n)
    cols = ['ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion', 'mae']
    def frame(method, shift, robust_shift):
        df = pd.DataFrame({'window_id': ids, 'timestamp': ids * 3600000, 'method': method,
                           'num_items': 100})
        df['ndcg@10'] = base + shift + rng.normal(0, 0.005, n)
        df['spearman_rho'] = base
        df['rsi@10'] = base + shift
        df.loc[0, 'rsi@10'] = np.nan
        df['robustness_distortion'] = 20 + robust_shift + rng.normal(0, 1, n)
        df['mae'] = 1.0
        df['ties_top21'] = False
        df['padded'] = False
        df['n_nonfinite'] = 0
        return df
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / 'run'
        (run / 'protocol').mkdir(parents=True)
        frame('WSPI', 0.02, 5.0).to_csv(run / 'protocol' / 'WSPI_protocol.csv', index=False)
        frame('AF', 0.0, 0.0).to_csv(run / 'protocol' / 'AF_protocol.csv', index=False)
        extra = frame('RRD', 0.0, 0.0)
        extra = extra[extra['window_id'] != 40]          # one missing window
        extra.to_csv(run / 'protocol' / 'RRD_protocol.csv', index=False)
        s, p, meta = sr.analyse(run, Path(tmp) / 'out', 24, 12, 'WSPI', 2000, 42, None)
        check('common windows = intersection', meta['n_common_windows'] == n - 1)
        q = p.set_index(['metric', 'method'])
        check('verdict ref_better when reference higher (ndcg@10)',
              q.loc[('ndcg@10', 'AF'), 'verdict'] == 'ref_better')
        check('verdict ref_worse when reference has larger distortion',
              q.loc[('robustness_distortion', 'AF'), 'verdict'] == 'ref_worse'
              and q.loc[('robustness_distortion', 'AF'), 'cliffs_delta_favours_ref'] < 0)
        check('identical series: p = 1, no wins or losses',
              q.loc[('spearman_rho', 'AF'), 'p_block'] == 1.0
              and q.loc[('spearman_rho', 'AF'), 'wins'] == 0
              and q.loc[('spearman_rho', 'AF'), 'verdict'] == 'n.s.')
        check('NaN windows dropped pairwise', q.loc[('rsi@10', 'AF'), 'n_windows'] == n - 2)
        check('sensitivity columns written', 'verdict_sens' in p.columns and 'ci_low_sens' in s.columns)
        ci = q.loc[('ndcg@10', 'AF')]
        check('diff CI contains the mean difference',
              ci['diff_ci_low'] <= ci['diff_ref_minus_method'] <= ci['diff_ci_high'])
        check('output files', all((Path(tmp) / 'out' / f).exists() for f in
                                  ('method_summary.csv', 'paired_tests.csv', 'metadata/stats_run.json')))


if __name__ == '__main__':
    for t in (t_cbb, t_cliff, t_rb, t_holm, t_blocks, t_run):
        t()
    print('ALL OK' if not FAILED else f'{len(FAILED)} FAILED: {FAILED}')
    sys.exit(1 if FAILED else 0)
