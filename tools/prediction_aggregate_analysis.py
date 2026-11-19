"""Aggregate view of all wavelet-domain methods of the popularity-prediction study on the test part.

Reads only existing CSV files under results/prediction and writes summary CSVs.
Usage: python tools/prediction_aggregate_analysis.py <results/prediction dir> <output dir>
"""
import os
import sys

import pandas as pd

MAIN = ['ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion']
FULL = ['Persistence', 'Holt', 'ARYW', 'MA64', 'SMA8', 'SMA16', 'SMA32',
        'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64']
CLASSIC = ['Persistence', 'Holt', 'ARYW', 'MA64']
SOURCES = {'dev': ['stats_test', 'stats_test_D', 'stats_explore'],
           'conf': ['confirm/stats_test']}
MEANS = {'dev': 'decision_D/test_means.csv',
         'conf': 'confirm/decision/test_means.csv'}
# same model in two domains: (data, stats folder, coefficient-domain ref, time-domain method)
PAIRS = [('dev', 'stats_explore', 'WSPI-F-YW', 'ARYW'),
         ('dev', 'stats_test_D', 'WSPI-M', 'MA64'),
         ('conf', 'confirm/stats_test', 'WSPI-M', 'MA64'),
         ('dev', 'stats_test', 'GBM', 'GBM+W'),
         ('dev', 'stats_test_B', 'WSPI-FA', 'ARIMA')]


def load_pairs(root):
    rows = []
    for data, folders in SOURCES.items():
        for d in folders:
            p = pd.read_csv(os.path.join(root, d, 'all_paired_tests.csv'))
            p = p[p.metric.isin(MAIN) & (p.part == 'test')].copy()
            p['data'] = data
            rows.append(p)
    p = pd.concat(rows)
    return p.drop_duplicates(['data', 'scenario', 'horizon', 'metric', 'reference', 'method'])


def gain_table(p):
    out = []
    for (data, ref), g in p.groupby(['data', 'reference']):
        if not ref.startswith('WSPI'):
            continue
        for set_name, base in [('classic', CLASSIC), ('full', FULL)]:
            for met in MAIN:
                gm = g[(g.metric == met) & g.method.isin(base)]
                n = {'ref_better': 0, 'n.s.': 0, 'ref_worse': 0}
                cells = []
                for (sc, h), c in gm.groupby(['scenario', 'horizon']):
                    lib = bool(c.lower_is_better.iloc[0])
                    b = c.loc[c.mean_method.idxmin()] if lib else c.loc[c.mean_method.idxmax()]
                    n[b.verdict] += 1
                    if b.verdict == 'ref_better':
                        cells.append(f'{sc}/h{h} vs {b.method}')
                out.append(dict(data=data, method=ref, baseline_set=set_name, metric=met,
                                cells=sum(n.values()), better_than_best=n['ref_better'],
                                ns_vs_best=n['n.s.'], worse_than_best=n['ref_worse'],
                                pairs=len(gm), pairs_better=int((gm.verdict == 'ref_better').sum()),
                                pairs_worse=int((gm.verdict == 'ref_worse').sum()),
                                gain_cells='; '.join(cells)))
    return pd.DataFrame(out)


def domain_pairs(root):
    out = []
    for data, d, ref, meth in PAIRS:
        p = pd.read_csv(os.path.join(root, d, 'all_paired_tests.csv'))
        p = p[p.metric.isin(MAIN) & (p.part == 'test') & (p.reference == ref) & (p.method == meth)]
        for met, g in p.groupby('metric'):
            out.append(dict(data=data, source=d, reference=ref, method=meth, metric=met,
                            cells=len(g), ref_better=int((g.verdict == 'ref_better').sum()),
                            ns=int((g.verdict == 'n.s.').sum()),
                            ref_worse=int((g.verdict == 'ref_worse').sum())))
    return pd.DataFrame(out)


def mean_views(root):
    tops, ranks = [], []
    for data, f in MEANS.items():
        m = pd.read_csv(os.path.join(root, f))
        m = m[m.metric.isin(MAIN)].copy()
        m['score'] = m['mean'].where(m.metric != 'robustness_distortion', -m['mean'])
        key = ['scenario', 'horizon', 'metric']
        m['rank'] = m.groupby(key)['score'].rank(ascending=False)
        top = m.loc[m.groupby(key)['score'].idxmax()]
        t = top.groupby(['metric', 'method']).size().reset_index(name='cells_first')
        t['data'] = data
        tops.append(t)
        ar = m.groupby(['scenario', 'horizon', 'method'])['rank'].mean().reset_index()
        s = ar.groupby('method')['rank'].mean().reset_index(name='mean_of_avg_rank')
        ar['pos'] = ar.groupby(['scenario', 'horizon'])['rank'].rank(method='min')
        s = s.merge(ar[ar.pos == 1].method.value_counts().rename('cells_best_avg_rank')
                    .reset_index(), on='method', how='left').fillna(0)
        s['data'] = data
        ranks.append(s.sort_values('mean_of_avg_rank'))
    return pd.concat(tops), pd.concat(ranks)


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    gain_table(load_pairs(root)).to_csv(os.path.join(out, 'gain_vs_best_all_methods.csv'), index=False)
    domain_pairs(root).to_csv(os.path.join(out, 'same_model_two_domains.csv'), index=False)
    tops, ranks = mean_views(root)
    tops.to_csv(os.path.join(out, 'first_place_by_metric.csv'), index=False)
    ranks.to_csv(os.path.join(out, 'average_rank_four_metrics.csv'), index=False)
    print('AGGREGATE OK')


if __name__ == '__main__':
    main()
