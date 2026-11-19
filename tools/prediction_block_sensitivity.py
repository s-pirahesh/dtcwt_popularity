r"""
Sensitivity of the paired tests to the block length, and two further tables
===========================================================================
Reads only the result CSV and JSON files of results/prediction; computes nothing
statistical itself (the verdicts, block p values, intervals and rank-biserial
correlations come from tools/stats_report.py through tools/run_prediction_stats.py).
The dominance and gain rules are those of tools/prediction_test_analysis.py
(imported from there, unchanged).

  --block    Sensitivity of the verdicts to the block length.  Compares the
             reports of the fixed rule b_h = max(b, h) with those of
             ``--block-rule window`` (b_h = max(b, N + h - 1)), written to the
             folders <stats folder><suffix> (default suffix _window):
               stats_test, stats_test_D, stats_test_B, confirm/stats_test.
             Outputs (<root>/decision_review/): block_verdict_changes.csv,
             block_dominance.csv, block_gains.csv, block_pairs.csv,
             block_sensitivity.json; with --figures also the SI tables
             tab_sit_block_verdicts.md and tab_sit_block_counts.md.
  --tables   Two SI tables from the existing reports (fixed block rule):
             effect sizes of the verdicts of the scored methods against the
             11 baselines (block rank-biserial correlation and relative
             difference), and dominance on the two accuracy metrics and on the
             two stability metrics separately.  Outputs: effect_sizes.csv,
             dominance_groups.csv, review_tables.json; with --figures the tables
             tab_sit_effect_sizes.md and tab_sit_dominance_groups.md.

    python tools/prediction_block_sensitivity.py --root results/prediction --block --figures figures_dir
    python tools/prediction_block_sensitivity.py --root results/prediction --tables --figures figures_dir
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prediction_test_analysis as A                                          # noqa: E402

MAIN = A.MAIN
BASES = A.BASE_X2
ACC = ('ndcg@10', 'spearman_rho')
STAB = ('rsi@10', 'robustness_distortion')
DEV = A.SCENARIOS
CONF = ('taxi2024_hourly', 'taxi2024_30min', 'taxi2024_5min', 'wikipedia_daily')
METRIC_NAMES = {'ndcg@10': 'NDCG@10', 'spearman_rho': '$\\rho$', 'rsi@10': 'RSI@10',
                'robustness_distortion': '$\\Delta$Rank'}
SCEN_NAMES = {'youtube_hourly': 'YouTube (1 h)', 'taxi_hourly': 'Taxi (1 h)',
              'taxi_30min': 'Taxi (30 min)', 'taxi_5min': 'Taxi (5 min)',
              'movielens_daily': 'MovieLens (day)', 'movielens_weekly': 'MovieLens (week)',
              'taxi2024_hourly': 'Taxi 2024 (1 h)', 'taxi2024_30min': 'Taxi 2024 (30 min)',
              'taxi2024_5min': 'Taxi 2024 (5 min)', 'wikipedia_daily': 'Wikipedia (day)'}
# (data, stats folder, references whose dominance and gains are counted, scenarios)
ANALYSES = (('Development', 'stats_test', ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W'), DEV),
            ('Development', 'stats_test_D', ('WSPI-M', 'WSPI-H'), DEV),
            ('Development', 'stats_test_B', ('WSPI-FA',), ('youtube_hourly', 'taxi_hourly')),
            ('Confirmation', 'confirm/stats_test', ('WSPI-H', 'WSPI-F2', 'WSPI-M', 'WSPI'), CONF))
# folder whose WSPI report gives the means that pick the best baseline (rule 2)
MEANS_FROM = {'stats_test': 'stats_test', 'stats_test_D': 'stats_test',
              'confirm/stats_test': 'confirm/stats_test'}
# (data, name, folder, reference report, method, turn the verdict around)
PAIRS = (('Development', 'WSPI-F-YW vs WSPI-M', 'stats_test_D', 'WSPI-M', 'WSPI-F-YW', True),
         ('Development', 'WSPI-F2 vs WSPI-H', 'stats_test_D', 'WSPI-H', 'WSPI-F2', True),
         ('Development', 'WSPI-H vs WSPI-M', 'stats_test_D', 'WSPI-H', 'WSPI-M', False),
         ('Confirmation', 'WSPI-F2 vs WSPI-H', 'confirm/stats_test', 'WSPI-H', 'WSPI-F2', True),
         ('Confirmation', 'WSPI-F-YW vs WSPI-M', 'confirm/stats_test', 'WSPI-M', 'WSPI-F-YW', True))
TURN = {'ref_better': 'ref_worse', 'ref_worse': 'ref_better', 'n.s.': 'n.s.'}
EFFECT = (('Development', 'stats_test', ('WSPI', 'WSPI-F2', 'WSPI-FS3'), DEV),
          ('Development', 'stats_test_D', ('WSPI-H',), DEV),
          ('Confirmation', 'confirm/stats_test', ('WSPI-H', 'WSPI-F2', 'WSPI'), CONF))
GROUP_ROWS = (('Development', 'stats_test', ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W'), DEV),
              ('Development', 'stats_test_D', ('WSPI-M', 'WSPI-H'), DEV),
              ('Confirmation', 'confirm/stats_test', ('WSPI', 'WSPI-F2', 'WSPI-M', 'WSPI-H'), CONF))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(root: Path, folder: str, inputs: dict) -> pd.DataFrame:
    p = root / folder / 'all_paired_tests.csv'
    inputs[p.relative_to(root).as_posix()] = sha(p)
    t = pd.read_csv(p, float_precision='round_trip')
    return t[t.part == 'test']


def load_means(root: Path, folder: str, inputs: dict) -> pd.DataFrame:
    p = root / folder / 'all_method_summary.csv'
    inputs[p.relative_to(root).as_posix()] = sha(p)
    return A.means_table(pd.read_csv(p, float_precision='round_trip'))


def dom_cells(t, refs, scen, metrics=MAIN):
    """Relation of each reference to each baseline per cell on the given metrics."""
    out = []
    for ref in refs:
        for b in BASES:
            for s in scen:
                for h in A.HORIZONS:
                    v = []
                    for met in metrics:
                        r = A.row(t, s, h, ref, b, met)
                        if r is None:
                            break
                        v.append(r.verdict)
                    if len(v) != len(metrics):
                        continue
                    out.append(dict(method=ref, baseline=b, scenario=s, horizon=h,
                                    relation=A.relation(v)))
    return pd.DataFrame(out)


def d_counts(cells):
    """D(X): cells where at least one baseline dominates X, per horizon."""
    rows = []
    for m, d in cells.groupby('method', sort=False):
        dom = d[d.relation == 'dominated'][['scenario', 'horizon']].drop_duplicates()
        r = dict(method=m, n_cells=d[['scenario', 'horizon']].drop_duplicates().shape[0], D=len(dom))
        for h in A.HORIZONS:
            r[f'D_h{h}'] = int((dom.horizon == h).sum())
        r['dominated_cells'] = '; '.join(f'{s}/h{h}' for s, h in dom.itertuples(index=False))
        rows.append(r)
    return pd.DataFrame(rows)


def gains(t, means, refs, scen):
    """Rule 2 against the best of the 11 baselines (A+X2) of each cell and metric."""
    n = {r: 0 for r in refs}
    for s in scen:
        for h in A.HORIZONS:
            for met in MAIN:
                mm = means[(means.scenario == s) & (means.horizon == h) & (means.metric == met)
                           & means.method.isin(BASES)].set_index('method')['mean']
                best_v = mm.min() if met in A.LOWER else mm.max()
                best = [x for x in BASES if x in mm.index and mm[x] == best_v][0]
                for ref in refs:
                    r = A.row(t, s, h, ref, best, met)
                    n[ref] += int(r is not None and r.verdict == 'ref_better')
    return n


def pair_counts(t, ref, meth, turn, scen):
    rel = {'dominates': 0, 'dominated': 0, 'trade-off': 0, 'equal': 0}
    for s in scen:
        for h in A.HORIZONS:
            v = [A.row(t, s, h, ref, meth, met) for met in MAIN]
            if any(x is None for x in v):
                continue
            v = [TURN[x.verdict] if turn else x.verdict for x in v]
            rel[A.relation(v)] += 1
    return rel


def md(rows, head, align):
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(align) + '|']
    lines += ['| ' + ' | '.join(str(x) for x in r) + ' |' for r in rows]
    return '\n'.join(lines) + '\n'


def note(tool_opt, sources):
    return (f'<!-- Generated by tools/prediction_block_sensitivity.py {tool_opt} from '
            f'{"; ".join(sources)}. Do not edit by hand; rebuild instead. -->\n\n')


# ---------------------------------------------------------------- --block
def block(root: Path, out: Path, suffix: str, figures: Path | None):
    inputs, vrows, drows, grows, prows = {}, [], [], [], []
    for data, folder, refs, scen in ANALYSES:
        t0 = load(root, folder, inputs)
        t1 = load(root, folder + suffix, inputs)
        key = ['scenario', 'horizon', 'reference', 'method', 'metric']
        a = t0[t0.metric.isin(MAIN)].set_index(key)
        b = t1[t1.metric.isin(MAIN)].set_index(key)
        if len(a) != len(b) or not a.index.sort_values().equals(b.index.sort_values()):
            raise SystemExit(f'{folder}: the two block rules do not have the same comparisons')
        j = a[['verdict', 'block', 'n_blocks']].join(b[['verdict', 'block', 'n_blocks']],
                                                     lsuffix='_max', rsuffix='_window')
        for (s, h), d in j.groupby(level=['scenario', 'horizon'], sort=False):
            v0, v1 = d.verdict_max, d.verdict_window
            vrows.append(dict(data=data, analysis=folder, scenario=s, horizon=int(h),
                              block_max=int(d.block_max.iloc[0]),
                              block_window=int(d.block_window.iloc[0]),
                              n_blocks_max=int(d.n_blocks_max.max()),
                              n_blocks_window=int(d.n_blocks_window.max()),
                              n_verdicts=len(d), same=int((v0 == v1).sum()),
                              sig_to_ns=int(((v0 != 'n.s.') & (v1 == 'n.s.')).sum()),
                              ns_to_sig=int(((v0 == 'n.s.') & (v1 != 'n.s.')).sum()),
                              reversed=int(((v0 != 'n.s.') & (v1 != 'n.s.') & (v0 != v1)).sum()),
                              sig_max=int((v0 != 'n.s.').sum()), sig_window=int((v1 != 'n.s.').sum())))
        if folder == 'stats_test_B':
            continue
        c0, c1 = d_counts(dom_cells(t0, refs, scen)), d_counts(dom_cells(t1, refs, scen))
        for r0 in c0.itertuples():
            r1 = c1[c1.method == r0.method].iloc[0]
            drows.append(dict(data=data, method=r0.method, n_cells=r0.n_cells, D_max=r0.D,
                              D_window=int(r1.D), **{f'D_h{h}_max': getattr(r0, f'D_h{h}') for h in A.HORIZONS},
                              **{f'D_h{h}_window': int(r1[f'D_h{h}']) for h in A.HORIZONS},
                              cells_max=r0.dominated_cells, cells_window=r1.dominated_cells))
        m0 = load_means(root, MEANS_FROM[folder], inputs)
        g0, g1 = gains(t0, m0, refs, scen), gains(t1, m0, refs, scen)
        for r in refs:
            grows.append(dict(data=data, method=r, comparisons=len(scen) * len(A.HORIZONS) * len(MAIN),
                              gains_max=g0[r], gains_window=g1[r]))
    for data, name, folder, ref, meth, turn in PAIRS:
        scen = CONF if data == 'Confirmation' else DEV
        p0 = pair_counts(load(root, folder, inputs), ref, meth, turn, scen)
        p1 = pair_counts(load(root, folder + suffix, inputs), ref, meth, turn, scen)
        prows.append(dict(data=data, pair=name, **{f'{k}_max': v for k, v in p0.items()},
                          **{f'{k}_window': v for k, v in p1.items()}))
    V, Dm, G, P = (pd.DataFrame(x) for x in (vrows, drows, grows, prows))
    V.to_csv(out / 'block_verdict_changes.csv', index=False)
    Dm.to_csv(out / 'block_dominance.csv', index=False)
    G.to_csv(out / 'block_gains.csv', index=False)
    P.to_csv(out / 'block_pairs.csv', index=False)

    def dget(data, m, col):
        return int(Dm[(Dm.data == data) & (Dm.method == m)][col].iloc[0])
    conf = P[(P.data == 'Confirmation') & (P.pair == 'WSPI-F2 vs WSPI-H')].iloc[0]
    screen = {}
    for rule in ('max', 'window'):
        dh, df2 = dget('Confirmation', 'WSPI-H', f'D_{rule}'), dget('Confirmation', 'WSPI-F2', f'D_{rule}')
        f2h = int(conf[f'dominates_{rule}'])
        screen[rule] = dict(D_WSPI_H=dh, D_WSPI_F2=df2, F2_dominates_H=f2h, T1=dh <= 3, T2=dh < df2,
                            T3=f2h <= 2, confirmed=bool(dh <= 3 and dh < df2 and f2h <= 2))
    tot = V.groupby('analysis')[['n_verdicts', 'same', 'sig_to_ns', 'ns_to_sig', 'reversed']].sum()
    head = dict(note='block rule max(b, h) (fixed) against max(b, N + h - 1), N = 64; same methods, '
                     'Holm families, B and seed; verdicts of the four main metrics on the test part',
                suffix=suffix, inputs=inputs,
                verdict_totals={a: {k: int(v) for k, v in r.items()} for a, r in tot.iterrows()},
                confirmation_conditions=screen)
    (out / 'block_sensitivity.json').write_text(json.dumps(head, indent=2))
    print(json.dumps({k: v for k, v in head.items() if k != 'inputs'}, indent=2))
    if figures is not None:
        srcs = ['decision_review/block_verdict_changes.csv']
        rows, last = [], None
        for (data, s, h), d in V[V.analysis != 'stats_test_B'].groupby(['data', 'scenario', 'horizon'], sort=False):
            rows.append(['' if data == last else data, SCEN_NAMES[s], int(h),
                         f'{int(d.block_max.iloc[0])} → {int(d.block_window.iloc[0])}',
                         f'{int(d.n_blocks_max.max())} → {int(d.n_blocks_window.max())}',
                         int(d.n_verdicts.sum()), int(d.sig_to_ns.sum()), int(d.ns_to_sig.sum()),
                         int(d.reversed.sum())])
            last = data
        (figures / 'tab_sit_block_verdicts.md').write_text(
            note('--block', srcs) + md(rows, ['Data', 'Scenario', '$h$', 'Block length',
                                               'Test blocks', 'Verdicts', 'Significant → n.s.',
                                               'n.s. → significant', 'Reversed'],
                                       [':--', ':--', ':-:', ':-:', ':-:', ':-:', ':-:', ':-:', ':-:']),
            encoding='utf-8')
        rows, last = [], None
        for r in Dm.itertuples():
            g = G[(G.data == r.data) & (G.method == r.method)].iloc[0]
            rows.append(['' if r.data == last else r.data, r.method,
                         f'{r.D_max} of {r.n_cells}', f'{r.D_window} of {r.n_cells}',
                         f'{g.gains_max} of {g.comparisons}', f'{g.gains_window} of {g.comparisons}'])
            last = r.data
        (figures / 'tab_sit_block_counts.md').write_text(
            note('--block', ['decision_review/block_dominance.csv', 'decision_review/block_gains.csv'])
            + md(rows, ['Data', 'Method', 'Dominated cells, $\\max(b,h)$', 'Dominated cells, $\\max(b,N+h-1)$',
                        'Gains, $\\max(b,h)$', 'Gains, $\\max(b,N+h-1)$'],
                 [':--', ':--', ':-:', ':-:', ':-:', ':-:']), encoding='utf-8')
        print('Tables written to', figures)


# ---------------------------------------------------------------- --tables
def tables(root: Path, out: Path, figures: Path | None):
    inputs, erows, grows = {}, [], []
    for data, folder, refs, scen in EFFECT:
        t = load(root, folder, inputs)
        t = t[t.method.isin(BASES) & t.metric.isin(MAIN) & t.scenario.isin(scen)]
        for ref in refs:
            for met in MAIN:
                d = t[(t.reference == ref) & (t.metric == met)]
                sig = d[d.verdict != 'n.s.']
                rel = (sig.diff_ref_minus_method.abs() / sig.mean_method.abs()).median() if len(sig) else float('nan')
                erows.append(dict(data=data, method=ref, metric=met, comparisons=len(d),
                                  better=int((d.verdict == 'ref_better').sum()),
                                  worse=int((d.verdict == 'ref_worse').sum()),
                                  median_abs_rb_significant=float(sig.rb_block_favours_ref.abs().median()) if len(sig) else float('nan'),
                                  share_abs_rb_below_0_5_significant=float((sig.rb_block_favours_ref.abs() < 0.5).mean()) if len(sig) else float('nan'),
                                  median_rel_diff_significant=float(rel)))
    for data, folder, refs, scen in GROUP_ROWS:
        t = load(root, folder, inputs)
        for name, mets in (('all', MAIN), ('accuracy', ACC), ('stability', STAB)):
            c = d_counts(dom_cells(t, refs, scen, mets))
            for r in c.itertuples():
                grows.append(dict(data=data, method=r.method, metrics=name, n_cells=r.n_cells, D=r.D))
    E, Gd = pd.DataFrame(erows), pd.DataFrame(grows)
    E.to_csv(out / 'effect_sizes.csv', index=False)
    Gd.to_csv(out / 'dominance_groups.csv', index=False)
    (out / 'review_tables.json').write_text(json.dumps(dict(inputs=inputs), indent=2))
    print(E.to_string())
    print(Gd.pivot_table(index=['data', 'method'], columns='metrics', values='D', sort=False).to_string())
    if figures is not None:
        rows, last = [], None
        for r in E.itertuples():
            tag = (r.data, r.method)
            rows.append(['' if last is not None and last[0] == r.data else r.data,
                         '' if tag == last else r.method, METRIC_NAMES[r.metric],
                         f'{r.better} / {r.worse}',
                         '–' if pd.isna(r.median_abs_rb_significant) else f'{r.median_abs_rb_significant:.2f}',
                         '–' if pd.isna(r.median_rel_diff_significant) else f'{100 * r.median_rel_diff_significant:.1f}%'])
            last = tag
        (figures / 'tab_sit_effect_sizes.md').write_text(
            note('--tables', ['decision_review/effect_sizes.csv']) +
            md(rows, ['Data', 'Method', 'Metric', 'Better / worse', 'Median rank-biserial correlation (absolute)',
                      'Median relative difference'], [':--', ':--', ':--', ':-:', ':-:', ':-:']),
            encoding='utf-8')
        pv = Gd.pivot_table(index=['data', 'method'], columns='metrics', values='D', sort=False)
        nc = Gd.groupby(['data', 'method'], sort=False).n_cells.first()
        rows, last = [], None
        for (data, m), r in pv.iterrows():
            n = int(nc[(data, m)])
            rows.append(['' if data == last else data, m, f'{int(r["all"])} of {n}',
                         f'{int(r["accuracy"])} of {n}', f'{int(r["stability"])} of {n}'])
            last = data
        (figures / 'tab_sit_dominance_groups.md').write_text(
            note('--tables', ['decision_review/dominance_groups.csv']) +
            md(rows, ['Data', 'Method', 'Four main metrics', 'NDCG@10 and $\\rho$', 'RSI@10 and $\\Delta$Rank'],
               [':--', ':--', ':-:', ':-:', ':-:']), encoding='utf-8')
        print('Tables written to', figures)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=None, help='default: <root>/decision_review')
    ap.add_argument('--block', action='store_true')
    ap.add_argument('--suffix', default='_window', help='folder suffix of the window-rule reports')
    ap.add_argument('--tables', action='store_true')
    ap.add_argument('--figures', default=None, help='folder for the SI markdown tables')
    a = ap.parse_args(argv)
    if not (a.block or a.tables):
        ap.error('give --block and/or --tables')
    root = Path(a.root).resolve()
    out = Path(a.out) if a.out else root / 'decision_review'
    out.mkdir(parents=True, exist_ok=True)
    figures = Path(a.figures) if a.figures else None
    if a.tables:
        tables(root, out, figures)
    if a.block:
        block(root, out, a.suffix, figures)
    print('Saved to', out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
