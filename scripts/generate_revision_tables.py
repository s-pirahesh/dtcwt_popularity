r"""
Generate the LaTeX result tables of the Scientific Reports revision (task T3.10)
================================================================================
All numbers come from the result CSVs under ``results/revision_v5`` through the
same loaders that draw the figures (scripts/generate_revision_figures.py:
v4_values, v4_tests), so a table and its figure can never disagree.  Nothing
is typed by hand.  Added in task T3.10 (chat 17, 27 Sep 2026).

Parameters
----------
  --results   folder of the revision results (normally results\revision_v5)
  --out       folder for the .tex files (Response\Tables)
  --data-out  folder for the source CSVs of the tables and figures
              (default <results>/T3.10_main_figures; CSV and JSON only)

Tables (.tex, booktabs + amssymb)
--------------------------------
  Main text
    T3.10_tab_main_default.tex   default configuration (baselines 7, wavelet-based 64)
    T3.10_tab_main_equal64.tex   equal window of 64 slots for all nine methods
        Rows: 4 scenarios x 9 methods; columns NDCG@10, rho, RSI@10, Delta Rank
        (mean over the common windows).  Bold = best value of the scenario at the
        printed precision.  Marker after a baseline: \blacktriangle = significantly
        better than WSPI, \triangledown = significantly worse (block Wilcoxon,
        Holm per scenario x metric, alpha 0.05; T1.6).  The 95 % CIs and the tie
        shares are in the SI tables; an automatic footnote names the methods with
        tied scores among the top 21 in >= 10 % of the windows of any scenario.
  SI
    T3.10_tab_si_ci_default.tex / _equal64.tex     mean [95 % CI] + Ties share
        (equal64 with the extra smoothers SMA, EWMA-eq, Holt)
    T3.10_tab_si_tests_default.tex / _equal64.tex  Cliff's delta (positive = WSPI
        better) and Holm-adjusted block p against WSPI
    T3.10_tab_si_movielens.tex                     MovieLens default, mean [95 % CI] + Ties
    T3.10_tab_si_movielens_equal64.tex             MovieLens equal window 64 (12 methods)
    T3.10_tab_si_movielens_tests.tex               MovieLens paired tests, both configurations

Source data (CSV, in --data-out)
--------------------------------
    values_all.csv   every mean / CI / ties share used by the tables and figures
    tests_all.csv    every paired test used by the tables
    control.csv      statistics mean vs summary_common_windows mean, per row
    metadata/generate_revision_tables_run.json  inputs with md5, versions, date

Usage (Windows, from the project root)
--------------------------------------
  python scripts\generate_revision_tables.py --results results\revision_v5 ^
         --out "...\Revisions\V4\Response\Tables" ^
         --data-out results\revision_v5\T3.10_main_figures
"""
import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_revision_figures as grf  # noqa: E402

ROOT = HERE.parent
SCEN = [s for s, _ in grf.V4_SCENARIOS]
SCEN_LABEL = dict(grf.V4_SCENARIOS)
ML = [s for s, _ in grf.ML_SCENARIOS]
ML_LABEL = dict(grf.ML_SCENARIOS)
MET = grf.MAIN_METRICS
MET_TEX = {'ndcg@10': 'NDCG@10', 'spearman_rho': r'$\rho$', 'rsi@10': 'RSI@10',
           'robustness_distortion': r'$\Delta$Rank'}
ARROW = {'ndcg@10': r'$\uparrow$', 'spearman_rho': r'$\uparrow$', 'rsi@10': r'$\uparrow$',
         'robustness_distortion': r'$\downarrow$'}
DEC = {'ndcg@10': 4, 'spearman_rho': 4, 'rsi@10': 4, 'robustness_distortion': 2}
EXTRA = ['SMA', 'EWMA-eq', 'Holt']
TIE_NOTE = 0.10   # footnote threshold for the share of windows with tied top-21 scores
UP = r'\textsuperscript{$\blacktriangle$}'
DOWN = r'\textsuperscript{$\triangledown$}'
CONFIG_TEXT = {'default': 'default configuration (baselines 7 slots, wavelet-based methods 64 slots)',
               'equal64': 'equal window of 64 slots for all methods'}


def fmt(x, d):
    return f'{x:.{d}f}'


def best_mask(v, met):
    """Methods holding the best value of one column at the printed precision."""
    r = v.round(DEC[met])
    b = r.min() if met == 'robustness_distortion' else r.max()
    return r == b


def marker(tests, sc, met, m):
    if m == 'WSPI':
        return ''
    t = tests[(tests['scenario'] == sc) & (tests['metric'] == met) & (tests['method'] == m)]
    if t.empty:
        return ''
    v = t['verdict'].iloc[0]
    return DOWN if v == 'ref_better' else UP if v == 'ref_worse' else ''


def tie_footnote(vals, methods):
    t = (vals[vals['metric'] == 'rsi@10'].groupby('method')['ties_top21_share'].max()
         .reindex(methods).dropna())
    t = t[t >= TIE_NOTE].sort_values(ascending=False)
    if t.empty:
        return 'No method had tied scores among the top 21 in 10\\% or more of the windows.'
    parts = ', '.join(f'{m} ({v:.2f})' for m, v in t.items())
    return ('Methods with tied scores among the top 21 in 10\\% or more of the windows of at '
            f'least one scenario (largest share): {parts}; their NDCG@10 and RSI@10 depend on '
            'the fixed tie rule (item order). Shares per scenario are in the SI.')


def main_table(vals, tests, config, label, scen, scen_label, methods=grf.V4_ORDER):
    nwin = {sc: int(vals[(vals['scenario'] == sc) & (vals['metric'] == 'ndcg@10')
                         & (vals['method'] == 'WSPI')]['n_windows'].iloc[0]) for sc in scen}
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{Results in the ' + CONFIG_TEXT[config] + '; mean over the common windows '
         '(' + ', '.join(f'{scen_label[s]} {nwin[s]:,}' for s in scen) + '). '
         r'Bold: best value of the scenario. ' + UP + ' / ' + DOWN +
         r': significantly better / worse than WSPI (block Wilcoxon test, Holm correction, $\alpha=0.05$). '
         r'95\% confidence intervals and effect sizes are given in the SI.}',
         r'\label{' + label + '}', r'\small',
         r'\begin{tabular}{@{}l' + 'r' * len(MET) + '@{}}', r'\toprule',
         'Method & ' + ' & '.join(f'{MET_TEX[m]} {ARROW[m]}' for m in MET) + r'\\']
    for sc in scen:
        L += [r'\midrule', r'\multicolumn{' + str(len(MET) + 1) + r'}{@{}l}{\textit{' + scen_label[sc] + r'}}\\']
        cols = {}
        for met in MET:
            v = vals[(vals['scenario'] == sc) & (vals['metric'] == met)].set_index('method')['mean'].reindex(methods)
            cols[met] = (v, best_mask(v, met))
        for m in methods:
            cells = []
            for met in MET:
                v, b = cols[met]
                s = fmt(v[m], DEC[met])
                s = r'\textbf{' + s + '}' if b[m] else s
                cells.append(s + marker(tests, sc, met, m))
            name = r'\textbf{WSPI}' if m == 'WSPI' else m
            L.append(name + ' & ' + ' & '.join(cells) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', '', r'\vspace{0.3em}',
          r'{\footnotesize\textit{' + tie_footnote(vals, methods) + '}}', r'\end{table}', '']
    return '\n'.join(L)


def ci_table(vals, config, label, scen, scen_label, methods, title):
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{' + title + r' Mean [95\% block-bootstrap confidence interval] over the common windows; '
         r'Ties = share of windows with tied scores among the top 21.}',
         r'\label{' + label + '}', r'\scriptsize', r'\resizebox{\linewidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'c' * len(MET) + 'c@{}}', r'\toprule',
         'Method & ' + ' & '.join(f'{MET_TEX[m]} {ARROW[m]}' for m in MET) + r' & Ties\\']
    for sc in scen:
        L += [r'\midrule', r'\multicolumn{' + str(len(MET) + 2) + r'}{@{}l}{\textit{' + scen_label[sc] + r'}}\\']
        for m in methods:
            cells = []
            for met in MET:
                r = vals[(vals['scenario'] == sc) & (vals['metric'] == met) & (vals['method'] == m)]
                if r.empty:
                    cells.append('--')
                    continue
                r = r.iloc[0]
                d = DEC[met]
                cells.append(f"{fmt(r['mean'], d)} [{fmt(r['ci_low'], d)}, {fmt(r['ci_high'], d)}]")
            r = vals[(vals['scenario'] == sc) & (vals['method'] == m)]
            ties = fmt(r['ties_top21_share'].iloc[0], 2) if not r.empty else '--'
            L.append(m + ' & ' + ' & '.join(cells) + f' & {ties}' + r'\\')
    L += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}', '']
    return '\n'.join(L)


def p_text(p):
    return r'$<$0.001' if p < 0.001 else f'{p:.3f}'


def tests_table(tests, label, scen, scen_label, methods, title, configs=None):
    configs = configs or sorted(tests['config'].unique())
    L = [r'\begin{table}[htbp]', r'\centering',
         r'\caption{' + title + r" Paired comparison of WSPI with each method on the common windows: Cliff's "
         r'$\delta$ (positive = WSPI better) and, in brackets, the Holm-adjusted $p$ of the block Wilcoxon test.}',
         r'\label{' + label + '}', r'\scriptsize', r'\resizebox{\linewidth}{!}{%',
         r'\begin{tabular}{@{}l' + 'c' * len(MET) + '@{}}', r'\toprule',
         'Method & ' + ' & '.join(MET_TEX[m] for m in MET) + r'\\']
    for cfg in configs:
        for sc in scen:
            head = scen_label[sc] + ('' if len(configs) == 1 else f' ({CONFIG_TEXT[cfg].split(" (")[0]})')
            L += [r'\midrule', r'\multicolumn{' + str(len(MET) + 1) + r'}{@{}l}{\textit{' + head + r'}}\\']
            for m in methods:
                if m == 'WSPI':
                    continue
                cells = []
                for met in MET:
                    t = tests[(tests['config'] == cfg) & (tests['scenario'] == sc)
                              & (tests['metric'] == met) & (tests['method'] == m)]
                    if t.empty:
                        cells.append('--')
                    else:
                        t = t.iloc[0]
                        cells.append(f"{t['cliffs_delta_favours_ref']:+.3f} ({p_text(t['p_block_holm'])})")
                L.append(m + ' & ' + ' & '.join(cells) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table}', '']
    return '\n'.join(L)


def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True, help='folder for the .tex tables')
    ap.add_argument('--data-out', default=None, help='folder for the source CSVs')
    a = ap.parse_args()
    res = Path(a.results) if Path(a.results).is_absolute() else ROOT / a.results
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    dout = Path(a.data_out) if a.data_out else res / 'T3.10_main_figures'
    dout = dout if dout.is_absolute() else ROOT / dout
    out.mkdir(parents=True, exist_ok=True)
    (dout / 'metadata').mkdir(parents=True, exist_ok=True)

    main9 = grf.V4_ORDER
    all12 = grf.V4_BASELINES + EXTRA + grf.V4_WAVELET
    V = {('default', 'main'): grf.v4_values(res, 'default', SCEN),
         ('equal64', 'main'): grf.v4_values(res, 'equal64', SCEN),
         ('default', 'ml'): grf.v4_values(res, 'default', ML),
         ('equal64', 'ml'): grf.v4_values(res, 'equal64', ML)}
    T = {('default', 'main'): grf.v4_tests(res, 'default', SCEN),
         ('equal64', 'main'): grf.v4_tests(res, 'equal64', SCEN),
         ('default', 'ml'): grf.v4_tests(res, 'default', ML),
         ('equal64', 'ml'): grf.v4_tests(res, 'equal64', ML)}

    files = {
        'T3.10_tab_main_default.tex': main_table(V['default', 'main'], T['default', 'main'], 'default',
                                                 'tab:main_default', SCEN, SCEN_LABEL),
        'T3.10_tab_main_equal64.tex': main_table(V['equal64', 'main'], T['equal64', 'main'], 'equal64',
                                                 'tab:main_equal64', SCEN, SCEN_LABEL),
        'T3.10_tab_si_ci_default.tex': ci_table(V['default', 'main'], 'default', 'tab:si_ci_default', SCEN,
                                                SCEN_LABEL, main9, 'Default configuration.'),
        'T3.10_tab_si_ci_equal64.tex': ci_table(V['equal64', 'main'], 'equal64', 'tab:si_ci_equal64', SCEN,
                                                SCEN_LABEL, all12, 'Equal window of 64 slots, with the '
                                                'simple smoothers SMA, EWMA-eq and Holt.'),
        'T3.10_tab_si_tests_default.tex': tests_table(T['default', 'main'], 'tab:si_tests_default', SCEN,
                                                      SCEN_LABEL, main9, 'Default configuration.'),
        'T3.10_tab_si_tests_equal64.tex': tests_table(T['equal64', 'main'], 'tab:si_tests_equal64', SCEN,
                                                      SCEN_LABEL, all12, 'Equal window of 64 slots.'),
        'T3.10_tab_si_movielens.tex': ci_table(V['default', 'ml'], 'default', 'tab:movielens', ML, ML_LABEL,
                                               main9, 'MovieLens, default configuration.'),
        'T3.10_tab_si_movielens_equal64.tex': ci_table(V['equal64', 'ml'], 'equal64', 'tab:movielens_equal64',
                                                       ML, ML_LABEL, all12,
                                                       'MovieLens, equal window of 64 slots.'),
        'T3.10_tab_si_movielens_tests.tex': tests_table(pd.concat([T['default', 'ml'], T['equal64', 'ml']]),
                                                        'tab:movielens_tests', ML, ML_LABEL, all12,
                                                        'MovieLens, both configurations.',
                                                        configs=['default', 'equal64']),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding='utf-8')
        print(f'[ok] {name}')

    vals = pd.concat(V.values(), ignore_index=True)
    tests = pd.concat(T.values(), ignore_index=True)
    vals.drop(columns=['summary_mean', 'rel_diff']).to_csv(dout / 'values_all.csv', index=False)
    keep = ['config', 'run_group', 'scenario', 'metric', 'reference', 'method', 'n_windows',
            'mean_reference', 'mean_method', 'diff_ref_minus_method', 'block', 'n_blocks',
            'p_block', 'p_block_holm', 'cliffs_delta_favours_ref', 'rb_block_favours_ref', 'verdict']
    tests[[c for c in keep if c in tests.columns]].to_csv(dout / 'tests_all.csv', index=False)
    ctrl = vals[['config', 'scenario', 'method', 'metric', 'mean', 'summary_mean', 'rel_diff']].copy()
    ctrl['pass'] = ctrl['rel_diff'] <= 1e-9
    ctrl.to_csv(dout / 'control.csv', index=False)

    inputs = sorted({grf.v4_summary_path(res, c, s) for c in ('default', 'equal64') for s in SCEN + ML})
    inputs += [res / 'T1.6_stats' / d / f for d in ('causal', 'T2.2_window_sweep', 'T3.9_movielens')
               for f in ('all_method_summary.csv', 'all_paired_tests.csv')]
    meta = dict(task='T3.10 tables and figure source data', script='scripts/generate_revision_tables.py',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                tie_footnote_threshold=TIE_NOTE,
                significance='block Wilcoxon after Holm (scenario x metric), alpha 0.05 (T1.6 verdict column)',
                control=dict(rows=int(len(ctrl)), max_rel_diff=float(ctrl['rel_diff'].max()),
                             all_pass=bool(ctrl['pass'].all())),
                inputs={str(p.relative_to(res)).replace('\\', '/'): md5(p) for p in inputs},
                tables=sorted(files), python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, platform=platform.platform())
    (dout / 'metadata' / 'generate_revision_tables_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'control: {len(ctrl)} rows, max relative difference {ctrl["rel_diff"].max():.2e}, '
          f'all pass = {bool(ctrl["pass"].all())}')
    print(f'{len(files)} tables written to {out}; source CSVs in {dout}')


if __name__ == '__main__':
    main()
