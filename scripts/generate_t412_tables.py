r"""
Six-scenario result tables of paper V5 (task T4.12)
====================================================
MovieLens joins the main results of Section 4 (28 Sep 2026).  This program writes the two main tables and the
four SI tables of intervals and paired tests with all six scenarios: YouTube,
NYC Yellow Taxi at three granularities and MovieLens daily and weekly.

Nothing is computed here.  Every value comes from the loaders of
scripts/generate_revision_figures.py (v4_values, v4_tests), which check the
T1.6 statistics against summary_common_windows.csv of every run, and the
formatting helpers of scripts/generate_revision_tables.py.  Neither file is
changed; the T3.10 tables stay as they were.

Main tables: one panel per metric, one column per scenario,
nine rows per panel.  Bold = best of the nine methods in the scenario at the
printed precision; \blacktriangle / \triangledown = significantly better /
worse than WSPI (block Wilcoxon, Holm per scenario x metric, alpha 0.05).

Parameters
----------
  --results   folder of the revision results (results\revision_v5); read only
  --out       folder for the .tex files and the run JSON

Output (in --out)
-----------------
  T4.12_tab_main_default.tex     Table 8 of the paper   (tab:main_default)
  T4.12_tab_main_equal64.tex     Table 9 of the paper   (tab:main_equal64)
  T4.12_tab_si_ci_default.tex    Supplementary Table S6 (tab:si_ci_default); the four SI tables are
                                 printed in two parts (YouTube and taxi; MovieLens, continued)
  T4.12_tab_si_tests_default.tex Supplementary Table S7 (tab:si_tests_default)
  T4.12_tab_si_ci_equal64.tex    Supplementary Table S8 (tab:si_ci_equal64)
  T4.12_tab_si_tests_equal64.tex Supplementary Table S9 (tab:si_tests_equal64)
  generate_t412_tables_run.json  inputs with md5, versions, date
Nothing is written to results.

Usage (Windows, from the project root)
--------------------------------------
  python scripts\generate_t412_tables.py --results results\revision_v5 --out <folder for the tables>
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
import generate_revision_figures as grf   # noqa: E402
import generate_revision_tables as grt    # noqa: E402

ROOT = HERE.parent
SCEN4 = [s for s, _ in grf.V4_SCENARIOS]
ML2 = [s for s, _ in grf.ML_SCENARIOS]
SCEN6 = SCEN4 + ML2
LABEL6 = {**dict(grf.V4_SCENARIOS), **dict(grf.ML_SCENARIOS)}
# column heads of the main tables: the labels of Table 5 (tab:data)
HEAD6 = {'youtube_hourly': r'\shortstack{YouTube\\(1h)}', 'taxi_hourly': r'\shortstack{Taxi\\Hourly}',
         'taxi_30min': r'\shortstack{Taxi\\30min}', 'taxi_5min': r'\shortstack{Taxi\\5min}',
         'movielens_daily': r'\shortstack{MovieLens\\(1d)}', 'movielens_weekly': r'\shortstack{MovieLens\\(1w)}'}
MET_PANEL = {'ndcg@10': r'NDCG@10 $\uparrow$', 'spearman_rho': r'Spearman $\rho$ $\uparrow$',
             'rsi@10': r'RSI@10 $\uparrow$', 'robustness_distortion': r'$\Delta$Rank $\downarrow$'}
# Supplementary numbers of the interval and test tables (order of first citation)
SI_NUM = {'default': ('S6', 'S7'), 'equal64': ('S8', 'S9')}


def tie_note(vals, cfg):
    t = (vals[vals['metric'] == 'rsi@10'].groupby('method')['ties_top21_share'].max()
         .reindex(grf.V4_ORDER).dropna())
    t = t[t >= grt.TIE_NOTE].sort_values(ascending=False, kind='stable')
    parts = ', '.join(f'{m} ({v:.2f})' for m, v in t.items())
    return ('Methods with tied scores among the top 21 in 10\\% or more of the windows of at least one '
            f'scenario (largest share): {parts}; their NDCG@10 and RSI@10 depend on the fixed tie rule '
            f'(item order). Shares per scenario are in Supplementary Table~{SI_NUM[cfg][0]}.')


def main_table6(vals, tests, cfg, label):
    nwin = {s: int(vals[(vals['scenario'] == s) & (vals['metric'] == 'ndcg@10')
                        & (vals['method'] == 'WSPI')]['n_windows'].iloc[0]) for s in SCEN6}
    ci, eff = SI_NUM[cfg]
    cap = ('Results in the ' + grt.CONFIG_TEXT[cfg] + '; mean over the common windows ('
           + ', '.join(f'{LABEL6[s]} {nwin[s]:,}' for s in SCEN6) + '). Bold: best value of the scenario. '
           + grt.UP + ' / ' + grt.DOWN + r': significantly better / worse than WSPI (block Wilcoxon test, '
           r'Holm correction, $\alpha=0.05$). 95\% confidence intervals are given in Supplementary Table~'
           + ci + ' and effect sizes in Supplementary Table~' + eff + '.')
    L = [r'\begin{table}[htbp]', r'\centering', r'\caption{' + cap + '}', r'\label{' + label + '}',
         r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{0.96}',
         r'\begin{tabular}{@{}l' + 'r' * len(SCEN6) + '@{}}', r'\toprule',
         'Method & ' + ' & '.join(HEAD6[s] for s in SCEN6) + r'\\']
    for met in grf.MAIN_METRICS:
        L += [r'\midrule', r'\multicolumn{' + str(len(SCEN6) + 1) + r'}{@{}l}{\textit{' + MET_PANEL[met] + r'}}\\']
        cols = {}
        for s in SCEN6:
            v = (vals[(vals['scenario'] == s) & (vals['metric'] == met)].set_index('method')['mean']
                 .reindex(grf.V4_ORDER))
            if v.isna().any():
                raise ValueError(f'{cfg}/{s}/{met}: missing method')
            cols[s] = (v, grt.best_mask(v, met))
        for m in grf.V4_ORDER:
            cells = []
            for s in SCEN6:
                v, b = cols[s]
                x = grt.fmt(v[m], grt.DEC[met])
                x = r'\textbf{' + x + '}' if b[m] else x
                cells.append(x + grt.marker(tests, s, met, m))
            L.append((r'\textbf{WSPI}' if m == 'WSPI' else m) + ' & ' + ' & '.join(cells) + r'\\')
    L += [r'\bottomrule', r'\end{tabular}', '', r'\vspace{0.3em}',
          r'{\footnotesize\textit{' + tie_note(vals, cfg) + '}}', r'\end{table}', '']
    return '\n'.join(L)


def split6(make, label, title):
    """One SI table in two parts on consecutive pages: YouTube and taxi, then MovieLens
    (the same table number; the second part uses \\ContinuedFloat of the caption package)."""
    a = make(SCEN4, label, title + ' YouTube and NYC Yellow Taxi.')
    b = make(ML2, label, title + ' MovieLens (continued).')
    b = b.replace('\\centering', '\\ContinuedFloat\n\\centering', 1)
    b = '\n'.join(l for l in b.split('\n') if not l.startswith('\\label{'))
    return a + b


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    res = Path(a.results) if Path(a.results).is_absolute() else ROOT / a.results
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    V = {c: grf.v4_values(res, c, SCEN6) for c in ('default', 'equal64')}
    T = {c: grf.v4_tests(res, c, SCEN6) for c in ('default', 'equal64')}
    all12 = grf.V4_BASELINES + grt.EXTRA + grf.V4_WAVELET
    files = {
        'T4.12_tab_main_default.tex': main_table6(V['default'], T['default'], 'default', 'tab:main_default'),
        'T4.12_tab_main_equal64.tex': main_table6(V['equal64'], T['equal64'], 'equal64', 'tab:main_equal64'),
        'T4.12_tab_si_ci_default.tex': split6(
            lambda sc, lb, ti: grt.ci_table(V['default'], 'default', lb, sc, LABEL6, grf.V4_ORDER, ti),
            'tab:si_ci_default', 'Default configuration.'),
        'T4.12_tab_si_tests_default.tex': split6(
            lambda sc, lb, ti: grt.tests_table(T['default'], lb, sc, LABEL6, grf.V4_ORDER, ti),
            'tab:si_tests_default', 'Default configuration.'),
        'T4.12_tab_si_ci_equal64.tex': split6(
            lambda sc, lb, ti: grt.ci_table(V['equal64'], 'equal64', lb, sc, LABEL6, all12, ti),
            'tab:si_ci_equal64', 'Equal window of 64 slots, with the simple smoothers SMA, EWMA-eq and Holt.'),
        'T4.12_tab_si_tests_equal64.tex': split6(
            lambda sc, lb, ti: grt.tests_table(T['equal64'], lb, sc, LABEL6, all12, ti),
            'tab:si_tests_equal64', 'Equal window of 64 slots.'),
    }
    for name, text in files.items():
        (out / name).write_text(text, encoding='utf-8', newline='\n')
        print(f'[ok] {name}')
    ctrl = pd.concat(V.values(), ignore_index=True)
    inputs = sorted({grf.v4_summary_path(res, c, s) for c in ('default', 'equal64') for s in SCEN6})
    inputs += [res / 'T1.6_stats' / d / f for d in ('causal', 'T2.2_window_sweep', 'T3.9_movielens')
               for f in ('all_method_summary.csv', 'all_paired_tests.csv')]
    meta = dict(task='T4.12 six-scenario tables', script='scripts/generate_t412_tables.py',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                control=dict(rows=int(len(ctrl)), max_rel_diff=float(ctrl['rel_diff'].max())),
                inputs={str(p.relative_to(res)).replace('\\', '/'): md5(p) for p in inputs},
                tables={n: hashlib.md5(t.encode('utf-8')).hexdigest() for n, t in files.items()},
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                platform=platform.platform())
    (out / 'generate_t412_tables_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'control: {len(ctrl)} rows, max relative difference {ctrl["rel_diff"].max():.2e}')
    print(f'{len(files)} tables written to {out}')


if __name__ == '__main__':
    main()
