r"""
Generate every LaTeX table of the paper and its Supplementary Information
=========================================================================
All numbers come from the result CSVs under --results (normally
results\revision_v5).  No number is typed by hand and no number is computed
beyond selection, rounding and the few derived quantities written to the
control CSV.  The main result tables use the same loaders as the figures
(scripts/generate_figures.py: v4_values, v4_tests), which check the
block-bootstrap statistics against summary_common_windows.csv of every run, so
a table and its figure can never disagree.  The result folders are read from
result_paths.json (next to this file).

Parts (--part), all written to --out (file name = the name in \input)
------------------------------------------------------------------------
  main   tab_main_default.tex, tab_main_equal64.tex   main result tables, six
             scenarios: one panel per metric, one column per scenario.  Bold
             value = best of the scenario at the printed precision;
             \blacktriangle / \triangledown = significantly better / worse than
             WSPI (block Wilcoxon, Holm per scenario x metric, alpha 0.05)
         tab_main4_default.tex, tab_main4_equal64.tex  the earlier four-scenario
             layout (YouTube and taxi only; not used in the paper)
         tables_main_run.json
         and, in --data-out (default: the folder 'figure_data' of
         result_paths.json; CSV and JSON only), the source data of the tables
         and figures: values_all.csv, tests_all.csv, control.csv,
         metadata/generate_tables_run.json
  si     si_ci_default, si_tests_default, si_ci_equal64, si_tests_equal64
             intervals and paired tests, six scenarios in two parts
         si_padding, si_features, si_levels, si_yt_gap, si_ml_profile,
         si_decomposition, si_robust_default, si_robust_equal64, si_hardware,
         si_runtime_grid, si_memory, si_runtime_real, si_ablation, si_fusion,
         si_selection, si_resp_strict, si_rsi_truth
         si_ci4_default, si_ci4_equal64, si_tests4_default, si_tests4_equal64,
         si_movielens, si_movielens_equal64, si_movielens_tests  (earlier
             four-scenario and MovieLens-only layout; not used in the SI)
         si_tables_sources.csv (table, source file, md5), si_tables_control.csv
         (every cross-check and derived value, with pass flag), tables_si_run.json
  all    both parts

The names of the three wavelet-based methods (DWT+AF, DTCWT+AF, WSPI) are
printed in bold where they label a row or a column.  In the SI tables,
\blacktriangle / \triangledown mark values significantly better / worse than the
reference of the table (WSPI; symmetric extension in si_padding; J = 3 in
si_levels) and \dagger in si_ablation and si_fusion marks values NOT
significantly different from the full index / the exponential form.

Table numbers of one document that are printed in the other (the SI and the
paper are compiled separately) are set in PAPER_TABLE and SI_NUM below.

Usage (from the project root)
-----------------------------
  python scripts/generate_tables.py --part all --results results/revision_v5 --out <folder>
Nothing is written to results except by --part main in --data-out (CSV and JSON).
"""
import argparse
import hashlib
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_figures as gf  # noqa: E402

ROOT = HERE.parent
PATHS = gf.PATHS

# Numbers of the paper tables cited in SI captions, and of the SI tables cited in
# paper captions (order of first citation; change here when the numbering changes).
PAPER_TABLE = {'tab:main_default': '8', 'tab:resp': '14'}
SI_NUM = {'default': ('S6', 'S7'), 'equal64': ('S8', 'S9'),
          'fig:si_spike_size': 'S4', 'tab:si_runtime_grid': 'S14', 'tab:si_runtime_real': 'S16'}


def src(key, *parts):
    """Source path relative to --results, for the SI tables (as recorded in si_tables_sources.csv)."""
    return '/'.join([PATHS[key], *parts])


SCEN = [s for s, _ in gf.V4_SCENARIOS]
SCEN_LABEL = dict(gf.V4_SCENARIOS)
ML = [s for s, _ in gf.ML_SCENARIOS]
ML_LABEL = dict(gf.ML_SCENARIOS)
MET = gf.MAIN_METRICS
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


def main_table(vals, tests, config, label, scen, scen_label, methods=gf.V4_ORDER):
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


# =================================================================== six-scenario result tables
SCEN4 = [s for s, _ in gf.V4_SCENARIOS]
ML2 = [s for s, _ in gf.ML_SCENARIOS]
SCEN6 = SCEN4 + ML2
LABEL6 = {**dict(gf.V4_SCENARIOS), **dict(gf.ML_SCENARIOS)}
# column heads of the main tables: the labels of Table 5 (tab:data)
HEAD6 = {'youtube_hourly': r'\shortstack{YouTube\\(1h)}', 'taxi_hourly': r'\shortstack{Taxi\\Hourly}',
         'taxi_30min': r'\shortstack{Taxi\\30min}', 'taxi_5min': r'\shortstack{Taxi\\5min}',
         'movielens_daily': r'\shortstack{MovieLens\\(1d)}', 'movielens_weekly': r'\shortstack{MovieLens\\(1w)}'}
MET_PANEL = {'ndcg@10': r'NDCG@10 $\uparrow$', 'spearman_rho': r'Spearman $\rho$ $\uparrow$',
             'rsi@10': r'RSI@10 $\uparrow$', 'robustness_distortion': r'$\Delta$Rank $\downarrow$'}


def tie_note(vals, cfg):
    t = (vals[vals['metric'] == 'rsi@10'].groupby('method')['ties_top21_share'].max()
         .reindex(gf.V4_ORDER).dropna())
    t = t[t >= TIE_NOTE].sort_values(ascending=False, kind='stable')
    parts = ', '.join(f'{m} ({v:.2f})' for m, v in t.items())
    return ('Methods with tied scores among the top 21 in 10\\% or more of the windows of at least one '
            f'scenario (largest share): {parts}; their NDCG@10 and RSI@10 depend on the fixed tie rule '
            f'(item order). Shares per scenario are in Supplementary Table~{SI_NUM[cfg][0]}.')


WAVELET_NAMES = ('DWT+AF', 'DTCWT+AF', 'WSPI')


def bold_wavelet_names(tex):
    """Print the names of the three wavelet-based methods in bold where they label a
    table row (first cell) or a column (\\shortstack header), as in Table 1 of the
    paper.  Only the name cell changes; bold values still mark the best value."""
    out = []
    for line in tex.split('\n'):
        for m in WAVELET_NAMES:
            if line.startswith(m + ' & '):
                line = r'\textbf{' + m + '}' + line[len(m):]
        line = re.sub(r'\\shortstack\{(DWT\+AF|DTCWT\+AF|WSPI)\\\\',
                      lambda g: r'\shortstack{\textbf{' + g.group(1) + '}' + '\\\\', line)
        out.append(line)
    return '\n'.join(out)


def main_table6(vals, tests, cfg, label):
    nwin = {s: int(vals[(vals['scenario'] == s) & (vals['metric'] == 'ndcg@10')
                        & (vals['method'] == 'WSPI')]['n_windows'].iloc[0]) for s in SCEN6}
    ci, eff = SI_NUM[cfg]
    cap = ('Results in the ' + CONFIG_TEXT[cfg] + '; mean over the common windows ('
           + ', '.join(f'{LABEL6[s]} {nwin[s]:,}' for s in SCEN6) + '). Bold: best value of the scenario. '
           + UP + ' / ' + DOWN + r': significantly better / worse than WSPI (block Wilcoxon test, '
           r'Holm correction, $\alpha=0.05$). 95\% confidence intervals are given in Supplementary Table~'
           + ci + ' and effect sizes in Supplementary Table~' + eff + '.')
    L = [r'\begin{table}[htbp]', r'\centering', r'\caption{' + cap + '}', r'\label{' + label + '}',
         r'\footnotesize', r'\setlength{\tabcolsep}{4pt}', r'\renewcommand{\arraystretch}{0.96}',
         r'\begin{tabular}{@{}l' + 'r' * len(SCEN6) + '@{}}', r'\toprule',
         'Method & ' + ' & '.join(HEAD6[s] for s in SCEN6) + r'\\']
    for met in gf.MAIN_METRICS:
        L += [r'\midrule', r'\multicolumn{' + str(len(SCEN6) + 1) + r'}{@{}l}{\textit{' + MET_PANEL[met] + r'}}\\']
        cols = {}
        for s in SCEN6:
            v = (vals[(vals['scenario'] == s) & (vals['metric'] == met)].set_index('method')['mean']
                 .reindex(gf.V4_ORDER))
            if v.isna().any():
                raise ValueError(f'{cfg}/{s}/{met}: missing method')
            cols[s] = (v, best_mask(v, met))
        for m in gf.V4_ORDER:
            cells = []
            for s in SCEN6:
                v, b = cols[s]
                x = fmt(v[m], DEC[met])
                x = r'\textbf{' + x + '}' if b[m] else x
                cells.append(x + marker(tests, s, met, m))
            L.append((r'\textbf{' + m + '}' if m in WAVELET_NAMES else m) + ' & ' + ' & '.join(cells) + r'\\')
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
    return bold_wavelet_names(a + b)


# =================================================================== Supplementary tables from the result CSVs
SCEN_SHORT = {'youtube_hourly': 'YouTube', 'taxi_hourly': 'Taxi 1h',
              'taxi_30min': 'Taxi 30m', 'taxi_5min': 'Taxi 5m'}
ORDER = ['AF', 'CompoundPop', 'EWMA', 'PFRF', 'RRD', 'VSE', 'DWT+AF', 'DTCWT+AF', 'WSPI']
WAVELET = ['DWT+AF', 'DTCWT+AF', 'WSPI']
MET4 = ['ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion']
MET3 = ['ndcg@10', 'rsi@10', 'robustness_distortion']
DAG = r'\textsuperscript{$\dagger$}'
TOL = 1e-9


class Ctx:
    """Collects sources and controls while the tables are built."""

    def __init__(self, results):
        self.results = results
        self.sources = []
        self.control = []
        self.table = None

    def read(self, rel, **kw):
        p = self.results / rel
        self.sources.append(dict(table=self.table, source=rel.replace('\\', '/'),
                                 md5=hashlib.md5(p.read_bytes()).hexdigest()))
        if rel.endswith('.json'):
            return json.loads(p.read_text(encoding='utf-8'))
        return pd.read_csv(p, **kw)

    def check(self, what, value, expected=None, ok=None, tol=TOL):
        if ok is None:
            ok = abs(float(value) - float(expected)) <= tol if expected is not None else True
        self.control.append(dict(table=self.table, check=what, value=value,
                                 expected='' if expected is None else expected, passed=bool(ok)))
        if not ok:
            raise SystemExit(f'[control failed] {self.table}: {what}: {value} vs {expected}')


# ---------------------------------------------------------------- formatting
def num(x, d):
    s = f'{x:.{d}f}'
    if s.startswith('-'):
        s = '$-$' + s[1:]
    return s


def signed(x, d):
    s = f'{x:+.{d}f}'
    return s.replace('-', '$-$')


def ci(m, lo, hi, d):
    return f'{num(m, d)} [{num(lo, d)}, {num(hi, d)}]'


def ci2(m, lo, hi, d):
    """Mean on the first line, interval on the second (narrow columns)."""
    return r'\shortstack{' + num(m, d) + r'\\{}[' + num(lo, d) + ', ' + num(hi, d) + ']}'


def sig3(x):
    """Three significant digits, plain notation."""
    if x == 0 or not np.isfinite(x):
        return '0'
    d = max(0, 2 - int(np.floor(np.log10(abs(x)))))
    return f'{x:,.{d}f}'


def mark(verdict):
    return DOWN if verdict == 'ref_better' else UP if verdict == 'ref_worse' else ''


def dagger(verdict):
    return DAG if verdict == 'n.s.' else ''


def head(caption, label, cols, size=r'\scriptsize', long=False, sep=None):
    L = []
    if not long:
        L += [r'\begin{table}[htbp]', r'\centering', size]
        if sep:
            L.append(r'\setlength{\tabcolsep}{' + sep + '}')
        L += [r'\caption{' + caption + '}', r'\label{' + label + '}',
              r'\begin{tabular}{@{}' + cols + '@{}}', r'\toprule']
    else:
        L += ['{' + size]
        if sep:
            L.append(r'\setlength{\tabcolsep}{' + sep + '}')
        L += [r'\begin{longtable}{@{}' + cols + '@{}}',
              r'\caption{' + caption + r'}\label{' + label + r'}\\', r'\toprule']
    return L


def foot(note=None, long=False, ncol=1):
    if long:
        L = [r'\bottomrule']
        if note:
            L.append(r'\multicolumn{' + str(ncol) + r'}{@{}p{\linewidth}@{}}{\footnotesize\textit{' + note + r'}}\\')
        L += [r'\end{longtable}', '}']
        return L
    L = [r'\bottomrule', r'\end{tabular}']
    if note:
        L += [r'\par\smallskip', r'\begin{minipage}{\linewidth}\footnotesize\textit{' + note + r'}\end{minipage}']
    L.append(r'\end{table}')
    return L


def group(text, ncol):
    return r'\multicolumn{' + str(ncol) + r'}{@{}l}{\textit{' + text + r'}}\\'


# ---------------------------------------------------------------- padding
def t_padding(c):
    c.table = 'padding'
    p = c.read(src('padding', 'padding_summary.csv'))
    t = c.read(src('stats_padding', 'all_paired_tests.csv'))
    share = c.read(src('padding', 'padded_share.csv'))
    ext = p[(p.layer == 'ext') & (p.subset == 'all')]
    modes = ['symmetric', 'reflect', 'edge', 'zero', 'periodic']
    methods = ['WSPI', 'DTCWT+AF', 'DWT+AF']

    def cell(sc, m, mode, met):
        r = ext[(ext.scenario == sc) & (ext.method == m) & (ext['mode'] == mode)]
        assert len(r) == 1
        v = r[met + '_mean'].iloc[0]
        s = num(v, DEC[met])
        if mode != 'symmetric':
            tt = t[(t.run_group == sc) & (t.scenario == m) & (t.metric == met)
                   & (t.reference == 'symmetric') & (t.method == mode)]
            assert len(tt) == 1
            s += mark(tt.verdict.iloc[0])
        return s

    # explicit padding (layer pad, windows 32-63 padded, later windows unchanged)
    pad = p[(p.layer == 'pad') & (p.subset == 'all') & (p.method == 'WSPI')]
    rng = {}
    for met in MET3:
        d = pad.pivot(index='scenario', columns='mode', values=met + '_mean')
        rng[met] = float(d.sub(d['reflect'], axis=0).abs().max().max())
        c.check(f'WSPI explicit padding: largest |mode - reflect| over the four scenarios, {met}', rng[met])
    sh = share[share.method == 'WSPI'].set_index('scenario')['padded_share']
    c.check('padded share YouTube', sh['youtube_hourly'])
    c.check('padded share taxi 5m', sh['taxi_5min'])
    # DWT+AF with zero extension against default WSPI
    gaps = []
    for sc in SCEN:
        w = ext[(ext.scenario == sc) & (ext.method == 'WSPI') & (ext['mode'] == 'symmetric')].iloc[0]
        d = ext[(ext.scenario == sc) & (ext.method == 'DWT+AF') & (ext['mode'] == 'zero')].iloc[0]
        c.check(f'{sc}: RSI@10 DWT+AF zero > WSPI symmetric',
                d['rsi@10_mean'] - w['rsi@10_mean'], ok=d['rsi@10_mean'] > w['rsi@10_mean'])
        gaps.append(w['ndcg@10_mean'] - d['ndcg@10_mean'])
    c.check('NDCG@10 gap WSPI - DWT+AF zero, min', min(gaps), ok=min(gaps) > 0)
    c.check('NDCG@10 gap WSPI - DWT+AF zero, max', max(gaps))

    cols = 'll' + 'r' * 6
    cap = (r'Padding and boundary extension of the three wavelet-based methods ($N=64$, $J=3$); '
           r'mean over the common windows. The rows vary the boundary extension of the transform '
           r'(all windows); symmetric is the default. ' + UP + ' / ' + DOWN +
           r': significantly better / worse than symmetric extension of the same method '
           r'(block Wilcoxon test, Holm correction, $\alpha=0.05$).')
    L = head(cap, 'tab:padding', cols)
    for pair in (SCEN[:2], SCEN[2:]):
        L.append(r' & & \multicolumn{3}{c}{' + SCEN_LABEL[pair[0]] + r'} & \multicolumn{3}{c}{'
                 + SCEN_LABEL[pair[1]] + r'}\\')
        L.append(r'\cmidrule(lr){3-5}\cmidrule(l){6-8}')
        L.append('Method & Extension & ' + ' & '.join(f'{MET_TEX[m]} {ARROW[m]}' for m in MET3) * 1
                 + ' & ' + ' & '.join(f'{MET_TEX[m]} {ARROW[m]}' for m in MET3) + r'\\')
        L.append(r'\midrule')
        for m in methods:
            for i, mode in enumerate(modes):
                row = [m if i == 0 else '', mode]
                row += [cell(sc, m, mode, met) for sc in pair for met in MET3]
                L.append(' & '.join(row) + r'\\')
            if m != methods[-1]:
                L.append(r'\addlinespace[2pt]')
        if pair is not None and pair[0] == SCEN[0]:
            L.append(r'\midrule')
    lo, hi = min(gaps), max(gaps)
    note = (f'Explicit padding (reflect by default) is used only in windows 32 to 63, before 64 slots of '
            f'history exist ({sh["youtube_hourly"] * 100:.1f}\\% of the common windows on YouTube and '
            f'{sh["taxi_5min"] * 100:.2f}\\% on the 5-minute taxi data). The other four padding modes '
            f'changed the means of WSPI by at most {rng["ndcg@10"]:.4f} in NDCG@10, '
            f'{rng["rsi@10"]:.4f} in RSI@10 and {rng["robustness_distortion"]:.2f} in $\\Delta$Rank. '
            'The DWT-based method is far more sensitive to the extension mode; with zero extension its '
            'RSI@10 exceeds that of the default WSPI in all four scenarios, at an NDCG@10 that is '
            f'{lo:.3f} to {hi:.3f} lower. We therefore do not attribute the stability of WSPI to '
            'shift-invariance alone.')
    return '\n'.join(L + foot(note)) + '\n'


# ---------------------------------------------------------------- features
def t_features(c):
    c.table = 'features'
    f = c.read(src('feature_relation', 'feature_relation_summary.csv'))
    rows_w = [('sp_R_WE', r'Spearman $(R, W_E)$'),
              ('sp_R_logmu', r'Spearman $(R, \log\mu_L)$'),
              ('sp_WE_logmu', r'Spearman $(W_E, \log\mu_L)$'),
              ('sp_mu_y', r'Spearman $(\mu_L, y)$'),
              ('pc_R_y_mu', r'Partial $(R, y \mid \mu_L)$'),
              ('pc_WE_y_mu', r'Partial $(W_E, y \mid \mu_L)$'),
              ('pc_S_y_mu', r'Partial $(R-W_E, y \mid \mu_L)$'),
              ('pc_R_y_muWE', r'Partial $(R, y \mid \mu_L, W_E)$'),
              ('pc_WE_y_muR', r'Partial $(W_E, y \mid \mu_L, R)$'),
              ('pc_D_y_muR', r'Partial $(D, y \mid \mu_L, R)$')]
    rows_p = [('n_item_windows', 'Item-windows', 0),
              ('eta2_WE_given_R', r'$\eta^2(W_E \mid R)$', 4),
              ('mi_R_WE_bits', r'Mutual information $(R; W_E)$, bits', 2),
              ('vif_logmu', r'VIF $\log\mu_L$', 2),
              ('vif_R', r'VIF $R$', 2),
              ('vif_WE', r'VIF $W_E$', 2)]
    ww = f[f.kind == 'within_window']
    pp = f[f.kind == 'pooled']
    sp = ww[ww.statistic == 'sp_R_WE'].set_index('scenario')['value']
    c.check('within-window Spearman(R, WE), max', sp.max())
    c.check('within-window Spearman(R, WE), min', sp.min())
    cap = (r'Relation of $R$ and $W_E$ over all item-windows scored by WSPI ($N=64$, $J=3$). '
           r'Upper part: mean over the windows of the within-window statistic [95\% block-bootstrap '
           r'confidence interval]; $y$ is the count in the next slot; partial correlations are Spearman '
           r'correlations of the residuals; $D=H(q)/\log_2 J$ is the evenness of the detail energy over the '
           r'detail levels. Lower part: pooled over all item-windows.')
    L = head(cap, 'tab:si_features', 'l' + 'c' * 4, sep='3pt')
    L.append('Statistic & ' + ' & '.join(SCEN_SHORT[s] for s in SCEN) + r'\\')
    L.append(r'\midrule')
    L.append(group('Within window', 5))
    for key, lab in rows_w:
        cells = []
        for sc in SCEN:
            r = ww[(ww.scenario == sc) & (ww.statistic == key)]
            assert len(r) == 1
            r = r.iloc[0]
            cells.append(ci2(r.value, r.ci_low, r.ci_high, 3))
        L.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    L.append(r'\midrule')
    L.append(group('Pooled', 5))
    for key, lab, d in rows_p:
        cells = []
        for sc in SCEN:
            r = pp[(pp.scenario == sc) & (pp.statistic == key)]
            assert len(r) == 1
            v = r.value.iloc[0]
            cells.append(f'{int(round(v)):,}' if d == 0 else num(v, d))
        L.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    nan = ww.groupby('scenario').n_nan.max()
    note = ('Statistics that depend on the next slot are undefined in windows where this slot is the '
            'same for all items or where fewer than ten items are scored; such windows are left out '
            f'(at most {int(nan.max())} windows per scenario).')
    return '\n'.join(L + foot(note)) + '\n'


# ---------------------------------------------------------------- levels
def t_levels(c):
    c.table = 'levels'
    s = c.read(src('stats_level_sweep', 'all_method_summary.csv'))
    t = c.read(src('stats_level_sweep', 'all_paired_tests.csv'))
    ls = c.read(src('level_sweep', 'level_sweep_summary.csv'))
    cap = (r'Decomposition level $J$ of WSPI and DTCWT+AF for $N=64$ and $N=32$; mean [95\% '
           r'block-bootstrap confidence interval] over the common windows. ' + UP + ' / ' + DOWN +
           r': significantly better / worse than $J=3$ with the same method and $N$ '
           r'(block Wilcoxon test, Holm correction, $\alpha=0.05$).')
    L = head(cap, 'tab:si_levels', 'lll' + 'l' * 4, long=True, sep='3pt')
    hdr = r'Method & $N$ & $J$ & ' + ' & '.join(f'{MET_TEX[m]} {ARROW[m]}' for m in MET4) + r'\\'
    L += [hdr, r'\midrule', r'\endfirsthead', r'\toprule', hdr, r'\midrule', r'\endhead']
    nchk = 0
    for sc in SCEN:
        L.append(group(SCEN_LABEL[sc], 7))
        for m in ['WSPI', 'DTCWT+AF']:
            first = True
            for N, Js in ((64, [2, 3, 4, 5]), (32, [2, 3, 4])):
                rg = f'{sc}/W{N:03d}'
                for j in Js:
                    cells = []
                    for met in MET4:
                        r = s[(s.run_group == rg) & (s.scenario == m) & (s.metric == met) & (s.method == f'J{j}')]
                        assert len(r) == 1, (rg, m, met, j)
                        r = r.iloc[0]
                        q = ls[(ls.scenario == sc) & (ls.window == N) & (ls.method == m) & (ls.level == j)]
                        assert len(q) == 1
                        if abs(q[met + '_mean'].iloc[0] - r['mean']) > 1e-9 * max(1, abs(r['mean'])):
                            c.check(f'{rg} {m} J{j} {met} mean', r['mean'], q[met + '_mean'].iloc[0])
                        nchk += 1
                        cell = ci(r['mean'], r.ci_low, r.ci_high, DEC[met])
                        if j != 3:
                            tt = t[(t.run_group == rg) & (t.scenario == m) & (t.metric == met)
                                   & (t.reference == 'J3') & (t.method == f'J{j}')]
                            assert len(tt) == 1
                            cell += mark(tt.verdict.iloc[0])
                        cells.append(cell)
                    L.append(' & '.join([m if first else '', str(N) if j == Js[0] else '', str(j)] + cells) + r'\\')
                    first = False
        if sc != SCEN[-1]:
            L.append(r'\midrule')
    c.check('means of the statistics equal level_sweep_summary (values compared)', nchk)
    return '\n'.join(L + foot(long=True, ncol=7)) + '\n'


# ---------------------------------------------------------------- YouTube gap
def t_yt_gap(c):
    c.table = 'yt_gap'
    g = c.read(src('youtube_provenance', 'gap_sensitivity.csv'))
    c.check('control_pass of all rows', int(g.control_pass.all()), ok=bool(g.control_pass.all()))
    n_all, n_ex = int(g.n_all.iloc[0]), int(g.n_excl.iloc[0])
    c.check('largest |change|', g['diff'].abs().max())
    c.check('rank changes', int((g.rank_all != g.rank_excl).sum()), ok=(g.rank_all == g.rank_excl).all())
    mets = ['ndcg@10', 'spearman_rho', 'rsi@10']
    cap = ('YouTube: mean over the common windows with and without the ' + str(n_all - n_ex) +
           ' windows whose input contains the collection gap of 19 May 2018 (All: ' + f'{n_all}'
           + ' windows; Without gap: ' + f'{n_ex}' + ' windows).')
    L = head(cap, 'tab:yt_gap', 'l' + 'r' * 6)
    L.append(' & ' + ' & '.join(r'\multicolumn{2}{c}{' + MET_TEX[m] + '}' for m in mets) + r'\\')
    L.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}')
    L.append('Method & ' + ' & '.join(['All', 'Without gap'] * 3) + r'\\')
    L.append(r'\midrule')
    for m in ORDER:
        cells = []
        for met in mets:
            r = g[(g.method == m) & (g.metric == met)]
            assert len(r) == 1
            cells += [num(r.mean_all.iloc[0], 4), num(r.mean_excl.iloc[0], 4)]
        L.append(m + ' & ' + ' & '.join(cells) + r'\\')
    note = (f'The largest change of a mean is {g["diff"].abs().max():.4f}; the order of the methods '
            'does not change in any metric.')
    return '\n'.join(L + foot(note)) + '\n'


# ---------------------------------------------------------------- robustness
COND = [('size2', r'$2\times$'), ('size5', r'$5\times$'), ('size10', r'$10\times$'),
        ('size20', r'$20\times$'), ('size50', r'$50\times$'),
        ('pos_middle', 'Middle slot'), ('pos_first', 'First slot'), ('pos_random', 'Random slot'),
        ('dur3', '3 slots'), ('dur6', '6 slots'),
        ('noise_poisson', 'Poisson noise'), ('noise_gauss_snr10', 'Gaussian, 10 dB'),
        ('noise_gauss_snr0', 'Gaussian, 0 dB')]


def t_decomp(c):
    c.table = 'decomposition'
    r = c.read(src('robustness', 'robustness_summary.csv'))
    w = r[(r.method == 'WSPI') & (r.config == 'default')]
    y = w[(w.scenario == 'youtube_hourly') & (w.condition == 'size10')].iloc[0]
    c.check('YouTube 10x: dlogmu', y.dlogmu_mean)
    c.check('YouTube 10x: dexpo', y.dexpo_mean)
    cap = (r'Change of the two terms of WSPI for the 50 perturbed items (default configuration; mean '
           r'over the common windows and five seeds). $\Delta\log\mu_L$: change of the log trend term; '
           r'$\Delta S$: change of $\alpha R-\beta W_E$; Up: share of perturbed items whose score rose. '
           r'Spike rows: $10\times$ the mean in the last slot unless stated; position and duration rows '
           r'use a $10\times$ spike.')
    cols = 'l' + 'r' * 12
    L = head(cap, 'tab:si_decomp', cols, sep='2.5pt')
    L.append(' & ' + ' & '.join(r'\multicolumn{3}{c}{' + SCEN_SHORT[s] + '}' for s in SCEN) + r'\\')
    L.append(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}')
    L.append('Perturbation & ' + ' & '.join([r'$\Delta\log\mu_L$', r'$\Delta S$', 'Up'] * 4) + r'\\')
    L.append(r'\midrule')
    for key, lab in COND:
        cells = []
        for sc in SCEN:
            q = w[(w.scenario == sc) & (w.condition == key)]
            assert len(q) == 1
            q = q.iloc[0]
            cells += [signed(q.dlogmu_mean, 3), signed(q.dexpo_mean, 3), num(q.up_mean, 2)]
        L.append(lab + ' & ' + ' & '.join(cells) + r'\\')
    return '\n'.join(L + foot()) + '\n'


def t_robust(c, config):
    c.table = 'robust_default' if config == 'default' else 'robust_equal64'
    r = c.read(src('robustness', 'robustness_summary.csv'))
    t = c.read(src('stats_robustness', 'all_paired_tests.csv'))
    conds = [('size10', r'$10\times$ last'), ('pos_middle', 'Middle'), ('pos_first', 'First'),
             ('pos_random', 'Random'), ('dur3', '3 slots'), ('dur6', '6 slots'),
             ('noise_poisson', 'Poisson'), ('noise_gauss_snr10', r'Gauss.\ 10\,dB'),
             ('noise_gauss_snr0', r'Gauss.\ 0\,dB')]
    rr = r[r.config == config]
    spread = []
    if config == 'default':
        for sc, v in zip(SCEN, [41.88, 10.24, 8.72, 4.84]):
            q = rr[(rr.scenario == sc) & (rr.method == 'WSPI') & (rr.condition == 'pos_random')].dr_mean.iloc[0]
            c.check(f'{sc}: WSPI random-slot Delta Rank (text of the failure-case subsection)', round(q, 2), v, tol=1e-9)
            top = rr[(rr.scenario == sc) & (rr.condition == 'pos_random')].sort_values('dr_mean').method.iloc[-1]
            c.check(f'{sc}: largest random-slot Delta Rank is WSPI', top, ok=top == 'WSPI')
    else:
        for sc in SCEN:
            for key, _ in conds:
                q = rr[(rr.scenario == sc) & (rr.condition == key)].set_index('method').dr_mean
                c.check(f'{sc} {key}: RRD below WSPI (text of the results subsection)', q['RRD'] - q['WSPI'],
                        ok=q['RRD'] < q['WSPI'])
    cfg = ('default configuration (baselines 7 slots, wavelet-based methods 64 slots)'
           if config == 'default' else 'equal window of 64 slots for all methods')
    cap = (r'Wider perturbations, ' + cfg + r': $\Delta$Rank (lower is better), mean over the common '
           r'windows and five seeds. Every method sees the same 50 low-activity items and the same '
           r'corruption. Columns: a $10\times$ spike in the last slot (reference cell), in the middle, '
           r'first or a random slot of the 64-slot window, lasting 3 or 6 slots, and continuous Poisson '
           r'or Gaussian noise (SNR 10 and 0\,dB). ' + UP + ' / ' + DOWN +
           r': significantly better / worse than WSPI (block Wilcoxon test, Holm correction, $\alpha=0.05$). '
           r'Spike sizes are shown in Supplementary Fig.~' + SI_NUM['fig:si_spike_size'] + '.')
    L = head(cap, 'tab:si_robust_' + ('default' if config == 'default' else 'equal64'),
             'l' + 'r' * len(conds), sep='3pt')
    L.append('Method & ' + ' & '.join(lab for _, lab in conds) + r'\\')
    L.append(r'\midrule')
    for sc in SCEN:
        L.append(group(SCEN_LABEL[sc], len(conds) + 1))
        for m in ORDER:
            cells = []
            for key, _ in conds:
                q = rr[(rr.scenario == sc) & (rr.method == m) & (rr.condition == key)]
                assert len(q) == 1
                q = q.iloc[0]
                spread.append((q.dr_seed_max - q.dr_seed_min, q.dr_mean))
                s = num(q.dr_mean, 2)
                if m != 'WSPI':
                    tt = t[(t.run_group == f'{sc}/{"default" if config == "default" else "eq64"}')
                           & (t.scenario == key) & (t.metric == 'robustness_distortion')
                           & (t.reference == 'WSPI') & (t.method == m)]
                    assert len(tt) == 1, (sc, key, m)
                    s += mark(tt.verdict.iloc[0])
                cells.append(s)
            L.append(m + ' & ' + ' & '.join(cells) + r'\\')
        if sc != SCEN[-1]:
            L.append(r'\midrule')
    sp_abs = max(a for a, _ in spread)
    sp = max(a / m for a, m in spread if m >= 1)
    c.check('largest seed range of a cell (ranks)', sp_abs)
    c.check('largest seed range / mean, cells with mean >= 1', sp)
    note = ('Low-activity items: items whose mean over the last 64 slots is non-zero and at most the median '
            'among the items eligible for all methods. Over the five seeds the mean $\\Delta$Rank of a cell '
            f'varied by at most {sp_abs:.2f} ranks, and by at most {sp * 100:.1f}\\% of its value in the cells with '
            'a mean of at least one rank.')
    if config == 'default':
        note += (' In the cell $10\\times$, last slot, one slot, the common test ranks the methods as the '
                 'test of Table~' + PAPER_TABLE['tab:main_default'] + ' does, except that RRD and VSE, whose values are close, swap places in '
                 'three of the four scenarios; WSPI is second after PFRF in both tests and all four scenarios.')
    return '\n'.join(L + foot(note)) + '\n'


# ---------------------------------------------------------------- cost
# PyWavelets 1.9.0 is the version installed for all V5 runs, but its generated
# pywt/version.py still says '1.8.0' (checked against the official cp312 win_amd64
# wheel of 1.9.0, git_revision c7bca20).  The run metadata therefore record 1.8.0;
# the paper, the SI and requirements.txt give the installed version.
PYWT_INSTALLED = {'1.8.0': '1.9.0'}


def t_hardware(c):
    c.table = 'hardware'
    b = c.read(src('runtime', 'bench', 'metadata', 'bench_run.json'))
    m = c.read(src('runtime', 'memory', 'metadata', 'memory_run.json'))
    h, hm = b['hardware'], m.get('hardware', {})
    for k in ('cpu_name', 'physical_cores', 'ram_bytes', 'platform'):
        if k in hm:
            c.check(f'same {k} in bench and memory runs', str(h[k]), ok=hm[k] == h[k])
    v = h['versions']
    rows = [('Processor', h['cpu_name']),
            ('Cores / logical processors', f'{h["physical_cores"]} / {h["logical_cpus"]}'),
            ('Memory', f'{h["ram_gib"]:.2f} GiB'),
            ('Operating system', f'Windows {h["release"]} ({h["version"]})'),
            ('Threads', f'{h["threads_setting"]} (OMP, MKL, OpenBLAS, NumExpr and vecLib limited to 1)'),
            ('BLAS', f'{h["blas"]["name"]} {h["blas"]["version"]}'),
            ('Python', f'{v["python"]} ({v["python_impl"]})'),
            ('NumPy / SciPy / pandas', f'{v["numpy"]} / {v["scipy"]} / {v["pandas"]}'),
            ('PyWavelets / dtcwt', f'{PYWT_INSTALLED.get(v["pywt"], v["pywt"])} / {v["dtcwt"]}'),
            ('Synthetic data', b['data'].replace('lognormal(1, 1)', r'log-normal ($\mu=1$, $\sigma=1$)')),
            ('Repeats', f'{b["repeats"]} (median and interquartile range reported)'),
            ('Batch size for large catalogues', f'{b["chunk"]:,} items'),
            ('Seed', str(b['seed']))]
    cap = ('Hardware and software of the cost measurements (Section 4.7 and Supplementary Tables '
           + SI_NUM['tab:si_runtime_grid'] + ' to ' + SI_NUM['tab:si_runtime_real'] + ').')
    L = head(cap, 'tab:si_hardware', 'll', size=r'\small')
    L.append(r'Item & Value\\')
    L.append(r'\midrule')
    for k, val in rows:
        L.append(f'{k} & {val}' + r'\\')
    return '\n'.join(L + foot()) + '\n'


def t_runtime_grid(c):
    c.table = 'runtime_grid'
    g = c.read(src('runtime', 'bench', 'runtime_grid.csv'))
    Ms = sorted(g.M.unique())
    w = g[(g.method == 'WSPI') & (g.N == 64) & (g.M == 10000)].us_per_item_median.iloc[0]
    c.check('WSPI N=64, 10^4 items, us per item', round(w, 2), 11.74)

    def mlab(M):
        return '1' if M == 1 else f'$10^{{{int(round(np.log10(M)))}}}$'
    cap = (r'Scoring time per item and window in microseconds: median [first quartile, third quartile] of '
           r'ten repeats, for each method, window length $N$ and number of items $M$ (synthetic counts, one '
           r'thread; batches of $10^4$ items for $M\ge 10^5$).')
    L = head(cap, 'tab:si_runtime_grid', 'lr' + 'r' * len(Ms), long=True, sep='3pt')
    hdr = r'Method & $N$ & ' + ' & '.join(r'$M=$' + mlab(M) for M in Ms) + r'\\'
    L += [hdr, r'\midrule', r'\endfirsthead', r'\toprule', hdr, r'\midrule', r'\endhead']
    for m in ORDER:
        sub = g[g.method == m]
        for i, N in enumerate(sorted(sub.N.unique())):
            cells = []
            for M in Ms:
                q = sub[(sub.N == N) & (sub.M == M)]
                assert len(q) == 1
                q = q.iloc[0]
                cells.append(r'\shortstack{' + sig3(q.us_per_item_median) + r'\\{}[' + sig3(q.us_per_item_q1)
                             + ', ' + sig3(q.us_per_item_q3) + ']}')
            L.append(' & '.join([m if i == 0 else '', str(N)] + cells) + r'\\')
        if m != ORDER[-1]:
            L.append(r'\addlinespace[2pt]')
    return '\n'.join(L + foot(long=True, ncol=len(Ms) + 2)) + '\n'


def t_memory(c):
    c.table = 'memory'
    tm = c.read(src('runtime', 'memory', 'tracemalloc_grid.csv'))
    rs = c.read(src('runtime', 'memory', 'rss_grid.csv'))
    Ns = [7, 32, 64, 128, 256]
    q = tm[(tm.method == 'WSPI') & (tm.N == 64) & (tm.batch == 10000)].traced_peak_bytes_per_item.iloc[0]
    c.check('WSPI traced bytes per item, N=64, batch 10^4', round(q), 3170)
    q = rs[(rs.method == 'WSPI') & (rs.N == 64) & (rs.case == 'chunked')].rss_sampled_peak_delta_bytes.iloc[0]
    c.check('WSPI RSS growth, 10^6 items in batches, N=64 (MiB)', round(q / 2**20), 39)
    cap = (r'Memory. Traced: peak memory allocated by Python while one batch of $10^4$ items is scored, '
           r'per item, in bytes (tracemalloc). Coef.: bytes of wavelet coefficients per item for $N=64$. '
           r'RSS: growth of the resident memory of the process in MiB while $10^6$ items are scored in '
           r'batches of $10^4$, or $10^5$ items in one batch.')
    cols = 'l' + 'r' * 5 + 'r' + 'r' * 4
    L = head(cap, 'tab:si_memory', cols, sep='3pt')
    L.append(r' & \multicolumn{5}{c}{Traced, bytes per item} & Coef. & \multicolumn{2}{c}{RSS, $10^6$ in batches} '
             r'& \multicolumn{2}{c}{RSS, $10^5$ at once}\\')
    L.append(r'\cmidrule(lr){2-6}\cmidrule(lr){7-7}\cmidrule(lr){8-9}\cmidrule(l){10-11}')
    L.append('Method & ' + ' & '.join(f'$N={n}$' for n in Ns) + r' & $N=64$ & $N=64$ & $N=256$ & $N=64$ & $N=256$\\')
    L.append(r'\midrule')
    for m in ORDER:
        cells = []
        for n in Ns:
            q = tm[(tm.method == m) & (tm.N == n) & (tm.batch == 10000)]
            cells.append('--' if q.empty else f'{q.traced_peak_bytes_per_item.iloc[0]:,.0f}')
        q = tm[(tm.method == m) & (tm.N == 64) & (tm.batch == 10000)]
        cells.append(f'{q.coef_bytes_per_item.iloc[0]:,.0f}')
        for case in ('chunked', 'batch'):
            for n in (64, 256):
                q = rs[(rs.method == m) & (rs.N == n) & (rs.case == case)]
                assert len(q) == 1
                cells.append(f'{q.rss_sampled_peak_delta_bytes.iloc[0] / 2**20:.1f}')
        L.append(m + ' & ' + ' & '.join(cells) + r'\\')
    note = ('The traced peak grows with the batch size, because the transform of a batch is held in memory at '
            'once. When a large catalogue is scored in batches, the memory of one batch is released and '
            'reused by the next, so the resident memory grows by far less than the traced peak times the '
            'number of items.')
    return '\n'.join(L + foot(note)) + '\n'


def t_runtime_real(c):
    c.table = 'runtime_real'
    r = c.read(src('runtime', 'real_summary_all.csv'))
    w = r[(r.method == 'WSPI') & (r.scenario == 'taxi_5min')].iloc[0]
    c.check('WSPI 5-minute taxi score share', round(w.score_share, 3), 0.339)
    cap = (r'Run time of the evaluation on the real data (default configuration, all windows, one thread). '
           r'Score: scoring the items; Robust.: the robustness test (50 rescorings and rankings per window); '
           r'Other: metrics and bookkeeping. Latency: time to score the whole catalogue of one window.')
    cols = 'l' + 'r' * 9
    L = head(cap, 'tab:si_runtime_real', cols, sep='3pt')
    L.append(r' & & \multicolumn{4}{c}{Seconds} & & \multicolumn{3}{c}{Latency, ms}\\')
    L.append(r'\cmidrule(lr){3-6}\cmidrule(l){8-10}')
    L.append(r'Method & Item-windows & Total & Score & Robust. & Other & Score share & Median & 95th pct. & Max\\')
    L.append(r'\midrule')
    for sc in SCEN:
        n = r[r.scenario == sc].n_windows
        L.append(group(SCEN_LABEL[sc] + f' ({int(n.max()):,} windows)', 10))
        for m in ORDER:
            q = r[(r.scenario == sc) & (r.method == m)]
            assert len(q) == 1
            q = q.iloc[0]
            L.append(' & '.join([m, f'{int(q.item_windows):,}', sig3(q.total_s), sig3(q.score_s),
                                 sig3(q.robust_s), sig3(q.other_s), num(q.score_share, 3),
                                 sig3(q.latency_ms_median), sig3(q.latency_ms_p95), sig3(q.latency_ms_max)]) + r'\\')
        if sc != SCEN[-1]:
            L.append(r'\midrule')
    return '\n'.join(L + foot()) + '\n'


# ---------------------------------------------------------------- ablation and fusion
ABL = [('WSPI', r'WSPI, $\mu_L e^{R-W_E}$'), ('Trend', r'Trend only, $\mu_L$'),
       ('Trend+R', r'Trend $+\,R$, $\mu_L e^{R}$'), ('Trend+WE', r'Trend $+\,W_E$, $\mu_L e^{-W_E}$'),
       ('DWT-WSPI', r'DWT: WSPI'), ('DWT-Trend', r'DWT: trend only'),
       ('DWT-Trend+R', r'DWT: trend $+\,R$'), ('DWT-Trend+WE', r'DWT: trend $+\,W_E$')]
FUS = [('WSPI', r'Exponential, $\mu_L e^{R-W_E}$'), ('Linear', r'Linear, $\mu_L(1+R-W_E)$'),
       ('Product', r'Product, $\mu_L R(1-W_E)$'), ('Additive', r'Additive, $\mu_L+R-W_E$')]


def _ablation(c, family, variants, label, cap):
    s = c.read(src('stats_ablation', 'all_method_summary.csv'))
    t = c.read(src('stats_ablation', 'all_paired_tests.csv'))
    a = c.read(src('ablation', 'ablation_summary.csv'))
    L = head(cap, label, 'l' + 'l' * 4, sep='3pt')
    L.append('Variant & ' + ' & '.join(f'{MET_TEX[m]} {ARROW[m]}' for m in MET4) + r'\\')
    L.append(r'\midrule')
    n = 0
    for sc in SCEN:
        nw = a[(a.scenario == sc) & (a.family == family)].n_windows
        L.append(group(SCEN_LABEL[sc] + f' ({int(nw.iloc[0]):,} windows)', 5))
        for v, lab in variants:
            cells = []
            for met in MET4:
                r = s[(s.run_group == sc) & (s.scenario == family) & (s.metric == met) & (s.method == v)]
                assert len(r) == 1, (sc, family, met, v)
                r = r.iloc[0]
                q = a[(a.scenario == sc) & (a.family == family) & (a.variant == v)][met + '_mean'].iloc[0]
                if abs(q - r['mean']) > 1e-9 * max(1, abs(q)):
                    c.check(f'{sc} {v} {met} mean', r['mean'], q)
                n += 1
                cell = ci(r['mean'], r.ci_low, r.ci_high, DEC[met])
                if v != 'WSPI':
                    tt = t[(t.run_group == sc) & (t.scenario == family) & (t.metric == met)
                           & (t.reference == 'WSPI') & (t.method == v)]
                    assert len(tt) == 1
                    cell += dagger(tt.verdict.iloc[0])
                cells.append(cell)
            L.append(lab + ' & ' + ' & '.join(cells) + r'\\')
        if sc != SCEN[-1]:
            L.append(r'\midrule')
    c.check('means of the statistics equal ablation_summary (values compared)', n)
    return '\n'.join(L + foot()) + '\n'


def t_ablation(c):
    c.table = 'ablation'
    cap = (r'Ablation of the WSPI components in all four scenarios ($N=64$, $J=3$, $\alpha=\beta=1$); mean '
           r'[95\% block-bootstrap confidence interval] over the common windows. DWT: db4 with symmetric '
           r'extension. ' + DAG + r' Not significantly different from the full index '
           r'(block Wilcoxon test, Holm correction).')
    return _ablation(c, 'ablation', ABL, 'tab:si_ablation', cap)


def t_fusion(c):
    c.table = 'fusion'
    cap = (r'Fusion function: four ways to combine the same features with unit weights ($N=64$, $J=3$); mean '
           r'[95\% block-bootstrap confidence interval] over the common windows. The product form ranks items '
           r'like the geometric mean. ' + DAG + r' Not significantly different from the exponential form '
           r'(block Wilcoxon test, Holm correction).')
    return _ablation(c, 'fusion', FUS, 'tab:si_fusion', cap)


# ---------------------------------------------------------------- selection
def t_selection(c):
    c.table = 'selection'
    s = c.read(src('param_grid', 'selection', 'selection.csv'))
    cap = (r'Selection of $(\alpha,\beta)$ and $J$ with the 30/70 split. On the tuning part (first 30\% of the '
           r'common windows) a setting is admissible if its NDCG@10 is at least 0.99 times the best; the '
           r'admissible setting with the highest RSI@10 is selected. The last three columns compare the '
           r'selected setting with the default on the test part (selected / default).')
    L = head(cap, 'tab:si_selection', 'llrrlccc', sep='3pt')
    L.append(r'Scenario & Parameter & Tune / test & Admissible & Selected & NDCG@10 $\uparrow$ & RSI@10 $\uparrow$ '
             r'& $\Delta$Rank $\downarrow$\\')
    L.append(r'\midrule')
    for sc in SCEN:
        for i, par in enumerate(['alpha_beta', 'J']):
            q = s[(s.scenario == sc) & (s.param == par)]
            assert len(q) == 1
            q = q.iloc[0]
            if par == 'J':
                sel = f'$J={int(q.selected_J)}$'
                plab = '$J$'
            else:
                sel = f'$({q.selected_alpha:g}, {q.selected_beta:g})$'
                plab = r'$(\alpha,\beta)$'
            cells = [SCEN_LABEL[sc] if i == 0 else '', plab, f'{int(q.n_tune):,} / {int(q.n_test):,}',
                     f'{int(q.n_feasible)} of {int(q.n_configs)}', sel,
                     f'{num(q["test_ndcg@10_selected"], 4)} / {num(q["test_ndcg@10_default"], 4)}',
                     f'{num(q["test_rsi@10_selected"], 4)} / {num(q["test_rsi@10_default"], 4)}',
                     f'{num(q.test_robustness_distortion_selected, 2)} / {num(q.test_robustness_distortion_default, 2)}']
            L.append(' & '.join(cells) + r'\\')
        if sc != SCEN[-1]:
            L.append(r'\addlinespace[2pt]')
    note = 'Default: $\\alpha=\\beta=1$ and $J=3$. The J rows use $\\alpha=\\beta=1$; the $(\\alpha,\\beta)$ rows use $J=3$.'
    return '\n'.join(L + foot(note)) + '\n'


# ---------------------------------------------------------------- MovieLens profile
def t_ml_profile(c):
    c.table = 'ml_profile'
    g = c.read(src('movielens', 'data_prep', 'granularity_profile.csv')).set_index('file')
    y = c.read(src('movielens', 'data_prep', 'year_profile.csv'))
    j = c.read(src('movielens', 'data_prep', 'prep_summary.json'))
    tot = y[y.year.astype(str) == 'all'].iloc[0]
    y = y[y.year.astype(str) != 'all']
    share = float(tot.share_on_user_first_day)
    c.check('first-day share of ratings 1998-2023 (row all)', round(share, 2), 0.54)
    c.check('sum of the yearly ratings = row all', int(y.ratings.sum()), int(tot.ratings), tol=0)
    c.check('row all = ratings in the period (prep_summary, daily)', int(tot.ratings),
            j['files']['daily']['ratings_in_period'], tol=0)
    c.check('weighted yearly first-day share = row all',
            float((y.ratings * y.share_on_user_first_day).sum() / y.ratings.sum()), share, tol=1e-9)
    cap = (r'MovieLens (ML-32M) in the period used, 1 January 1998 to 12 October 2023. (a) Data sets after '
           r'the filter on the total count ($\ge 24$ ratings). (b) Ratings per year (all movies); first-day '
           r'share: share of the ratings made on the user\textquotesingle s first day of rating.')
    L = head(cap, 'tab:si_ml_profile', 'lrrrr', size=r'\footnotesize')
    L.append(r'\multicolumn{5}{@{}l}{\textit{(a) Data sets}}\\')
    L.append(r' & \multicolumn{2}{r}{Daily} & \multicolumn{2}{r}{Weekly}\\')
    L.append(r'\midrule')
    rows = [('Movies', lambda r: f'{int(r["items"]):,}'), ('Slots', lambda r: f'{int(r.slots):,}'),
            ('First / last slot', lambda r: f'{r.first_slot} / {r.last_slot}'),
            ('Ratings', lambda r: f'{int(r.ratings):,}'),
            ('Share of zero movie-slots', lambda r: num(r.zero_share_item_slot, 3)),
            ('Eligible movies per window, wavelet (median)', lambda r: f'{int(r.wavelet_W64_min32_eligible_median):,}'),
            ('Eligible movies per window, baselines (median)', lambda r: f'{r.baseline_W7_min3_eligible_median:,.1f}'.replace('.0', '')),
            ('Largest count in a slot (median)', lambda r: f'{r.max_count_per_slot_median:g}'),
            ('Tenth-largest count in a slot (median)', lambda r: f'{r.count_rank10_per_slot_median:g}')]
    for lab, f in rows:
        L.append(lab + r' & \multicolumn{2}{r}{' + f(g.loc['daily']) + r'} & \multicolumn{2}{r}{' + f(g.loc['weekly']) + r'}\\')
    L.append(r'\midrule')
    L.append(r'\multicolumn{5}{@{}l}{\textit{(b) Per year}}\\')
    L.append(r'Year & Ratings & \shortstack{Active\\movies} & \shortstack{Days with\\data} & \shortstack{First-day\\share}\\')
    L.append(r'\midrule')
    for _, r in y.iterrows():
        L.append(f'{int(float(r.year))} & {int(r.ratings):,} & {int(r.active_movies):,} & {int(r.days_with_data)} & '
                 f'{num(r.share_on_user_first_day, 3)}' + r'\\')
    L.append(r'\midrule')
    L.append(f'1998--2023 & {int(tot.ratings):,} & {int(tot.active_movies):,} & {int(tot.days_with_data):,} & '
             f'{num(share, 3)}' + r'\\')
    return '\n'.join(L + foot()) + '\n'


# ---------------------------------------------------------------- responsiveness
RESP_COLS = [('WSPI', 'default'), ('DTCWT+AF', 'default'), ('DWT+AF', 'default'), ('AF', 'default'),
             ('RRD', 'default'), ('RRD', 'equal64'), ('CompoundPop', 'equal64')]


def t_resp_strict(c):
    c.table = 'resp_strict'
    s = c.read(src('responsiveness', 'responsiveness_summary.csv'))
    main = s[(s.variant == 'main') & (s.config == 'default') & (s.method == 'WSPI')].set_index('scenario')
    c.check('main variant YouTube WSPI miss rate (paper table of responsiveness)', round(main.loc['youtube_hourly'].miss_rate, 3), 0.312)
    st = s[s.variant == 'strict']
    cap = (r'Responsiveness with a stricter entry rule: the item was outside the ground-truth Top-20 (instead of '
           r'the Top-10) in the six slots before the entry. Miss rate / median delay in slots, as in Table~' + PAPER_TABLE['tab:resp'] + r'; '
           r'-- : median undefined (more than half of the entries missed). The window length of each method is '
           r'given in parentheses.')
    L = head(cap, 'tab:si_resp_strict', 'lr' + 'c' * len(RESP_COLS), sep='3.3pt')
    L.append('Scenario & Entries & ' + ' & '.join(r'\shortstack{' + m + r'\\(' + ('64' if (m in WAVELET or cfg == 'equal64') else '7') + ')}'
                                                 for m, cfg in RESP_COLS) + r'\\')
    L.append(r'\midrule')
    for sc in SCEN:
        n = st[(st.scenario == sc) & (st.config == 'default') & (st.method == 'WSPI')].n_events.iloc[0]
        cells = []
        for m, cfg in RESP_COLS:
            q = st[(st.scenario == sc) & (st.config == cfg) & (st.method == m)]
            assert len(q) == 1
            q = q.iloc[0]
            c.check(f'{sc} {m} {cfg}: same entries in both configurations', int(q.n_events), int(n), tol=0)
            med = '--' if bool(q.median_undefined) else f'{q.median_delay:g}'
            cells.append(f'{num(q.miss_rate, 3)} / {med}')
        L.append(f'{SCEN_LABEL[sc]} & {int(n):,} & ' + ' & '.join(cells) + r'\\')
    return '\n'.join(L + foot()) + '\n'


def t_rsi_truth(c):
    c.table = 'rsi_truth'
    f = c.read(src('responsiveness', 'all_rsi_failures.csv'))
    d = f[f.config == 'default']
    af = d[d.method == 'AF'].set_index('scenario')
    c.check('AF rho min (text of the failure-case subsection)', round(af.spearman_rsi_vs_truth_seen.min(), 2), 0.24)
    c.check('AF rho max', round(af.spearman_rsi_vs_truth_seen.max(), 2), 0.58)
    c.check('WSPI rho min', round(af.spearman_ref_rsi_vs_truth_seen.min(), 2), -0.09)
    c.check('WSPI rho max', round(af.spearman_ref_rsi_vs_truth_seen.max(), 2), 0.18)
    cap = (r'Rank changes and real changes. Spearman correlation, over the common windows, between the RSI@10 '
           r'of a method in a window and the RSI@10 of the ground truth visible to that window (the change of '
           r'the true Top-10 between the two slots before it). A higher value means that the rank changes of '
           r'the method follow real changes.')
    L = head(cap, 'tab:si_rsi_truth', 'l' + 'r' * 4)
    L.append('Method & ' + ' & '.join(SCEN_SHORT[s] for s in SCEN) + r'\\')
    L.append(r'\midrule')
    for cfg, lab in (('default', '(a) Default configuration'), ('equal64', '(b) Equal window of 64 slots')):
        L.append(group(lab, 5))
        g = f[f.config == cfg]
        for m in ORDER:
            cells = []
            for sc in SCEN:
                if m == 'WSPI':
                    q = g[(g.scenario == sc)].spearman_ref_rsi_vs_truth_seen
                    assert q.nunique() == 1
                    cells.append(num(q.iloc[0], 3))
                else:
                    q = g[(g.scenario == sc) & (g.method == m)]
                    assert len(q) == 1
                    cells.append(num(q.spearman_rsi_vs_truth_seen.iloc[0], 3))
            L.append(m + ' & ' + ' & '.join(cells) + r'\\')
        L.append(r'\midrule')
    L.append(group('(c) WSPI against AF, default configuration', 5))
    rows = [('Share of windows with lower RSI@10 than AF', 'share_ref_rsi_lower'),
            ('Visible truth RSI@10 in these windows', 'truth_seen_when_ref_lower'),
            ('Visible truth RSI@10 in the other windows', 'truth_seen_other_windows')]
    for lab, col in rows:
        L.append(lab + ' & ' + ' & '.join(num(af.loc[sc, col], 3) for sc in SCEN) + r'\\')
    return '\n'.join(L + foot()) + '\n'


# =================================================================== main
SI_TABLES = [('si_padding.tex', t_padding), ('si_features.tex', t_features), ('si_levels.tex', t_levels),
             ('si_yt_gap.tex', t_yt_gap), ('si_decomposition.tex', t_decomp),
             ('si_robust_default.tex', lambda c: t_robust(c, 'default')),
             ('si_robust_equal64.tex', lambda c: t_robust(c, 'eq64')),
             ('si_hardware.tex', t_hardware), ('si_runtime_grid.tex', t_runtime_grid),
             ('si_memory.tex', t_memory), ('si_runtime_real.tex', t_runtime_real),
             ('si_ablation.tex', t_ablation), ('si_fusion.tex', t_fusion),
             ('si_selection.tex', t_selection), ('si_ml_profile.tex', t_ml_profile),
             ('si_resp_strict.tex', t_resp_strict), ('si_rsi_truth.tex', t_rsi_truth)]


def result_tables(res):
    """Values and paired tests of the default and equal-window runs, and the main
    and SI result tables built from them."""
    main9 = gf.V4_ORDER
    all12 = gf.V4_BASELINES + EXTRA + gf.V4_WAVELET
    V = {('default', 'main'): gf.v4_values(res, 'default', SCEN),
         ('equal64', 'main'): gf.v4_values(res, 'equal64', SCEN),
         ('default', 'ml'): gf.v4_values(res, 'default', ML),
         ('equal64', 'ml'): gf.v4_values(res, 'equal64', ML)}
    T = {('default', 'main'): gf.v4_tests(res, 'default', SCEN),
         ('equal64', 'main'): gf.v4_tests(res, 'equal64', SCEN),
         ('default', 'ml'): gf.v4_tests(res, 'default', ML),
         ('equal64', 'ml'): gf.v4_tests(res, 'equal64', ML)}
    V6 = {c: gf.v4_values(res, c, SCEN6) for c in ('default', 'equal64')}
    T6 = {c: gf.v4_tests(res, c, SCEN6) for c in ('default', 'equal64')}
    main = {
        'tab_main_default.tex': main_table6(V6['default'], T6['default'], 'default', 'tab:main_default'),
        'tab_main_equal64.tex': main_table6(V6['equal64'], T6['equal64'], 'equal64', 'tab:main_equal64'),
        'tab_main4_default.tex': main_table(V['default', 'main'], T['default', 'main'], 'default',
                                            'tab:main_default', SCEN, SCEN_LABEL),
        'tab_main4_equal64.tex': main_table(V['equal64', 'main'], T['equal64', 'main'], 'equal64',
                                            'tab:main_equal64', SCEN, SCEN_LABEL),
    }
    si = {
        'si_ci_default.tex': split6(
            lambda sc, lb, ti: ci_table(V6['default'], 'default', lb, sc, LABEL6, gf.V4_ORDER, ti),
            'tab:si_ci_default', 'Default configuration.'),
        'si_tests_default.tex': split6(
            lambda sc, lb, ti: tests_table(T6['default'], lb, sc, LABEL6, gf.V4_ORDER, ti),
            'tab:si_tests_default', 'Default configuration.'),
        'si_ci_equal64.tex': split6(
            lambda sc, lb, ti: ci_table(V6['equal64'], 'equal64', lb, sc, LABEL6, all12, ti),
            'tab:si_ci_equal64', 'Equal window of 64 slots, with the simple smoothers SMA, EWMA-eq and Holt.'),
        'si_tests_equal64.tex': split6(
            lambda sc, lb, ti: tests_table(T6['equal64'], lb, sc, LABEL6, all12, ti),
            'tab:si_tests_equal64', 'Equal window of 64 slots.'),
        'si_ci4_default.tex': ci_table(V['default', 'main'], 'default', 'tab:si_ci_default', SCEN,
                                       SCEN_LABEL, main9, 'Default configuration.'),
        'si_ci4_equal64.tex': ci_table(V['equal64', 'main'], 'equal64', 'tab:si_ci_equal64', SCEN,
                                       SCEN_LABEL, all12, 'Equal window of 64 slots, with the '
                                       'simple smoothers SMA, EWMA-eq and Holt.'),
        'si_tests4_default.tex': tests_table(T['default', 'main'], 'tab:si_tests_default', SCEN,
                                             SCEN_LABEL, main9, 'Default configuration.'),
        'si_tests4_equal64.tex': tests_table(T['equal64', 'main'], 'tab:si_tests_equal64', SCEN,
                                             SCEN_LABEL, all12, 'Equal window of 64 slots.'),
        'si_movielens.tex': ci_table(V['default', 'ml'], 'default', 'tab:movielens', ML, ML_LABEL,
                                     main9, 'MovieLens, default configuration.'),
        'si_movielens_equal64.tex': ci_table(V['equal64', 'ml'], 'equal64', 'tab:movielens_equal64',
                                             ML, ML_LABEL, all12, 'MovieLens, equal window of 64 slots.'),
        'si_movielens_tests.tex': tests_table(pd.concat([T['default', 'ml'], T['equal64', 'ml']]),
                                              'tab:movielens_tests', ML, ML_LABEL, all12,
                                              'MovieLens, both configurations.',
                                              configs=['default', 'equal64']),
    }
    return V, T, V6, main, si


def input_files(res):
    inputs = sorted({gf.v4_summary_path(res, c, s) for c in ('default', 'equal64') for s in SCEN + ML})
    inputs += [gf.rp(res, k, f) for k in ('stats_default', 'stats_window_sweep', 'stats_movielens')
               for f in ('all_method_summary.csv', 'all_paired_tests.csv')]
    return {str(p.relative_to(res)).replace('\\', '/'): md5(p) for p in inputs}


def write(out, files):
    for name, text in files.items():
        (out / name).write_text(text, encoding='utf-8', newline='\n')
        print(f'[ok] {name}')


def run_main(res, out, dout, V, T, V6, files):
    write(out, files)
    vals = pd.concat(V.values(), ignore_index=True)
    tests = pd.concat(T.values(), ignore_index=True)
    (dout / 'metadata').mkdir(parents=True, exist_ok=True)
    vals.drop(columns=['summary_mean', 'rel_diff']).to_csv(dout / 'values_all.csv', index=False)
    keep = ['config', 'run_group', 'scenario', 'metric', 'reference', 'method', 'n_windows',
            'mean_reference', 'mean_method', 'diff_ref_minus_method', 'block', 'n_blocks',
            'p_block', 'p_block_holm', 'cliffs_delta_favours_ref', 'rb_block_favours_ref', 'verdict']
    tests[[c for c in keep if c in tests.columns]].to_csv(dout / 'tests_all.csv', index=False)
    ctrl = vals[['config', 'scenario', 'method', 'metric', 'mean', 'summary_mean', 'rel_diff']].copy()
    ctrl['pass'] = ctrl['rel_diff'] <= 1e-9
    ctrl.to_csv(dout / 'control.csv', index=False)
    c6 = pd.concat(V6.values(), ignore_index=True)
    inputs = input_files(res)
    common = dict(created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                  platform=platform.platform())
    meta = dict(program='scripts/generate_tables.py --part main', tie_footnote_threshold=TIE_NOTE,
                significance='block Wilcoxon after Holm (scenario x metric), alpha 0.05 (verdict column)',
                control=dict(rows=int(len(ctrl)), max_rel_diff=float(ctrl['rel_diff'].max()),
                             all_pass=bool(ctrl['pass'].all())),
                inputs=inputs, tables=sorted(files), **common)
    (dout / 'metadata' / 'generate_tables_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    meta6 = dict(program='scripts/generate_tables.py --part main',
                 control=dict(rows=int(len(c6)), max_rel_diff=float(c6['rel_diff'].max())),
                 inputs=inputs, tables={n: hashlib.md5(t.encode('utf-8')).hexdigest() for n, t in files.items()},
                 **common)
    (out / 'tables_main_run.json').write_text(json.dumps(meta6, indent=2), encoding='utf-8')
    print(f'control: {len(ctrl)} rows, max relative difference {ctrl["rel_diff"].max():.2e}, '
          f'all pass = {bool(ctrl["pass"].all())}')
    print(f'control (six scenarios): {len(c6)} rows, max relative difference {c6["rel_diff"].max():.2e}')
    print(f'{len(files)} main tables written to {out}; source CSVs in {dout}')


def run_si(res, out, files):
    write(out, files)
    c = Ctx(res)
    written = {n: hashlib.md5(t.encode('utf-8')).hexdigest() for n, t in files.items()}
    for name, func in SI_TABLES:
        tex = bold_wavelet_names(func(c))
        p = out / name
        p.write_text(tex, encoding='utf-8', newline='\n')
        written[name] = hashlib.md5(p.read_bytes()).hexdigest()
        print(f'[ok] {name}')
    pd.DataFrame(c.sources).drop_duplicates().to_csv(out / 'si_tables_sources.csv', index=False)
    pd.DataFrame(c.control).to_csv(out / 'si_tables_control.csv', index=False)
    meta = dict(program='scripts/generate_tables.py --part si',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                platform=platform.platform(), results=str(res), tables=written,
                controls=len(c.control), controls_passed=int(sum(x['passed'] for x in c.control)))
    (out / 'tables_si_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'{len(written)} SI tables written; {meta["controls_passed"]} of {meta["controls"]} controls passed')


def main():
    ap = argparse.ArgumentParser(description='LaTeX tables of the paper and the SI')
    ap.add_argument('--part', required=True, choices=['main', 'si', 'all'])
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', required=True, help='folder for the .tex files')
    ap.add_argument('--data-out', default=None,
                    help="folder for the source CSVs (default: 'figure_data' of result_paths.json)")
    a = ap.parse_args()
    res = Path(a.results) if Path(a.results).is_absolute() else ROOT / a.results
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    if out.resolve().is_relative_to(res.resolve()):
        raise SystemExit('--out must not be inside --results')
    dout = Path(a.data_out) if a.data_out else gf.rp(res, 'figure_data')
    dout = dout if dout.is_absolute() else ROOT / dout
    out.mkdir(parents=True, exist_ok=True)
    V, T, V6, main_files, si_files = result_tables(res)
    if a.part in ('main', 'all'):
        run_main(res, out, dout, V, T, V6, main_files)
    if a.part in ('si', 'all'):
        run_si(res, out, si_files)


if __name__ == '__main__':
    main()
