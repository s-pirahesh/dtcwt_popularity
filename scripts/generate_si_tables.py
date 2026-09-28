r"""
Generate the new LaTeX tables of the Supplementary Information of paper V5 (task T4.10)
======================================================================================
Seventeen SI tables are built here directly from the result CSVs under
``results/revision_v5``.  No number is typed by hand and no number is computed
beyond selection, rounding and the few derived quantities named below (a range,
a ratio, a weighted share); each of these is written to the control CSV.
The seven SI tables made in task T3.10 (Tables S5-S8, S19, S21, S22) are not
built here; they are used unchanged from Response/Tables.

Added for paper V5 (chat 28, 28 Sep 2026; plan: Reports/R27_T4.10a_SI_Plan.md,
sections 3-5).  No existing file of the project is changed or imported.

Tables (file name -> SI number, label)
--------------------------------------
  S01_padding.tex          S1   tab:padding            padding and boundary extension (T3.4)
  S02_features.tex         S2   tab:si_features        relation of R and W_E (T3.7)
  S03_levels.tex           S3   tab:si_levels          decomposition level J (T3.1)
  S04_yt_gap.tex           S4   tab:yt_gap             YouTube collection gap (T3.11)
  S09_decomposition.tex    S9   tab:si_decomp          change of the WSPI terms under perturbation (T3.5)
  S10_robust_default.tex   S10  tab:si_robust_default  wider perturbations, default configuration (T3.5)
  S11_robust_equal64.tex   S11  tab:si_robust_equal64  wider perturbations, equal window of 64 (T3.5)
  S12_hardware.tex         S12  tab:si_hardware        hardware and software of the cost study (T3.8)
  S13_runtime_grid.tex     S13  tab:si_runtime_grid    scoring time per item (T3.8)
  S14_memory.tex           S14  tab:si_memory          memory (T3.8)
  S15_runtime_real.tex     S15  tab:si_runtime_real    run time on the real data (T3.8)
  S16_ablation.tex         S16  tab:si_ablation        eight ablation variants, four scenarios (T3.3)
  S17_fusion.tex           S17  tab:si_fusion          four fusion functions (T3.3)
  S18_selection.tex        S18  tab:si_selection       30/70 selection of alpha, beta and J (T3.2)
  S20_ml_profile.tex       S20  tab:si_ml_profile      MovieLens data profile (T3.9)
  S23_resp_strict.tex      S23  tab:si_resp_strict     responsiveness, strict entry rule (T2.4)
  S24_rsi_truth.tex        S24  tab:si_rsi_truth       RSI@10 and the visible change of the truth (T2.4)

Markers
-------
  \blacktriangle / \triangledown after a value: significantly better / worse than
  the reference of the table (WSPI; symmetric extension in S1; J = 3 in S3),
  block Wilcoxon test with Holm correction (verdict column of the T1.6 statistics).
  \dagger in S16 and S17 (as in Table 12 of the paper): NOT significantly
  different from the full index / the exponential form.

Other output (same folder, CSV and JSON only)
---------------------------------------------
  si_tables_sources.csv   table, label, source file, md5 of the source
  si_tables_control.csv   every cross-check and derived value, with pass flag
  si_tables_run.json      date, versions, md5 of every .tex written

Usage (Windows, from the project root)
--------------------------------------
  set PYTHONDONTWRITEBYTECODE=1
  python scripts\generate_si_tables.py --results results\revision_v5 --out "...\Response\Tables\SI"
Nothing is written to results.
"""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCEN = ['youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min']
SCEN_LABEL = {'youtube_hourly': 'YouTube Hourly', 'taxi_hourly': 'NYC Yellow Taxi Hourly',
              'taxi_30min': 'NYC Yellow Taxi 30m', 'taxi_5min': 'NYC Yellow Taxi 5m'}
SCEN_SHORT = {'youtube_hourly': 'YouTube', 'taxi_hourly': 'Taxi 1h',
              'taxi_30min': 'Taxi 30m', 'taxi_5min': 'Taxi 5m'}
ORDER = ['AF', 'CompoundPop', 'EWMA', 'PFRF', 'RRD', 'VSE', 'DWT+AF', 'DTCWT+AF', 'WSPI']
WAVELET = ['DWT+AF', 'DTCWT+AF', 'WSPI']
MET4 = ['ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion']
MET3 = ['ndcg@10', 'rsi@10', 'robustness_distortion']
MET_TEX = {'ndcg@10': 'NDCG@10', 'spearman_rho': r'$\rho$', 'rsi@10': 'RSI@10',
           'robustness_distortion': r'$\Delta$Rank'}
ARROW = {'ndcg@10': r'$\uparrow$', 'spearman_rho': r'$\uparrow$', 'rsi@10': r'$\uparrow$',
         'robustness_distortion': r'$\downarrow$'}
DEC = {'ndcg@10': 4, 'spearman_rho': 4, 'rsi@10': 4, 'robustness_distortion': 2}
UP = r'\textsuperscript{$\blacktriangle$}'
DOWN = r'\textsuperscript{$\triangledown$}'
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


# ---------------------------------------------------------------- S1 padding
def t_padding(c):
    c.table = 'S1'
    p = c.read('T3.4_padding/padding_summary.csv')
    t = c.read('T1.6_stats/T3.4_padding/all_paired_tests.csv')
    share = c.read('T3.4_padding/padded_share.csv')
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


# ---------------------------------------------------------------- S2 features
def t_features(c):
    c.table = 'S2'
    f = c.read('T3.7_feature_relation/feature_relation_summary.csv')
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


# ---------------------------------------------------------------- S3 levels
def t_levels(c):
    c.table = 'S3'
    s = c.read('T1.6_stats/T3.1_level_sweep/all_method_summary.csv')
    t = c.read('T1.6_stats/T3.1_level_sweep/all_paired_tests.csv')
    ls = c.read('T3.1_level_sweep/level_sweep_summary.csv')
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


# ---------------------------------------------------------------- S4 YouTube gap
def t_yt_gap(c):
    c.table = 'S4'
    g = c.read('T3.11_youtube_provenance/gap_sensitivity.csv')
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
    c.table = 'S9'
    r = c.read('T3.5_robustness/robustness_summary.csv')
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
    c.table = 'S10' if config == 'default' else 'S11'
    r = c.read('T3.5_robustness/robustness_summary.csv')
    t = c.read('T1.6_stats/T3.5_robustness/all_paired_tests.csv')
    conds = [('size10', r'$10\times$ last'), ('pos_middle', 'Middle'), ('pos_first', 'First'),
             ('pos_random', 'Random'), ('dur3', '3 slots'), ('dur6', '6 slots'),
             ('noise_poisson', 'Poisson'), ('noise_gauss_snr10', r'Gauss.\ 10\,dB'),
             ('noise_gauss_snr0', r'Gauss.\ 0\,dB')]
    rr = r[r.config == config]
    spread = []
    if config == 'default':
        for sc, v in zip(SCEN, [41.88, 10.24, 8.72, 4.84]):
            q = rr[(rr.scenario == sc) & (rr.method == 'WSPI') & (rr.condition == 'pos_random')].dr_mean.iloc[0]
            c.check(f'{sc}: WSPI random-slot Delta Rank (text of Subsection 4.11)', round(q, 2), v, tol=1e-9)
            top = rr[(rr.scenario == sc) & (rr.condition == 'pos_random')].sort_values('dr_mean').method.iloc[-1]
            c.check(f'{sc}: largest random-slot Delta Rank is WSPI', top, ok=top == 'WSPI')
    else:
        for sc in SCEN:
            for key, _ in conds:
                q = rr[(rr.scenario == sc) & (rr.condition == key)].set_index('method').dr_mean
                c.check(f'{sc} {key}: RRD below WSPI (text of Subsection 4.3)', q['RRD'] - q['WSPI'],
                        ok=q['RRD'] < q['WSPI'])
    cfg = ('default configuration (baselines 7 slots, wavelet-based methods 64 slots)'
           if config == 'default' else 'equal window of 64 slots for all methods')
    cap = (r'Wider perturbations, ' + cfg + r': $\Delta$Rank (lower is better), mean over the common '
           r'windows and five seeds. Every method sees the same 50 low-activity items and the same '
           r'corruption. Columns: a $10\times$ spike in the last slot (reference cell), in the middle, '
           r'first or a random slot of the 64-slot window, lasting 3 or 6 slots, and continuous Poisson '
           r'or Gaussian noise (SNR 10 and 0\,dB). ' + UP + ' / ' + DOWN +
           r': significantly better / worse than WSPI (block Wilcoxon test, Holm correction, $\alpha=0.05$). '
           r'Spike sizes are shown in Supplementary Fig.~S3.')
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
                 'test of Table~8 does, except that RRD and VSE, whose values are close, swap places in '
                 'three of the four scenarios; WSPI is second after PFRF in both tests and all four scenarios.')
    return '\n'.join(L + foot(note)) + '\n'


# ---------------------------------------------------------------- cost (T3.8)
def t_hardware(c):
    c.table = 'S12'
    b = c.read('T3.8_runtime/bench/metadata/bench_run.json')
    m = c.read('T3.8_runtime/memory/metadata/memory_run.json')
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
            ('PyWavelets / dtcwt', f'{v["pywt"]} / {v["dtcwt"]}'),
            ('Synthetic data', b['data'].replace('lognormal(1, 1)', r'log-normal ($\mu=1$, $\sigma=1$)')),
            ('Repeats', f'{b["repeats"]} (median and interquartile range reported)'),
            ('Batch size for large catalogues', f'{b["chunk"]:,} items'),
            ('Seed', str(b['seed']))]
    cap = 'Hardware and software of the cost measurements (Section 4.7 and Supplementary Tables S13 to S15).'
    L = head(cap, 'tab:si_hardware', 'll', size=r'\small')
    L.append(r'Item & Value\\')
    L.append(r'\midrule')
    for k, val in rows:
        L.append(f'{k} & {val}' + r'\\')
    return '\n'.join(L + foot()) + '\n'


def t_runtime_grid(c):
    c.table = 'S13'
    g = c.read('T3.8_runtime/bench/runtime_grid.csv')
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
    c.table = 'S14'
    tm = c.read('T3.8_runtime/memory/tracemalloc_grid.csv')
    rs = c.read('T3.8_runtime/memory/rss_grid.csv')
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
    c.table = 'S15'
    r = c.read('T3.8_runtime/real_summary_all.csv')
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
    s = c.read('T1.6_stats/T3.3_ablation/all_method_summary.csv')
    t = c.read('T1.6_stats/T3.3_ablation/all_paired_tests.csv')
    a = c.read('T3.3_ablation/ablation_summary.csv')
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
    c.table = 'S16'
    cap = (r'Ablation of the WSPI components in all four scenarios ($N=64$, $J=3$, $\alpha=\beta=1$); mean '
           r'[95\% block-bootstrap confidence interval] over the common windows. DWT: db4 with symmetric '
           r'extension. ' + DAG + r' Not significantly different from the full index '
           r'(block Wilcoxon test, Holm correction).')
    return _ablation(c, 'ablation', ABL, 'tab:si_ablation', cap)


def t_fusion(c):
    c.table = 'S17'
    cap = (r'Fusion function: four ways to combine the same features with unit weights ($N=64$, $J=3$); mean '
           r'[95\% block-bootstrap confidence interval] over the common windows. The product form ranks items '
           r'like the geometric mean. ' + DAG + r' Not significantly different from the exponential form '
           r'(block Wilcoxon test, Holm correction).')
    return _ablation(c, 'fusion', FUS, 'tab:si_fusion', cap)


# ---------------------------------------------------------------- S18 selection
def t_selection(c):
    c.table = 'S18'
    s = c.read('T3.2_param_grid/selection/selection.csv')
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


# ---------------------------------------------------------------- S20 MovieLens profile
def t_ml_profile(c):
    c.table = 'S20'
    g = c.read('T3.9_movielens/data_prep/granularity_profile.csv').set_index('file')
    y = c.read('T3.9_movielens/data_prep/year_profile.csv')
    j = c.read('T3.9_movielens/data_prep/prep_summary.json')
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
    c.table = 'S23'
    s = c.read('T2.4_responsiveness/responsiveness_summary.csv')
    main = s[(s.variant == 'main') & (s.config == 'default') & (s.method == 'WSPI')].set_index('scenario')
    c.check('main variant YouTube WSPI miss rate (Table 14)', round(main.loc['youtube_hourly'].miss_rate, 3), 0.312)
    st = s[s.variant == 'strict']
    cap = (r'Responsiveness with a stricter entry rule: the item was outside the ground-truth Top-20 (instead of '
           r'the Top-10) in the six slots before the entry. Miss rate / median delay in slots, as in Table~14; '
           r'-- : median undefined (more than half of the entries missed). The window length of each method is '
           r'given in parentheses.')
    L = head(cap, 'tab:si_resp_strict', 'lr' + 'c' * len(RESP_COLS), sep='3.5pt')
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
    c.table = 'S24'
    f = c.read('T2.4_responsiveness/all_rsi_failures.csv')
    d = f[f.config == 'default']
    af = d[d.method == 'AF'].set_index('scenario')
    c.check('AF rho min (text of Subsection 4.11)', round(af.spearman_rsi_vs_truth_seen.min(), 2), 0.24)
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


# ---------------------------------------------------------------- main
TABLES = [('S01_padding.tex', t_padding), ('S02_features.tex', t_features), ('S03_levels.tex', t_levels),
          ('S04_yt_gap.tex', t_yt_gap), ('S09_decomposition.tex', t_decomp),
          ('S10_robust_default.tex', lambda c: t_robust(c, 'default')),
          ('S11_robust_equal64.tex', lambda c: t_robust(c, 'eq64')),
          ('S12_hardware.tex', t_hardware), ('S13_runtime_grid.tex', t_runtime_grid),
          ('S14_memory.tex', t_memory), ('S15_runtime_real.tex', t_runtime_real),
          ('S16_ablation.tex', t_ablation), ('S17_fusion.tex', t_fusion),
          ('S18_selection.tex', t_selection), ('S20_ml_profile.tex', t_ml_profile),
          ('S23_resp_strict.tex', t_resp_strict), ('S24_rsi_truth.tex', t_rsi_truth)]


def main():
    ap = argparse.ArgumentParser(description='SI tables of paper V5 (T4.10)')
    ap.add_argument('--results', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    if a.out.resolve().is_relative_to(a.results.resolve()):
        raise SystemExit('--out must not be inside --results')
    a.out.mkdir(parents=True, exist_ok=True)
    c = Ctx(a.results)
    written = {}
    for name, func in TABLES:
        tex = func(c)
        p = a.out / name
        p.write_text(tex, encoding='utf-8', newline='\n')
        written[name] = hashlib.md5(p.read_bytes()).hexdigest()
        print(f'[ok] {name}')
    pd.DataFrame(c.sources).drop_duplicates().to_csv(a.out / 'si_tables_sources.csv', index=False)
    pd.DataFrame(c.control).to_csv(a.out / 'si_tables_control.csv', index=False)
    meta = dict(program='scripts/generate_si_tables.py', task='T4.10',
                created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                platform=platform.platform(), results=str(a.results), tables=written,
                controls=len(c.control), controls_passed=int(sum(x['passed'] for x in c.control)))
    (a.out / 'si_tables_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'{len(written)} tables written; {meta["controls_passed"]} of {meta["controls"]} controls passed')


if __name__ == '__main__':
    main()
