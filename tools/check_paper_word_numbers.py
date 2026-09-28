#!/usr/bin/env python3
"""Check the numbers written in words in the V5 paper and its SI against the result files (task T4.11b).

check_paper_numbers.py (T4.11) checks every number written with digits. This program checks the rest:
every number word (one ... twenty, half, twice, double) and every quantifier (all, every, each, both,
none, never, always, most, whole, single) from \\begin{document} to the reference list of both files.
Each such word gets one category:

  claim       a statement about results ("in all six scenarios", "three of the four", "never falls").
              The statement is recomputed from the result CSVs (values, block-test verdicts, per-window
              series) and the count or condition is compared (status ok or mismatch).
  count       a count of the study design ("six baselines", "four scenarios", "five seeds"). It is
              compared with the configuration or metadata file that records it (status count-ok).
  structure   a count of items listed in the text itself ("three questions ... (a) (b) (c)", "four
              stages", "Table 12 shows six variants"); the items are counted in the tex (structure-ok).
  design      a fixed setting or a definition that no result file records; the reason is given.
  language    ordinary wording, not a quantity ("each item", "at most" before a checked number,
              "most recent"); the reason is given.

A few result statements next to these words carry no number word but make the same kind of claim
("the product form is more robust on the taxi data but less robust on YouTube"). They are checked as
extra claims (kind=claim).

Every word must be covered by exactly one rule. A rule is tied to a literal piece of one line (its
anchor). If the sentence changes, the anchor is not found and the program reports a RULE PROBLEM.

The program only reads. It writes one CSV report (--out), prints a summary, and returns 1 if any
claim or count differs, a word has no rule, or a rule does not match the text.

Example (Windows, from the project root, on a temporary built copy of V5/source):
  python tools\\check_paper_word_numbers.py --main "%TMPB%\\WSPI_ScientificReports.tex"
      --si "%TMPB%\\WSPI_SI.tex" --results results\\revision_v5
      --out "%RESP%\\Reports\\R30_T4.11b_word_check.csv"
"""
import sys
sys.dont_write_bytecode = True

import argparse, csv, json, re
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_paper_numbers as C  # noqa: E402  shared result loaders and file names

WORDS = ('one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|twenty'
         '|half|twice|both|neither|none|every|all|each|single|double|doubles|doubled|doubling|majority|most'
         '|never|always|whole')
WORD_RE = re.compile(r'(?<![A-Za-z\\-])(' + WORDS + r')(?![A-Za-z])', re.I)

M4, ML, M6 = C.M4, C.ML, C.M6
TAXI3 = M4[1:]
BASE6, BASE5, ALL9 = C.BASE6, C.BASE5, C.ALL9
WAV3 = ['DWT+AF', 'DTCWT+AF', 'WSPI']
LONG3 = ['RRD', 'VSE', 'CompoundPop']
NDCG, RHO, RSI, DR = C.NDCG, C.RHO, C.RSI, C.DR
SL = {'youtube_hourly': 'YouTube', 'taxi_hourly': 'taxi 1h', 'taxi_30min': 'taxi 30m', 'taxi_5min': 'taxi 5m',
      'movielens_daily': 'ML daily', 'movielens_weekly': 'ML weekly'}
VA, TA, CFG, AB, ABT, RB, RBT, RE_, RET, EX, SH, SY, RS, RG, BJ, YP = (
    C.VA, C.TA, C.CFG, C.AB, C.ABT, C.RB, C.RBT, C.RE_, C.RET, C.EX, C.SH, C.SY, C.RS, C.RG, C.BJ, C.YP)
PD = 'T3.4_padding/padding_summary.csv'
WS = 'T1.5_leakage_audit/causal/window_structure.csv'
PT = 'T1.5_leakage_audit/causal/perturbation_test.csv'
SHT = 'T1.6_stats/T3.6_shift_invariance/all_paired_tests.csv'
SYJ = 'T3.6_shift_invariance/synthetic/metadata/synthetic_run.json'
PROT = 'T1.5_causal_universe/T1.4_protocol_v5'


# ------------------------------------------------------------------ results of a check
class Res:
    def __init__(self, ok, found, source):
        self.ok, self.found, self.source = ok, found, source
        self.what = ''


def over(cases, want, files):
    """cases: (label, holds, detail). want: 'all', 'none' or the number of cases that must hold."""
    k = sum(1 for c in cases if c[1])
    n = len(cases)
    ok = (k == n) if want == 'all' else (k == 0) if want == 'none' else (k == want)
    bad = [f'{c[0]}: {c[2]}' for c in cases if not c[1]]
    found = f'{k} of {n}'
    if bad:
        found += ' | not: ' + '; '.join(bad[:8]) + (' ...' if len(bad) > 8 else '')
    return Res(ok, found, ', '.join(dict.fromkeys(files)))


def val(cfg, sc, m, met):
    return float(C.V(sc, m, met, cfg).value)


def best(cfg, sc, met, methods):
    v = {m: val(cfg, sc, m, met) for m in methods}
    b = (min if met == DR else max)(v, key=v.get)
    return b, v


def vd(cfg, sc, met, m):
    return C.verdict(cfg, sc, met, m)


def jload(rel):
    return json.loads((C.RESULTS / rel).read_text(encoding='utf-8'))


def cfg_df():
    d = C.load(CFG)
    return d[d.run_group == 'T1.4_protocol_v5']


def ml_meta(cfg, sc):
    rel = f'T3.9_movielens/{cfg}/{sc}' + ('/W064' if cfg == 'equal64' else '') + '/metadata/protocol_v5_run.json'
    return jload(rel), rel


def eq64_meta(sc):
    rel = f'T2.2_window_sweep/{sc}/W064/metadata/protocol_v5_run.json'
    return jload(rel), rel


CHECKS = {}
_DONE = {}


def check(name, what):
    def deco(fn):
        CHECKS[name] = (fn, what)
        return fn
    return deco


def run_check(name):
    if name not in _DONE:
        fn, what = CHECKS[name]
        try:
            r = fn()
        except Exception as e:  # a missing row or file is an error, never a silent pass
            r = Res(None, f'{type(e).__name__}: {e}', '')
        r.what = what
        _DONE[name] = r
    return _DONE[name]


# ------------------------------------------------------------------ design counts (count)
@check('n_base', 'methods with the 7-slot default window (config table of the paper runs, each scenario)')
def _():
    d = cfg_df()
    ns = {SL[s]: int((g.window_slots == 7).sum()) for s, g in d.groupby('scenario')}
    return Res(set(ns.values()) == {6}, str(ns), CFG)


@check('n_wav', 'methods with the 64-slot default window = the wavelet-based methods (config table)')
def _():
    d = cfg_df()
    ns = {SL[s]: sorted(g.method[g.window_slots == 64]) for s, g in d.groupby('scenario')}
    ok = all(v == sorted(WAV3) for v in ns.values())
    return Res(ok, f"3 in each scenario: {ns['YouTube']}", CFG)


@check('n_methods', 'distinct methods in the config table of the paper runs')
def _():
    d = cfg_df()
    return Res(d.method.nunique() == 9 and all(g.method.nunique() == 9 for _, g in d.groupby('scenario')),
               f'{d.method.nunique()} methods', CFG)


@check('n_competitors', 'methods other than WSPI in the config table')
def _():
    d = cfg_df()
    return Res(d.method.nunique() - 1 == 8, f'{d.method.nunique() - 1}', CFG)


@check('n_scen4', 'scenarios of the paper runs (YouTube and the three taxi granularities)')
def _():
    s = sorted(cfg_df().scenario.unique())
    return Res(len(s) == 4, f'{len(s)}: {s}', CFG)


@check('n_scen6', 'scenarios of the default configuration in the main result table')
def _():
    d = C.load(VA)
    s = sorted(d[d.config == 'default'].scenario.unique())
    return Res(len(s) == 6, f'{len(s)}', VA)


@check('n_taxi', 'taxi granularities in the config table')
def _():
    s = sorted(x for x in cfg_df().scenario.unique() if x.startswith('taxi'))
    return Res(len(s) == 3, f'{len(s)}: {s}', CFG)


@check('n_ml', 'MovieLens granularities in the main result table')
def _():
    s = sorted(x for x in C.load(VA).scenario.unique() if x.startswith('movielens'))
    return Res(len(s) == 2, f'{len(s)}: {s}', VA)


@check('n_datasets', 'datasets (scenario prefixes) in the main result table')
def _():
    s = sorted({x.split('_')[0] for x in C.load(VA).scenario.unique()})
    return Res(len(s) == 3, f'{len(s)}: {s}', VA)


@check('n_configs', 'configurations in the main result table')
def _():
    s = sorted(C.load(VA).config.unique())
    return Res(len(s) == 2, f'{len(s)}: {s}', VA)


@check('holm8', 'paired tests per scenario and metric in the default configuration (Holm family)')
def _():
    d = C.load(TA)
    n = d[d.config == 'default'].groupby(['scenario', 'metric']).size()
    return Res(set(n) == {8}, f'sizes {sorted(set(n))}', TA)


@check('n_pad5', 'explicit padding modes in the padding study (without "none")')
def _():
    d = C.load(PD)
    m = sorted(set(d[d.layer == 'pad']['mode']) - {'none'})
    return Res(len(m) == 5, f'{len(m)}: {m}', PD)


@check('n_ext5', 'boundary extension modes in the padding study')
def _():
    d = C.load(PD)
    m = sorted(set(d[d.layer == 'ext']['mode']))
    return Res(len(m) == 5, f'{len(m)}: {m}', PD)


@check('seeds5', 'seeds of the extended robustness study (metadata of each scenario)')
def _():
    s = {SL[sc]: len(jload(f'T3.5_robustness/{sc}/metadata/robustness_run.json')['seeds']) for sc in M4}
    return Res(set(s.values()) == {5}, str(s), 'T3.5_robustness/<scenario>/metadata/robustness_run.json')


@check('dur6', 'duration of the longest spike of the extended robustness study')
def _():
    sp = {x['name']: x for x in jload('T3.5_robustness/youtube_hourly/metadata/robustness_run.json')['spikes']}
    return Res(sp['dur6']['duration'] == 6, f"dur6 duration {sp['dur6']['duration']}",
               'T3.5_robustness/youtube_hourly/metadata/robustness_run.json')


def _event_rule(key, want):
    s = {SL[sc]: jload(f'T2.4_responsiveness/{sc}/metadata/responsiveness_run.json')['event_rule'][key] for sc in M4}
    return Res(set(s.values()) == {want}, f'{key} {s}', 'T2.4_responsiveness/<scenario>/metadata/responsiveness_run.json')


@check('L6', 'entry rule: slots the item stays in the true Top-10 (L)')
def _():
    return _event_rule('L', 6)


@check('P6', 'entry rule: slots outside the true Top-10 before the entry (P)')
def _():
    return _event_rule('P', 6)


@check('ablation8', 'variants of the ablation family (four with DTCWT, four with DWT)')
def _():
    d = C.load(AB)
    v = list(dict.fromkeys(d[d.family == 'ablation'].variant))
    return Res(len(v) == 8, f'{len(v)}: {v}', AB)


@check('fusion_alt3', 'fusion forms other than the exponential one')
def _():
    d = C.load(AB)
    v = [x for x in dict.fromkeys(d[d.family == 'fusion'].variant) if x != 'WSPI']
    return Res(len(v) == 3, f'{len(v)}: {v}', AB)


@check('gap10', 'hours of the YouTube collection gap on 19 May 2018')
def _():
    g = C.Q(YP, 'value', key='gap_hours')
    return Res(int(float(g.value)) == 10, f'{int(float(g.value))} hours', YP)


@check('eight_slots', 'slots per approximation coefficient = 2^J with the J of WSPI')
def _():
    j = int(C.cfgv('WSPI', 'level_J').value)
    return Res(2 ** j == 8, f'J={j}, 2^J={2 ** j}', CFG)


@check('level3', 'detail bands of WSPI = J')
def _():
    j = int(C.cfgv('WSPI', 'level_J').value)
    return Res(j == 3, f'J={j}', CFG)


@check('thread1', 'threads of the cost measurement')
def _():
    t = jload(BJ)['threads']
    return Res(str(t) == '1', f'threads={t}', BJ)


@check('million', 'largest catalogue of the cost benchmark')
def _():
    m = max(jload(BJ)['items'])
    return Res(m == 1_000_000, f'{m:,} items', BJ)


@check('five_hours', '64 slots of the 5-minute taxi data, in hours')
def _():
    sm = float(C.cfgv('WSPI', 'slot_minutes', 'taxi_5min').value)
    h = 64 * sm / 60
    return Res(round(h) == 5, f'64 x {sm:g} min = {h:.2f} h', CFG)


@check('steps8', 'steps of the synthetic moving event (shifts 0..8)')
def _():
    s = jload(SYJ)['shifts']
    return Res(len(s) - 1 == 8, f'{len(s) - 1} steps', SYJ)


@check('burst3', 'length of the synthetic burst')
def _():
    b = jload(SYJ)['shapes']['burst3']
    return Res(len(b) == 3, f'{len(b)} slots', SYJ)


@check('block_day_week', 'bootstrap block of YouTube (24 hourly slots) and of the hourly taxi data (168), in days')
def _():
    d = C.load(TA)
    d = d[d.config == 'default']
    by = int(d[d.scenario == 'youtube_hourly'].block.iloc[0])
    bt = int(d[d.scenario == 'taxi_hourly'].block.iloc[0])
    s = float(C.cfgv('WSPI', 'slot_minutes').value)
    ok = by * s == 1440 and bt * s == 7 * 1440
    return Res(ok, f'YouTube {by} x {s:g} min = {by * s / 1440:g} day; taxi 1h {bt} slots = {bt * s / 1440:g} days',
               f'{TA}, {CFG}')


@check('slot_lengths', 'slot lengths of the six scenarios')
def _():
    d = cfg_df()
    sm = {SL[s]: float(g.slot_minutes.iloc[0]) for s, g in d.groupby('scenario')}
    for sc in ML:
        sm[SL[sc]] = float(ml_meta('default', sc)[0]['slot_minutes'])
    ok = set(sm.values()) == {60.0, 30.0, 5.0, 1440.0, 10080.0}
    return Res(ok, str({k: f'{v:g} min' for k, v in sm.items()}), f'{CFG}, T3.9_movielens/default/<s>/metadata')


@check('horizon_one', 'gap between the last input slot and the test slot, every audited run')
def _():
    d = C.load(WS)
    g = sorted(set(d.max_train_to_test_gap_slots))
    return Res(g == [1], f'{len(d)} runs, gaps {g}', WS)


@check('eq64_all', 'window of all nine methods in the equal-window runs (metadata)')
def _():
    cases = []
    for sc in M4 + ML:
        m = (eq64_meta(sc) if sc in M4 else ml_meta('equal64', sc))[0]['methods']
        w = {k: v['window_slots'] for k, v in m.items() if k in ALL9}
        cases.append((SL[sc], len(w) == 9 and set(w.values()) == {64}, str(sorted(set(w.values())))))
    return over(cases, 'all', ['T2.2_window_sweep/<s>/W064/metadata', 'T3.9_movielens/equal64/<s>/W064/metadata'])


# ------------------------------------------------------------------ result claims (claim)
@check('rsi_top_wav6', 'default config: the method with the highest RSI@10 of the nine is a wavelet-based one')
def _():
    cases = []
    for sc in M6:
        b, v = best('default', sc, RSI, ALL9)
        cases.append((SL[sc], b in WAV3, f'{b} {v[b]:.4f}'))
    return over(cases, 'all', [VA])


@check('rsi_top_counts', 'default config: how often WSPI and DTCWT+AF have the highest RSI@10 of the nine')
def _():
    tops = {SL[sc]: best('default', sc, RSI, ALL9)[0] for sc in M6}
    c = Counter(tops.values())
    return Res(c['WSPI'] == 3 and c['DTCWT+AF'] == 3, f"WSPI {c['WSPI']}, DTCWT+AF {c['DTCWT+AF']} ({tops})", VA)


def _dr_low(scen):
    cases = []
    for sc in scen:
        b, v = best('default', sc, DR, [m for m in ALL9 if m != 'PFRF'])
        cases.append((SL[sc], b == 'WSPI', f'{b} {v[b]:.2f}'))
    return over(cases, 'all', [VA])


@check('dr_low6', 'default config: WSPI has the lowest dRank of the eight methods other than PFRF')
def _():
    return _dr_low(M6)


@check('sym_ndcg3', 'extension modes, WSPI: symmetric gives the highest NDCG@10 (want 3 of 4)')
def _():
    d = C.load(PD)
    d = d[(d.layer == 'ext') & (d.method == 'WSPI') & (d.subset == 'all')]
    cases = []
    for sc in M4:
        g = d[d.scenario == sc].set_index('mode')['ndcg@10_mean']
        cases.append((SL[sc], g.idxmax() == 'symmetric', f'best {g.idxmax()} {g.max():.4f}'))
    return over(cases, 3, [PD])


@check('edge_dr_every', 'extension modes, WSPI: edge has a higher dRank than symmetric')
def _():
    d = C.load(PD)
    d = d[(d.layer == 'ext') & (d.method == 'WSPI') & (d.subset == 'all')].set_index(['scenario', 'mode'])['robustness_distortion_mean']
    cases = [(SL[sc], d[(sc, 'edge')] > d[(sc, 'symmetric')], f"{d[(sc, 'edge')]:.2f} vs {d[(sc, 'symmetric')]:.2f}") for sc in M4]
    return over(cases, 'all', [PD])


@check('wspi_min_mode', 'extension modes: WSPI has the lowest dRank of the three wavelet-based methods (each mode and scenario)')
def _():
    d = C.load(PD)
    d = d[(d.layer == 'ext') & (d.subset == 'all')]
    cases = []
    for (sc, mo), g in d.groupby(['scenario', 'mode']):
        s = g.set_index('method')['robustness_distortion_mean']
        cases.append((f'{SL[sc]} {mo}', s.idxmin() == 'WSPI' and len(s) == 3, f'lowest {s.idxmin()}'))
    return over(cases, 'all', [PD])


@check('gap_never_truth', 'hours of the YouTube gap used as a test slot')
def _():
    g = C.Q(YP, 'value', key='gap_hours_used_as_test_slot')
    return Res(int(float(g.value)) == 0, f'{int(float(g.value))}', YP)


@check('audit_every_run', 'the window-structure audit covers every run of the config table')
def _():
    a = set(map(tuple, C.load(WS)[['scenario', 'run_group', 'method']].values))
    b = set(map(tuple, C.load(CFG)[['scenario', 'run_group', 'method']].values))
    return Res(a == b, f'{len(a)} audited runs, {len(b)} runs in the config table', f'{WS}, {CFG}')


@check('audit_each_window', 'input of each window ends before its test slot (all audited runs)')
def _():
    d = C.load(WS)
    ok = bool(d.last_train_before_test.all() and d.train_end_equals_test_slot.all() and d.timestamps_match.all())
    return Res(ok, f'{int(d.last_train_before_test.sum())} of {len(d)} runs', WS)


@check('posctl_every', 'positive control (an evaluator that sees the test slot) detected in every tested window')
def _():
    d = C.load(PT)
    p = d[d.positive_control.astype(str) == 'True']
    cases = [(f'{SL[r.scenario]} {r.method}', r.n_windows_changed == r.n_windows_tested,
              f'{r.n_windows_changed}/{r.n_windows_tested}') for r in p.itertuples()]
    return over(cases, 'all', [PT])


@check('config_same', 'configuration of each method is the same on all datasets (window, entry rule, scorer, J, seed, tie rule)')
def _():
    d = cfg_df()
    cases = []
    for m in ALL9:
        g = d[d.method == m]
        rows = {SL[r.scenario]: (int(r.window_slots), int(r.min_obs), r.scorer) for r in g.itertuples()}
        for sc in ML:
            mm = ml_meta('default', sc)[0]['methods'][m]
            rows[SL[sc]] = (int(mm['window_slots']), int(mm['min_obs']), mm['scorer'])
        cases.append((m, len(set(rows.values())) == 1 and len(rows) == 6, str(sorted(set(rows.values())))))
    other = set(map(tuple, d[['level_J', 'seed', 'tie_rule', 'rsi_by_item', 'mode']].astype(str).values))
    for sc in ML:
        j = ml_meta('default', sc)[0]
        other.add((str(j['level']), str(j['seed']), j['tie_rule'], str(j['rsi_by_item']), j['mode']))
    cases.append(('J, seed, tie rule, RSI by item, mode', len(other) == 1, str(other)))
    return over(cases, 'all', [CFG, 'T3.9_movielens/default/<s>/metadata/protocol_v5_run.json'])


def _rsi_base(scen):
    cases = []
    for sc in scen:
        for b in BASE6:
            hi = val('default', sc, 'WSPI', RSI) > val('default', sc, b, RSI)
            v = vd('default', sc, RSI, b)
            want = 'n.s.' if (sc, b) == ('youtube_hourly', 'PFRF') else 'ref_better'
            cases.append((f'{SL[sc]} {b}', hi and v == want, f'higher={hi}, {v}'))
    return over(cases, 'all', [VA, TA])


@check('rsi_gt_base6', 'default, six scenarios: WSPI RSI@10 above each of the six baselines; significant except PFRF on YouTube (n.s.)')
def _():
    return _rsi_base(M6)


@check('rrd_rsi_eq6', 'equal window, six scenarios: RRD has a higher RSI@10 than WSPI')
def _():
    cases = [(SL[sc], val('equal64', sc, 'RRD', RSI) > val('equal64', sc, 'WSPI', RSI),
              f"{val('equal64', sc, 'RRD', RSI):.4f} vs {val('equal64', sc, 'WSPI', RSI):.4f}, {vd('equal64', sc, RSI, 'RRD')}")
             for sc in M6]
    return over(cases, 'all', [VA, TA])


@check('acc_gt_long_eq6', 'equal window, six scenarios: WSPI NDCG@10 and rho significantly above RRD, VSE and CompoundPop')
def _():
    cases = [(f'{SL[sc]} {met} {m}', vd('equal64', sc, met, m) == 'ref_better', vd('equal64', sc, met, m))
             for sc in M6 for met in (NDCG, RHO) for m in LONG3]
    return over(cases, 'all', [TA])


@check('not_most_stable', 'settings (configuration x scenario) in which WSPI does not have the highest RSI@10 of the nine (want at least one)')
def _():
    cases = []
    for cfg in ('default', 'equal64'):
        for sc in M6:
            b, v = best(cfg, sc, RSI, ALL9)
            cases.append((f'{cfg} {SL[sc]}', b != 'WSPI', f'highest {b}'))
    k = sum(c[1] for c in cases)
    return Res(k >= 1, f'WSPI not highest in {k} of {len(cases)} settings: ' +
               '; '.join(f'{c[0]}: {c[2]}' for c in cases if c[1]), VA)


@check('taxi_ndcg_all', 'taxi, default: every method except PFRF has a significantly higher NDCG@10 than WSPI; PFRF not')
def _():
    cases = []
    for sc in TAXI3:
        for m in ALL9:
            if m == 'WSPI':
                continue
            v = vd('default', sc, NDCG, m)
            cases.append((f'{SL[sc]} {m}', (v != 'ref_worse') if m == 'PFRF' else (v == 'ref_worse'), v))
    return over(cases, 'all', [TA])


@check('tm_rho_all', 'taxi and MovieLens, default: WSPI rho significantly above each of the six baselines')
def _():
    cases = [(f'{SL[sc]} {b}', vd('default', sc, RHO, b) == 'ref_better', vd('default', sc, RHO, b)) for sc in TAXI3 + ML for b in BASE6]
    return over(cases, 'all', [TA])


@check('most_stable_base6', 'default: highest RSI@10 of the five baselines other than PFRF = RRD (YouTube, taxi, ML daily), VSE (ML weekly)')
def _():
    cases = []
    for sc in M6:
        b, v = best('default', sc, RSI, BASE5)
        want = 'VSE' if sc == 'movielens_weekly' else 'RRD'
        cases.append((SL[sc], b == want, f'{b} {v[b]:.4f}'))
    return over(cases, 'all', [VA])


@check('tm_top2', 'taxi and MovieLens, default: the two highest RSI@10 of the nine are WSPI and DTCWT+AF')
def _():
    cases = []
    for sc in TAXI3 + ML:
        v = {m: val('default', sc, m, RSI) for m in ALL9}
        top = sorted(v, key=v.get, reverse=True)[:2]
        cases.append((SL[sc], set(top) == {'WSPI', 'DTCWT+AF'}, str(top)))
    return over(cases, 'all', [VA])


@check('dr_sig_all', 'default, six scenarios: WSPI dRank significantly lower than each method other than PFRF')
def _():
    cases = [(f'{SL[sc]} {m}', vd('default', sc, DR, m) == 'ref_better', vd('default', sc, DR, m))
             for sc in M6 for m in ALL9 if m not in ('WSPI', 'PFRF')]
    return over(cases, 'all', [TA])


@check('pfrf_lowest', 'default, six scenarios: PFRF has the lowest dRank of the nine')
def _():
    cases = []
    for sc in M6:
        b, v = best('default', sc, DR, ALL9)
        cases.append((SL[sc], b == 'PFRF', f'{b} {v[b]:.2f}'))
    return over(cases, 'all', [VA])


@check('pfrf_tied', 'default, six scenarios: share of windows in which PFRF has tied scores among the top 21 (nearly every window: at least 0.999 in each scenario)')
def _():
    cases = []
    for sc in M6:
        s = float(C.V(sc, 'PFRF', RSI, 'default', 'ties_top21_share').value)
        cases.append((SL[sc], s >= 0.999, f'{s:g}'))
    return over(cases, 'all', [VA])


@check('spike_opposite', '10x spike, default, WSPI: log mu_L rises and alpha R - beta W_E falls')
def _():
    cases = []
    for sc in M4:
        a = float(C.rb(sc, 'default', 'WSPI', 'size10', 'dlogmu_mean').value)
        b = float(C.rb(sc, 'default', 'WSPI', 'size10', 'dexpo_mean').value)
        cases.append((SL[sc], a > 0 and b < 0, f'dlogmu {a:+.3f}, dexpo {b:+.3f}'))
    return over(cases, 'all', [RB])


@check('size_lowest', 'common perturbation, default: WSPI lowest dRank of the eight methods other than PFRF at each spike size')
def _():
    cases = []
    for sc in M4:
        for cond in ('size2', 'size5', 'size10', 'size20', 'size50'):
            v = {m: float(C.rb(sc, 'default', m, cond).value) for m in ALL9 if m != 'PFRF'}
            b = min(v, key=v.get)
            cases.append((f'{SL[sc]} {cond}', b == 'WSPI', f'lowest {b}'))
    return over(cases, 'all', [RB])


@check('rrd_eq_more_robust', 'common perturbation, equal window: RRD dRank below WSPI in every condition (value and block test)')
def _():
    d = C.load(RBT)
    rb_ = C.load(RB)
    cases = []
    for sc in M4:
        for cond in sorted(rb_[(rb_.scenario == sc) & (rb_.config == 'eq64')].condition.unique()):
            lo = float(C.rb(sc, 'eq64', 'RRD', cond).value) < float(C.rb(sc, 'eq64', 'WSPI', cond).value)
            r = d[(d.run_group == f'{sc}/eq64') & (d.scenario == cond) & (d.method == 'RRD') & (d.metric == 'robustness_distortion')]
            v = r.verdict.iloc[0] if len(r) == 1 else f'{len(r)} rows'
            cases.append((f'{SL[sc]} {cond}', lo and v == 'ref_worse', f'lower={lo}, {v}'))
    return over(cases, 'all', [RB, RBT])


def _series(sc, roll):
    """RSI@10 per common window of the nine methods, as the centred moving average of Figures 6 and 7."""
    pdir = C.RESULTS / PROT / sc / 'protocol'
    fr = {m: pd.read_csv(pdir / f'{m}_protocol.csv', usecols=['window_id', RSI]).set_index('window_id')[RSI] for m in ALL9}
    common = sorted(set.intersection(*(set(s.index) for s in fr.values())))
    df = pd.DataFrame({m: fr[m].loc[common].to_numpy() for m in ALL9})
    return df.rolling(roll, center=True, min_periods=roll // 2).mean(), len(common)


@check('yt_fig_most', 'Figure 6 (YouTube, 24-h centred moving average): share of the period in which WSPI and DTCWT+AF each lie above both AF and RRD (most = above one half)')
def _():
    df, n = _series('youtube_hourly', 24)
    lo = df[['AF', 'RRD']].max(axis=1)
    a, b = float((df['WSPI'] > lo).mean()), float((df['DTCWT+AF'] > lo).mean())
    return Res(a > 0.5 and b > 0.5, f'WSPI {a:.3f}, DTCWT+AF {b:.3f} of {n} windows', f'{PROT}/youtube_hourly/protocol/*_protocol.csv')


@check('yt_fig_half', 'Figure 6: share of the period in which PFRF lies above WSPI (about half: 0.4 to 0.6)')
def _():
    df, n = _series('youtube_hourly', 24)
    a = float((df['PFRF'] > df['WSPI']).mean())
    return Res(0.4 <= a <= 0.6, f'{a:.3f} of {n} windows', f'{PROT}/youtube_hourly/protocol/*_protocol.csv')


@check('taxi_fig_top', 'Figure 7 (hourly taxi, 168-h centred moving average): WSPI or DTCWT+AF is above all methods except PFRF at every point')
def _():
    df, n = _series('taxi_hourly', 168)
    df = df.dropna()   # points without a value are not drawn
    top = df[['WSPI', 'DTCWT+AF']].max(axis=1)
    rest = df[[m for m in ALL9 if m not in ('WSPI', 'DTCWT+AF', 'PFRF')]].max(axis=1)
    k, p = int((top > rest).sum()), float((top > df['PFRF']).mean())
    return Res(k == len(df), f'{k} of {len(df)} drawn points (also above PFRF: {p:.3f})', f'{PROT}/taxi_hourly/protocol/*_protocol.csv')


@check('taxi_fig_apart', 'Figure 7: the lower of WSPI and DTCWT+AF stays above the highest of AF, RRD and DWT+AF at every point')
def _():
    df, n = _series('taxi_hourly', 168)
    df = df.dropna()
    gap = df[['WSPI', 'DTCWT+AF']].min(axis=1) - df[['AF', 'RRD', 'DWT+AF']].max(axis=1)
    return Res(float(gap.min()) > 0, f'smallest gap {gap.min():.4f} over {len(df)} drawn points', f'{PROT}/taxi_hourly/protocol/*_protocol.csv')


@check('taxi5_top', '5-minute taxi, default: WSPI has the highest RSI@10 of the nine')
def _():
    b, v = best('default', 'taxi_5min', RSI, ALL9)
    return Res(b == 'WSPI', f'{b} {v[b]:.4f}', VA)


@check('taxi_dr_lowest', 'taxi, default: WSPI lowest dRank of the eight methods other than PFRF')
def _():
    return _dr_low(TAXI3)


@check('taxi_best_ndcg', 'taxi, default: the best baseline other than PFRF has a significantly higher NDCG@10 than WSPI')
def _():
    cases = []
    for sc in TAXI3:
        b, v = best('default', sc, NDCG, BASE5)
        cases.append((SL[sc], v[b] > val('default', sc, 'WSPI', NDCG) and vd('default', sc, NDCG, b) == 'ref_worse',
                      f'{b} {v[b]:.4f}, {vd("default", sc, NDCG, b)}'))
    return over(cases, 'all', [VA, TA])


@check('time_doubling', 'cost benchmark, WSPI, 10^4 items: time ratio for each doubling of N from 64 to 256 (roughly doubled: 1.8 to 2.2)')
def _():
    d = C.load(RG)
    d = d[(d.method == 'WSPI') & (d.M == 10000)].set_index('N')['us_per_item_median']
    r = [float(d[128] / d[64]), float(d[256] / d[128])]
    return Res(all(1.8 <= x <= 2.2 for x in r), f'64->128 x{r[0]:.2f}, 128->256 x{r[1]:.2f}', RG)


@check('score_share_minor', 'real runs, WSPI: share of the evaluation run time spent scoring (most is elsewhere = below one half)')
def _():
    d = C.load(RS)
    cases = [(SL[r.scenario], r.score_share < 0.5, f'{r.score_share:.3f}') for r in d[d.method == 'WSPI'].itertuples()]
    return over(cases, 'all', [RS])


def _ab(sc, var, col, fam='ablation'):
    return float(C.ab(sc, var, col, fam).value)


def _abt(sc, m, met):
    d = C.load(ABT)
    r = d[(d.run_group == sc) & (d.scenario == 'ablation') & (d.method == m) & (d.metric == met)]
    if len(r) != 1:
        raise LookupError(f'{ABT}: {len(r)} rows for {sc} {m} {met}')
    return r.verdict.iloc[0]


@check('terms_each_part', 'ablation: each structural term alone puts dRank between the full index and the trend term; the full index is the lowest')
def _():
    cases = []
    for sc in M4:
        w, t, r, e = (_ab(sc, v, DR) for v in ('WSPI', 'Trend', 'Trend+R', 'Trend+WE'))
        cases.append((SL[sc], w < r < t and w < e < t, f'WSPI {w:.2f}, +R {r:.2f}, +WE {e:.2f}, trend {t:.2f}'))
    return over(cases, 'all', [AB])


@check('terms_rsi', 'ablation: RSI@10 of WSPI vs the trend term: n.s. on YouTube, taxi 1h and 30m; significantly higher on 5m')
def _():
    cases = []
    for sc in M4:
        v = _abt(sc, 'Trend', RSI)
        cases.append((SL[sc], v == ('ref_better' if sc == 'taxi_5min' else 'n.s.'), v))
    return over(cases, 'all', [ABT])


@check('dwt_dr_double', 'ablation: dRank of WSPI with DWT is more than twice that of WSPI')
def _():
    cases = []
    for sc in M4:
        q = _ab(sc, 'DWT-WSPI', DR) / _ab(sc, 'WSPI', DR)
        cases.append((SL[sc], q > 2, f'x{q:.2f}'))
    return over(cases, 'all', [AB])


@check('dwt_ndcg_3of4', 'ablation: WSPI with DWT has a higher NDCG@10 than WSPI in 3 of 4 scenarios, not on 5m')
def _():
    cases = []
    for sc in M4:
        a, b = _ab(sc, 'DWT-WSPI', NDCG), _ab(sc, 'WSPI', NDCG)
        cases.append((SL[sc], a > b, f'{a:.4f} vs {b:.4f}, {_abt(sc, "DWT-WSPI", NDCG)}'))
    r = over(cases, 3, [AB, ABT])
    r.ok = r.ok and not cases[3][1]
    return r


@check('trend_gap_same', 'ablation: trend term with DWT vs with DTCWT, as for the full index: lower RSI@10 and a dRank at least 1.5 times higher')
def _():
    cases = []
    for sc in M4:
        q = _ab(sc, 'DWT-Trend', DR) / _ab(sc, 'Trend', DR)
        r = _ab(sc, 'DWT-Trend', RSI) < _ab(sc, 'Trend', RSI)
        cases.append((SL[sc], q >= 1.5 and r, f'dRank x{q:.2f}, RSI lower={r}'))
    return over(cases, 'all', [AB])


@check('dwt_largest_drop', 'ablation: replacing DTCWT by DWT lowers RSI@10 and raises dRank of WSPI more than removing R, W_E or both')
def _():
    cases = []
    for sc in M4:
        w_r, w_d = _ab(sc, 'WSPI', RSI), _ab(sc, 'WSPI', DR)
        dd = {v: (w_r - _ab(sc, v, RSI), _ab(sc, v, DR) - w_d) for v in ('DWT-WSPI', 'Trend', 'Trend+R', 'Trend+WE')}
        ok = all(dd['DWT-WSPI'][0] > dd[v][0] and dd['DWT-WSPI'][1] > dd[v][1] for v in ('Trend', 'Trend+R', 'Trend+WE'))
        cases.append((SL[sc], ok, 'RSI drop ' + ', '.join(f'{k} {x[0]:+.4f}' for k, x in dd.items()) +
                      '; dRank rise ' + ', '.join(f'{k} {x[1]:+.2f}' for k, x in dd.items())))
    return over(cases, 'all', [AB])


@check('e1_zero', 'shift test: energy of DTCWT level 1 does not change (coefficient of variation 0)')
def _():
    d = C.load(SH)
    d = d[(d['transform'] == 'DTCWT') & (d.metric == 'cv_E_1')]
    cases = [(SL[sc], abs(float(x)) < 1e-12, f'{float(x):.1e}') for sc, x in zip(d.scenario, d['mean'])]
    return over(cases, 'all', [SH])


@check('shift_sig_all', 'shift test: every DTCWT vs DWT difference (6 statistics x 4 scenarios) significant in favour of DTCWT')
def _():
    d = C.load(SHT)
    cases = [(f'{SL[r.scenario]} {r.metric}', r.verdict == 'ref_better', r.verdict) for r in d.itertuples()]
    return over(cases, 'all', [SHT])


def _sy(shape, age, name, metric):
    d = C.load(SY)
    r = d[(d['shape'] == shape) & (d.age == age) & (d.name == name) & (d.metric == metric)]
    if len(r) != 1:
        raise LookupError(f'{SY}: {len(r)} rows for {shape} {age} {name} {metric}')
    return float(r['mean'].iloc[0])


@check('dtcwtaf_never', 'synthetic event, recent part: share of steps in which the DTCWT+AF score falls (never = 0)')
def _():
    cases = [(s, _sy(s, 'recent', 'DTCWT+AF', 'share_wrong') == 0, f"{_sy(s, 'recent', 'DTCWT+AF', 'share_wrong'):g}") for s in ('spike', 'burst3')]
    return over(cases, 'all', [SY])


@check('ewma_always', 'synthetic event, recent part: the EWMA score never falls and its mean step is positive (a rise at every single step is not stored)')
def _():
    cases = [(s, _sy(s, 'recent', 'EWMA-eq', 'share_wrong') == 0 and _sy(s, 'recent', 'EWMA-eq', 'mean_dlog') > 0,
              f"falls {_sy(s, 'recent', 'EWMA-eq', 'share_wrong'):g}, mean dlog {_sy(s, 'recent', 'EWMA-eq', 'mean_dlog'):.4f}") for s in ('spike', 'burst3')]
    return over(cases, 'all', [SY])


@check('sma_flat', 'synthetic event, recent part: the 64-slot SMA score does not change')
def _():
    cases = [(s, _sy(s, 'recent', 'SMA', 'mean_abs_dlog') < 1e-12, f"{_sy(s, 'recent', 'SMA', 'mean_abs_dlog'):g}") for s in ('spike', 'burst3')]
    return over(cases, 'all', [SY])


@check('mid_half', 'synthetic event, middle of the window: the WSPI score falls in about half of the steps with either transform (0.4 to 0.6)')
def _():
    cases = []
    for n in ('WSPI', 'DWT-WSPI'):
        for s in ('spike', 'burst3'):
            x = _sy(s, 'middle', n, 'share_wrong')
            cases.append((f'{n} {s}', 0.4 <= x <= 0.6, f'{x:.3f}'))
    return over(cases, 'all', [SY])


@check('linear_more_robust', 'fusion: the linear form has a lower dRank than the exponential form')
def _():
    cases = [(SL[sc], _ab(sc, 'Linear', DR, 'fusion') < _ab(sc, 'WSPI', DR, 'fusion'),
              f"{_ab(sc, 'Linear', DR, 'fusion'):.2f} vs {_ab(sc, 'WSPI', DR, 'fusion'):.2f}") for sc in M4]
    return over(cases, 'all', [AB])


@check('product_mixed', 'fusion: the product form has a lower dRank than the exponential form on taxi and a higher one on YouTube')
def _():
    cases = []
    for sc in M4:
        p, w = _ab(sc, 'Product', DR, 'fusion'), _ab(sc, 'WSPI', DR, 'fusion')
        cases.append((SL[sc], (p > w) if sc == 'youtube_hourly' else (p < w), f'{p:.2f} vs {w:.2f}'))
    return over(cases, 'all', [AB])


@check('ml_daily_ndcg', 'MovieLens daily, default: WSPI NDCG@10 significantly above each of the six baselines')
def _():
    cases = [(b, vd('default', 'movielens_daily', NDCG, b) == 'ref_better', vd('default', 'movielens_daily', NDCG, b)) for b in BASE6]
    return over(cases, 'all', [TA])


@check('ml_ndcg_best', 'MovieLens, default: highest NDCG@10 of the nine = DTCWT+AF (daily), DWT+AF (weekly)')
def _():
    cases = [(SL[sc], best('default', sc, NDCG, ALL9)[0] == w, best('default', sc, NDCG, ALL9)[0])
             for sc, w in (('movielens_daily', 'DTCWT+AF'), ('movielens_weekly', 'DWT+AF'))]
    return over(cases, 'all', [VA])


@check('ml_rho_best', 'MovieLens, default: WSPI has the highest rho of the nine (daily and weekly)')
def _():
    cases = [(SL[sc], best('default', sc, RHO, ALL9)[0] == 'WSPI', best('default', sc, RHO, ALL9)[0]) for sc in ML]
    return over(cases, 'all', [VA])


@check('rsi_best_wspi3', 'default: WSPI has the highest RSI@10 of the nine on YouTube, taxi 5m and ML daily; DTCWT+AF on ML weekly, significant')
def _():
    cases = [(SL[sc], best('default', sc, RSI, ALL9)[0] == 'WSPI', best('default', sc, RSI, ALL9)[0])
             for sc in ('youtube_hourly', 'taxi_5min', 'movielens_daily')]
    b = best('default', 'movielens_weekly', RSI, ALL9)[0]
    v = vd('default', 'movielens_weekly', RSI, 'DTCWT+AF')
    cases.append(('ML weekly', b == 'DTCWT+AF' and v == 'ref_worse', f'{b}, {v}'))
    return over(cases, 'all', [VA, TA])


@check('daily_cycle_most', 'hourly and 30-minute taxi entries: share whose zone has another entry one day earlier or later, within one hour (most = above one half)')
def _():
    cases = []
    for sc in TAXI3[:2]:
        ev = pd.read_csv(C.RESULTS / 'T2.4_responsiveness' / sc / 'events.csv', usecols=['item_id', 't0'])
        sm = float(C.cfgv('WSPI', 'slot_minutes', sc).value)
        day, tol = int(round(1440 / sm)), int(round(60 / sm))
        hit = 0
        for _, g in ev.groupby('item_id'):
            t = pd.Series(sorted(g.t0))
            for x in t:
                if (t - x).abs().between(day - tol, day + tol).any():
                    hit += 1
        cases.append((SL[sc], hit / len(ev) > 0.5, f'{hit / len(ev):.3f} of {len(ev)}'))
    return over(cases, 'all', ['T2.4_responsiveness/<taxi>/events.csv', CFG])


@check('median_half_rule', 'responsiveness: the median is undefined exactly when more than half of the entries are missed')
def _():
    d = C.load(RE_)
    bad = d[(d.median_undefined.astype(str) == 'True') != (d.miss_rate > 0.5)]
    return Res(len(bad) == 0, f'{len(d) - len(bad)} of {len(d)} rows agree', RE_)


@check('af_none', 'responsiveness, default: AF misses no entry')
def _():
    cases = [(SL[sc], float(C.resp(sc, 'AF', 'miss_rate').value) == 0, f"{float(C.resp(sc, 'AF', 'miss_rate').value):g}") for sc in M4]
    return over(cases, 'all', [RE_])


def _ret(sc, m, cfg='default'):
    d = C.load(RET)
    r = d[(d.scenario == sc) & (d.config == cfg) & (d.variant == 'main') & (d.reference == 'WSPI') & (d.method == m)]
    if len(r) != 1:
        raise LookupError(f'{RET}: {len(r)} rows for {sc} {cfg} {m}')
    return r.verdict.iloc[0]


@check('delay_sig', 'responsiveness, default: WSPI mean delay significantly longer than that of AF and of DTCWT+AF')
def _():
    cases = [(f'{SL[sc]} {m}', _ret(sc, m) == 'ref_slower', _ret(sc, m)) for sc in M4 for m in ('AF', 'DTCWT+AF')]
    return over(cases, 'all', [RET])


@check('long_slower', 'responsiveness, equal window: WSPI significantly faster than RRD, VSE and CompoundPop on taxi; n.s. on YouTube')
def _():
    cases = []
    for sc in M4:
        for m in LONG3:
            v = _ret(sc, m, 'equal64')
            cases.append((f'{SL[sc]} {m}', v == ('n.s.' if sc == 'youtube_hourly' else 'ref_faster'), v))
    return over(cases, 'all', [RET])


@check('af_one_slot', 'worst hourly-taxi example of Figure 10: delay of AF')
def _():
    d = C.load(EX)
    r = d[(d.scenario == 'taxi_hourly') & (d.example == 'worst_for_reference')]
    return Res(float(r.delay_AF.iloc[0]) == 1, f"AF {r.delay_AF.iloc[0]:g} slot, WSPI {r.delay_WSPI.iloc[0]:g}", EX)


@check('inelig_zero', 'responsiveness, default: share of missed entries caused by the 32-slot rule, wavelet-based methods, YouTube, taxi 1h and 30m')
def _():
    cases = [(f'{SL[sc]} {m}', float(C.resp(sc, m, 'share_miss_ineligible_whole_run').value) == 0,
              f"{float(C.resp(sc, m, 'share_miss_ineligible_whole_run').value):g}") for sc in M4[:3] for m in WAV3]
    return over(cases, 'all', [RE_])


@check('random_largest', 'common perturbation, default, spike in a random slot: WSPI has the largest dRank of the nine')
def _():
    cases = []
    for sc in M4:
        v = {m: float(C.rb(sc, 'default', m, 'pos_random').value) for m in ALL9}
        b = max(v, key=v.get)
        cases.append((SL[sc], b == 'WSPI', f'largest {b}'))
    return over(cases, 'all', [RB])


@check('af_delay_12', 'responsiveness, default: median delay of AF is one or two slots')
def _():
    cases = [(SL[sc], float(C.resp(sc, 'AF', 'median_delay').value) in (1.0, 2.0), f"{float(C.resp(sc, 'AF', 'median_delay').value):g}") for sc in M4]
    return over(cases, 'all', [RE_])


@check('af_best_ndcg', 'default: AF has the highest NDCG@10 of the nine on YouTube, taxi 1h and 30m, not on 5m')
def _():
    cases = []
    for sc in M4:
        b = best('default', sc, NDCG, ALL9)[0]
        cases.append((SL[sc], (b == 'AF') == (sc != 'taxi_5min'), b))
    return over(cases, 'all', [VA])


@check('s_ndcg_short', 'Table 15: scenarios in which at least one of the six baselines (7 slots) has a significantly higher NDCG@10 than WSPI (5 of 6)')
def _():
    cases = []
    for sc in M6:
        w = [b for b in BASE6 if vd('default', sc, NDCG, b) == 'ref_worse']
        cases.append((SL[sc], len(w) >= 1, ', '.join(w) or 'none'))
    return over(cases, 5, [TA])


def _long(met, want, methods=LONG3):
    cases = [(f'{SL[sc]} {m}', vd('equal64', sc, met, m) == want, vd('equal64', sc, met, m)) for sc in M6 for m in methods]
    return over(cases, 'all', [TA])


@check('s_ndcg_long', 'Table 15: WSPI NDCG@10 significantly above RRD, VSE and CompoundPop (64 slots), six scenarios')
def _():
    return _long(NDCG, 'ref_better')


@check('s_rho_long', 'Table 15: WSPI rho significantly above RRD, VSE and CompoundPop (64 slots), six scenarios')
def _():
    return _long(RHO, 'ref_better')


@check('s_rho_short', 'Table 15: WSPI rho significantly above the six baselines on taxi and MovieLens, significantly below AF on YouTube')
def _():
    cases = [(f'{SL[sc]} {b}', vd('default', sc, RHO, b) == 'ref_better', vd('default', sc, RHO, b)) for sc in M6[1:] for b in BASE6]
    v = vd('default', 'youtube_hourly', RHO, 'AF')
    cases.append(('YouTube AF', v == 'ref_worse', v))
    return over(cases, 'all', [TA])


@check('s_rsi_long', 'Table 15: WSPI RSI@10 significantly below RRD (64 slots), six scenarios')
def _():
    return _long(RSI, 'ref_worse', ['RRD'])


@check('s_dr_long', 'Table 15: WSPI dRank significantly above RRD (64 slots) (4 of 6)')
def _():
    cases = [(SL[sc], vd('equal64', sc, DR, 'RRD') == 'ref_worse', vd('equal64', sc, DR, 'RRD')) for sc in M6]
    return over(cases, 4, [TA])


@check('si_ml_eq_acc', 'MovieLens, equal window: WSPI NDCG@10 and rho significantly above SMA, RRD and VSE')
def _():
    cases = [(f'{SL[sc]} {met} {m}', vd('equal64', sc, met, m) == 'ref_better', vd('equal64', sc, met, m))
             for sc in ML for met in (NDCG, RHO) for m in ('SMA', 'RRD', 'VSE')]
    return over(cases, 'all', [TA])


@check('si_ml_eq_rsi', 'MovieLens, equal window: SMA, RRD and VSE have a higher RSI@10 than WSPI')
def _():
    cases = [(f'{SL[sc]} {m}', val('equal64', sc, m, RSI) > val('equal64', sc, 'WSPI', RSI),
              f"{val('equal64', sc, m, RSI):.4f} vs {val('equal64', sc, 'WSPI', RSI):.4f}, {vd('equal64', sc, RSI, m)}")
             for sc in ML for m in ('SMA', 'RRD', 'VSE')]
    return over(cases, 'all', [VA, TA])


# ------------------------------------------------------------------ words and rules
class WTok:
    __slots__ = ('file', 'line', 'col', 'end', 'word', 'ctx', 'status', 'category', 'expected', 'found',
                 'check', 'source', 'rule')

    def __init__(self, file, line, col, end, word, ctx):
        self.file, self.line, self.col, self.end, self.word, self.ctx = file, line, col, end, word, ctx
        self.status, self.category = 'unchecked', ''
        self.expected = self.found = self.check = self.source = self.rule = ''


class Spec:
    def __init__(self, cat, key=None, expect='', note=''):
        self.cat, self.key, self.expect, self.note = cat, key, expect, note


def Claim(key, expect):
    return Spec('claim', key, expect)


def Count(key, expect):
    return Spec('count', key, expect)


def Struct(key, expect):
    return Spec('structure', key, expect)


def Design(note):
    return Spec('design', note=note)


def Lang(note):
    return Spec('language', note=note)


def apply_spec(t, sp, rule):
    t.rule, t.category = rule, sp.cat
    if sp.key is None:
        t.status, t.source = sp.cat, sp.note
        return
    r = run_check(sp.key)
    t.check, t.expected, t.found = sp.key, sp.expect, r.found
    t.source = f'{r.what} | {r.source}'
    if r.ok is None:
        t.status = 'error'
    elif not r.ok:
        t.status = 'mismatch'
    else:
        t.status = 'ok' if sp.cat == 'claim' else f'{sp.cat}-ok'


class Engine:
    def __init__(self, files):
        self.lines, self.toks, self.problems, self.extra, self.body = {}, [], [], [], {}
        for tag, path in files.items():
            ls = Path(path).read_text(encoding='utf-8').split('\n')
            self.lines[tag] = ls
            a = next(i for i, l in enumerate(ls) if l.startswith(r'\begin{document}'))
            b = next((i for i, l in enumerate(ls) if l.startswith(r'\begin{thebibliography}')), len(ls))
            self.body[tag] = (a + 1, b)
            for i in range(a + 1, b):
                s = C.masked(ls[i])
                for m in WORD_RE.finditer(s):
                    x, y = m.span(1)
                    self.toks.append(WTok(tag, i + 1, x, y, ls[i][x:y], ls[i][max(0, x - 70):y + 50]))

    def hits(self, tag, anchor):
        a, b = self.body[tag]
        return [(i, m.start()) for i, l in enumerate(self.lines[tag]) if a <= i < b for m in re.finditer(re.escape(anchor), l)]

    def text(self, tag, anchor, specs, rule=None):
        """The words inside `anchor` (a literal piece of one line), in order, get these specs."""
        h = self.hits(tag, anchor)
        rule = rule or anchor[:70]
        if len(h) != 1:
            self.problems.append(f'[{tag}] anchor found {len(h)} times: {anchor!r}')
            return
        i, j = h[0]
        span = [t for t in self.toks if t.file == tag and t.line == i + 1 and j <= t.col and t.end <= j + len(anchor)]
        if len(span) != len(specs):
            self.problems.append(f'[{tag}:{i + 1}] {len(span)} words but {len(specs)} specs in {anchor!r}: {[t.word for t in span]}')
            return
        for t, sp in zip(span, specs):
            if t.status != 'unchecked':
                self.problems.append(f'[{tag}:{i + 1}] word {t.word!r} covered twice ({t.rule} / {rule})')
            apply_spec(t, sp, rule)

    def pattern(self, regex, spec, tags=None, rule=None, group=1):
        """Every unchecked word at `group` of `regex` gets `spec` (used for plain wording only)."""
        rx = re.compile(regex)
        for tag, ls in self.lines.items():
            if tags and tag not in tags:
                continue
            for i, l in enumerate(ls):
                for m in rx.finditer(l):
                    a, b = m.span(group)
                    for t in self.toks:
                        if t.file == tag and t.line == i + 1 and a <= t.col and t.end <= b and t.status == 'unchecked':
                            apply_spec(t, spec, rule or regex)

    def claim(self, tag, anchor, key, expect):
        """A result statement without a number word, next to the checked words."""
        h = self.hits(tag, anchor)
        if len(h) != 1:
            self.problems.append(f'[{tag}] claim anchor found {len(h)} times: {anchor!r}')
            return
        r = run_check(key)
        st = 'error' if r.ok is None else 'ok' if r.ok else 'mismatch'
        self.extra.append(dict(kind='claim', file=tag, line=h[0][0] + 1, col=h[0][1], word='', category='claim', status=st,
                               expected=expect, found=r.found, check=key, source=f'{r.what} | {r.source}',
                               rule=anchor[:70], context=anchor))

    # -------- structure checks read the tex itself
    def line_of(self, tag, anchor):
        h = self.hits(tag, anchor)
        if len(h) != 1:
            raise LookupError(f'anchor found {len(h)} times: {anchor!r}')
        return self.lines[tag][h[0][0]], h[0][0]


E = None   # the engine, set in main(); the structure checks below read it


def _markers(tag, anchor, present, absent=()):
    l, _ = E.line_of(tag, anchor)
    miss = [p for p in present if p not in l]
    extra = [p for p in absent if p in l]
    return Res(not miss and not extra, f'{len(present) - len(miss)} of {len(present)} listed' +
               (f'; missing {miss}' if miss else '') + (f'; also {extra}' if extra else ''), 'text of the same paragraph')


def _list_after(tag, anchor, n, stop='.'):
    l, _ = E.line_of(tag, anchor)
    rest = l[l.index(anchor) + len(anchor):]
    rest = rest[:rest.index(stop)] if stop in rest else rest
    items = [x.strip() for x in re.split(r',\s*(?:and\s+)?|\s+and\s+', rest) if x.strip()]
    return Res(len(items) == n, f'{len(items)} items: {items}', 'text of the same sentence')


def _table_rows(tag, label):
    ls = E.lines[tag]
    i0 = next(i for i, l in enumerate(ls) if f'\\label{{{label}}}' in l)
    i = next(k for k in range(i0, len(ls)) if '\\midrule' in ls[k])
    rows = []
    for k in range(i + 1, len(ls)):
        if '\\bottomrule' in ls[k]:
            break
        if '&' in ls[k]:
            rows.append(ls[k].split('&')[0].strip())
    return rows


@check('stages4', 'Stage headings of Subsection 3.4 (text)')
def _():
    s = re.findall(r'\\textbf\{Stage (\d) ---', '\n'.join(E.lines['main']))
    return Res(s == ['1', '2', '3', '4'], f'stages {s}', 'main text')


@check('features3', 'structural features listed in Stage 3 (text)')
def _():
    return _markers('main', 'Stage 3 --- Structural feature extraction.', ['$\\mu_L$', '$R$', '$W_E$'])


@check('aspects4', 'rows of the metrics table (text)')
def _():
    r = _table_rows('main', 'tab:metrics')
    return Res(len(r) == 4, f'{len(r)} rows: {r}', 'Table tab:metrics')


@check('ablation_rows6', 'rows of the ablation table (text)')
def _():
    r = _table_rows('main', 'tab:ablation')
    return Res(len(r) == 6, f'{len(r)} rows', 'Table tab:ablation')


@check('areas4', 'areas named in the introduction of Section 2 (text)')
def _():
    return _markers('main', 'falls into four main areas.', ['The first', 'The second', 'The third', 'The fourth'], ['The fifth'])


@check('gaps3', 'research gaps listed (text)')
def _():
    return _markers('main', 'three distinct research gaps.', ['First,', 'Second,', 'Third,'], ['Fourth,'])


@check('questions3', 'questions listed in Subsection 3.1 (text)')
def _():
    l, _ = E.line_of('main', 'We separate three questions.')
    n = len(re.findall(r'\\textit\{', l[:l.index('A good score')]))
    return Res(n == 3, f'{n} questions in italics', 'text of the same paragraph')


@check('criteria3', 'criteria (a), (b), (c) listed (text)')
def _():
    return _markers('main', 'A good score should therefore meet three criteria', ['(a)', '(b)', '(c)'], ['(d)'])


@check('props3', 'properties of the popularity definition listed (text)')
def _():
    return _list_after('main', 'This definition has three core properties: ', 3)


@check('ways2', 'items of the list after "two fundamental ways" (text)')
def _():
    _, i = E.line_of('main', 'in two fundamental ways:')
    ls = E.lines['main']
    k = next(j for j in range(i, len(ls)) if '\\end{itemize}' in ls[j])
    n = sum(1 for j in range(i, k) if ls[j].lstrip().startswith('\\item'))
    return Res(n == 2, f'{n} items', 'itemize after the sentence')


@check('principles2', 'fusion principles listed (text)')
def _():
    return _markers('main', 'The fusion design rests on two principles.', ['First,', 'Second,'], ['Third,'])


@check('benefits2', 'benefits listed (text)')
def _():
    return _list_after('main', 'This independence brings two benefits: ', 2)


@check('effects2', 'effect sizes listed (text)')
def _():
    return _list_after('main', 'we also report two effect sizes, ', 2)


@check('base6_list', 'baselines listed (text)')
def _():
    return _list_after('main', 'against six baseline methods: ', 6)


@check('points3', 'points of the ablation discussion (text)')
def _():
    a = _markers('main', 'The results show three points. First,', ['First,'])
    b = E.hits('main', 'Second, replacing DTCWT with DWT')
    c = E.hits('main', 'Third, we compared the exponential fusion')
    return Res(a.ok and len(b) == 1 and len(c) == 1, f'First, Second, Third found: {a.ok}, {len(b)}, {len(c)}', 'Subsection 4.8')


@check('directions4', 'future-work directions listed (text)')
def _():
    return _markers('main', 'Four directions follow from these limits.', ['The first', 'The second', 'The third', 'The fourth'], ['The fifth'])


@check('props2_ml', 'limiting properties of MovieLens listed (text)')
def _():
    return _markers('main', 'Two properties of this dataset limit what it can show.', ['First,', 'Second,'], ['Third,'])


@check('dwt_props3', 'DTCWT properties listed (text)')
def _():
    return _list_after('main', 'offers three structural properties over classic DWT: ', 3)


@check('factors3', 'factors of Hamdeni et al. listed (text)')
def _():
    return _list_after('main', 'identified three main factors in computing a popularity score: ', 3)


# ------------------------------------------------------------------ reasons of the design and language rules
LIT = 'describes the cited literature, not a result of this study'
DIST = 'distributive wording ("each item", "each slot"), not a count'
PLAIN = 'plain wording, not a quantity'
BOUND = '"at most" before a number that check_paper_numbers.py checks'
THEORY = 'property of the transform from its definition [27,37], not a measured result'
DEF = 'definition of the method or protocol, not a result'
FIG = 'layout of the figure, set in the plotting code'
CONCL = 'conclusion stated with its numbers, which check_paper_numbers.py checks'
L, D = Lang, Design


def rules_main(E):
    T = lambda anchor, specs: E.text('main', anchor, specs)
    # abstract
    T('score each item from its raw counts', [L(DIST)])
    T('study three training-free methods', [Count('n_wav', '3')])
    T('with six conventional baselines on YouTube, NYC Yellow Taxi (three granularities) and MovieLens (two)',
      [Count('n_base', '6'), Count('n_taxi', '3'), Count('n_ml', '2')])
    T('has the highest ranking stability (RSI@10) in all six scenarios', [Claim('rsi_top_wav6', 'all 6'), Count('n_scen6', '6')])
    # introduction
    T("estimating each item's popularity", [L(DIST)])
    T('This problem is most acute', [L(PLAIN)])
    T('online media, urban mobility and recommender systems, the three settings we study in this paper', [Count('n_datasets', '3')])
    T('a three-factor composite model', [L(LIT)])
    T('Each method brings one aspect of demand', [L(LIT), L(LIT)])
    T('The demand signal is processed at a single scale', [L(LIT)])
    T('A method that works at one scale', [L(PLAIN)])
    T('Most also model popularity at a single scale', [L(LIT), L(LIT)])
    T('shifting the input by one sample at the window boundary can spuriously', [D(THEORY)])
    T('process the demand signal at a single scale: they build', [L(PLAIN)])
    T('are blended into one quantity', [L(PLAIN)])
    T('We study three training-free wavelet-based methods', [Count('n_wav', '3')])
    T('The demand signal of each item is decomposed', [L(DIST)])
    T('Three structural features are then extracted', [Struct('features3', '3')])
    T('the total interactions each item receives', [L(DIST)])
    T('we design a four-aspect protocol', [Struct('aspects4', '4')])
    T('We also measure how fast each method responds', [L(DIST)])
    T('and propose three training-free wavelet-based methods', [Count('n_wav', '3')])
    T('that is monotonic in each feature', [L(DIST)])
    T('It integrates three structural features extracted from DTCWT: the trend volume and two measures',
      [Struct('features3', '3'), D('the two structural terms R and W_E (Subsection 3.5)')])
    T('All results come with block-bootstrap confidence intervals', [D('reporting rule of the paper (Subsection 4.1)')])
    T('evaluation on three public datasets with different dynamics (YouTube, NYC Yellow Taxi at three granularities and MovieLens at two), '
      'over 131,443 evaluation windows, in two configurations',
      [Count('n_datasets', '3'), Count('n_taxi', '3'), Count('n_ml', '2'), Count('n_configs', '2')])
    T(r'In the default configuration, one of the three wavelet-based methods has the highest RSI@10 in all six scenarios, and WSPI has '
      r'the lowest $\Delta$Rank of all methods except PFRF in all six.',
      [Claim('rsi_top_wav6', 'one wavelet-based method highest'), Count('n_wav', '3'), Claim('rsi_top_wav6', 'all 6'),
       Count('n_scen6', '6'), Claim('dr_low6', 'lowest except PFRF'), Claim('dr_low6', 'all 6'), Count('n_scen6', '6')])
    # related work
    T('falls into four main areas', [Struct('areas4', '4')])
    T('We review each area below', [L(DIST)])
    T('identified three main factors', [Struct('factors3', '3')])
    T('They stressed that every metric should', [L(LIT)])
    T('total requests in each time interval', [L(LIT)])
    T('to the requests in each interval', [L(LIT)])
    T('accounts for all three factors at once, it is regarded as the most complete', [L(LIT), L(LIT), L(LIT)])
    T('combined three parameters', [L(LIT)])
    T("raises or lowers each file's score in every period", [L(LIT), L(LIT)])
    T('All these methods share a common weakness', [L(LIT)])
    T('Most of these approaches model popularity at a single scale', [L(LIT), L(LIT)])
    T('One point is important here. All of this work', [L(PLAIN), L(LIT)])
    T('offers three structural properties over classic DWT', [Struct('dwt_props3', '3')])
    T('by using two parallel trees with a half-sample delay', [D(THEORY), D(THEORY)])
    T('three distinct research gaps', [Struct('gaps3', '3')])
    T('First, none of the access-statistics methods', [L(LIT)])
    T('model the score at a single scale. Third', [L(LIT)])
    T('This paper addresses this gap with three training-free methods', [Count('n_wav', '3')])
    T('For an item that is active in every slot, RRD [8] equals an SMA. These methods work at one scale', [D(DEF), L(PLAIN)])
    T('moves with every spike. The wavelet-based methods separate the scales inside one window', [L(PLAIN), L(PLAIN)])
    T('the three wavelet-based methods of this paper are the only reviewed', [Count('n_wav', '3')])
    # methods
    T('the index and its four stages', [Struct('stages4', '4')])
    T('(one hour, 30 minutes, 5 minutes, one day or one week, depending on the dataset)', [Count('slot_lengths', 'slot lengths')] * 3)
    T('At each ranking time', [L(DIST)])
    T('$N=64$ for the three wavelet-based methods', [Count('n_wav', '3')])
    T('gives each item a score', [L(DIST)])
    T('The function uses one item at a time', [D(DEF)])
    T('After each ranking the window moves forward by one slot', [L(DIST), D(DEF)])
    T('We separate three questions', [Struct('questions3', '3')])
    T('meet three criteria', [Struct('criteria3', '3')])
    T('access frequency weights each slot by its age', [L(DIST)])
    T('uses a two-channel filter bank', [D(THEORY)])
    T('Shifting the input by one sample at the window boundary can change', [D(THEORY)])
    T('It runs two real DWT trees', [D(THEORY)])
    T('At the first level the two trees use the same filters with a one-sample offset', [D(THEORY), D(THEORY)])
    T('is delayed by half a sample', [D(THEORY)])
    T('the wavelets of the two trees form', [D(THEORY)])
    T('Because the two parts are close to quadrature', [D(THEORY)])
    T('When the input moves by one sample', [D(THEORY)])
    T('The low-pass outputs of the two trees are real and are combined into one vector', [D(THEORY), D(THEORY)])
    T('twice that of a single tree', [D(THEORY), D(THEORY)])
    T('in two fundamental ways', [Struct('ways2', '2')])
    T('By extracting three structural features', [Struct('features3', '3')])
    T('The index has four main stages', [Struct('stages4', '4')])
    T('The most recent observation sits at the end', [L(PLAIN)])
    T('also extend the signal at both ends, at every level and in every window',
      [D('boundary extension inside the dtcwt library (colfilter/coldfilt), R10 section 3')] * 3)
    T('compares five padding modes and five extension modes', [Count('n_pad5', '5'), Count('n_ext5', '5')])
    T('by at most 0.002 and its rank distortion by at most 0.2', [L(BOUND), L(BOUND)])
    T('Symmetric extension gave the highest NDCG@10 in three scenarios', [Claim('sym_ndcg3', '3 of 4')])
    T('but it had a higher rank distortion in every scenario', [Claim('edge_dr_every', 'all 4')])
    T('WSPI kept the lowest rank distortion of the three wavelet-based methods under every mode',
      [Count('n_wav', '3'), Claim('wspi_min_mode', 'all modes x scenarios')])
    T(r'Three features ($\mu_L$, $R$, and $W_E$) are extracted', [Struct('features3', '3')])
    T('The three features are combined through', [Struct('features3', '3')])
    T(r'\caption{Four-stage WSPI pipeline.}', [Struct('stages4', '4')])
    T('three structural features are extracted from the resulting coefficients', [Struct('features3', '3')])
    T('the approximation energy and all detail sub-bands', [D(DEF)])
    T('The ratio $R$ always lies in $[0,1]$', [D('follows from the definition of R')])
    T('The two terms therefore act as one structural factor', [L(CONCL), L(CONCL)])
    T('the three extracted features are combined into a single score', [Struct('features3', '3'), D(DEF)])
    T('The fusion design rests on two principles', [Struct('principles2', '2')])
    T('Second, the two other structural features', [D('R and W_E')])
    T('The exponential model satisfies both principles', [Struct('principles2', '2')])
    T('the weights of the two structural features', [D(DEF)])
    T('both lie in $[0,1]$', [D('follows from the definitions of R and W_E')])
    T('the exponent always stays in', [D('follows from alpha=beta=1 and R, W_E in [0,1]')])
    T('a neutral default: both structural terms get unit weight', [D(DEF)])
    T(r'16 approximation coefficients, each covering eight slots, so $\mu_L$ tracks the trend at scales of about eight slots',
      [L(DIST), Count('eight_slots', '8'), Count('eight_slots', '8')])
    T('summarizes, as pseudocode, the four stages', [Struct('stages4', '4')])
    T('It runs independently for each data item', [L(DIST)])
    T('This independence brings two benefits', [Struct('benefits2', '2')])
    T('full parallel processing of all items', [L(PLAIN)])
    # evaluation setup
    T("It refers to each item's share", [L(DIST)])
    T('This definition has three core properties', [Struct('props3', '3')])
    T('in each evaluation window the ground truth is the number of interactions that each item receives', [L(DIST), L(DIST)])
    T('rests on four complementary aspects', [Struct('aspects4', '4')])
    T('summarizes these four aspects', [Struct('aspects4', '4')])
    T('against six baseline methods', [Count('n_base', '6')])
    T('we implement and evaluate two wavelet-based methods, DWT+AF and DTCWT+AF', [D('named in the same sentence')])
    T('these form the three wavelet-based methods', [Count('n_wav', '3')])
    T('Evaluation metrics of the four-aspect protocol', [Struct('aspects4', '4')])
    T('Top-$K$ sets in two consecutive windows', [D(DEF)])
    T('The four aspects answer different questions', [Struct('aspects4', '4')])
    T('never alone', [L('advice to the reader')])
    T('Each metric is averaged over the common windows, the windows that all nine methods evaluate',
      [L(DIST), D(DEF), Count('n_methods', '9')])
    T('Consecutive windows share most of their input', [L(PLAIN)])
    T('The block is one day on YouTube (24 slots) and one week on the taxi data',
      [Count('block_day_week', '1 day'), Count('block_day_week', '1 week')])
    T('a one-day block serves as a sensitivity check', [D('sensitivity block of T1.6 (tools/stats_report.py)')])
    T('Each method is compared with WSPI', [L(DIST)])
    T('Holm correction over the eight comparisons within each scenario and metric', [Count('holm8', '8'), L(DIST)])
    T('Every claim of a difference rests on this block test', [D('reporting rule; the claims checked here use the block verdicts')])
    T('we also report two effect sizes', [Struct('effects2', '2')])
    T('we use three data sources', [Count('n_datasets', '3')])
    T("We use each zone's share", [L(DIST)])
    T('keeping the most requested content', [L(PLAIN)])
    T('is examined at three aggregation levels', [Count('n_taxi', '3')])
    T('YouTube videos viewCount every hour', [D('title of the source dataset')])
    T('cumulative counts every hour', [D('recording interval of the source (695 hourly snapshots)')])
    T('the difference between two consecutive cumulative counts', [D(DEF)])
    T('(ten hours without data for any video)', [Count('gap10', '10')])
    T('The collection gap is never used as ground truth', [Claim('gap_never_truth', '0 test slots')])
    T('filter uses the whole period', [D(DEF)])
    T('in at least one window', [D(DEF)])
    T('No statistic of the whole period is used', [D(DEF)])
    T('checks every run', [Claim('audit_every_run', 'all runs')])
    T('the input of each window ends before its test slot', [Claim('audit_each_window', 'all')])
    T('was detected in every tested window', [Claim('posctl_every', 'all')])
    T('lists the configuration of every method. It is the same on all datasets',
      [Claim('config_same', 'same'), Claim('config_same', 'same')])
    T('Configuration of the nine methods, the same on all datasets', [Count('n_methods', '9'), Claim('config_same', 'same')])
    T('Product of 1.2 for each active slot and 0.8 for each inactive slot', [D(DEF), D(DEF)])
    T('in the equal-window comparison all methods use $N=64$', [Count('eq64_all', 'all 64')])
    T('Shared by all methods and datasets: a horizon of one slot', [Count('horizon_one', 'all'), Count('horizon_one', '1')])
    T('All methods were implemented in Python and run inside a single, unified pipeline, so that WSPI and every baseline share',
      [D('implementation (code release)')] * 3)
    T('the exact versions used for all reported runs', [D('versions of the run metadata, checked by check_paper_numbers.py')])
    T('including the four-aspect evaluation protocol and all baselines', [Struct('aspects4', '4'), D('code release')])
    T('exact library versions used for all reported runs', [D('versions of the run metadata, checked by check_paper_numbers.py')])
    # results 4.3
    T('WSPI is evaluated together with eight competing methods in six scenarios', [Count('n_competitors', '8'), Count('n_scen6', '6')])
    T('NYC Yellow Taxi data at three granularities (hourly', [Count('n_taxi', '3')])
    T('and MovieLens data at two (daily and weekly)', [Count('n_ml', '2')])
    T('We report two configurations', [Count('n_configs', '2')])
    T('the six conventional baselines use a 7-slot window and the three wavelet-based methods a 64-slot window',
      [Count('n_base', '6'), Count('n_wav', '3')])
    T('The two groups therefore do not see', [D('the two method groups of the same sentence')])
    T('all nine methods use the same 64-slot window', [Count('eq64_all', 'all 64'), Count('n_methods', '9')])
    T('active in every slot of the window, RRD equals', [D(DEF)])
    T('the moving-average reference in both tables', [D('Tables 8 and 9')])
    T('the three wavelet-based methods in stronger colors', [Count('n_wav', '3')])
    T('The two tables show one pattern', [D('Tables 8 and 9'), L(PLAIN)])
    T(r'It has the lowest $\Delta$Rank of all methods except PFRF in all six scenarios, and a higher RSI@10 than all six baselines',
      [Claim('dr_low6', 'lowest except PFRF'), Claim('dr_low6', 'all 6'), Count('n_scen6', '6'),
       Claim('rsi_gt_base6', 'all 6, significant except PFRF YouTube'), Count('n_base', '6')])
    T('RRD has a higher RSI@10 in all six scenarios', [Claim('rrd_rsi_eq6', 'all 6'), Count('n_scen6', '6')])
    T('RRD, VSE and CompoundPop in all six scenarios', [Claim('acc_gt_long_eq6', 'all 6'), Count('n_scen6', '6')])
    T('WSPI is therefore not the most stable method in every setting. It is more',
      [Claim('not_most_stable', 'not highest in some setting')] * 2)
    E.claim('main', 'AF has the highest NDCG@10 on YouTube and on the hourly and 30-minute taxi data', 'af_best_ndcg', 'YouTube, 1h, 30m')
    T(r'in 10\% or more of the windows of at least one scenario (largest share): PFRF (1.00), VSE (0.99)',
      [D('footnote rule of the table generator; the list is checked by check_paper_numbers.py')])
    T('equal window of 64 slots for all methods', [Count('eq64_all', 'all 64')])
    T(r'in 10\% or more of the windows of at least one scenario (largest share): PFRF (1.00), VSE (0.63)',
      [D('footnote rule of the table generator; the list is checked by check_paper_numbers.py')])
    T('every method except PFRF has a significantly higher NDCG@10 than WSPI', [Claim('taxi_ndcg_all', 'all 7 x 3')])
    T('which suits a one-slot horizon', [Count('horizon_one', '1')])
    T('NDCG@10 in the four YouTube and taxi scenarios', [Count('n_scen4', '4')])
    T('On daily data WSPI is significantly more accurate than all six baselines', [Claim('ml_daily_ndcg', 'all 6'), Count('n_base', '6')])
    E.claim('main', 'On MovieLens a wavelet-based method ranks best: DTCWT+AF on daily data', 'ml_ndcg_best', 'DTCWT+AF daily, DWT+AF weekly')
    T('Spearman correlation between each ranking', [L(DIST)])
    T(r'higher $\rho$ than all six baselines in every scenario',
      [Claim('tm_rho_all', 'all 6'), Count('n_base', '6'), Claim('tm_rho_all', 'all 5')])
    T(r'WSPI has the highest $\rho$ of all nine methods at both granularities',
      [Claim('ml_rho_best', 'highest of nine'), Count('n_methods', '9'), Claim('ml_rho_best', 'both')])
    T(r'All methods have a low $\rho$ on daily MovieLens data (at most', [D('the nine methods; the bound is checked by check_paper_numbers.py'), L(BOUND)])
    T('The two accuracy metrics thus disagree', [D('NDCG@10 and rho')])
    T('WSPI orders the whole list better', [L(PLAIN)])
    T(r'Spearman $\rho$ in the four YouTube and taxi scenarios', [Count('n_scen4', '4')])
    E.claim('main', 'WSPI has the highest value on YouTube (0.9456), on the 5-minute taxi data', 'rsi_best_wspi3', 'WSPI 3, DTCWT+AF ML weekly significant')
    T('All six baselines have a significantly lower RSI@10 than WSPI', [Claim('rsi_gt_base6', 'all 6, except PFRF YouTube'), Count('n_base', '6')])
    T('The most stable baseline apart from PFRF is RRD', [Claim('most_stable_base6', 'RRD 5, VSE ML weekly')])
    T('on the three taxi granularities', [Count('n_taxi', '3')])
    T('the two DTCWT-based methods have the two highest values in every scenario',
      [D('WSPI and DTCWT+AF'), Claim('tm_top2', 'top 2'), Claim('tm_top2', 'all 5')])
    T('shift-invariance is one source of this stability, but not the only one', [L(CONCL), L(CONCL)])
    T('RSI@10 in the four YouTube and taxi scenarios', [Count('n_scen4', '4')])
    T(r'WSPI has the lowest $\Delta$Rank of all methods except PFRF in all six scenarios: 37.12',
      [Claim('dr_low6', 'lowest except PFRF'), Claim('dr_low6', 'all 6'), Count('n_scen6', '6')])
    T('All these differences are significant. The next method', [Claim('dr_sig_all', 'all 7 x 6')])
    T(r'PFRF has the lowest $\Delta$Rank of all methods, but', [Claim('pfrf_lowest', 'all 6')])
    T('tied among the top 21 items in nearly every window', [Claim('pfrf_tied', 'nearly every window')])
    T('Two terms of the index react to a spike in opposite directions', [Claim('spike_opposite', 'opposite signs')])
    T('The test above uses one spike size, one position and one duration', [D('main-table test: 10x, last slot, one slot (Table 4)')] * 3)
    T('Here every method sees the same corrupted data', [D('design of the common perturbation (T3.5, R11)')])
    T('over five random seeds', [Count('seeds5', '5')])
    T('at every spike size in all four scenarios', [Claim('size_lowest', 'every size'), Claim('size_lowest', 'all 4'), Count('n_scen4', '4')])
    T('A six-slot burst', [Count('dur6', '6')])
    T('are computed over the whole window. This penalty', [D(DEF)])
    T('RRD was more robust than WSPI in every setting', [Claim('rrd_eq_more_robust', 'every condition x scenario')])
    T(r'under a $10\times$ spike in the four YouTube and taxi scenarios', [Count('n_scen4', '4')])
    T('for most of the period', [Claim('yt_fig_most', 'most')])
    T('lies above WSPI in about half of the plotted period', [Claim('yt_fig_half', 'about half')])
    T('lie close together at the top during the whole year', [Claim('taxi_fig_top', 'whole year')])
    T('as a one-week (168-hour) moving average', [D(FIG)])
    T('The two groups stay apart during the whole year', [D('the two groups named before'), Claim('taxi_fig_apart', 'whole year')])
    T('one-week moving average).}', [D(FIG)])
    T('was examined at three aggregation levels: hourly', [Count('n_taxi', '3')])
    T('and, at each level, the best of the six baselines except PFRF', [L(DIST), Count('n_base', '6')])
    T('WSPI has the highest RSI@10 of all nine methods', [Claim('taxi5_top', 'highest'), Count('n_methods', '9')])
    T(r'At every level, WSPI keeps the lowest $\Delta$Rank of all methods except PFRF',
      [Claim('taxi_dr_lowest', 'all 3'), Claim('taxi_dr_lowest', 'lowest except PFRF')])
    T('has a higher NDCG@10 than WSPI at every level', [Claim('taxi_best_ndcg', 'all 3')])
    T('The three levels differ in more than resolution', [Count('n_taxi', '3')])
    T('but only about five hours at the 5-minute level', [Count('five_hours', 'about 5')])
    T('the best of the six baselines except PFRF at each level', [Count('n_base', '6'), L(DIST)])
    # 4.7 cost
    T('For one window, every method reads the last', [L(PLAIN), L(PLAIN)])
    T('so the sum over all levels stays', [D(THEORY)])
    T('the six baselines can be updated in $O(1)$', [Count('n_base', '6')])
    T('AF and EWMA need a single value', [D('recursive form of AF and EWMA')])
    T('recompute the transform of the whole window at each slot', [D(DEF), L(DIST)])
    T('the last $N$ samples of each item', [L(DIST)])
    T('or 1~KiB in double precision', [L(PLAIN)])
    T('WSPI adds three sums and one exponential to the transform', [D('operations of the WSPI score (methods/wspi_assessment.py)')] * 2)
    T('AF and EWMA keep a single value', [D('recursive form of AF and EWMA')])
    T('the transform is recomputed at each slot. Measured', [L(DIST)])
    T('All methods ran in the same Python implementation', [D('implementation (code release)')])
    T('with one thread', [Count('thread1', '1')])
    T('one call per window, as in the evaluation', [D('batch scoring of the evaluation')])
    T('One million items took 12.5~s', [Count('million', '10^6')])
    T('It roughly doubled with each doubling of $N$ from 64 to 256', [Claim('time_doubling', 'about x2 each step')] * 3)
    T('A call for a single item cost', [D('single-item benchmark (Table 11)')])
    T('while scoring one million items in batches', [Count('million', '10^6')])
    T('scoring the whole catalogue of one window took', [L(PLAIN), L(PLAIN)])
    T('(at most 130~ms)', [L(BOUND)])
    T('Most of the evaluation run time came from the evaluation itself', [Claim('score_share_minor', 'score share < 1/2')])
    T('it can rescore the full catalogue at every slot', [L(PLAIN)])
    T('Measured cost on one CPU core', [Count('thread1', '1')])
    T('Single-item', [D('single-item benchmark (Table 11)')])
    T('median time to score the whole catalogue of one window on the real data', [L(PLAIN), L(PLAIN)])
    # 4.8 ablation, shift test, fusion
    T('To measure the contribution of each component, we evaluate six variants', [L(DIST), Struct('ablation_rows6', '6')])
    T('the trend with one structural term', [D(DEF)])
    T('All variants use $N=64$', [D('settings of the ablation runs (T3.3, R09)')])
    T('and each variant is tested against the full index', [L(DIST)])
    T('The results show three points', [Struct('points3', '3')])
    T('Each term alone removes part of this gap, and the full index is the most robust',
      [Claim('terms_each_part', 'each term'), Claim('terms_each_part', 'full index lowest')])
    T('The two terms do not change RSI@10 significantly', [Claim('terms_rsi', 'n.s. 3, higher 5m')])
    T('raise it slightly (by at most 0.0076). They lower NDCG@10 by at most 0.0042', [L(BOUND), L(BOUND)])
    T(r'$\Delta$Rank more than doubles', [Claim('dwt_dr_double', 'more than x2')])
    T('A similar gap appears between the two trend-only variants', [D('Trend with DTCWT and with DWT')])
    E.claim('main', 'A similar gap appears between the two trend-only variants', 'trend_gap_same', 'similar gap')
    T('In three of the four scenarios DWT gives a higher NDCG@10', [Claim('dwt_ndcg_3of4', '3 of 4'), Count('n_scen4', '4')])
    T('and not always present, cost in accuracy', [Claim('dwt_ndcg_3of4', 'not on 5m')])
    T('we shifted every real 64-slot window of every eligible item', [D('design of the shift test, part A (T3.6)')] * 2)
    T('Both transforms use a periodic boundary here', [D('setting of part A of the shift test (R12)')])
    T('so its energy does not change at all', [Claim('e1_zero', 'cv 0')])
    T('All these differences are significant in every scenario', [Claim('shift_sig_all', 'all')] * 2)
    T('towards the newest slot, one slot at a time, and scored it with the settings of each method',
      [D('step of the synthetic test (shifts 0..8)'), L(DIST)])
    T('DTCWT+AF never falls', [Claim('dtcwtaf_never', '0')])
    T('the equivalent EWMA always rises', [Claim('ewma_always', 'never falls, rises on average')])
    T('the WSPI score falls in about half of the steps with either transform', [Claim('mid_half', 'about half')])
    T('This is one source of the stability of WSPI, but not the only one', [L(CONCL), L(CONCL)])
    E.claim('main', 'A 64-slot moving average does not change', 'sma_flat', 'no change')
    T('we compared the exponential fusion with three alternatives', [Count('fusion_alt3', '3')])
    T('The RSI@10 of each alternative differs', [L(DIST)])
    T('the linear form is slightly more robust in every scenario', [Claim('linear_more_robust', 'all 4')])
    E.claim('main', 'the product form is more robust on the taxi data but less robust on YouTube', 'product_mixed', 'taxi lower, YouTube higher')
    # 4.9 sensitivity
    T(r'The index has two coefficients, $\alpha$ and $\beta$, and one level', [D(DEF), D(DEF)])
    T('We split the evaluation windows of each scenario in time', [L(DIST)])
    T(r'within 1\% of the best one', [L(PLAIN)])
    T('shows the one-dimensional slices', [L(PLAIN)])
    T(r'one-dimensional slices of the $7\times7$ grid, each coefficient varied', [L(PLAIN), L(DIST)])
    T('change little over the whole grid', [L(PLAIN)])
    T('their spread is at most 0.0104', [L(BOUND)])
    T('over all splits', [L(PLAIN)])
    T('without both terms', [D(DEF)])
    T('the accuracy--stability balance of each dataset', [L(DIST)])
    # MovieLens paragraph of 4.2 and the MovieLens figure (T4.12)
    T('Each rating counts as one interaction with a movie', [D(DEF), D(DEF)])
    T('The data are used at two granularities', [Count('n_ml', '2')])
    T('because each window ranks about', [L(DIST)])
    T('Two properties of this dataset limit what it can show', [Struct('props2_ml', '2')])
    # 4.11 responsiveness and failure cases
    T('how fast each method follows genuine entries into the true Top-10. An entry', [L(DIST)])
    T('stays there for at least six consecutive slots, and was outside it in the six slots before',
      [Count('L6', '6'), Count('P6', '6')])
    T('The four scenarios contain', [Count('n_scen4', '4')])
    T('On the hourly and 30-minute taxi data most of them follow the daily cycle', [Claim('daily_cycle_most', 'most, hourly and 30-minute')])
    T('outside the Top-20 in the six slots before the entry', [Count('P6', '6')])
    T('Each cell gives the miss rate', [L(DIST)])
    T('The window length $N$ of each method', [L(DIST)])
    T('more than half of the entries were missed', [Claim('median_half_rule', 'rule holds')])
    T('AF misses none', [Claim('af_none', '0')])
    T('all these differences are significant. The long-window', [Claim('delay_sig', 'all 8')])
    T('On the three taxi scenarios WSPI is significantly faster than all three',
      [Count('n_taxi', '3'), Claim('long_slower', 'all 3 x 3'), D('RRD, VSE and CompoundPop of the same paragraph')])
    T('against one slot for AF', [Claim('af_one_slot', '1')])
    T('Each example has two panels', [D(FIG), D(FIG)])
    T('during its whole stay', [D(DEF)])
    T('In the other three scenarios the share is zero for all three methods',
      [D('YouTube, taxi 1h and 30m'), Claim('inelig_zero', 'zero, all'), Count('n_wav', '3')])
    T('most likely comes from', [L('hedged interpretation')])
    T(r'WSPI had the largest $\Delta$Rank of all nine methods in all four scenarios',
      [Claim('random_largest', 'largest'), Count('n_methods', '9'), Claim('random_largest', 'all 4'), Count('n_scen4', '4')])
    # discussion, Table 15, conclusion
    T('Across the three datasets and the six evaluation scenarios, the results show one pattern',
      [Count('n_datasets', '3'), Count('n_scen6', '6'), L(PLAIN)])
    T('not the most stable method in every setting. Table', [Claim('not_most_stable', 'not highest in some setting')] * 2)
    T('One source of this behavior is the transform', [L(PLAIN)])
    T('moving the data by one or a few slots', [L(PLAIN)])
    T('from one window to the next', [L(PLAIN)])
    T('causes the largest drop in both stability metrics, at a small gain in NDCG@10 in three of the four scenarios',
      [Claim('dwt_largest_drop', 'both'), Claim('dwt_ndcg_3of4', '3 of 4'), Count('n_scen4', '4')])
    T('the two terms mainly add robustness', [D('R and W_E')])
    T('are computed over the whole window and carry', [D(DEF)])
    T('each change of the Top-$K$ set has a cost and one-off spikes', [L(DIST), L(PLAIN)])
    T('must be caught within one or two slots', [Claim('af_delay_12', 'median 1 or 2')] * 2)
    T('runs in a single pass', [D(DEF)])
    T('On one CPU core it rescored', [Count('thread1', '1')])
    T('the whole catalogue of a window', [L(PLAIN)])
    T('so it can update at every slot', [L(PLAIN)])
    T('It also held up in three different settings', [Count('n_datasets', '3')])
    T('recomputes the transform at each slot, about 105 times', [L(DIST)])
    T('What we have shown holds for these three public datasets', [Count('n_datasets', '3')])
    T('Short-window baselines: the six baselines with their default 7-slot window', [Count('n_base', '6')])
    T('The six scenarios are YouTube, NYC Yellow Taxi at three granularities and MovieLens at two',
      [Count('n_scen6', '6'), Count('n_taxi', '3'), Count('n_ml', '2')])
    T('All differences stated are significant', [D('the row claims below are checked with their block verdicts')])
    T(r'Lower than at least one baseline in five of the six scenarios & $+$ Higher in all six scenarios',
      [Claim('s_ndcg_short', 'at least one'), Claim('s_ndcg_short', '5 of 6'), Count('n_scen6', '6'),
       Claim('s_ndcg_long', 'all 6'), Count('n_scen6', '6')])
    T(r'lower than AF on YouTube & $+$ Higher in all six scenarios', [Claim('s_rho_long', 'all 6'), Count('n_scen6', '6')])
    E.claim('main', r'Higher on taxi and MovieLens; $-$ lower than AF on YouTube', 's_rho_short', 'taxi and ML higher, YouTube lower than AF')
    T(r'Higher in all six scenarios (not significant against PFRF on YouTube) & $-$ Lower than RRD in all six scenarios',
      [Claim('rsi_gt_base6', 'all 6, except PFRF YouTube n.s.'), Count('n_scen6', '6'), Claim('s_rsi_long', 'all 6'), Count('n_scen6', '6')])
    T('Higher than RRD in four of the six scenarios', [Claim('s_dr_long', '4 of 6'), Count('n_scen6', '6')])
    T('the transform is recomputed at each slot ($O(N)$)', [L(DIST)])
    T('studied three wavelet-based methods that need no training', [Count('n_wav', '3')])
    T('extracts three structural features: trend-weighted volume', [Struct('features3', '3')])
    T('YouTube, NYC Yellow Taxi at three granularities and MovieLens at two, over 131,443 windows, with block-bootstrap confidence intervals and paired tests',
      [Count('n_taxi', '3'), Count('n_ml', '2')])
    T('and each result depends on the configuration', [L(PLAIN)])
    T('one of the three wavelet-based methods had the highest RSI@10 in all six scenarios (WSPI in three, DTCWT+AF in three)',
      [Claim('rsi_top_wav6', 'one wavelet-based method highest'), Count('n_wav', '3'), Claim('rsi_top_wav6', 'all 6'),
       Count('n_scen6', '6'), Claim('rsi_top_counts', 'WSPI 3'), Claim('rsi_top_counts', 'DTCWT+AF 3')])
    E.claim('main', 'replacing DTCWT with DWT caused the largest loss of stability', 'dwt_largest_drop', 'largest')
    T('on one CPU core it scored one million items', [Count('thread1', '1'), Count('million', '10^6')])
    T('recomputes the transform at each slot. The robustness', [L(DIST)])
    T('The evaluation covers three public datasets', [Count('n_datasets', '3')])
    T('Four directions follow from these limits', [Struct('directions4', '4')])
    T('beyond the three used here', [Struct('features3', '3')])
    T('All authors reviewed and approved', [L(PLAIN)])
    T('the four-aspect evaluation protocol, and all baselines used in this study', [Struct('aspects4', '4'), D('code release')])
    T('All experiments used the settings in Tables', [D('reproducibility statement')])
    E.claim('main', 'against six baseline methods: AF, EWMA, RRD, VSE, CompoundPop, and PFRF', 'base6_list', '6 named')


def rules_si(E):
    T = lambda anchor, specs: E.text('si', anchor, specs)
    T('All values are means over the common evaluation windows', [D(DEF)])
    T('The nine methods, their window lengths', [Count('n_methods', '9')])
    T('compares the methods in two configurations', [Count('n_configs', '2')])
    T('mean over the windows common to all methods and all $N$', [D(DEF), D(DEF)])
    T('Two boundary operations act on the wavelet coefficients', [D('padding and boundary extension (Stage 1)')])
    T('The boundary extension of the transform acts in every window', [D('boundary extension inside the dtcwt library')])
    T('compares five extension modes for the three wavelet-based methods', [Count('n_ext5', '5'), Count('n_wav', '3')])
    T('over all item-windows, and Supplementary', [D(DEF)])
    T('over all item-windows scored by WSPI', [D(DEF)])
    T('solid, all detail energy in one band; dashed, detail energy spread evenly over the three detail bands',
      [D(DEF), D(DEF), Count('level3', '3')])
    T('a collection gap of ten hours on 19 May 2018', [Count('gap10', '10')])
    T('The gap is never used as ground truth', [Claim('gap_never_truth', '0 test slots')])
    T('injects one spike into the last slot', [D('main-table test (Table 4)')])
    T('shows how the two terms of WSPI respond', [D('log mu_L and alpha R - beta W_E')])
    T('(spike in the last slot, one slot)', [D('main-table test (Table 4)')])
    T('Every method sees the same items and the same spike', [D('design of the common perturbation (T3.5, R11)')])
    T('mean over the common windows and five seeds', [Count('seeds5', '5')])
    T('reports the measured cost of the nine methods', [Count('n_methods', '9')])
    T('Measured cost of the nine methods on one CPU core', [Count('n_methods', '9'), Count('thread1', '1')])
    T('Time to score all items of one window', [L(PLAIN), L(PLAIN)])
    T('default window of each method', [L(DIST)])
    T('shows six variants of the index', [Struct('ablation_rows6', '6')])
    T('gives all eight variants in all four scenarios', [D(PLAIN), Count('ablation8', '8'), D(PLAIN), Count('n_scen4', '4')])
    T('compares the exponential fusion with three alternatives', [Count('fusion_alt3', '3')])
    T('shows its two parts', [D('parts A and B of the shift test')])
    T('every 64-slot window of every eligible item is shifted', [D('design of the shift test, part A (T3.6)')] * 2)
    T('Synthetic three-slot burst', [Count('burst3', '3')])
    T('moved one slot at a time towards the newest slot', [D('step of the synthetic test (shifts 0..8)')])
    T('each method with its own settings', [L(DIST)])
    T('Share of the eight steps', [Count('steps8', '8')])
    T('columns: the four scenarios', [Count('n_scen4', '4')])
    T('Darker cells are better within each panel; the panel title gives the range, and each cell gives its mean', [L(DIST), L(DIST)])
    T('The values of the one-dimensional slices', [L(PLAIN)])
    T('behind Tables~8 and~9 of the main text, for all six scenarios', [Count('n_scen6', 'all 6')] * 2)
    T(r'WSPI had a higher NDCG@10 and $\rho$ than all of them', [Claim('si_ml_eq_acc', 'all 3 x 2 x 2')])
    E.claim('si', 'had a higher RSI@10 than WSPI on MovieLens', 'si_ml_eq_rsi', 'SMA, RRD, VSE higher')
    T('how fast each method follows genuine entries', [L(DIST)])
    T('relates the rank changes of each method', [L(DIST)])
    T('appears in the Top-10 of each method (an entry stays at least six slots in the true Top-10 after at least six slots outside it)',
      [L(DIST), Count('L6', '6'), Count('P6', '6')])
    T('the level at the right end equals one minus the miss rate', [L(PLAIN)])


# ------------------------------------------------------------------ main
def main():
    global E
    ap = argparse.ArgumentParser(description='Check the numbers written in words in the V5 paper and SI (T4.11b)')
    ap.add_argument('--main', required=True, type=Path, help='WSPI_ScientificReports.tex (a built copy)')
    ap.add_argument('--si', required=True, type=Path, help='WSPI_SI.tex of the same copy')
    ap.add_argument('--results', required=True, type=Path, help='results/revision_v5')
    ap.add_argument('--out', required=True, type=Path, help='CSV report (one row per word and per extra claim)')
    a = ap.parse_args()
    C.RESULTS = a.results

    E = Engine({'main': a.main, 'si': a.si})
    rules_main(E)
    rules_si(E)

    rows = [dict(kind='word', file=t.file, line=t.line, col=t.col, word=t.word, category=t.category or '-', status=t.status,
                 expected=t.expected, found=t.found, check=t.check, source=t.source, rule=t.rule, context=t.ctx.strip())
            for t in E.toks]
    rows += E.extra
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    words = [r for r in rows if r['kind'] == 'word']
    c = Counter(r['status'] for r in words)
    print(f'words: {len(words)}  ' + '  '.join(f'{k}={v}' for k, v in sorted(c.items())))
    ce = Counter(r['status'] for r in E.extra)
    print(f'extra claims: {len(E.extra)}  ' + '  '.join(f'{k}={v}' for k, v in sorted(ce.items())))
    used = sorted({r['check'] for r in rows if r['check']})
    print(f'checks used: {len(used)} of {len(CHECKS)} defined')
    for p in E.problems:
        print('RULE PROBLEM', p)
    bad = [r for r in rows if r['status'] in ('mismatch', 'error', 'unchecked')]
    for r in bad:
        print(f"{r['status'].upper():9s} [{r['file']}:{r['line']}] {r['word'] or '(claim)':>8s} | {r['check'] or r['context'][:60]} | {r['found']}")
    n_bad = len(bad) + len(E.problems)
    print('ALL WORD NUMBERS AGREE WITH THE RESULT FILES' if n_bad == 0 else f'{n_bad} item(s) to review')
    return 0 if n_bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
