"""Test-part tables of the popularity-prediction study (run C and the later runs).

Reads only result files and writes CSV/JSON to <root>/decision_test/.
No new statistics: every verdict is the ``verdict`` column of
stats_test/all_paired_tests.csv; every mean is a column of the same file.

Tables
  gain_claims_test.csv     rule 2 on the test part: each reference method
                           against the best traditional baseline of each cell
                           and main metric; two baseline sets (A: Persistence,
                           Holt, ARYW, MA64; A+X2: plus SMA8-32, EWMA8-64)
  gain_summary_test.csv    counts of gain_claims_test.csv
  pareto_test.csv          Pareto rule on the test part for pairs of
                           wavelet methods (reference better in >= 1 main
                           metric and worse in none)
  verdict_counts_test.csv  verdict counts over the 18 cells for every
                           reference x method x main metric
  pairs_test.csv           cell rows of the comparisons named in PAIRS
  means_test.csv           test-part mean of every method, cell and main metric
  runtime_test.csv         scorer time per method and cell; GBM training time
  runtime_ratio.csv        time ratios inside one run, WSPI as the unit
  sensitivity_range.csv    range of the 18 WSPI-F2 configurations per cell
  nlms_late.csv            summary of the late positions, both scenarios
  analysis_test.json       inputs (sha256) and headline counts

Exploratory test (--explore)
  Reads stats_explore/all_paired_tests.csv (the test-part reports with the
  extra reference WSPI-F-YW, tools/run_prediction_stats.py --references) and
  decision_test/gain_claims_test.csv (best baseline of each cell and metric).
  Writes only into <root>/decision_explore/ (the tables above are not
  written):
  verdict_counts_explore.csv   verdict counts over the 18 cells, WSPI-F-YW
                               against every method of the test report
  pairs_explore.csv            the main-metric rows of those reports
  pareto_explore.csv           Pareto rule, WSPI-F-YW against every method
  best_baseline_explore.csv    WSPI-F-YW against the best baseline of
                               decision_test (set A+X2); description only,
                               outside the gain rule (chosen after seeing data)
  analysis_explore.json        inputs (sha256) and counts

Tables of the paper on the test part (--paper)
  Reads stats_test/all_paired_tests.csv, stats_explore/all_paired_tests.csv,
  stats_test_B/all_paired_tests.csv and decision/domain_effect.csv (time ratio and
  fallback share of run B, which do not depend on the part).  Writes only into
  <root>/decision_paper/ (decision_test/ and decision_explore/ are not written).
  "X dominates Y" in a cell: X better in >= 1 main metric and worse in none
  (column verdict).  Every pair is read from the report whose reference is X.
  dominance_cells.csv          every method with a test-part reference report
                               (WSPI, WSPI-F2, WSPI-FS3, GBM, GBM+W; WSPI-F-YW from
                               the exploratory reports) against each of the 11
                               baselines A+X2, per cell: the four verdicts and the
                               relation (dominated, dominates, trade-off, equal)
  dominance_by_baseline.csv    counts of the relations over the 18 cells
  dominance_summary.csv        D = cells where >= 1 baseline dominates the method,
                               per horizon and in all; and the cells where the
                               method dominates >= 1 baseline
  positive_facts.csv           the comparisons with named baselines of the design
                               review (WSPI against EWMA8, SMA8, EWMA16 and the other
                               moving averages; GBM+W against GBM): verdict counts
                               per main metric and relation counts
  pareto_gates_test.csv        Pareto rule of the gates on the test part: WSPI-FG
                               against WSPI-F2 and WSPI-FS3 against WSPI-F2, per cell
                               (read from the WSPI-F2 report with the verdict turned
                               around), with the rule of the registered decision
                               (>= 4 of 6 scenarios in >= 2 of 3 horizons)
  domain_effect_test.csv       run B on the test part: WSPI-FA against WSPI and
                               ARIMA per cell and main metric, with time ratio and
                               fallback share
  analysis_paper.json          inputs (sha256) and the headline counts

Reading rule of run D (--run-d tune, then --run-d test)
  Applies the fixed reading rule of the design review (section 9) to the two
  window-mean level methods of run D: WSPI-M (control) and WSPI-H (candidate).
  The thresholds are constants of this file and were fixed before the run.
  Writes only into <root>/decision_D/.  ``--run-d test`` refuses to run until
  ``--run-d tune`` has written screen_tune.json (the tuning part is read first).
  --run-d tune   reads stats_tune_D/ only
    tune_dominance_cells.csv, tune_dominance_by_baseline.csv,
    tune_dominance_summary.csv   as in --paper, for WSPI-H, WSPI-M, WSPI-F2, WSPI
                                 and WSPI-F-YW on the tuning part
    tune_pairs.csv               relation per cell of the named pairs
    tune_pair_counts.csv         verdict counts per main metric of those pairs
    tune_means.csv               tuning-part mean of every method (WSPI-H report)
    screen_tune.json             conditions T1-T3 and the result of the screen
  --run-d test   reads stats_test_D/, decision_paper/dominance_summary.csv (the
                 fixed test-part D of WSPI, WSPI-F2 and WSPI-F-YW) and
                 decision_test/gain_claims_test.csv (best baseline of each cell)
    test_dominance_cells.csv, test_dominance_by_baseline.csv,
    test_dominance_summary.csv   WSPI-H and WSPI-M on the test part
    test_pairs.csv, test_pair_counts.csv, test_means.csv   as above
    test_gain_claims.csv         rule 2 for WSPI-H against the best baseline of
                                 each cell and main metric (sets A and A+X2)
    inference_D.json             T1-T3 on the test part, the final state
                                 (accepted, inconsistent, rejected), the mechanism
                                 question (WSPI-F-YW against WSPI-M) and the cost
                                 of the clamp (WSPI-H against WSPI-M)
  WSPI-F2 against WSPI-H is read from the WSPI-H report with the verdict turned
  around, on both parts, because the test part has no WSPI-F2 report with WSPI-H
  in its family; the count from the WSPI-F2 report is added on the tuning part.
  WSPI-F-YW against WSPI-M is read from the WSPI-M report in the same way.

Reading rule of the confirmation data (--confirm tune, then --confirm test)
  The reading rule below was fixed before the held-out data were downloaded: the 12 cells of taxi2024_hourly,
  taxi2024_30min, taxi2024_5min and wikipedia_daily, thresholds T1 D(WSPI-H) <= 3,
  T2 D(WSPI-H) < D(WSPI-F2), T3 WSPI-F2 dominates WSPI-H in at most 2 cells.
  Reads <root>/confirm/stats_<part>/ (references WSPI-H, WSPI-F2, WSPI-M, WSPI)
  and writes only into <root>/confirm/decision/.  Every cell must be complete.
  --confirm tune   <part>_dominance_*.csv, <part>_pairs.csv, <part>_pair_counts.csv,
                   <part>_means.csv as above; screen_tune.json (T1-T3 on the tuning
                   part; reported, not part of the verdict)
  --confirm test   needs screen_tune.json; the same tables for the test part,
                   test_gain_claims.csv (rule 2 for WSPI-H; best baseline from the
                   test means of the WSPI report) and confirmation.json (T1-T3 on
                   the test part and the final state: confirmed or not confirmed;
                   gains; descriptive counts of the named pairs)
  WSPI-F2 against WSPI-H is read from the WSPI-H report with the verdict turned
  around; the count from the WSPI-F2 report is written next to it.

Usage
  python tools/prediction_test_analysis.py --root results/prediction [--out DIR]
  python tools/prediction_test_analysis.py --root results/prediction --explore [--out DIR]
  python tools/prediction_test_analysis.py --root results/prediction --paper [--out DIR]
  python tools/prediction_test_analysis.py --root results/prediction --run-d tune [--out DIR]
  python tools/prediction_test_analysis.py --root results/prediction --run-d test [--out DIR]
  python tools/prediction_test_analysis.py --root results/prediction --confirm tune [--out DIR]
  python tools/prediction_test_analysis.py --root results/prediction --confirm test [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SCENARIOS = ('youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min',
             'movielens_daily', 'movielens_weekly')
HORIZONS = (1, 8, 24)
MAIN = ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion')
LOWER = ('robustness_distortion',)
BASE_A = ('Persistence', 'Holt', 'ARYW', 'MA64')
BASE_X2 = BASE_A + ('SMA8', 'SMA16', 'SMA32', 'EWMA8', 'EWMA16', 'EWMA32', 'EWMA64')
REFS = ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W')
PARETO = (('WSPI-FS3', 'WSPI-F2'), ('WSPI-FS3', 'WSPI'), ('WSPI-F2', 'WSPI'),
          ('WSPI-F2', 'WSPI-FG'), ('WSPI', 'WSPI-FG'), ('WSPI-F2', 'WSPI-S'),
          ('WSPI', 'WSPI-S'), ('WSPI-F2', 'WSPI-F-YW'), ('GBM', 'GBM+W'))
PAIRS = PARETO + tuple((r, g) for r in ('WSPI', 'WSPI-F2', 'WSPI-FS3') for g in ('GBM', 'GBM+W'))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def means_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Test mean of every method from the WSPI reference report, as in
    tools/prediction_decision.py (rule 2 picks the best baseline there).
    A window drops out of a mean only where the metric is undefined
    (Spearman rho of a constant score vector)."""
    s = summary[(summary.part == 'test') & (summary.reference_report == 'WSPI') & summary.main_metric]
    s = s[['scenario', 'horizon', 'metric', 'method', 'n_windows', 'mean', 'ci_low', 'ci_high']]
    return s.sort_values(['scenario', 'horizon', 'metric', 'method']).reset_index(drop=True)


def row(t, s, h, ref, meth, met):
    d = t[(t.scenario == s) & (t.horizon == h) & (t.reference == ref)
          & (t.method == meth) & (t.metric == met)]
    if len(d) != 1:
        return None
    return d.iloc[0]


def gain_claims(t, means):
    out = []
    for bset, names in (('A', BASE_A), ('A+X2', BASE_X2)):
        for s in SCENARIOS:
            for h in HORIZONS:
                for met in MAIN:
                    mm = means[(means.scenario == s) & (means.horizon == h) & (means.metric == met)
                               & means.method.isin(names)].set_index('method')['mean']
                    best_v = mm.min() if met in LOWER else mm.max()
                    best = [n for n in names if n in mm.index and mm[n] == best_v][0]
                    for ref in REFS:
                        r = row(t, s, h, ref, best, met)
                        out.append(dict(baseline_set=bset, scenario=s, horizon=h, metric=met,
                                        method=ref, best_baseline=best,
                                        mean_method=r.mean_reference, mean_baseline=r.mean_method,
                                        diff_method_minus_baseline=r.diff_ref_minus_method,
                                        diff_ci_low=r.diff_ci_low, diff_ci_high=r.diff_ci_high,
                                        cliffs_delta=r.cliffs_delta_favours_ref,
                                        p_block_holm=r.p_block_holm, verdict=r.verdict,
                                        gain=bool(r.verdict == 'ref_better')))
    return pd.DataFrame(out)


def pareto(t):
    out = []
    for ref, meth in PARETO:
        for s in SCENARIOS:
            for h in HORIZONS:
                v = {}
                for met in MAIN:
                    r = row(t, s, h, ref, meth, met)
                    v[met] = None if r is None else r.verdict
                if None in v.values():
                    continue
                vals = list(v.values())
                out.append(dict(reference=ref, method=meth, scenario=s, horizon=h,
                                **{'verdict_' + k: x for k, x in v.items()},
                                n_better=vals.count('ref_better'), n_worse=vals.count('ref_worse'),
                                success=vals.count('ref_better') >= 1 and vals.count('ref_worse') == 0))
    return pd.DataFrame(out)


def runtime(root: Path):
    out = []
    for s in SCENARIOS:
        for h in HORIZONS:
            c = pd.read_csv(root / 'run_C' / s / f'h{h}' / 'scorer_timing.csv')
            a = pd.read_csv(root / 'run_A' / s / f'h{h}' / 'scorer_timing.csv')
            for run, d in (('C', c), ('A', a)):
                for _, r in d.iterrows():
                    out.append(dict(scenario=s, horizon=h, run=run, method=r.method,
                                    rows=r.rows, scorer_seconds=r.scorer_seconds,
                                    ms_per_row=r.ms_per_row, train_seconds=None))
            for meth, fn in (('GBM', 'learned_GBM.json'), ('GBM+W', 'learned_GBM_W.json')):
                j = json.loads((root / 'run_C' / s / f'h{h}' / 'metadata' / fn).read_text())
                for o in out:
                    if o['scenario'] == s and o['horizon'] == h and o['run'] == 'C' and o['method'] == meth:
                        o['train_seconds'] = j['rows_seconds'] + j['fit_seconds']
                        o['train_rows'] = j['n_rows_used']
    return pd.DataFrame(out)


def runtime_ratio(rt: pd.DataFrame) -> pd.DataFrame:
    """Ratios inside one run only (same machine load), WSPI as the common unit.
    WSPI-F2 is not in run C, so GBM / WSPI-F2 = (GBM / WSPI in C) / (WSPI-F2 / WSPI in A)."""
    out = []
    for (s, h), d in rt.groupby(['scenario', 'horizon'], sort=False):
        c = d[d.run == 'C'].set_index('method')
        a = d[d.run == 'A'].set_index('method')
        w_c, w_a = c.loc['WSPI', 'scorer_seconds'], a.loc['WSPI', 'scorer_seconds']
        f2_rel = a.loc['WSPI-F2', 'scorer_seconds'] / w_a
        row = dict(scenario=s, horizon=h, rows=int(c.loc['WSPI', 'rows']),
                   wspi_f2_over_wspi_runA=f2_rel,
                   wspi_fs3_over_wspi_runC=c.loc['WSPI-FS3', 'scorer_seconds'] / w_c)
        for g in ('GBM', 'GBM+W'):
            tot = c.loc[g, 'scorer_seconds'] + c.loc[g, 'train_seconds']
            row[f'{g}_train_seconds'] = c.loc[g, 'train_seconds']
            row[f'{g}_score_seconds'] = c.loc[g, 'scorer_seconds']
            row[f'{g}_total_over_wspi_runC'] = tot / w_c
            row[f'{g}_total_over_wspi_f2'] = tot / w_c / f2_rel
        out.append(row)
    return pd.DataFrame(out)


def sens_range(root: Path):
    s = pd.read_csv(root / 'stats_test' / 'sensitivity_summary.csv')
    out = []
    for (sc, h), d in s.groupby(['scenario', 'horizon'], sort=False):
        dflt = d[d['default']].iloc[0]
        for met in MAIN:
            col = met + '_mean'
            lo, hi = d[col].min(), d[col].max()
            pick = d[col].idxmin() if met in LOWER else d[col].idxmax()
            out.append(dict(scenario=sc, horizon=h, metric=met, n_configs=len(d),
                            default=dflt[col], min=lo, max=hi,
                            range_rel_to_default=(hi - lo) / abs(dflt[col]),
                            best_config=d.loc[pick, 'method'], best_value=d.loc[pick, col]))
    return pd.DataFrame(out)


def nlms(root: Path):
    out = []
    for s in ('youtube_hourly', 'taxi_hourly'):
        d = pd.read_csv(root / 'run_C_nlms' / s / 'h1' / 'nlms_convergence_late.csv')
        d.insert(0, 'scenario', s)
        out.append(d)
    return pd.concat(out)


EXPLORE_REF = 'WSPI-F-YW'


def explore(root: Path, out: Path):
    """Tables of the exploratory test-part reports with reference WSPI-F-YW."""
    src = root / 'stats_explore' / 'all_paired_tests.csv'
    gsrc = root / 'decision_test' / 'gain_claims_test.csv'
    t = pd.read_csv(src, float_precision='round_trip')
    t = t[(t.part == 'test') & (t.reference == EXPLORE_REF)]
    tm = t[t.main_metric]
    vc = (tm.groupby(['reference', 'method', 'metric'])['verdict'].value_counts()
          .unstack(fill_value=0).reindex(columns=['ref_better', 'n.s.', 'ref_worse'], fill_value=0)
          .reset_index())
    vc.to_csv(out / 'verdict_counts_explore.csv', index=False)
    tm.to_csv(out / 'pairs_explore.csv', index=False)
    prow = []
    for meth in sorted(tm.method.unique()):
        for s in SCENARIOS:
            for h in HORIZONS:
                v = {}
                for met in MAIN:
                    r = row(t, s, h, EXPLORE_REF, meth, met)
                    v[met] = None if r is None else r.verdict
                if None in v.values():
                    continue
                vals = list(v.values())
                prow.append(dict(reference=EXPLORE_REF, method=meth, scenario=s, horizon=h,
                                 **{'verdict_' + k: x for k, x in v.items()},
                                 n_better=vals.count('ref_better'),
                                 n_worse=vals.count('ref_worse'),
                                 success=vals.count('ref_better') >= 1
                                 and vals.count('ref_worse') == 0))
    p = pd.DataFrame(prow)
    p.to_csv(out / 'pareto_explore.csv', index=False)
    g = pd.read_csv(gsrc)
    best = (g[g.baseline_set == 'A+X2'][['scenario', 'horizon', 'metric', 'best_baseline']]
            .drop_duplicates())
    brow = []
    for _, b in best.iterrows():
        r = row(t, b.scenario, b.horizon, EXPLORE_REF, b.best_baseline, b.metric)
        brow.append(dict(baseline_set='A+X2', scenario=b.scenario, horizon=int(b.horizon),
                         metric=b.metric, method=EXPLORE_REF, best_baseline=b.best_baseline,
                         mean_method=r.mean_reference, mean_baseline=r.mean_method,
                         diff_method_minus_baseline=r.diff_ref_minus_method,
                         diff_ci_low=r.diff_ci_low, diff_ci_high=r.diff_ci_high,
                         cliffs_delta=r.cliffs_delta_favours_ref, p_block_holm=r.p_block_holm,
                         verdict=r.verdict))
    bb = pd.DataFrame(brow)
    bb.to_csv(out / 'best_baseline_explore.csv', index=False)
    head = dict(
        note='exploratory: WSPI-F-YW as reference was chosen after seeing the test-part '
             'means; outside the gain rule',
        inputs={src.relative_to(root).as_posix(): sha(src),
                gsrc.relative_to(root).as_posix(): sha(gsrc)},
        reference=EXPLORE_REF, n_cells=int(t.groupby(['scenario', 'horizon']).ngroups),
        n_methods=int(tm.method.nunique()),
        vs_MA64={met: {k: int(v) for k, v in vc[(vc.method == 'MA64') & (vc.metric == met)]
                       [['ref_better', 'n.s.', 'ref_worse']].iloc[0].items()} for met in MAIN},
        pareto_success={m: int(p[p.method == m].success.sum()) for m in sorted(p.method.unique())},
        better_than_best_baseline_A_X2={met: int((bb[bb.metric == met].verdict == 'ref_better').sum())
                                        for met in MAIN},
    )
    (out / 'analysis_explore.json').write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2))
    print('Saved to', out)


# ---- tables of the paper on the test part (--paper) ---------------------------

DOM_METHODS = ('WSPI', 'WSPI-F2', 'WSPI-FS3', 'GBM', 'GBM+W')     # stats_test references
DOM_EXPLORE = ('WSPI-F-YW',)                                       # stats_explore reference
POSITIVE = (('WSPI', ('EWMA8', 'SMA8', 'EWMA16', 'EWMA32', 'SMA16', 'SMA32', 'EWMA64',
                      'MA64', 'ARYW')),
            ('GBM+W', ('GBM',)))
GATE_PAIRS = (('WSPI-FG', 'WSPI-F2'), ('WSPI-FS3', 'WSPI-F2'))       # (method, reference report)
FLIP = {'ref_better': 'ref_worse', 'ref_worse': 'ref_better', 'n.s.': 'n.s.'}


def relation(verdicts):
    """Relation of the reference X to the method Y from the four verdicts of the X report."""
    nb, nw = verdicts.count('ref_better'), verdicts.count('ref_worse')
    if nb and not nw:
        return 'dominates'
    if nw and not nb:
        return 'dominated'
    return 'trade-off' if nb else 'equal'


def cell_verdicts(t, s, h, ref, meth):
    v = {}
    for met in MAIN:
        r = row(t, s, h, ref, meth, met)
        if r is None:
            return None
        v[met] = r.verdict
    return v


def dominance(t, methods, baselines=BASE_X2, source='stats_test', scenarios=SCENARIOS):
    out = []
    for ref in methods:
        for b in baselines:
            for s in scenarios:
                for h in HORIZONS:
                    v = cell_verdicts(t, s, h, ref, b)
                    if v is None:
                        continue
                    vals = list(v.values())
                    out.append(dict(source=source, method=ref, baseline=b, scenario=s, horizon=h,
                                    **{'verdict_' + k: x for k, x in v.items()},
                                    n_better=vals.count('ref_better'),
                                    n_worse=vals.count('ref_worse'), relation=relation(vals)))
    return pd.DataFrame(out)


def dominance_tables(cells):
    rel = ('dominated', 'dominates', 'trade-off', 'equal')
    by_b = (cells.groupby(['source', 'method', 'baseline'], sort=False)['relation']
            .value_counts().unstack(fill_value=0).reindex(columns=rel, fill_value=0).reset_index())
    by_b['n_cells'] = by_b[list(rel)].sum(axis=1)
    rows = []
    for (src, m), d in cells.groupby(['source', 'method'], sort=False):
        dom = d[d.relation == 'dominated'].groupby(['scenario', 'horizon']).baseline.apply(sorted)
        win = d[d.relation == 'dominates'].groupby(['scenario', 'horizon']).size()
        n_cells = d.groupby(['scenario', 'horizon']).ngroups
        r = dict(source=src, method=m, n_cells=n_cells, D=len(dom), not_dominated=n_cells - len(dom),
                 dominates_any=len(win))
        for h in HORIZONS:
            r[f'D_h{h}'] = int(sum(1 for (s, hh) in dom.index if hh == h))
        r['dominated_cells'] = '; '.join(f'{s}/h{h}: {",".join(b)}' for (s, h), b in dom.items())
        rows.append(r)
    return by_b, pd.DataFrame(rows)


def positive_facts(t):
    out = []
    for ref, names in POSITIVE:
        for b in names:
            d = t[(t.reference == ref) & (t.method == b) & t.metric.isin(MAIN)]
            r = dict(method=ref, baseline=b)
            for met in MAIN:
                v = d[d.metric == met].verdict
                r[f'{met}_better'] = int((v == 'ref_better').sum())
                r[f'{met}_ns'] = int((v == 'n.s.').sum())
                r[f'{met}_worse'] = int((v == 'ref_worse').sum())
            rels = [relation(list(cv.values())) for s in SCENARIOS for h in HORIZONS
                    for cv in [cell_verdicts(t, s, h, ref, b)] if cv is not None]
            r.update(n_cells=len(rels), dominates=rels.count('dominates'),
                     dominated=rels.count('dominated'), trade_off=rels.count('trade-off'),
                     equal=rels.count('equal'))
            out.append(r)
    return pd.DataFrame(out)


def pareto_gates(t):
    out = []
    for meth, ref in GATE_PAIRS:
        for s in SCENARIOS:
            for h in HORIZONS:
                v = cell_verdicts(t, s, h, ref, meth)
                if v is None:
                    continue
                v = {k: FLIP[x] for k, x in v.items()}          # from the method's side
                vals = list(v.values())
                out.append(dict(method=meth, against=ref, scenario=s, horizon=h,
                                **{'verdict_' + k: x for k, x in v.items()},
                                n_better=vals.count('ref_better'), n_worse=vals.count('ref_worse'),
                                success=vals.count('ref_better') >= 1 and vals.count('ref_worse') == 0,
                                source_report=f'reference {ref}, verdict turned around'))
    p = pd.DataFrame(out)
    rule = {}
    for meth, ref in GATE_PAIRS:
        d = p[p.method == meth]
        per_h = {int(h): int(g.success.sum()) for h, g in d.groupby('horizon')}
        rule[f'{meth}>{ref}'] = dict(successes_per_horizon=per_h,
                                     horizons_with_4_of_6=sum(v >= 4 for v in per_h.values()),
                                     rule_met=sum(v >= 4 for v in per_h.values()) >= 2)
    return p, rule


def domain_test(tb, de):
    out = []
    for s in ('youtube_hourly', 'taxi_hourly'):
        for h in HORIZONS:
            e = de[(de.scenario == s) & (de.horizon == h)]
            for meth in ('WSPI', 'ARIMA'):
                for met in MAIN:
                    r = row(tb, s, h, 'WSPI-FA', meth, met)
                    if r is None:
                        continue
                    out.append(dict(scenario=s, horizon=h, metric=met, method='WSPI-FA', against=meth,
                                    mean_wspi_fa=r.mean_reference, mean_against=r.mean_method,
                                    diff=r.diff_ref_minus_method, diff_ci_low=r.diff_ci_low,
                                    diff_ci_high=r.diff_ci_high, n_windows=r.n_windows,
                                    n_blocks=r.n_blocks, p_block_holm=r.p_block_holm,
                                    verdict=r.verdict,
                                    time_ratio_fa_to_arima=float(e.time_ratio_fa_to_arima.iloc[0]),
                                    fallback_share=float(e.fallback_share.iloc[0])))
    return pd.DataFrame(out)


def paper(root: Path, out: Path):
    src = root / 'stats_test' / 'all_paired_tests.csv'
    esrc = root / 'stats_explore' / 'all_paired_tests.csv'
    bsrc = root / 'stats_test_B' / 'all_paired_tests.csv'
    dsrc = root / 'decision' / 'domain_effect.csv'
    t = pd.read_csv(src, float_precision='round_trip')
    t = t[t.part == 'test']
    te = pd.read_csv(esrc, float_precision='round_trip')
    te = te[(te.part == 'test') & te.reference.isin(DOM_EXPLORE)]
    tb = pd.read_csv(bsrc, float_precision='round_trip')
    tb = tb[tb.part == 'test']
    cells = pd.concat([dominance(t, DOM_METHODS), dominance(te, DOM_EXPLORE, source='stats_explore')],
                      ignore_index=True)
    cells.to_csv(out / 'dominance_cells.csv', index=False)
    by_b, summ = dominance_tables(cells)
    by_b.to_csv(out / 'dominance_by_baseline.csv', index=False)
    summ.to_csv(out / 'dominance_summary.csv', index=False)
    pf = positive_facts(t)
    pf.to_csv(out / 'positive_facts.csv', index=False)
    pg, rule = pareto_gates(t)
    pg.to_csv(out / 'pareto_gates_test.csv', index=False)
    dt = domain_test(tb, pd.read_csv(dsrc, float_precision='round_trip'))
    dt.to_csv(out / 'domain_effect_test.csv', index=False)
    counts = (dt.groupby(['against', 'metric'])['verdict'].value_counts().unstack(fill_value=0)
              .reindex(columns=['ref_better', 'n.s.', 'ref_worse'], fill_value=0))
    head = dict(
        note='dominance: X better in >= 1 main metric and worse in none (column verdict), '
             'read from the report with reference X; WSPI-F-YW is exploratory',
        inputs={p.relative_to(root).as_posix(): sha(p) for p in (src, esrc, bsrc, dsrc)},
        baselines_A_X2=list(BASE_X2),
        D={r.method: dict(all=int(r.D), h1=int(r.D_h1), h8=int(r.D_h8), h24=int(r.D_h24),
                          not_dominated=int(r.not_dominated)) for r in summ.itertuples()},
        WSPI_F2_dominated_dominates_neither={
            r['baseline']: f"{r['dominated']} / {r['dominates']} / {r['trade-off'] + r['equal']}"
            for _, r in by_b[by_b.method == 'WSPI-F2'].iterrows()},
        WSPI_dominated_by={r.baseline: int(r.dominated)
                           for r in by_b[by_b.method == 'WSPI'].itertuples()},
        WSPI_dominates={r.baseline: int(r.dominates)
                        for r in by_b[by_b.method == 'WSPI'].itertuples()},
        GBM_W_vs_GBM_better={met: int(pf[(pf.method == 'GBM+W')][f'{met}_better'].iloc[0])
                             for met in MAIN},
        pareto_gates=rule,
        domain_effect_counts={f'{a}|{m}': {k: int(v) for k, v in counts.loc[(a, m)].items()}
                              for a, m in counts.index},
    )
    (out / 'analysis_paper.json').write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2))
    print('Saved to', out)


# ---- reading rule of run D (--run-d) ------------------------------------------

D_CANDIDATE, D_CONTROL = 'WSPI-H', 'WSPI-M'
D_TUNE_METHODS = ('WSPI-H', 'WSPI-M', 'WSPI-F2', 'WSPI', 'WSPI-F-YW')    # stats_tune_D references
D_TEST_METHODS = ('WSPI-H', 'WSPI-M')                                     # stats_test_D references
D_T1_MAX = 5                # T1: D(WSPI-H) <= 5
D_T3_MAX = 3                # T3: WSPI-F2 dominates WSPI-H in at most 3 cells
D_MECH_CELLS, D_MECH_YW_MAX = 10, 5      # mechanism question, step 4
D_CLAMP_COST = 6            # step 5: WSPI-H worse than WSPI-M in dRank in more than 6 cells
# (name, reference report, method, turn the verdict around)
D_PAIRS_BOTH = (('WSPI-F2 vs WSPI-H', 'WSPI-H', 'WSPI-F2', True),
                ('WSPI-H vs WSPI-M', 'WSPI-H', 'WSPI-M', False),
                ('WSPI-F-YW vs WSPI-M', 'WSPI-M', 'WSPI-F-YW', True),
                ('WSPI-H vs WSPI', 'WSPI-H', 'WSPI', False),
                ('WSPI-H vs WSPI-F-YW', 'WSPI-H', 'WSPI-F-YW', False),
                ('WSPI-M vs WSPI', 'WSPI-M', 'WSPI', False))
D_PAIRS_TUNE = (('WSPI-F2 vs WSPI-H (WSPI-F2 report)', 'WSPI-F2', 'WSPI-H', False),)


def d_pairs(t, pairs, scenarios=SCENARIOS):
    """Relation per cell and verdict counts per main metric; the first name of a
    pair is the side the verdicts and the relation are written from."""
    cells, counts = [], []
    for name, ref, meth, turn in pairs:
        per_met = {met: [] for met in MAIN}
        for s in scenarios:
            for h in HORIZONS:
                v = cell_verdicts(t, s, h, ref, meth)
                if v is None:
                    raise SystemExit(f'missing rows: {name}, {s}, h{h}')
                if turn:
                    v = {k: FLIP[x] for k, x in v.items()}
                vals = list(v.values())
                for met in MAIN:
                    per_met[met].append(v[met])
                cells.append(dict(pair=name, scenario=s, horizon=h,
                                  **{'verdict_' + k: x for k, x in v.items()},
                                  n_better=vals.count('ref_better'), n_worse=vals.count('ref_worse'),
                                  relation=relation(vals),
                                  source_report=f'reference {ref}' + (', verdict turned around' if turn else '')))
        for met in MAIN:
            counts.append(dict(pair=name, metric=met, better=per_met[met].count('ref_better'),
                               ns=per_met[met].count('n.s.'), worse=per_met[met].count('ref_worse')))
    return pd.DataFrame(cells), pd.DataFrame(counts)


def d_rel_counts(cells, name):
    d = cells[cells.pair == name]
    out = {k: int((d.relation == k).sum()) for k in ('dominates', 'dominated', 'trade-off', 'equal')}
    for h in HORIZONS:
        out[f'dominates_h{h}'] = int(((d.relation == 'dominates') & (d.horizon == h)).sum())
        out[f'dominated_h{h}'] = int(((d.relation == 'dominated') & (d.horizon == h)).sum())
    return out


def d_means(root, folder, part):
    s = pd.read_csv(root / folder / 'all_method_summary.csv', float_precision='round_trip')
    s = s[(s.part == part) & (s.reference_report == D_CANDIDATE) & s.main_metric]
    s = s[['scenario', 'horizon', 'metric', 'method', 'n_windows', 'mean', 'ci_low', 'ci_high']]
    return s.sort_values(['scenario', 'horizon', 'metric', 'method']).reset_index(drop=True)


def d_screen(d_h, d_f2, f2_dominates_h):
    c = dict(T1=bool(d_h <= D_T1_MAX), T2=bool(d_h < d_f2), T3=bool(f2_dominates_h <= D_T3_MAX))
    return dict(D_WSPI_H=int(d_h), D_WSPI_F2=int(d_f2), WSPI_F2_dominates_WSPI_H=int(f2_dominates_h),
                thresholds=dict(T1=f'D(WSPI-H) <= {D_T1_MAX}', T2='D(WSPI-H) < D(WSPI-F2), same part',
                                T3=f'WSPI-F2 dominates WSPI-H in at most {D_T3_MAX} cells'),
                conditions=c, passed=all(c.values()))


def run_d(root: Path, out: Path, part: str):
    folder = f'stats_{part}_D'
    src = root / folder / 'all_paired_tests.csv'
    screen_file = out / 'screen_tune.json'
    if part == 'test' and not screen_file.exists():
        raise SystemExit('the tuning part is read first: run --run-d tune before --run-d test')
    t = pd.read_csv(src, float_precision='round_trip')
    t = t[t.part == part]
    methods = D_TUNE_METHODS if part == 'tune' else D_TEST_METHODS
    cells = dominance(t, methods, source=folder)
    by_b, summ = dominance_tables(cells)
    pairs = D_PAIRS_BOTH + (D_PAIRS_TUNE if part == 'tune' else ())
    pc, pn = d_pairs(t, pairs)
    cells.to_csv(out / f'{part}_dominance_cells.csv', index=False)
    by_b.to_csv(out / f'{part}_dominance_by_baseline.csv', index=False)
    summ.to_csv(out / f'{part}_dominance_summary.csv', index=False)
    pc.to_csv(out / f'{part}_pairs.csv', index=False)
    pn.to_csv(out / f'{part}_pair_counts.csv', index=False)
    d_means(root, folder, part).to_csv(out / f'{part}_means.csv', index=False)
    D = {r.method: dict(all=int(r.D), h1=int(r.D_h1), h8=int(r.D_h8), h24=int(r.D_h24),
                        not_dominated=int(r.not_dominated), dominates_any=int(r.dominates_any))
         for r in summ.itertuples()}
    rel = {name: d_rel_counts(pc, name) for name, *_ in pairs}
    inputs = {src.relative_to(root).as_posix(): sha(src),
              f'{folder}/all_method_summary.csv': sha(root / folder / 'all_method_summary.csv')}
    if part == 'tune':
        head = dict(part='tune', inputs=inputs, baselines_A_X2=list(BASE_X2), D=D, pairs=rel,
                    screen=d_screen(D['WSPI-H']['all'], D['WSPI-F2']['all'],
                                    rel['WSPI-F2 vs WSPI-H']['dominates']))
        screen_file.write_text(json.dumps(head, indent=2))
        print(json.dumps(head, indent=2))
        print('Saved to', out)
        return
    # ---- test part ----
    psrc = root / 'decision_paper' / 'dominance_summary.csv'
    gsrc = root / 'decision_test' / 'gain_claims_test.csv'
    fixed = pd.read_csv(psrc).set_index('method').D
    best = pd.read_csv(gsrc)
    best = best[best.method == 'WSPI'][['baseline_set', 'scenario', 'horizon', 'metric', 'best_baseline']]
    g = []
    for b in best.itertuples():
        r = row(t, b.scenario, b.horizon, D_CANDIDATE, b.best_baseline, b.metric)
        g.append(dict(baseline_set=b.baseline_set, scenario=b.scenario, horizon=b.horizon, metric=b.metric,
                      method=D_CANDIDATE, best_baseline=b.best_baseline,
                      mean_method=r.mean_reference, mean_baseline=r.mean_method,
                      diff_method_minus_baseline=r.diff_ref_minus_method,
                      diff_ci_low=r.diff_ci_low, diff_ci_high=r.diff_ci_high,
                      cliffs_delta=r.cliffs_delta_favours_ref, p_block_holm=r.p_block_holm,
                      verdict=r.verdict, gain=bool(r.verdict == 'ref_better')))
    g = pd.DataFrame(g)
    g.to_csv(out / 'test_gain_claims.csv', index=False)
    tune = json.loads(screen_file.read_text())
    scr = d_screen(D['WSPI-H']['all'], int(fixed['WSPI-F2']), rel['WSPI-F2 vs WSPI-H']['dominates'])
    n_pass = int(tune['screen']['passed']) + int(scr['passed'])
    state = {2: 'accepted', 1: 'inconsistent', 0: 'rejected'}[n_pass]
    mech = rel['WSPI-F-YW vs WSPI-M']
    yw, m_dom = mech['dominates'], mech['dominated']
    if yw >= D_MECH_CELLS:
        outcome = 'AR(2) dynamics add to the return to the mean'
    elif 18 - yw >= D_MECH_CELLS and yw < D_MECH_YW_MAX:
        outcome = 'return to the mean; AR(2) dynamics add nothing'
    else:
        outcome = 'mixed'
    cost = pn[(pn.pair == 'WSPI-H vs WSPI-M')].set_index('metric')
    drank_worse = int(cost.loc['robustness_distortion', 'worse'])
    head = dict(part='test', inputs={**inputs, psrc.relative_to(root).as_posix(): sha(psrc),
                                     gsrc.relative_to(root).as_posix(): sha(gsrc),
                                     'decision_D/screen_tune.json': sha(screen_file)},
                baselines_A_X2=list(BASE_X2), D=D,
                D_fixed_test={k: int(fixed[k]) for k in ('WSPI', 'WSPI-F2', 'WSPI-F-YW')},
                pairs=rel, screen_tune=tune['screen'], screen_test=scr, final_state=state,
                gains_WSPI_H={b: f"{int(g[g.baseline_set == b].gain.sum())} of {int((g.baseline_set == b).sum())}"
                              for b in ('A', 'A+X2')},
                gain_verdicts_WSPI_H_A_X2={k: int(v) for k, v in
                                           g[g.baseline_set == 'A+X2'].verdict.value_counts().items()},
                mechanism=dict(WSPI_F_YW_dominates=yw, WSPI_M_dominates=m_dom,
                               other=18 - yw - m_dom, trade_off=mech['trade-off'], equal=mech['equal'],
                               outcome=outcome),
                clamp_cost=dict(counts={m: dict(better=int(cost.loc[m, 'better']), ns=int(cost.loc[m, 'ns']),
                                                worse=int(cost.loc[m, 'worse'])) for m in MAIN},
                                dRank_worse_cells=drank_worse,
                                clamp_lowers_stability=bool(drank_worse > D_CLAMP_COST)))
    (out / 'inference_D.json').write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2))
    print('Saved to', out)


# =============================================================================
# Confirmation data (the held-out scenarios; reading rule fixed beforehand)
# =============================================================================
C_SCENARIOS = ('taxi2024_hourly', 'taxi2024_30min', 'taxi2024_5min', 'wikipedia_daily')
C_REFS = ('WSPI-H', 'WSPI-F2', 'WSPI-M', 'WSPI')          # references of confirm/stats_*
C_T1_MAX = 3                # T1: D(WSPI-H) <= 3 (12 cells)
C_T3_MAX = 2                # T3: WSPI-F2 dominates WSPI-H in at most 2 cells
C_PAIRS = (('WSPI-F2 vs WSPI-H', 'WSPI-H', 'WSPI-F2', True),
           ('WSPI-F2 vs WSPI-H (WSPI-F2 report)', 'WSPI-F2', 'WSPI-H', False),
           ('WSPI-H vs WSPI-M', 'WSPI-H', 'WSPI-M', False),
           ('WSPI-F-YW vs WSPI-M', 'WSPI-M', 'WSPI-F-YW', True),
           ('WSPI-H vs WSPI', 'WSPI-H', 'WSPI', False),
           ('WSPI-M vs WSPI', 'WSPI-M', 'WSPI', False))


def c_screen(d_h, d_f2, f2_dominates_h):
    c = dict(T1=bool(d_h <= C_T1_MAX), T2=bool(d_h < d_f2), T3=bool(f2_dominates_h <= C_T3_MAX))
    return dict(D_WSPI_H=int(d_h), D_WSPI_F2=int(d_f2), WSPI_F2_dominates_WSPI_H=int(f2_dominates_h),
                thresholds=dict(T1=f'D(WSPI-H) <= {C_T1_MAX}',
                                T2='D(WSPI-H) < D(WSPI-F2), same data and part (equal fails)',
                                T3=f'WSPI-F2 dominates WSPI-H in at most {C_T3_MAX} cells'),
                conditions=c, passed=all(c.values()))


def confirm(root: Path, out: Path, part: str):
    """Reading rule of the confirmation data (thresholds T1 to T3 of the module header).
    The tuning part is read and written first; the test part needs that file."""
    folder = f'confirm/stats_{part}'
    src = root / folder / 'all_paired_tests.csv'
    ssrc = root / folder / 'all_method_summary.csv'
    screen_file = out / 'screen_tune.json'
    if part == 'test' and not screen_file.exists():
        raise SystemExit('the tuning part is read first: run --confirm tune before --confirm test')
    t = pd.read_csv(src, float_precision='round_trip')
    t = t[t.part == part]
    n_cells = len(C_SCENARIOS) * len(HORIZONS)
    cells = dominance(t, C_REFS, source=folder, scenarios=C_SCENARIOS)
    for m in C_REFS:
        n = cells[cells.method == m].groupby(['scenario', 'horizon']).baseline.nunique()
        if len(n) != n_cells or (n != len(BASE_X2)).any():
            raise SystemExit(f'incomplete reports for {m}: every one of the {n_cells} cells needs '
                             f'all {len(BASE_X2)} baselines')
    by_b, summ = dominance_tables(cells)
    pc, pn = d_pairs(t, C_PAIRS, scenarios=C_SCENARIOS)
    cells.to_csv(out / f'{part}_dominance_cells.csv', index=False)
    by_b.to_csv(out / f'{part}_dominance_by_baseline.csv', index=False)
    summ.to_csv(out / f'{part}_dominance_summary.csv', index=False)
    pc.to_csv(out / f'{part}_pairs.csv', index=False)
    pn.to_csv(out / f'{part}_pair_counts.csv', index=False)
    summary = pd.read_csv(ssrc, float_precision='round_trip')
    sm = summary[(summary.part == part) & summary.main_metric]
    sm = sm[['reference_report', 'scenario', 'horizon', 'metric', 'method', 'n_windows', 'mean',
             'ci_low', 'ci_high']]
    sm[sm.reference_report == 'WSPI-H'].drop(columns='reference_report') \
        .sort_values(['scenario', 'horizon', 'metric', 'method']) \
        .to_csv(out / f'{part}_means.csv', index=False)
    D = {r.method: dict(all=int(r.D), h1=int(r.D_h1), h8=int(r.D_h8), h24=int(r.D_h24),
                        not_dominated=int(r.not_dominated), dominates_any=int(r.dominates_any))
         for r in summ.itertuples()}
    rel = {name: d_rel_counts(pc, name) for name, *_ in C_PAIRS}
    scr = c_screen(D['WSPI-H']['all'], D['WSPI-F2']['all'], rel['WSPI-F2 vs WSPI-H']['dominates'])
    cross = rel['WSPI-F2 vs WSPI-H (WSPI-F2 report)']['dominates']
    inputs = {f'{folder}/all_paired_tests.csv': sha(src),
              f'{folder}/all_method_summary.csv': sha(ssrc)}
    if part == 'tune':
        head = dict(part='tune', role='read and written first; reported, not part of the verdict',
                    inputs=inputs, cells=n_cells, baselines_A_X2=list(BASE_X2), D=D, pairs=rel,
                    WSPI_F2_dominates_WSPI_H_from_WSPI_F2_report=cross, screen=scr)
        screen_file.write_text(json.dumps(head, indent=2))
        print(json.dumps(head, indent=2))
        print('Saved to', out)
        return
    # ---- test part: the verdict, then rule 2 and the descriptive counts ----
    means = sm[sm.reference_report == 'WSPI']
    g = []
    for bset, names in (('A', BASE_A), ('A+X2', BASE_X2)):
        for s in C_SCENARIOS:
            for h in HORIZONS:
                for met in MAIN:
                    mm = means[(means.scenario == s) & (means.horizon == h) & (means.metric == met)
                               & means.method.isin(names)].set_index('method')['mean']
                    if len(mm) != len(names):
                        raise SystemExit(f'missing means: {s} h{h} {met}')
                    best_v = mm.min() if met in LOWER else mm.max()
                    best = [n for n in names if mm[n] == best_v][0]
                    r = row(t, s, h, 'WSPI-H', best, met)
                    g.append(dict(baseline_set=bset, scenario=s, horizon=h, metric=met,
                                  method='WSPI-H', best_baseline=best,
                                  mean_method=r.mean_reference, mean_baseline=r.mean_method,
                                  diff_method_minus_baseline=r.diff_ref_minus_method,
                                  diff_ci_low=r.diff_ci_low, diff_ci_high=r.diff_ci_high,
                                  cliffs_delta=r.cliffs_delta_favours_ref,
                                  p_block_holm=r.p_block_holm, verdict=r.verdict,
                                  gain=bool(r.verdict == 'ref_better')))
    g = pd.DataFrame(g)
    g.to_csv(out / 'test_gain_claims.csv', index=False)
    tune = json.loads(screen_file.read_text())
    state = 'confirmed' if scr['passed'] else 'not confirmed'
    head = dict(part='test', inputs={**inputs, 'confirm/decision/screen_tune.json': sha(screen_file)},
                cells=n_cells, baselines_A_X2=list(BASE_X2), D=D, pairs=rel,
                WSPI_F2_dominates_WSPI_H_from_WSPI_F2_report=cross,
                screen_tune=tune['screen'], screen_test=scr, final_state=state,
                gains_WSPI_H={b: f"{int(g[g.baseline_set == b].gain.sum())} of {int((g.baseline_set == b).sum())}"
                              for b in ('A', 'A+X2')},
                gain_verdicts_WSPI_H_A_X2={f'{m}|{k}': int(v) for (m, k), v in
                                           g[g.baseline_set == 'A+X2'].groupby('metric').verdict
                                           .value_counts().items()},
                descriptive_counts={name: {r.metric: dict(better=int(r.better), ns=int(r.ns),
                                                          worse=int(r.worse))
                                           for r in pn[pn.pair == name].itertuples()}
                                    for name, *_ in C_PAIRS})
    (out / 'confirmation.json').write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2))
    print('Saved to', out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', default=None,
                    help='output folder (default <root>/decision_test, '
                         'with --explore <root>/decision_explore, '
                         'with --paper <root>/decision_paper, '
                         'with --run-d <root>/decision_D)')
    ap.add_argument('--explore', action='store_true',
                    help='only the tables of the exploratory reports (reference WSPI-F-YW)')
    ap.add_argument('--paper', action='store_true',
                    help='only the tables of the paper on the test part (dominance, '
                         'named-baseline comparisons, gates, domain effect)')
    ap.add_argument('--confirm', choices=('tune', 'test'), default=None,
                    help='only the reading rule of the confirmation data on one part '
                         '(<root>/confirm/stats_<part>); tune first, then test; '
                         'output <root>/confirm/decision')
    ap.add_argument('--run-d', choices=('tune', 'test'), default=None,
                    help='only the reading rule of run D (WSPI-M, WSPI-H) on one part; '
                         'tune first, then test')
    a = ap.parse_args(argv)
    root = Path(a.root)
    if a.confirm:
        out = Path(a.out) if a.out else root / 'confirm' / 'decision'
        out.mkdir(parents=True, exist_ok=True)
        confirm(root, out, a.confirm)
        return
    if a.run_d:
        out = Path(a.out) if a.out else root / 'decision_D'
        out.mkdir(parents=True, exist_ok=True)
        run_d(root, out, a.run_d)
        return
    if a.paper:
        out = Path(a.out) if a.out else root / 'decision_paper'
        out.mkdir(parents=True, exist_ok=True)
        paper(root, out)
        return
    if a.explore:
        out = Path(a.out) if a.out else root / 'decision_explore'
        out.mkdir(parents=True, exist_ok=True)
        explore(root, out)
        return
    out = Path(a.out) if a.out else root / 'decision_test'
    out.mkdir(parents=True, exist_ok=True)
    src = root / 'stats_test' / 'all_paired_tests.csv'
    t = pd.read_csv(src, float_precision='round_trip')
    t = t[t.part == 'test']
    ssrc = root / 'stats_test' / 'all_method_summary.csv'
    means = means_table(pd.read_csv(ssrc, float_precision='round_trip'))
    means.to_csv(out / 'means_test.csv', index=False)

    g = gain_claims(t, means)
    g.to_csv(out / 'gain_claims_test.csv', index=False)
    gs = (g.groupby(['baseline_set', 'method', 'metric'])['verdict'].value_counts()
          .unstack(fill_value=0).reindex(columns=['ref_better', 'n.s.', 'ref_worse'], fill_value=0)
          .reset_index())
    gs.to_csv(out / 'gain_summary_test.csv', index=False)

    p = pareto(t)
    p.to_csv(out / 'pareto_test.csv', index=False)

    tm = t[t.main_metric]
    vc = (tm.groupby(['reference', 'method', 'metric'])['verdict'].value_counts()
          .unstack(fill_value=0).reindex(columns=['ref_better', 'n.s.', 'ref_worse'], fill_value=0)
          .reset_index())
    vc.to_csv(out / 'verdict_counts_test.csv', index=False)

    keep = pd.Series(False, index=tm.index)
    for ref, meth in PAIRS:
        keep |= (tm.reference == ref) & (tm.method == meth)
    tm[keep].to_csv(out / 'pairs_test.csv', index=False)

    rt = runtime(root)
    rt.to_csv(out / 'runtime_test.csv', index=False)
    runtime_ratio(rt).to_csv(out / 'runtime_ratio.csv', index=False)
    sens_range(root).to_csv(out / 'sensitivity_range.csv', index=False)
    nlms(root).to_csv(out / 'nlms_late.csv', index=False)

    head = dict(
        inputs={src.relative_to(root).as_posix(): sha(src),
                'stats_test/all_method_summary.csv': sha(ssrc),
                'stats_test/sensitivity_summary.csv': sha(root / 'stats_test' / 'sensitivity_summary.csv'),
                'stats_tune/kappa_selection.json': sha(root / 'stats_tune' / 'kappa_selection.json')},
        selected_kappa=json.loads((root / 'stats_tune' / 'kappa_selection.json').read_text())['selected_kappa'],
        gains={f'{b}|{m}': int(g[(g.baseline_set == b) & (g.method == m)].gain.sum())
               for b in ('A', 'A+X2') for m in REFS},
        comparisons_per_method=len(SCENARIOS) * len(HORIZONS) * len(MAIN),
        pareto_success={f'{r}>{m}': int(p[(p.reference == r) & (p.method == m)].success.sum())
                        for r, m in PARETO},
        baselines_A=list(BASE_A), baselines_A_X2=list(BASE_X2),
    )
    (out / 'analysis_test.json').write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2))
    print('Saved to', out)


if __name__ == '__main__':
    main()
