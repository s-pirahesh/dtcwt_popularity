r"""
Apply the pre-registered decision rules to the prediction results
=================================================================
Reads only the CSV files written by ``tools/run_prediction_stats.py`` and
``tools/run_prediction.py`` and applies, without any free choice, the rules
that were fixed before the runs.  A verdict is always the column ``verdict``
of ``paired_tests.csv`` (two-sided Wilcoxon on block means, Holm within one
scenario x horizon x metric, alpha 0.05): ``ref_better``, ``ref_worse`` or
``n.s.``.  The four main metrics are NDCG@10, Spearman rho, RSI@10 and
DeltaRank (``robustness_distortion``, lower is better).

Rule 1, narrative (gate of WSPI-FG against the gate of WSPI-F2)
  cell     one scenario x one horizon.  The cell is a success if, in the report
           with reference WSPI-FG, the comparison with WSPI-F2 is ``ref_better``
           on at least one main metric and ``ref_worse`` on none (Pareto).
  horizon  a success if at least 4 of the 6 scenarios are successes.
  result   narrative A if at least 2 of the 3 horizons are successes, else B.

Rule 2, gain claims (against the best traditional baseline)
  For every scenario x horizon x main metric the best traditional baseline is
  the method with the best mean among Persistence, Holt, ARYW and MA64 on the
  common windows of run A (ties: that order).  A gain of WSPI, WSPI-FG or
  WSPI-F2 is claimed only where the report with that method as reference gives
  ``ref_better`` against that baseline.

Rule 3, domain effect (run B)
  Verdict of the report with reference WSPI-FA against ARIMA, with the ratio
  of the scorer times WSPI-FA / ARIMA and the share of WSPI-FA fits that fell
  back to persistence (``scorer_timing.csv``).

Negative results.  Listed mechanically, one row per scenario x horizon x main
metric and kind:
  gate_worse_than_structural     WSPI-FG against WSPI-F2 is ``ref_worse``
  forecast_worse_than_wspi       reference WSPI against a coefficient-domain
                                 forecast method is ``ref_better``
  worse_than_best_baseline       WSPI, WSPI-FG or WSPI-F2 against the best
                                 traditional baseline is ``ref_worse``
  no_gain_over_best_baseline     the same comparison is ``n.s.``
  domain_worse_than_raw          WSPI-FA against ARIMA is ``ref_worse``
  bound_violation                a row of ``bound_check.csv`` with status VIOLATION

Outputs (``--out``, default ``<root>/decision``):
  pareto_cells.csv, pareto_horizons.csv, gain_claims.csv, domain_effect.csv,
  bound_check.csv (all runs stacked), negative_results.csv, decision.json

If a cell of rule 1 is missing the narrative is UNDETERMINED and the exit code
is 1; the other tables hold what exists.

    python tools\prediction_decision.py --root results\paper2
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = ('youtube_hourly', 'taxi_hourly', 'taxi_30min', 'taxi_5min',
             'movielens_daily', 'movielens_weekly')
RUN_B_SCENARIOS = ('youtube_hourly', 'taxi_hourly')
HORIZONS = (1, 8, 24)
MAIN_METRICS = ('ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion')
LOWER_IS_BETTER = ('robustness_distortion',)
BASELINES = ('Persistence', 'Holt', 'ARYW', 'MA64')
CLAIM_METHODS = ('WSPI', 'WSPI-FG', 'WSPI-F2')
FORECAST_METHODS = ('WSPI-FG', 'WSPI-F2', 'WSPI-F-YW', 'WSPI-F', 'WSPI-FT')
MIN_SCENARIOS = 4
MIN_HORIZONS = 2
VERDICTS = ('ref_better', 'ref_worse', 'n.s.')


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lookup(paired: pd.DataFrame, run, scenario, horizon, reference, method, metric):
    """One row of the stacked paired tests, or None."""
    d = paired[(paired['run'] == run) & (paired['scenario'] == scenario)
               & (paired['horizon'] == horizon) & (paired['reference'] == reference)
               & (paired['method'] == method) & (paired['metric'] == metric)]
    if len(d) != 1:
        return None
    return d.iloc[0]


def pareto(paired: pd.DataFrame, scenarios=SCENARIOS, horizons=HORIZONS):
    rows = []
    for h in horizons:
        for s in scenarios:
            row = dict(scenario=s, horizon=h)
            v = []
            for c in MAIN_METRICS:
                r = lookup(paired, 'A', s, h, 'WSPI-FG', 'WSPI-F2', c)
                row[f'verdict_{c}'] = r['verdict'] if r is not None else 'missing'
                v.append(row[f'verdict_{c}'])
            row['n_better'] = v.count('ref_better')
            row['n_worse'] = v.count('ref_worse')
            row['complete'] = 'missing' not in v
            row['success'] = bool(row['complete'] and row['n_better'] >= 1 and row['n_worse'] == 0)
            rows.append(row)
    cells = pd.DataFrame(rows)
    hrows = []
    for h in horizons:
        d = cells[cells['horizon'] == h]
        hrows.append(dict(horizon=h, n_scenarios=len(d), n_complete=int(d['complete'].sum()),
                          n_success=int(d['success'].sum()),
                          success=bool(d['success'].sum() >= MIN_SCENARIOS)))
    hz = pd.DataFrame(hrows)
    complete = bool(cells['complete'].all()) and len(cells) == len(SCENARIOS) * len(HORIZONS)
    n_ok = int(hz['success'].sum())
    narrative = ('A' if n_ok >= MIN_HORIZONS else 'B') if complete else 'UNDETERMINED'
    return cells, hz, narrative


def gain_claims(paired: pd.DataFrame, summary: pd.DataFrame, scenarios=SCENARIOS,
                horizons=HORIZONS):
    rows = []
    s_a = summary[(summary['run'] == 'A') & (summary['reference_report'] == 'WSPI')]
    for s in scenarios:
        for h in horizons:
            for c in MAIN_METRICS:
                d = s_a[(s_a['scenario'] == s) & (s_a['horizon'] == h) & (s_a['metric'] == c)
                        & s_a['method'].isin(BASELINES)]
                if d.empty:
                    continue
                means = {m: float(d.loc[d['method'] == m, 'mean'].iloc[0])
                         for m in BASELINES if (d['method'] == m).any()}
                pick = min if c in LOWER_IS_BETTER else max
                best_value = pick(means.values())
                best = [m for m in BASELINES if m in means and means[m] == best_value][0]
                for ref in CLAIM_METHODS:
                    r = lookup(paired, 'A', s, h, ref, best, c)
                    if r is None:
                        continue
                    rows.append(dict(scenario=s, horizon=h, metric=c, method=ref,
                                     best_baseline=best, mean_method=r['mean_reference'],
                                     mean_baseline=r['mean_method'],
                                     diff_method_minus_baseline=r['diff_ref_minus_method'],
                                     diff_ci_low=r['diff_ci_low'], diff_ci_high=r['diff_ci_high'],
                                     n_blocks=r['n_blocks'], p_block_holm=r['p_block_holm'],
                                     verdict=r['verdict'], gain=bool(r['verdict'] == 'ref_better')))
    return pd.DataFrame(rows)


def domain_effect(paired: pd.DataFrame, root: Path, horizons=HORIZONS):
    rows = []
    for s in RUN_B_SCENARIOS:
        for h in horizons:
            tf = root / 'run_B' / s / f'h{h}' / 'scorer_timing.csv'
            ratio = share = sec_fa = sec_ar = np.nan
            if tf.exists():
                t = pd.read_csv(tf).set_index('method')
                if 'WSPI-FA' in t.index and 'ARIMA' in t.index:
                    sec_fa = float(t.loc['WSPI-FA', 'scorer_seconds'])
                    sec_ar = float(t.loc['ARIMA', 'scorer_seconds'])
                    ratio = sec_fa / sec_ar if sec_ar > 0 else np.nan
                    share = float(t.loc['WSPI-FA', 'fallback_share'])
            for c in MAIN_METRICS:
                r = lookup(paired, 'B', s, h, 'WSPI-FA', 'ARIMA', c)
                if r is None:
                    continue
                rows.append(dict(scenario=s, horizon=h, metric=c, mean_wspi_fa=r['mean_reference'],
                                 mean_arima=r['mean_method'],
                                 diff_fa_minus_arima=r['diff_ref_minus_method'],
                                 diff_ci_low=r['diff_ci_low'], diff_ci_high=r['diff_ci_high'],
                                 n_blocks=r['n_blocks'], p_block_holm=r['p_block_holm'],
                                 verdict=r['verdict'], scorer_seconds_wspi_fa=sec_fa,
                                 scorer_seconds_arima=sec_ar, time_ratio_fa_to_arima=ratio,
                                 fallback_share=share))
    return pd.DataFrame(rows)


def bound_table(root: Path):
    parts = []
    for run in ('A', 'B'):
        for f in sorted((root / f'run_{run}').glob('*/h*/bound_check.csv')):
            d = pd.read_csv(f)
            d.insert(0, 'scenario', f.parent.parent.name)
            d.insert(0, 'run', run)
            parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def negative_results(paired, cells, claims, domain, bounds):
    rows = []

    def add(kind, scenario, horizon, metric, reference, method, r=None, note=''):
        rows.append(dict(kind=kind, scenario=scenario, horizon=horizon, metric=metric,
                         reference=reference, method=method,
                         diff_ref_minus_method=r['diff_ref_minus_method'] if r is not None else np.nan,
                         p_block_holm=r['p_block_holm'] if r is not None else np.nan,
                         verdict=r['verdict'] if r is not None else '', note=note))

    for _, cell in cells.iterrows():
        for c in MAIN_METRICS:
            if cell[f'verdict_{c}'] == 'ref_worse':
                add('gate_worse_than_structural', cell['scenario'], cell['horizon'], c,
                    'WSPI-FG', 'WSPI-F2',
                    lookup(paired, 'A', cell['scenario'], cell['horizon'], 'WSPI-FG', 'WSPI-F2', c))
    d = paired[(paired['run'] == 'A') & (paired['reference'] == 'WSPI')
               & paired['method'].isin(FORECAST_METHODS) & paired['metric'].isin(MAIN_METRICS)
               & (paired['verdict'] == 'ref_better')]
    for _, r in d.iterrows():
        add('forecast_worse_than_wspi', r['scenario'], r['horizon'], r['metric'], 'WSPI',
            r['method'], r)
    for _, r in claims.iterrows():
        if r['verdict'] in ('ref_worse', 'n.s.'):
            kind = 'worse_than_best_baseline' if r['verdict'] == 'ref_worse' \
                else 'no_gain_over_best_baseline'
            rows.append(dict(kind=kind, scenario=r['scenario'], horizon=r['horizon'],
                             metric=r['metric'], reference=r['method'], method=r['best_baseline'],
                             diff_ref_minus_method=r['diff_method_minus_baseline'],
                             p_block_holm=r['p_block_holm'], verdict=r['verdict'], note=''))
    for _, r in domain.iterrows():
        if r['verdict'] == 'ref_worse':
            rows.append(dict(kind='domain_worse_than_raw', scenario=r['scenario'],
                             horizon=r['horizon'], metric=r['metric'], reference='WSPI-FA',
                             method='ARIMA', diff_ref_minus_method=r['diff_fa_minus_arima'],
                             p_block_holm=r['p_block_holm'], verdict=r['verdict'], note=''))
    if len(bounds):
        for _, r in bounds[bounds['status'] != 'OK'].iterrows():
            add('bound_violation', r['scenario'], r['horizon'], '', r['method'], 'WSPI', None,
                f"run {r['run']}: ratio in [{r['min_ratio']}, {r['max_ratio']}], "
                f"bounds [{r['lower_bound']}, {r['upper_bound']}]")
    cols = ['kind', 'scenario', 'horizon', 'metric', 'reference', 'method',
            'diff_ref_minus_method', 'p_block_holm', 'verdict', 'note']
    return pd.DataFrame(rows, columns=cols)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='results/paper2')
    ap.add_argument('--stats', default=None, help='default: <root>/stats')
    ap.add_argument('--out', default=None, help='default: <root>/decision')
    a = ap.parse_args(argv)
    root = _abs(a.root)
    stats = _abs(a.stats) if a.stats else root / 'stats'
    out = _abs(a.out) if a.out else root / 'decision'
    pf, sf = stats / 'all_paired_tests.csv', stats / 'all_method_summary.csv'
    if not pf.exists() or not sf.exists():
        print(f'missing {pf} or {sf}: run tools/run_prediction_stats.py first')
        return 1
    paired = pd.read_csv(pf, float_precision='round_trip')
    summary = pd.read_csv(sf, float_precision='round_trip')
    unknown = sorted(set(paired['verdict'].dropna()) - set(VERDICTS))
    if unknown:
        print(f'unknown verdict values: {unknown}')
        return 1

    cells, hz, narrative = pareto(paired)
    claims = gain_claims(paired, summary)
    domain = domain_effect(paired, root)
    bounds = bound_table(root)
    neg = negative_results(paired, cells, claims, domain, bounds)

    out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(out / 'pareto_cells.csv', index=False, encoding='utf-8')
    hz.to_csv(out / 'pareto_horizons.csv', index=False, encoding='utf-8')
    claims.to_csv(out / 'gain_claims.csv', index=False, encoding='utf-8')
    domain.to_csv(out / 'domain_effect.csv', index=False, encoding='utf-8')
    bounds.to_csv(out / 'bound_check.csv', index=False, encoding='utf-8')
    neg.to_csv(out / 'negative_results.csv', index=False, encoding='utf-8')
    n_cells_expected = len(SCENARIOS) * len(HORIZONS)
    decision = dict(
        narrative=narrative,
        rule=dict(pair='WSPI-FG (reference) against WSPI-F2', main_metrics=list(MAIN_METRICS),
                  cell='>= 1 ref_better and 0 ref_worse', horizon=f'>= {MIN_SCENARIOS} of 6 scenarios',
                  narrative_A=f'>= {MIN_HORIZONS} of 3 horizons'),
        successful_horizons=[int(h) for h in hz.loc[hz['success'], 'horizon']],
        successful_cells=int(cells['success'].sum()),
        complete_cells=int(cells['complete'].sum()), expected_cells=n_cells_expected,
        gain_rows=int(len(claims)), gains_claimed=int(claims['gain'].sum()) if len(claims) else 0,
        domain_rows=int(len(domain)),
        bound_rows=int(len(bounds)),
        bound_violations=int((bounds['status'] != 'OK').sum()) if len(bounds) else 0,
        negative_results={k: int(v) for k, v in neg['kind'].value_counts().items()},
        inputs={'all_paired_tests.csv': _sha(pf), 'all_method_summary.csv': _sha(sf)},
        created=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / 'decision.json').write_text(json.dumps(decision, indent=2), encoding='utf-8')

    show = cells.copy()
    show.columns = [c.replace('verdict_', '') for c in show.columns]
    with pd.option_context('display.width', 200, 'display.max_rows', 100):
        print(show.to_string(index=False))
        print()
        print(hz.to_string(index=False))
    print(f"\ncells complete: {decision['complete_cells']} of {n_cells_expected}; "
          f"gains claimed: {decision['gains_claimed']} of {decision['gain_rows']} comparisons; "
          f"bound violations: {decision['bound_violations']}")
    print(f'Saved to {out}')
    print(f'NARRATIVE {narrative}')
    return 0 if narrative != 'UNDETERMINED' else 1


if __name__ == '__main__':
    sys.exit(main())
