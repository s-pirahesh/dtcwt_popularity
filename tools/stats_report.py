r"""
Statistical report for one protocol-V5 run folder (revision-srep-v5, T1.6 / E0)
==============================================================================
Reads ``<run>/protocol/<method>_protocol.csv`` (window-by-window results written
by tools/run_v5_eval.py) and writes, for every metric column:

  method_summary.csv   per method: n, mean, SD, median and a 95 % CI of the
                       mean from a circular block bootstrap.  Rows are the
                       windows common to all methods (same rows as
                       comparison/summary_common_windows.csv, so the means are
                       identical to that file; checked at run time).
  paired_tests.csv     reference method (default WSPI) against every other
                       method, paired by window_id on the windows where both
                       values are finite:
                         - mean difference (reference - other) + block-bootstrap 95 % CI
                         - win / loss / tie counts (oriented: "win" = reference better)
                         - Wilcoxon signed-rank on the windows       (p_window)
                         - Wilcoxon signed-rank on non-overlapping block means
                           of the difference                          (p_block)
                         - Holm correction of both p values; the family is
                           one scenario x one metric (all comparisons of that metric)
                         - matched-pairs rank-biserial correlation (window and block)
                         - Cliff's delta between the two per-window distributions
                       Effect sizes are oriented: positive = reference better.
                       ``verdict`` uses the block test after Holm (alpha 0.05).
  metadata/stats_run.json   arguments, library versions, date, runtime.

Why block-level tests: consecutive windows share 63 of 64 slots, so per-window
values are strongly autocorrelated and the window-level Wilcoxon p value is
far too small.  The block test uses one value per block (a day for YouTube, a
week for taxi; decision of 24 Sep 2026, tracker section E) and is the basis of
every claim in the paper.  The window-level p value is reported for
completeness only.

Bootstrap: circular block bootstrap of the mean (Politis & Romano 1992), block
length --block, B resamples, percentile interval.  The resampled block starts
depend only on (seed, n, block), so every method and every difference of one
metric is resampled with the same windows.  --block-sens adds a second block
length (sensitivity; columns with suffix _sens).

Examples (from the project root, Windows):

  python tools\stats_report.py --run results\revision_v5\T2.1_baselines_W16\taxi_hourly ^
         --block 168 --block-sens 24 --out results\revision_v5\T1.6_stats\T2.1_baselines_W16\taxi_hourly

  # stack every stats folder under a root into two tables
  python tools\stats_report.py --collect results\revision_v5\T1.6_stats

Full command list: Revisions/V4/Response/Runbooks/RUN_T1.6_T1.5.md
Unit test: tools/test_stats_report.py
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import scipy
    from scipy.stats import rankdata, wilcoxon
except ImportError:  # pragma: no cover
    sys.exit('scipy is required:  pip install scipy')

ROOT = Path(__file__).resolve().parent.parent

METHOD_ORDER = ['AF', 'EWMA', 'RRD', 'VSE', 'CompoundPop', 'PFRF',
                'DWT+AF', 'DTCWT+AF', 'WSPI']
ID_COLUMNS = {'window_id', 'timestamp', 'method', 'num_items'}
NON_METRIC = {'ties_top21', 'padded', 'n_nonfinite'}
LOWER_IS_BETTER = {'mae', 'robustness_distortion'}
MAIN_METRICS = ['ndcg@10', 'spearman_rho', 'rsi@10', 'robustness_distortion']
ALPHA = 0.05


# =============================================================================
# Statistics
# =============================================================================
def cbb_starts(n: int, block: int, B: int, seed: int) -> np.ndarray:
    """Random block starts (B x k) for a circular block bootstrap of length n.
    Deterministic in (seed, n, block)."""
    k = -(-n // block)
    rng = np.random.default_rng([seed, n, block])
    return rng.integers(0, n, size=(B, k))


def cbb_means(x: np.ndarray, block: int, starts: np.ndarray) -> np.ndarray:
    """Bootstrap means of x: k-1 full circular blocks + one block of length r,
    so every resample has exactly n values."""
    n = len(x)
    B, k = starts.shape
    r = n - (k - 1) * block                     # 1 <= r <= block
    xx = np.concatenate([x, x[:block]])         # circular extension
    cs = np.concatenate([[0.0], np.cumsum(xx)])
    full = starts[:, :k - 1]
    s = (cs[full + block] - cs[full]).sum(axis=1)
    last = starts[:, k - 1]
    s += cs[last + r] - cs[last]
    return s / n


def percentile_ci(boot: np.ndarray):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def rank_biserial(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (zeros dropped, as in
    Wilcoxon's zero_method='wilcox').  Positive = d mostly > 0."""
    d = d[d != 0]
    if len(d) == 0:
        return float('nan')
    r = rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) - P(a < b) over all cross pairs, computed from ranks
    (Mann-Whitney U with average ranks for ties)."""
    na, nb = len(a), len(b)
    r = rankdata(np.concatenate([a, b]))
    u = r[:na].sum() - na * (na + 1) / 2.0
    return float(2.0 * u / (na * nb) - 1.0)


def wilcoxon_p(d: np.ndarray) -> float:
    if np.count_nonzero(d) == 0:
        return 1.0
    return float(wilcoxon(d, zero_method='wilcox', alternative='two-sided').pvalue)


def block_means(d: np.ndarray, block: int) -> np.ndarray:
    """Means of consecutive non-overlapping blocks; the incomplete tail is dropped."""
    nb = len(d) // block
    return d[:nb * block].reshape(nb, block).mean(axis=1)


def holm(p) -> np.ndarray:
    """Holm-Bonferroni adjusted p values (NaN kept as NaN)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind='stable')]
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        out[idx] = running
    return out


# =============================================================================
# Loading
# =============================================================================
def load_run(run: Path):
    pdir = run / 'protocol'
    files = sorted(pdir.glob('*_protocol.csv'))
    if not files:
        sys.exit(f'no protocol CSV in {pdir}')
    names = [f.name[:-len('_protocol.csv')] for f in files]
    names = [m for m in METHOD_ORDER if m in names] + \
            [m for m in names if m not in METHOD_ORDER]
    res = {m: pd.read_csv(pdir / f'{m}_protocol.csv') for m in names}
    first = res[names[0]]
    metrics = [c for c in first.columns if c not in ID_COLUMNS | NON_METRIC]
    return res, metrics


# =============================================================================
# One run
# =============================================================================
def analyse(run: Path, out: Path, block: int, block_sens, reference: str,
            B: int, seed: int, metrics_arg):
    t0 = time.time()
    res, metrics = load_run(run)
    if metrics_arg:
        metrics = [m for m in metrics_arg if m in metrics]
    if reference not in res:
        sys.exit(f'reference {reference} not in {list(res)}')
    ids = sorted(set.intersection(*[set(df['window_id']) for df in res.values()]))
    idx = pd.Index(ids, name='window_id')
    tab = {m: df.set_index('window_id').reindex(idx) for m, df in res.items()}
    blocks = [('', block)] + ([('_sens', block_sens)] if block_sens else [])
    others = [m for m in res if m != reference]

    # ---- per-method summary -------------------------------------------------
    srows = []
    for c in metrics:
        for m, t in tab.items():
            x = t[c].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            r = {'metric': c, 'method': m, 'n_windows': len(x),
                 'mean': x.mean(), 'sd': x.std(ddof=1), 'median': float(np.median(x)),
                 'lower_is_better': c in LOWER_IS_BETTER, 'main_metric': c in MAIN_METRICS}
            for suf, L in blocks:
                bm = cbb_means(x, L, cbb_starts(len(x), L, B, seed))
                r[f'ci_low{suf}'], r[f'ci_high{suf}'] = percentile_ci(bm)
                r[f'block{suf}'] = L
            srows.append(r)
    summary = pd.DataFrame(srows)

    # ---- paired tests ---------------------------------------------------------
    prows = []
    for c in metrics:
        sign = -1.0 if c in LOWER_IS_BETTER else 1.0
        a_all = tab[reference][c].to_numpy(dtype=float)
        for m in others:
            b_all = tab[m][c].to_numpy(dtype=float)
            ok = np.isfinite(a_all) & np.isfinite(b_all)
            a, b = a_all[ok], b_all[ok]
            d = a - b
            od = sign * d                                   # > 0: reference better
            r = {'metric': c, 'reference': reference, 'method': m,
                 'lower_is_better': c in LOWER_IS_BETTER, 'main_metric': c in MAIN_METRICS,
                 'n_windows': int(ok.sum()),
                 'mean_reference': a.mean(), 'mean_method': b.mean(),
                 'diff_ref_minus_method': d.mean(),
                 'wins': int((od > 0).sum()), 'losses': int((od < 0).sum()),
                 'ties': int((od == 0).sum()),
                 'p_window': wilcoxon_p(d),
                 'rb_window_favours_ref': sign * rank_biserial(d),
                 'cliffs_delta_favours_ref': sign * cliffs_delta(a, b)}
            for suf, L in blocks:
                bd = cbb_means(d, L, cbb_starts(len(d), L, B, seed))
                r[f'diff_ci_low{suf}'], r[f'diff_ci_high{suf}'] = percentile_ci(bd)
                bmn = block_means(d, L)
                r[f'block{suf}'] = L
                r[f'n_blocks{suf}'] = len(bmn)
                r[f'p_block{suf}'] = wilcoxon_p(bmn) if len(bmn) else float('nan')
                r[f'rb_block{suf}_favours_ref'] = sign * rank_biserial(bmn)
            prows.append(r)
    paired = pd.DataFrame(prows)

    # Holm within one metric (family = this scenario x this metric)
    pcols = ['p_window'] + [f'p_block{suf}' for suf, _ in blocks]
    for pc in pcols:
        paired[f'{pc}_holm'] = np.nan
        for c in metrics:
            sel = paired['metric'] == c
            paired.loc[sel, f'{pc}_holm'] = holm(paired.loc[sel, pc].to_numpy())
    for suf, _ in blocks:
        od = np.where(paired['lower_is_better'], -1.0, 1.0) * paired['diff_ref_minus_method']
        sig = paired[f'p_block{suf}_holm'] < ALPHA
        paired[f'verdict{suf}'] = np.where(~sig, 'n.s.',
                                           np.where(od > 0, 'ref_better', 'ref_worse'))

    # ---- consistency check with the run's own summary ----------------------
    check = {}
    sc = run / 'comparison' / 'summary_common_windows.csv'
    if sc.exists():
        s0 = pd.read_csv(sc).set_index('method')
        worst = 0.0
        for _, r in summary.iterrows():
            col = f"{r['metric']}_mean"
            if col in s0.columns and r['method'] in s0.index:
                worst = max(worst, abs(float(s0.loc[r['method'], col]) - r['mean']))
        check = {'summary_common_windows_max_abs_diff': worst,
                 'summary_common_windows_match': bool(worst < 1e-9)}
        if worst >= 1e-9:
            print(f'WARNING: means differ from {sc} by {worst:g}')

    # ---- write --------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metadata').mkdir(exist_ok=True)
    summary.to_csv(out / 'method_summary.csv', index=False)
    paired.to_csv(out / 'paired_tests.csv', index=False)
    meta = {
        'task': 'T1.6 (E0) statistical report',
        'source_run': str(run),
        'reference': reference, 'methods': list(res),
        'n_common_windows': len(ids),
        'first_common_window': int(ids[0]), 'last_common_window': int(ids[-1]),
        'block': block, 'block_sens': block_sens,
        'bootstrap': 'circular block bootstrap of the mean, percentile 95% CI',
        'B': B, 'seed': seed,
        'wilcoxon': "two-sided, zero_method='wilcox'",
        'holm_family': 'one scenario x one metric',
        'verdict': f'block Wilcoxon after Holm, alpha={ALPHA}',
        'effect_size_orientation': 'positive = reference better',
        'lower_is_better': sorted(LOWER_IS_BETTER),
        'metrics': metrics, 'main_metrics': MAIN_METRICS,
        **check,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'scipy': scipy.__version__},
        'host': platform.platform(),
        'date_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'runtime_seconds': round(time.time() - t0, 2),
    }
    (out / 'metadata' / 'stats_run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    return summary, paired, meta


def print_main(paired: pd.DataFrame, blocks_sens: bool):
    cols = ['metric', 'method', 'n_windows', 'diff_ref_minus_method',
            'diff_ci_low', 'diff_ci_high', 'n_blocks', 'p_block_holm',
            'rb_block_favours_ref', 'cliffs_delta_favours_ref', 'verdict']
    if blocks_sens:
        cols.append('verdict_sens')
    d = paired[paired['main_metric']][cols]
    with pd.option_context('display.width', 200, 'display.max_rows', 200,
                           'display.float_format', '{:.4g}'.format):
        print(d.to_string(index=False))


# =============================================================================
# Collect
# =============================================================================
def collect(root: Path):
    for name in ('method_summary', 'paired_tests'):
        parts = []
        for f in sorted(root.rglob(f'{name}.csv')):
            rel = f.parent.relative_to(root).parts
            df = pd.read_csv(f)
            df.insert(0, 'scenario', rel[-1] if rel else '')
            df.insert(0, 'run_group', '/'.join(rel[:-1]))
            parts.append(df)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(root / f'all_{name}.csv', index=False)
            print(f'{root / f"all_{name}.csv"}: {len(parts)} folders')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--run', help='run folder with protocol/*.csv')
    ap.add_argument('--out', help='output folder (always give it explicitly)')
    ap.add_argument('--block', type=int, help='block length in slots (main)')
    ap.add_argument('--block-sens', type=int, default=None, help='second block length (sensitivity)')
    ap.add_argument('--reference', default='WSPI')
    ap.add_argument('--B', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--metrics', nargs='*', default=None)
    ap.add_argument('--collect', help='stack all method_summary/paired_tests under this folder')
    a = ap.parse_args()

    def absp(p):
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    if a.collect:
        collect(absp(a.collect))
        return
    if not (a.run and a.out and a.block):
        ap.error('--run, --out and --block are required (or use --collect)')
    summary, paired, meta = analyse(absp(a.run), absp(a.out), a.block, a.block_sens,
                                    a.reference, a.B, a.seed, a.metrics)
    print(f"common windows: {meta['n_common_windows']}   block {a.block}"
          f"{'' if not a.block_sens else f' (sens {a.block_sens})'}   B={a.B}   "
          f"summary check: {meta.get('summary_common_windows_match', 'n/a')}   "
          f"{meta['runtime_seconds']} s")
    print_main(paired, bool(a.block_sens))
    print(f'\nSaved to {absp(a.out)}')


if __name__ == '__main__':
    main()
